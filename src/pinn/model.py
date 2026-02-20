"""
model.py — Hybrid Physics-Guided Surrogate (PgNN) for Seismic RC Response Prediction
======================================================================================

Implements the production Physics-Guided surrogate model (PgNN): a 1D-CNN temporal encoder followed
by fully-connected regression layers.  The network predicts inter-story
drift ratios (IDR) from ground-motion acceleration time series.

Architecture
------------
    Encoder (1D-CNN)
    ├── Conv1d(1→32, k=7, s=2) + SiLU
    ├── Conv1d(32→64, k=5, s=2) + SiLU
    ├── Conv1d(64→128, k=3, s=2) + SiLU
    └── AdaptiveAvgPool1d(16)
    Head (FC)
    ├── Linear(128×16 → 256) + SiLU + Dropout
    ├── Linear(256 → 128) + SiLU + Dropout
    ├── Linear(128 → 64) + SiLU
    ├── Linear(64 → 32) + SiLU
    └── Linear(32 → n_stories)

Activation: SiLU (Swish with β=1) — smooth C¹-continuous, preferred for
PINNs because physics-loss gradients d²L/dx² require smooth activations.

References
----------
    [3] Raissi, Perdikaris, Karniadakis (2019). J. Comput. Phys., 378, 686-707.
    [9] Ramachandran, Zoph, Le (2017). arXiv:1710.05941 (Swish activation).

Unit System
-----------
    Input  : acceleration time series (m/s²), shape (B, 1, seq_len)
    Output : IDR per story (dimensionless),   shape (B, n_stories)

Author: Mikisbell
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PINNConfig:
    """Hyper-parameters for the Hybrid-PINN model.

    Attributes
    ----------
    seq_len : int
        Length of acceleration input sequence (samples).
    n_stories : int
        Number of stories → output dimension.
    enc_channels : tuple[int, ...]
        Channel progression for 1D-CNN encoder layers.
    enc_kernels : tuple[int, ...]
        Kernel sizes for encoder conv layers.
    pool_size : int
        Adaptive average-pool output length after encoder.
    fc_dims : tuple[int, ...]
        Hidden dimensions for FC head.
    dropout : float
        Dropout probability in FC head (0 = no dropout).
    """

    seq_len: int = 2048
    n_stories: int = 3
    enc_channels: tuple[int, ...] = (32, 64, 128)
    enc_kernels: tuple[int, ...] = (7, 5, 3)
    pool_size: int = 16
    fc_dims: tuple[int, ...] = (256, 128, 64, 32)
    dropout: float = 0.05
    use_attention: bool = True
    attn_heads: int = 4
    attn_dropout: float = 0.1
    output_sequence: bool = False  # v2.0: Predict full time history (Seq2Seq)


# ═══════════════════════════════════════════════════════════════════════════
# 1D-CNN Encoder
# ═══════════════════════════════════════════════════════════════════════════


class TemporalEncoder(nn.Module):
    """1D-CNN encoder for seismic acceleration time series.

    Extracts local temporal patterns (wave-front arrivals, spectral content)
    from raw acceleration signals via strided convolutions.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for single-component ground motion).
    channels : tuple[int, ...]
        Output channels for each conv layer.
    kernels : tuple[int, ...]
        Kernel sizes for each conv layer.
    pool_size : int
        Output length of the adaptive average-pooling layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128),
        kernels: tuple[int, ...] = (7, 5, 3),
        pool_size: int = 16,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernels), "channels and kernels must match"

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k in zip(channels, kernels, strict=False):
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=k, stride=2, padding=k // 2),
                    nn.BatchNorm1d(c_out),
                    nn.SiLU(inplace=True),
                ]
            )
            c_in = c_out

        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.out_features = channels[-1] * pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, seq_len) — single-channel acceleration.

        Returns
        -------
        torch.Tensor
            Shape (B, out_features) — flattened feature vector.
        """
        z = self.net(x)  # (B, C_last, T_reduced)
        z = self.pool(z)  # (B, C_last, pool_size)
        return z.view(z.size(0), -1)


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Self-Attention
# ═══════════════════════════════════════════════════════════════════════════


class TemporalAttention(nn.Module):
    """Multi-head self-attention over CNN feature map positions.

    Operates on the encoder output *before* flattening, treating the
    ``pool_size`` positions as a sequence of tokens.  This allows the
    network to learn **which temporal regions** of the ground motion
    contribute most to each story's drift.

    Parameters
    ----------
    embed_dim : int
        Dimension of each token (= last encoder channel count).
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, T) — CNN encoder output before flattening.

        Returns
        -------
        torch.Tensor
            Shape (B, C, T) — attention-refined feature map.
        """
        # (B, C, T) → (B, T, C) for attention
        x_t = x.transpose(1, 2)

        # Multi-head self-attention with residual
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + attn_out)

        # Feed-forward with residual
        x_t = self.norm2(x_t + self.ff(x_t))

        # (B, T, C) → (B, C, T)
        return x_t.transpose(1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# FC Regression Head
# ═══════════════════════════════════════════════════════════════════════════


class RegressionHead(nn.Module):
    """Fully-connected head mapping encoded features to IDR predictions.

    Parameters
    ----------
    in_features : int
        Dimension of encoder output.
    fc_dims : tuple[int, ...]
        Hidden layer dimensions.
    n_outputs : int
        Output dimension (= n_stories).
    dropout : float
        Dropout probability applied after the first two FC layers.
    """

    def __init__(
        self,
        in_features: int,
        fc_dims: tuple[int, ...] = (256, 128, 64, 32),
        n_outputs: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim_in = in_features
        for i, dim_out in enumerate(fc_dims):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.SiLU(inplace=True))
            # Dropout only on the wider layers (first two)
            if i < 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ═══════════════════════════════════════════════════════════════════════════
# Full PINN Model
# ═══════════════════════════════════════════════════════════════════════════


class HybridPINN(nn.Module):
    """Hybrid Physics-Informed Neural Network for seismic IDR prediction.

    Combines a 1D-CNN temporal encoder with an FC regression head.
    The model is "physics-guided" through FEM training data (f_int, mass_matrix)
    and physics-tensor-informed story weights derived from OpenSeesPy NLTHA.
    The architecture itself is a standard encoder-regressor (1D-CNN + FC head).

    Parameters
    ----------
    config : PINNConfig
        Model hyper-parameters.

    Examples
    --------
    >>> cfg = PINNConfig(seq_len=2048, n_stories=5)
    >>> model = HybridPINN(cfg)
    >>> x = torch.randn(4, 1, 2048)      # batch of 4 accelerograms
    >>> y = model(x)                       # (4, 5) IDR predictions
    >>> y.shape
    torch.Size([4, 5])
    """

    def __init__(self, config: PINNConfig | None = None) -> None:
        super().__init__()
        self.config = config or PINNConfig()

        self.encoder = TemporalEncoder(
            in_channels=1,
            channels=self.config.enc_channels,
            kernels=self.config.enc_kernels,
            pool_size=self.config.pool_size,
        )

        # Optional temporal self-attention between encoder and head
        # v1.6: Attention now preserves sequence length, pooling happens AFTER attention
        self.attention: TemporalAttention | None = None
        if self.config.use_attention:
            self.attention = TemporalAttention(
                embed_dim=self.config.enc_channels[-1],
                n_heads=self.config.attn_heads,
                dropout=self.config.attn_dropout,
            )

        # Head input features depends on output mode
        if self.config.output_sequence:
            # Seq2Seq: Head applied per time-step, so input is just C
            head_in_features = self.config.enc_channels[-1]
        else:
            # Scalar: Input is flattened pooled vector (C * pool_size)
            head_in_features = self.encoder.out_features

        self.head = RegressionHead(
            in_features=head_in_features,
            fc_dims=self.config.fc_dims,
            n_outputs=self.config.n_stories,
            dropout=self.config.dropout,
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info("HybridPINN initialised: %d parameters", n_params)

    def _init_weights(self) -> None:
        """Kaiming initialisation for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: acceleration → IDR predictions.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, seq_len) — ground acceleration time series.

        Returns
        -------
        torch.Tensor
            Shape (B, n_stories) — predicted inter-story drift ratio per story.
        """
        if self.config.output_sequence:
            # ── v2.0 Seq2Seq Mode ──────────────────────────────────────
            # x: (B, 1, T) -> z: (B, C, T_reduced)
            t_in = x.size(2)
            z = self.encoder.net(x)

            if self.attention is not None:
                z = self.attention(z)  # (B, C, T_reduced)

            # Upsample to original sequence length
            z = nn.functional.interpolate(
                z, size=t_in, mode="linear", align_corners=False
            )  # (B, C, T)

            # Prepare for per-step regression: (B, C, T) -> (B, T, C)
            z = z.transpose(1, 2)

            # Apply head (MLP) to each time step
            # Linear accepts (B, *, H_in), acts on last dim
            out = self.head(z)  # (B, T, N)

            # (B, T, N) -> (B, N, T)
            return out.transpose(1, 2)

        else:
            # ── v1.6 Scalar Mode ───────────────────────────────────────
            if self.attention is not None:
                # 1. Feature extraction
                z = self.encoder.net(x)  # (B, C, T_reduced)
                # 2. Temporal Attention
                z = self.attention(z)  # (B, C, T_reduced)
                # 3. Pooling
                z = self.encoder.pool(z)  # (B, C, pool_size)
                # 4. Flatten
                z = z.reshape(z.size(0), -1)
            else:
                # Standard path: features -> pool -> flatten
                z = self.encoder(x)  # (B, C * pool_size)
            return self.head(z)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a human-readable model summary."""
        lines = [
            "HybridPINN Summary",
            "=" * 50,
            f"  Encoder channels  : {self.config.enc_channels}",
            f"  Encoder kernels   : {self.config.enc_kernels}",
            f"  Pool size         : {self.config.pool_size}",
            f"  FC dimensions     : {self.config.fc_dims}",
            f"  Output stories    : {self.config.n_stories}",
            f"  Dropout           : {self.config.dropout}",
            f"  Input seq_len     : {self.config.seq_len}",
            f"  Trainable params  : {self.count_parameters():,}",
            "=" * 50,
        ]
        return "\n".join(lines)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> HybridPINN:
        """Load model from a saved checkpoint.

        Parameters
        ----------
        path : str
            Path to .pt checkpoint file (saved via trainer.py).
        device : str
            Target device for loading.

        Returns
        -------
        HybridPINN
            Loaded model in eval mode.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt.get("config", PINNConfig())
        if isinstance(config, dict):
            config = PINNConfig(**config)
        model = cls(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info("Loaded HybridPINN from %s (%d params)", path, model.count_parameters())
        return model


# ═══════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════


def build_pinn(
    seq_len: int = 2048,
    n_stories: int = 3,
    dropout: float = 0.05,
) -> HybridPINN:
    """Build a HybridPINN with sensible defaults.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    n_stories : int
        Number of building stories (output dimension).
    dropout : float
        Dropout rate for FC head.

    Returns
    -------
    HybridPINN
        Constructed model ready for training.
    """
    cfg = PINNConfig(seq_len=seq_len, n_stories=n_stories, dropout=dropout)
    return HybridPINN(cfg)
