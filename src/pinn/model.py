"""
model.py — Hybrid-PINN Architecture for Seismic RC Response Prediction
=======================================================================

Implements the production PINN model: a 1D-CNN temporal encoder followed
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

        layers.append(nn.AdaptiveAvgPool1d(pool_size))
        self.net = nn.Sequential(*layers)
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
        z = self.net(x)  # (B, C_last, pool_size)
        return z.view(z.size(0), -1)


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
    The model is "physics-informed" through the loss function (see loss.py),
    not through architectural constraints — the architecture itself is a
    standard encoder-regressor.

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

        self.head = RegressionHead(
            in_features=self.encoder.out_features,
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
        z = self.encoder(x)
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
