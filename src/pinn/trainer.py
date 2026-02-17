"""
trainer.py — PINN Training Loop with HRPUB-Reproducible Protocol
=================================================================

Implements the full training pipeline for the HybridPINN model:

    Optimizer  : AdamW (lr=1e-3, weight_decay=1e-4)
    Scheduler  : CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
    Validation : 70/15/15 split (handled upstream by NLTHAPipeline)
    Early Stop : Patience = 50 epochs monitoring validation loss
    Logging    : Per-epoch losses, LR, and wall-clock time
    Checkpoints: Best model saved to data/models/pinn_best.pt

The trainer supports three modes:
    1. **Data-only** : L = L_data  (pure supervised, no physics constraint)
    2. **Hybrid**    : L = λ_d·L_data + λ_p·L_physics + λ_b·L_bc
    3. **Adaptive**  : Same as Hybrid with self-adaptive weight balancing

References
----------
    [3] Raissi, Perdikaris, Karniadakis (2019). J. Comput. Phys., 378.
    [11] Loshchilov, Hutter (2019). Decoupled Weight Decay. ICLR 2019.
    [12] Loshchilov, Hutter (2017). SGDR: Warm Restarts. ICLR 2017.

Author: Mikisbell
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pinn.loss import AdaptiveLossWeights, HybridPINNLoss, LossWeights
from src.pinn.model import HybridPINN, PINNConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    """Hyper-parameters for the training loop.

    All values chosen following the PI specification and HRPUB
    reproducibility requirements.
    """

    # Optimiser
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Scheduler — Cosine annealing with warm restarts
    scheduler_t0: int = 50
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6

    # Training
    epochs: int = 500
    batch_size: int = 64
    grad_clip_norm: float = 1.0  # Max gradient norm (0 = disabled)

    # Early stopping
    patience: int = 50

    # Checkpoints
    checkpoint_dir: str = "data/models"
    save_every: int = 50  # Save intermediate checkpoint every N epochs

    # Loss weights
    lambda_data: float = 1.0
    lambda_phys: float = 0.1
    lambda_bc: float = 0.01
    adaptive_weights: bool = False

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cpu"  # "cpu" or "cuda"

    # Logging
    log_every: int = 10  # Print metrics every N epochs


# ═══════════════════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════════════════


class EarlyStopping:
    """Monitor validation loss and stop when improvement stalls.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before stopping.
    min_delta : float
        Minimum improvement to qualify as an improvement.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float("inf")
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        bool
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered at patience=%d (best_val=%.6f)",
                    self.patience,
                    self.best_loss,
                )
        return self.should_stop


# ═══════════════════════════════════════════════════════════════════════════
# Training History
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainHistory:
    """Records per-epoch metrics for analysis and plotting."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_l_data: list[float] = field(default_factory=list)
    train_l_physics: list[float] = field(default_factory=list)
    train_l_bc: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    epoch_time_s: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time_s: float = 0.0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        components: dict[str, float],
        lr: float,
        elapsed: float,
    ) -> None:
        """Append one epoch of metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_l_data.append(components.get("L_data", 0.0))
        self.train_l_physics.append(components.get("L_physics", 0.0))
        self.train_l_bc.append(components.get("L_bc", 0.0))
        self.learning_rate.append(lr)
        self.epoch_time_s.append(elapsed)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def to_dict(self) -> dict[str, Any]:
        """Serialise history to a JSON-compatible dict."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_l_data": self.train_l_data,
            "train_l_physics": self.train_l_physics,
            "train_l_bc": self.train_l_bc,
            "learning_rate": self.learning_rate,
            "epoch_time_s": self.epoch_time_s,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time_s": self.total_time_s,
        }

    def save(self, path: str | Path) -> None:
        """Save history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Training history saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════


class PINNTrainer:
    """End-to-end training loop for the HybridPINN.

    Parameters
    ----------
    model : HybridPINN
        The neural network to train.
    config : TrainConfig
        Training hyper-parameters.

    Examples
    --------
    >>> model = HybridPINN(PINNConfig())
    >>> trainer = PINNTrainer(model, TrainConfig(epochs=100))
    >>> history = trainer.fit(train_loader, val_loader)
    >>> trainer.save_checkpoint("data/models/pinn_best.pt")
    """

    def __init__(
        self,
        model: HybridPINN,
        config: TrainConfig | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        # Loss function
        weights = LossWeights(
            lambda_data=self.config.lambda_data,
            lambda_phys=self.config.lambda_phys,
            lambda_bc=self.config.lambda_bc,
        )
        self.loss_fn = HybridPINNLoss(weights)

        # Adaptive weight balancing (optional)
        self.adaptive: AdaptiveLossWeights | None = None
        if self.config.adaptive_weights:
            self.adaptive = AdaptiveLossWeights(self.loss_fn)
            logger.info("Adaptive loss weight balancing enabled")

        # Optimiser: AdamW (decoupled weight decay, Loshchilov & Hutter 2019)
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler: Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser,
            T_0=self.config.scheduler_t0,
            T_mult=self.config.scheduler_t_mult,
            eta_min=self.config.scheduler_eta_min,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.config.patience)

        # History
        self.history = TrainHistory()

        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        logger.info(
            "PINNTrainer initialised: device=%s, epochs=%d, lr=%.1e, patience=%d",
            self.device,
            self.config.epochs,
            self.config.lr,
            self.config.patience,
        )

    # ── Training step ──────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        """Run one training epoch.

        Returns
        -------
        epoch_loss : float
            Average total loss over all batches.
        avg_components : dict
            Average per-component losses.
        """
        self.model.train()
        total_loss = 0.0
        comp_sums: dict[str, float] = {"L_data": 0.0, "L_physics": 0.0, "L_bc": 0.0}
        n_batches = 0

        for batch in loader:
            # Unpack batch — support variable-length tuples
            x, y_target = batch[0].to(self.device), batch[1].to(self.device)

            # Optional physics tensors (positions 2-7 in batch tuple)
            physics_kwargs: dict[str, torch.Tensor] = {}
            if len(batch) > 2 and batch[2] is not None:
                physics_kwargs["mass_matrix"] = batch[2].to(self.device)
            if len(batch) > 3 and batch[3] is not None:
                physics_kwargs["damping_matrix"] = batch[3].to(self.device)
            if len(batch) > 4 and batch[4] is not None:
                physics_kwargs["accel_response"] = batch[4].to(self.device)
            if len(batch) > 5 and batch[5] is not None:
                physics_kwargs["vel_response"] = batch[5].to(self.device)
            if len(batch) > 6 and batch[6] is not None:
                physics_kwargs["f_int"] = batch[6].to(self.device)
            if len(batch) > 7 and batch[7] is not None:
                physics_kwargs["ground_accel"] = batch[7].to(self.device)

            # Forward pass
            y_pred = self.model(x)

            # Compute loss
            loss, components = self.loss_fn(
                pred=y_pred,
                target=y_target,
                **physics_kwargs,
            )

            # Backward pass
            self.optimiser.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

            self.optimiser.step()

            # Adaptive weight update
            if self.adaptive is not None:
                self.adaptive.step(components)

            # Accumulate
            total_loss += loss.item()
            for key in comp_sums:
                comp_sums[key] += components[key].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_comps = {k: v / max(n_batches, 1) for k, v in comp_sums.items()}
        return avg_loss, avg_comps

    # ── Validation step ────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Compute validation loss (data-only MSE for early stopping).

        Returns
        -------
        float
            Average validation MSE loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x = batch[0].to(self.device)
            y_target = batch[1].to(self.device)
            y_pred = self.model(x)
            loss = nn.functional.mse_loss(y_pred, y_target)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── Main training loop ─────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainHistory:
        """Train the model for the configured number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            Training data: each batch is (x, y_target, *optional_physics).
        val_loader : DataLoader
            Validation data: each batch is (x, y_target).

        Returns
        -------
        TrainHistory
            Per-epoch metrics.
        """
        logger.info(
            "Starting training: %d epochs, %d batches/epoch", self.config.epochs, len(train_loader)
        )

        t_start = time.perf_counter()
        best_state: dict[str, Any] = {}

        for epoch in range(1, self.config.epochs + 1):
            t_epoch = time.perf_counter()

            # Train
            train_loss, train_comps = self._train_epoch(train_loader)

            # Validate
            val_loss = self._validate(val_loader)

            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimiser.param_groups[0]["lr"]

            # Record history
            elapsed = time.perf_counter() - t_epoch
            self.history.update(epoch, train_loss, val_loss, train_comps, current_lr, elapsed)

            # Save best model (check best_epoch since update() already set it)
            if self.history.best_epoch == epoch:
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    },
                    "optimiser_state_dict": self.optimiser.state_dict(),
                    "val_loss": val_loss,
                    "config": asdict(self.model.config) if self.model.config else {},
                    "train_config": asdict(self.config),
                }

            # Periodic logging
            if epoch % self.config.log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — train=%.6f  val=%.6f  "
                    "[data=%.4e  phys=%.4e  bc=%.4e]  lr=%.2e  (%.1fs)",
                    epoch,
                    self.config.epochs,
                    train_loss,
                    val_loss,
                    train_comps["L_data"],
                    train_comps["L_physics"],
                    train_comps["L_bc"],
                    current_lr,
                    elapsed,
                )

            # Periodic checkpoint
            if self.config.save_every > 0 and epoch % self.config.save_every == 0:
                self._save_intermediate(epoch)

            # Early stopping check
            if self.early_stopping.step(val_loss):
                logger.info("Training stopped at epoch %d", epoch)
                break

        # End of training
        self.history.total_time_s = time.perf_counter() - t_start

        # Restore and save best model
        if best_state:
            self.model.load_state_dict(best_state["model_state_dict"])
            ckpt_path = Path(self.config.checkpoint_dir) / "pinn_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
            logger.info(
                "Best model saved: epoch=%d, val_loss=%.6f → %s",
                best_state["epoch"],
                best_state["val_loss"],
                ckpt_path,
            )

        # Save training history
        hist_path = Path(self.config.checkpoint_dir) / "train_history.json"
        self.history.save(hist_path)

        logger.info(
            "Training complete: %d epochs in %.1fs (best epoch=%d, val=%.6f)",
            len(self.history.train_loss),
            self.history.total_time_s,
            self.history.best_epoch,
            self.history.best_val_loss,
        )
        return self.history

    # ── Checkpoint helpers ─────────────────────────────────────────────

    def _save_intermediate(self, epoch: int) -> None:
        """Save an intermediate checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"pinn_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "config": asdict(self.model.config) if self.model.config else {},
            },
            path,
        )
        logger.debug("Intermediate checkpoint: %s", path)

    def save_checkpoint(self, path: str | Path) -> None:
        """Manually save current model state.

        Parameters
        ----------
        path : str or Path
            Target .pt file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": asdict(self.model.config) if self.model.config else {},
                "train_config": asdict(self.config),
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: create data loaders from tensors
# ═══════════════════════════════════════════════════════════════════════════


def create_loaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders from pre-processed tensor arrays.

    Parameters
    ----------
    x_train, y_train : torch.Tensor
        Training inputs/targets.
    x_val, y_val : torch.Tensor
        Validation inputs/targets.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes (0 = main thread).

    Returns
    -------
    train_loader, val_loader : DataLoader
        Ready for :meth:`PINNTrainer.fit`.
    """
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
# Quick-start (synthetic data for CI / smoke test)
# ═══════════════════════════════════════════════════════════════════════════


def smoke_test(epochs: int = 5, batch_size: int = 16) -> TrainHistory:
    """Run a minimal training loop with synthetic data.

    Useful for CI pipelines and verifying that all components integrate
    correctly before running with real NLTHA data.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for synthetic data.

    Returns
    -------
    TrainHistory
        Training metrics.
    """
    logger.info("Running PINN smoke test (synthetic data)...")

    cfg = PINNConfig(seq_len=512, n_stories=5)
    model = HybridPINN(cfg)

    # Synthetic data: random accelerograms → random IDR targets
    n_train, n_val = 128, 32
    x_train = torch.randn(n_train, 1, cfg.seq_len)
    y_train = torch.rand(n_train, cfg.n_stories) * 0.05  # IDR 0-5%
    x_val = torch.randn(n_val, 1, cfg.seq_len)
    y_val = torch.rand(n_val, cfg.n_stories) * 0.05

    train_loader, val_loader = create_loaders(x_train, y_train, x_val, y_val, batch_size=batch_size)

    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        patience=epochs + 1,  # No early stopping for smoke test
        log_every=1,
        checkpoint_dir="data/models/smoke_test",
    )

    trainer = PINNTrainer(model, train_cfg)
    history = trainer.fit(train_loader, val_loader)

    logger.info(
        "Smoke test complete: final_train=%.6f, final_val=%.6f",
        history.train_loss[-1],
        history.val_loss[-1],
    )
    return history


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    smoke_test(epochs=10)
