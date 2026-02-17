"""
train.py — Train the HybridPINN on processed NLTHA data
========================================================

Loads train/val tensors from ``data/processed/``, creates DataLoaders,
and trains the HybridPINN using the protocol specified in the manuscript
(AdamW + cosine annealing + early stopping).

Outputs saved to ``data/models/``:
    - ``pinn_best.pt``       — best checkpoint (lowest val loss)
    - ``train_history.json``  — per-epoch metrics

Usage::

    python -m src.pinn.train                    # default 500 epochs
    python -m src.pinn.train --epochs 200       # shorter run
    python -m src.pinn.train --adaptive         # adaptive loss weights

Author: Mikisbell
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.pinn.model import HybridPINN, PINNConfig
from src.pinn.trainer import PINNTrainer, TrainConfig, create_loaders

logger = logging.getLogger(__name__)


def load_processed(
    processed_dir: str = "data/processed",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load train and val tensors from the pipeline output.

    Returns
    -------
    x_train, y_train, x_val, y_val : torch.Tensor
    """
    d = Path(processed_dir)

    train_data = torch.load(d / "train.pt", weights_only=True)
    val_data = torch.load(d / "val.pt", weights_only=True)

    return train_data["x"], train_data["y"], val_data["x"], val_data["y"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HybridPINN on NLTHA data")
    parser.add_argument("--epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--adaptive", action="store_true", help="Adaptive loss weights")
    parser.add_argument("--seq-len", type=int, default=2048, help="Input sequence length")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data dir")
    parser.add_argument("--checkpoint-dir", default="data/models", help="Model output dir")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    logger.info("Loading processed data from %s", args.processed_dir)
    x_train, y_train, x_val, y_val = load_processed(args.processed_dir)
    logger.info(
        "Train: x=%s y=%s | Val: x=%s y=%s",
        list(x_train.shape),
        list(y_train.shape),
        list(x_val.shape),
        list(y_val.shape),
    )

    # Create DataLoaders
    train_loader, val_loader = create_loaders(
        x_train, y_train, x_val, y_val, batch_size=args.batch_size
    )

    # Build model
    model_cfg = PINNConfig(seq_len=args.seq_len, n_stories=5)
    model = HybridPINN(model_cfg)
    logger.info("Model: %d parameters", model.count_parameters())
    logger.info("\n%s", model.summary())

    # Configure training
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        adaptive_weights=args.adaptive,
        checkpoint_dir=args.checkpoint_dir,
        log_every=10,
        save_every=100,
        # Data-dominated loss (supervised training with light physics)
        lambda_data=1.0,
        lambda_phys=0.01,
        lambda_bc=0.001,
    )

    # Train
    trainer = PINNTrainer(model, train_cfg)
    history = trainer.fit(train_loader, val_loader)

    # Summary
    logger.info(
        "Training complete: best_epoch=%d, best_val=%.6f, total=%.1fs",
        history.best_epoch,
        history.best_val_loss,
        history.total_time_s,
    )

    # Quick inference test
    model.eval()
    with torch.no_grad():
        sample = x_val[:4]
        pred = model(sample)
        logger.info("Sample predictions (val[:4]):\n%s", pred.numpy())


if __name__ == "__main__":
    main()
