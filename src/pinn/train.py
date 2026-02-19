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
from typing import Any

import torch

from src.pinn.model import HybridPINN, PINNConfig
from src.pinn.trainer import PINNTrainer, TrainConfig, create_loaders

logger = logging.getLogger(__name__)


def load_processed(
    processed_dir: str = "data/processed",
) -> dict[str, Any]:
    """Load train/val/test data from the pipeline output.

    Returns
    -------
    dict
         Dictionary with keys 'train', 'val', 'test', each containing
         the data dictionary loaded from .pt files.
    """
    d = Path(processed_dir)
    data = {}
    for split in ["train", "val", "test"]:
        p = d / f"{split}.pt"
        if p.exists():
            data[split] = torch.load(p, weights_only=True)
        else:
            # Fallback for old pipeline runs or missing test set
            logger.warning("Missing processed file: %s", p)
            if split == "test" and "val" in data:
                # Use val as test if missing
                data["test"] = data["val"]

    return data


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
    parser.add_argument("--lambda-phys", type=float, default=0.1, help="Physics loss weight")
    parser.add_argument(
        "--n-stories", type=int, default=5, help="Number of building stories (output dim)"
    )
    parser.add_argument("--output-sequence", action="store_true", help="Predict full time history")
    parser.add_argument(
        "--experiment-name", type=str, default="pinn_experiment", help="Experiment name for logging"
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    logger.info("Loading processed data from %s", args.processed_dir)
    data = load_processed(args.processed_dir)

    # Check shapes
    if "train" in data and "x" in data["train"]:
        x_shape = data["train"]["x"].shape
        y_shape = data["train"]["y"].shape
        logger.info("Train: x=%s y=%s", list(x_shape), list(y_shape))

    # Extract global physics constants (mass matrix)
    mass_matrix = None
    if "train" in data and "mass_matrix" in data["train"]:
        mass_matrix = data["train"]["mass_matrix"]
        logger.info("Loaded mass matrix: %s", list(mass_matrix.shape))

    # Compute per-story inverse-variance weights from scaler params
    story_weights = None
    scaler_path = Path(args.processed_dir) / "scaler_params.json"
    if scaler_path.exists():
        import json

        with open(scaler_path) as f:
            scaler_params = json.load(f)
        if "target" in scaler_params and "std" in scaler_params["target"]:
            std_per_story = torch.tensor(scaler_params["target"]["std"], dtype=torch.float32)
            # Inverse-variance: w_i = 1/std_i, normalised so sum = n_stories
            inv_std = 1.0 / std_per_story.clamp(min=1e-6)
            story_weights = inv_std * (len(inv_std) / inv_std.sum())
            logger.info("Story weights (inverse-variance): %s", story_weights.tolist())

    # Create DataLoaders
    # create_loaders returns 3 loaders now
    train_loader, val_loader, test_loader = create_loaders(data, batch_size=args.batch_size)

    # Build model
    model_cfg = PINNConfig(
        seq_len=args.seq_len,
        n_stories=args.n_stories,
        output_sequence=args.output_sequence,
    )
    model = HybridPINN(model_cfg)
    logger.info("Model: %d parameters", model.count_parameters())
    # logger.info("\n%s", model.summary())

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
        # Hybrid loss weights
        lambda_data=1.0,
        lambda_phys=args.lambda_phys,
        lambda_bc=0.01,
    )

    # Train
    trainer = PINNTrainer(model, train_cfg, mass_matrix=mass_matrix, story_weights=story_weights)
    history = trainer.train(train_loader, val_loader)

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
        # Get one batch from val_loader
        batch = next(iter(val_loader))
        x_sample = batch[0][:4].to(trainer.device)
        pred = model(x_sample)
        logger.info("Sample predictions (val[:4]):\n%s", pred.cpu().numpy())


if __name__ == "__main__":
    main()
