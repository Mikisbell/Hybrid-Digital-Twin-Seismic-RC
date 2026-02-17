"""
evaluate.py — Evaluate the trained HybridPINN on the held-out test set
======================================================================

Produces:
    1. Per-story RMSE and R² metrics (physical units)
    2. Figure 4 — Training and validation loss curves
    3. Figure 5 — Predicted vs. actual peak IDR scatter plot
    4. Figure 6 — Per-story error distribution (box plot)

All figures are saved at ≥300 DPI via ``FigureManager`` for direct
inclusion in the HRPUB manuscript.

Usage::

    python -m src.pinn.evaluate

Author: Mikisbell
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pinn.model import HybridPINN
from src.utils.figure_manager import FigureManager

logger = logging.getLogger(__name__)

STORY_LABELS = ["Story 1", "Story 2", "Story 3", "Story 4", "Story 5"]
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
FIG_DIR = Path("manuscript/figures")


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════


def _load_scaler() -> tuple[np.ndarray, np.ndarray]:
    """Load target mean and std for denormalization."""
    with open(PROCESSED_DIR / "scaler_params.json") as f:
        params = json.load(f)
    mean = np.array(params["target"]["mean"])
    std = np.array(params["target"]["std"])
    return mean, std


def _denormalize(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert standardized IDR back to physical values."""
    return y * std + mean


def _load_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load test tensors."""
    data = torch.load(PROCESSED_DIR / "test.pt", weights_only=True)
    return data["x"], data["y"]


def _load_model() -> HybridPINN:
    """Load the best checkpoint."""
    ckpt_path = MODELS_DIR / "pinn_best.pt"
    model = HybridPINN.from_checkpoint(ckpt_path)
    model.eval()
    logger.info("Loaded model from %s", ckpt_path)
    return model


def _load_history() -> dict:
    """Load training history."""
    with open(MODELS_DIR / "train_history.json") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    """Compute RMSE and R² per story and overall.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape (N, 5), physical IDR values.

    Returns
    -------
    dict with keys: rmse_per_story, r2_per_story, rmse_overall, r2_overall
    """
    # Per-story
    rmse_list: list[float] = []
    r2_list: list[float] = []
    for i in range(y_true.shape[1]):
        diff = y_true[:, i] - y_pred[:, i]
        rmse = float(np.sqrt(np.mean(diff**2)))
        ss_res = float(np.sum(diff**2))
        ss_tot = float(np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse_list.append(rmse)
        r2_list.append(r2)

    # Overall (flatten)
    diff_all = y_true.flatten() - y_pred.flatten()
    rmse_all = float(np.sqrt(np.mean(diff_all**2)))
    ss_res_all = float(np.sum(diff_all**2))
    ss_tot_all = float(np.sum((y_true.flatten() - y_true.flatten().mean()) ** 2))
    r2_all = 1.0 - ss_res_all / max(ss_tot_all, 1e-12)

    return {
        "rmse_per_story": rmse_list,
        "r2_per_story": r2_list,
        "rmse_overall": rmse_all,
        "r2_overall": r2_all,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════


def plot_loss_curves(history: dict, fm: FigureManager) -> None:
    """Figure 4: Training and validation loss curves."""
    fig, ax = plt.subplots(figsize=FigureManager.DOUBLE_COLUMN)

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.semilogy(epochs, history["train_loss"], "b-", linewidth=1.0, label="Train loss")
    ax.semilogy(epochs, history["val_loss"], "r--", linewidth=1.0, label="Val loss")

    best_ep = history["best_epoch"]
    best_val = history["best_val_loss"]
    ax.axvline(best_ep, color="green", linestyle=":", alpha=0.7, label=f"Best epoch ({best_ep})")
    ax.plot(best_ep, best_val, "g*", markersize=12, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("HybridPINN Training Convergence")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, alpha=0.3)

    fm.save(
        fig,
        caption=(
            "Training and validation loss convergence for the HybridPINN model. "
            f"Best validation loss ({best_val:.4f}) reached at epoch {best_ep}. "
            "Training used AdamW optimizer with cosine annealing warm restarts."
        ),
        label="loss_curves",
    )
    plt.close(fig)


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    fm: FigureManager,
) -> None:
    """Figure 5: Predicted vs. actual peak IDR scatter."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.0), sharey=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (ax, color) in enumerate(zip(axes, colors, strict=False)):
        ax.scatter(
            y_true[:, i] * 100,
            y_pred[:, i] * 100,
            c=color,
            alpha=0.6,
            s=25,
            edgecolors="white",
            linewidths=0.3,
        )

        # Perfect prediction line
        lims = [0, max(y_true[:, i].max(), y_pred[:, i].max()) * 100 * 1.1]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Actual IDR (%)")
        if i == 0:
            ax.set_ylabel("Predicted IDR (%)")
        ax.set_title(STORY_LABELS[i], fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # R² annotation
        r2 = metrics["r2_per_story"][i]
        rmse = metrics["rmse_per_story"][i] * 100  # %
        ax.text(
            0.05,
            0.92,
            f"R²={r2:.3f}\nRMSE={rmse:.3f}%",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )

    fig.suptitle("Predicted vs. Actual Peak Inter-Story Drift Ratio", fontsize=11, y=1.02)

    fm.save(
        fig,
        caption=(
            "Predicted vs. actual peak inter-story drift ratio (IDR) for each story "
            f"on the held-out test set (N={y_true.shape[0]}). "
            f"Overall RMSE = {metrics['rmse_overall'] * 100:.3f}%, "
            f"R² = {metrics['r2_overall']:.3f}."
        ),
        label="pred_vs_actual",
    )
    plt.close(fig)


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fm: FigureManager,
) -> None:
    """Figure 6: Per-story error distribution (box plot)."""
    fig, ax = plt.subplots(figsize=FigureManager.DOUBLE_COLUMN)

    # Percentage error per story
    errors = (y_pred - y_true) * 100  # Convert from ratio to %
    error_list = [errors[:, i] for i in range(errors.shape[1])]

    bp = ax.boxplot(
        error_list,
        tick_labels=STORY_LABELS,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Prediction Error (IDR %)")
    ax.set_title("Per-Story Prediction Error Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    fm.save(
        fig,
        caption=(
            "Box plot of prediction errors (predicted − actual IDR) for each story "
            "on the held-out test set. The dashed line indicates zero error."
        ),
        label="error_distribution",
    )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data and model
    x_test, y_test_norm = _load_test_data()
    model = _load_model()
    history = _load_history()
    mean, std = _load_scaler()

    # Predict
    with torch.no_grad():
        y_pred_norm = model(x_test).numpy()
    y_test_norm_np = y_test_norm.numpy()

    # Denormalize to physical IDR
    y_true = _denormalize(y_test_norm_np, mean, std)
    y_pred = _denormalize(y_pred_norm, mean, std)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION (N=%d)", y_true.shape[0])
    logger.info("=" * 60)
    for i in range(5):
        logger.info(
            "  %s: RMSE = %.5f (%.3f%%)  R² = %.4f",
            STORY_LABELS[i],
            metrics["rmse_per_story"][i],
            metrics["rmse_per_story"][i] * 100,
            metrics["r2_per_story"][i],
        )
    logger.info("-" * 60)
    logger.info(
        "  Overall: RMSE = %.5f (%.3f%%)  R² = %.4f",
        metrics["rmse_overall"],
        metrics["rmse_overall"] * 100,
        metrics["r2_overall"],
    )
    logger.info("=" * 60)

    # Save metrics to JSON
    metrics_path = MODELS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Generate publication figures
    fm = FigureManager(output_dir=str(FIG_DIR), dpi=300)

    plot_loss_curves(history, fm)
    plot_pred_vs_actual(y_true, y_pred, metrics, fm)
    plot_error_distribution(y_true, y_pred, fm)

    logger.info("Publication figures saved to %s", FIG_DIR)


if __name__ == "__main__":
    main()
