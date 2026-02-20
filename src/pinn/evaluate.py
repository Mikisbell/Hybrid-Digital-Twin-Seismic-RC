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

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pinn.model import HybridPINN
from src.utils.figure_manager import FigureManager

logger = logging.getLogger(__name__)


def _story_labels(n: int) -> list[str]:
    """Generate story labels dynamically."""
    return [f"Story {i}" for i in range(1, n + 1)]


PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("manuscript/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HybridPINN model")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory containing trained model checkpoints",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed data (test.pt, scaler_params.json)",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════


def _load_scaler(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load target mean and std for denormalization."""
    with open(data_dir / "scaler_params.json") as f:
        params = json.load(f)
    mean = np.array(params["target"]["mean"])
    std = np.array(params["target"]["std"])
    return mean, std


def _denormalize(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert standardized IDR back to physical values."""
    return y * std + mean


def _load_test_data(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load test tensors."""
    data = torch.load(data_dir / "test.pt", weights_only=True)
    return data["x"], data["y"]


def _load_model(models_dir: Path) -> HybridPINN:
    """Load the best checkpoint."""
    ckpt_path = models_dir / "pinn_best.pt"
    model = HybridPINN.from_checkpoint(ckpt_path)
    model.eval()
    logger.info("Loaded model from %s", ckpt_path)
    return model


def _load_history(models_dir: Path) -> dict:
    """Load training history."""
    with open(models_dir / "train_history.json") as f:
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
    story_labels: list[str] | None = None,
) -> None:
    """Figure 5: Predicted vs. actual peak IDR scatter."""
    n_stories = y_true.shape[1]
    if story_labels is None:
        story_labels = _story_labels(n_stories)
    fig, axes = plt.subplots(1, n_stories, figsize=(2.8 * n_stories, 3.0), sharey=True)

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
        ax.set_title(story_labels[i], fontsize=9)
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
    story_labels: list[str] | None = None,
) -> None:
    """Figure 6: Per-story error distribution (box plot)."""
    n_stories = y_true.shape[1]
    if story_labels is None:
        story_labels = _story_labels(n_stories)
    fig, ax = plt.subplots(figsize=FigureManager.DOUBLE_COLUMN)

    # Percentage error per story
    errors = (y_pred - y_true) * 100  # Convert from ratio to %
    error_list = [errors[:, i] for i in range(errors.shape[1])]

    bp = ax.boxplot(
        error_list,
        tick_labels=story_labels,
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


# ─────────────────────────────────────────────────────────────────────────────
# N-story accuracy profile (Figure 7)
# ─────────────────────────────────────────────────────────────────────────────

# Known N=3 baseline results (v1.6, PEER real data)
_N3_R2 = [0.763, 0.758, 0.549]

# Theoretical upper-bound envelope for N=10 (Section 5.4.3).
# Based on modal participation argument: first-mode dominated floors retain
# high accuracy; whiplash zone (floors 8-10) degrades. Updated automatically
# when N=10 test_metrics.json is available.
_N10_R2_THEORY_LOW = [0.70, 0.68, 0.65, 0.60, 0.57, 0.55, 0.52, 0.49, 0.47, 0.45]
_N10_R2_THEORY_HIGH = [0.80, 0.78, 0.75, 0.70, 0.67, 0.64, 0.60, 0.57, 0.54, 0.52]


def plot_r2_accuracy_profile(
    fm: FigureManager,
    n10_metrics_path: Path | None = None,
) -> None:
    """Figure 7: R² per story for N=3 (measured) and N=10 (measured or predicted).

    Shows the characteristic accuracy degradation toward the whiplash zone
    (upper stories) driven by higher-mode participation.  When real N=10
    metrics are available via *n10_metrics_path*, they replace the theoretical
    envelope.

    Parameters
    ----------
    fm:
        FigureManager instance for saving at publication DPI.
    n10_metrics_path:
        Optional path to ``test_metrics.json`` for the N=10 model.
        If None or missing, the theoretical envelope is plotted as a
        shaded region labelled "Expected range (N=10)".
    """
    fig, ax = plt.subplots(figsize=FigureManager.DOUBLE_COLUMN)

    # ── N=3 measured ──────────────────────────────────────────────────────────
    stories_n3 = list(range(1, len(_N3_R2) + 1))
    ax.plot(
        stories_n3,
        _N3_R2,
        "o-",
        color="#1f77b4",
        linewidth=1.8,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        label=f"N=3 (measured, v1.6)  $R^2_{{overall}}={np.mean(_N3_R2):.3f}$",
        zorder=4,
    )

    # ── N=10: measured or theoretical envelope ────────────────────────────────
    n10_r2_measured: list[float] | None = None
    if n10_metrics_path is not None and Path(n10_metrics_path).exists():
        with open(n10_metrics_path) as f:
            n10_data = json.load(f)
        n10_r2_measured = n10_data.get("r2_per_story")

    if n10_r2_measured is not None:
        stories_n10 = list(range(1, len(n10_r2_measured) + 1))
        ax.plot(
            stories_n10,
            n10_r2_measured,
            "s-",
            color="#d62728",
            linewidth=1.8,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            label=f"N=10 (measured, v2.0)  $R^2_{{overall}}={np.mean(n10_r2_measured):.3f}$",
            zorder=4,
        )
        n10_label = "measured"
    else:
        # Theoretical envelope (shaded band)
        stories_n10 = list(range(1, 11))
        ax.fill_between(
            stories_n10,
            _N10_R2_THEORY_LOW,
            _N10_R2_THEORY_HIGH,
            color="#d62728",
            alpha=0.20,
            label="N=10 — Expected range (Section 5.4.3)",
            zorder=2,
        )
        ax.plot(
            stories_n10,
            [(lo + hi) / 2 for lo, hi in zip(_N10_R2_THEORY_LOW, _N10_R2_THEORY_HIGH, strict=True)],
            "--",
            color="#d62728",
            linewidth=1.2,
            alpha=0.7,
            zorder=3,
        )
        n10_label = "predicted"

    # ── Zone annotations ──────────────────────────────────────────────────────
    ax.axvspan(7.5, 10.5, alpha=0.06, color="#ff7f0e", zorder=1)
    ax.text(
        9.0,
        ax.get_ylim()[0] + 0.02 if ax.get_ylim()[0] > 0 else 0.42,
        "Whiplash\nzone",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#d62728",
        fontstyle="italic",
    )

    # ── Collapse-level reference ───────────────────────────────────────────────
    ax.axhline(
        0.5,
        color="gray",
        linestyle=":",
        linewidth=0.9,
        alpha=0.6,
        label="$R^2 = 0.50$ (minimum acceptable)",
    )

    ax.set_xlabel("Story Number")
    ax.set_ylabel("Coefficient of Determination ($R^2$)")
    ax.set_title(
        f"Per-Story Prediction Accuracy: N=3 (measured) vs N=10 ({n10_label})",
        pad=10,
    )
    ax.set_xlim(0.5, max(len(stories_n3), len(stories_n10)) + 0.5)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks(range(1, max(len(stories_n3), len(stories_n10)) + 1))
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", fontsize=8)
    ax.grid(True, alpha=0.3)

    fm.save(
        fig,
        caption=(
            "Per-story $R^2$ accuracy profile comparing the N=3 (v1.6, PEER real data) "
            "measured results with the N=10 (v2.0) "
            + ("measured results. " if n10_r2_measured else "theoretical prediction envelope. ")
            + "The shaded orange region (Floors 8–10) marks the whiplash zone where "
            "higher-mode participation degrades prediction accuracy. "
            "The dashed grey line indicates $R^2 = 0.50$ (minimum acceptable threshold)."
        ),
        label="r2_accuracy_profile",
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
    args = parse_args()
    model_dir = args.model_dir
    processed_dir = args.processed_dir

    # Load data and model
    x_test, y_test_norm = _load_test_data(processed_dir)
    model = _load_model(model_dir)
    history = _load_history(model_dir)
    mean, std = _load_scaler(processed_dir)

    # Predict
    with torch.no_grad():
        y_pred_norm = model(x_test).numpy()
    y_test_norm_np = y_test_norm.numpy()

    # Denormalize to physical IDR
    y_true = _denormalize(y_test_norm_np, mean, std)
    y_pred = _denormalize(y_pred_norm, mean, std)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)

    n_stories = y_true.shape[1]
    story_labels = _story_labels(n_stories)

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION (N=%d)", y_true.shape[0])
    logger.info("=" * 60)
    for i in range(n_stories):
        logger.info(
            "  %s: RMSE = %.5f (%.3f%%)  R² = %.4f",
            story_labels[i],
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
    # Save metrics to JSON
    metrics_path = model_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Generate publication figures
    fm = FigureManager(output_dir=str(FIG_DIR), dpi=300)

    plot_loss_curves(history, fm)
    plot_pred_vs_actual(y_true, y_pred, metrics, fm, story_labels)
    plot_error_distribution(y_true, y_pred, fm, story_labels)

    # Figure 7: R² per story — N=3 measured vs N=10 measured/predicted.
    # Automatically switches from theoretical envelope to real data when the
    # N=10 checkpoint has been evaluated and test_metrics.json is present.
    n10_metrics = Path("data/models_n10") / "test_metrics.json"
    plot_r2_accuracy_profile(fm, n10_metrics_path=n10_metrics)

    logger.info("Publication figures saved to %s", FIG_DIR)


if __name__ == "__main__":
    main()
