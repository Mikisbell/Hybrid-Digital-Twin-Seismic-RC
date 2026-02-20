"""
scripts/fragility_curves.py
Generate seismic fragility curves P(DS ≥ ds | IM) from PgNN predictions.

Usage:
    python scripts/fragility_curves.py \
        --model-dir data/models_n10_v2 \
        --processed-dir data/processed/peer_10story_scalar

Damage States (FEMA P-58 / ASCE 41 for RC moment frames):
    IO  = 0.5%  (Immediate Occupancy)
    LS  = 1.5%  (Life Safety)
    CP  = 2.5%  (Collapse Prevention)

IM: Peak Ground Acceleration (PGA) extracted from each test record.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import norm

from src.pinn.model import HybridPINN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAMAGE_STATES = {
    "IO": 0.005,  # 0.5% IDR
    "LS": 0.015,  # 1.5% IDR
    "CP": 0.025,  # 2.5% IDR
}

DS_COLORS = {"IO": "#2ecc71", "LS": "#f39c12", "CP": "#e74c3c"}
DS_LABELS = {
    "IO": "Immediate Occupancy (0.5%)",
    "LS": "Life Safety (1.5%)",
    "CP": "Collapse Prevention (2.5%)",
}


# ---------------------------------------------------------------------------
# Fragility model
# ---------------------------------------------------------------------------


def fragility_lognormal(im: np.ndarray, theta: float, beta: float) -> np.ndarray:
    """Lognormal fragility function: P(DS ≥ ds | IM = im)."""
    return norm.cdf(np.log(im / theta) / beta)


def fit_fragility_mle(
    im: np.ndarray, exceedance: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """MLE fit of lognormal fragility parameters (Baker 2015).

    Parameters
    ----------
    im : array
        Intensity measure values (e.g., PGA in g).
    exceedance : array
        Binary: 1 if IDR ≥ threshold, 0 otherwise.

    Returns
    -------
    theta, beta : median capacity and dispersion, or (None, None) if fit fails.
    """
    valid = im > 0
    im_v = im[valid]
    y_v = exceedance[valid]

    if y_v.sum() < 3 or (1 - y_v).sum() < 3:
        return None, None

    median_exceed = np.median(im_v[y_v > 0]) if y_v.sum() > 0 else np.median(im_v)
    p0 = [median_exceed, 0.4]

    try:
        popt, _ = curve_fit(
            fragility_lognormal,
            im_v,
            y_v,
            p0=p0,
            bounds=([1e-4, 0.05], [np.inf, 3.0]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions(
    model_dir: Path, processed_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model, run inference on test set, return (pga, y_true, y_pred) in physical IDR.

    Returns arrays in physical space (denormalized).
    """
    # Load test data
    test = torch.load(processed_dir / "test.pt", weights_only=True)
    x_test, y_test_norm = test["x"], test["y"]

    # Load scaler
    with open(processed_dir / "scaler_params.json") as f:
        sp = json.load(f)
    mean = np.array(sp["target"]["mean"])
    std = np.array(sp["target"]["std"])

    # Load model and predict
    model = HybridPINN.from_checkpoint(str(model_dir / "pinn_best.pt"))
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_test).numpy()

    # Denormalize
    y_true = y_test_norm.numpy() * std + mean
    y_pred = y_pred_norm * std + mean

    # Extract PGA (max absolute acceleration from input)
    x_np = x_test.numpy().squeeze(1)  # (N, seq_len)

    # Check if input is normalised — if so, use raw ground_accel from test.pt
    if "ground_accel" in test:
        ga = test["ground_accel"].numpy()
        # ground_accel might be (N, n_stories, seq_len) — use first component
        if ga.ndim == 3:
            ga = ga[:, 0, :]
        pga = np.max(np.abs(ga), axis=1)
    else:
        # Fallback: use normalised input (relative PGA ordering preserved)
        pga = np.max(np.abs(x_np), axis=1)

    return pga, y_true, y_pred


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_fragility_all_ds(
    im_range: np.ndarray,
    params: dict,
    pga: np.ndarray,
    y_pred: np.ndarray,
    n_stories: int,
    output_dir: Path,
) -> None:
    """Figure 8: Fragility curves — all damage states, building-level (max IDR over stories)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Building-level: max IDR across all stories
    max_idr = np.max(y_pred, axis=1)

    for ds_name, threshold in DAMAGE_STATES.items():
        exceed = (max_idr >= threshold).astype(float)
        theta, beta = fit_fragility_mle(pga, exceed)
        if theta is None:
            continue

        prob = fragility_lognormal(im_range, theta, beta)
        ax.plot(
            im_range,
            prob,
            color=DS_COLORS[ds_name],
            linewidth=2.5,
            label=f"{DS_LABELS[ds_name]}\n($\\theta$={theta:.3f} g, $\\beta$={beta:.2f})",
        )

        # Empirical points
        n_bins = 8
        bin_edges = np.logspace(np.log10(pga.min()), np.log10(pga.max()), n_bins + 1)
        for i in range(n_bins):
            mask = (pga >= bin_edges[i]) & (pga < bin_edges[i + 1])
            if mask.sum() > 5:
                center = np.sqrt(bin_edges[i] * bin_edges[i + 1])
                emp_prob = exceed[mask].mean()
                ax.scatter(center, emp_prob, color=DS_COLORS[ds_name], s=30, alpha=0.6, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Peak Ground Acceleration, PGA (g)", fontsize=11)
    ax.set_ylabel("P(DS $\\geq$ ds | PGA)", fontsize=11)
    ax.set_title(f"Seismic Fragility Curves — {n_stories}-Story RC Frame (PgNN)", fontsize=12)
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = output_dir / "fragility_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_fragility_per_story(
    im_range: np.ndarray,
    pga: np.ndarray,
    y_pred: np.ndarray,
    n_stories: int,
    output_dir: Path,
) -> None:
    """Figure 9: Life Safety fragility curves per story — shows whiplash effect."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (n_stories - 1)) for i in range(n_stories)]

    threshold = DAMAGE_STATES["LS"]
    theta_list = []

    for story_idx in range(n_stories):
        exceed = (y_pred[:, story_idx] >= threshold).astype(float)
        theta, beta = fit_fragility_mle(pga, exceed)
        if theta is None:
            theta_list.append(np.nan)
            continue

        theta_list.append(theta)
        prob = fragility_lognormal(im_range, theta, beta)
        ax.plot(
            im_range,
            prob,
            color=colors[story_idx],
            linewidth=1.8,
            label=f"Story {story_idx + 1} ($\\theta$={theta:.3f} g)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Peak Ground Acceleration, PGA (g)", fontsize=11)
    ax.set_ylabel("P(IDR $\\geq$ 1.5% | PGA)", fontsize=11)
    ax.set_title(f"Life Safety Fragility per Story — {n_stories}-Story RC Frame", fontsize=12)
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    fig.tight_layout()
    out = output_dir / "fragility_per_story.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)

    return theta_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seismic fragility curves from PgNN")
    parser.add_argument("--model-dir", type=Path, default=Path("data/models_n10_v2"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/peer_10story_scalar")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript/figures"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading predictions from %s", args.model_dir)
    pga, y_true, y_pred = load_predictions(args.model_dir, args.processed_dir)
    n_stories = y_pred.shape[1]
    logger.info(
        "Test set: %d samples, %d stories, PGA range: %.3f–%.3f g",
        len(pga),
        n_stories,
        pga.min(),
        pga.max(),
    )

    # IM range for smooth curves
    im_range = np.logspace(np.log10(max(pga.min(), 1e-3)), np.log10(pga.max() * 1.5), 200)

    # Figure 8: Building-level fragility (all damage states)
    logger.info("Generating building-level fragility curves...")
    plot_fragility_all_ds(im_range, {}, pga, y_pred, n_stories, args.output_dir)

    # Figure 9: Per-story fragility (Life Safety)
    logger.info("Generating per-story fragility curves (LS)...")
    theta_list = plot_fragility_per_story(im_range, pga, y_pred, n_stories, args.output_dir)

    # Summary table
    logger.info("=" * 60)
    logger.info("FRAGILITY SUMMARY (Life Safety, IDR ≥ 1.5%%)")
    logger.info("=" * 60)
    for i, theta in enumerate(theta_list):
        if np.isnan(theta):
            logger.info("  Story %2d: insufficient data", i + 1)
        else:
            logger.info("  Story %2d: θ = %.4f g", i + 1, theta)
    logger.info("=" * 60)

    # Save params to JSON
    results = {}
    max_idr = np.max(y_pred, axis=1)
    for ds_name, threshold in DAMAGE_STATES.items():
        exceed = (max_idr >= threshold).astype(float)
        theta, beta = fit_fragility_mle(pga, exceed)
        results[ds_name] = {"theta": theta, "beta": beta, "threshold_idr": threshold}

    out_json = args.model_dir / "fragility_params.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Fragility parameters saved to %s", out_json)


if __name__ == "__main__":
    main()
