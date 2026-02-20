"""
infer.py — Quick Inference Demo for the Hybrid Digital Twin
============================================================

Loads a trained HybridPINN checkpoint and runs real-time inference on a
ground-motion record that was never seen during training.  Reports per-story
IDR predictions and measures warm-inference latency.

Two input modes
---------------
1. **Test-set sample** (default): loads one record from ``data/processed/test.pt``
2. **Raw AT2 file**: reads a PEER NGA-West2 AT2 file directly via ``--at2``

Usage::

    # Mode 1: sample from the held-out test set (index 0 by default)
    python -m src.pinn.infer

    # Mode 1: pick a specific test-set record
    python -m src.pinn.infer --index 7

    # Mode 2: unseen raw ground-motion record
    python -m src.pinn.infer --at2 data/external/peer_nga/RSN169_IMPVALL.H_H-DLT262.AT2

    # Save the IDR bar-chart figure
    python -m src.pinn.infer --save-fig

Author: Mikisbell
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.config import GlobalConfig
from src.pinn.model import HybridPINN

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# AT2 parser (subset of data_factory logic, no heavy deps)
# ─────────────────────────────────────────────────────────────────────────────

_G_TO_MS2 = 9.81  # m/s²


def _parse_at2(path: Path) -> tuple[np.ndarray, float]:
    """Read a PEER NGA-West2 AT2 file.

    Returns
    -------
    accel : np.ndarray
        Acceleration in m/s².
    dt : float
        Time step in seconds.
    """
    lines = path.read_text().splitlines()
    # Header is 4 lines; line 4 contains NPTS and DT
    header = lines[3].upper()
    npts, dt = None, None
    for token in header.replace(",", " ").split():
        if "NPTS=" in token:
            npts = int(token.split("=")[1])
        elif "DT=" in token:
            dt = float(token.split("=")[1])
    if dt is None or npts is None:
        raise ValueError(f"Could not parse NPTS/DT from header: {lines[3]}")

    values: list[float] = []
    for line in lines[4:]:
        values.extend(float(v) for v in line.split() if v)

    accel = np.array(values[:npts], dtype=np.float32) * _G_TO_MS2
    return accel, dt


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing (mirrors pipeline.py normalisation)
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_tensor(accel: np.ndarray, seq_len: int) -> torch.Tensor:
    """Zero-mean / unit-std normalise, pad or truncate to seq_len.

    Returns
    -------
    torch.Tensor of shape (1, 1, seq_len)  — batch-of-one, single-channel.
    """
    # Normalise
    mu, sigma = accel.mean(), accel.std()
    if sigma < 1e-8:
        sigma = 1.0
    accel = (accel - mu) / sigma

    # Pad or truncate
    if len(accel) >= seq_len:
        accel = accel[:seq_len]
    else:
        pad = np.zeros(seq_len - len(accel), dtype=np.float32)
        accel = np.concatenate([accel, pad])

    return torch.from_numpy(accel).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ─────────────────────────────────────────────────────────────────────────────


def _measure_latency(model: HybridPINN, x: torch.Tensor, n_runs: int = 200) -> float:
    """Return mean warm-inference latency in milliseconds."""
    with torch.no_grad():
        # Warm-up (fills caches, JIT compiles)
        for _ in range(10):
            model(x)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        t1 = time.perf_counter()

    return (t1 - t0) / n_runs * 1_000  # ms


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────


def _print_results(idr: np.ndarray, latency_ms: float, source: str) -> None:
    width = 52
    bar_max = 30

    print()
    print("=" * width)
    print("  Hybrid Digital Twin — Seismic IDR Prediction")
    print(f"  Input : {source}")
    print("=" * width)
    print(f"  {'Story':<10} {'Peak IDR':>10}  {'Bar':}")
    print(f"  {'-' * 8:<10} {'-' * 8:>10}  {'-' * bar_max}")

    max_idr = max(abs(idr)) if abs(idr).max() > 0 else 1e-6
    for i, val in enumerate(idr, 1):
        pct = val * 100
        bar_len = int(abs(val) / max_idr * bar_max)
        bar = "█" * bar_len
        flag = " ← COLLAPSE RISK" if abs(val) > 0.025 else ""
        print(f"  Story {i:<4}  {pct:>8.4f} %  {bar}{flag}")

    print("=" * width)
    print(f"  Inference latency (warm, CPU): {latency_ms:.2f} ms")
    real_time = "YES" if latency_ms <= 100 else "NO"
    print(f"  Real-time threshold ≤100 ms : {real_time}")
    print("=" * width)
    print()


def _save_figure(idr: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping figure.")
        return

    n = len(idr)
    stories = [f"S{i}" for i in range(1, n + 1)]
    colors = ["#d62728" if v > 0.025 else "#1f77b4" for v in idr]

    fig, ax = plt.subplots(figsize=(max(4, n * 0.9), 4))
    ax.barh(stories, [v * 100 for v in idr], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(2.5, color="#d62728", linestyle="--", linewidth=1.0, label="Collapse limit (2.5%)")
    ax.set_xlabel("Peak IDR (%)")
    ax.set_title("HybridPINN — Predicted Inter-Story Drift Ratios")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick inference demo for the Hybrid Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory containing pinn_best.pt (default: data/models)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed data directory (used in test-set mode)",
    )
    parser.add_argument(
        "--at2",
        type=Path,
        default=None,
        help="Path to a raw PEER AT2 file for unseen-record mode",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index into the test set (test-set mode only, default: 0)",
    )
    parser.add_argument(
        "--save-fig",
        action="store_true",
        help="Save IDR bar chart to manuscript/figures/infer_demo.png",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── Load GlobalConfig to get seq_len ─────────────────────────────────────
    try:
        g_cfg = GlobalConfig.from_processed_dir(args.processed_dir)
        seq_len = g_cfg.seq_len
        logger.info("GlobalConfig: n_stories=%d, seq_len=%d", g_cfg.n_stories, seq_len)
    except FileNotFoundError:
        seq_len = 2048
        logger.warning("No global_config.json — using seq_len=%d", seq_len)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = args.model_dir / "pinn_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt}. Train the model first:\n  python src/pinn/train.py"
        )
    model = HybridPINN.from_checkpoint(ckpt)
    model.eval()
    logger.info("Model loaded (%d parameters)", model.count_parameters())

    # ── Prepare input tensor ──────────────────────────────────────────────────
    if args.at2 is not None:
        # Mode 2: raw AT2 file (unseen record)
        if not args.at2.exists():
            raise FileNotFoundError(f"AT2 file not found: {args.at2}")
        accel, dt = _parse_at2(args.at2)
        logger.info(
            "AT2 record: %d samples, dt=%.4f s, PGA=%.3f m/s²",
            len(accel),
            dt,
            float(np.abs(accel).max()),
        )
        x = _prepare_tensor(accel, seq_len)
        source = args.at2.name
    else:
        # Mode 1: test-set sample
        test_path = args.processed_dir / "test.pt"
        if not test_path.exists():
            raise FileNotFoundError(
                f"No test.pt found in {args.processed_dir}. Run pipeline first:\n"
                "  python src/preprocessing/pipeline.py"
            )
        test_data = torch.load(test_path, weights_only=True)
        idx = args.index
        if idx >= len(test_data["x"]):
            raise IndexError(
                f"--index {idx} out of range (test set has {len(test_data['x'])} samples)"
            )
        x = test_data["x"][idx : idx + 1]  # (1, 1, seq_len)
        source = f"test.pt[{idx}]"
        logger.info("Using test-set sample %d", idx)

    # ── Inference + latency ───────────────────────────────────────────────────
    with torch.no_grad():
        pred = model(x)  # (1, n_stories)

    latency_ms = _measure_latency(model, x)

    # ── Denormalise if scaler available ───────────────────────────────────────
    idr = pred.squeeze(0).numpy()
    scaler_path = args.processed_dir / "scaler_params.json"
    if scaler_path.exists():
        with open(scaler_path) as f:
            sp = json.load(f)
        if "target" in sp:
            mean = np.array(sp["target"]["mean"])
            std = np.array(sp["target"]["std"])
            idr = idr * std + mean
            logger.info("IDR denormalized using scaler_params.json")
    else:
        logger.warning("No scaler_params.json — predictions are in normalised units")

    # ── Print results ─────────────────────────────────────────────────────────
    _print_results(idr, latency_ms, source)

    # ── Optional figure ───────────────────────────────────────────────────────
    if args.save_fig:
        fig_path = Path("manuscript/figures/infer_demo.png")
        _save_figure(idr, fig_path)


if __name__ == "__main__":
    main()
