"""
pipeline.py — NLTHA Data Processing Pipeline for PINN Training
===============================================================

Transforms raw OpenSeesPy NLTHA time-series outputs into PyTorch-ready
tensors for the HybridPINN model.

Each raw CSV contains a full time-history simulation with columns::

    time, ground_accel, disp_1..5, vel_1..5, accel_1..5, drift_1..5, base_shear

The pipeline produces:
    - **Input tensors**  : ``(N, 1, seq_len)`` — ground acceleration
    - **Target tensors** : ``(N, 5)`` — peak absolute IDR per story

Pipeline stages:
    1. Ingest     : Load raw CSV + metadata JSON per simulation
    2. Validate   : Physical bounds, convergence check, NaN detection
    3. Augment    : Window slicing, amplitude scaling, noise injection
    4. Tensorise  : Pad/truncate to seq_len, build (x, y) tensors
    5. Split      : Record-level train/val/test (70/15/15)
    6. Export     : Save .pt tensors + scaler params + metadata JSON

Usage::

    python -m src.preprocessing.pipeline              # full run
    python -m src.preprocessing.pipeline --no-augment  # no augmentation
    python -m src.preprocessing.pipeline --dry-run     # preview only

Author: Mikisbell
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """Configuration for the NLTHA → PINN pipeline."""

    raw_dir: str = "data/raw"
    out_dir: str = "data/processed"

    # PINN model expects fixed-length sequences
    seq_len: int = 2048
    n_stories: int = 5

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Physical validation bounds
    max_idr: float = 0.10  # 10% IDR = structural collapse
    max_pga: float = 50.0  # ~5g in m/s² (raw CSV stores m/s², not g)

    # Data augmentation
    augment: bool = True
    n_windows: int = 3  # overlapping windows per record
    amplitude_scales: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5)
    noise_sigma: float = 0.005  # Gaussian noise std (fraction of PGA)

    # Normalisation
    normalise_input: bool = True  # zero-mean, unit-variance per sample
    normalise_targets: bool = True  # global target scaler

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            msg = f"Split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Data container for a single simulation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SimulationRecord:
    """One NLTHA simulation loaded from raw data."""

    name: str
    ground_accel: np.ndarray  # (T,) in g units
    drift: np.ndarray  # (T, n_stories)
    dt: float
    pga: float  # peak |ground_accel|
    peak_idr: np.ndarray  # (n_stories,) — max |drift| per story
    peak_idr_overall: float
    converged: bool
    source_file: str


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering helpers
# ═══════════════════════════════════════════════════════════════════════════


def compute_arias_intensity(acc: np.ndarray, dt: float) -> float:
    r"""Arias intensity: :math:`I_a = (\pi/2g) \int a^2(t)\,dt`."""
    g = 9.81
    return float((np.pi / (2 * g)) * np.trapezoid(acc**2, dx=dt))


def compute_pgv(acc: np.ndarray, dt: float) -> float:
    """Peak Ground Velocity via numerical integration (m/s)."""
    vel = np.cumsum(acc) * dt
    return float(np.max(np.abs(vel)))


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline class
# ═══════════════════════════════════════════════════════════════════════════


class NLTHAPipeline:
    """End-to-end pipeline: raw NLTHA CSV → PINN-ready .pt tensors.

    Parameters
    ----------
    config : PipelineConfig or None
        Pipeline settings.  Defaults to production config.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.metadata: dict[str, Any] = {
            "created": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
        }

    # ── Stage 1: Ingest ────────────────────────────────────────────────

    def ingest(self) -> list[SimulationRecord]:
        """Load all raw NLTHA simulations from data/raw/."""
        raw = Path(self.config.raw_dir)
        records: list[SimulationRecord] = []

        csv_files = sorted(
            f for f in raw.glob("Synthetic_*.csv") if not f.name.startswith("factory_summary")
        )

        for csv_path in csv_files:
            meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
            try:
                rec = self._load_one(csv_path, meta_path)
                records.append(rec)
            except Exception as e:
                logger.warning("Skipping %s: %s", csv_path.name, e)

        logger.info("Ingested %d simulation records", len(records))
        self.metadata["n_records_raw"] = len(records)
        return records

    def _load_one(self, csv_path: Path, meta_path: Path) -> SimulationRecord:
        """Load a single simulation CSV + metadata JSON."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        ground_accel = df["ground_accel"].values.astype(np.float32)
        drift_cols = [f"drift_{i}" for i in range(1, self.config.n_stories + 1)]
        drift = df[drift_cols].values.astype(np.float32)  # (T, 5)

        dt = 0.01
        converged = True
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            dt = meta.get("ground_motion", {}).get("dt", 0.01)
            converged = meta.get("results", {}).get("converged", True)

        pga = float(np.max(np.abs(ground_accel)))
        peak_idr = np.max(np.abs(drift), axis=0)
        peak_idr_overall = float(np.max(peak_idr))

        return SimulationRecord(
            name=csv_path.stem,
            ground_accel=ground_accel,
            drift=drift,
            dt=dt,
            pga=pga,
            peak_idr=peak_idr,
            peak_idr_overall=peak_idr_overall,
            converged=converged,
            source_file=csv_path.name,
        )

    # ── Stage 2: Validate ──────────────────────────────────────────────

    def validate(self, records: list[SimulationRecord]) -> list[SimulationRecord]:
        """Filter out invalid or non-converged records."""
        valid: list[SimulationRecord] = []
        for rec in records:
            if not rec.converged:
                logger.info("  Skipping %s: did not converge", rec.name)
                continue
            if rec.peak_idr_overall > self.config.max_idr:
                logger.info(
                    "  Skipping %s: IDR %.4f > %.2f (collapse)",
                    rec.name,
                    rec.peak_idr_overall,
                    self.config.max_idr,
                )
                continue
            if rec.pga > self.config.max_pga:
                logger.info(
                    "  Skipping %s: PGA %.2f > %.1f",
                    rec.name,
                    rec.pga,
                    self.config.max_pga,
                )
                continue
            if np.any(np.isnan(rec.ground_accel)) or np.any(np.isnan(rec.drift)):
                logger.info("  Skipping %s: contains NaN", rec.name)
                continue
            valid.append(rec)

        n_removed = len(records) - len(valid)
        if n_removed > 0:
            logger.info("Validation: removed %d/%d records", n_removed, len(records))
        else:
            logger.info("Validation: all %d records passed", len(valid))

        self.metadata["n_records_valid"] = len(valid)
        return valid

    # ── Stage 3: Augment ───────────────────────────────────────────────

    def augment(self, records: list[SimulationRecord]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate augmented (input, target) pairs from valid records.

        Augmentation strategies:
            1. **Window slicing** — overlapping windows of ``seq_len``
               capturing different earthquake phases.
            2. **Amplitude scaling** — linearly scale accelerogram and
               target IDR (valid for small-to-moderate nonlinearity).
            3. **Noise injection** — additive Gaussian noise on input.

        Returns
        -------
        inputs : list[np.ndarray]
            Each array has shape ``(seq_len,)``.
        targets : list[np.ndarray]
            Each array has shape ``(n_stories,)``.
        """
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        seq_len = self.config.seq_len

        for rec in records:
            accel = rec.ground_accel
            n = len(accel)

            windows = self._extract_windows(accel, n, seq_len)

            for win_accel in windows:
                idr_target = rec.peak_idr.copy()

                if self.config.augment:
                    for scale in self.config.amplitude_scales:
                        scaled_accel = win_accel * scale
                        scaled_idr = idr_target * scale

                        inputs.append(scaled_accel)
                        targets.append(scaled_idr)

                        # With noise (only at unit scale)
                        if self.config.noise_sigma > 0 and scale == 1.0:
                            noise = self._rng.normal(
                                0,
                                self.config.noise_sigma * rec.pga,
                                size=seq_len,
                            ).astype(np.float32)
                            inputs.append(win_accel + noise)
                            targets.append(idr_target)
                else:
                    inputs.append(win_accel)
                    targets.append(idr_target)

        logger.info(
            "Augmentation: %d records → %d samples (%.1fx)",
            len(records),
            len(inputs),
            len(inputs) / max(len(records), 1),
        )
        self.metadata["n_samples_augmented"] = len(inputs)
        return inputs, targets

    def _extract_windows(self, accel: np.ndarray, n: int, seq_len: int) -> list[np.ndarray]:
        """Extract overlapping windows from an acceleration record."""
        windows: list[np.ndarray] = []

        if n <= seq_len:
            padded = np.zeros(seq_len, dtype=np.float32)
            padded[:n] = accel
            windows.append(padded)
        else:
            n_win = self.config.n_windows if self.config.augment else 1
            if n_win == 1:
                windows.append(accel[:seq_len].astype(np.float32))
            else:
                stride = max(1, (n - seq_len) // (n_win - 1))
                starts = sorted({min(i * stride, n - seq_len) for i in range(n_win)})
                for s in starts:
                    windows.append(accel[s : s + seq_len].astype(np.float32))

        return windows

    # ── Stage 4: Tensorise ─────────────────────────────────────────────

    def tensorise(
        self, inputs: list[np.ndarray], targets: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Convert to PyTorch tensors and normalise.

        Returns
        -------
        x : torch.Tensor
            Shape ``(N, 1, seq_len)`` — normalised ground acceleration.
        y : torch.Tensor
            Shape ``(N, n_stories)`` — normalised peak IDR per story.
        scaler_params : dict
            Parameters needed to invert normalisation at inference.
        """
        x = np.stack(inputs)[:, np.newaxis, :]  # (N, 1, seq_len)
        y = np.stack(targets)  # (N, n_stories)

        scaler_params: dict[str, Any] = {}

        if self.config.normalise_input:
            x_mean = x.mean(axis=2, keepdims=True)
            x_std = x.std(axis=2, keepdims=True)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            x = (x - x_mean) / x_std
            scaler_params["input"] = {"method": "per_sample_standard"}

        if self.config.normalise_targets:
            y_mean = y.mean(axis=0)
            y_std = y.std(axis=0)
            y_std = np.where(y_std < 1e-8, 1.0, y_std)
            y = (y - y_mean) / y_std
            scaler_params["target"] = {
                "method": "global_standard",
                "mean": y_mean.tolist(),
                "std": y_std.tolist(),
            }

        x_tensor = torch.from_numpy(x.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))

        logger.info(
            "Tensorised: x=%s (%.1f MB), y=%s",
            list(x_tensor.shape),
            x_tensor.element_size() * x_tensor.nelement() / 1e6,
            list(y_tensor.shape),
        )
        return x_tensor, y_tensor, scaler_params

    # ── Stage 5: Split ─────────────────────────────────────────────────

    def split(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Shuffle and split into train/val/test."""
        n = x.shape[0]
        idx = self._rng.permutation(n)

        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "train": (x[idx[:n_train]], y[idx[:n_train]]),
            "val": (
                x[idx[n_train : n_train + n_val]],
                y[idx[n_train : n_train + n_val]],
            ),
            "test": (x[idx[n_train + n_val :]], y[idx[n_train + n_val :]]),
        }

        for name, (xp, _yp) in splits.items():
            logger.info("  %s: %d samples", name, xp.shape[0])

        self.metadata["split_sizes"] = {k: v[0].shape[0] for k, v in splits.items()}
        return splits

    # ── Stage 6: Export ────────────────────────────────────────────────

    def export(
        self,
        splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
        scaler_params: dict[str, Any],
    ) -> None:
        """Save tensors, scaler params, and metadata to data/processed/."""
        out = Path(self.config.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, (x, y) in splits.items():
            torch.save({"x": x, "y": y}, out / f"{name}.pt")

        with open(out / "scaler_params.json", "w") as f:
            json.dump(scaler_params, f, indent=2)

        self.metadata["exported"] = datetime.now(timezone.utc).isoformat()
        with open(out / "pipeline_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info("Exported to %s/", out)

    # ── Full pipeline ──────────────────────────────────────────────────

    def run(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Execute: ingest → validate → augment → tensorise → split → export."""
        logger.info("=" * 60)
        logger.info("NLTHA DATA PIPELINE")
        logger.info("=" * 60)

        records = self.ingest()
        if not records:
            logger.error("No data in %s. Run data_factory first.", self.config.raw_dir)
            return {}

        records = self.validate(records)
        if not records:
            logger.error("No valid records after validation.")
            return {}

        inputs, targets = self.augment(records)
        x, y, scaler_params = self.tensorise(inputs, targets)
        splits = self.split(x, y)
        self.export(splits, scaler_params)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return splits


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLTHA data pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    cfg = PipelineConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        augment=not args.no_augment,
    )

    pipe = NLTHAPipeline(cfg)

    if args.dry_run:
        records = pipe.ingest()
        records = pipe.validate(records)
        inputs, targets = pipe.augment(records)
        logger.info("[DRY RUN] Would produce %d samples, seq_len=%d", len(inputs), cfg.seq_len)
    else:
        pipe.run()
