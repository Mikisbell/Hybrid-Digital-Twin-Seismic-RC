"""
data_factory.py — Seismic Ground Motion Data Factory
=====================================================

End-to-end pipeline for generating the PINN training dataset:

    1. Parse PEER NGA-West2 AT2 flat files from data/external/peer_nga/
    2. Apply spectrum-compatible scaling to ASCE 7-22 design spectrum
    3. Run NLTHA for each record via OpenSeesPy (batch mode)
    4. Produce structured CSVs in data/raw/ for NLTHAPipeline

Selection Criteria (per manuscript §3.2)
-----------------------------------------
    - Database:  PEER NGA-West2
    - Magnitude: 6.0 ≤ Mw ≤ 7.5
    - Distance:  10 km ≤ Rjb ≤ 50 km
    - Site class: C/D (180 ≤ Vs30 ≤ 760 m/s)
    - Records:   ≥200 (two horizontal components → 400+ time histories)

PEER NGA-West2 AT2 Format
--------------------------
    Header (4 lines):
        Line 1: Record description
        Line 2: Supplementary info
        Line 3: "NPTS= XXXX, DT= X.XXXXX SEC"
        Line 4: (may be blank or contain acceleration units)
    Data: Space-separated acceleration values (in g or cm/s²)

Usage
-----
    # Step 1: Download records from https://ngawest2.berkeley.edu/
    # Place .AT2 files in data/external/peer_nga/
    # Optionally place the flat-file CSV (NGA_West2_flatfile.csv)

    # Step 2: Run the factory
    python -m src.preprocessing.data_factory

    # Step 3: Run the ML pipeline
    python -m src.preprocessing.pipeline

Author: Mikisbell
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time as timer
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.opensees_analysis.nltha_runner import (
    GroundMotionRecord,
    NLTHAConfig,
)
from src.opensees_analysis.ospy_model import RCFrameModel

logger = logging.getLogger(__name__)

try:
    import openseespy.opensees as ops  # noqa: F401

    OPS_AVAILABLE = True
except ImportError:
    OPS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionCriteria:
    """Ground motion selection criteria per manuscript §3.2.

    Attributes
    ----------
    mw_min, mw_max : float
        Magnitude range.
    rjb_min, rjb_max : float
        Joyner-Boore distance range (km).
    vs30_min, vs30_max : float
        Shear-wave velocity range (m/s) for site class C/D.
    min_records : int
        Minimum number of records required.
    """

    mw_min: float = 6.0
    mw_max: float = 7.5
    rjb_min: float = 10.0
    rjb_max: float = 50.0
    vs30_min: float = 180.0
    vs30_max: float = 760.0
    min_records: int = 200


@dataclass
class DesignSpectrum:
    """ASCE 7-22 design response spectrum parameters.

    Attributes
    ----------
    sds : float
        Design spectral acceleration at short periods (g).
    sd1 : float
        Design spectral acceleration at 1 s (g).
    tl : float
        Long-period transition period (s).
    damping : float
        Damping ratio for target spectrum.
    """

    sds: float = 1.0
    sd1: float = 0.6
    tl: float = 8.0
    damping: float = 0.05


@dataclass
class FactoryConfig:
    """Configuration for the Data Factory.

    Attributes
    ----------
    peer_dir : str
        Directory containing PEER NGA-West2 .AT2 files.
    flatfile_path : str
        Path to NGA-West2 flatfile CSV (optional metadata source).
    output_dir : str
        NLTHA output directory (data/raw/).
    summary_path : str
        Path for the master summary CSV.
    selection : SelectionCriteria
        Ground motion selection criteria.
    spectrum : DesignSpectrum
        Target design spectrum for scaling.
    scale_period_range : tuple[float, float]
        Period range for spectrum matching (T_low, T_high) in seconds.
    max_scale_factor : float
        Maximum allowed scale factor (rejects records beyond this).
    nltha_config : NLTHAConfig
        Analysis configuration for OpenSeesPy.
    n_workers : int
        Number of parallel workers (1 = sequential).
    seed : int
        Random seed for reproducibility.
    """

    peer_dir: str = "data/external/peer_nga"
    flatfile_path: str = "data/external/NGA_West2_flatfile.csv"
    output_dir: str = "data/raw"
    summary_path: str = "data/raw/factory_summary.csv"
    selection: SelectionCriteria = field(default_factory=SelectionCriteria)
    spectrum: DesignSpectrum = field(default_factory=DesignSpectrum)
    scale_period_range: tuple[float, float] = (0.2, 2.0)
    max_scale_factor: float = 5.0
    nltha_config: NLTHAConfig = field(default_factory=NLTHAConfig)
    n_workers: int = 1
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════
# AT2 Parser
# ═══════════════════════════════════════════════════════════════════════════


def parse_at2(filepath: str | Path) -> tuple[np.ndarray, float, dict[str, str]]:
    """Parse a PEER NGA-West2 AT2 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .AT2 file.

    Returns
    -------
    acceleration : np.ndarray
        Acceleration values (in the file's native units).
    dt : float
        Time step in seconds.
    header_info : dict
        Parsed header metadata (description, npts, dt, units).

    Raises
    ------
    ValueError
        If the file cannot be parsed or NPTS/DT not found.
    """
    filepath = Path(filepath)
    header_info: dict[str, str] = {"filename": filepath.name}

    with open(filepath) as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"AT2 file too short: {filepath}")

    # Parse header
    header_info["description"] = lines[0].strip()
    header_info["supplementary"] = lines[1].strip()

    # Line 3: "NPTS= XXXX, DT= X.XXXXX SEC" (various formats)
    npts_dt_line = lines[2].strip()
    header_info["npts_dt_line"] = npts_dt_line

    # Extract NPTS and DT using regex (handles multiple PEER formats)
    npts_match = re.search(r"NPTS\s*=?\s*(\d+)", npts_dt_line, re.IGNORECASE)
    dt_match = re.search(r"DT\s*=?\s*([\d.Ee+-]+)", npts_dt_line, re.IGNORECASE)

    if not npts_match or not dt_match:
        # Try alternative format: "  XXXX   X.XXXXX   NPTS, DT"
        alt_match = re.match(r"\s*(\d+)\s+([\d.Ee+-]+)", npts_dt_line)
        if alt_match:
            npts = int(alt_match.group(1))
            dt = float(alt_match.group(2))
        else:
            raise ValueError(f"Cannot parse NPTS/DT from: '{npts_dt_line}' in {filepath}")
    else:
        npts = int(npts_match.group(1))
        dt = float(dt_match.group(1))

    header_info["npts"] = str(npts)
    header_info["dt"] = str(dt)

    # Detect units from line 4 or header
    units_line = lines[3].strip().lower() if len(lines) > 3 else ""
    if "cm/sec" in units_line or "cm/s" in units_line:
        header_info["units"] = "cm/s2"
    elif "g" in units_line or "gal" in units_line.lower():
        header_info["units"] = "g"
    else:
        # Default assumption: acceleration in g
        header_info["units"] = "g"

    # Parse data values (skip header lines, typically 4)
    data_start = 4
    # Some AT2 files have variable header length — find first numeric line
    for i in range(2, min(10, len(lines))):
        try:
            float(lines[i].split()[0])
            data_start = i
            break
        except (ValueError, IndexError):
            continue

    values: list[float] = []
    for line in lines[data_start:]:
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                continue

    acceleration = np.array(values[:npts])

    if len(acceleration) < npts:
        logger.warning(
            "AT2 %s: expected %d points, got %d",
            filepath.name,
            npts,
            len(acceleration),
        )

    return acceleration, dt, header_info


def parse_at2_directory(
    peer_dir: str | Path,
) -> list[tuple[np.ndarray, float, dict[str, str]]]:
    """Parse all AT2 files in a directory.

    Parameters
    ----------
    peer_dir : str or Path
        Directory containing .AT2 files.

    Returns
    -------
    list of (acceleration, dt, header_info) tuples.
    """
    peer_path = Path(peer_dir)
    if not peer_path.exists():
        logger.error("PEER directory not found: %s", peer_path)
        return []

    at2_files = sorted(peer_path.glob("*.AT2")) + sorted(peer_path.glob("*.at2"))
    logger.info("Found %d AT2 files in %s", len(at2_files), peer_path)

    records = []
    for fpath in at2_files:
        try:
            acc, dt, info = parse_at2(fpath)
            records.append((acc, dt, info))
        except (ValueError, OSError) as exc:
            logger.warning("Skipping %s: %s", fpath.name, exc)

    logger.info("Successfully parsed %d / %d records", len(records), len(at2_files))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# PEER Flatfile parser (optional metadata enrichment)
# ═══════════════════════════════════════════════════════════════════════════


def load_peer_flatfile(path: str | Path) -> dict[int, dict]:
    """Load the NGA-West2 flatfile CSV for metadata (Mw, Rjb, Vs30).

    Parameters
    ----------
    path : str or Path
        Path to ``NGA_West2_flatfile.csv``.

    Returns
    -------
    dict mapping RSN → {Mw, Rjb, Vs30, fault_type, ...}
    """
    import pandas as pd

    fpath = Path(path)
    if not fpath.exists():
        logger.warning("Flatfile not found: %s — metadata filtering disabled", fpath)
        return {}

    df = pd.read_csv(fpath, low_memory=False)
    logger.info("Loaded flatfile: %d records", len(df))

    catalog: dict[int, dict] = {}
    for _, row in df.iterrows():
        rsn = int(row.get("Record Sequence Number", row.get("RSN", 0)))
        if rsn == 0:
            continue
        catalog[rsn] = {
            "rsn": rsn,
            "event": str(row.get("Earthquake Name", "")),
            "mw": float(row.get("Magnitude", 0)),
            "rjb": float(row.get("Joyner-Boore Dist. (km)", row.get("Rjb (km)", 0))),
            "vs30": float(row.get("Vs30 (m/s)", row.get("Vs30", 0))),
            "fault_type": str(row.get("Mechanism", "")),
        }

    return catalog


def filter_by_criteria(
    catalog: dict[int, dict],
    criteria: SelectionCriteria,
) -> dict[int, dict]:
    """Filter flatfile records by selection criteria.

    Parameters
    ----------
    catalog : dict
        RSN → metadata from flatfile.
    criteria : SelectionCriteria
        Manuscript §3.2 selection bounds.

    Returns
    -------
    dict of RSN → metadata for records passing all criteria.
    """
    selected = {}
    for rsn, meta in catalog.items():
        if not (criteria.mw_min <= meta["mw"] <= criteria.mw_max):
            continue
        if not (criteria.rjb_min <= meta["rjb"] <= criteria.rjb_max):
            continue
        if not (criteria.vs30_min <= meta["vs30"] <= criteria.vs30_max):
            continue
        selected[rsn] = meta

    logger.info(
        "Flatfile filter: %d → %d records (Mw=[%.1f,%.1f], Rjb=[%.0f,%.0f]km, Vs30=[%.0f,%.0f]m/s)",
        len(catalog),
        len(selected),
        criteria.mw_min,
        criteria.mw_max,
        criteria.rjb_min,
        criteria.rjb_max,
        criteria.vs30_min,
        criteria.vs30_max,
    )
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Spectrum-compatible scaling (ASCE 7-22)
# ═══════════════════════════════════════════════════════════════════════════


def asce7_design_spectrum(
    periods: np.ndarray,
    sds: float,
    sd1: float,
    tl: float = 8.0,
) -> np.ndarray:
    """Compute ASCE 7-22 design response spectrum.

    Parameters
    ----------
    periods : np.ndarray
        Array of spectral periods (s).
    sds : float
        Short-period design spectral acceleration (g).
    sd1 : float
        1 s design spectral acceleration (g).
    tl : float
        Long-period transition period (s).

    Returns
    -------
    np.ndarray
        Design spectral acceleration Sa(T) in g.
    """
    t0 = 0.2 * sd1 / sds  # Short-period corner
    ts = sd1 / sds  # Transition period

    sa = np.zeros_like(periods)
    for i, t in enumerate(periods):
        if t <= 0:
            sa[i] = sds
        elif t < t0:
            sa[i] = sds * (0.4 + 0.6 * t / t0)
        elif t <= ts:
            sa[i] = sds
        elif t <= tl:
            sa[i] = sd1 / t
        else:
            sa[i] = sd1 * tl / (t * t)

    return sa


def compute_response_spectrum(
    acc: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float = 0.05,
) -> np.ndarray:
    """Compute pseudo-acceleration response spectrum via Newmark-β.

    Uses the piecewise-exact method for efficiency on long records.

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time series (g or m/s²).
    dt : float
        Time step (s).
    periods : np.ndarray
        Target spectral periods (s).
    damping : float
        Damping ratio.

    Returns
    -------
    np.ndarray
        Spectral acceleration Sa(T) in same units as input.
    """
    sa = np.zeros(len(periods))

    for i, t in enumerate(periods):
        if t == 0:
            sa[i] = np.max(np.abs(acc))
            continue

        omega = 2.0 * np.pi / t
        omega_d = omega * np.sqrt(1.0 - damping**2)
        xi_omega = damping * omega

        # Recurrence coefficients (piecewise-exact for constant accel segments)
        exp_term = np.exp(-xi_omega * dt)
        cos_term = np.cos(omega_d * dt)
        sin_term = np.sin(omega_d * dt)

        a11 = exp_term * (cos_term + xi_omega / omega_d * sin_term)
        a12 = exp_term * sin_term / omega_d
        a21 = -(omega**2) * exp_term * sin_term / omega_d
        a22 = exp_term * (cos_term - xi_omega / omega_d * sin_term)

        # Simplified: use direct numerical integration for robustness
        u = 0.0
        v = 0.0
        sd = 0.0

        for j in range(len(acc) - 1):
            u_new = a11 * u + a12 * v - dt * (a12 * acc[j])
            v_new = a21 * u + a22 * v - dt * (a22 * acc[j])
            # Simplified SDOF with piecewise-linear excitation
            # Use Newmark average acceleration for robustness
            u_new = a11 * u + a12 * v + ((1.0 - a11) * acc[j] + (dt - a12) * acc[j]) / omega**2
            v_new = a21 * u + a22 * v

            # Direct Duhamel integral (most robust for arbitrary signals)
            u = u_new
            v = v_new
            sd = max(sd, abs(u))

        # Pseudo-acceleration: Sa = ω² × Sd
        sa[i] = sd * omega**2

    return sa


def compute_scale_factor(
    record_sa: np.ndarray,
    target_sa: np.ndarray,
    periods: np.ndarray,
    period_range: tuple[float, float],
) -> float:
    """Compute spectrum-compatible scale factor.

    Minimises the MSE between scaled record spectrum and target in the
    specified period range.

    Parameters
    ----------
    record_sa : np.ndarray
        Record's response spectrum.
    target_sa : np.ndarray
        Target design spectrum.
    periods : np.ndarray
        Spectral periods.
    period_range : tuple[float, float]
        (T_low, T_high) for matching.

    Returns
    -------
    float
        Optimal scale factor (least-squares).
    """
    mask = (periods >= period_range[0]) & (periods <= period_range[1])
    if not np.any(mask):
        return 1.0

    rec = record_sa[mask]
    tgt = target_sa[mask]

    # Avoid division by zero
    denom = np.dot(rec, rec)
    if denom < 1e-12:
        return 1.0

    # Least-squares: SF = (rec · tgt) / (rec · rec)
    sf = float(np.dot(rec, tgt) / denom)
    return max(sf, 0.01)  # Enforce positive


# ═══════════════════════════════════════════════════════════════════════════
# Ground Motion Record builder
# ═══════════════════════════════════════════════════════════════════════════


def build_ground_motion_records(
    config: FactoryConfig,
) -> list[GroundMotionRecord]:
    """Build scaled GroundMotionRecord objects from AT2 files.

    Pipeline:
        1. Parse AT2 files
        2. (Optional) Filter by flatfile metadata
        3. Compute response spectra
        4. Scale to ASCE 7-22 design spectrum
        5. Filter by max scale factor

    Parameters
    ----------
    config : FactoryConfig
        Factory configuration.

    Returns
    -------
    list[GroundMotionRecord]
        Scaled ground motion records ready for NLTHA.
    """
    logger.info("Building ground motion records from %s", config.peer_dir)

    # Step 1: Parse AT2 files
    raw_records = parse_at2_directory(config.peer_dir)
    if not raw_records:
        logger.warning("No AT2 files found. Generating synthetic records instead.")
        return _generate_synthetic_suite(config)

    # Step 2: Load flatfile metadata (optional)
    catalog = load_peer_flatfile(config.flatfile_path)
    filtered_rsns: set[int] | None = None
    if catalog:
        selected = filter_by_criteria(catalog, config.selection)
        filtered_rsns = set(selected.keys())

    # Step 3: Compute target spectrum
    periods = np.geomspace(0.01, 10.0, 200)
    target_sa = asce7_design_spectrum(
        periods, config.spectrum.sds, config.spectrum.sd1, config.spectrum.tl
    )

    # Step 4: Process each record
    gm_records: list[GroundMotionRecord] = []

    for acc, dt, info in raw_records:
        name = info.get("description", info.get("filename", "unknown"))

        # Extract RSN from filename (e.g., "RSN953_NORTHR_MUL009.AT2")
        rsn_match = re.search(r"RSN(\d+)", info.get("filename", ""), re.IGNORECASE)
        rsn = int(rsn_match.group(1)) if rsn_match else None

        # Filter by flatfile criteria if available
        if filtered_rsns is not None and rsn is not None and rsn not in filtered_rsns:
            continue

        # Determine units
        units = info.get("units", "g")

        # Convert to g for spectrum computation
        acc_g = acc.copy()
        if units == "cm/s2":
            acc_g = acc / 981.0
        elif units == "m/s2":
            acc_g = acc / 9.81

        # Compute response spectrum
        record_sa = compute_response_spectrum(acc_g, dt, periods, config.spectrum.damping)

        # Compute scale factor
        sf = compute_scale_factor(record_sa, target_sa, periods, config.scale_period_range)

        # Reject if scale factor too large
        if sf > config.max_scale_factor:
            logger.debug("Rejected %s: SF=%.2f > %.1f", name, sf, config.max_scale_factor)
            continue

        # Build record name
        safe_name = info.get("filename", "unknown").replace(".AT2", "").replace(".at2", "")

        gm = GroundMotionRecord(
            name=safe_name,
            acceleration=acc_g,  # Store in g
            dt=dt,
            units="g",
            scale_factor=sf,
            source="PEER NGA-West2",
        )
        gm_records.append(gm)

    logger.info(
        "Ground motion suite: %d records (from %d parsed, SF ≤ %.1f)",
        len(gm_records),
        len(raw_records),
        config.max_scale_factor,
    )

    return gm_records


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic suite (fallback when no PEER data available)
# ═══════════════════════════════════════════════════════════════════════════


def _generate_synthetic_suite(
    config: FactoryConfig,
    n_records: int = 20,
) -> list[GroundMotionRecord]:
    """Generate synthetic ground motions for testing when PEER data is unavailable.

    Creates a variety of records with different frequency content and
    intensity to validate the pipeline before real data is available.

    Parameters
    ----------
    config : FactoryConfig
        Factory configuration (used for seed).
    n_records : int
        Number of synthetic records to generate.

    Returns
    -------
    list[GroundMotionRecord]
        Synthetic records for pipeline testing.
    """
    from src.opensees_analysis.nltha_runner import generate_synthetic_record

    rng = np.random.default_rng(config.seed)
    records: list[GroundMotionRecord] = []

    logger.info("Generating %d synthetic ground motions (PEER data not available)", n_records)

    for i in range(n_records):
        pga = rng.uniform(0.1, 0.6)  # PGA in g
        freq = rng.uniform(0.5, 5.0)  # Dominant frequency (Hz)
        duration = rng.uniform(10.0, 30.0)  # Duration (s)

        gm = generate_synthetic_record(
            duration=duration,
            dt=0.01,
            pga_g=pga,
            freq_hz=freq,
            name=f"Synthetic_{i + 1:03d}_PGA{pga:.2f}g_f{freq:.1f}Hz",
        )
        records.append(gm)

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Model builder function (for batch runner)
# ═══════════════════════════════════════════════════════════════════════════


def _build_fresh_model() -> None:
    """Build a clean RC frame model for each NLTHA run.

    Called by the batch runner before each analysis to ensure
    a pristine model state (no residual displacements/forces).
    """
    model = RCFrameModel()
    model.build()
    model.apply_gravity()
    model.setup_rayleigh_damping()


# ═══════════════════════════════════════════════════════════════════════════
# Master orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class DataFactory:
    """Orchestrates the full data generation campaign.

    Workflow:
        1. Build ground motion suite (AT2 → GroundMotionRecord)
        2. Run batch NLTHA (OpenSeesPy)
        3. Save master summary CSV

    Parameters
    ----------
    config : FactoryConfig
        Factory configuration.

    Examples
    --------
    >>> factory = DataFactory()
    >>> factory.run()       # Full pipeline
    >>> factory.run(dry_run=True)  # Parse + scale only (no NLTHA)
    """

    def __init__(self, config: FactoryConfig | None = None) -> None:
        self.config = config or FactoryConfig()
        self.gm_records: list[GroundMotionRecord] = []
        self.results: list[dict] = []

    def run(self, dry_run: bool = False) -> list[dict]:
        """Execute the full data generation campaign.

        Parameters
        ----------
        dry_run : bool
            If True, build the GM suite and report statistics but skip NLTHA.

        Returns
        -------
        list[dict]
            NLTHA results for each record (empty if dry_run).
        """
        logger.info("=" * 70)
        logger.info("  DATA FACTORY — Seismic Ground Motion Campaign")
        logger.info("=" * 70)
        t_start = timer.perf_counter()

        # Step 1: Build ground motion suite
        self.gm_records = build_ground_motion_records(self.config)

        if not self.gm_records:
            logger.error("No ground motion records available. Aborting.")
            return []

        # Report ground motion statistics
        self._report_gm_stats()

        if dry_run:
            logger.info("DRY RUN: Skipping NLTHA. %d records ready.", len(self.gm_records))
            return []

        # Step 2: Verify OpenSeesPy
        if not OPS_AVAILABLE:
            logger.error("OpenSeesPy required for NLTHA. Install: pip install openseespy")
            return []

        # Step 3: Run batch NLTHA
        logger.info("Starting batch NLTHA: %d records", len(self.gm_records))
        self.results = self._run_batch()

        # Step 4: Save summary
        self._save_summary()

        elapsed = timer.perf_counter() - t_start
        n_ok = sum(1 for r in self.results if r.get("converged", False))
        logger.info("=" * 70)
        logger.info(
            "  CAMPAIGN COMPLETE: %d/%d converged in %.0f s (%.1f min)",
            n_ok,
            len(self.results),
            elapsed,
            elapsed / 60,
        )
        logger.info("  Output: %s", self.config.output_dir)
        logger.info("  Summary: %s", self.config.summary_path)
        logger.info("=" * 70)

        return self.results

    def _run_batch(self) -> list[dict]:
        """Run NLTHA for all ground motions sequentially.

        Rebuilds the model before each run for clean state.
        """
        from src.opensees_analysis.nltha_runner import run_batch

        return run_batch(
            ground_motions=self.gm_records,
            model_builder=_build_fresh_model,
            config=self.config.nltha_config,
            n_stories=5,
            n_bays=3,
        )

    def _report_gm_stats(self) -> None:
        """Log summary statistics of the ground motion suite."""
        n = len(self.gm_records)
        pgas = [float(np.max(np.abs(gm.acceleration)) * gm.scale_factor) for gm in self.gm_records]
        durations = [gm.duration for gm in self.gm_records]
        sfs = [gm.scale_factor for gm in self.gm_records]

        logger.info("Ground Motion Suite Statistics:")
        logger.info("  Records:         %d", n)
        logger.info(
            "  PGA (g):         min=%.3f, max=%.3f, mean=%.3f",
            min(pgas),
            max(pgas),
            np.mean(pgas),
        )
        logger.info(
            "  Duration (s):    min=%.1f, max=%.1f, mean=%.1f",
            min(durations),
            max(durations),
            np.mean(durations),
        )
        logger.info(
            "  Scale factor:    min=%.2f, max=%.2f, mean=%.2f",
            min(sfs),
            max(sfs),
            np.mean(sfs),
        )

    def _save_summary(self) -> None:
        """Save master summary CSV with record metadata and peak responses."""
        summary_path = Path(self.config.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "record_name",
            "source",
            "scale_factor",
            "duration_s",
            "pga_g",
            "converged",
            "wall_clock_s",
            "n_steps",
            "max_idr_1",
            "max_idr_2",
            "max_idr_3",
            "max_idr_4",
            "max_idr_5",
            "max_idr_overall",
            "peak_base_shear_kN",
            "output_file",
        ]

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for gm, result in zip(self.gm_records, self.results, strict=False):
                drifts = result.get("max_drift", [0] * 5)
                row = [
                    gm.name,
                    gm.source,
                    f"{gm.scale_factor:.4f}",
                    f"{gm.duration:.1f}",
                    f"{float(np.max(np.abs(gm.acceleration)) * gm.scale_factor):.4f}",
                    result.get("converged", False),
                    result.get("duration_s", 0),
                    result.get("n_steps", 0),
                ]
                # Per-story max IDR
                for i in range(5):
                    row.append(f"{drifts[i]:.6f}" if i < len(drifts) else "0.0")
                row.append(f"{max(drifts):.6f}" if drifts else "0.0")
                row.append(f"{result.get('peak_base_shear', 0):.1f}")
                row.append(result.get("output_file", ""))
                writer.writerow(row)

        logger.info("Summary saved: %s (%d records)", summary_path, len(self.results))

        # Also save as JSON for programmatic access
        json_path = summary_path.with_suffix(".json")
        summary_data = {
            "n_records": len(self.results),
            "n_converged": sum(1 for r in self.results if r.get("converged", False)),
            "config": {
                "selection": asdict(self.config.selection),
                "spectrum": asdict(self.config.spectrum),
                "scale_period_range": list(self.config.scale_period_range),
                "max_scale_factor": self.config.max_scale_factor,
            },
        }
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for the Data Factory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Seismic Ground Motion Data Factory for PINN Training"
    )
    parser.add_argument(
        "--peer-dir",
        type=str,
        default="data/external/peer_nga",
        help="Directory with PEER NGA-West2 .AT2 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for NLTHA results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and scale GM records without running NLTHA",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="Generate N synthetic records (0 = use PEER data)",
    )
    parser.add_argument(
        "--max-sf",
        type=float,
        default=5.0,
        help="Maximum allowed scale factor (default: 5.0)",
    )
    parser.add_argument(
        "--sds",
        type=float,
        default=1.0,
        help="ASCE 7-22 SDS (default: 1.0g)",
    )
    parser.add_argument(
        "--sd1",
        type=float,
        default=0.6,
        help="ASCE 7-22 SD1 (default: 0.6g)",
    )
    args = parser.parse_args()

    config = FactoryConfig(
        peer_dir=args.peer_dir,
        output_dir=args.output_dir,
        spectrum=DesignSpectrum(sds=args.sds, sd1=args.sd1),
        max_scale_factor=args.max_sf,
    )

    factory = DataFactory(config)

    if args.synthetic > 0:
        logger.info("Using %d synthetic records (overriding PEER data)", args.synthetic)
        factory.gm_records = _generate_synthetic_suite(config, n_records=args.synthetic)
        if not args.dry_run:
            factory.results = factory._run_batch()
            factory._save_summary()
        else:
            factory._report_gm_stats()
    else:
        factory.run(dry_run=args.dry_run)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
