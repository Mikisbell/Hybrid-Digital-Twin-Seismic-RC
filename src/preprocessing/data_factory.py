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
    n_stories: int = 5
    n_bays: int = 3
    n_workers: int = 1
    limit: int = 0  # 0 = no limit
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

    with open(filepath, errors="replace") as f:
        # Read first 50 lines to cover header and start of data
        lines = [f.readline() for _ in range(50)]
        # Read the rest if needed later, but usually not for header parsing

    if len(lines) < 4:
        raise ValueError(f"AT2 file too short: {filepath}")

    # Parse header - search for NPTS and DT
    npts = 0
    dt = 0.0
    found_header = False
    header_end_line = 0

    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        if "NPTS" in line.upper() and ("DT" in line.upper() or "SEC" in line.upper()):
            npts_match = re.search(r"NPTS\s*=?\s*(\d+)", line, re.IGNORECASE)
            dt_match = re.search(r"DT\s*=?\s*([\d.Ee+-]+)", line, re.IGNORECASE)

            if not npts_match or not dt_match:
                # Try alternative format: "  XXXX   X.XXXXX   NPTS, DT"
                alt_match = re.match(r"\s*(\d+)\s+([\d.Ee+-]+)", line.strip())
                if alt_match:
                    npts = int(alt_match.group(1))
                    dt = float(alt_match.group(2))
                    found_header = True
                    header_end_line = i
                    break
            else:
                npts = int(npts_match.group(1))
                dt = float(dt_match.group(1))
                found_header = True
                header_end_line = i
                break

    if not found_header:
        raise ValueError(f"Cannot parse NPTS/DT from header in {filepath}")

    header_info["npts"] = str(npts)
    header_info["dt"] = str(dt)

    # Detect units - typically on the line BEFORE or AFTER the NPTS line
    # PEER standard: Unit line is usually line 3 (index 2) or line 4 (index 3)
    # We'll just search for typical unit strings in the header
    units_found = "g"  # Default
    for line in lines[: header_end_line + 1]:
        lower_line = line.lower()
        if "cm/sec" in lower_line or "cm/s" in lower_line:
            units_found = "cm/s2"
            break  # High confidence
        elif "in/sec" in lower_line or "in/s" in lower_line:
            units_found = "in/s2"
        elif "g" in lower_line or "gal" in lower_line:
            # Be careful not to match 'g' in text like 'Strong Motion'
            if re.search(r"\bunits\s+of\s+g\b", lower_line) or re.search(r"\bg\b", lower_line):
                units_found = "g"

    header_info["units"] = units_found

    # Parse data values (skip header lines)
    # Data starts after the header line. Typically next line.
    data_start = header_end_line + 1

    # Verify data start by checking if line is numeric
    for i in range(data_start, min(data_start + 5, len(lines))):
        try:
            # split and check first token
            parts = lines[i].split()
            if parts:
                float(parts[0])
                data_start = i
                break
        except ValueError:
            continue

    # Now read all lines (re-open to read full file properly or use existing buffer?)
    # Easier to re-read everything and skip
    with open(filepath, errors="replace") as f:
        all_lines = f.readlines()

    values: list[float] = []
    for line in all_lines[data_start:]:
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                continue

    acceleration = np.array(values)
    # Truncate to NPTS if we have more, or warn if less
    if len(acceleration) > npts:
        acceleration = acceleration[:npts]
    elif len(acceleration) < npts:
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
    """Compute pseudo-acceleration response spectrum via piecewise-exact method.

    Uses the Nigam-Jennings (1969) recurrence relations for an SDOF system
    subjected to piecewise-linear excitation.  This is the standard approach
    adopted by PEER and ASCE for ground-motion characterization.

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
        Spectral pseudo-acceleration Sa(T) in same units as input.

    References
    ----------
    Nigam, N.C. and Jennings, P.C. (1969). "Calculation of response spectra
    from strong-motion earthquake records." Bull. Seismol. Soc. Am., 59(2).
    """
    # Vectorized implementation of Nigam-Jennings (1969)
    # Vectors over periods (shape: [num_periods])

    # Handle zero/negative periods: max(abs(acc))
    valid_mask = periods > 0
    sa = np.zeros_like(periods)
    sa[~valid_mask] = np.max(np.abs(acc))

    if not np.any(valid_mask):
        return sa

    t = periods[valid_mask]
    omega = 2.0 * np.pi / t
    omega2 = omega**2
    xi = damping
    omega_d = omega * np.sqrt(1.0 - xi**2)
    xi_omega = xi * omega

    # Precompute coefficients (vectorized over periods)
    exp_term = np.exp(-xi_omega * dt)
    cos_d = np.cos(omega_d * dt)
    sin_d = np.sin(omega_d * dt)

    a11 = exp_term * (cos_d + (xi_omega / omega_d) * sin_d)
    a12 = exp_term * sin_d / omega_d
    a21 = -omega2 * a12
    a22 = exp_term * (cos_d - (xi_omega / omega_d) * sin_d)

    one_over_omega2 = 1.0 / omega2
    t1 = (2.0 * xi**2 - 1.0) / (omega2 * dt)
    t2 = 2.0 * xi / (omega**3 * dt)

    b11 = exp_term * ((t1 + xi / omega) * sin_d / omega_d + (t2 + one_over_omega2) * cos_d) - t2
    b12 = -(exp_term * (t1 * sin_d / omega_d + t2 * cos_d)) - one_over_omega2 + t2
    b21 = exp_term * (
        (t1 + xi / omega) * (cos_d - xi_omega * sin_d / omega_d)
        - (t2 + one_over_omega2) * (omega_d * sin_d + xi_omega * cos_d)
    ) + 1.0 / (omega2 * dt)
    b22 = -exp_term * (
        t1 * (cos_d - xi_omega * sin_d / omega_d) - t2 * (omega_d * sin_d + xi_omega * cos_d)
    ) - 1.0 / (omega2 * dt)

    # Initialize state vectors [num_valid_periods]
    u = np.zeros_like(t)
    v = np.zeros_like(t)
    sd_max = np.zeros_like(t)

    # Time-stepping loop (vectorized over periods)
    # We still loop over time, but do all periods at once
    n = len(acc)
    p = -acc  # Excitation array

    for j in range(n - 1):
        p_j = p[j]
        p_j1 = p[j + 1]

        # Update state
        u_new = a11 * u + a12 * v + b11 * p_j + b12 * p_j1
        v_new = a21 * u + a22 * v + b21 * p_j + b22 * p_j1

        u = u_new
        v = v_new

        # Track max displacement
        abs_u = np.abs(u)
        mask_update = abs_u > sd_max
        sd_max[mask_update] = abs_u[mask_update]

    # Pseudo-acceleration: Sa = ω² * Sd
    sa[valid_mask] = sd_max * omega2

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
# ASCE 7-22 §16.2 Suite-Level Spectral Matching
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SpectralMatchResult:
    """Result of the suite-level spectral matching check.

    Attributes
    ----------
    passed : bool
        True if suite mean ≥ threshold × target across the matching range.
    ratio_min : float
        Minimum ratio of suite-mean Sa to target Sa in the matching range.
    threshold : float
        Acceptance threshold (default: 0.9 per ASCE 7-22 §16.2.2).
    n_records : int
        Number of records in the suite.
    periods : np.ndarray
        Spectral periods used.
    suite_mean_sa : np.ndarray
        Mean response spectrum of the scaled suite.
    target_sa : np.ndarray
        Target design spectrum.
    individual_sa : list[np.ndarray]
        Scaled Sa for each record.
    scale_factors : list[float]
        Applied scale factors.
    """

    passed: bool
    ratio_min: float
    threshold: float
    n_records: int
    periods: np.ndarray = field(repr=False)
    suite_mean_sa: np.ndarray = field(repr=False)
    target_sa: np.ndarray = field(repr=False)
    individual_sa: list[np.ndarray] = field(default_factory=list, repr=False)
    scale_factors: list[float] = field(default_factory=list, repr=False)


def validate_suite_spectrum(
    records_sa: list[np.ndarray],
    scale_factors: list[float],
    target_sa: np.ndarray,
    periods: np.ndarray,
    period_range: tuple[float, float],
    threshold: float = 0.9,
) -> SpectralMatchResult:
    """Validate that a scaled GM suite satisfies ASCE 7-22 §16.2.2.

    The standard requires: for each period in [0.2·T₁, 1.5·T₁], the **mean**
    response spectrum of the scaled suite shall not fall below *threshold*
    (90 %) of the target spectrum.

    Parameters
    ----------
    records_sa : list[np.ndarray]
        Unscaled response spectra, one per record.
    scale_factors : list[float]
        Per-record scale factors from ``compute_scale_factor``.
    target_sa : np.ndarray
        Target ASCE 7-22 design spectrum.
    periods : np.ndarray
        Spectral periods common to all spectra.
    period_range : tuple[float, float]
        Period range for matching (T_low, T_high).
    threshold : float
        Minimum acceptable ratio (0.9 per code).

    Returns
    -------
    SpectralMatchResult
        Detailed pass/fail result with diagnostic arrays.
    """
    mask = (periods >= period_range[0]) & (periods <= period_range[1])
    if not np.any(mask):
        logger.warning("No periods in matching range [%.2f, %.2f] s", *period_range)
        return SpectralMatchResult(
            passed=False,
            ratio_min=0.0,
            threshold=threshold,
            n_records=len(records_sa),
            periods=periods,
            suite_mean_sa=np.zeros_like(target_sa),
            target_sa=target_sa,
        )

    # Compute scaled spectra
    scaled = [sa * sf for sa, sf in zip(records_sa, scale_factors, strict=False)]
    suite_mean = np.mean(scaled, axis=0)

    # Ratio check in matching range
    tgt_mask = target_sa[mask]
    mean_mask = suite_mean[mask]

    # Avoid division by zero
    ratios = np.where(tgt_mask > 1e-12, mean_mask / tgt_mask, 999.0)
    ratio_min = float(np.min(ratios))
    passed = bool(ratio_min >= threshold)

    if passed:
        logger.info(
            "Suite spectral check PASSED: min ratio = %.3f ≥ %.2f (%d records)",
            ratio_min,
            threshold,
            len(records_sa),
        )
    else:
        logger.warning(
            "Suite spectral check FAILED: min ratio = %.3f < %.2f (%d records)",
            ratio_min,
            threshold,
            len(records_sa),
        )

    return SpectralMatchResult(
        passed=passed,
        ratio_min=ratio_min,
        threshold=threshold,
        n_records=len(records_sa),
        periods=periods,
        suite_mean_sa=suite_mean,
        target_sa=target_sa,
        individual_sa=scaled,
        scale_factors=scale_factors,
    )


def iterative_suite_scaling(
    records_sa: list[np.ndarray],
    target_sa: np.ndarray,
    periods: np.ndarray,
    period_range: tuple[float, float],
    max_sf: float = 5.0,
    threshold: float = 0.9,
    max_iterations: int = 10,
) -> tuple[list[float], SpectralMatchResult]:
    """Iteratively adjust individual scale factors until the suite passes.

    Algorithm:
        1. Compute initial per-record SF via least-squares.
        2. Validate suite mean vs target.
        3. If fail: multiply all SFs by (threshold / ratio_min) and re-check.
        4. Repeat until pass or max_iterations.

    Parameters
    ----------
    records_sa : list[np.ndarray]
        Unscaled response spectra.
    target_sa : np.ndarray
        Target design spectrum.
    periods : np.ndarray
        Spectral periods.
    period_range : tuple[float, float]
        Period range for matching.
    max_sf : float
        Maximum allowed per-record scale factor.
    threshold : float
        ASCE 7-22 acceptance threshold (0.9).
    max_iterations : int
        Maximum correction iterations.

    Returns
    -------
    scale_factors : list[float]
        Final per-record scale factors.
    result : SpectralMatchResult
        Final suite validation result.
    """
    # Step 1: initial per-record SF
    scale_factors = [
        compute_scale_factor(sa, target_sa, periods, period_range) for sa in records_sa
    ]

    for iteration in range(max_iterations):
        result = validate_suite_spectrum(
            records_sa, scale_factors, target_sa, periods, period_range, threshold
        )

        if result.passed:
            logger.info("Suite matching converged after %d iteration(s).", iteration + 1)
            return scale_factors, result

        # Correction: boost all SFs so the weakest period meets the threshold
        boost = threshold / result.ratio_min if result.ratio_min > 0 else 2.0

        scale_factors = [min(sf * boost, max_sf) for sf in scale_factors]
        logger.info(
            "  Iteration %d: boosted SFs by %.3f (min ratio was %.3f)",
            iteration + 1,
            boost,
            result.ratio_min,
        )

    # Final check after last iteration
    result = validate_suite_spectrum(
        records_sa, scale_factors, target_sa, periods, period_range, threshold
    )
    if not result.passed:
        logger.warning(
            "Suite matching did NOT converge after %d iterations (ratio=%.3f).",
            max_iterations,
            result.ratio_min,
        )

    return scale_factors, result


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

    # Step 4: Process each record — compute spectra and initial SFs
    candidates: list[tuple[np.ndarray, float, dict, np.ndarray]] = []  # (acc_g, dt, info, Sa)

    for acc, dt, info in raw_records:
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
        candidates.append((acc_g, dt, info, record_sa))

    if not candidates:
        logger.warning("No valid records after parsing + flatfile filter.")
        return _generate_synthetic_suite(config)

    # Step 5: ASCE 7-22 §16.2 iterative suite-level spectral matching
    all_sa = [c[3] for c in candidates]
    scale_factors, match_result = iterative_suite_scaling(
        records_sa=all_sa,
        target_sa=target_sa,
        periods=periods,
        period_range=config.scale_period_range,
        max_sf=config.max_scale_factor,
        threshold=0.9,
    )

    # Step 6: Build GroundMotionRecords, rejecting over-scaled entries
    gm_records: list[GroundMotionRecord] = []

    for (acc_g, dt, info, _sa), sf in zip(candidates, scale_factors, strict=False):
        if sf > config.max_scale_factor:
            logger.debug(
                "Rejected %s: SF=%.2f > %.1f",
                info.get("filename", "?"),
                sf,
                config.max_scale_factor,
            )
            continue

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
    logger.info(
        "Suite spectral match: %s (min ratio=%.3f, threshold=%.2f)",
        "PASSED" if match_result.passed else "FAILED",
        match_result.ratio_min,
        match_result.threshold,
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


def _build_fresh_model(n_stories: int = 5, n_bays: int = 3) -> None:
    """Build a clean RC frame model for each NLTHA run.

    Called by the batch runner before each analysis to ensure
    a pristine model state (no residual displacements/forces).
    """
    from src.opensees_analysis.ospy_model import FrameGeometry, ModelConfig

    frame_cfg = FrameGeometry(n_stories=n_stories, n_bays=n_bays)
    model_cfg = ModelConfig(frame=frame_cfg)
    model = RCFrameModel(config=model_cfg)
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

        # Step 5: Sync to Notion (non-blocking — failures are logged, not raised)
        self._sync_to_notion()

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
        records = self.gm_records
        if self.config.limit > 0:
            logger.info("Limiting execution to first %d records", self.config.limit)
            records = records[: self.config.limit]

        if not records:
            logger.warning("No records to process.")
            return []

        logger.info("Preparing NLTHA batch for %d records...", len(records))
        from functools import partial

        from src.opensees_analysis.nltha_runner import run_batch

        # Ensure NLTHA config inherits the factory's output directory
        self.config.nltha_config.output_dir = self.config.output_dir

        # Bind n_stories/n_bays so the builder creates the correct frame
        builder = partial(
            _build_fresh_model,
            n_stories=self.config.n_stories,
            n_bays=self.config.n_bays,
        )

        return run_batch(
            ground_motions=records,
            model_builder=builder,
            config=self.config.nltha_config,
            n_stories=self.config.n_stories,
            n_bays=self.config.n_bays,
            n_workers=self.config.n_workers,
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
        ]
        # Dynamic drift headers for N stories
        for i in range(1, self.config.n_stories + 1):
            headers.append(f"max_idr_{i}")
        headers.extend(["max_idr_overall", "peak_base_shear_kN", "output_file"])

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for gm, result in zip(self.gm_records, self.results, strict=False):
                drifts = result.get("max_drift", [0.0] * self.config.n_stories)
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
                # Ensure we have enough drift values, or pad with 0.0
                n_drifts = len(drifts)
                for i in range(self.config.n_stories):
                    val = drifts[i] if i < n_drifts else 0.0
                    row.append(f"{val:.6f}")

                # Overall max
                overall_max = max(drifts) if drifts else 0.0
                row.append(f"{overall_max:.6f}")

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

    def _sync_to_notion(self) -> None:
        """Sync NLTHA results to Notion Simulation Log database.

        Uses ``NotionResearchLogger`` to create one entry per record.
        Failures are logged as warnings — they never abort the pipeline.
        """
        try:
            from src.utils.sync_results import NotionResearchLogger
        except ImportError:
            logger.info("Notion sync skipped (notion-client not installed).")
            return

        try:
            notion = NotionResearchLogger()
        except (ValueError, ImportError) as exc:
            logger.warning("Notion sync skipped: %s", exc)
            return

        n_logged = 0
        for gm, result in zip(self.gm_records, self.results, strict=False):
            converged = result.get("converged", False)
            drifts = result.get("max_drift", [0.0] * 5)
            pga = float(np.max(np.abs(gm.acceleration)) * gm.scale_factor)

            try:
                notion.log_simulation(
                    ground_motion=gm.name,
                    max_drift=max(drifts) if drifts else 0.0,
                    peak_acceleration=pga,
                    convergence_status="Converged" if converged else "Diverged",
                    num_stories=5,
                    phase="Methods",
                    notes=(
                        f"SF={gm.scale_factor:.3f} | "
                        f"Duration={gm.duration:.1f}s | "
                        f"Source={gm.source} | "
                        f"IDR=[{', '.join(f'{d:.5f}' for d in drifts)}]"
                    ),
                    source_ref="Data Factory — commit SHA: see git log",
                )
                n_logged += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Notion log failed for %s: %s", gm.name, exc)

        logger.info("Notion sync: %d / %d records logged.", n_logged, len(self.results))


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
    parser.add_argument(
        "--n-stories",
        type=int,
        default=5,
        help="Number of building stories (default: 5)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records (0 = no limit)",
    )
    args = parser.parse_args()

    config = FactoryConfig(
        peer_dir=args.peer_dir,
        output_dir=args.output_dir,
        summary_path=str(Path(args.output_dir) / "factory_summary.csv"),
        spectrum=DesignSpectrum(sds=args.sds, sd1=args.sd1),
        max_scale_factor=args.max_sf,
        n_stories=args.n_stories,
        n_workers=args.n_workers,
        limit=args.limit,
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

    # Persist GlobalConfig so downstream scripts (pipeline, train) can validate
    # that they are operating on consistent structural parameters.
    if not args.dry_run:
        from src.config import GlobalConfig

        GlobalConfig(n_stories=args.n_stories, n_bays=args.n_bays).save(args.output_dir)
        logger.info("GlobalConfig saved to %s (n_stories=%d)", args.output_dir, args.n_stories)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
