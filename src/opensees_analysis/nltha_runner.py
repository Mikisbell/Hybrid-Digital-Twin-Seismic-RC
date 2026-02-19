"""
nltha_runner.py — Nonlinear Time History Analysis Runner
========================================================

Executes dynamic analysis on the RC frame model with robust convergence
handling, adaptive time-stepping, and structured output for the data pipeline.

Convergence Strategy (layered)
------------------------------
    1. Newton with initial dt
    2. ModifiedNewton if Newton fails
    3. Bisect dt by 2 (up to 5 levels) with Newton
    4. NewtonLineSearch as last resort
    5. KrylovNewton for severely ill-conditioned steps

This addresses the "Valley of Death" concern: high-intensity records
that cause numerical instabilities in nonlinear RC elements.

Output Format
-------------
    Each run produces a CSV file in data/raw/ with columns:
    time, ground_accel, disp_1..disp_5, vel_1..vel_5, accel_1..accel_5,
    drift_1..drift_5, base_shear

    Plus a JSON metadata file with analysis parameters and convergence info.

Author: Mikisbell
"""

from __future__ import annotations

import csv
import json
import logging
import time as timer
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import openseespy.opensees as ops

    OPS_AVAILABLE = True
except ImportError:
    OPS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NLTHAConfig:
    """Configuration for NLTHA execution."""

    # Time integration
    dt: float = 0.01  # Analysis time step (s)
    dt_min: float = 1.0e-6  # Minimum adaptive dt (s)
    max_bisections: int = 5  # Max dt bisections before failure

    # Convergence
    tol: float = 1.0e-8  # Norm tolerance
    max_iter: int = 100  # Max Newton iterations
    test_type: str = "NormDispIncr"  # "NormDispIncr" | "NormUnbalance" | "EnergyIncr"

    # Newmark integration
    gamma: float = 0.5  # Newmark gamma
    beta: float = 0.25  # Newmark beta (average acceleration)

    # System
    system_type: str = "BandGeneral"  # "BandGeneral" | "UmfPack" | "SparseSYM"
    numberer: str = "RCM"
    constraints: str = "Transformation"

    # Output
    output_dir: str = "data/raw"
    save_every_n: int = 1  # Save every N steps (1 = all)


@dataclass
class GroundMotionRecord:
    """Represents a single ground motion record."""

    name: str  # e.g. "RSN953_Northridge_MUL009"
    acceleration: np.ndarray  # Time series (m/s² or g)
    dt: float  # Record time step (s)
    units: str = "g"  # "g" | "m/s2" | "cm/s2"
    scale_factor: float = 1.0  # Spectrum-compatible scaling
    source: str = "PEER NGA-West2"  # Data source

    @property
    def duration(self) -> float:
        return len(self.acceleration) * self.dt

    @property
    def npts(self) -> int:
        return len(self.acceleration)

    def get_acceleration_mps2(self) -> np.ndarray:
        """Return acceleration in m/s² with scale factor applied."""
        acc = self.acceleration * self.scale_factor
        if self.units == "g":
            acc = acc * 9.81
        elif self.units == "cm/s2":
            acc = acc / 100.0
        return acc


# ═══════════════════════════════════════════════════════════════════════════
# Convergence engine
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveAnalyzer:
    """Multi-strategy convergence handler for NLTHA.

    Implements a cascading algorithm hierarchy:
    1. Newton (fastest)
    2. ModifiedNewton (more robust)
    3. Bisected dt + Newton
    4. NewtonLineSearch (slower but handles steep softening)
    5. KrylovNewton (last resort for ill-conditioned tangent)
    """

    def __init__(self, config: NLTHAConfig) -> None:
        self.config = config
        self.convergence_log: list[dict] = []
        self._total_substeps = 0
        self._total_failures = 0
        self._total_retries = 0

    def analyze_one_step(self, dt_target: float) -> bool:
        """Attempt to advance one time step with adaptive strategies.

        Returns True if the step succeeded (possibly with substeps).
        """
        # Strategy 1: Standard Newton at full dt
        if self._try_algorithm("Newton", dt_target):
            return True

        self._total_retries += 1

        # Strategy 2: ModifiedNewton at full dt
        if self._try_algorithm("ModifiedNewton", dt_target):
            return True

        # Strategy 3: Bisect time step
        for level in range(1, self.config.max_bisections + 1):
            dt_sub = dt_target / (2**level)
            if dt_sub < self.config.dt_min:
                break

            n_substeps = 2**level
            logger.debug(
                "Bisection level %d: %d substeps at dt=%.2e s",
                level,
                n_substeps,
                dt_sub,
            )

            success = True
            for _ in range(n_substeps):
                if not self._try_algorithm("Newton", dt_sub) and not self._try_algorithm(
                    "NewtonLineSearch", dt_sub
                ):
                    success = False
                    break
                self._total_substeps += 1

            if success:
                return True

        # Strategy 4: KrylovNewton at full dt (last resort)
        if self._try_algorithm("KrylovNewton", dt_target):
            return True

        # All strategies exhausted
        self._total_failures += 1
        return False

    def _try_algorithm(self, algo_name: str, dt: float) -> bool:
        """Try a single algorithm at a given dt."""
        ops.algorithm(algo_name)
        ops.integrator("Newmark", self.config.gamma, self.config.beta)
        ops.test(self.config.test_type, self.config.tol, self.config.max_iter)
        ops.analysis("Transient")

        ok = ops.analyze(1, dt)
        return ok == 0

    @property
    def stats(self) -> dict:
        return {
            "total_substeps": self._total_substeps,
            "total_retries": self._total_retries,
            "total_failures": self._total_failures,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main NLTHA Runner
# ═══════════════════════════════════════════════════════════════════════════


class NLTHARunner:
    """Execute and record NLTHA on an RC frame model.

    Usage
    -----
        from src.opensees_analysis.ospy_model import RCFrameModel
        from src.opensees_analysis.nltha_runner import NLTHARunner, GroundMotionRecord

        model = RCFrameModel()
        model.build()
        model.apply_gravity()
        model.setup_rayleigh_damping()

        gm = GroundMotionRecord(
            name="RSN953_Northridge",
            acceleration=np.loadtxt("gm.txt"),
            dt=0.01,
        )

        runner = NLTHARunner(n_stories=5, n_bays=3)
        result = runner.run(gm)
    """

    def __init__(
        self,
        n_stories: int = 5,
        n_bays: int = 3,
        config: NLTHAConfig | None = None,
    ) -> None:
        self.n_stories = n_stories
        self.n_bays = n_bays
        self.config = config or NLTHAConfig()
        self._analyzer = AdaptiveAnalyzer(self.config)

    def run(self, gm: GroundMotionRecord) -> dict:
        """Execute NLTHA for a single ground motion record.

        Returns
        -------
        dict with keys:
            converged: bool
            duration_s: float (wall-clock)
            n_steps: int
            max_drift: list[float] (per story)
            max_disp: list[float] (per story)
            peak_base_shear: float
            output_file: str (path to CSV)
            metadata_file: str (path to JSON)
            convergence_stats: dict
        """
        if not OPS_AVAILABLE:
            raise RuntimeError("OpenSeesPy required for NLTHA")

        logger.info("Starting NLTHA: %s (%.1f s, dt=%.4f s)", gm.name, gm.duration, gm.dt)
        t_start = timer.perf_counter()

        # ── Setup ground motion load pattern ────────────────────────────
        acc_mps2 = gm.get_acceleration_mps2()
        n_pts = len(acc_mps2)
        dt_gm = gm.dt

        # TimeSeries: Path (piece-wise linear from array)
        ts_tag = 100
        ops.timeSeries("Path", ts_tag, "-dt", dt_gm, "-values", *acc_mps2.tolist())

        # UniformExcitation pattern (horizontal DOF = 1)
        pat_tag = 100
        ops.pattern("UniformExcitation", pat_tag, 1, "-accel", ts_tag)

        # ── Setup recorders (in-memory) ─────────────────────────────────
        # We'll record manually for flexibility
        floor_nodes = []  # Top node of each story (leftmost column)
        for story in range(1, self.n_stories + 1):
            floor_nodes.append(story * 100)

        # Story heights for IDR
        story_heights = []
        for story in range(1, self.n_stories + 1):
            h = 3.5 if story == 1 else 3.0  # Match FrameGeometry defaults
            story_heights.append(h)

        # ── Analysis setup ──────────────────────────────────────────────
        # Wipe previous (static) analysis before setting up transient
        ops.wipeAnalysis()
        ops.constraints(self.config.constraints)
        ops.numberer(self.config.numberer)
        ops.system(self.config.system_type)

        # ── Time-stepping loop ──────────────────────────────────────────
        dt_analysis = self.config.dt
        total_time = gm.duration
        current_time = 0.0

        # Storage
        times: list[float] = []
        ground_accels: list[float] = []
        displacements: list[list[float]] = [[] for _ in range(self.n_stories)]
        velocities: list[list[float]] = [[] for _ in range(self.n_stories)]
        accelerations: list[list[float]] = [[] for _ in range(self.n_stories)]
        drifts: list[list[float]] = [[] for _ in range(self.n_stories)]
        base_shears: list[float] = []

        converged = True
        step = 0

        while current_time < total_time - dt_analysis / 2:
            ok = self._analyzer.analyze_one_step(dt_analysis)

            if not ok:
                logger.error(
                    "NLTHA failed at t=%.4f s (%.1f%% complete) for %s",
                    current_time,
                    100 * current_time / total_time,
                    gm.name,
                )
                converged = False
                break

            current_time = ops.getTime()
            step += 1

            # Record every N steps
            if step % self.config.save_every_n == 0:
                times.append(current_time)

                # Interpolate ground acceleration at current time
                idx = min(int(current_time / dt_gm), n_pts - 1)
                ground_accels.append(float(acc_mps2[idx]))

                # Floor responses
                total_base_shear = 0.0
                for i, node in enumerate(floor_nodes):
                    story = i + 1
                    disp = ops.nodeDisp(node, 1)  # Horizontal displacement
                    vel = ops.nodeVel(node, 1)  # Horizontal velocity
                    acc = ops.nodeAccel(node, 1)  # Horizontal acceleration

                    displacements[i].append(disp)
                    velocities[i].append(vel)
                    accelerations[i].append(acc)

                    # Inter-story drift ratio
                    disp_below = 0.0 if story == 1 else ops.nodeDisp((story - 1) * 100, 1)
                    idr = (disp - disp_below) / story_heights[i]
                    drifts[i].append(idr)

                # Base shear (sum of base reactions)
                ops.reactions()  # Must compute reactions before reading them
                for bay in range(self.n_bays + 1):
                    base_tag = bay
                    total_base_shear += ops.nodeReaction(base_tag, 1)
                base_shears.append(total_base_shear)

        t_elapsed = timer.perf_counter() - t_start

        # ── Compute peak responses ──────────────────────────────────────
        max_drift = [
            float(np.max(np.abs(drifts[i]))) if drifts[i] else 0.0 for i in range(self.n_stories)
        ]
        max_disp = [
            float(np.max(np.abs(displacements[i]))) if displacements[i] else 0.0
            for i in range(self.n_stories)
        ]
        peak_base_shear = float(np.max(np.abs(base_shears))) if base_shears else 0.0

        # ── Save results ────────────────────────────────────────────────
        output_file, metadata_file = self._save_results(
            gm,
            times,
            ground_accels,
            displacements,
            velocities,
            accelerations,
            drifts,
            base_shears,
            converged,
            t_elapsed,
            max_drift,
            max_disp,
            peak_base_shear,
        )

        result = {
            "converged": converged,
            "duration_s": round(t_elapsed, 2),
            "n_steps": step,
            "max_drift": max_drift,
            "max_disp": max_disp,
            "peak_base_shear": peak_base_shear,
            "output_file": str(output_file),
            "metadata_file": str(metadata_file),
            "convergence_stats": self._analyzer.stats,
        }

        status = "CONVERGED ✅" if converged else "FAILED ❌"
        logger.info(
            "NLTHA %s: %s in %.1f s (%d steps). Max IDR=%.4f (story %d)",
            gm.name,
            status,
            t_elapsed,
            step,
            max(max_drift) if max_drift else 0.0,
            (max_drift.index(max(max_drift)) + 1) if max_drift else 0,
        )

        return result

    def _save_results(
        self,
        gm: GroundMotionRecord,
        times: list[float],
        ground_accels: list[float],
        displacements: list[list[float]],
        velocities: list[list[float]],
        accelerations: list[list[float]],
        drifts: list[list[float]],
        base_shears: list[float],
        converged: bool,
        elapsed: float,
        max_drift: list[float],
        max_disp: list[float],
        peak_base_shear: float,
    ) -> tuple[Path, Path]:
        """Save CSV data and JSON metadata to data/raw/."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_name = gm.name.replace("/", "_").replace(" ", "_")
        csv_path = out_dir / f"{safe_name}.csv"
        json_path = out_dir / f"{safe_name}_meta.json"

        # ── CSV output ──────────────────────────────────────────────────
        headers = ["time", "ground_accel"]
        for i in range(1, self.n_stories + 1):
            headers.extend([f"disp_{i}", f"vel_{i}", f"accel_{i}", f"drift_{i}"])
        headers.append("base_shear")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for idx in range(len(times)):
                row = [times[idx], ground_accels[idx]]
                for i in range(self.n_stories):
                    row.extend(
                        [
                            displacements[i][idx],
                            velocities[i][idx],
                            accelerations[i][idx],
                            drifts[i][idx],
                        ]
                    )
                row.append(base_shears[idx])
                writer.writerow(row)

        # ── JSON metadata ───────────────────────────────────────────────
        metadata = {
            "ground_motion": {
                "name": gm.name,
                "source": gm.source,
                "dt": gm.dt,
                "npts": gm.npts,
                "duration": gm.duration,
                "scale_factor": gm.scale_factor,
                "units": gm.units,
            },
            "analysis": {
                "dt": self.config.dt,
                "integrator": "Newmark",
                "gamma": self.config.gamma,
                "beta": self.config.beta,
                "tolerance": self.config.tol,
                "max_iterations": self.config.max_iter,
                "adaptive_dt_min": self.config.dt_min,
                "max_bisections": self.config.max_bisections,
            },
            "results": {
                "converged": converged,
                "wall_clock_s": round(elapsed, 2),
                "n_steps_recorded": len(times),
                "max_drift_per_story": max_drift,
                "max_drift_overall": max(max_drift) if max_drift else 0.0,
                "critical_story": (max_drift.index(max(max_drift)) + 1) if max_drift else 0,
                "max_disp_per_story": max_disp,
                "peak_base_shear_kN": peak_base_shear,
            },
            "convergence": self._analyzer.stats,
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Results saved: %s, %s", csv_path.name, json_path.name)
        return csv_path, json_path


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner for parametric studies
# ═══════════════════════════════════════════════════════════════════════════


def _worker_task(args: tuple) -> dict:
    """Worker function for parallel NLTHA execution.

    Must be top-level for multiprocessing pickling.
    """
    gm, model_builder, config, n_stories, n_bays = args

    # Ensure clean state in worker process
    if OPS_AVAILABLE:
        ops.wipe()

    # Rebuild model
    model_builder()

    # Run analysis
    runner = NLTHARunner(n_stories=n_stories, n_bays=n_bays, config=config)
    try:
        result = runner.run(gm)
    except Exception as exc:
        logger.error("Worker failed for %s: %s", gm.name, exc)
        result = {
            "converged": False,
            "error": str(exc),
            "ground_motion": gm.name,
            "max_drift": [0.0] * n_stories,
            "peak_base_shear": 0.0,
        }

    # Cleanup
    if OPS_AVAILABLE:
        ops.wipe()

    return result


def run_batch(
    ground_motions: list[GroundMotionRecord],
    model_builder: callable,
    config: NLTHAConfig | None = None,
    n_stories: int = 5,
    n_bays: int = 3,
    n_workers: int = 1,
) -> list[dict]:
    """Run NLTHA for multiple ground motions (sequential or parallel).

    Parameters
    ----------
    ground_motions : list[GroundMotionRecord]
        List of ground motion records to analyze.
    model_builder : callable
        Function that builds and prepares the model.
    config : NLTHAConfig, optional
        Analysis configuration.
    n_stories, n_bays : int
        Frame dimensions.
    n_workers : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    list[dict]
        Analysis results for each ground motion.
    """
    import multiprocessing

    results = []
    total = len(ground_motions)

    if n_workers > 1:
        logger.info("Starting parallel batch NLTHA with %d workers", n_workers)

        # Prepare task arguments
        tasks = [(gm, model_builder, config, n_stories, n_bays) for gm in ground_motions]

        # Execute in pool
        with multiprocessing.Pool(processes=n_workers) as pool:
            # imap_unordered is faster as we don't strictly need order,
            # but we want to track progress. using imap to keep order is fine/easier for debugging logic
            # or actually unordered is fine since we return full results dicts.
            # let's use imap to be safe with pickling/iteration
            for i, result in enumerate(pool.imap(_worker_task, tasks), 1):
                result["batch_index"] = i
                result["batch_total"] = total
                results.append(result)

                if i % 5 == 0 or i == total:
                    logger.info("Batch progress: %d/%d completed", i, total)

    else:
        logger.info("Starting sequential batch NLTHA (1 worker)")
        for i, gm in enumerate(ground_motions, 1):
            logger.info("━" * 60)
            logger.info("Batch run %d/%d: %s", i, total, gm.name)
            logger.info("━" * 60)

            # Rebuild model from scratch (clean state)
            model_builder()

            runner = NLTHARunner(n_stories=n_stories, n_bays=n_bays, config=config)
            result = runner.run(gm)
            result["batch_index"] = i
            result["batch_total"] = total
            results.append(result)

            # Clean up
            ops.wipe()

    # Summary
    n_converged = sum(1 for r in results if r.get("converged", False))
    logger.info(
        "Batch complete: %d/%d converged (%.0f%%)",
        n_converged,
        total,
        100 * n_converged / total if total > 0 else 0,
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic ground motion for testing
# ═══════════════════════════════════════════════════════════════════════════


def generate_synthetic_record(
    duration: float = 20.0,
    dt: float = 0.01,
    pga_g: float = 0.3,
    freq_hz: float = 2.0,
    name: str = "Synthetic_Test",
) -> GroundMotionRecord:
    """Generate a synthetic ground motion for model verification.

    Uses a modulated sinusoidal signal with Gaussian envelope.
    NOT for production — only for testing convergence and pipeline.
    """
    n_pts = int(duration / dt) + 1
    t = np.linspace(0, duration, n_pts)

    # Gaussian envelope (peak at duration/3)
    t_peak = duration / 3
    sigma = duration / 6
    envelope = np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)

    # Modulated sinusoid with frequency content
    acc = pga_g * envelope * np.sin(2 * np.pi * freq_hz * t)

    # Add secondary frequency for realistic broadband content
    acc += 0.3 * pga_g * envelope * np.sin(2 * np.pi * freq_hz * 2.5 * t + 0.7)

    return GroundMotionRecord(
        name=name,
        acceleration=acc,
        dt=dt,
        units="g",
        scale_factor=1.0,
        source="Synthetic (verification only)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.opensees_analysis.ospy_model import RCFrameModel

    print("=" * 60)
    print("  NLTHA Runner — Synthetic Ground Motion Test")
    print("=" * 60)

    # Build model
    model = RCFrameModel()
    model.build()
    T1 = model.get_fundamental_period()
    print(f"T1 = {T1:.4f} s")

    model.apply_gravity()
    model.setup_rayleigh_damping()

    # Synthetic record
    gm = generate_synthetic_record(
        duration=15.0,
        dt=0.01,
        pga_g=0.25,
        freq_hz=1.0 / T1,
    )
    print(f"Ground motion: {gm.name} ({gm.duration:.0f}s, PGA={gm.acceleration.max():.3f}g)")

    # Run NLTHA
    runner = NLTHARunner(n_stories=5, n_bays=3)
    result = runner.run(gm)

    # Report
    print(f"\n{'=' * 60}")
    print("  NLTHA Results")
    print(f"{'=' * 60}")
    print(f"  Status:        {'CONVERGED ✅' if result['converged'] else 'FAILED ❌'}")
    print(f"  Wall-clock:    {result['duration_s']:.1f} s")
    print(f"  Steps:         {result['n_steps']}")
    print(f"  Peak base shear: {result['peak_base_shear']:.1f} kN")
    print(f"\n  {'Story':<8} {'Max Drift':<12} {'Max Disp (m)':<14}")
    print(f"  {'-' * 34}")
    for i in range(5):
        print(f"  {i + 1:<8} {result['max_drift'][i]:<12.6f} {result['max_disp'][i]:<14.6f}")
    print(f"\n  Convergence: {result['convergence_stats']}")
    print(f"  Output: {result['output_file']}")
