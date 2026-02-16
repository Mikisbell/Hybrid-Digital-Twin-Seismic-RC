"""
benchmark_latency.py — PINN Inference Latency Benchmarking
==========================================================

Validates the "real-time" claim by measuring inference latency under
controlled conditions.  A structural reviewer will reject "real-time"
if single-sample inference exceeds 100 ms on CPU.

Usage
-----
    python -m src.pinn.benchmark_latency --model_path data/models/pinn_best.pt
    python -m src.pinn.benchmark_latency --dummy   # synthetic model for CI

Output
------
    Prints a Markdown-formatted table suitable for manuscript/04_results.md
    and saves results to data/models/benchmark_results.json.

References
----------
    HRPUB requirement: every quantitative claim must be reproducible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Try importing torch; if unavailable, report and exit gracefully.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Dummy PINN model (used when --dummy flag is set or no model file exists)
# Architecture mirrors the planned model: 1D-CNN encoder → FC layers
# ═══════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:

    class DummyPINN(nn.Module):
        """Lightweight surrogate that mirrors the production PINN shape.

        Input : (batch, 1, seq_len)  — single-channel acceleration series
        Output: (batch, 5)           — predicted IDR per story
        """

        def __init__(self, seq_len: int = 2048) -> None:
            super().__init__()
            # 1D-CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
                nn.SiLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(16),
            )
            # Fully-connected head (256-128-64-32-5)
            self.head = nn.Sequential(
                nn.Linear(64 * 16, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Linear(32, 5),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)  # (B, 64, 16)
            z = z.view(z.size(0), -1)  # (B, 1024)
            return self.head(z)  # (B, 5)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarking engine
# ═══════════════════════════════════════════════════════════════════════════

REALTIME_THRESHOLD_MS = 100.0  # Maximum acceptable latency for "real-time"


def _generate_dummy_input(
    batch_size: int = 1, seq_len: int = 2048, device: str = "cpu"
) -> torch.Tensor:
    """Create a synthetic acceleration time-series tensor."""
    return torch.randn(batch_size, 1, seq_len, device=device)


def preprocess_signal(raw_signal: torch.Tensor) -> torch.Tensor:
    """Simulate real-world pre-processing of seismic signal.

    In a real deployment scenario, T_total = T_sensing + T_pre + T_inference.
    This function measures T_pre: baseline correction, bandpass filtering,
    and normalization that would occur before inference.

    Parameters
    ----------
    raw_signal : torch.Tensor
        Raw acceleration signal (batch, 1, seq_len).

    Returns
    -------
    torch.Tensor
        Preprocessed signal ready for inference.
    """
    # Baseline correction (remove mean)
    signal = raw_signal - raw_signal.mean(dim=-1, keepdim=True)
    # Normalization (zero mean, unit variance per sample)
    std = signal.std(dim=-1, keepdim=True).clamp(min=1e-8)
    signal = signal / std
    # Taper edges (5% cosine taper to reduce spectral leakage)
    seq_len = signal.shape[-1]
    taper_len = max(1, seq_len // 20)
    taper = torch.ones(seq_len, device=signal.device)
    ramp = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, taper_len, device=signal.device)))
    taper[:taper_len] = ramp
    taper[-taper_len:] = ramp.flip(0)
    signal = signal * taper
    return signal


def benchmark(
    model: nn.Module,
    device: str = "cpu",
    seq_len: int = 2048,
    n_warmup: int = 50,
    n_iterations: int = 1000,
    batch_sizes: tuple[int, ...] = (1, 8, 32, 128),
    include_preprocessing: bool = True,
) -> dict:
    """Run the full benchmarking protocol.

    Returns a dict with:
      - cold_start_ms   : first inference after model load
      - warm_mean_ms    : mean latency over *n_iterations* (batch=1)
      - warm_median_ms  : median latency
      - warm_p95_ms     : 95th percentile
      - warm_p99_ms     : 99th percentile
      - warm_std_ms     : standard deviation
      - batch_throughput: {batch_size: samples_per_second}
      - device          : cpu / cuda
      - realtime_ok     : bool (warm_mean_ms <= threshold)
    """
    model = model.to(device).eval()
    results: dict = {
        "device": device,
        "seq_len": seq_len,
        "include_preprocessing": include_preprocessing,
    }

    # ── Cold start (full cycle: preprocess + inference) ─────────────────
    x_cold = _generate_dummy_input(1, seq_len, device)
    with torch.no_grad():
        t0 = time.perf_counter()
        if include_preprocessing:
            x_cold = preprocess_signal(x_cold)
        _ = model(x_cold)
        t1 = time.perf_counter()
    results["cold_start_ms"] = round((t1 - t0) * 1000, 3)

    # ── Measure preprocessing latency separately ────────────────────────
    pre_latencies: list[float] = []
    for _ in range(min(n_iterations, 200)):
        x_pre = _generate_dummy_input(1, seq_len, device)
        t0 = time.perf_counter()
        _ = preprocess_signal(x_pre)
        t1 = time.perf_counter()
        pre_latencies.append((t1 - t0) * 1000)
    results["preprocess_mean_ms"] = round(statistics.mean(pre_latencies), 3)
    results["preprocess_p95_ms"] = round(sorted(pre_latencies)[int(0.95 * len(pre_latencies))], 3)

    # ── Warm-up phase (discard) ─────────────────────────────────────────
    x_warm = _generate_dummy_input(1, seq_len, device)
    if include_preprocessing:
        x_warm = preprocess_signal(x_warm)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_warm)

    # ── Warm inference (batch=1, full cycle: T_pre + T_inference) ───────
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(n_iterations):
            x = _generate_dummy_input(1, seq_len, device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if include_preprocessing:
                x = preprocess_signal(x)
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    latencies_sorted = sorted(latencies)
    results["warm_mean_ms"] = round(statistics.mean(latencies), 3)
    results["warm_median_ms"] = round(statistics.median(latencies), 3)
    results["warm_std_ms"] = round(statistics.stdev(latencies), 3)
    idx_p95 = int(0.95 * len(latencies_sorted))
    idx_p99 = int(0.99 * len(latencies_sorted))
    results["warm_p95_ms"] = round(latencies_sorted[idx_p95], 3)
    results["warm_p99_ms"] = round(latencies_sorted[idx_p99], 3)
    results["n_iterations"] = n_iterations

    # ── Batch throughput ────────────────────────────────────────────────
    throughput: dict[int, float] = {}
    for bs in batch_sizes:
        x_batch = _generate_dummy_input(bs, seq_len, device)
        with torch.no_grad():
            # warm up
            for _ in range(10):
                _ = model(x_batch)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                _ = model(x_batch)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        elapsed = t1 - t0
        throughput[bs] = round((bs * 100) / elapsed, 1)
    results["batch_throughput"] = throughput

    # ── Verdict ─────────────────────────────────────────────────────────
    results["realtime_ok"] = results["warm_mean_ms"] <= REALTIME_THRESHOLD_MS

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-print for manuscript / terminal
# ═══════════════════════════════════════════════════════════════════════════


def print_results(results: dict) -> None:
    """Print benchmark results as a Markdown table."""
    verdict = "PASS ✅" if results["realtime_ok"] else "FAIL ❌"

    print("\n## PINN Inference Latency Benchmark\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Device | {results['device']} |")
    print(f"| Sequence length | {results['seq_len']} samples |")
    incl = "T_pre + T_inf" if results.get("include_preprocessing") else "T_inf only"
    print(f"| Cycle measured | {incl} |")
    print(f"| Preprocessing mean | {results.get('preprocess_mean_ms', 0):.3f} ms |")
    print(f"| Preprocessing P95 | {results.get('preprocess_p95_ms', 0):.3f} ms |")
    print(f"| Cold start | {results['cold_start_ms']:.3f} ms |")
    print(f"| Warm mean (n={results['n_iterations']}) | {results['warm_mean_ms']:.3f} ms |")
    print(f"| Warm median | {results['warm_median_ms']:.3f} ms |")
    print(f"| Warm P95 | {results['warm_p95_ms']:.3f} ms |")
    print(f"| Warm P99 | {results['warm_p99_ms']:.3f} ms |")
    print(f"| Std dev | {results['warm_std_ms']:.3f} ms |")
    print(f"| **Real-time (≤ {REALTIME_THRESHOLD_MS} ms)** | **{verdict}** |")
    print("\n> **Note**: T_total = T_sensing + T_pre + T_inference. ")
    print("> This benchmark measures T_pre + T_inference. ")
    print("> T_sensing depends on hardware (accelerometer sampling rate).")

    print("\n### Batch Throughput\n")
    print("| Batch Size | Samples/sec |")
    print("|------------|-------------|")
    for bs, sps in results["batch_throughput"].items():
        print(f"| {bs} | {sps:,.1f} |")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required.  pip install torch", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="PINN inference latency benchmark (real-time validation)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved .pt model (TorchScript or state_dict)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use a dummy model with production-equivalent architecture",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Input sequence length (default: 2048)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu for deployment target)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of warm iterations (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/benchmark_results.json",
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    # Load or create model
    if args.dummy or args.model_path is None:
        print("Using dummy PINN model (production-equivalent architecture)")
        model = DummyPINN(seq_len=args.seq_len)
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        try:
            model = torch.jit.load(str(model_path), map_location=args.device)
        except Exception:
            # Try loading as state_dict into DummyPINN
            model = DummyPINN(seq_len=args.seq_len)
            state = torch.load(str(model_path), map_location=args.device)
            model.load_state_dict(state)

    # Run benchmark
    print(f"Benchmarking on {args.device.upper()} with seq_len={args.seq_len}...")
    results = benchmark(
        model,
        device=args.device,
        seq_len=args.seq_len,
        n_iterations=args.iterations,
    )

    # Print results
    print_results(results)

    # Save to JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert int keys to str for JSON
    results_json = results.copy()
    results_json["batch_throughput"] = {str(k): v for k, v in results["batch_throughput"].items()}
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
