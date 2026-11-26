"""Minimal, honest GPU benchmark harness.

Usage:
    from benchmarks.benchmark import bench, check, record
    ok, err = check(out, ref, atol=1e-5, rtol=1e-5)
    stats = bench(lambda: my_kernel(x, y))
    record("vector_add", "v1_naive", shape=(n,), dtype="fp32", layout="contiguous",
           stats=stats, correct=ok, max_err=err, bytes_moved=3 * n * 4)

Smoke test:  python benchmarks/benchmark.py
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import statistics
import subprocess
from pathlib import Path

import torch

RESULTS = Path(__file__).with_name("results.csv")
FIELDS = [
    "timestamp", "name", "version", "shape", "dtype", "layout", "config",
    "gpu", "compute_capability", "driver", "cuda_runtime", "torch", "triton", "nvcc",
    "warmup", "iters", "l2_flush", "median_us", "p10_us", "p90_us", "stdev_us",
    "correct", "max_err", "gbps", "tflops", "notes",
]


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "n/a"


def env_info() -> dict:
    """Environment recorded with every result."""
    info = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda or "n/a"}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = "n/a"
    nvcc = _cmd(["nvcc", "--version"]).splitlines()
    info["nvcc"] = nvcc[-1] if nvcc else "n/a"
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(dev)
        info["gpu"] = torch.cuda.get_device_name(dev)
        info["compute_capability"] = f"{major}.{minor}"
        drv = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        info["driver"] = drv.splitlines()[dev] if drv not in ("", "n/a") else "n/a"
    else:
        info.update(gpu="none", compute_capability="n/a", driver="n/a")
    return info


_l2_buffer = None


def _flush_l2() -> None:
    """Overwrite a buffer larger than L2 so each timed run starts cold."""
    global _l2_buffer
    if _l2_buffer is None:
        _l2_buffer = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _l2_buffer.zero_()


def bench(fn, warmup: int = 10, iters: int = 100, l2_flush: bool = True) -> dict:
    """Time fn() with CUDA events, one event pair per iteration. Returns microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if l2_flush:
            _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    times.sort()
    q = lambda p: times[min(len(times) - 1, int(p * len(times)))]
    return {
        "warmup": warmup, "iters": iters, "l2_flush": l2_flush,
        "median_us": statistics.median(times), "p10_us": q(0.10), "p90_us": q(0.90),
        "stdev_us": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def check(out: torch.Tensor, ref: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5):
    """Compare against a trusted reference. Returns (ok, max_abs_err)."""
    if out.shape != ref.shape:
        return False, float("inf")
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item() if diff.numel() else 0.0
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    return bool(ok), max_err


def record(name: str, version: str, *, shape, dtype: str, layout: str, stats: dict,
           correct: bool, max_err: float = 0.0, config: str = "",
           bytes_moved: int | None = None, flops: int | None = None, notes: str = "") -> dict:
    """Append one result row (with environment) to results.csv."""
    sec = stats["median_us"] * 1e-6
    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "name": name, "version": version, "shape": "x".join(map(str, shape)),
        "dtype": dtype, "layout": layout, "config": config,
        **env_info(), **stats,
        "correct": correct, "max_err": f"{max_err:.3e}",
        "gbps": f"{bytes_moved / sec / 1e9:.1f}" if bytes_moved else "",
        "tflops": f"{flops / sec / 1e12:.2f}" if flops else "",
        "notes": notes,
    }
    for k in ("median_us", "p10_us", "p90_us", "stdev_us"):
        row[k] = f"{row[k]:.2f}"
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


if __name__ == "__main__":
    print(env_info())
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible: environment check only.")
    n = 1 << 24
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    ok, err = check(a + b, (a.cpu() + b.cpu()).cuda())
    stats = bench(lambda: a + b)
    row = record("vector_add", "torch_baseline", shape=(n,), dtype="fp32",
                 layout="contiguous", stats=stats, correct=ok, max_err=err,
                 bytes_moved=3 * n * 4, notes="smoke test")
    print(f"median {row['median_us']} us, {row['gbps']} GB/s, correct={ok}")
