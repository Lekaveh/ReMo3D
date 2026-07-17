"""Demonstrate JIT-compile amortization across len512 samples (GPU only).

Runs the GPU solver on several len512 samples back-to-back (same tool set, so
the grid shapes — hence the compiled kernels — are reused after the first
sample) and prints per-sample wall time. Sample 0 pays compilation; later
samples are near-pure GPU throughput. This is the fair GPU-vs-CPU picture: the
CPU pipeline does this workload in ~24 s/sample (20 cores, gmsh) with no
compile step.

Usage:
    python scripts/gpu_len512_amortize.py --n-samples 4 [--shared] [--path per-tool]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp  # noqa: E402
from remo3d.gpu_solver.driver import compute_logs_gpu  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--shared", action="store_true")
    p.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)
    dtype = jnp.float32 if args.dtype == "f32" else jnp.float64

    print(f"len512, {len(TOOLS)} tools x 256 depths = 1280 solves/sample; "
          f"path={'shared' if args.shared else 'per-tool'}, {args.dtype}")
    times = []
    for si in range(args.n_samples):
        d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
        formation = np.asarray(d["formation_model"], float)
        borehole = np.asarray(d["borehole_model"], float)
        depths = borehole[:, 0].astype(float)
        t0 = time.perf_counter()
        compute_logs_gpu(TOOLS, depths, formation, borehole,
                         borehole_geometry_type="radius", backend="fv",
                         dtype=dtype, tol=args.tol, batch_size=args.batch_size,
                         shared_grid=args.shared)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  sample {si}: {dt:6.1f}s  ({1000 * dt / 1280:.0f} ms/solve)"
              + ("   <- includes compile" if si == 0 else ""))

    warm = times[1:] or times
    print(f"\nsample 0 (cold): {times[0]:.1f}s; "
          f"warm mean: {np.mean(warm):.1f}s "
          f"({1000 * np.mean(warm) / 1280:.0f} ms/solve, "
          f"{1280 / np.mean(warm):.0f} solves/s)")
    # CPU reference: the 20-worker MPI pipeline PRODUCES ALL NaN on these noisy
    # 256-layer len512 models (worker meshing/solve fails per-task -> NaN fill;
    # see cpu_len512_baseline.py output). The only valid CPU reference is the
    # single-process gmsh path in compare_len512.py: ~3.5 s/solve, 1 core.
    print("CPU ref: 20-worker pipeline returns all-NaN on len512 (fails); "
          "valid single-process gmsh = 3.5 s/solve/core.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
