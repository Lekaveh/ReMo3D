# -*- coding: utf-8 -*-
"""Phase G1: GPU global block-Thomas — correctness + throughput benchmark.

Correctness: fp64, one sample, vs the CPU global control
(benchmark_data/gpu_solver/global_len512/global_sample_0.npz — the exact
same operator, so agreement should be ~1e-10). fp32 is then compared to fp64.

Throughput: warm s/sample over N samples at batch sizes B, against the v1
GPU baseline (7.65 s/sample) and the E1 gate (<3.06 s/sample).

Usage:
    python scripts/gpu_v2_global_gpu_bench.py [--samples 8] [--batch 2]
        [--dtype f64|f32] [--skip-correctness]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from remo3d.gpu_solver import global_op, global_gpu  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"
CPU_REF = (ROOT / "benchmark_data" / "gpu_solver" / "global_len512"
           / "global_sample_0.npz")

V1_GPU_BASELINE = 7.65
GATE = 0.4 * V1_GPU_BASELINE


def load_samples(n):
    fs, bs = [], []
    for si in range(n):
        d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
        fs.append(np.asarray(d["formation_model"], float))
        bs.append(np.asarray(d["borehole_model"], float))
    return np.stack(fs), np.stack(bs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dtype", choices=["f64", "mixed"], default="mixed")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args(argv)
    precision = "f64" if args.dtype == "f64" else "mixed"

    print(f"device: {jax.devices()[0]}", flush=True)
    fs, bs = load_samples(max(args.samples, 1))
    depths = bs[0][:, 0]

    t0 = time.perf_counter()
    p = global_op.build_global_tasks(TOOLS, depths, fs[0], bs[0])
    print(f"tasks: grid {len(p['r_nodes'])}x{len(p['z_nodes'])} "
          f"n_free={p['n_free']:,} k={len(p['uniq_src'])} "
          f"({time.perf_counter() - t0:.1f}s host)", flush=True)

    solver = global_gpu.make_solver(p, precision=precision)

    # --- correctness on sample 0 ---
    if not args.skip_correctness:
        t0 = time.perf_counter()
        X0 = solver(fs[:1], bs[:1])
        X0.block_until_ready()
        print(f"compile+first solve: {time.perf_counter() - t0:.1f}s",
              flush=True)
        logs = global_gpu.extract_logs(p, X0)[0]
        if CPU_REF.exists():
            ref = np.load(CPU_REF, allow_pickle=True)
            ref_logs = {str(t): np.asarray(ref["logs"], float)[ti]
                        for ti, t in enumerate(ref["tools"])}
            worst = max(float(np.max(np.abs(logs[t] - ref_logs[t])
                                     / np.abs(ref_logs[t]))) for t in TOOLS)
            print(f"vs CPU global control ({args.dtype}): worst rel "
                  f"{worst:.3e}", flush=True)

    # --- throughput ---
    B = args.batch
    n = (args.samples // B) * B
    if n == 0:
        return 0
    times = []
    for lo in range(0, n, B):
        t0 = time.perf_counter()
        X0 = solver(fs[lo:lo + B], bs[lo:lo + B])
        X0.block_until_ready()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  batch [{lo}:{lo + B}]: {dt:6.2f}s "
              f"({dt / B:5.2f} s/sample)"
              + ("   <- includes compile" if lo == 0 and args.skip_correctness
                 else ""), flush=True)
    warm = times[1:] or times
    per = float(np.mean(warm)) / B
    print(f"\n{args.dtype} B={B}: warm {per:.2f} s/sample "
          f"(v1 7.65 -> x{V1_GPU_BASELINE / per:.1f}; "
          f"gate <{GATE:.2f}s: {'PASS' if per < GATE else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
