# -*- coding: utf-8 -*-
"""v2 GPU global solver on the optim_bench workload: time + accuracy.

Runs the mixed-precision global factor-once/solve-many solver over the 100
smooth_noise samples (5 normal tools x 128 depths, 0..25.4 m) and compares:

  time     — warm s/sample vs the recorded pipeline wall times
             (V1_baseline CG ~41.2 s, Vd_direct_forced ~11.8 s,
              axis-study pinned direct b5 ~9.6 s);
  accuracy — per-tool max/mean relative difference vs the stored pipeline
             logs (Vd = forced-direct, the exact CPU reference; V1 = CG
             baseline as context). NOTE (Phase 0 finding): the pipeline uses
             the scalar-mud-per-solve convention and truncated per-depth
             domains; with the noisy RM logs of this dataset (±4%, jumps to
             ~5%) systematic sub-% to % differences at RM-jump depths are
             the CONVENTION, not solver error — see
             wiki/findings/gpu-solver-v2.md.

Usage:
    python scripts/gpu_v2_optim_bench.py [--samples 100] [--batch 10]
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

from remo3d.gpu_solver import global_op, global_gpu  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "smooth_noise" / "data"
PIPE_DIR = ROOT / "benchmark_data" / "optim_bench" / "full_pipeline"
OUT = ROOT / "benchmark_data" / "gpu_solver" / "global_optim_bench.npz"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args(argv)

    ref = {v: np.load(PIPE_DIR / f"{v}.npz", allow_pickle=True)
           for v in ("V1_baseline", "Vd_direct_forced")}
    depths = np.asarray(ref["Vd_direct_forced"]["depths"], float)
    ref_tools = [str(t) for t in ref["Vd_direct_forced"]["tools"]]
    assert ref_tools == TOOLS

    n = args.samples
    s0 = args.start
    fs, bs = [], []
    for si in range(s0, s0 + n):
        d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
        fs.append(np.asarray(d["formation_model"], float))
        bs.append(np.asarray(d["borehole_model"], float))
    fs, bs = np.stack(fs), np.stack(bs)

    print(f"device: {jax.devices()[0]}; {n} samples, "
          f"{len(TOOLS)} tools x {len(depths)} depths", flush=True)
    p = global_op.build_global_tasks(TOOLS, depths, fs[0], bs[0])
    print(f"grid {len(p['r_nodes'])}x{len(p['z_nodes'])} "
          f"n_free={p['n_free']:,} k={len(p['uniq_src'])} "
          f"(dedup x{p['n_tasks'] / len(p['uniq_src']):.2f})", flush=True)
    solver = global_gpu.make_solver(p, precision="mixed")

    B = args.batch
    logs = np.empty((n, len(TOOLS), len(depths)))
    times = []
    for lo in range(0, (n // B) * B, B):
        t0 = time.perf_counter()
        X0 = solver(fs[lo:lo + B], bs[lo:lo + B])
        X0.block_until_ready()
        dt = time.perf_counter() - t0
        times.append(dt)
        for b, sample_logs in enumerate(global_gpu.extract_logs(p, X0)):
            for ti, t in enumerate(TOOLS):
                logs[lo + b, ti] = sample_logs[t]
        print(f"  batch [{lo:3d}:{lo + B:3d}]: {dt:6.2f}s "
              f"({dt / B:5.2f} s/sample)"
              + ("   <- includes compile" if lo == 0 else ""), flush=True)
    if n % B:
        lo = (n // B) * B
        X0 = solver(fs[lo - (B - n % B):lo + (n % B)][-B:],
                    bs[lo - (B - n % B):lo + (n % B)][-B:])
        for b, sample_logs in enumerate(global_gpu.extract_logs(p, X0)):
            si = lo - (B - n % B) + b
            for ti, t in enumerate(TOOLS):
                logs[si, ti] = sample_logs[t]

    out = (OUT if s0 == 0 else
           OUT.with_name(OUT.stem + f"_s{s0}" + OUT.suffix))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, logs=logs, depths=depths, tools=TOOLS,
                        n_samples=n, start=s0)

    warm = times[1:] or times
    per = float(np.mean(warm)) / B
    v1_t = float(np.mean(ref["V1_baseline"]["wall_time"][s0:s0 + n]))
    vd_t = float(np.mean(ref["Vd_direct_forced"]["wall_time"][s0:s0 + n]))
    print(f"\nTIME: v2 GPU warm {per:.2f} s/sample "
          f"| V1 CG pipeline {v1_t:.2f}s (x{v1_t / per:.0f}) "
          f"| Vd forced-direct {vd_t:.2f}s (x{vd_t / per:.0f}) "
          f"| axis b5 9.60s (x{9.60 / per:.0f})")

    print("\nACCURACY vs stored pipeline logs (rel):")
    for name in ("Vd_direct_forced", "V1_baseline"):
        rl = np.asarray(ref[name]["logs"], float)[s0:s0 + n]
        rel = np.abs(logs - rl) / np.abs(rl)
        line = "  ".join(
            f"{t}: max {np.nanmax(rel[:, ti]):.2%} "
            f"mean {np.nanmean(rel[:, ti]):.3%}"
            for ti, t in enumerate(TOOLS))
        print(f"  vs {name:16s}: {line}")
    rl1 = np.asarray(ref["V1_baseline"]["logs"], float)[s0:s0 + n]
    rld = np.asarray(ref["Vd_direct_forced"]["logs"], float)[s0:s0 + n]
    spread = np.abs(rl1 - rld) / np.abs(rld)
    print(f"  context: V1-vs-Vd internal spread max {np.nanmax(spread):.2%} "
          f"mean {np.nanmean(spread):.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
