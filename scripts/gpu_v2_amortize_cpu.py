# -*- coding: utf-8 -*-
"""E1 (GPU_SOLVER_V2_PLAN Phase 0): factor-once/solve-many CPU control.

Builds the shared global operator per len512 sample (global_op), factors it
ONCE with the global block-Thomas (dense Schur Cholesky per z-row — v1's
algorithm, but one factorization per SAMPLE instead of per (tool, depth)),
and solves all 1280 tasks as deduplicated RHS columns (BLAS-3 dpotrs blocks). Reports the
time breakdown and the amortized s/sample against the v1 GPU baseline
(7.65 s/sample) and the 32-worker CPU pipeline (~40 s/sample), plus accuracy
vs the stored v1 direct-solver logs (benchmark_data/gpu_solver/direct_len512,
themselves validated to 0.4% worst vs fresh NGSolve).

This is deliberately solver-agnostic evidence: if the amortized CPU time
already beats the per-depth GPU path, the architecture question is settled
before any cuDSS work.

Usage:
    python scripts/gpu_v2_amortize_cpu.py [--samples 3] [--block 64]
        [--no-dedup]  (solve all 1280 columns instead of the 531 unique)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# BLAS pinned to 1 thread BEFORE numpy loads: the hot kernels are 111x111 —
# threaded OpenBLAS on them is a measured >10x slowdown (oversubscription).
# Parallelism lives in the RHS blocks (global_op.solve_logs thread pool).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remo3d.gpu_solver import global_op  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"
V1_DIR = ROOT / "benchmark_data" / "gpu_solver" / "direct_len512"

V1_GPU_BASELINE = 7.65   # s/sample, warm (WORK_SUMMARY A.5)
CPU_PIPELINE = 40.0      # s/sample, forced-direct 32 workers


def run_sample(si, block=32, n_jobs=16, dedup=True):
    d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
    formation = np.asarray(d["formation_model"], float)
    borehole = np.asarray(d["borehole_model"], float)
    depths = borehole[:, 0].astype(float)

    t0 = time.perf_counter()
    p = global_op.build_global_problem(TOOLS, depths, formation, borehole)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    fac = global_op.factor_block_thomas(p)
    t_factor = time.perf_counter() - t0

    if not dedup:
        # Pretend every task is its own column (no reciprocity/dedup win).
        all_src = np.concatenate([p["tasks"][t]["src"] for t in TOOLS])
        p["uniq_src"] = all_src
        ofs = 0
        for t in TOOLS:
            n = len(p["depths"])
            p["tasks"][t]["col"] = np.arange(ofs, ofs + n)
            ofs += n

    t0 = time.perf_counter()
    Ra = global_op.solve_logs(p, fac, block=block, n_jobs=n_jobs)
    t_solve = time.perf_counter() - t0

    total = t_build + t_factor + t_solve
    ncols = len(p["uniq_src"])
    print(f"sample {si}: build {t_build:5.2f}s  factor {t_factor:5.2f}s  "
          f"solve[{ncols} cols] {t_solve:5.2f}s  -> total {total:6.2f}s")

    # Accuracy vs stored v1 GPU logs (fp32 direct block-Thomas).
    ref_file = V1_DIR / f"direct_sample_{si}.npz"
    if ref_file.exists():
        ref = np.load(ref_file, allow_pickle=True)
        ref_logs = np.asarray(ref["logs"], float)
        ref_tools = [str(t) for t in ref["tools"]]
        rel = {}
        for ti, t in enumerate(ref_tools):
            r = np.abs(Ra[t] - ref_logs[ti]) / np.abs(ref_logs[ti])
            rel[t] = (float(np.nanmax(r)), float(np.nanmean(r)))
        worst = max(v[0] for v in rel.values())
        print("  vs v1 GPU logs: worst {:.3%}; per tool "
              .format(worst)
              + "  ".join(f"{t}: max {v[0]:.3%} mean {v[1]:.3%}"
                          for t, v in rel.items()))
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args(argv)

    totals = [run_sample(si, block=args.block, n_jobs=args.jobs,
                         dedup=not args.no_dedup)
              for si in range(args.samples)]
    mean = float(np.mean(totals))
    print(f"\namortized: {mean:.2f} s/sample "
          f"(v1 GPU {V1_GPU_BASELINE:.2f}s -> x{V1_GPU_BASELINE / mean:.2f}; "
          f"CPU pipeline {CPU_PIPELINE:.0f}s -> x{CPU_PIPELINE / mean:.1f}; "
          f"E1 gate <{0.4 * V1_GPU_BASELINE:.2f}s: "
          f"{'PASS' if mean < 0.4 * V1_GPU_BASELINE else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
