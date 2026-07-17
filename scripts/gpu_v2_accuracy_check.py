# -*- coding: utf-8 -*-
"""E2 (GPU_SOLVER_V2_PLAN Phase 0): three-way accuracy arbitration.

The global factor-once solve (v2) and the stored per-window v1 GPU logs
disagree by up to ~8-17% at isolated points (mean ~1.3%). Both are our own
discretizations; only a fresh NGSolve solve can arbitrate. This script:

  1. solves the sample with the global operator (v2) and stores the logs;
  2. ranks (tool, depth) points by |v2 - v1| relative difference;
  3. re-solves the worst K points (plus a few agreeing controls) with the
     single-process NGSolve gmsh forced-direct reference
     (compare_len512.ngsolve_direct_log, thread-pinned);
  4. prints depth, v1, v2, NGSolve and each solver's error vs NGSolve.

Usage:
    python scripts/gpu_v2_accuracy_check.py [--sample 0] [--worst 4]
        [--controls 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from remo3d.gpu_solver import global_op  # noqa: E402
from compare_len512 import ngsolve_direct_log  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"
V1_DIR = ROOT / "benchmark_data" / "gpu_solver" / "direct_len512"
OUT_DIR = ROOT / "benchmark_data" / "gpu_solver" / "global_len512"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--worst", type=int, default=4,
                    help="worst points per tool to re-solve with NGSolve")
    ap.add_argument("--controls", type=int, default=2,
                    help="agreeing points per tool as controls")
    args = ap.parse_args(argv)

    d = np.load(SAMPLES_DIR / f"sample_{args.sample}.npz", allow_pickle=True)
    formation = np.asarray(d["formation_model"], float)
    borehole = np.asarray(d["borehole_model"], float)
    depths = borehole[:, 0].astype(float)

    # --- v2 global solve ---
    t0 = time.perf_counter()
    p = global_op.build_global_problem(TOOLS, depths, formation, borehole)
    fac = global_op.factor_block_thomas(p)
    Ra = global_op.solve_logs(p, fac, block=128, n_jobs=4)
    print(f"v2 global solve: {time.perf_counter() - t0:.1f}s "
          f"(n={p['n_free']:,}, {len(p['uniq_src'])} cols)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"global_sample_{args.sample}.npz",
        logs=np.stack([Ra[t] for t in TOOLS]), depths=depths, tools=TOOLS)

    # --- v1 stored logs + disagreement profile ---
    ref = np.load(V1_DIR / f"direct_sample_{args.sample}.npz",
                  allow_pickle=True)
    v1 = {str(t): np.asarray(ref["logs"], float)[ti]
          for ti, t in enumerate(ref["tools"])}

    print("\nv2-vs-v1 disagreement profile (rel):")
    picks = {}
    for t in TOOLS:
        rel = np.abs(Ra[t] - v1[t]) / np.abs(v1[t])
        order = np.argsort(rel)[::-1]
        worst = order[:args.worst]
        ctrl = order[len(order) // 2:len(order) // 2 + args.controls]
        picks[t] = np.concatenate([worst, ctrl])
        prof = "  ".join(f"{depths[k]:5.1f}m:{rel[k]:.2%}" for k in worst)
        print(f"  {t}: max {rel.max():.2%} mean {rel.mean():.3%} "
              f"| worst @ {prof}")

    # --- NGSolve arbitration at the picked points ---
    try:
        import ngsolve as ngs
        ngs.SetNumThreads(1)          # worker thread-pinning lesson (B.3)
    except Exception:
        pass

    print("\nthree-way check (err vs fresh NGSolve):")
    rows = []
    for t in TOOLS:
        idx = picks[t]
        ra_ng, t_ng = ngsolve_direct_log(t, depths[idx], formation, borehole,
                                         progress_every=0)
        for k, ng in zip(idx, ra_ng):
            e1 = abs(v1[t][k] - ng) / abs(ng)
            e2 = abs(Ra[t][k] - ng) / abs(ng)
            tag = "worst" if k in idx[:args.worst] else "ctrl"
            rows.append((t, depths[k], v1[t][k], Ra[t][k], ng, e1, e2, tag))
        print(f"  [{t}] {len(idx)} NGSolve solves in {t_ng:.0f}s", flush=True)

    print(f"\n{'tool':10s} {'depth':>6s} {'v1':>9s} {'v2':>9s} "
          f"{'NGSolve':>9s} {'v1 err':>8s} {'v2 err':>8s}")
    for t, z, a, b, ng, e1, e2, tag in rows:
        print(f"{t:10s} {z:6.1f} {a:9.4f} {b:9.4f} {ng:9.4f} "
              f"{e1:8.3%} {e2:8.3%}  {tag}")

    w = [r for r in rows if r[7] == "worst"]
    print(f"\nsummary over worst points: "
          f"v1 err mean {np.mean([r[5] for r in w]):.3%} "
          f"max {np.max([r[5] for r in w]):.3%} | "
          f"v2 err mean {np.mean([r[6] for r in w]):.3%} "
          f"max {np.max([r[6] for r in w]):.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
