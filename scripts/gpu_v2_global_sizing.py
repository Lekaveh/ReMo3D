# -*- coding: utf-8 -*-
"""E0 + E3 (GPU_SOLVER_V2_PLAN Phase 0): global-operator sizing & RHS audit.

Builds the global factor-once/solve-many operator for the len512 workload —
per-tool grids (variant a) and the shared all-tools grid (variant b) — and
reports grid/operator sizes, band-factor memory, and the deduplicated RHS
counts (source form vs reciprocal form). No factorization here; timing is
scripts/gpu_v2_amortize_cpu.py (E1).

Usage: python scripts/gpu_v2_global_sizing.py [--sample 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remo3d.gpu_solver import global_op  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"


def describe(name, tools, depths, formation, borehole):
    t0 = time.perf_counter()
    p = global_op.build_global_problem(tools, depths, formation, borehole)
    dt = time.perf_counter() - t0
    nr, nz = len(p["r_nodes"]), len(p["z_nodes"])
    band_gb = (p["bandwidth"] + 1) * p["n_free"] * 8 / 1e9
    print(f"{name:26s} h={p['h_min']:.4g} R={p['domain_radius']:5.1f} "
          f"grid {nr}x{nz} n_free={p['n_free']:9,d} nnz={p['A'].nnz:10,d} "
          f"band={p['bandwidth']:3d} factor~{band_gb:5.2f} GB "
          f"tasks={p['n_tasks']:4d} uniq_src={len(p['uniq_src']):4d} "
          f"recip={p['n_src_recip']:4d} build={dt:5.1f}s")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args(argv)

    d = np.load(SAMPLES_DIR / f"sample_{args.sample}.npz", allow_pickle=True)
    formation = np.asarray(d["formation_model"], float)
    borehole = np.asarray(d["borehole_model"], float)
    depths = borehole[:, 0].astype(float)

    print(f"len512 sample {args.sample}: {len(depths)} depths "
          f"[{depths[0]}, {depths[-1]}] step {depths[1] - depths[0]:.3g}, "
          f"{len(TOOLS)} tools\n")

    print("-- variant a: one global grid per tool "
          "(<=5 factorizations/sample) --")
    tot = 0
    for t in TOOLS:
        p = describe(t, [t], depths, formation, borehole)
        tot += p["n_free"]
    print(f"{'sum over tools':26s} n_free={tot:9,d}\n")

    print("-- variant b: one shared grid for all tools "
          "(1 factorization/sample) --")
    p = describe("shared[5 tools]", TOOLS, depths, formation, borehole)

    n1280 = p["n_tasks"]
    print(f"\nE3 audit (shared grid): {n1280} (tool,depth) tasks -> "
          f"{len(p['uniq_src'])} unique source columns "
          f"(x{n1280 / len(p['uniq_src']):.2f} dedup); "
          f"reciprocal form would need {p['n_src_recip']} columns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
