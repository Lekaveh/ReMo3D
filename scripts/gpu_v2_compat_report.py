# -*- coding: utf-8 -*-
"""Final CPU-compat report: compat vs stored refs vs fresh unbatched NGSolve.

Pulls together:
  * benchmark_data/gpu_solver/global_optim_bench_cpuconv.npz (+ _s50) —
    the 100-sample compat-convention run;
  * benchmark_data/gpu_solver/global_optim_bench.npz (+ _s50) — native v2;
  * benchmark_data/optim_bench/full_pipeline/Vd_direct_forced.npz — stored
    pipeline logs (batch_size=5!);
  * benchmark_data/gpu_solver/ngsolve_protocol.npz — fresh UNBATCHED
    NGSolve, samples 0 & 56, every 4th depth (the clean CPU convention).

Usage: python scripts/gpu_v2_compat_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "benchmark_data" / "gpu_solver"
TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]


def load_pair(base, tag):
    a = np.load(G / f"{base}{tag}.npz", allow_pickle=True)
    b = np.load(G / f"{base}_s50{tag}.npz", allow_pickle=True)
    return np.concatenate([np.asarray(a["logs"]), np.asarray(b["logs"])])


def main():
    compat = load_pair("global_optim_bench", "_cpuconv")
    native = load_pair("global_optim_bench", "")
    vd = np.asarray(np.load(ROOT / "benchmark_data" / "optim_bench"
                            / "full_pipeline" / "Vd_direct_forced.npz",
                            allow_pickle=True)["logs"], float)

    print("== vs STORED pipeline logs (batch_size=5!) — 100 samples ==")
    for name, L in (("compat", compat), ("native", native)):
        rel = np.abs(L - vd) / np.abs(vd)
        per = "  ".join(f"{t}: max {np.nanmax(rel[:, ti]):.2%} "
                        f"mean {np.nanmean(rel[:, ti]):.3%}"
                        for ti, t in enumerate(TOOLS))
        print(f"  {name:6s}: {per}")

    ng = np.load(G / "ngsolve_protocol.npz", allow_pickle=True)
    idx = np.asarray(ng["depth_idx"], int)
    print("\n== vs FRESH UNBATCHED NGSolve (clean CPU convention; "
          f"samples {list(np.asarray(ng['samples']))}, {len(idx)} depths) ==")
    for name, L in (("compat", compat), ("native", native)):
        worst, mean_acc = 0.0, []
        per_tool = {}
        for si in np.asarray(ng["samples"], int):
            for ti, t in enumerate(TOOLS):
                ref = np.asarray(ng[f"s{si}_{t}"], float)
                rel = np.abs(L[si, ti, idx] - ref) / np.abs(ref)
                per_tool.setdefault(t, []).append(rel)
                worst = max(worst, float(np.nanmax(rel)))
                mean_acc.append(rel)
        per = "  ".join(f"{t}: max {np.nanmax(np.concatenate(v)):.2%}"
                        for t, v in per_tool.items())
        print(f"  {name:6s}: worst {worst:.3%} "
              f"mean {np.nanmean(np.concatenate(mean_acc)):.3%} | {per}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
