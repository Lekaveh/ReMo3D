"""Pivot the axis-study variants into three 2-D tables: SEC / batch / condense.

The axis study (``benchmark_optimizations.py --variants axes``) crosses three
ablation axes with the solver {CG, direct}:

  * SEC       — single-electrode-computation mode {on, off}
  * batch     — depths per mesh/assembly {1, 5, 10}
  * condense  — static condensation {True, False}

All three share the base cell (SEC on, batch 5, condense True), so the 14 table
cells come from only 10 distinct runs. This script reads the per-variant
``full_pipeline/*.npz`` (same source as analyze_benchmark.py), computes wall
time / speedup / accuracy vs the V1 baseline, and lays each axis out as a table
with the {CG, direct} columns side by side. Output: ``summary_axes.md``.

Usage:
    python scripts/summarize_axes.py --results benchmark_data/optim_bench
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BASELINE = "V1_baseline"

# name grid: (row label) -> {"cg": variant, "dir": variant}, per axis. The base
# cell (sec1/b5/cndT) is reused across all three axes.
BASE_CG, BASE_DIR = "Ax_sec1_b5_cndT_cg", "Ax_sec1_b5_cndT_dir"
AXES = {
    "SEC (single-electrode computation) — batch 5, condense on": [
        ("SEC on  (reuse 1 solve / N tools)", "Ax_sec1_b5_cndT_cg", "Ax_sec1_b5_cndT_dir"),
        ("SEC off (1 solve per tool)",        "Ax_sec0_b5_cndT_cg", "Ax_sec0_b5_cndT_dir"),
    ],
    "Batch size (depths per mesh/assembly) — SEC on, condense on": [
        ("batch 1  (exact, no geom shift)", "Ax_sec1_b1_cndT_cg",  "Ax_sec1_b1_cndT_dir"),
        ("batch 5  (default)",              "Ax_sec1_b5_cndT_cg",  "Ax_sec1_b5_cndT_dir"),
        ("batch 10 (larger shift)", "Ax_sec1_b10_cndT_cg", "Ax_sec1_b10_cndT_dir"),
        ("batch 15 (largest shift; ~19 tasks underfill 24 workers)", None, "Ax_sec1_b15_cndT_dir"),
    ],
    "Static condensation — SEC on, batch 5": [
        ("condense on  (Schur elim.)", "Ax_sec1_b5_cndT_cg", "Ax_sec1_b5_cndT_dir"),
        ("condense off (full system)", "Ax_sec1_b5_cndF_cg", "Ax_sec1_b5_cndF_dir"),
    ],
    "Worker thread pinning (fixes direct-solver TaskManager oversubscription) — SEC on, batch 5, condense on": [
        # Unpinned references: the CG base cell ran unpinned (CG is single-threaded
        # per worker, pinning-neutral); Vd_direct_forced is the same config as the
        # direct base cell, run unpinned in the 2026-07-16 study.
        ("unpinned workers (Vd_direct_forced = same config)", "Ax_sec1_b5_cndT_cg", "Vd_direct_forced"),
        ("pinned workers (1 thread/worker fix)",              "Ax_pin_cg",          "Ax_sec1_b5_cndT_dir"),
    ],
}


def _rel_err(cand, base):
    return np.abs(cand - base) / np.maximum(np.abs(base), 1e-30)


def load_npzs(results_dir):
    out = {}
    for path in sorted(glob.glob(str(Path(results_dir) / "full_pipeline" / "*.npz"))):
        d = np.load(path, allow_pickle=True)
        out[str(d["variant"])] = {
            "logs": d["logs"], "wall_time": d["wall_time"], "nan_count": d["nan_count"],
            "flags": json.loads(str(d["flags"])),
        }
    return out


def stats(data, name, base_wall, base_logs):
    """Return (mean_wall, speedup, max_rel, mean_rel, nan) or None if absent."""
    if name not in data:
        return None
    d = data[name]
    mean_wall = float(np.nanmean(d["wall_time"]))
    speedup = (base_wall / mean_wall) if base_wall else float("nan")
    nan = int(np.sum(d["nan_count"]))
    if base_logs is not None and base_logs.shape == d["logs"].shape:
        finite = np.isfinite(base_logs) & np.isfinite(d["logs"])
        rel = _rel_err(d["logs"][finite], base_logs[finite]) if np.any(finite) else np.array([np.nan])
        max_rel, mean_rel = float(np.max(rel)), float(np.mean(rel))
    else:
        max_rel = mean_rel = float("nan")
    return mean_wall, speedup, max_rel, mean_rel, nan


def cell(st):
    if st is None:
        return "—"
    mean_wall, speedup, max_rel, _, nan = st
    tag = f" ⚠{nan}nan" if nan else ""
    return f"{mean_wall:.2f}s · {speedup:.2f}× · err {max_rel:.1e}{tag}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("benchmark_data/optim_bench"))
    args = p.parse_args(argv)

    data = load_npzs(args.results)
    if not data:
        raise SystemExit(f"No variant .npz under {args.results}/full_pipeline/")
    base_wall = float(np.nanmean(data[BASELINE]["wall_time"])) if BASELINE in data else None
    base_logs = data[BASELINE]["logs"] if BASELINE in data else None
    if base_wall is None:
        print(f"[warn] {BASELINE}.npz absent — speedup/accuracy columns will be blank")

    n = len(data.get(BASE_CG, data[next(iter(data))])["wall_time"])
    md = ["# ReMo3D axis study — SEC / batch / condense × {CG, direct}", "",
          f"Cell = **mean s/sample · speedup vs V1 · max rel-err vs V1**. "
          f"Baseline = `{BASELINE}` (original `main`). {n} samples, 5 tools, gmsh, domain 40 m. "
          f"Base cell (SEC on · batch 5 · condense on) is shared across all three tables.", "",
          "**Threading note:** all `direct` cells were re-run with **thread-pinned workers** "
          "(2026-07-17 fix: `REMO3D_WORKER_THREADS=1` env caps + `ngs.SetNumThreads`) after the "
          "unpinned direct numbers were shown to suffer NGSolve-TaskManager oversubscription. "
          "CG cells are unpinned but pinning-neutral (CG runs 1 thread/worker regardless). "
          "The last table quantifies the pinning effect itself.", ""]

    for title, rows in AXES.items():
        md += [f"## {title}", "",
               "| level | CG (multigrid) | direct (sparse-Cholesky) |",
               "|---|---|---|"]
        for label, cg_name, dir_name in rows:
            cg = cell(stats(data, cg_name, base_wall, base_logs))
            dr = cell(stats(data, dir_name, base_wall, base_logs))
            md.append(f"| {label} | {cg} | {dr} |")
        md.append("")

    out = Path(args.results) / "summary_axes.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[summarize_axes] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
