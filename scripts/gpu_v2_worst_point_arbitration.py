# -*- coding: utf-8 -*-
"""Arbitrate the worst full/-dataset points: fresh NGSolve vs stored refs vs v2.

The 1000-sample native-v2 run (full_v2_native_n1000_seed42.npz) has isolated
huge relative diffs vs the stored batch5/R=40 NGSolve logs (max 158% at
A2.0M0.5N, sample_331 z=33.2). This script recomputes those points with a
fresh UNBATCHED single-process NGSolve in the refs' own convention (scalar
mud, per-depth z-window, R=40) and at R=160 (boundary control), to attribute
each discrepancy to (a) the stored refs' batch_size=5 artifact, (b) boundary
convention, or (c) v2 discretization error.

Run under plain python in the remo3d env (single-process, no MPI spawn).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "scripts"), str(ROOT / "remo3d"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FULL = ROOT / "benchmark_data" / "full"
OUT = ROOT / "benchmark_data" / "gpu_solver" / "full_worst_arbitration.npz"

# (sample file, tool, depths, radii) — the worst offenders of the 1000-run
CASES = [
    ("large_noise/3/data/sample_331.npz", "A2.0M0.5N",
     [32.6, 32.8, 33.0, 33.2, 33.4], [40.0, 160.0]),
    ("large_noise/3/data/sample_331.npz", "A0.4M0.1N",
     [31.0, 31.2, 31.4, 31.6], [40.0, 160.0]),
    ("smooth_noise/6/data/sample_359.npz", "A8.0M1.0N",
     [39.4, 39.8, 40.2], [40.0, 160.0]),
]


def fresh_log(tool_name, depths, formation, borehole, domain_radius,
              fe_order=3):
    """compare_len512.ngsolve_direct_log with an explicit domain radius."""
    import ngsolve as ngs
    import gmsh_functions as gmf
    import ngsolve_functions as ngsf
    from remo3d.remo3d import Model
    ngs.ngsglobals.msg_level = 0
    os.makedirs(gmf.TMP_DIR, exist_ok=True)

    helper = Model([tool_name])
    tp = helper.tools[tool_name]
    tool_geometry = tp[0, :3].astype(float)
    source_terms = tp[1, :3].astype(float)
    K = float(tp[0, 3])
    depth_shift = float(tp[1, 3])

    Ra = np.full(len(depths), np.nan)
    for k, md in enumerate(depths):
        z_sim = float(md) + depth_shift
        mud = float(np.interp(z_sim, borehole[:, 0], borehole[:, 2]))
        local_formation, local_borehole, sigma = gmf.SelectGmshDataRange(
            borehole[:, :2], formation, 0.0, mud, z_sim, domain_radius)
        mesh = ngs.Mesh(gmf.ConstructGmsh2dModel(
            domain_radius, tool_geometry, source_terms,
            local_formation, local_borehole, 0))
        sigma_cf = ngs.CoefficientFunction(sigma)
        fes, gfu = ngsf.SolveBVP(
            mesh, sigma_cf, tool_geometry, source_terms,
            "dirichlet_boundary", "multigrid", True, order=fe_order,
            symmetric=True, direct_solver=True)
        measuring = tool_geometry[source_terms == 0.0]
        Ra[k] = abs(K * (gfu(mesh(0.0, float(measuring[1])))
                         - gfu(mesh(0.0, float(measuring[0])))))
    return Ra


def main():
    v2 = np.load(ROOT / "benchmark_data" / "gpu_solver"
                 / "full_v2_native_n1000_seed42.npz", allow_pickle=True)
    v2_files = list(v2["files"])
    v2_depths = np.asarray(v2["depths"], float)
    tools = list(v2["tools"])

    results = {}
    for f, tool, zs, radii in CASES:
        d = np.load(FULL / f, allow_pickle=True)
        fm = np.asarray(d["formation_model"], float)
        bh = np.asarray(d["borehole_model"], float)
        ml = d["model_logs"].item()
        si, ti = v2_files.index(f), tools.index(tool)
        di = np.searchsorted(v2_depths, zs)

        fresh = {}
        for R in radii:
            t0 = time.perf_counter()
            fresh[R] = fresh_log(tool, zs, fm, bh, R)
            print(f"{f} {tool} R={R:.0f}: {len(zs)} solves "
                  f"in {time.perf_counter() - t0:.0f}s", flush=True)

        stored = np.asarray(ml[tool], float)[di, 1]
        v2n = np.asarray(v2["logs"])[si, ti, di]
        key = f"{Path(f).parent.parent.name}_{Path(f).stem}_{tool}"
        results[key] = np.stack([np.asarray(zs), stored, v2n]
                                + [fresh[R] for R in radii])

        print(f"\n== {f} {tool} ==")
        hdr = "    z     stored     v2  " + "".join(
            f"  fresh R={R:<5.0f}" for R in radii)
        print(hdr)
        for j, z in enumerate(zs):
            row = (f"  {z:5.1f} {stored[j]:8.3f} {v2n[j]:8.3f}"
                   + "".join(f"     {fresh[R][j]:8.3f}" for R in radii))
            print(row, flush=True)
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, cases=[c[0] + "|" + c[1] for c in CASES],
                        **results)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
