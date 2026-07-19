# -*- coding: utf-8 -*-
"""Arbitrate ALL big v2-vs-stored discrepancies of the 1000-sample run.

Point selection (from full_v2_native_n1000_seed42.npz, rel = |v2-ref|/|ref|):
  * every non-A8.0 point with rel > 10%          (exhaustive, ~195);
  * every A8.0 point with rel > 25%              (exhaustive, ~190);
  * a seeded stratified A8.0 sample from the 10-25% tail (3 bins x 15).

Each point is recomputed with fresh UNBATCHED single-process NGSolve in the
stored refs' own convention (scalar mud, R=40) and at a converged control
radius (R=160 short/mid tools, R=320 for A8.0), then classified:

  batch5   — fresh R=40 agrees with v2/control but NOT with stored: the
             stored value is a batch_size=5 artifact;
  R40      — fresh R=40 reproduces stored, control moves to v2: for A8.0
             this is the boundary-truncation convention, for short tools an
             R=40 gmsh mesh defect (p-refinement-stable, see
             gpu_v2_worst_point_arbitration.py probe);
  v2_err   — fresh R=40 = control = stored, v2 stands alone: a GENUINE v2
             error;
  other    — anything else (listed individually).

Runs on a process pool (single-threaded workers). Usage:
    python scripts/gpu_v2_full_arbitration_all.py [--workers 16] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "benchmark_data" / "full"
RUN = ROOT / "benchmark_data" / "gpu_solver" / "full_v2_native_n1000_seed42.npz"
OUT = ROOT / "benchmark_data" / "gpu_solver" / "full_arbitration_all.npz"

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
AGREE = 0.05                       # 5% relative = "same value"


def _init_worker():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    for p in (str(ROOT / "scripts"), str(ROOT / "remo3d"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def solve_point(task):
    """(file, tool, z, R) -> fresh unbatched NGSolve Ra (refs' convention)."""
    f, tool, z, R = task
    _init_worker()
    import ngsolve as ngs
    import gmsh_functions as gmf
    import ngsolve_functions as ngsf
    from remo3d.remo3d import Model
    ngs.ngsglobals.msg_level = 0
    os.makedirs(gmf.TMP_DIR, exist_ok=True)

    d = np.load(FULL / f, allow_pickle=True)
    fm = np.asarray(d["formation_model"], float)
    bh = np.asarray(d["borehole_model"], float)

    helper = Model([tool])
    tp = helper.tools[tool]
    tool_geometry = tp[0, :3].astype(float)
    source_terms = tp[1, :3].astype(float)
    K = float(tp[0, 3])
    z_sim = float(z) + float(tp[1, 3])
    mud = float(np.interp(z_sim, bh[:, 0], bh[:, 2]))
    lf, lb, sigma = gmf.SelectGmshDataRange(
        bh[:, :2], fm, 0.0, mud, z_sim, R)
    mesh = ngs.Mesh(gmf.ConstructGmsh2dModel(
        R, tool_geometry, source_terms, lf, lb, os.getpid()))
    fes, gfu = ngsf.SolveBVP(
        mesh, ngs.CoefficientFunction(sigma), tool_geometry, source_terms,
        "dirichlet_boundary", "multigrid", True, order=3,
        symmetric=True, direct_solver=True)
    measuring = tool_geometry[source_terms == 0.0]
    return abs(K * (gfu(mesh(0.0, float(measuring[1])))
                    - gfu(mesh(0.0, float(measuring[0])))))


def select_points():
    d = np.load(RUN, allow_pickle=True)
    logs, refs = np.asarray(d["logs"]), np.asarray(d["refs"])
    files, depths = list(d["files"]), np.asarray(d["depths"], float)
    rel = np.abs(logs - refs) / np.abs(refs)

    pts = []   # (si, ti, di, kind)
    for si, ti, di in zip(*np.where(rel[:, :4] > 0.10)):
        pts.append((si, ti, di, "non8"))
    for si, di in zip(*np.where(rel[:, 4] > 0.25)):
        pts.append((si, 4, di, "a8_tail"))
    rng = np.random.default_rng(7)
    a8 = rel[:, 4]
    for lo, hi in ((0.10, 0.15), (0.15, 0.20), (0.20, 0.25)):
        cand = np.argwhere((a8 > lo) & (a8 <= hi))
        pick = cand[rng.choice(len(cand), min(15, len(cand)), replace=False)]
        for si, di in pick:
            pts.append((int(si), 4, int(di), f"a8_{lo:.2f}"))
    meta = [dict(file=str(files[si]), tool=TOOLS[ti], z=float(depths[di]),
                 stored=float(refs[si, ti, di]), v2=float(logs[si, ti, di]),
                 rel=float(rel[si, ti, di]), kind=kind)
            for si, ti, di, kind in pts]
    return meta


def classify(m):
    ag = lambda a, b: abs(a - b) / max(abs(b), 1e-30) < AGREE
    f40, fhi = m["fresh40"], m["freshhi"]
    if ag(f40, m["stored"]) and ag(fhi, m["v2"]) and not ag(f40, fhi):
        return "conv_R40" if m["tool"] == "A8.0M1.0N" else "mesh_R40"
    if not ag(f40, m["stored"]) and ag(m["v2"], fhi):
        return "batch5"
    if ag(f40, m["stored"]) and ag(f40, fhi) and not ag(m["v2"], fhi):
        return "v2_err"
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only first N points")
    args = ap.parse_args(argv)

    meta = select_points()
    if args.limit:
        meta = meta[:args.limit]
    kinds = {}
    for m in meta:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(f"{len(meta)} points to arbitrate: {kinds}", flush=True)

    tasks, owners = [], []
    for i, m in enumerate(meta):
        hi = 320.0 if m["tool"] == "A8.0M1.0N" else 160.0
        tasks += [(m["file"], m["tool"], m["z"], 40.0),
                  (m["file"], m["tool"], m["z"], hi)]
        owners += [(i, "fresh40"), (i, "freshhi")]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as ex:
        done = 0
        for (i, slot), ra in zip(owners, ex.map(solve_point, tasks,
                                                chunksize=4)):
            meta[i][slot] = float(ra)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} solves, "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"all {len(tasks)} solves in {time.perf_counter() - t0:.0f}s",
          flush=True)

    for m in meta:
        m["class"] = classify(m)
        m["v2_vs_conv"] = (abs(m["v2"] - m["freshhi"])
                           / max(abs(m["freshhi"]), 1e-30))
        m["stored_vs_conv"] = (abs(m["stored"] - m["freshhi"])
                               / max(abs(m["freshhi"]), 1e-30))

    np.savez_compressed(OUT, meta=np.array(meta, dtype=object),
                        agree=AGREE)
    print(f"saved {OUT}\n")

    print("== classification x tool ==")
    classes = ("conv_R40", "mesh_R40", "batch5", "v2_err", "other")
    print(f"{'tool':11s} " + "".join(f"{c:>9s}" for c in classes) + "    total")
    for t in TOOLS:
        row = [m for m in meta if m["tool"] == t]
        if not row:
            continue
        cnt = {c: sum(1 for m in row if m["class"] == c) for c in classes}
        print(f"{t:11s} " + "".join(f"{cnt[c]:9d}" for c in classes)
              + f" {len(row):8d}")

    v2c = np.array([m["v2_vs_conv"] for m in meta])
    stc = np.array([m["stored_vs_conv"] for m in meta])
    print(f"\nacross ALL {len(meta)} arbitrated points "
          f"(vs converged fresh NGSolve):")
    print(f"  v2     : mean {v2c.mean():.2%}  median {np.median(v2c):.2%}  "
          f"max {v2c.max():.2%}")
    print(f"  stored : mean {stc.mean():.2%}  median {np.median(stc):.2%}  "
          f"max {stc.max():.2%}")

    for cls in ("v2_err", "other"):
        rows = [m for m in meta if m["class"] == cls]
        if rows:
            print(f"\n== {cls} points ({len(rows)}) ==")
            for m in sorted(rows, key=lambda m: -m["v2_vs_conv"]):
                print(f"  {m['file']} {m['tool']} z={m['z']:.1f}: "
                      f"stored {m['stored']:.4f} v2 {m['v2']:.4f} "
                      f"fresh40 {m['fresh40']:.4f} "
                      f"freshhi {m['freshhi']:.4f} "
                      f"[v2 vs conv {m['v2_vs_conv']:.1%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
