"""Validation step 3: full Ex1 log via the GPU driver vs Results_1.txt.

Reproduces the scripts/forward.py workload (8 tools, 0-25 m @ 0.1 m,
invasion zones blanked exactly like forward.py does) on the batched GPU
solver, then diffs against the frozen NGSolve validation log
notebooks/validation/Results_1.txt.

The 1e-4 gate in forward.py is an NGSolve-vs-NGSolve regression bound; two
different discretizations cannot meet it. The physical acceptance target here
is max relative difference per tool below ~1% (see the plan). Reports per-tool
max/mean rel diff and total wall time.

Usage:
    python scripts/gpu_full_ex1.py [--backend fv] [--dtype f64] [--batch-size 64]
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
REMO3D_DIR = ROOT / "remo3d"
if str(REMO3D_DIR) not in sys.path:
    sys.path.append(str(REMO3D_DIR))

import jax.numpy as jnp  # noqa: E402

from remo3d.gpu_solver.driver import compute_logs_gpu  # noqa: E402

TOOLS = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N",
         "N0.5M2.0A", "M4.0A0.5B", "N2.0M0.5A", "N11.0M0.5A"]
VALIDATION_FILE = ROOT / "notebooks" / "validation" / "Results_1.txt"
FORMATION_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Formation.txt"
BOREHOLE_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Borehole.txt"


def load_validation(path):
    """Results_1.txt: header row, units row, then DEPTH + one column per tool."""
    with open(path) as fh:
        header = fh.readline().split()
    data = np.loadtxt(path, skiprows=2)
    cols = {name: data[:, k] for k, name in enumerate(header)}
    return cols


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["fv", "fem"], default="fv")
    p.add_argument("--dtype", choices=["f32", "f64"], default="f64")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--tools", nargs="+", default=TOOLS)
    p.add_argument("--shared", action="store_true",
                   help="Solve all tools on one shared MP-centred grid "
                        "(single nested-vmap batch).")
    args = p.parse_args(argv)

    # forward.py blanks the invasion zones (formation[:, 2:4] = nan)
    formation = np.loadtxt(FORMATION_FILE, skiprows=2)
    formation[:, 2:4] = np.nan
    depths = np.arange(0, 25.1, 0.1)

    dtype = jnp.float64 if args.dtype == "f64" else jnp.float32
    t0 = time.perf_counter()
    logs = compute_logs_gpu(
        args.tools, depths, formation, str(BOREHOLE_FILE),
        borehole_geometry_type="diameter", backend=args.backend,
        dtype=dtype, tol=args.tol, batch_size=args.batch_size,
        shared_grid=args.shared, verbose=True)
    wall = time.perf_counter() - t0
    n_solves = len(args.tools) * len(depths)
    print(f"\nGPU total: {wall:.1f}s for {n_solves} solves "
          f"({1000.0 * wall / n_solves:.1f} ms/solve, "
          f"{n_solves / wall:.1f} solves/s)")

    val = load_validation(VALIDATION_FILE)
    v_depth = val["DEPTH"]
    print(f"\n{'tool':>12} {'max_rel':>9} {'mean_rel':>9} {'max_abs':>9}  worst_depth")
    overall = 0.0
    for tool in args.tools:
        if tool not in val:
            print(f"{tool:>12}   (not in validation file)")
            continue
        ref = val[tool]
        got = np.interp(v_depth, logs[tool][:, 0], logs[tool][:, 1])
        ok = np.isfinite(ref) & np.isfinite(got)
        rel = np.abs(got[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-30)
        k = int(np.argmax(rel))
        overall = max(overall, float(rel.max()))
        print(f"{tool:>12} {rel.max():>9.2e} {rel.mean():>9.2e} "
              f"{np.abs(got[ok]-ref[ok]).max():>9.4f}  {v_depth[ok][k]:.1f} m")
    print(f"\noverall max rel diff vs Results_1.txt: {overall:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
