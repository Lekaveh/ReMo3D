"""Phase-2 benchmark: vmap-batched GPU solves vs single-solve throughput.

One tool -> one depth-relative grid (shared geometry, shared RHS): the batch
axis is the measurement depth, exactly the forward.py workload shape. Measures

  * JIT compile time (one-off) and steady-state batch wall time,
  * throughput in solves/second for several batch sizes,
  * fp64 vs fp32 accuracy drift (A6000 fp64 is 1/32 rate, fp32 is the speed
    path -- but correctness is checked against fp64),
  * optional NGSolve single-process baseline on the same (tool, depths) for a
    like-for-like CPU reference (--ngsolve-baseline N picks N depths).

Usage:
    python scripts/benchmark_gpu_batch.py --tool A2.0M0.5N --n-depths 251
    python scripts/benchmark_gpu_batch.py --batch-sizes 32 64 128 251 --dtype both
"""

from __future__ import annotations

import argparse
import csv
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from remo3d.gpu_solver import grid as ggrid  # noqa: E402
from remo3d.gpu_solver import operator_fv, operator_fem  # noqa: E402
from remo3d.gpu_solver import solve as gsolve  # noqa: E402
from remo3d.gpu_solver import tool as gtool  # noqa: E402
from remo3d.sensitivity import _load_formation, _load_borehole  # noqa: E402

BACKENDS = {"fv": operator_fv, "fem": operator_fem}
FORMATION_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Formation.txt"
BOREHOLE_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Borehole.txt"


def build_problem(tool_name, depths, formation, borehole, h_min, growth,
                  backend, dtype):
    """Shared grid/geometry + per-depth sigma stack and RHS."""
    cfg = gtool.tool_config(tool_name)
    if h_min is None:
        h_min = gtool.default_h_min(cfg)
    domain_radius = max(10.0 * cfg["span"], 5.0)
    elec_rel = [0.0, cfg["dz_M"], cfg["dz_N"]]
    fz = formation[:, 2]
    r_foci = [float(np.median(borehole[:, 1]))] + [float(v) for v in fz[~np.isnan(fz)]]
    r_nodes, z_rel = ggrid.build_grid(domain_radius, 0.0, elec_rel,
                                      r_foci=r_foci, h_min=h_min, growth=growth)
    geom = BACKENDS[backend].build_geometry(r_nodes, z_rel)
    j_C = ggrid.node_index(0.0, z_rel)
    j_M = ggrid.node_index(cfg["dz_M"], z_rel)
    j_N = ggrid.node_index(cfg["dz_N"], z_rel)
    rhs = operator_fv.point_source_rhs(
        (len(z_rel), len(r_nodes)), [((j_C, 0), 1.0)], geom["dirichlet"],
        dtype=dtype)

    sigmas = np.empty((len(depths), len(z_rel) - 1, len(r_nodes) - 1))
    for k, md in enumerate(depths):
        z_sim = md + cfg["depth_shift"]
        mud = float(np.interp(z_sim, borehole[:, 0], borehole[:, 2]))
        sigmas[k] = ggrid.sample_sigma(r_nodes, z_rel + z_sim, formation,
                                       borehole, mud)

    # Cast geometry + sigma to the requested dtype.
    geom = {k: (v.astype(dtype) if np.issubdtype(np.asarray(v).dtype, np.floating)
                else v) for k, v in geom.items()}
    sigmas = jnp.asarray(sigmas, dtype=dtype)
    return cfg, geom, rhs, sigmas, (j_M, j_N), (len(r_nodes), len(z_rel))


def run_batch(sigmas, rhs, geom, backend, tol, maxiter):
    """Batched solve with shared rhs; returns device array of solutions."""
    fn = lambda s: gsolve.solve_one(s, rhs, geom, backend=backend,
                                    tol=tol, maxiter=maxiter)
    return jax.vmap(fn)(sigmas)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", default="A2.0M0.5N")
    p.add_argument("--n-depths", type=int, default=251)
    p.add_argument("--depth-range", nargs=2, type=float, default=[0.0, 25.0])
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=[1, 8, 32, 64, 128, 251])
    p.add_argument("--backends", nargs="+", default=["fv"], choices=["fv", "fem"])
    p.add_argument("--dtype", choices=["f32", "f64", "both"], default="both")
    p.add_argument("--h-min", type=float, default=None,
                   help="Default: per-tool heuristic (tool.default_h_min).")
    p.add_argument("--growth", type=float, default=1.15)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=20000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--ngsolve-baseline", type=int, default=0,
                   help="Also time NGSolve single-process on N depths (0=skip).")
    p.add_argument("--output", type=Path,
                   default=ROOT / "benchmark_data" / "gpu_solver")
    args = p.parse_args(argv)

    formation = _load_formation([args.tool], str(FORMATION_FILE))
    borehole = _load_borehole([args.tool], str(BOREHOLE_FILE),
                              borehole_geometry_type="diameter")
    depths = np.linspace(args.depth_range[0], args.depth_range[1], args.n_depths)

    dtypes = {"f32": [jnp.float32], "f64": [jnp.float64],
              "both": [jnp.float64, jnp.float32]}[args.dtype]

    print(f"tool {args.tool}, {args.n_depths} depths, h_min={args.h_min}, "
          f"device={jax.devices()[0]}")
    rows = []
    ref64 = {}
    for backend in args.backends:
        for dtype in dtypes:
            dname = np.dtype(dtype).name
            cfg, geom, rhs, sigmas, (j_M, j_N), (nr, nz) = build_problem(
                args.tool, depths, formation, borehole, args.h_min,
                args.growth, backend, dtype)
            print(f"\n[{backend}/{dname}] grid {nr} x {nz} = {nr*nz} nodes")

            for bs in args.batch_sizes:
                bs = min(bs, len(depths))
                sig_b = sigmas[:bs]
                # compile
                t0 = time.perf_counter()
                u = run_batch(sig_b, rhs, geom, backend, args.tol, args.maxiter)
                u.block_until_ready()
                t_compile_total = time.perf_counter() - t0
                # steady state
                best = np.inf
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    u = run_batch(sig_b, rhs, geom, backend, args.tol,
                                  args.maxiter)
                    u.block_until_ready()
                    best = min(best, time.perf_counter() - t0)
                Ra = gtool.apparent_resistivity(np.asarray(u), cfg, j_M, j_N)
                key = (backend, bs)
                if dtype == jnp.float64:
                    ref64[key] = Ra
                drift = (np.max(np.abs(Ra - ref64[key]) / np.abs(ref64[key]))
                         if key in ref64 and dtype != jnp.float64 else 0.0)
                thr = bs / best
                rows.append({
                    "backend": backend, "dtype": dname, "batch": bs,
                    "nodes": nr * nz, "t_compile_s": round(t_compile_total - best, 3),
                    "t_batch_s": round(best, 4),
                    "solves_per_s": round(thr, 2),
                    "ms_per_solve": round(1000.0 * best / bs, 2),
                    "max_rel_vs_f64": f"{drift:.2e}",
                    "Ra_first": f"{Ra.flat[0]:.5f}",
                })
                print(f"  batch {bs:>4}: {best:>8.3f}s  "
                      f"{thr:>8.1f} solves/s  {1000*best/bs:>7.2f} ms/solve  "
                      f"drift {drift:.1e}")

    if args.ngsolve_baseline > 0:
        from validate_gpu_solver import ngsolve_reference  # noqa: E402
        nd = min(args.ngsolve_baseline, len(depths))
        pick = np.linspace(0, len(depths) - 1, nd).round().astype(int)
        t0 = time.perf_counter()
        for di in pick:
            ngsolve_reference(args.tool, float(depths[di]), formation, borehole)
        t_ng = time.perf_counter() - t0
        per = t_ng / nd
        print(f"\nNGSolve single-process: {per:.2f} s/solve "
              f"({nd} depths, {t_ng:.1f}s total) -> {1.0/per:.2f} solves/s/core")
        rows.append({"backend": "ngsolve", "dtype": "f64", "batch": nd,
                     "nodes": 0, "t_compile_s": 0.0, "t_batch_s": round(t_ng, 3),
                     "solves_per_s": round(nd / t_ng, 2),
                     "ms_per_solve": round(1000.0 * per, 1),
                     "max_rel_vs_f64": "", "Ra_first": ""})

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "gpu_batch_benchmark.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
