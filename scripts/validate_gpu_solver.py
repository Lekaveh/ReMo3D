"""Golden-reference validation of the structured-grid GPU solver.

Single-process (no MPI, like benchmark_solver_phases.py): for each (tool,
depth) it solves the same physical problem twice —

  * reference: NGSolve/Netgen body-fitted FEM via ngsolve_functions.SolveBVP
    (order 3, symmetric+condense+direct-auto = the V5 optimized variant);
  * candidate: remo3d.gpu_solver on a graded structured (r, z) grid, with
    both discretization backends ("fv" and "fem");

and reports apparent resistivities and relative differences.

Modes:
    homo  — homogeneous medium (analytic target R_a = rho), grid-refinement
            sweep for both backends; no NGSolve needed.
    ex1   — the canonical Ex1 formation/borehole (invasion zones included),
            a set of tools x depths vs the NGSolve reference.

Usage:
    python scripts/validate_gpu_solver.py --mode homo
    python scripts/validate_gpu_solver.py --mode ex1 --tools A2.0M0.5N N2.0M0.5A
    python scripts/validate_gpu_solver.py --mode ex1 --h-min 0.005

Output: printed table + <output>/gpu_validation_<mode>.csv
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
# Appended LAST so the remo3d/ package dir wins for `import remo3d`, while
# top-level `import netgen_functions` (NGSolve reference) still resolves.
REMO3D_DIR = ROOT / "remo3d"
if str(REMO3D_DIR) not in sys.path:
    sys.path.append(str(REMO3D_DIR))

from remo3d.gpu_solver import grid as ggrid          # noqa: E402
from remo3d.gpu_solver import operator_fv, operator_fem, solve as gsolve, tool as gtool  # noqa: E402
from remo3d.sensitivity import _load_formation, _load_borehole  # noqa: E402
from remo3d.remo3d import Model                       # noqa: E402

BACKENDS = {"fv": operator_fv, "fem": operator_fem}

FORMATION_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Formation.txt"
BOREHOLE_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Borehole.txt"
DEFAULT_TOOLS = ["M1.0A0.1B", "A2.0M0.5N", "N2.0M0.5A", "B5.7A0.4M"]
DEFAULT_DEPTHS = [2.0, 5.0, 9.0, 11.5, 15.0, 20.0, 23.0]


# ---------------------------------------------------------------------------
# GPU-solver single solve
# ---------------------------------------------------------------------------

def gpu_solve_one(tool_name, measurement_depth, formation, borehole,
                  backend="fv", h_min=None, growth=1.15, domain_radius=None,
                  tol=1e-10, cache={}):
    """Solve one (tool, depth) on the structured grid; returns (Ra, wall_s, n_nodes).

    The grid, geometry factors and electrode indices depend only on
    (tool, backend, h_min, domain_radius) — NOT on the measurement depth,
    because the domain is centred on the current electrode (grid nodes are
    depth-relative). They are cached across depths; sigma is resampled per
    depth. This mirrors how the batched path will work (one grid per tool).
    """
    cfg = gtool.tool_config(tool_name)
    if h_min is None:
        h_min = gtool.default_h_min(cfg)
    if domain_radius is None:
        domain_radius = max(10.0 * cfg["span"], 5.0)

    key = (tool_name, backend, h_min, growth, float(domain_radius))
    if key not in cache:
        # Depth-relative grid centred on the current electrode at z=0.
        elec_rel = [0.0, cfg["dz_M"], cfg["dz_N"]]
        r_foci = [float(np.median(borehole[:, 1]))]
        fz = formation[:, 2]
        r_foci += [float(v) for v in fz[~np.isnan(fz)]]
        r_nodes, z_rel = ggrid.build_grid(domain_radius, 0.0, elec_rel,
                                          r_foci=r_foci, h_min=h_min,
                                          growth=growth)
        geom = BACKENDS[backend].build_geometry(r_nodes, z_rel)
        j_C = ggrid.node_index(0.0, z_rel)
        j_M = ggrid.node_index(cfg["dz_M"], z_rel)
        j_N = ggrid.node_index(cfg["dz_N"], z_rel)
        rhs = operator_fv.point_source_rhs(
            (len(z_rel), len(r_nodes)), [((j_C, 0), 1.0)], geom["dirichlet"])
        cache[key] = (r_nodes, z_rel, geom, rhs, j_M, j_N)
    r_nodes, z_rel, geom, rhs, j_M, j_N = cache[key]

    z_sim = measurement_depth + cfg["depth_shift"]
    mud_res = float(np.interp(z_sim, borehole[:, 0], borehole[:, 2]))
    sigma = ggrid.sample_sigma_aniso(r_nodes, z_rel + z_sim, formation,
                                     borehole, mud_res)

    t0 = time.perf_counter()
    u = gsolve.solve_one(sigma, rhs, geom, backend=backend, tol=tol)
    u.block_until_ready()
    wall = time.perf_counter() - t0
    Ra = float(gtool.apparent_resistivity(np.asarray(u), cfg, j_M, j_N))
    return Ra, wall, len(r_nodes) * len(z_rel)


# ---------------------------------------------------------------------------
# NGSolve reference solve (single-process, benchmark_solver_phases pattern)
# ---------------------------------------------------------------------------

def ngsolve_reference(tool_name, measurement_depth, formation, borehole,
                      fe_order=3, preconditioner="multigrid", domain_radius=None):
    """Reference R_a from the NGSolve pipeline for one (tool, depth)."""
    import ngsolve as ngs
    import netgen_functions as ngf
    import ngsolve_functions as ngsf
    ngs.ngsglobals.msg_level = 0

    helper = Model([tool_name])
    tp = helper.tools[tool_name]
    tool_geometry = tp[0, :3].astype(float)      # relative to current electrode
    source_terms = tp[1, :3].astype(float)
    K = float(tp[0, 3])
    depth_shift = float(tp[1, 3])

    if domain_radius is None:
        domain_radius = max(10.0 * float(np.max(np.abs(tool_geometry))), 5.0)

    z_sim = measurement_depth + depth_shift
    mud_res = float(np.interp(z_sim, borehole[:, 0], borehole[:, 2]))

    t0 = time.perf_counter()
    local_formation, local_borehole, sigma = ngf.SelectNetgenDataRange(
        borehole[:, :2], formation, mud_res, z_sim, domain_radius)
    mesh_data = ngf.ConstructNetgen2dModel(
        domain_radius, tool_geometry, source_terms, local_formation, local_borehole)
    mesh = ngs.Mesh(mesh_data)
    sigma_cf = ngs.CoefficientFunction(sigma)
    fes, gfu, m = ngsf.SolveBVP(
        mesh, sigma_cf, tool_geometry, source_terms, [2],
        preconditioner, True, order=fe_order, symmetric=True,
        direct_solver="auto", return_metrics=True)
    wall = time.perf_counter() - t0

    measuring = tool_geometry[source_terms == 0.0]
    if measuring.shape[0] == 2:
        Ra = abs(K * (gfu(mesh(0.0, float(measuring[1])))
                      - gfu(mesh(0.0, float(measuring[0])))))
    else:
        Ra = abs(K * gfu(mesh(0.0, float(measuring[0]))))
    return float(Ra), wall, int(m.get("dofs_total") or 0)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_homo(args):
    rho = args.rho
    formation = np.array([[-1e4, 1e4, np.nan, np.nan, rho]])
    borehole = np.array([[-1e4, 1e-6, rho], [1e4, 1e-6, rho]])  # vanishing borehole
    rows = []
    print(f"\n=== homogeneous medium, rho = {rho} Ohm.m (target R_a = rho) ===")
    print(f"{'tool':>12} {'h_min':>7} {'backend':>7} {'nodes':>8} "
          f"{'R_a':>9} {'rel_err':>9} {'t_solve':>8}")
    for tool_name in args.tools:
        for h_min in args.h_sweep:
            for backend in args.backends:
                Ra, wall, n = gpu_solve_one(
                    tool_name, 10.0, formation, borehole, backend=backend,
                    h_min=h_min, tol=args.tol)
                rel = abs(Ra - rho) / rho
                rows.append({"mode": "homo", "tool": tool_name, "depth": 10.0,
                             "backend": backend, "h_min": h_min, "nodes": n,
                             "Ra_ref": rho, "Ra_gpu": Ra, "rel_err": rel,
                             "t_gpu": wall, "t_ref": ""})
                print(f"{tool_name:>12} {h_min:>7.4f} {backend:>7} {n:>8d} "
                      f"{Ra:>9.4f} {rel:>9.2e} {wall:>8.2f}s")
    return rows


def run_ex1(args):
    formation = _load_formation(args.tools, str(FORMATION_FILE))
    if args.no_fz:
        formation[:, 2:4] = np.nan
    borehole = _load_borehole(args.tools, str(BOREHOLE_FILE),
                              borehole_geometry_type="diameter")
    rows = []
    fz_note = "without" if args.no_fz else "with"
    print(f"\n=== Ex1 ({fz_note} invasion zones) vs NGSolve order-{args.fe_order} ===")
    print(f"{'tool':>12} {'depth':>6} {'Ra_ngsolve':>11} "
          + "".join(f"{'Ra_' + b:>10} {'rel_' + b:>9}" for b in args.backends)
          + f" {'t_ng':>7} " + "".join(f"{'t_' + b:>7}" for b in args.backends))
    for tool_name in args.tools:
        h_eff = args.h_min or gtool.default_h_min(gtool.tool_config(tool_name))
        for depth in args.depths:
            Ra_ref, t_ref, dofs = ngsolve_reference(
                tool_name, depth, formation, borehole, fe_order=args.fe_order)
            cells = []
            for backend in args.backends:
                Ra, wall, n = gpu_solve_one(
                    tool_name, depth, formation, borehole, backend=backend,
                    h_min=h_eff, tol=args.tol)
                rel = abs(Ra - Ra_ref) / abs(Ra_ref)
                rows.append({"mode": "ex1", "tool": tool_name, "depth": depth,
                             "backend": backend, "h_min": h_eff, "nodes": n,
                             "Ra_ref": Ra_ref, "Ra_gpu": Ra, "rel_err": rel,
                             "t_gpu": wall, "t_ref": t_ref})
                cells.append((Ra, rel, wall))
            print(f"{tool_name:>12} {depth:>6.1f} {Ra_ref:>11.4f} "
                  + "".join(f"{Ra:>10.4f} {rel:>9.2e}" for Ra, rel, _ in cells)
                  + f" {t_ref:>6.1f}s "
                  + "".join(f"{w:>6.1f}s" for _, _, w in cells))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["homo", "ex1"], default="ex1")
    p.add_argument("--tools", nargs="+", default=DEFAULT_TOOLS)
    p.add_argument("--depths", nargs="+", type=float, default=DEFAULT_DEPTHS)
    p.add_argument("--backends", nargs="+", default=["fv", "fem"],
                   choices=["fv", "fem"])
    p.add_argument("--h-min", type=float, default=None,
                   help="Structured-grid minimum spacing (m). Default: per-tool "
                        "heuristic min_gap/20 clipped to [0.0025, 0.01].")
    p.add_argument("--h-sweep", nargs="+", type=float,
                   default=[0.02, 0.01, 0.005],
                   help="homo mode: h_min refinement sweep.")
    p.add_argument("--growth", type=float, default=1.15)
    p.add_argument("--rho", type=float, default=10.0, help="homo mode: rho.")
    p.add_argument("--tol", type=float, default=1e-10, help="CG rel tolerance.")
    p.add_argument("--fe-order", type=int, default=3,
                   help="NGSolve reference order. Default 3.")
    p.add_argument("--no-fz", action="store_true",
                   help="ex1: blank invasion zones like scripts/forward.py.")
    p.add_argument("--output", type=Path, default=ROOT / "benchmark_data" / "gpu_solver")
    args = p.parse_args(argv)

    rows = run_homo(args) if args.mode == "homo" else run_ex1(args)

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"gpu_validation_{args.mode}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} rows -> {out}")

    worst = max(rows, key=lambda r: r["rel_err"])
    print(f"worst rel_err: {worst['rel_err']:.2e} "
          f"({worst['tool']} depth {worst['depth']} {worst['backend']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
