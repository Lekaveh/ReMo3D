"""Instrument B — direct (single-process) per-phase solver micro-benchmark.

Complements ``benchmark_optimizations.py`` (Instrument A, the full MPI pipeline).
This bypasses MPI and calls the solver directly (like ``benchmark_task0.py``), so
it attributes time cleanly to each phase — setup / assembly / factorization /
rhs / solve — and records DOF counts, solver type, and CG iterations that the MPI
worker path discards.

For each (sample, depth) it builds one Netgen mesh per domain radius (using a
synthetic normal tool, exactly like benchmark_task0.py) and then, on that
identical mesh, sweeps the optimization flag combinations, calling
``ngsolve_functions.SolveBVP(..., return_metrics=True)``. Runs single-process, so
launch it as a plain ``python`` command (no mpiexec needed). It is cheap
(~0.1-0.5 s per solve), so it can cover all 100 samples.

Notes:
  * A synthetic normal tool (spacing ``--spacing``) is used for meshing + solve.
    The per-phase timing, DOF counts, solver type and CG iterations are what
    matter here and are essentially tool-independent; the apparent resistivity
    is a sanity cross-check (direct vs CG must agree), not a pipeline match.
  * ``reuse_assembly`` is a batch/worker concept (amortizing one assembly over
    many RHS) not exercised here — it is measured by Instrument A. The
    ``assembly`` + ``factorization`` phase times recorded here are exactly what
    reuse amortizes, so the two instruments together quantify its benefit.

Usage:
    python scripts/benchmark_solver_phases.py --n-samples 20 --depths 8
    python scripts/benchmark_solver_phases.py --n-samples 100 --depths 6 --repeats 3

Output: <output>/solver_phases.csv (one row per sample/depth/combo).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# Import the solver/mesh helpers as top-level modules from the repo's remo3d/
# source folder (same pattern as scripts/benchmark_task0.py).
ROOT = Path(__file__).resolve().parents[1]
REMO3D_DIR = ROOT / "remo3d"
if str(REMO3D_DIR) not in sys.path:
    sys.path.insert(0, str(REMO3D_DIR))

import ngsolve as ngs  # noqa: E402
import netgen_functions as ngf  # noqa: E402
import ngsolve_functions as ngsf  # noqa: E402

ngs.ngsglobals.msg_level = 0

DEFAULT_SAMPLES_DIR = ROOT / "benchmark_data" / "smooth_noise" / "data"
DEFAULT_OUTPUT = ROOT / "benchmark_data" / "optim_bench"

# Flag combos (mirror benchmark_optimizations variants, deduped for SolveBVP where
# reuse_assembly makes no difference). name -> pipeline variant it corresponds to.
COMBOS = [
    {"name": "baseline",    "variant": "V1_baseline",    "symmetric": False, "direct_solver": False,  "condense": False, "fe_order": 3, "domain_radius": 40.0},
    {"name": "symmetric",   "variant": "V2_symmetric",   "symmetric": True,  "direct_solver": False,  "condense": False, "fe_order": 3, "domain_radius": 40.0},
    {"name": "condense",    "variant": "V4_condense",    "symmetric": True,  "direct_solver": False,  "condense": True,  "fe_order": 3, "domain_radius": 40.0},
    {"name": "direct",      "variant": "V5_all_on",      "symmetric": True,  "direct_solver": "auto", "condense": True,  "fe_order": 3, "domain_radius": 40.0},
    {"name": "order2",      "variant": "V6_order2",      "symmetric": True,  "direct_solver": "auto", "condense": True,  "fe_order": 2, "domain_radius": 40.0},
    {"name": "domain_auto", "variant": "V7_domain_auto", "symmetric": True,  "direct_solver": "auto", "condense": True,  "fe_order": 3, "domain_radius": "auto"},
]


def synthetic_tool(spacing):
    """A two-electrode normal tool: source at 0, measure at `spacing` (as benchmark_task0)."""
    tool_geometry = np.array([0.0, spacing], dtype=float)
    source_terms = np.array([1.0, 0.0], dtype=float)
    geometric_factor = 4.0 * math.pi * spacing
    return tool_geometry, source_terms, geometric_factor


def pick_depths(depths, n):
    if n >= len(depths):
        return list(range(len(depths)))
    return sorted(set(np.linspace(0, len(depths) - 1, n).round().astype(int).tolist()))


def apparent_resistivity(gfu, mesh, geometry, source_terms, geometric_factor):
    measuring = geometry[source_terms == 0.0]
    if measuring.shape[0] == 1:
        return abs(geometric_factor * gfu(mesh(0.0, float(measuring[0]))))
    if measuring.shape[0] == 2:
        return abs(geometric_factor * (gfu(mesh(0.0, float(measuring[1]))) - gfu(mesh(0.0, float(measuring[0])))))
    return float("nan")


def build_mesh(borehole, formation, mud_res, depth, domain_radius, geometry, source_terms):
    local_formation, local_borehole, sigma = ngf.SelectNetgenDataRange(
        borehole[:, :2], formation, mud_res, depth, domain_radius,
    )
    mesh_data = ngf.ConstructNetgen2dModel(
        domain_radius, geometry, source_terms, local_formation, local_borehole,
    )
    return ngs.Mesh(mesh_data), ngs.CoefficientFunction(sigma)


def run(args):
    geometry, source_terms, geometric_factor = synthetic_tool(args.spacing)
    auto_r = max(10.0 * args.spacing, 5.0)

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "solver_phases.csv"

    fields = ["sample", "depth_index", "depth", "combo", "variant",
              "symmetric", "direct_solver", "condense", "fe_order", "domain_radius",
              "dofs_total", "dofs_free", "solver_type", "cg_iterations", "final_residual_norm",
              "t_setup", "t_assembly", "t_factorization", "t_rhs", "t_solve", "t_total",
              "apparent_resistivity"]
    n_rows = 0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for idx in range(args.sample_start, args.sample_start + args.n_samples):
            path = samples_dir / f"sample_{idx}.npz"
            if not path.is_file():
                print(f"[warn] missing {path}; skipping")
                continue
            d = np.load(path, allow_pickle=True)
            formation = np.asarray(d["formation_model"], dtype=float)
            borehole = np.asarray(d["borehole_model"], dtype=float)
            depths = borehole[:, 0].astype(float)
            depth_idx_list = pick_depths(depths, args.depths)

            for di in depth_idx_list:
                depth = float(depths[di])
                mud_res = float(np.interp(depth, borehole[:, 0], borehole[:, 2]))
                # Build one mesh per distinct domain radius, reuse across combos.
                meshes = {}
                for radius_key in {c["domain_radius"] for c in COMBOS}:
                    radius = auto_r if radius_key == "auto" else float(radius_key)
                    try:
                        meshes[radius_key] = build_mesh(borehole, formation, mud_res, depth, radius, geometry, source_terms)
                    except Exception as e:
                        print(f"[warn] meshing failed sample {idx} depth {di} r={radius}: {e!r}")
                        meshes[radius_key] = None

                for combo in COMBOS:
                    mesh_sigma = meshes[combo["domain_radius"]]
                    if mesh_sigma is None:
                        continue
                    mesh, sigma_cf = mesh_sigma
                    radius = auto_r if combo["domain_radius"] == "auto" else float(combo["domain_radius"])
                    best = None
                    apparent = float("nan")
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        fes, gfu, m = ngsf.SolveBVP(
                            mesh, sigma_cf, geometry, source_terms, [2],
                            args.preconditioner, combo["condense"],
                            order=combo["fe_order"], symmetric=combo["symmetric"],
                            direct_solver=combo["direct_solver"], return_metrics=True,
                        )
                        t_total = time.perf_counter() - t0
                        if best is None or t_total < best[1]:
                            best = (m, t_total)
                        apparent = apparent_resistivity(gfu, mesh, geometry, source_terms, geometric_factor)
                    m, t_total = best
                    tm = m.get("timings", {})
                    writer.writerow({
                        "sample": idx, "depth_index": di, "depth": f"{depth:.4f}",
                        "combo": combo["name"], "variant": combo["variant"],
                        "symmetric": combo["symmetric"], "direct_solver": combo["direct_solver"],
                        "condense": combo["condense"], "fe_order": combo["fe_order"],
                        "domain_radius": f"{radius:.2f}",
                        "dofs_total": m.get("dofs_total"), "dofs_free": m.get("dofs_free"),
                        "solver_type": m.get("solver_type"), "cg_iterations": m.get("cg_iterations"),
                        "final_residual_norm": m.get("final_residual_norm"),
                        "t_setup": f"{tm.get('setup', float('nan')):.6f}",
                        "t_assembly": f"{tm.get('assembly', float('nan')):.6f}",
                        "t_factorization": f"{tm.get('factorization', float('nan')):.6f}",
                        "t_rhs": f"{tm.get('rhs', float('nan')):.6f}",
                        "t_solve": f"{tm.get('solve', float('nan')):.6f}",
                        "t_total": f"{t_total:.6f}",
                        "apparent_resistivity": f"{apparent:.6f}",
                    })
                    n_rows += 1
            print(f"[sample {idx}] {len(depth_idx_list)} depths x {len(COMBOS)} combos done", flush=True)

    print(f"\n[done] wrote {n_rows} rows -> {csv_path}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=20, help="Number of samples. Default: 20.")
    p.add_argument("--sample-start", type=int, default=0, help="First sample index. Default: 0.")
    p.add_argument("--depths", type=int, default=8, help="Evenly-spaced depths per sample. Default: 8.")
    p.add_argument("--repeats", type=int, default=1, help="Solve repeats per combo (keep min time). Default: 1.")
    p.add_argument("--spacing", type=float, default=1.0, help="Synthetic normal-tool electrode spacing (m). Default: 1.0.")
    p.add_argument("--preconditioner", default="multigrid", help="Preconditioner for the CG path. Default: multigrid.")
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.n_samples < 1 or args.depths < 1 or args.repeats < 1:
        raise SystemExit("--n-samples, --depths, --repeats must be >= 1")
    if args.spacing <= 0:
        raise SystemExit("--spacing must be > 0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
