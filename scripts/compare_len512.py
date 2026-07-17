"""Head-to-head on one len512 sample: NGSolve forced-direct vs GPU solver.

Loads benchmark_data/len512/smooth_noise/data/sample_<n>.npz (radius
convention, 51.2 m pseudowell, 256 layers with per-cell noise — the hardest
sigma-upscaling case) and computes apparent-resistivity logs for the
benchmark tool set two ways:

  * NGSolve single-process, direct solver FORCED (symmetric+condense+order 3,
    sparse-Cholesky — the exact 3.48x CPU winner from the optim benchmark);
  * remo3d.gpu_solver driver (fv backend, MG-PCG, fp32, chunked vmap).

Reports per-tool accuracy (GPU vs NGSolve-direct) and wall times. If the
sample carries its generation-time logs (model_logs from the MPI pipeline),
they are cross-checked too.

Usage:
    python scripts/compare_len512.py [--sample 0] [--tools A2.0M0.5N ...]
    python scripts/compare_len512.py --ngsolve-depths 64   # subsample CPU ref
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

DEFAULT_TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"


def ngsolve_direct_log(tool_name, depths, formation, borehole, fe_order=3,
                       progress_every=64):
    """NGSolve single-process log with the direct solver forced.

    Uses the gmsh mesher: the noisy per-cell len512 formations make Netgen's
    SplineGeometry meshing fail, and the len512 logs were generated with gmsh
    in the first place.
    """
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
    domain_radius = max(10.0 * float(np.max(np.abs(tool_geometry))), 5.0)

    Ra = np.full(len(depths), np.nan)
    t0 = time.perf_counter()
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
        if progress_every and (k + 1) % progress_every == 0:
            print(f"    [{tool_name}] {k + 1}/{len(depths)} depths, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    return Ra, time.perf_counter() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--tools", nargs="+", default=DEFAULT_TOOLS)
    p.add_argument("--ngsolve-depths", type=int, default=0,
                   help="Subsample the CPU reference to N depths (0 = all).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args(argv)

    path = SAMPLES_DIR / f"sample_{args.sample}.npz"
    d = np.load(path, allow_pickle=True)
    formation = np.asarray(d["formation_model"], dtype=float)
    borehole = np.asarray(d["borehole_model"], dtype=float)  # radius, metres
    depths = borehole[:, 0].astype(float)
    print(f"sample_{args.sample}: {len(depths)} depths "
          f"({depths[0]:.1f}-{depths[-1]:.1f} m), {len(formation)} layers, "
          f"{int((~np.isnan(formation[:, 2])).sum())} invaded")

    gen_logs = None
    if "model_logs" in d:
        try:
            gen_logs = d["model_logs"].item()
        except Exception:
            gen_logs = None

    # --- GPU: all tools at once on the shared MP-centred grid ---
    import jax.numpy as jnp
    from remo3d.gpu_solver.driver import compute_logs_gpu
    dtype = jnp.float32 if args.dtype == "f32" else jnp.float64
    t0 = time.perf_counter()
    gpu_logs = compute_logs_gpu(
        args.tools, depths, formation, borehole,
        borehole_geometry_type="radius", backend="fv", dtype=dtype,
        tol=args.tol, batch_size=args.batch_size, shared_grid=True,
        verbose=True)
    t_gpu = time.perf_counter() - t0
    n_solves = len(args.tools) * len(depths)
    print(f"GPU total: {t_gpu:.1f}s for {n_solves} solves "
          f"({1000 * t_gpu / n_solves:.0f} ms/solve)\n")

    # --- NGSolve direct (per tool, possibly subsampled depths) ---
    if args.ngsolve_depths and args.ngsolve_depths < len(depths):
        idx = np.linspace(0, len(depths) - 1,
                          args.ngsolve_depths).round().astype(int)
    else:
        idx = np.arange(len(depths))
    ref_depths = depths[idx]

    print(f"{'tool':>11} {'ng_direct':>10} {'gpu':>8} "
          f"{'max_rel':>9} {'mean_rel':>9} {'p95_rel':>9}"
          + ("  vs_gen(max/mean)" if gen_logs else ""))
    t_ng_total = 0.0
    worst = (0.0, "", 0.0)
    for tool_name in args.tools:
        Ra_ng, t_ng = ngsolve_direct_log(tool_name, ref_depths,
                                         formation, borehole)
        t_ng_total += t_ng
        Ra_gpu = gpu_logs[tool_name][idx, 1]
        ok = np.isfinite(Ra_ng) & np.isfinite(Ra_gpu)
        rel = np.abs(Ra_gpu[ok] - Ra_ng[ok]) / np.abs(Ra_ng[ok])
        k = int(np.argmax(rel))
        if rel.max() > worst[0]:
            worst = (float(rel.max()), tool_name, float(ref_depths[ok][k]))
        extra = ""
        if gen_logs and tool_name in gen_logs:
            gl = np.asarray(gen_logs[tool_name], dtype=float)
            gen_on_ref = np.interp(ref_depths, gl[:, 0], gl[:, 1])
            grel = np.abs(Ra_gpu[ok] - gen_on_ref[ok]) / np.abs(gen_on_ref[ok])
            extra = f"  {grel.max():.2e}/{grel.mean():.2e}"
        print(f"{tool_name:>11} {t_ng:>9.1f}s {t_gpu / len(args.tools):>7.1f}s "
              f"{rel.max():>9.2e} {rel.mean():>9.2e} "
              f"{np.percentile(rel, 95):>9.2e}{extra}")

    per_ng = t_ng_total / (len(args.tools) * len(ref_depths))
    per_gpu = t_gpu / n_solves
    print(f"\nNGSolve-direct: {t_ng_total:.1f}s for "
          f"{len(args.tools) * len(ref_depths)} solves "
          f"({1000 * per_ng:.0f} ms/solve, 1 core)")
    print(f"GPU (incl. setup): {1000 * per_gpu:.0f} ms/solve "
          f"-> x{per_ng / per_gpu:.1f} vs 1 core "
          f"(x{per_ng / per_gpu / 20:.2f} vs ideal 20-core pool)")
    print(f"worst GPU-vs-direct point: {worst[0]:.2e} "
          f"({worst[1]} @ {worst[2]:.1f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
