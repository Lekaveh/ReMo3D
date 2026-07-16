"""Instrument A — full-pipeline ablation benchmark of the ReMo3D solver optimizations.

For each optimization *variant* (a set of on/off flags) and each stored sample,
this runs the full MPI forward pipeline (``Model.simulate_logs``) over all tools
and records:
  * the per-tool apparent-resistivity logs, and
  * the end-to-end wall time (plus optional per-task solver metrics).

The baseline variant (all optimizations off) reproduces the original unoptimized
solver; every other variant turns one more optimization on. Comparing a variant's
logs to the baseline gives accuracy; comparing wall times gives the realized
speedup.

A single ``Model`` and one MPI worker pool are reused across every variant and
sample — the new flags are ordinary ``simulate_logs`` kwargs, so no re-spawn and
no git checkout is needed.

Usage (auto-relaunches under ``mpiexec -n 1``):
    python scripts/benchmark_optimizations.py --n-samples 100 --variants all --cpu-workers 24
    python scripts/benchmark_optimizations.py --n-samples 100 --variants headline --collect-metrics
    python scripts/benchmark_optimizations.py --variants V1_baseline,V5_all_on --n-samples 20

Outputs (under --output, default benchmark_data/optim_bench/):
    full_pipeline/<variant>.npz   logs[sample,tool,depth], wall_time[sample], ...
    walltime.csv                  one row per (variant, sample)
    accuracy.csv                  one row per (variant, sample, tool) vs baseline
    solver_metrics.csv            per-task metrics (only with --collect-metrics)
    manifest.json                 run configuration
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- Load the repo's remo3d source (v1.4.0) -----------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from remo3d_loader import load_remo3d  # noqa: E402

remo3d = load_remo3d("folder", quiet=True)
Model = remo3d.Model

DEFAULT_TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
DEFAULT_SAMPLES_DIR = ROOT / "benchmark_data" / "smooth_noise" / "data"
DEFAULT_OUTPUT = ROOT / "benchmark_data" / "optim_bench"


# ==========================================================================
# Variant matrix — cumulative ablation from "all off" (baseline) to "all on"
# ==========================================================================
# Flags: symmetric (T1), reuse_assembly (T2), direct_solver (T5),
# condense (T0 corrected condensation), fe_order (T4), domain_radius (T8).
# Baseline uses condense=False: that is the path that is bit-equivalent to the
# original `main` code (main's condense=True has a harmonic-extension ordering
# bug that optim fixes, so the condensed paths are not comparable).

VARIANTS = [
    {"name": "V1_baseline", "desc": "all optimizations off (== main, condense=False)",
     "symmetric": False, "reuse_assembly": False, "direct_solver": False, "condense": False, "fe_order": 3, "domain_radius": 40.0},
    {"name": "V2_symmetric", "desc": "+ symmetric SPD form (T1)",
     "symmetric": True, "reuse_assembly": False, "direct_solver": False, "condense": False, "fe_order": 3, "domain_radius": 40.0},
    {"name": "V3_reuse", "desc": "+ assemble-once / solve-many (T2)",
     "symmetric": True, "reuse_assembly": True, "direct_solver": False, "condense": False, "fe_order": 3, "domain_radius": 40.0},
    {"name": "V4_condense", "desc": "+ static condensation (T0 corrected)",
     "symmetric": True, "reuse_assembly": True, "direct_solver": False, "condense": True, "fe_order": 3, "domain_radius": 40.0},
    {"name": "V5_all_on", "desc": "+ direct sparse-Cholesky (T5) == optim default",
     "symmetric": True, "reuse_assembly": True, "direct_solver": "auto", "condense": True, "fe_order": 3, "domain_radius": 40.0},
    {"name": "V6_order2", "desc": "V5 with fe_order=2 (T4 accuracy tradeoff)",
     "symmetric": True, "reuse_assembly": True, "direct_solver": "auto", "condense": True, "fe_order": 2, "domain_radius": 40.0},
    {"name": "V7_domain_auto", "desc": "V5 with domain_radius=auto (T8 accuracy tradeoff)",
     "symmetric": True, "reuse_assembly": True, "direct_solver": "auto", "condense": True, "fe_order": 3, "domain_radius": "auto"},
    {"name": "V8_all_changes", "desc": "ALL optimizations on together (T1+T2+T0+T5 + order2 T4 + domain-auto T8)",
     "symmetric": True, "reuse_assembly": True, "direct_solver": "auto", "condense": True, "fe_order": 2, "domain_radius": "auto"},
    {"name": "Vb_order2_only", "desc": "baseline with ONLY fe_order=2 (T4 in isolation, all else off)",
     "symmetric": False, "reuse_assembly": False, "direct_solver": False, "condense": False, "fe_order": 2, "domain_radius": 40.0},
]
VARIANTS_BY_NAME = {v["name"]: v for v in VARIANTS}
BASELINE_NAME = "V1_baseline"
HEADLINE = ["V1_baseline", "V5_all_on"]
FLAG_KEYS = ["symmetric", "reuse_assembly", "direct_solver", "condense", "fe_order", "domain_radius"]


def _ensure_mpi_launcher():
    """Re-exec under ``mpiexec -n 1`` if not already under a process manager.

    Bare ``python script.py`` deadlocks on this host because MPI singleton-init
    Spawn hangs; ``mpiexec -n 1`` gives Spawn a working process manager.
    """
    if os.environ.get("REMO3D_MPI_RELAUNCHED") == "1":
        return
    if os.environ.get("PMI_RANK") is not None or os.environ.get("PMI_SIZE") is not None:
        return
    import shutil

    mpiexec = Path(sys.executable).with_name("mpiexec")
    mpiexec_path = str(mpiexec) if mpiexec.exists() else shutil.which("mpiexec")
    if not mpiexec_path:
        print("[warn] mpiexec not found; running singleton (Spawn may hang).", flush=True)
        return
    env = dict(os.environ, REMO3D_MPI_RELAUNCHED="1")
    argv = [mpiexec_path, "-n", "1", sys.executable, *sys.argv]
    print(f"[main] relaunching under: {' '.join(argv[:4])} ...", flush=True)
    os.execve(mpiexec_path, argv, env)


# ==========================================================================
# Sample loading
# ==========================================================================


def load_samples(samples_dir, sample_start, n_samples):
    """Load formation/borehole arrays + measurement depths for a contiguous index range."""
    samples_dir = Path(samples_dir)
    out = []
    for idx in range(sample_start, sample_start + n_samples):
        path = samples_dir / f"sample_{idx}.npz"
        if not path.is_file():
            print(f"[warn] missing sample: {path} (skipping)")
            continue
        d = np.load(path, allow_pickle=True)
        formation = np.asarray(d["formation_model"], dtype=float)
        borehole = np.asarray(d["borehole_model"], dtype=float)
        depths = borehole[:, 0].astype(float)  # measurement grid == borehole DEPT
        out.append({"id": idx, "formation": formation, "borehole": borehole, "depths": depths})
    if not out:
        raise SystemExit(f"No samples found in {samples_dir} for range "
                         f"[{sample_start}, {sample_start + n_samples}).")
    n_depths = out[0]["depths"].shape[0]
    for s in out:
        if s["depths"].shape[0] != n_depths:
            raise SystemExit(f"Sample {s['id']} has {s['depths'].shape[0]} depths, expected {n_depths}. "
                             "All samples must share the depth grid for a rectangular result array.")
    return out, n_depths


# ==========================================================================
# Run a single variant across all samples on the shared worker pool
# ==========================================================================


def run_variant(model, variant, samples, tools, n_depths, mesh_generator, batch_size, collect_metrics):
    """Run one variant over all samples; return logs array, wall times, nan counts, metrics rows."""
    n_samples = len(samples)
    n_tools = len(tools)
    logs_arr = np.full((n_samples, n_tools, n_depths), np.nan, dtype=float)
    wall = np.full(n_samples, np.nan, dtype=float)
    nan_counts = np.zeros(n_samples, dtype=int)
    metrics_rows = []

    kwargs = {k: variant[k] for k in ("symmetric", "reuse_assembly", "direct_solver", "condense", "fe_order")}
    variant_started = time.perf_counter()

    for si, s in enumerate(samples):
        model.set_model_parameters(
            s["formation"].copy(), s["borehole"].copy(),
            borehole_geometry_type="radius", dip=0,
        )
        model.simulate_logs(
            s["depths"].copy(),
            domain_radius=variant["domain_radius"],
            batch_size=batch_size,
            mesh_generator=mesh_generator,
            collect_metrics=collect_metrics,
            **kwargs,
        )
        wall[si] = float(getattr(model, "wall_time", np.nan))
        for ti, tool in enumerate(tools):
            logs_arr[si, ti, :] = model.logs[tool][:, 1]
        nan_counts[si] = int(np.sum(~np.isfinite(logs_arr[si])))

        if collect_metrics and getattr(model, "task_metrics", None):
            for m in model.task_metrics:
                row = dict(m)
                row["variant"] = variant["name"]
                row["sample"] = s["id"]
                metrics_rows.append(row)

        print(f"  [{variant['name']}] sample {s['id']} ({si + 1}/{n_samples})  "
              f"wall={wall[si]:.2f}s  nan={nan_counts[si]}", flush=True)

    total = time.perf_counter() - variant_started
    print(f"[variant] {variant['name']} done: {n_samples} samples in {total:.1f}s "
          f"(mean {np.nanmean(wall):.2f}s/sample)", flush=True)
    return logs_arr, wall, nan_counts, metrics_rows


def save_variant_npz(out_dir, variant, samples, tools, depths, logs_arr, wall, nan_counts, mesh_generator, batch_size):
    out_dir = Path(out_dir) / "full_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{variant['name']}.npz",
        variant=variant["name"],
        desc=variant["desc"],
        flags=json.dumps({k: variant[k] for k in FLAG_KEYS}),
        sample_ids=np.array([s["id"] for s in samples], dtype=int),
        tools=np.array(tools),
        depths=np.asarray(depths, dtype=float),
        logs=logs_arr,          # [sample, tool, depth]
        wall_time=wall,         # [sample]
        nan_count=nan_counts,   # [sample]
        mesh_generator=mesh_generator,
        batch_size=batch_size,
    )


# ==========================================================================
# Post-processing: accuracy vs baseline + CSV writers
# ==========================================================================


def _relative_error(candidate, baseline):
    scale = np.maximum(np.abs(baseline), 1.0e-30)
    return np.abs(candidate - baseline) / scale


def write_walltime_csv(path, results, samples, tools, mesh_generator, batch_size):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "sample", "wall_time_s", "n_depths", "n_tools",
                    "mesh_generator", "domain_radius", "fe_order",
                    "symmetric", "reuse_assembly", "direct_solver", "condense", "nan_count"])
        for name, res in results.items():
            v = VARIANTS_BY_NAME[name]
            for si, s in enumerate(samples):
                w.writerow([name, s["id"], f"{res['wall'][si]:.6f}", res["logs"].shape[2], len(tools),
                            mesh_generator, v["domain_radius"], v["fe_order"],
                            v["symmetric"], v["reuse_assembly"], v["direct_solver"], v["condense"],
                            int(res["nan_count"][si])])


def write_accuracy_csv(path, results, samples, tools, baseline_name):
    if baseline_name not in results:
        print(f"[warn] baseline {baseline_name} not run; skipping accuracy.csv")
        return
    base = results[baseline_name]["logs"]  # [sample, tool, depth]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "sample", "tool", "max_rel_err_vs_baseline", "mean_rel_err", "n_nan"])
        for name, res in results.items():
            cand = res["logs"]
            for si, s in enumerate(samples):
                for ti, tool in enumerate(tools):
                    b = base[si, ti]
                    c = cand[si, ti]
                    finite = np.isfinite(b) & np.isfinite(c)
                    n_nan = int(np.sum(~np.isfinite(c)))
                    if np.any(finite):
                        rel = _relative_error(c[finite], b[finite])
                        w.writerow([name, s["id"], tool, f"{np.max(rel):.3e}", f"{np.mean(rel):.3e}", n_nan])
                    else:
                        w.writerow([name, s["id"], tool, "nan", "nan", n_nan])


def write_metrics_csv(path, metrics_rows):
    if not metrics_rows:
        return
    cols = ["variant", "sample", "rank", "task_index", "n_solves", "wall",
            "assembly_time", "solve_time", "dofs_total", "dofs_free",
            "solver_type", "cg_iterations", "reuse_assembly"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in metrics_rows:
            w.writerow(row)


# ==========================================================================
# CLI
# ==========================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=100, help="Number of samples. Default: 100.")
    p.add_argument("--sample-start", type=int, default=0, help="First sample index. Default: 0.")
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR,
                   help=f"Directory of sample_<n>.npz. Default: {DEFAULT_SAMPLES_DIR}.")
    p.add_argument("--variants", default="all",
                   help="'all', 'headline' (baseline+all_on), or a comma list of variant names. Default: all.")
    p.add_argument("--tools", default=",".join(DEFAULT_TOOLS), help="Comma-separated tool names.")
    p.add_argument("--cpu-workers", type=int, default=24, help="MPI CPU workers. Default: 24.")
    p.add_argument("--mesh-generator", default="gmsh", choices=["gmsh", "netgen"],
                   help="Mesh generator (held constant across variants). gmsh is robust for the\n"
                        "128-layer benchmark models; netgen can fail to mesh them. Default: gmsh.")
    p.add_argument("--batch-size", type=int, default=5, help="Depths per mesh/assembly. Default: 5.")
    p.add_argument("--collect-metrics", action="store_true",
                   help="Gather per-task solver metrics (DOFs, solver_type, CG iters, phase times).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output dir. Default: {DEFAULT_OUTPUT}.")
    p.add_argument("--list-variants", action="store_true", help="Print the variant matrix and exit.")
    return p.parse_args(argv)


def select_variants(spec):
    if spec == "all":
        return list(VARIANTS)
    if spec == "headline":
        return [VARIANTS_BY_NAME[n] for n in HEADLINE]
    chosen = []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        if name not in VARIANTS_BY_NAME:
            raise SystemExit(f"Unknown variant {name!r}. Known: {', '.join(VARIANTS_BY_NAME)}")
        chosen.append(VARIANTS_BY_NAME[name])
    if not chosen:
        raise SystemExit("No variants selected.")
    return chosen


def main(argv=None):
    args = parse_args(argv)

    if args.list_variants:
        for v in VARIANTS:
            print(f"{v['name']:16s} {v['desc']}")
            print(f"{'':16s} " + "  ".join(f"{k}={v[k]}" for k in FLAG_KEYS))
        return 0

    _ensure_mpi_launcher()

    if args.n_samples < 1:
        raise SystemExit("--n-samples must be >= 1")
    if args.cpu_workers < 1:
        raise SystemExit("--cpu-workers must be >= 1")

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    selected = select_variants(args.variants)
    samples, n_depths = load_samples(args.samples_dir, args.sample_start, args.n_samples)
    depths = samples[0]["depths"]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] variants={[v['name'] for v in selected]}")
    print(f"[main] samples={len(samples)} (from {args.samples_dir})  tools={tools}")
    print(f"[main] cpu_workers={args.cpu_workers}  mesh={args.mesh_generator}  "
          f"batch_size={args.batch_size}  collect_metrics={args.collect_metrics}")
    print(f"[main] output -> {out_dir}")

    # Manifest up front so a crash still records the intended configuration.
    manifest = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "variants": [{k: v[k] for k in ["name", "desc", *FLAG_KEYS]} for v in selected],
        "baseline": BASELINE_NAME,
        "n_samples": len(samples),
        "sample_ids": [s["id"] for s in samples],
        "tools": tools,
        "cpu_workers": args.cpu_workers,
        "mesh_generator": args.mesh_generator,
        "batch_size": args.batch_size,
        "collect_metrics": bool(args.collect_metrics),
        "remo3d_version": getattr(remo3d, "__version__", "unknown"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    results = {}
    all_metrics = []
    model = Model(tools=tools)
    model.initialize_workers(cpu_workers=args.cpu_workers, gpu_workers=0)
    try:
        for variant in selected:
            print(f"\n[variant] === {variant['name']}: {variant['desc']} ===", flush=True)
            logs_arr, wall, nan_counts, metrics_rows = run_variant(
                model, variant, samples, tools, n_depths,
                args.mesh_generator, args.batch_size, args.collect_metrics,
            )
            results[variant["name"]] = {"logs": logs_arr, "wall": wall, "nan_count": nan_counts}
            all_metrics.extend(metrics_rows)
            # Checkpoint this variant immediately (survives a later crash/timeout).
            save_variant_npz(out_dir, variant, samples, tools, depths, logs_arr, wall, nan_counts,
                             args.mesh_generator, args.batch_size)
            write_walltime_csv(out_dir / "walltime.csv", results, samples, tools,
                               args.mesh_generator, args.batch_size)
            if args.collect_metrics:
                write_metrics_csv(out_dir / "solver_metrics.csv", all_metrics)
    finally:
        try:
            model.shutdown_workers()
        except Exception as e:
            print(f"[main] worker shutdown warning: {e!r}")

    write_accuracy_csv(out_dir / "accuracy.csv", results, samples, tools, BASELINE_NAME)

    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Console summary: mean wall time + speedup vs baseline
    print("\n===== summary (mean wall time per sample) =====")
    base_mean = np.nanmean(results[BASELINE_NAME]["wall"]) if BASELINE_NAME in results else None
    for name, res in results.items():
        m = np.nanmean(res["wall"])
        speed = f"  speedup x{base_mean / m:.2f}" if base_mean else ""
        print(f"  {name:16s} {m:7.2f}s/sample{speed}  (nan {int(res['nan_count'].sum())})")
    print(f"\n[main] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
