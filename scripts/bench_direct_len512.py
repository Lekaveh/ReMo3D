"""Multi-sample throughput benchmark of the direct solver on len512.

Runs N len512 samples (5 normal tools x 256 depths = 1280 solves each)
back-to-back through the whole-model batched Thomas solve. The tool grids are
canonical (shape depends only on the tool), so the JIT compile is paid once
for the entire run regardless of N; per-sample work is sigma sampling (GPU) +
band assembly + chunked solve.

Target (user): >=10x faster than the NGSolve forced-direct CPU reference.
Reference (user-measured): the real pipeline with 32 workers does a len512
sample (5 tools x 256 depths) in ~40 s. Target: <=4 s/sample.

Usage:
    python scripts/bench_direct_len512.py --n-samples 1
    python scripts/bench_direct_len512.py --n-samples 10 --chunk 64
    python scripts/bench_direct_len512.py --n-samples 100 --chunk 64
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

from remo3d.gpu_solver import direct as gdirect  # noqa: E402
from remo3d.gpu_solver import driver as gdrv  # noqa: E402
from remo3d.gpu_solver import tool as gtool  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "len512" / "smooth_noise" / "data"
CPU_REF_S_PER_SAMPLE = 40.0    # user-measured: 32-worker NGSolve-direct pipeline


def run_sample(si, chunk, dtype, out_dir=None):
    d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
    formation = np.asarray(d["formation_model"], float)
    borehole = np.asarray(d["borehole_model"], float)
    depths = borehole[:, 0].astype(float)

    t0 = time.perf_counter()
    problems = [gdrv.build_tool_problem(t, depths, formation, borehole,
                                        backend="fv", dtype=dtype,
                                        precond="jacobi")
                for t in TOOLS]
    bands, b, layout = gdirect.build_model_batch(problems)
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    us = gdirect.solve_model_batch(bands, b, layout, chunk=chunk)
    t_solve = time.perf_counter() - t0

    logs = np.empty((len(TOOLS), len(depths)))
    for ti, (p, u) in enumerate(zip(problems, us)):
        logs[ti] = gtool.apparent_resistivity(u, p["cfg"], p["j_M"], p["j_N"])
    if out_dir is not None:
        np.savez_compressed(out_dir / f"direct_sample_{si}.npz",
                            logs=logs, depths=depths, tools=TOOLS)
    return t_setup, t_solve, logs


def run_group(sample_ids, chunk, dtype):
    """Assemble ALL systems of a group of samples into one batch and solve.

    This is the "at once" path: the whole group's (sample x tool x depth)
    systems go through solve_model_batch together, so the batched-Cholesky
    per-step latency is amortized over a much larger batch (7.5 vs 40 us per
    system at B=1024 vs 64). Bands for the group live in host memory; the
    solve streams chunks of `chunk` systems to the GPU.
    """
    all_bands = {"a_r": [], "a_z": [], "s": []}
    all_b, layouts = [], []
    t0 = time.perf_counter()
    for si in sample_ids:
        d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
        formation = np.asarray(d["formation_model"], float)
        borehole = np.asarray(d["borehole_model"], float)
        depths = borehole[:, 0].astype(float)
        problems = [gdrv.build_tool_problem(t, depths, formation, borehole,
                                            backend="fv", dtype=dtype,
                                            precond="jacobi")
                    for t in TOOLS]
        bands, b, layout = gdirect.build_model_batch(problems)
        for k in all_bands:
            all_bands[k].append(np.asarray(bands[k]))
        all_b.append(np.asarray(b))
        layouts.append((layout, [p["cfg"] for p in problems],
                        [(p["j_M"], p["j_N"]) for p in problems]))
    import jax.numpy as _jnp
    bands = {k: _jnp.asarray(np.concatenate(v)) for k, v in all_bands.items()}
    b = _jnp.asarray(np.concatenate(all_b))
    flat_layout = [row for (layout, _, _) in layouts for row in layout]
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    us = gdirect.solve_model_batch(bands, b, flat_layout, chunk=chunk)
    t_solve = time.perf_counter() - t0
    return t_setup, t_solve, b.shape[0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--group", type=int, default=0,
                   help="Samples per combined batch (0 = sequential loop). "
                        "e.g. 4 solves 4 samples' 5120 systems at once.")
    p.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    p.add_argument("--save", action="store_true",
                   help="Save per-sample logs npz next to the benchmark data.")
    args = p.parse_args(argv)
    dtype = jnp.float32 if args.dtype == "f32" else jnp.float64
    out_dir = None
    if args.save:
        out_dir = ROOT / "benchmark_data" / "gpu_solver" / "direct_len512"
        out_dir.mkdir(parents=True, exist_ok=True)

    mode = f"group={args.group}" if args.group else "sequential"
    print(f"len512 direct: {args.n_samples} sample(s), chunk={args.chunk}, "
          f"{mode}, {args.dtype}; CPU ref (32-worker direct) = "
          f"{CPU_REF_S_PER_SAMPLE:.0f} s/sample")
    t_all0 = time.perf_counter()

    if args.group:
        setups, solves = [], []
        ids = list(range(args.start, args.start + args.n_samples))
        for g0 in range(0, len(ids), args.group):
            grp = ids[g0:g0 + args.group]
            t_setup, t_solve, nsys = run_group(grp, args.chunk, dtype)
            setups.append(t_setup)
            solves.append(t_solve)
            tag = "  <- incl. compile" if g0 == 0 else ""
            print(f"  group {grp[0]:3d}-{grp[-1]:3d} ({nsys} systems): "
                  f"setup {t_setup:6.2f}s  solve {t_solve:6.2f}s  "
                  f"-> {t_solve / len(grp):.2f}s/sample solve{tag}", flush=True)
        t_all = time.perf_counter() - t_all0
        # warm = exclude the first group (compile)
        warm_groups = max(len(setups) - 1, 1)
        warm = ((np.sum(setups[1:]) + np.sum(solves[1:])) /
                (warm_groups * args.group)) if len(setups) > 1 else \
               t_all / args.n_samples
    else:
        setups, solves = [], []
        for k in range(args.n_samples):
            si = args.start + k
            t_setup, t_solve, _ = run_sample(si, args.chunk, dtype, out_dir)
            setups.append(t_setup)
            solves.append(t_solve)
            tag = "  <- includes JIT compile" if k == 0 else ""
            print(f"  sample {si:3d}: setup {t_setup:6.2f}s  "
                  f"solve {t_solve:6.2f}s{tag}", flush=True)
        t_all = time.perf_counter() - t_all0
        warm = ((np.sum(setups[1:]) + np.sum(solves[1:]))
                / max(args.n_samples - 1, 1)) if args.n_samples > 1 else \
               t_all / args.n_samples

    per = t_all / args.n_samples
    print(f"\nTOTAL {t_all:.1f}s for {args.n_samples} sample(s) "
          f"-> {per:.2f} s/sample (incl. compile); warm {warm:.2f} s/sample")
    speedup = CPU_REF_S_PER_SAMPLE / warm
    print(f"vs NGSolve-direct 32w (40s): x{speedup:.1f} warm "
          f"({'TARGET MET (>=10x)' if speedup >= 10 else 'below 10x target'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
