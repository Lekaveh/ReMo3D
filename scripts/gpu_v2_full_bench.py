# -*- coding: utf-8 -*-
"""v2 compat-mode benchmark on the full/ dataset: 1000 random samples.

The benchmark_data/full/ samples (large_noise / smooth_noise /
unphysical_noise, 256 depths 0..51.0 m, 5 normal tools) store model_logs
computed by the CPU NGSolve pipeline at batch_size=5, domain_radius=40 —
so the matched-convention v2 mode is compat (mud="cpu", R=40), as in
gpu_v2_optim_bench / the wiki gpu-solver-v2 finding.

Selection is seeded (default 42) over ALL samples found under full/, so
every shard of a multi-GPU run picks the same 1000 and takes a
deterministic slice (--range lo:hi — weighted split across unequal GPUs).

Usage:
    python scripts/gpu_v2_full_bench.py [--n 1000] [--batch 4] [--seed 42]
                                        [--range 0:590] [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
FULL = ROOT / "benchmark_data" / "full"
OUTDIR = ROOT / "benchmark_data" / "gpu_solver"


def select_samples(n, seed):
    files = sorted(FULL.glob("*/*/data/sample_*.npz"))
    if not files:
        raise SystemExit(f"no samples under {FULL}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in sorted(idx)], len(files)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--range", dest="rng", default=None, metavar="LO:HI",
                    help="slice of the seeded selection this process runs")
    ap.add_argument("--convention", choices=["v2", "cpu"], default="cpu",
                    help="cpu = compat (scalar mud, pipeline R=40, matches "
                         "the refs' convention); v2 = native (physical "
                         "RM(z), auto R)")
    ap.add_argument("--smoke", action="store_true",
                    help="8 samples, no npz output")
    args = ap.parse_args(argv)

    import jax  # noqa: E402  (after env vars)
    from remo3d.gpu_solver import global_op, global_gpu  # noqa: E402

    picked, total = select_samples(args.n, args.seed)
    if args.smoke:
        picked = picked[:8]
    lo_hi = (tuple(int(x) for x in args.rng.split(":")) if args.rng
             else (0, len(picked)))
    mine = picked[lo_hi[0]:lo_hi[1]]
    print(f"dataset: {total} samples; selected {len(picked)} (seed "
          f"{args.seed}); range {lo_hi[0]}:{lo_hi[1]} -> "
          f"{len(mine)} samples; device {jax.devices()[0]}", flush=True)

    # -- load samples + stored NGSolve refs (batch5, R=40) -----------------
    t0 = time.perf_counter()
    fs, bs, refs, depths = [], [], [], None
    for f in mine:
        d = np.load(f, allow_pickle=True)
        fs.append(np.asarray(d["formation_model"], float))
        bs.append(np.asarray(d["borehole_model"], float))
        ml = d["model_logs"].item()
        rows = []
        for t in TOOLS:
            a = np.asarray(ml[t], float)
            if depths is None:
                depths = a[:, 0].copy()
            assert np.allclose(a[:, 0], depths), f"depth grid differs: {f}"
            rows.append(a[:, 1])
        refs.append(np.stack(rows))
    fs, bs, refs = np.stack(fs), np.stack(bs), np.stack(refs)
    print(f"loaded {len(mine)} samples in {time.perf_counter() - t0:.1f}s; "
          f"{len(TOOLS)} tools x {len(depths)} depths "
          f"[{depths[0]}..{depths[-1]}]; ref nans: "
          f"{int(np.isnan(refs).sum())}", flush=True)

    # -- build problem + solver --------------------------------------------
    if args.convention == "cpu":
        # compat: scalar mud per depth + the refs' fixed pipeline R=40
        p = global_op.build_global_tasks(TOOLS, depths, fs[0], bs[0],
                                         domain_radius=40.0)
        solver = global_gpu.make_solver(p, precision="mixed", mud="cpu",
                                        chunk_cols=96)
    else:
        # native v2: physical RM(z) column, auto boundary (R=720 here)
        p = global_op.build_global_tasks(TOOLS, depths, fs[0], bs[0])
        solver = global_gpu.make_solver(p, precision="mixed")
    print(f"grid {len(p['r_nodes'])}x{len(p['z_nodes'])} "
          f"n_free={p['n_free']:,} k={len(p['uniq_src'])} "
          f"(dedup x{p['n_tasks'] / len(p['uniq_src']):.2f}) "
          f"R={p['domain_radius']:.0f}", flush=True)

    # -- solve in batches --------------------------------------------------
    B = args.batch
    n = len(mine)
    logs = np.full((n, len(TOOLS), len(depths)), np.nan)
    times = []
    t_run = time.perf_counter()
    for lo in range(0, n, B):
        hi = min(lo + B, n)
        # keep the compiled batch shape: pad the tail by repeating last
        sl = np.arange(lo, hi)
        pad = np.concatenate([sl, np.repeat(sl[-1:], B - len(sl))])
        t0 = time.perf_counter()
        X0 = solver(fs[pad], bs[pad])
        X0.block_until_ready()
        dt = time.perf_counter() - t0
        times.append(dt)
        for b, sample_logs in enumerate(global_gpu.extract_logs(p, X0)):
            if lo + b >= n:
                break
            for ti, t in enumerate(TOOLS):
                logs[lo + b, ti] = sample_logs[t]
        done = hi
        warm = times[1:] or times
        eta = (n - done) / B * float(np.mean(warm))
        print(f"  [{done:4d}/{n}] batch {dt:6.2f}s ({dt / B:5.2f} s/sample)"
              + ("   <- includes compile" if lo == 0 else
                 f"   ETA {eta / 60:.0f} min"), flush=True)
    wall = time.perf_counter() - t_run

    if not args.smoke:
        mode = "compat" if args.convention == "cpu" else "native"
        out = OUTDIR / (f"full_v2_{mode}_n{args.n}_seed{args.seed}"
                        + (f"_r{lo_hi[0]}_{lo_hi[1]}" if args.rng else "")
                        + ".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, logs=logs, refs=refs, depths=depths, tools=TOOLS,
            files=np.array([str(f.relative_to(FULL)) for f in mine]),
            batch_times=np.asarray(times), batch=B, wall=wall)
        print(f"saved {out}", flush=True)

    # -- stats -------------------------------------------------------------
    warm = times[1:] or times
    print(f"\nTIME: total {wall / 60:.1f} min for {n} samples | "
          f"warm {np.mean(warm) / B:.2f} s/sample "
          f"(median {np.median(warm) / B:.2f}) | "
          f"compile+first batch {times[0]:.1f}s")

    rel = np.abs(logs - refs) / np.abs(refs)
    print(f"\nACCURACY vs stored NGSolve (batch5, R=40) — relative diff "
          f"[{args.convention} convention"
          + (", incl. convention gap" if args.convention == "v2" else "")
          + "]:")
    for ti, t in enumerate(TOOLS):
        r = rel[:, ti]
        print(f"  {t}: mean {np.nanmean(r):.3%}  median "
              f"{np.nanmedian(r):.3%}  p95 {np.nanpercentile(r, 95):.3%}  "
              f"p99 {np.nanpercentile(r, 99):.3%}  max {np.nanmax(r):.2%}")
    print(f"  ALL      : mean {np.nanmean(rel):.3%}  median "
          f"{np.nanmedian(rel):.3%}  p95 {np.nanpercentile(rel, 95):.3%}  "
          f"p99 {np.nanpercentile(rel, 99):.3%}  max {np.nanmax(rel):.2%}")

    cats = np.array([str(f.relative_to(FULL)).split("/")[0] for f in mine])
    print("\nby category:")
    for c in sorted(set(cats)):
        m = cats == c
        print(f"  {c:17s} ({int(m.sum()):4d}): mean {np.nanmean(rel[m]):.3%}"
              f"  p99 {np.nanpercentile(rel[m], 99):.3%}"
              f"  max {np.nanmax(rel[m]):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
