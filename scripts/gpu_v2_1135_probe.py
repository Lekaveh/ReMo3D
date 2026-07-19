# -*- coding: utf-8 -*-
"""Probe the genuine-v2-error sample (large_noise/0/sample_1135).

Fresh NGSolve (R=40 and R=160) and the stored refs agree at ~3.0 while v2
native gives ~2.69 at e.g. A1.0 z=11.4 (-11%). The sample's signature:
caliper 0.26-0.30 m (huge borehole), RM~0.17 (conductive mud), invasion out
to 0.70 m. Probes:
  1. baseline mixed-precision native   (expect the wrong ~2.69);
  2. f64 native                        (precision hypothesis);
  3. mixed + radial plateau h=0.005    (radial-resolution hypothesis);
  4. mixed compat mud="cpu" R=40       (mud-convention hypothesis).

Usage: python scripts/gpu_v2_1135_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remo3d.gpu_solver import global_op, global_gpu, grid as ggrid  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLE = ROOT / "benchmark_data" / "full" / "large_noise/0/data/sample_1135.npz"
# the four arbitrated v2_err points: (tool_idx, z)
POINTS = [(1, 44.4), (1, 11.4), (1, 36.2), (0, 11.4)]
EXPECT = {(1, 44.4): 2.874, (1, 11.4): 3.005, (1, 36.2): 3.017,
          (0, 11.4): 0.750}          # converged fresh NGSolve


def run(tag, mud="log", precision="mixed", domain_radius=None, r_h=None):
    d = np.load(SAMPLE, allow_pickle=True)
    fm = np.asarray(d["formation_model"], float)[None]
    bh = np.asarray(d["borehole_model"], float)[None]
    depths = np.asarray(d["model_logs"].item()[TOOLS[0]], float)[:, 0]

    orig = ggrid.canonical_radial_nodes
    if r_h is not None:
        ggrid.canonical_radial_nodes = (
            lambda R, h, r_fine=0.6, growth=1.15, h_max=None:
            orig(R, r_h, r_fine=r_fine, growth=growth, h_max=h_max))
    try:
        p = global_op.build_global_tasks(TOOLS, depths, fm[0], bh[0],
                                         domain_radius=domain_radius)
        solver = global_gpu.make_solver(
            p, precision=precision, mud=mud,
            chunk_cols=96 if mud == "cpu" else None)
    finally:
        ggrid.canonical_radial_nodes = orig

    X0 = solver(fm, bh)
    sample_logs = next(iter(global_gpu.extract_logs(p, X0)))
    print(f"-- {tag} (grid {len(p['r_nodes'])}x{len(p['z_nodes'])} "
          f"R={p['domain_radius']:.0f}) --")
    for ti, z in POINTS:
        di = int(np.searchsorted(depths, z))
        v = float(sample_logs[TOOLS[ti]][di])
        e = EXPECT[(ti, z)]
        print(f"  {TOOLS[ti]} z={z}: {v:8.4f}  (converged {e:.3f}, "
              f"diff {abs(v - e) / e:6.1%})", flush=True)


def main():
    run("1. native mixed (baseline)")
    run("2. native f64", precision="f64")
    run("3. native mixed, radial h=0.005", r_h=0.005)
    run("4. compat mud=cpu R=40", mud="cpu", domain_radius=40.0)


if __name__ == "__main__":
    main()
