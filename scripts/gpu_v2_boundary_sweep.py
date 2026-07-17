# -*- coding: utf-8 -*-
"""E2 residual: boundary-saturation sweep of the global grid.

The global grid puts the Dirichlet boundary at ``domain_radius`` R — both the
radial extent and the z-pads beyond the electrode range. This sweep solves
the same samples at growing R and reports the log change vs the largest R:
the truncation error of each R level. Far-field nodes grow only
logarithmically with R (geometric tails), so a generous R is nearly free —
the question is where the error saturates.

Samples: s0 (typical) and s24 (the sample whose A8.0 edge points showed the
largest boundary sensitivity in the optim_bench arbitration).

Usage:
    python scripts/gpu_v2_boundary_sweep.py [--radii 45 90 180 360 720]
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

from remo3d.gpu_solver import global_op, global_gpu  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES_DIR = ROOT / "benchmark_data" / "smooth_noise" / "data"
SAMPLE_IDS = [0, 24]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[45.0, 90.0, 180.0, 360.0, 720.0])
    args = ap.parse_args(argv)
    radii = sorted(args.radii)

    fs, bs = [], []
    for si in SAMPLE_IDS:
        d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
        fs.append(np.asarray(d["formation_model"], float))
        bs.append(np.asarray(d["borehole_model"], float))
    fs, bs = np.stack(fs), np.stack(bs)
    depths = bs[0][:, 0]

    logs_by_R = {}
    for R in radii:
        t0 = time.perf_counter()
        p = global_op.build_global_tasks(TOOLS, depths, fs[0], bs[0],
                                         domain_radius=R)
        solver = global_gpu.make_solver(p, precision="mixed")
        X0 = solver(fs, bs)
        X0.block_until_ready()
        logs = global_gpu.extract_logs(p, X0)
        logs_by_R[R] = logs
        print(f"R={R:6.0f}: grid {len(p['r_nodes'])}x{len(p['z_nodes'])} "
              f"n_free={p['n_free']:9,d}  ({time.perf_counter() - t0:.1f}s "
              f"incl. compile)", flush=True)

    ref = logs_by_R[radii[-1]]
    edge = (depths <= depths[0] + 2.0) | (depths >= depths[-1] - 2.0)
    print(f"\ntruncation error vs R={radii[-1]:.0f} "
          f"(max over {len(SAMPLE_IDS)} samples x 128 depths):")
    print(f"{'R':>6s} " + " ".join(f"{t:>10s}" for t in TOOLS)
          + f" {'worst':>8s} {'worst@edge':>10s}")
    for R in radii[:-1]:
        cells, worst, worst_e = [], 0.0, 0.0
        for ti, t in enumerate(TOOLS):
            r = np.stack([np.abs(logs_by_R[R][b][t] - ref[b][t])
                          / np.abs(ref[b][t]) for b in range(len(SAMPLE_IDS))])
            cells.append(float(np.max(r)))
            worst = max(worst, float(np.max(r)))
            worst_e = max(worst_e, float(np.max(r[:, edge])))
        print(f"{R:6.0f} " + " ".join(f"{c:10.3%}" for c in cells)
              + f" {worst:8.3%} {worst_e:10.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
