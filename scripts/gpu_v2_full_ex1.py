# -*- coding: utf-8 -*-
"""G2 validation ladder: full Ex1 via the v2 global solver vs Results_1.txt.

Reproduces the forward.py workload (8 tools incl. laterals, 0-25 m @ 0.1 m,
invasion blanked) on the global factor-once/solve-many GPU solver — ALL 8
tools on ONE global grid (no radius bucketing needed, unlike v1's
depth-relative grids) — and diffs against the frozen NGSolve log.

Two runs:
  * matched-R (v1 convention max(10*span,5) = 110 m): apples-to-apples with
    the frozen reference — this is the discretization-parity number;
  * default-R (adopted 80*span = 880 m): the production configuration; its
    extra deviation vs the frozen log at long-tool edge depths is the
    REFERENCE's truncation error (see wiki finding, E2).

Ex1's RM log is smooth (1.1000..1.1008), so the mud convention is a
non-issue here by construction.

Usage:
    python scripts/gpu_v2_full_ex1.py
"""

from __future__ import annotations

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

from remo3d.sensitivity import _load_borehole  # noqa: E402
from remo3d.gpu_solver import global_op, global_gpu  # noqa: E402

TOOLS = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N",
         "N0.5M2.0A", "M4.0A0.5B", "N2.0M0.5A", "N11.0M0.5A"]
VALIDATION_FILE = ROOT / "notebooks" / "validation" / "Results_1.txt"
FORMATION_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Formation.txt"
BOREHOLE_FILE = ROOT / "notebooks" / "Input" / "Ex1" / "Borehole.txt"


def main(argv=None):
    formation = np.loadtxt(FORMATION_FILE, skiprows=2)
    formation[:, 2:4] = np.nan          # forward.py blanks invasion
    borehole = _load_borehole(TOOLS, str(BOREHOLE_FILE),
                              borehole_geometry_type="diameter")
    depths = np.round(np.arange(251) * 0.1, 6)

    with open(VALIDATION_FILE) as fh:
        header = fh.readline().split()
    data = np.loadtxt(VALIDATION_FILE, skiprows=2)
    ref = {name: data[:, k] for k, name in enumerate(header)}
    assert np.allclose(ref["DEPTH"], depths)

    for label, R in (("matched-R (v1 conv, 110)", 110.0),
                     ("default-R (adopted, 880)", None)):
        t0 = time.perf_counter()
        p = global_op.build_global_tasks(TOOLS, depths, formation, borehole,
                                         domain_radius=R)
        solver = global_gpu.make_solver(p, precision="mixed")
        X0 = solver(np.asarray(formation)[None], np.asarray(borehole)[None])
        X0.block_until_ready()
        logs = global_gpu.extract_logs(p, X0)[0]
        dt = time.perf_counter() - t0

        print(f"\n== {label}: grid {len(p['r_nodes'])}x{len(p['z_nodes'])} "
              f"n_free={p['n_free']:,} k={len(p['uniq_src'])} "
              f"(dedup x{p['n_tasks'] / len(p['uniq_src']):.2f}) "
              f"| {dt:.1f}s incl. compile ==")
        overall = 0.0
        for t in TOOLS:
            rel = np.abs(logs[t] - ref[t]) / np.abs(ref[t])
            overall = max(overall, float(np.max(rel)))
            print(f"  {t:12s} max {np.max(rel):8.3%}  mean {np.mean(rel):7.3%}"
                  f"  worst@z={depths[int(np.argmax(rel))]:5.1f}m")
        print(f"  overall max vs Results_1.txt: {overall:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
