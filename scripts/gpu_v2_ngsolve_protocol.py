# -*- coding: utf-8 -*-
"""Protocol reference: fresh UNBATCHED NGSolve on an optim_bench subset.

The stored full_pipeline references were produced with batch_size=5 (shared
meshes, up to 0.4 m geometric offset) — measured up to 15% off their own
unbatched convention at isolated points. This script produces the clean
CPU-convention reference (per-depth gmsh mesh, forced direct, scalar mud,
R = max(10*span, 5)) for samples 0 and 56 at every 4th depth, saved to
benchmark_data/gpu_solver/ngsolve_protocol.npz for the final compat
comparison.

Run under plain python (single-process, no MPI spawn).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# insertion order matters: ROOT must end up FIRST so `import remo3d` resolves
# the package, not remo3d/remo3d.py as a top-level module
for _p in (str(ROOT / "scripts"), str(ROOT / "remo3d"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compare_len512 import ngsolve_direct_log  # noqa: E402

TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
SAMPLES = [0, 56]
STRIDE = 4
SAMPLES_DIR = ROOT / "benchmark_data" / "smooth_noise" / "data"
OUT = ROOT / "benchmark_data" / "gpu_solver" / "ngsolve_protocol.npz"


def main():
    try:
        import ngsolve as ngs
        ngs.SetNumThreads(1)
    except Exception:
        pass

    depth_idx = None
    results = {}
    t00 = time.perf_counter()
    for si in SAMPLES:
        d = np.load(SAMPLES_DIR / f"sample_{si}.npz", allow_pickle=True)
        fm = np.asarray(d["formation_model"], float)
        bh = np.asarray(d["borehole_model"], float)
        depths = bh[:, 0].astype(float)
        depth_idx = np.arange(0, len(depths), STRIDE)
        for t in TOOLS:
            ra, dt = ngsolve_direct_log(t, depths[depth_idx], fm, bh,
                                        progress_every=0)
            results[f"s{si}_{t}"] = ra
            print(f"s{si} {t}: {len(depth_idx)} solves in {dt:.0f}s",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, depth_idx=depth_idx, samples=SAMPLES,
                        tools=TOOLS, stride=STRIDE,
                        **results)
    print(f"total {time.perf_counter() - t00:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
