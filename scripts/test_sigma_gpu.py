"""Equivalence + speed check: sigma_gpu (jnp) vs grid (numpy) samplers.

Runs both anisotropic subcell samplers on the Ex1 model and a len512 sample
and reports the max absolute difference (target: float64 round-off) and the
wall times.

Usage:
    python scripts/test_sigma_gpu.py
"""

from __future__ import annotations

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

from remo3d.gpu_solver import grid as ggrid  # noqa: E402
from remo3d.gpu_solver import sigma_gpu  # noqa: E402
from remo3d.sensitivity import _load_formation, _load_borehole  # noqa: E402


def check(name, formation, borehole, depths, shift, subsample):
    r, zr = ggrid.build_grid(25.0, 0.0, [0.0, 2.0, 2.5],
                             r_foci=ggrid.dedup_foci(
                                 [float(np.median(borehole[:, 1]))]
                                 + [float(v) for v in formation[:, 2]
                                    if np.isfinite(v)], 0.04),
                             h_min=0.01)
    z_sims = depths + shift
    muds = np.interp(z_sims, borehole[:, 0], borehole[:, 2])

    t0 = time.perf_counter()
    nr_, nz_ = ggrid.sample_sigma_aniso_batch(r, zr, z_sims, formation,
                                              borehole, muds,
                                              subsample=subsample)
    t_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    gr_, gz_ = sigma_gpu.sample_sigma_aniso_gpu(r, zr, z_sims, formation,
                                                borehole, muds,
                                                subsample=subsample)
    import jax
    jax.block_until_ready(gz_)
    t_gpu_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    gr_, gz_ = sigma_gpu.sample_sigma_aniso_gpu(r, zr, z_sims, formation,
                                                borehole, muds,
                                                subsample=subsample)
    jax.block_until_ready(gz_)
    t_gpu = time.perf_counter() - t0

    dr = float(np.abs(np.asarray(gr_) - nr_).max())
    dz = float(np.abs(np.asarray(gz_) - nz_).max())
    print(f"{name:>8} s={subsample}: D={len(depths)} grid {len(zr)}x{len(r)} | "
          f"max|d_sig_r|={dr:.2e} max|d_sig_z|={dz:.2e} | "
          f"numpy {t_np:.2f}s, gpu {t_gpu:.2f}s (compile {t_gpu_compile:.1f}s) "
          f"-> x{t_np / max(t_gpu, 1e-9):.0f}")
    return max(dr, dz)


def main():
    worst = 0.0

    formation = _load_formation(["A2.0M0.5N"],
                                str(ROOT / "notebooks/Input/Ex1/Formation.txt"))
    borehole = _load_borehole(["A2.0M0.5N"],
                              str(ROOT / "notebooks/Input/Ex1/Borehole.txt"),
                              borehole_geometry_type="diameter")
    depths = np.arange(0, 25.1, 0.1)
    worst = max(worst, check("Ex1", formation, borehole, depths, -2.25, 4))
    worst = max(worst, check("Ex1", formation, borehole, depths, -2.25, 1))

    d = np.load(ROOT / "benchmark_data/len512/smooth_noise/data/sample_0.npz",
                allow_pickle=True)
    formation = np.asarray(d["formation_model"], float)
    borehole = np.asarray(d["borehole_model"], float)
    depths = borehole[:, 0].astype(float)
    worst = max(worst, check("len512", formation, borehole, depths, -2.25, 4))

    ok = worst < 1e-11
    print(f"\nworst diff {worst:.2e} -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
