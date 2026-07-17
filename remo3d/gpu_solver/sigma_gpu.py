# -*- coding: utf-8 -*-
"""GPU (jnp) port of the anisotropic subcell sigma sampler.

grid.sample_sigma_aniso_batch is memory-bound numpy and dominated driver setup
time (~45-70 s per tool on len512). The lookup is searchsorted / interp /
where — all available in jnp — so the whole sampling runs on the device and
the sigma stacks are born device-resident.

Formulas are 1:1 with grid.sample_sigma_aniso_batch (see its docstring for
the physics); the equivalence test in scripts/test_sigma_gpu.py holds them to
float64 round-off against the numpy path.

Layout note: the (D, nz-1, s_z, nr-1, s_r) intermediate for a full 256-depth
stack in fp64 is ~6 GB, so sampling is chunked over depths with a fixed chunk
shape (one JIT specialization); the tail chunk is padded.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def _sigma_at_points_jnp(R, Z, tops0, bots, fz_radius, fz_value, uz_value,
                         bh_z, bh_r, mud):
    """jnp twin of grid._sigma_at_points (piecewise-constant lookup).

    Formation columns are passed pre-split (searchsorted needs 1D bots);
    mud broadcasts against R/Z (per-depth column with a leading batch axis).
    """
    layer = jnp.searchsorted(bots, Z, side="right")
    layer = jnp.clip(layer, 0, len(bots) - 1)

    in_fz = (~jnp.isnan(fz_radius[layer])) & (R < fz_radius[layer])
    res = jnp.where(in_fz, fz_value[layer], uz_value[layer])

    r_bh = jnp.interp(Z, bh_z, bh_r)
    res = jnp.where(R < r_bh, mud, res)
    return 1.0 / res


@partial(jax.jit, static_argnames=("subsample",))
def _sample_chunk(r_nodes, z_nodes_rel, z_sims, muds,
                  tops0, bots, fz_radius, fz_value, uz_value,
                  bh_z, bh_r, subsample):
    """Anisotropic (sigma_r, sigma_z) for one fixed-size chunk of depths."""
    r = r_nodes
    z = z_nodes_rel
    s = subsample

    frac = (jnp.arange(s) + 0.5) / s
    z_sub = z[:-1, None] + jnp.diff(z)[:, None] * frac[None, :]   # (nzc, s)
    r_sub = r[:-1, None] + jnp.diff(r)[:, None] * frac[None, :]   # (nrc, s)
    wr = r_sub / jnp.maximum(r_sub.sum(axis=1, keepdims=True), 1e-300)

    # axes: (d, nzc, s_z, nrc, s_r)
    R = r_sub[None, None, None, :, :]
    Z = z_sub[None, :, :, None, None] + z_sims[:, None, None, None, None]
    mud = muds[:, None, None, None, None]
    sig = _sigma_at_points_jnp(R, Z, tops0, bots, fz_radius, fz_value,
                               uz_value, bh_z, bh_r, mud)

    if s == 1:
        flat = sig[:, :, 0, :, 0]
        return flat, flat

    harm_z = 1.0 / jnp.mean(1.0 / sig, axis=2)                # (d, nzc, nrc, s_r)
    sig_z = (harm_z * wr[None, None, :, :]).sum(axis=3)

    inv_r = 1.0 / jnp.maximum(R, 1e-300)
    w_res = inv_r / inv_r.sum(axis=4, keepdims=True)
    harm_r = 1.0 / (w_res / sig).sum(axis=4)                  # (d, nzc, s_z, nrc)
    sig_r = harm_r.mean(axis=2)
    return sig_r, sig_z


def sample_sigma_aniso_gpu(r_nodes, z_nodes_rel, z_sims, formation, borehole,
                           muds, subsample=4, chunk=32, dtype=jnp.float64):
    """Device-resident (sigma_r, sigma_z) stacks over a batch of depths.

    Mirrors grid.sample_sigma_aniso_batch (same arguments and semantics) but
    computes on the GPU and returns jnp arrays of ``dtype`` with shape
    (D, nz-1, nr-1). Internals run in float64 for exact agreement with the
    numpy path; the result is cast to ``dtype`` at the end.
    """
    r = jnp.asarray(np.asarray(r_nodes, dtype=float))
    z = jnp.asarray(np.asarray(z_nodes_rel, dtype=float))
    z_sims = np.asarray(z_sims, dtype=float)
    muds = np.asarray(muds, dtype=float)
    formation = np.asarray(formation, dtype=float)
    borehole = np.asarray(borehole, dtype=float)

    args = (jnp.asarray(formation[:, 0]), jnp.asarray(formation[:, 1]),
            jnp.asarray(formation[:, 2]), jnp.asarray(formation[:, 3]),
            jnp.asarray(formation[:, 4]),
            jnp.asarray(borehole[:, 0]), jnp.asarray(borehole[:, 1]))

    D = len(z_sims)
    chunk = int(min(chunk, D))
    parts_r, parts_z = [], []
    for lo in range(0, D, chunk):
        hi = min(lo + chunk, D)
        # pad the tail to the fixed chunk shape (single JIT specialization)
        zs = np.pad(z_sims[lo:hi], (0, chunk - (hi - lo)), mode="edge")
        md = np.pad(muds[lo:hi], (0, chunk - (hi - lo)), mode="edge")
        cr, cz = _sample_chunk(r, z, jnp.asarray(zs), jnp.asarray(md),
                               *args, subsample=int(subsample))
        parts_r.append(cr[: hi - lo].astype(dtype))
        parts_z.append(cz[: hi - lo].astype(dtype))

    if len(parts_r) == 1:
        return parts_r[0], parts_z[0]
    return jnp.concatenate(parts_r, axis=0), jnp.concatenate(parts_z, axis=0)
