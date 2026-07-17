# -*- coding: utf-8 -*-
"""Vertex-centered finite-volume (box method) operator, matrix-free, JAX.

Discretizes the axisymmetric DC problem

    -div( 2*pi*r * sigma(r,z) * grad u ) = sum_l s_l * delta(r=0, z=z_l)

on a rectilinear graded (r, z) grid. Unknowns live on grid NODES; sigma is
piecewise constant per CELL (see grid.sample_sigma). The control volume of a
node is bounded by cell midlines, so every CV face crosses cell interiors
where sigma is constant:

  * radial face between nodes (i, j) and (i+1, j): area 2*pi*r_{i+1/2}*dz,
    split at z_j into the two adjacent cells -> conductance
        Tr[j, i] = 2*pi * r_{i+1/2} / dr_i * (sigma[j-1, i]*dz_{j-1}/2
                                              + sigma[j, i]*dz_j/2)
  * vertical face between nodes (i, j) and (i, j+1): annulus split at r_i ->
        Tz[j, i] = pi / dz_j * (sigma[j, i-1]*(r_i^2 - r_{i-1/2}^2)
                                + sigma[j, i]*(r_{i+1/2}^2 - r_i^2))

Boundary halves are dropped naturally (axis face has zero area -> the r=0
Neumann condition is automatic). The 2*pi factor lives in the operator, like
the 2*pi*r weight in the NGSolve bilinear form (ngsolve_functions.py:150), so
a unit point source equals unit total current.

Dirichlet u=0 on the outer boundary (r=R, z=z_min, z=z_max) is enforced by an
identity row mask: apply_A returns u at masked nodes, and the RHS is zeroed
there. This keeps the operator SPD on the free subspace, which CG requires.

sigma enters the face conductances linearly, and the geometry factors are
shared across a batch of solves on the same grid — build_geometry() once,
face_conductances() per sigma.
"""

import numpy as np
import jax.numpy as jnp


def build_geometry(r_nodes, z_nodes):
    """Precompute sigma-independent geometry factors (host side, numpy).

    Returns a dict of arrays ready to combine with cell sigma:
      gr_lo, gr_hi : (nz, nr-1)  radial-face weights for sigma[j-1, i] / sigma[j, i]
      gz_lo, gz_hi : (nz-1, nr)  vertical-face weights for sigma[j, i-1] / sigma[j, i]
      dirichlet    : (nz, nr) bool, True on outer Dirichlet nodes
    """
    r = np.asarray(r_nodes, dtype=float)
    z = np.asarray(z_nodes, dtype=float)
    nr, nz = len(r), len(z)
    dr = np.diff(r)                      # (nr-1,)
    dz = np.diff(z)                      # (nz-1,)
    r_mid = 0.5 * (r[:-1] + r[1:])       # (nr-1,) face radii r_{i+1/2}

    # ---- radial faces: between node columns i, i+1 at node row j ----
    # face segment below z_j lies in cell (j-1, i), above in cell (j, i)
    base = 2.0 * np.pi * r_mid[None, :] / dr[None, :]          # (1, nr-1)
    seg_lo = np.concatenate([[0.0], dz]) * 0.5                 # (nz,) lower half-height
    seg_hi = np.concatenate([dz, [0.0]]) * 0.5                 # (nz,) upper half-height
    gr_lo = base * seg_lo[:, None]                             # weight of sigma[j-1, i]
    gr_hi = base * seg_hi[:, None]                             # weight of sigma[j, i]

    # ---- vertical faces: between node rows j, j+1 at node column i ----
    # annulus [r_{i-1/2}, r_i] in cell (j, i-1), [r_i, r_{i+1/2}] in cell (j, i)
    a_in = np.concatenate([[0.0], np.pi * (r[1:-1] ** 2 - r_mid[:-1] ** 2), [np.pi * (r[-1] ** 2 - r_mid[-1] ** 2)]])
    a_out = np.concatenate([[np.pi * r_mid[0] ** 2], np.pi * (r_mid[1:] ** 2 - r[1:-1] ** 2), [0.0]])
    gz_lo = a_in[None, :] / dz[:, None]                        # weight of sigma[j, i-1]
    gz_hi = a_out[None, :] / dz[:, None]                       # weight of sigma[j, i]

    dirichlet = np.zeros((nz, nr), dtype=bool)
    dirichlet[0, :] = True
    dirichlet[-1, :] = True
    dirichlet[:, -1] = True

    return {
        "gr_lo": gr_lo, "gr_hi": gr_hi,
        "gz_lo": gz_lo, "gz_hi": gz_hi,
        "dirichlet": dirichlet,
    }


def face_conductances(sigma, geom):
    """Combine cell sigma with geometry -> (Tr, Tz).

    sigma is either one (nz-1, nr-1) array used for both directions, or a
    (sigma_r, sigma_z) pair from grid.sample_sigma_aniso: radial faces use
    sigma_r, vertical faces use sigma_z (transmissibility upscaling for cells
    crossed by material boundaries).

    Pure jnp; differentiable and vmappable in sigma.
      Tr : (nz, nr-1)  radial-face conductances
      Tz : (nz-1, nr)  vertical-face conductances
    """
    if isinstance(sigma, (tuple, list)):
        sigma_r, sigma_z = (jnp.asarray(s) for s in sigma)
    else:
        sigma_r = sigma_z = jnp.asarray(sigma)
    nzc, nrc = sigma_r.shape                   # cells: nz-1, nr-1
    # pad sigma with a zero row below/above for the j-1 / j cell lookups
    pad = jnp.zeros((1, nrc), dtype=sigma_r.dtype)
    s_below = jnp.concatenate([pad, sigma_r], axis=0)   # sigma[j-1, i] at node row j
    s_above = jnp.concatenate([sigma_r, pad], axis=0)   # sigma[j, i]   at node row j
    Tr = geom["gr_lo"] * s_below + geom["gr_hi"] * s_above

    padc = jnp.zeros((nzc, 1), dtype=sigma_z.dtype)
    s_left = jnp.concatenate([padc, sigma_z], axis=1)   # sigma[j, i-1] at node col i
    s_right = jnp.concatenate([sigma_z, padc], axis=1)  # sigma[j, i]   at node col i
    Tz = geom["gz_lo"] * s_left + geom["gz_hi"] * s_right
    return Tr, Tz


def apply_A(u, Tr, Tz, dirichlet):
    """Matrix-free A @ u for the 5-point FV stencil with Dirichlet identity rows.

    u : (nz, nr); returns same shape. Pure jnp, vmappable over (u, Tr, Tz).
    """
    fr = Tr * (u[:, :-1] - u[:, 1:])          # (nz, nr-1) flux i -> i+1
    fz = Tz * (u[:-1, :] - u[1:, :])          # (nz-1, nr) flux j -> j+1
    out = jnp.zeros_like(u)
    out = out.at[:, :-1].add(fr)
    out = out.at[:, 1:].add(-fr)
    out = out.at[:-1, :].add(fz)
    out = out.at[1:, :].add(-fz)
    return jnp.where(dirichlet, u, out)


def diagonal(Tr, Tz, dirichlet):
    """Diagonal of A (for Jacobi preconditioning); 1.0 on Dirichlet rows."""
    d = jnp.zeros((Tz.shape[0] + 1, Tr.shape[1] + 1), dtype=Tr.dtype)
    d = d.at[:, :-1].add(Tr)
    d = d.at[:, 1:].add(Tr)
    d = d.at[:-1, :].add(Tz)
    d = d.at[1:, :].add(Tz)
    return jnp.where(dirichlet, 1.0, d)


def point_source_rhs(shape, source_nodes, dirichlet, dtype=jnp.float64):
    """RHS with unit-current nodal deltas.

    source_nodes : sequence of ((j, i), weight) — node indices and s_l in {+1,-1}.
    The delta weight equals the injected current, matching AddPointSource with
    the 2*pi factor kept in the operator.
    """
    b = jnp.zeros(shape, dtype=dtype)
    for (j, i), w in source_nodes:
        b = b.at[j, i].add(w)
    return jnp.where(dirichlet, 0.0, b)
