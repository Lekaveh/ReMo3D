# -*- coding: utf-8 -*-
"""Q1 (bilinear) FEM operator on the same rectilinear grid, matrix-free, JAX.

Discretizes the axisymmetric weak form used by NGSolve
(ngsolve_functions.py:150):

    a(u, v) = integral 2*pi*r * sigma(r,z) * grad(u).grad(v) dr dz

with bilinear elements on grid cells and sigma constant per cell. The local
4x4 stiffness of element (j, i) factors as sigma_e * G_e where G_e is a
geometry-only matrix

    G_ab = integral_cell 2*pi*r * grad(N_a).grad(N_b) dr dz

evaluated EXACTLY with 2x2 Gauss quadrature (the integrand is polynomial of
degree <= 3 in each variable). G is precomputed once per grid (shared across a
whole vmap batch); only sigma varies per solve.

Local corner ordering c = 0..3: (j, i), (j, i+1), (j+1, i), (j+1, i+1).

Point sources at grid nodes reduce to nodal deltas (the FEM load is
s_l * N_d(x_l), and N_d at a node is a Kronecker delta) — identical to the FV
RHS, so operator_fv.point_source_rhs is reused. Dirichlet handling matches
operator_fv (identity rows via mask).
"""

import numpy as np
import jax.numpy as jnp

from .operator_fv import point_source_rhs  # noqa: F401  (shared RHS builder)

_GAUSS = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))


def build_geometry(r_nodes, z_nodes):
    """Precompute geometry-only element matrices G (host side, numpy).

    Returns dict:
      G         : (nz-1, nr-1, 4, 4) geometry stiffness (multiply by sigma_e)
      dirichlet : (nz, nr) bool mask of outer Dirichlet nodes
    """
    r = np.asarray(r_nodes, dtype=float)
    z = np.asarray(z_nodes, dtype=float)
    nr, nz = len(r), len(z)
    a = np.diff(r)                                   # (nr-1,) radial widths
    b = np.diff(z)                                   # (nz-1,) axial heights
    A, B = np.meshgrid(a, b)                         # (nz-1, nr-1)
    R0 = np.broadcast_to(r[:-1], A.shape)            # left edge radius per element

    # Split by derivative direction so direction-dependent cell sigma
    # (grid.sample_sigma_aniso) can weight them separately:
    #   K_e = sigma_r * G_rr + sigma_z * G_zz.
    G_rr = np.zeros(A.shape + (4, 4))
    G_zz = np.zeros(A.shape + (4, 4))
    for gx in _GAUSS:                                # xi = gx * a
        for gy in _GAUSS:                            # eta = gy * b
            xi_a = gx                                # xi / a
            eta_b = gy                               # eta / b
            # dN/dxi and dN/deta at this Gauss point, per element (broadcast):
            dNx = np.stack([-(1 - eta_b) / A, (1 - eta_b) / A,
                            -eta_b / A, eta_b / A], axis=-1)      # (..., 4)
            dNy = np.stack([-(1 - xi_a) / B, -xi_a / B,
                            (1 - xi_a) / B, xi_a / B], axis=-1)   # (..., 4)
            rad = R0 + xi_a * A                                    # r at Gauss pt
            w = (A * 0.5) * (B * 0.5)                              # Gauss weight
            fac = (2.0 * np.pi) * rad * w                          # (..., )
            G_rr += fac[..., None, None] * dNx[..., :, None] * dNx[..., None, :]
            G_zz += fac[..., None, None] * dNy[..., :, None] * dNy[..., None, :]

    dirichlet = np.zeros((nz, nr), dtype=bool)
    dirichlet[0, :] = True
    dirichlet[-1, :] = True
    dirichlet[:, -1] = True

    return {"G_rr": G_rr, "G_zz": G_zz, "dirichlet": dirichlet}


def _split_sigma(sigma):
    if isinstance(sigma, (tuple, list)):
        return jnp.asarray(sigma[0]), jnp.asarray(sigma[1])
    s = jnp.asarray(sigma)
    return s, s


def _gather_corners(u):
    """u (nz, nr) -> (nz-1, nr-1, 4) corner values per element."""
    return jnp.stack([u[:-1, :-1], u[:-1, 1:], u[1:, :-1], u[1:, 1:]], axis=-1)


def apply_A(u, sigma, geom):
    """Matrix-free A @ u for the Q1 stencil; identity on Dirichlet nodes.

    u     : (nz, nr)
    sigma : (nz-1, nr-1) cell conductivities, or a (sigma_r, sigma_z) pair
            weighting the radial / vertical stiffness parts separately
    geom  : dict from build_geometry (G_rr, G_zz shared across batch)
    """
    s_r, s_z = _split_sigma(sigma)
    u4 = _gather_corners(u)                               # (..., 4)
    y4 = (s_r[..., None] * jnp.einsum("...ab,...b->...a", geom["G_rr"], u4)
          + s_z[..., None] * jnp.einsum("...ab,...b->...a", geom["G_zz"], u4))
    out = jnp.zeros_like(u)
    out = out.at[:-1, :-1].add(y4[..., 0])
    out = out.at[:-1, 1:].add(y4[..., 1])
    out = out.at[1:, :-1].add(y4[..., 2])
    out = out.at[1:, 1:].add(y4[..., 3])
    return jnp.where(geom["dirichlet"], u, out)


def diagonal(sigma, geom):
    """Diagonal of A (Jacobi preconditioner); 1.0 on Dirichlet rows."""
    s_r, s_z = _split_sigma(sigma)
    Gd = geom["G_rr"]
    Gz = geom["G_zz"]
    d = jnp.zeros(geom["dirichlet"].shape,
                  dtype=jnp.result_type(s_r.dtype, Gd.dtype))
    for c, (jz, ir) in enumerate((( slice(None, -1), slice(None, -1)),
                                  (slice(None, -1), slice(1, None)),
                                  (slice(1, None), slice(None, -1)),
                                  (slice(1, None), slice(1, None)))):
        d = d.at[jz, ir].add(s_r * Gd[..., c, c] + s_z * Gz[..., c, c])
    return jnp.where(geom["dirichlet"], 1.0, d)
