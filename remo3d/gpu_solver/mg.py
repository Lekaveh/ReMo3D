# -*- coding: utf-8 -*-
"""Geometric multigrid preconditioner for the structured-grid solver (JAX).

Jacobi-PCG needs O(10^3) iterations on these grids: the axisymmetric weight
(2*pi*r) plus grading spreads the stencil coefficients over ~4 orders of
magnitude, and the sigma contrast adds more. A V-cycle brings that down to
tens of iterations.

Design choices, tuned to this problem:

  * Coarse levels are INDEPENDENT graded grids built by the same build_grid
    with h_min doubled per level (not injected node subsets). Sigma is
    RE-SAMPLED from the formation/borehole model on every level, so material
    boundaries stay sharp instead of being averaged — the levels see the same
    physics, just coarser.
  * Prolongation P is tensor-product bilinear interpolation between
    rectilinear grids (host-precomputed 1D indices/weights); restriction is
    exactly P^T via jax.linear_transpose, which keeps the V-cycle symmetric.
  * Smoother: damped Jacobi (omega=2/3), nu1=nu2 sweeps; coarsest level is
    solved with a short Jacobi-CG. Symmetric smoothing keeps M SPD, so the
    V-cycle is a valid CG preconditioner.
  * Everything is shape-static per level -> jit/vmap-friendly; the batch axis
    only carries per-level sigma stacks.

Entry points:
    build_hierarchy(...)   -- host-side; returns per-level geometry + P/R.
    sample_level_sigmas(...) -- per-depth sigma stacks for every level.
    vcycle_preconditioner(hier, sigmas_l) -> M(v) callable for solve.pcg.
"""

import numpy as np
import jax
import jax.numpy as jnp

from . import grid as ggrid
from . import operator_fv, operator_fem

_BACKENDS = {"fv": operator_fv, "fem": operator_fem}


# ---------------------------------------------------------------------------
# Host-side hierarchy construction
# ---------------------------------------------------------------------------

def _interp_1d_weights(x_fine, x_coarse):
    """Indices/weights so f(x_fine) = w*fc[i] + (1-w)*fc[i+1] (linear)."""
    idx = np.searchsorted(x_coarse, x_fine, side="right") - 1
    idx = np.clip(idx, 0, len(x_coarse) - 2)
    x0 = x_coarse[idx]
    x1 = x_coarse[idx + 1]
    t = (x_fine - x0) / (x1 - x0)
    t = np.clip(t, 0.0, 1.0)
    return idx.astype(np.int32), (1.0 - t)


def make_prolongation(r_fine, z_fine, r_coarse, z_coarse, dtype=jnp.float64):
    """Bilinear tensor-product interpolation coarse -> fine as a jax fn.

    Weights are cast to ``dtype`` so P preserves the solve dtype (otherwise
    x64-mode numpy weights would silently promote fp32 fields to fp64 and
    break the linear_transpose cotangent dtype)."""
    ir, wr = _interp_1d_weights(r_fine, r_coarse)
    iz, wz = _interp_1d_weights(z_fine, z_coarse)
    ir = jnp.asarray(ir); wr = jnp.asarray(wr, dtype=dtype)
    iz = jnp.asarray(iz); wz = jnp.asarray(wz, dtype=dtype)

    def P(uc):
        # z-direction: (nz_c, nr_c) -> (nz_f, nr_c)
        u1 = wz[:, None] * uc[iz, :] + (1.0 - wz)[:, None] * uc[iz + 1, :]
        # r-direction: (nz_f, nr_c) -> (nz_f, nr_f)
        return wr[None, :] * u1[:, ir] + (1.0 - wr)[None, :] * u1[:, ir + 1]

    return P


def build_hierarchy(domain_radius, elec_rel, r_foci, h_min, growth=1.15,
                    backend="fv", n_levels=None, min_nodes=1500,
                    dtype=jnp.float64, r_fine=0.6):
    """Build the multigrid hierarchy for one tool's depth-relative grid.

    Level 0 is the fine grid (identical to build_grid output the solver uses);
    each next level doubles h_min. Returns a dict with per-level grids,
    geometry factors, prolongations P_l (level l+1 -> l) and their transposes.
    r_foci=None uses the data-independent canonical radial grid (default in the
    driver, for shape-stable compilation caching).
    """
    levels = []
    l = 0
    while True:
        h = h_min * (2.0 ** l)
        r_nodes, z_nodes = ggrid.build_grid(domain_radius, 0.0, elec_rel,
                                            r_foci=r_foci, h_min=h,
                                            growth=growth, r_fine=r_fine)
        geom = _BACKENDS[backend].build_geometry(r_nodes, z_nodes)
        geom = {k: (jnp.asarray(v, dtype=dtype)
                    if np.issubdtype(np.asarray(v).dtype, np.floating)
                    else jnp.asarray(v))
                for k, v in geom.items()}
        levels.append({"r": r_nodes, "z": z_nodes, "geom": geom})
        n = len(r_nodes) * len(z_nodes)
        l += 1
        if n_levels is not None and l >= n_levels:
            break
        if n_levels is None and (n <= min_nodes or l >= 8):
            break

    # Prolongations between consecutive levels (coarse l+1 -> fine l) and
    # their exact transposes for restriction.
    for l in range(len(levels) - 1):
        fine, coarse = levels[l], levels[l + 1]
        P = make_prolongation(fine["r"], fine["z"], coarse["r"], coarse["z"],
                              dtype=dtype)
        shape_c = (len(coarse["z"]), len(coarse["r"]))
        Pt = jax.linear_transpose(P, jnp.zeros(shape_c, dtype=dtype))
        fine["P"] = P
        fine["Pt"] = lambda v, _Pt=Pt: _Pt(v)[0]

    return {"levels": levels, "backend": backend, "dtype": dtype}


def sample_level_sigmas(hier, z_sim, formation, borehole, mud_resistivity,
                        subsample=4):
    """Re-sample anisotropic cell sigma on every level for one depth.

    Returns a list of (sigma_r, sigma_z) pairs (see grid.sample_sigma_aniso).
    """
    out = []
    for lev in hier["levels"]:
        out.append(ggrid.sample_sigma_aniso(
            lev["r"], lev["z"] + z_sim, formation, borehole, mud_resistivity,
            subsample=subsample))
    return out


# ---------------------------------------------------------------------------
# Device-side V-cycle
# ---------------------------------------------------------------------------

def _level_ops(backend, geom, sigma):
    mod = _BACKENDS[backend]
    if backend == "fv":
        Tr, Tz = mod.face_conductances(sigma, geom)
        mask = geom["dirichlet"]
        return (lambda u: mod.apply_A(u, Tr, Tz, mask),
                mod.diagonal(Tr, Tz, mask))
    return (lambda u: mod.apply_A(u, sigma, geom),
            mod.diagonal(sigma, geom))


def _jacobi_smooth(apply_A, inv_diag, b, x, sweeps, omega):
    for _ in range(sweeps):
        x = x + omega * inv_diag * (b - apply_A(x))
    return x


def _coarse_solve(apply_A, inv_diag, b, iters):
    """Fixed-iteration Jacobi-CG on the coarsest level (shape-static)."""
    x = jnp.zeros_like(b)
    r = b
    z = inv_diag * r
    p = z
    rz = jnp.vdot(r, z)

    def body(_, state):
        x, r, p, rz = state
        Ap = apply_A(p)
        denom = jnp.vdot(p, Ap)
        alpha = jnp.where(denom > 0, rz / denom, 0.0)
        x = x + alpha * p
        r = r - alpha * Ap
        z = inv_diag * r
        rz_new = jnp.vdot(r, z)
        beta = jnp.where(rz > 0, rz_new / rz, 0.0)
        p = z + beta * p
        return x, r, p, rz_new

    x, _, _, _ = jax.lax.fori_loop(0, iters, body, (x, r, p, rz))
    return x


def vcycle_preconditioner(hier, sigmas_l, nu=2, omega=2.0 / 3.0,
                          coarse_iters=60):
    """Return M(v) ~= A^{-1} v — one V-cycle from a zero initial guess.

    sigmas_l: list of per-level cell-sigma arrays (device or numpy). Static
    over a solve; the returned callable closes over the per-level operators.
    """
    backend = hier["backend"]
    ops = []
    for lev, sig in zip(hier["levels"], sigmas_l):
        apply_A, diag = _level_ops(backend, lev["geom"], sig)
        ops.append((apply_A, 1.0 / diag))

    n_lev = len(ops)

    def cycle(l, b):
        apply_A, inv_diag = ops[l]
        if l == n_lev - 1:
            return _coarse_solve(apply_A, inv_diag, b, coarse_iters)
        x = _jacobi_smooth(apply_A, inv_diag, b, jnp.zeros_like(b), nu, omega)
        r = b - apply_A(x)
        rc = hier["levels"][l]["Pt"](r)
        # Dirichlet rows must stay identity across levels: zero the coarse
        # residual there so the correction does not leak through the boundary.
        rc = jnp.where(hier["levels"][l + 1]["geom"]["dirichlet"], 0.0, rc)
        ec = cycle(l + 1, rc)
        x = x + hier["levels"][l]["P"](ec)
        x = jnp.where(hier["levels"][l]["geom"]["dirichlet"], b, x)
        return _jacobi_smooth(apply_A, inv_diag, b, x, nu, omega)

    return lambda v: cycle(0, v)


# ---------------------------------------------------------------------------
# MG-preconditioned CG (shape-static, vmap-friendly)
# ---------------------------------------------------------------------------

def pcg(apply_A, b, M, tol=1e-8, maxiter=200):
    """Standard PCG with callable operator/preconditioner.

    Not jitted here: apply_A/M are fresh closures per call, so jitting at this
    level would recompile every time. Trace it inside a jitted caller (see
    solve.make_mg_solver) instead. Runs a shape-static while loop (converged
    batch elements idle until all are done under vmap — same semantics as
    jax.scipy cg).
    """
    x = jnp.zeros_like(b)
    r = b
    z = M(r)
    p = z
    rz = jnp.vdot(r, z)
    bnorm = jnp.linalg.norm(b)
    eps = tol * bnorm

    def cond(state):
        _, r, _, _, k = state
        return jnp.logical_and(jnp.linalg.norm(r) > eps, k < maxiter)

    def body(state):
        x, r, p, rz, k = state
        Ap = apply_A(p)
        denom = jnp.vdot(p, Ap)
        alpha = jnp.where(denom > 0, rz / denom, 0.0)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        rz_new = jnp.vdot(r, z)
        beta = jnp.where(rz > 0, rz_new / rz, 0.0)
        p = z + beta * p
        return x, r, p, rz_new, k + 1

    x, r, _, _, k = jax.lax.while_loop(cond, body, (x, r, p, rz, 0))
    return x, k
