# -*- coding: utf-8 -*-
"""Matrix-free PCG solve for both discretization backends (JAX).

`solve_one(sigma, rhs, geom, backend=...)` solves A u = rhs with Jacobi-
preconditioned CG from jax.scipy. The geometry dict `geom` comes from the
backend's build_geometry (shared across a batch on the same grid); `sigma`
and `rhs` are per-solve. Everything is functional, so batching is

    solve_batch = jax.vmap(lambda s, b: solve_one(s, b, geom, backend), ...)

Phase 1 uses plain CG + Jacobi (correctness first); a geometric-multigrid
preconditioner is a Phase 3 upgrade behind the same interface.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from . import operator_fv
from . import operator_fem


def _fv_operator(sigma, geom):
    Tr, Tz = operator_fv.face_conductances(sigma, geom)
    mask = geom["dirichlet"]
    return (lambda u: operator_fv.apply_A(u, Tr, Tz, mask),
            operator_fv.diagonal(Tr, Tz, mask))


def _fem_operator(sigma, geom):
    return (lambda u: operator_fem.apply_A(u, sigma, geom),
            operator_fem.diagonal(sigma, geom))


_BACKENDS = {"fv": _fv_operator, "fem": _fem_operator}


@partial(jax.jit, static_argnames=("backend", "tol", "maxiter"))
def solve_one(sigma, rhs, geom, backend="fv", tol=1e-8, maxiter=20000):
    """Solve A(sigma) u = rhs; returns u with the same shape as rhs.

    sigma : (nz-1, nr-1) cell conductivities
    rhs   : (nz, nr) load vector (see operator_fv.point_source_rhs)
    geom  : backend-matching dict from build_geometry(r_nodes, z_nodes)
    """
    apply_A, diag = _BACKENDS[backend](sigma, geom)
    inv_diag = 1.0 / diag
    u, _ = cg(apply_A, rhs, tol=tol, maxiter=maxiter,
              M=lambda v: inv_diag * v)
    return u


def solve_batch(sigmas, rhss, geom, backend="fv", tol=1e-8, maxiter=20000):
    """vmap of solve_one over leading batch axis of (sigmas, rhss)."""
    fn = lambda s, b: solve_one(s, b, geom, backend=backend,
                                tol=tol, maxiter=maxiter)
    return jax.vmap(fn)(sigmas, rhss)


def make_shared_mg_solver(hier, tol=1e-8, maxiter=300, nu=2, coarse_iters=60):
    """MG-PCG solver for the shared-grid multi-tool batch.

    The measurement-point-centred grid makes sigma per DEPTH (shared by all
    tools); tools differ only in their RHS (source node). The returned
    callable takes (sigmas_l, rhs_stack) where sigmas_l is a tuple of
    per-level (sigma_r, sigma_z) pairs with a leading depth axis and
    rhs_stack is (n_tools, nz, nr), and returns (u, iters) with u of shape
    (n_depths, n_tools, nz, nr).

    Nested vmap: outer over depths (sigma batched, rhs shared), inner over
    tools (rhs batched, sigma and the V-cycle preconditioner shared) — the
    preconditioner is built once per depth and serves every tool.
    """
    from . import mg as _mg

    def _one_depth(sigmas_l, rhs_stack):
        apply_A, _ = _mg._level_ops(hier["backend"],
                                    hier["levels"][0]["geom"], sigmas_l[0])
        M = _mg.vcycle_preconditioner(hier, sigmas_l, nu=nu,
                                      coarse_iters=coarse_iters)
        return jax.vmap(lambda b: _mg.pcg(apply_A, b, M, tol=tol,
                                          maxiter=maxiter))(rhs_stack)

    run = jax.vmap(_one_depth, in_axes=(0, None))
    return jax.jit(lambda sigmas_l, rhs_stack: run(tuple(sigmas_l), rhs_stack))


def make_mg_solver(hier, tol=1e-8, maxiter=300, nu=2, coarse_iters=60,
                   batched=True):
    """Build a jitted MG-preconditioned CG solver bound to a hierarchy.

    hier comes from mg.build_hierarchy (per tool/grid, sigma-independent).
    The returned callable takes (sigmas_l, rhs) where sigmas_l is a tuple of
    per-level cell-sigma arrays (leading batch axis when batched=True, shared
    rhs) and returns (u, n_iterations).

    jit is applied at this level so the per-call closures (V-cycle, operator)
    are traced once per shape; the hierarchy itself is baked into the trace.
    """
    from . import mg as _mg

    def _run_one(sigmas_l, rhs):
        apply_A, _ = _mg._level_ops(hier["backend"],
                                    hier["levels"][0]["geom"], sigmas_l[0])
        M = _mg.vcycle_preconditioner(hier, sigmas_l, nu=nu,
                                      coarse_iters=coarse_iters)
        return _mg.pcg(apply_A, rhs, M, tol=tol, maxiter=maxiter)

    if batched:
        run = jax.vmap(_run_one, in_axes=(0, None))
    else:
        run = _run_one
    return jax.jit(lambda sigmas_l, rhs: run(tuple(sigmas_l), rhs))
