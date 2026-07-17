# -*- coding: utf-8 -*-
"""Batched block-tridiagonal direct solver (block Cholesky / Thomas), JAX.

Why: the MG-PCG path needs 100-250 iterations per solve on these grids (weak
Jacobi smoothing on graded axisymmetric stencils) and the whole vmap batch
waits for its worst member — measured 33-61 ms/solve. A direct factorization
has a FIXED operation count: every batch member marches in lockstep, and the
per-step primitive (batched Cholesky / GEMM on nr x nr blocks) is exactly what
GPUs saturate with large batches. FLOP count: nz blocks x ~3*nr^3 x batch
~ 1 TFLOP per full model — tens of milliseconds of fp32 compute on an A6000.

Structure: on the rectilinear (r, z) grid the FV operator is block
tridiagonal in z — diagonal blocks D_j are TRIDIAGONAL in r (radial
couplings), off-diagonal blocks are DIAGONAL (vertical couplings c_j).
Block-Thomas with Cholesky factors:

    S_0 = D_0
    S_j = D_j - diag(c_{j-1}) S_{j-1}^{-1} diag(c_{j-1})     (dense nr x nr)
    L_j = chol(S_j)                                          (stored)

    forward:  g_j = b_j - c_{j-1} * S_{j-1}^{-1} g_{j-1}
    backward: x_{nz-1} = S^{-1} g;  x_j = S_j^{-1} (g_j - c_j * x_{j+1})

Dirichlet u=0 rows are eliminated SYMMETRICALLY (row+column zeroed, unit
diagonal, neighbour diagonals keep the coupling weight) — identical solution
to the identity-row convention of operator_fv.apply_A with zero RHS.

fp32 robustness: the system is Jacobi-scaled (unit diagonal) before
factorization, and an optional iterative-refinement step recomputes the
residual with the matrix-free apply_A and re-solves — one step recovers ~all
of the fp32 loss for these condition numbers (validated against the MG path).

Memory: stored factors are (nz, B, nr, nr) — e.g. 341 x 256 x 103^2 fp32
= 3.7 GB. Batch in depth chunks accordingly.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from . import operator_fv


def build_bands(sigma, geom):
    """FV operator as symmetric block-tridiagonal bands, Jacobi-scaled.

    sigma : (nz-1, nr-1) cell array or (sigma_r, sigma_z) pair.
    geom  : operator_fv.build_geometry dict (with 'dirichlet' mask).

    Returns dict with unit-diagonal scaled bands and the scaling vector:
      a_r   : (nz, nr-1) radial couplings  (j, i)-(j, i+1), Dirichlet-zeroed
      a_z   : (nz-1, nr) vertical couplings (j, i)-(j+1, i), Dirichlet-zeroed
      s     : (nz, nr) Jacobi scale = 1/sqrt(diag)
    The scaled system solves  (S A S) (S^{-1} u) = S b  with unit diagonal.
    """
    Tr, Tz = operator_fv.face_conductances(sigma, geom)
    mask = geom["dirichlet"]                      # (nz, nr) True on boundary

    # Unscaled diagonal: sum of ALL incident conductances (the coupling to a
    # Dirichlet neighbour stays on the diagonal — symmetric elimination).
    diag = jnp.zeros(mask.shape, dtype=Tr.dtype)
    diag = diag.at[:, :-1].add(Tr)
    diag = diag.at[:, 1:].add(Tr)
    diag = diag.at[:-1, :].add(Tz)
    diag = diag.at[1:, :].add(Tz)
    diag = jnp.where(mask, 1.0, diag)

    # Off-diagonals zeroed where either endpoint is Dirichlet.
    a_r = -Tr * (~mask[:, :-1] & ~mask[:, 1:])
    a_z = -Tz * (~mask[:-1, :] & ~mask[1:, :])

    s = 1.0 / jnp.sqrt(diag)
    a_r = a_r * s[:, :-1] * s[:, 1:]
    a_z = a_z * s[:-1, :] * s[1:, :]
    return {"a_r": a_r, "a_z": a_z, "s": s}


def _dense_D(a_r_row):
    """Tridiagonal D_j (unit diagonal) as dense (..., nr, nr)."""
    nr = a_r_row.shape[-1] + 1
    D = jnp.zeros(a_r_row.shape[:-1] + (nr, nr), dtype=a_r_row.dtype)
    idx = jnp.arange(nr)
    D = D.at[..., idx, idx].set(1.0)
    i = jnp.arange(nr - 1)
    D = D.at[..., i, i + 1].set(a_r_row)
    D = D.at[..., i + 1, i].set(a_r_row)
    return D


def factorize(bands):
    """Forward block elimination; returns per-row Cholesky factors.

    bands arrays may carry a leading batch axis B:
      a_r (B, nz, nr-1), a_z (B, nz-1, nr) -> Ls (nz, B, nr, nr).
    """
    a_r = bands["a_r"]
    a_z = bands["a_z"]
    batched = a_r.ndim == 3
    if not batched:
        a_r = a_r[None]
        a_z = a_z[None]
    B, nz, nrm1 = a_r.shape
    nr = nrm1 + 1

    a_r_j = jnp.moveaxis(a_r, 1, 0)               # (nz, B, nr-1)
    a_z_j = jnp.moveaxis(a_z, 1, 0)               # (nz-1, B, nr)
    # prepend a zero coupling row so step j consumes c_{j-1}
    c_prev = jnp.concatenate(
        [jnp.zeros((1, B, nr), a_z.dtype), a_z_j], axis=0)   # (nz, B, nr)

    def step(carry, inp):
        L_prev = carry                            # (B, nr, nr)
        a_r_row, c = inp                          # (B, nr-1), (B, nr)
        D = _dense_D(a_r_row)                     # (B, nr, nr)
        # W = S_{j-1}^{-1} diag(c):  solve L L^T W = diag(c)
        rhs = c[:, None, :] * jnp.eye(nr, dtype=c.dtype)[None]   # diag(c)
        W = jax.scipy.linalg.cho_solve((L_prev, True), rhs)
        S = D - c[:, :, None] * W                 # diag(c) @ W
        L = jnp.linalg.cholesky(S)
        return L, L

    L0 = jnp.tile(jnp.eye(nr, dtype=a_r.dtype)[None], (B, 1, 1))
    _, Ls = jax.lax.scan(step, L0, (a_r_j, c_prev))
    return Ls                                     # (nz, B, nr, nr)


def solve_factored(Ls, bands, b):
    """Block-Thomas solve for one RHS field b (B, nz, nr) -> u (B, nz, nr).

    b is the PHYSICAL rhs; scaling (s) is applied internally, the returned u
    is physical. Factors Ls come from factorize(bands).
    """
    s = bands["s"]
    a_z = bands["a_z"]
    if b.ndim == 2:
        b = b[None]
    B = b.shape[0]
    bs = b * (s if s.ndim == 3 else s[None])      # scaled rhs
    bs_j = jnp.moveaxis(bs, 1, 0)                 # (nz, B, nr)
    az = a_z if a_z.ndim == 3 else a_z[None].repeat(B, 0)
    az_j = jnp.moveaxis(az, 1, 0)                 # (nz-1, B, nr)
    nr = bs.shape[-1]
    c_prev = jnp.concatenate(
        [jnp.zeros((1, B, nr), bs.dtype), az_j], axis=0)

    # forward: g_j = b_j - c_{j-1} * S_{j-1}^{-1} g_{j-1}
    def fwd(carry, inp):
        g_prev_solved = carry                     # S_{j-1}^{-1} g_{j-1}
        L, bj, c = inp
        g = bj - c * g_prev_solved
        g_solved = jax.scipy.linalg.cho_solve(
            (L, True), g[..., None])[..., 0]
        return g_solved, (g, g_solved)

    z0 = jnp.zeros((B, nr), bs.dtype)
    _, (gs, gs_solved) = jax.lax.scan(fwd, z0, (Ls, bs_j, c_prev))

    # backward: x_j = S_j^{-1} (g_j - c_j * x_{j+1})
    c_next = jnp.concatenate(
        [az_j, jnp.zeros((1, B, nr), bs.dtype)], axis=0)      # c_j (j -> j+1)

    def bwd(carry, inp):
        x_next = carry
        L, g, c = inp
        x = jax.scipy.linalg.cho_solve(
            (L, True), (g - c * x_next)[..., None])[..., 0]
        return x, x

    _, xs_rev = jax.lax.scan(bwd, z0, (Ls[::-1], gs[::-1], c_next[::-1]))
    x = jnp.moveaxis(xs_rev[::-1], 0, 1)          # (B, nz, nr) scaled
    return x * (s if s.ndim == 3 else s[None])    # unscale


def pad_bands(bands, rhs, nz_pad, nr_pad):
    """Pad one tool's scaled bands + rhs to a common (nz_pad, nr_pad) shape.

    Padding rows/columns are decoupled identity equations (unit diagonal, no
    couplings, zero rhs): they solve to zero and do not perturb the real
    unknowns, so systems from different tool grids can share one batched
    Thomas scan. bands from build_bands (unbatched); rhs (nz, nr) physical.
    """
    a_r, a_z, s = bands["a_r"], bands["a_z"], bands["s"]
    nz, nr = s.shape
    dz, dr = nz_pad - nz, nr_pad - nr
    a_r = jnp.pad(a_r, ((0, dz), (0, dr)))        # extra couplings = 0
    a_z = jnp.pad(a_z, ((0, dz), (0, dr)))
    s = jnp.pad(s, ((0, dz), (0, dr)), constant_values=1.0)
    rhs = jnp.pad(rhs, ((0, dz), (0, dr)))
    return {"a_r": a_r, "a_z": a_z, "s": s}, rhs


def build_model_batch(problems):
    """Concatenate every (tool, depth) system of a model into ONE batch.

    Tools live on different canonical grids, so their scaled bands + rhs are
    padded to the common max shape with decoupled identity rows/columns
    (unit diagonal, zero couplings, zero rhs) that solve to zero without
    perturbing the real unknowns. Host-side; returns (bands, b, layout).
    """
    nz_pad = max(len(p["z_rel"]) for p in problems)
    nr_pad = max(len(p["r_nodes"]) for p in problems)
    a_r, a_z, s, bs = [], [], [], []
    layout = []
    for p in problems:
        geom = p["geom"]
        sig_r, sig_z = p["sigmas"]          # level-0 pair (mg or jacobi build)
        B_t = sig_r.shape[0]
        bt = jax.vmap(lambda sr, sz: build_bands((sr, sz), geom))(sig_r, sig_z)
        nz, nr = len(p["z_rel"]), len(p["r_nodes"])
        dz, dr = nz_pad - nz, nr_pad - nr
        a_r.append(jnp.pad(bt["a_r"], ((0, 0), (0, dz), (0, dr))))
        a_z.append(jnp.pad(bt["a_z"], ((0, 0), (0, dz), (0, dr))))
        s.append(jnp.pad(bt["s"], ((0, 0), (0, dz), (0, dr)),
                         constant_values=1.0))
        rhs_t = jnp.broadcast_to(p["rhs"], (B_t,) + p["rhs"].shape)
        bs.append(jnp.pad(rhs_t, ((0, 0), (0, dz), (0, dr))))
        layout.append((B_t, nz, nr))
    bands = {"a_r": jnp.concatenate(a_r), "a_z": jnp.concatenate(a_z),
             "s": jnp.concatenate(s)}
    return bands, jnp.concatenate(bs), layout


@jax.jit
def _thomas_chunk(bands, b):
    Ls = factorize(bands)
    return solve_factored(Ls, bands, b)


def solve_model_batch(bands, b, layout, chunk=256):
    """Chunked whole-model Thomas solve -> list of per-tool u (B_t, nz, nr).

    chunk bounds the stored factors ((nz_pad, chunk, nr_pad^2) fp32); each
    chunk is one scan pass, so prefer the largest chunk that fits memory —
    the per-step latency is nearly flat in batch size.
    """
    B = b.shape[0]
    outs = []
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        pad = chunk - (hi - lo)
        cb = {k: (jnp.concatenate([v[lo:hi], v[hi - 1:hi].repeat(pad, 0)])
                  if pad else v[lo:hi]) for k, v in bands.items()}
        rb = (jnp.concatenate([b[lo:hi], b[hi - 1:hi].repeat(pad, 0)])
              if pad else b[lo:hi])
        outs.append(np.asarray(_thomas_chunk(cb, rb)[: hi - lo]))
    u = np.concatenate(outs)
    res, lo = [], 0
    for B_t, nz, nr in layout:
        res.append(u[lo:lo + B_t, :nz, :nr])
        lo += B_t
    return res


def make_direct_solver(geom, refine=1):
    """Jitted batched direct solve: (sigma_batch, rhs_stack) -> u.

    sigma_batch : (B, nz-1, nr-1) or pair of such — one system per depth.
    rhs_stack   : (T, nz, nr) shared point-source RHS per tool.
    Returns u   : (B, T, nz, nr).

    refine iterative-refinement steps recompute the residual with the
    matrix-free fp32 operator and re-solve with the stored factors — cheap
    (two extra triangular sweeps) and recovers fp32 factorization loss.
    """
    mask = geom["dirichlet"]

    @jax.jit
    def run(sigma_batch, rhs_stack):
        bands = jax.vmap(lambda s: build_bands(s, geom))(sigma_batch)
        Ls = factorize(bands)

        Tr, Tz = jax.vmap(
            lambda s: operator_fv.face_conductances(s, geom))(sigma_batch)

        def solve_one_tool(rhs):
            b = jnp.broadcast_to(rhs, (Ls.shape[1],) + rhs.shape)
            u = solve_factored(Ls, bands, b)
            for _ in range(refine):
                r = b - jax.vmap(
                    lambda uu, tr, tz: operator_fv.apply_A(uu, tr, tz, mask)
                )(u, Tr, Tz)
                u = u + solve_factored(Ls, bands, r)
            return u

        u = jax.vmap(solve_one_tool, out_axes=1)(rhs_stack)   # (B, T, nz, nr)
        return u

    return run
