# -*- coding: utf-8 -*-
"""GPU global factor-once/solve-many block-Thomas (Phase G1).

The whole per-sample pipeline runs on the device inside one jitted function:

  sigma (z-varying mud, jnp)  ->  face conductances (operator_fv, jnp)
  ->  block-tridiagonal rows (d_main, d_sub, c)
  ->  factor: lax.scan over z-rows, dense Schur Cholesky per row
  ->  solve:  forward + reverse lax.scan with all RHS columns at once
  ->  axis potentials (n_rows, k) -> host Ra extraction.

Identical math to global_op.factor_block_thomas / solve_block_thomas (the
validated CPU control); the batch axis over SAMPLES is a jax.vmap — the grid
(and hence the source-column table) is sample-independent, so a batch shares
one compiled executable and one RHS table.

Memory note: the forward-sweep stack W is (n_rows, m, k) — ~5.8 GB in fp64
for the len512 shared grid at k=544 — so the sample batch B is bounded by
device memory; B=2 (fp64) / B=4 (fp32) fit comfortably on a 48 GB A6000.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, cho_solve

from . import operator_fv
from .sigma_gpu import _sigma_at_points_jnp


# ---------------------------------------------------------------- sigma ----

def _sigma_zmud_jnp(r_nodes, z_nodes, form_cols, bh_cols, subsample):
    """jnp twin of global_op.sample_sigma_aniso_zmud (z-varying mud)."""
    tops0, bots, fzr, fzv, uzv = form_cols
    bh_z, bh_r, bh_rm = bh_cols
    r, z, s = r_nodes, z_nodes, subsample

    frac = (jnp.arange(s) + 0.5) / s
    z_sub = z[:-1, None] + jnp.diff(z)[:, None] * frac[None, :]
    r_sub = r[:-1, None] + jnp.diff(r)[:, None] * frac[None, :]
    wr = r_sub / jnp.maximum(r_sub.sum(axis=1, keepdims=True), 1e-300)

    R = r_sub[None, None, :, :]                      # (1, 1, nrc, s)
    Z = z_sub[:, :, None, None]                      # (nzc, s, 1, 1)
    mud = jnp.interp(Z, bh_z, bh_rm)
    sig = _sigma_at_points_jnp(R, Z, tops0, bots, fzr, fzv, uzv,
                               bh_z, bh_r, mud)      # (nzc, s, nrc, s)

    harm_z = 1.0 / jnp.mean(1.0 / sig, axis=1)
    sigma_z = (harm_z * wr[None, :, :]).sum(axis=2)

    inv_r = 1.0 / jnp.maximum(jnp.broadcast_to(R, sig.shape), 1e-300)
    w_res = inv_r / inv_r.sum(axis=3, keepdims=True)
    harm_r = 1.0 / (w_res / sig).sum(axis=3)
    sigma_r = harm_r.mean(axis=1)
    return sigma_r, sigma_z


# ----------------------------------------------------- factor and solve ----

def _factor_scan(d_main, d_sub, c_pad, out_dtype=None):
    """L stack (n_rows, m, m) of the per-row Schur Cholesky factors (lower).

    Carry = S_j; step j emits L_j = chol(S_j) and builds
    S_{j+1} = D_{j+1} - diag(c_j) S_j^{-1} diag(c_j)
            = D_{j+1} - W_j^T W_j,   W_j = L_j^{-1} diag(c_j).
    The final step consumes a dummy D (identity) and zero coupling.

    The RECURSION must run in fp64: over ~12k rows the Schur complement's
    small eigenvalues sink below fp32 epsilon and Cholesky NaNs (measured:
    first NaN at row ~2153 even with Jacobi scaling). The *emitted* factors
    may be cast to ``out_dtype`` (fp32) — downstream triangular solves with
    accurate factors are stable (Ra error ~4e-5, measured).
    """
    m = d_main.shape[1]

    def D(d, ds):
        return (jnp.diag(d) + jnp.diag(ds, 1) + jnp.diag(ds, -1))

    d_next = jnp.concatenate([d_main[1:], jnp.ones((1, m), d_main.dtype)])
    ds_next = jnp.concatenate([d_sub[1:], jnp.zeros((1, m - 1), d_sub.dtype)])

    def body(S, inp):
        d, ds, cj = inp
        L = jnp.linalg.cholesky(S)
        W = solve_triangular(L, jnp.diag(cj), lower=True)
        S_next = D(d, ds) - W.T @ W
        return S_next, (L.astype(out_dtype) if out_dtype is not None else L)

    _, Ls = jax.lax.scan(body, D(d_main[0], d_sub[0]),
                         (d_next, ds_next, c_pad))
    return Ls


def _solve_scan(Ls, c_pad, b_axis, full=False):
    """Solve for k RHS that are unit axis sources.

    b_axis : (n_rows, k) one-hot rows (source value at radial index 0).
    Forward sweep stores w_j = S_j^{-1} v_j (the (n_rows, m, k) stack);
    reverse sweep emits the axis (i=0) row of x_j -> (n_rows, k), or the
    full solution (n_rows, m, k) when ``full=True`` (needed by the
    CPU-convention refinement loop).
    """
    def fwd(w_prev, inp):
        L, b_row, c_prev = inp
        v = (-c_prev[:, None] * w_prev).at[0].add(b_row)
        w = cho_solve((L, True), v)
        return w, w

    m = Ls.shape[1]
    k = b_axis.shape[1]
    c_prev = jnp.concatenate([jnp.zeros((1, m), c_pad.dtype), c_pad[:-1]])
    w0 = jnp.zeros((m, k), Ls.dtype)
    _, W = jax.lax.scan(fwd, w0, (Ls, b_axis, c_prev))

    def bwd(x_next, inp):
        L, w, cj = inp
        x = w - cho_solve((L, True), cj[:, None] * x_next)
        return x, (x if full else x[0])

    _, X = jax.lax.scan(bwd, jnp.zeros((m, k), Ls.dtype),
                        (Ls, W, c_pad), reverse=True)
    return X


def _solve_scan_dense(Ls, c_pad, B):
    """Same two sweeps for a DENSE RHS block B (n_rows, m, k), full output."""
    def fwd(w_prev, inp):
        L, b, c_prev = inp
        w = cho_solve((L, True), b - c_prev[:, None] * w_prev)
        return w, w

    m = Ls.shape[1]
    c_prev = jnp.concatenate([jnp.zeros((1, m), c_pad.dtype), c_pad[:-1]])
    w0 = jnp.zeros(B.shape[1:], Ls.dtype)
    _, W = jax.lax.scan(fwd, w0, (Ls, B, c_prev))

    def bwd(x_next, inp):
        L, w, cj = inp
        x = w - cho_solve((L, True), cj[:, None] * x_next)
        return x, x

    _, X = jax.lax.scan(bwd, jnp.zeros(B.shape[1:], Ls.dtype),
                        (Ls, W, c_pad), reverse=True)
    return X


def _apply_block(dm, ds, cc, X):
    """Block-tridiagonal apply A X — elementwise, no scan.

    dm (n_rows, m) diagonal, ds (n_rows, m-1) radial off-diagonal,
    cc (n_rows-1, m) vertical coupling; X (n_rows, m, k).
    """
    Y = dm[:, :, None] * X
    Y = Y.at[:, 1:, :].add(ds[:, :, None] * X[:, :-1, :])
    Y = Y.at[:, :-1, :].add(ds[:, :, None] * X[:, 1:, :])
    Y = Y.at[1:, :, :].add(cc[:, :, None] * X[:-1, :, :])
    Y = Y.at[:-1, :, :].add(cc[:, :, None] * X[1:, :, :])
    return Y


# ----------------------------------------------------------- the solver ----

def make_solver(problem, precision="mixed", mud="log", n_refine=3,
                subsample=4, k_pad_to=16, chunk_cols=None):
    """Compile the per-sample-batch pipeline for a fixed grid/task table.

    problem : dict from global_op.build_global_tasks (sample-independent).
    precision:
      "mixed" (default) — σ/assembly/factor recursion in fp64, emitted
        factors and both solve sweeps in fp32 (Ra error ~4e-5; ~4x faster
        solves, half the factor-stack memory);
      "f64" — everything fp64 (validation reference);
      pure fp32 is NOT offered: the factor recursion NaNs (see _factor_scan).
    mud:
      "log" (default) — the physical z-varying RM column (v2 convention);
      "cpu" — CPU-pipeline imitation: each task sees the whole borehole
        filled with the scalar RM(z_sim) of ITS simulation depth. Face
        conductances are linear in the mud conductivity for pure-mud cells,
        so A(mu) = A_ref + (mu - mu_ref) * dA — ONE factorization at mu_ref
        plus ``n_refine`` preconditioned-refinement sweeps per solve handles
        every column's own mu exactly (caliper-crossing cells are linearized
        around mu_ref; |mu - mu_ref| <~ 5% makes the quadratic residue
        negligible and the refinement contraction ~|dmu|/mu per sweep).
        Combine with domain_radius = max(10*span, 5) in build_global_tasks
        to match the CPU boundary convention as well.
    Returns solve(formations, boreholes) -> X0 (B, n_rows, k) device array;
    formations/boreholes are stacked (B, ...) sample arrays.
    """
    if precision not in ("mixed", "f64"):
        raise ValueError("precision must be 'mixed' or 'f64'")
    if mud not in ("log", "cpu"):
        raise ValueError("mud must be 'log' or 'cpu'")
    dtype = jnp.float64
    sol_dtype = jnp.float32 if precision == "mixed" else jnp.float64
    r_nodes = jnp.asarray(problem["r_nodes"], dtype)
    z_nodes = jnp.asarray(problem["z_nodes"], dtype)
    geom = operator_fv.build_geometry(problem["r_nodes"], problem["z_nodes"])
    geom = {kk: (jnp.asarray(v, dtype)
                 if np.issubdtype(np.asarray(v).dtype, np.floating)
                 else jnp.asarray(v)) for kk, v in geom.items()}
    m = problem["bandwidth"]

    uniq = problem["uniq_src"]
    k = len(uniq)
    k_full = -(-k // k_pad_to) * k_pad_to
    rows = uniq // m                     # free z-row of each source (axis i=0)
    cols = np.arange(k)
    b_axis = np.zeros((problem["n_free"] // m, k_full))
    b_axis[rows, cols] = 1.0             # padded tail columns stay zero
    b_axis = jnp.asarray(b_axis, dtype)  # scaled per sample, then cast
    # simulation depth of each source column (mud="cpu": one scalar RM each);
    # padded tail repeats the first z (their b columns are zero anyway)
    z_src = problem["z_nodes"][rows + 1]
    z_src = jnp.asarray(np.pad(z_src, (0, k_full - k), mode="edge"), dtype)

    def blocks(Tr, Tz):
        """Face conductances -> block-tridiagonal row arrays."""
        d_full = jnp.zeros(geom["dirichlet"].shape, dtype)
        d_full = d_full.at[:, :-1].add(Tr).at[:, 1:].add(Tr)
        d_full = d_full.at[:-1, :].add(Tz).at[1:, :].add(Tz)
        d_main = d_full[1:-1, :m]
        d_sub = -Tr[1:-1, :m - 1]
        # couplings j -> j+1 for j = 0..n_rows-2, plus a zero row so the
        # factor scan's final (dummy) step sees no coupling
        c_pad = jnp.concatenate([-Tz[1:-1, :m], jnp.zeros((1, m), dtype)])
        return d_main, d_sub, c_pad

    def one_sample(formation, borehole):
        form_cols = tuple(formation[:, i] for i in range(5))
        bh_z, bh_r, bh_rm = borehole[:, 0], borehole[:, 1], borehole[:, 2]

        if mud == "cpu":
            # Parameterize by mud CONDUCTIVITY nu = 1/RM: face conductances
            # are exactly linear in nu for pure-mud cells (resistivity is
            # not), so the refinement target A(nu) is exact there and only
            # the caliper-crossing cells carry a (tiny) secant residue.
            nu_cols = 1.0 / jnp.interp(z_src, bh_z, bh_rm)   # (k_full,)
            nu_ref = jnp.mean(nu_cols[:k])
            delta = 0.05 * nu_ref

            def sig_at(nu):
                return _sigma_zmud_jnp(r_nodes, z_nodes, form_cols,
                                       (bh_z, bh_r,
                                        jnp.full_like(bh_rm, 1.0 / nu)),
                                       subsample)

            # Quadratic fit through nu_ref, nu_ref±delta: pure-mud cells are
            # exactly linear in nu; the caliper-crossing cells' harmonic
            # mixes are captured to O((dnu/nu)^3) by the parabola (the
            # secant-only version left ~1e-2 Ra residue on short tools).
            sigma = sig_at(nu_ref)
            sigma_p = sig_at(nu_ref + delta)
            sigma_m = sig_at(nu_ref - delta)
        else:
            sigma = _sigma_zmud_jnp(r_nodes, z_nodes, form_cols,
                                    (bh_z, bh_r, bh_rm), subsample)

        Tr, Tz = operator_fv.face_conductances(sigma, geom)
        d_main, d_sub, c_pad = blocks(Tr, Tz)

        # Symmetric Jacobi scaling (unit diagonal): face conductances span
        # ~7 orders of magnitude (mm plateau cells to 10-m far-field cells);
        # scaling conditions the recursion and the fp32 solves — same cure
        # as v1 direct.py. x = s * x' recovers the solution; only the axis
        # column of s is needed for that.
        s = 1.0 / jnp.sqrt(d_main)
        s_next = jnp.concatenate([s[1:], jnp.ones((1, m), dtype)])
        d_main_s = jnp.ones_like(d_main)
        d_sub_s = d_sub * s[:, :-1] * s[:, 1:]
        c_pad_s = c_pad * s * s_next
        b_axis_s = (b_axis * s[:, :1]).astype(sol_dtype)

        Ls = _factor_scan(d_main_s, d_sub_s, c_pad_s,
                          out_dtype=(sol_dtype if sol_dtype != dtype
                                     else None))
        c32 = c_pad_s.astype(sol_dtype)

        if mud == "log":
            X0 = _solve_scan(Ls, c32, b_axis_s)
            return X0 * s[:, :1].astype(sol_dtype)

        # mud="cpu": A(nu) ~ A_ref + dA, dA = dnu*A1 + dnu^2*A2 (nonzero on
        # mud cells only). Solve A x = b as the NEUMANN SERIES on the single
        # nu_ref factorization M:  x = sum_n (-M^-1 dA)^n x_ref.
        # The series never subtracts b - A x, so the singular-source
        # cancellation of a residual-based refinement is gone; but the
        # STENCIL SUM (dA x)_i itself cancels (the discrete divergence of a
        # near-harmonic field is far smaller than its terms), so the dA
        # APPLY must run in fp64 — an all-fp32 series stalls at ~2e-3
        # (measured), fp64 applies + fp32 SOLVES reach the native ~1e-4
        # floor with only two transient fp64 fields per sample.
        # Contraction per term ~ max|dnu|/nu_ref (<~8%).
        Trp, Tzp = operator_fv.face_conductances(sigma_p, geom)
        Trm, Tzm = operator_fv.face_conductances(sigma_m, geom)

        def scaled_blocks(Tr_, Tz_):
            a, b, c = blocks(Tr_, Tz_)
            return (a * s * s, b * s[:, :-1] * s[:, 1:],
                    (c * s * s_next)[:-1])

        A1 = scaled_blocks((Trp - Trm) / (2 * delta),
                           (Tzp - Tzm) / (2 * delta))
        A2 = scaled_blocks((Trp + Trm - 2 * Tr) / (delta * delta) * 0.5,
                           (Tzp + Tzm - 2 * Tz) / (delta * delta) * 0.5)
        dmu = nu_cols - nu_ref
        dmu2 = dmu * dmu

        # chunk_cols bounds the live (n_rows, m, kc) fields — the refinement
        # temporaries are what limit the sample batch B. The chunks MUST run
        # sequentially (lax.map); a python loop leaves them dependency-free
        # and XLA schedules them concurrently, keeping the full peak.
        def series(b_sl, dmu_sl, dmu2_sl):
            X = _solve_scan(Ls, c32, b_sl, full=True)
            T = X
            for _ in range(n_refine):
                T64 = T.astype(dtype)
                DT = (dmu_sl[None, None, :] * _apply_block(*A1, T64)
                      + dmu2_sl[None, None, :] * _apply_block(*A2, T64))
                T = -_solve_scan_dense(Ls, c32, DT.astype(sol_dtype))
                X = X + T
            return X[:, 0, :]

        kc = int(chunk_cols) if chunk_cols else k_full
        n_chunks = -(-k_full // kc)
        if n_chunks == 1:
            X0 = series(b_axis_s, dmu, dmu2)
        else:
            pad = n_chunks * kc - k_full
            b_p = jnp.pad(b_axis_s, ((0, 0), (0, pad)))
            dmu_p = jnp.pad(dmu, (0, pad))
            dmu2_p = jnp.pad(dmu2, (0, pad))
            out = jax.lax.map(
                lambda t: series(*t),
                (jnp.stack(jnp.split(b_p, n_chunks, axis=1)),
                 dmu_p.reshape(n_chunks, kc),
                 dmu2_p.reshape(n_chunks, kc)))
            X0 = out.transpose(1, 0, 2).reshape(b_p.shape[0],
                                                n_chunks * kc)[:, :k_full]
        return X0 * s[:, :1].astype(sol_dtype)

    solve = jax.jit(jax.vmap(one_sample))

    def run(formations, boreholes):
        return solve(jnp.asarray(formations, dtype),
                     jnp.asarray(boreholes, dtype))

    return run


def extract_logs(problem, X0):
    """Host-side Ra extraction from axis potentials.

    X0 : (B, n_rows, k) device/np array. Returns list (len B) of dicts
    tool -> (n_depths,) Ra.
    """
    X0 = np.asarray(X0)
    m = problem["bandwidth"]
    out = []
    for b in range(X0.shape[0]):
        logs = {}
        for t in problem["tools"]:
            tk = problem["tasks"][t]
            dU = (X0[b, tk["M"] // m, tk["col"]]
                  - X0[b, tk["N"] // m, tk["col"]])
            logs[t] = tk["K"] * np.abs(dU)
        out.append(logs)
    return out
