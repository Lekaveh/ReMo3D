# -*- coding: utf-8 -*-
"""Global (whole-interval) operator — the factor-once/solve-many path (v2, E0-E3).

v1 solves one local BVP per (tool, depth): a depth-centred grid, a fresh
batched block-Thomas factorization per solve (see driver.build_tool_problem).
But sigma(r, z) is fixed per pseudowell — the ~1280 solves of a sample differ
only by the source position. This module builds ONE global (r, z) grid
spanning every measurement depth, assembles the same 5-point FV operator
EXPLICITLY, and exposes it as CSR and as an upper LAPACK band (with z-major
numbering the half-bandwidth is nr-1 ~ 110, so a banded Cholesky is the
natural CPU control solver: factor once, solve all sources as RHS columns).

Reuses grid.canonical_radial_nodes and operator_fv.build_geometry (both pure
numpy). The sigma sampler mirrors grid.sample_sigma_aniso with one change:
mud resistivity is interpolated at each sub-point z — the global column spans
the whole RM log, whereas a local solve uses one scalar RM at its simulation
depth.

See GPU_SOLVER_V2_PLAN.md (Phase 0) and wiki sources/deep-research-gpu-solver.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.linalg.lapack import dpotrf, dpotrs

from . import grid as ggrid
from . import operator_fv
from . import tool as gtool


# ---------------------------------------------------------------- grids ----

def build_global_z_nodes(fine_lo, fine_hi, h, pad, growth=1.15, h_max=None):
    """Uniform fine plateau [fine_lo, fine_hi] at spacing h + geometric tails.

    The plateau is snapped to the absolute lattice k*h (anchored at z=0) so
    that every measurement depth and electrode offset that is a multiple of h
    lands exactly on a node. Tails grow geometrically until they are ``pad``
    away from the plateau edges (Dirichlet boundary distance, the analogue of
    v1's z_center +- R).
    """
    if h_max is None:
        h_max = pad / 8.0
    k_lo = int(np.floor(fine_lo / h))
    k_hi = int(np.ceil(fine_hi / h))
    plateau = np.arange(k_lo, k_hi + 1, dtype=float) * h

    def tail(start, direction):
        out, x, step = [], start, h
        target = start + direction * pad
        while (x - target) * direction < 0:
            step = min(step * growth, h_max)
            x = x + direction * step
            out.append(x)
        return out

    lo_tail = tail(plateau[0], -1.0)
    hi_tail = tail(plateau[-1], +1.0)
    return np.concatenate([sorted(lo_tail), plateau, hi_tail])


# ---------------------------------------------------------------- sigma ----

def sample_sigma_aniso_zmud(r_nodes, z_nodes, formation, borehole,
                            subsample=4):
    """grid.sample_sigma_aniso with depth-interpolated mud resistivity.

    Identical transmissibility upscaling (sigma_z: harmonic in z / arithmetic
    radius-weighted in r; sigma_r: harmonic 1/r-weighted in r / arithmetic in
    z); the only difference is mud: RM is interpolated at every sub-point z
    instead of one scalar per solve, because one global column spans the whole
    RM log. Returns (sigma_r, sigma_z), each (nz-1, nr-1).
    """
    r = np.asarray(r_nodes, dtype=float)
    z = np.asarray(z_nodes, dtype=float)
    s = int(subsample)
    frac = (np.arange(s) + 0.5) / s
    z_sub = z[:-1, None] + np.diff(z)[:, None] * frac[None, :]
    r_sub = r[:-1, None] + np.diff(r)[:, None] * frac[None, :]
    wr = r_sub / np.maximum(r_sub.sum(axis=1, keepdims=True), 1e-300)

    shape = (len(z) - 1, s, len(r) - 1, s)   # (jz, sz, ir, sr)
    R = np.broadcast_to(r_sub[None, None, :, :], shape)
    Z = np.broadcast_to(z_sub[:, :, None, None], shape)
    mud = np.interp(Z, borehole[:, 0], borehole[:, 2])
    sig = ggrid._sigma_at_points(R, Z, formation, borehole, mud)

    harm_z = 1.0 / np.mean(1.0 / sig, axis=1)
    sigma_z = (harm_z * wr[None, :, :]).sum(axis=2)

    inv_r = 1.0 / np.maximum(R, 1e-300)
    w_res = inv_r / inv_r.sum(axis=3, keepdims=True)
    harm_r = 1.0 / (w_res / sig).sum(axis=3)
    sigma_r = harm_r.mean(axis=1)
    return sigma_r, sigma_z


# ------------------------------------------------------------- assembly ----

def _face_conductances_np(sigma_r, sigma_z, geom):
    """numpy twin of operator_fv.face_conductances (jnp)."""
    nzc, nrc = sigma_r.shape
    pad = np.zeros((1, nrc))
    Tr = (geom["gr_lo"] * np.vstack([pad, sigma_r])
          + geom["gr_hi"] * np.vstack([sigma_r, pad]))
    padc = np.zeros((nzc, 1))
    Tz = (geom["gz_lo"] * np.hstack([padc, sigma_z])
          + geom["gz_hi"] * np.hstack([sigma_z, padc]))
    return Tr, Tz


def assemble_csr(r_nodes, z_nodes, sigma_pair):
    """Explicit SPD operator on the free (non-Dirichlet) nodes.

    Free nodes are numbered z-major — fid(j, i) = (j-1)*(nr-1) + i for
    j in [1, nz-2], i in [0, nr-2] — so the nonzero off-diagonals sit at
    offsets 1 (radial) and nr-1 (vertical): half-bandwidth = nr-1.
    Dirichlet u=0 is eliminated symmetrically (couplings to boundary nodes
    dropped, their conductance kept on the diagonal), exactly the identity-row
    convention of operator_fv.apply_A restricted to the free subspace.

    Returns (A_csr, fid) with fid: (nz, nr) int array, -1 on Dirichlet nodes.
    """
    geom = operator_fv.build_geometry(r_nodes, z_nodes)
    Tr, Tz = _face_conductances_np(sigma_pair[0], sigma_pair[1], geom)
    dirichlet = geom["dirichlet"]
    nz, nr = dirichlet.shape

    free = ~dirichlet
    fid = -np.ones((nz, nr), dtype=np.int64)
    fid[free] = np.arange(free.sum())

    diag = np.zeros((nz, nr))
    diag[:, :-1] += Tr
    diag[:, 1:] += Tr
    diag[:-1, :] += Tz
    diag[1:, :] += Tz

    rows = [fid[free]]
    cols = [fid[free]]
    vals = [diag[free]]

    er = free[:, :-1] & free[:, 1:]           # radial edges (j,i)-(j,i+1)
    a, b, v = fid[:, :-1][er], fid[:, 1:][er], -Tr[er]
    rows += [a, b]; cols += [b, a]; vals += [v, v]

    ez = free[:-1, :] & free[1:, :]           # vertical edges (j,i)-(j+1,i)
    a, b, v = fid[:-1, :][ez], fid[1:, :][ez], -Tz[ez]
    rows += [a, b]; cols += [b, a]; vals += [v, v]

    n = int(free.sum())
    A = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)).tocsr()
    return A, fid, diag, Tr, Tz


def band_upper_from_csr(A, bandwidth):
    """Upper LAPACK band storage ab[u + i - j, j] = A[i, j] (u = bandwidth).

    Only offsets {0, 1, bandwidth} are nonzero for this stencil, but the loop
    stays generic (and cheap: A.diagonal is O(nnz)).
    """
    n = A.shape[0]
    ab = np.zeros((bandwidth + 1, n))
    for d in (0, 1, bandwidth):
        ab[bandwidth - d, d:] = A.diagonal(d)
    return ab


# ----------------------------------------------------------- the problem ----

def _node_indices(z_nodes, coords, tol=1e-6):
    """Vectorized exact-node lookup (see grid.node_index)."""
    z = np.asarray(z_nodes)
    c = np.asarray(coords, dtype=float)
    j = np.clip(np.searchsorted(z, c), 1, len(z) - 1)
    j = np.where(np.abs(z[j - 1] - c) < np.abs(z[j] - c), j - 1, j)
    bad = np.abs(z[j] - c) > tol
    if bad.any():
        raise ValueError(
            "electrode coordinates off-grid, worst miss {:.3e} m".format(
                float(np.max(np.abs(z[j] - c)))))
    return j


def build_global_tasks(tools, depths, formation, borehole, h_min=None,
                       domain_radius=None, growth=1.15, fine_margin=0.5):
    """Sample-independent part of the global problem: grid + task indexing.

    The grid depends only on (tools, depths) — canonical radial nodes and the
    lattice-snapped z plateau — so it is IDENTICAL across samples; formation /
    borehole are only used for σ later. Returns the dict skeleton shared by
    the CPU (build_global_problem) and GPU (global_gpu) paths.
    """
    cfgs = {t: gtool.tool_config(t) for t in tools}
    if h_min is None:
        h_min = min(gtool.default_h_min(c) for c in cfgs.values())
    if domain_radius is None:
        # 80*span, not v1's 10*span: the boundary sweep (gpu_v2_boundary_sweep)
        # measured 6.2% truncation error at 10*span for A8.0 vs 0.065% at
        # 80*span, and far-field nodes are logarithmic (~+14% DOF). NOTE:
        # matched-convention comparisons against v1/NGSolve production logs
        # must pass their truncated radius (max(10*span, 5)) explicitly.
        domain_radius = max(max(80.0 * c["span"], 45.0) for c in cfgs.values())

    depths = np.asarray(depths, dtype=float)
    elec = []
    for c in cfgs.values():
        zs = depths + c["depth_shift"]
        elec += [zs, zs + c["dz_M"], zs + c["dz_N"]]
    elec = np.concatenate(elec)

    z_nodes = build_global_z_nodes(elec.min() - fine_margin,
                                   elec.max() + fine_margin,
                                   h_min, pad=float(domain_radius),
                                   growth=growth)
    r_nodes = ggrid.canonical_radial_nodes(float(domain_radius),
                                           max(h_min, 1e-2), growth=growth)

    nz, nr = len(z_nodes), len(r_nodes)
    fid = -np.ones((nz, nr), dtype=np.int64)
    fid[1:-1, :-1] = np.arange((nz - 2) * (nr - 1)).reshape(nz - 2, nr - 1)

    tasks = {}
    for t, c in cfgs.items():
        zs = depths + c["depth_shift"]
        j_src = _node_indices(z_nodes, zs)
        j_M = _node_indices(z_nodes, zs + c["dz_M"])
        j_N = _node_indices(z_nodes, zs + c["dz_N"])
        tasks[t] = {
            "src": fid[j_src, 0], "M": fid[j_M, 0], "N": fid[j_N, 0],
            "K": c["K"],
        }
        assert (tasks[t]["src"] >= 0).all()

    all_src = np.concatenate([tk["src"] for tk in tasks.values()])
    uniq_src, inv = np.unique(all_src, return_inverse=True)
    ofs = 0
    for tk in tasks.values():
        tk["col"] = inv[ofs:ofs + len(depths)]
        ofs += len(depths)

    # Reciprocity accounting (E3): solving from the measuring electrodes
    # instead would need one column per unique M/N node.
    uniq_recip = np.unique(np.concatenate(
        [np.concatenate([tk["M"], tk["N"]]) for tk in tasks.values()]))

    return {
        "tools": list(tools), "depths": depths,
        "r_nodes": r_nodes, "z_nodes": z_nodes, "fid": fid,
        "h_min": float(h_min), "domain_radius": float(domain_radius),
        "bandwidth": nr - 1,
        "tasks": tasks, "n_free": int((nz - 2) * (nr - 1)),
        "uniq_src": uniq_src, "n_tasks": int(len(tools) * len(depths)),
        "n_src_recip": int(len(uniq_recip)),
    }


def build_global_problem(tools, depths, formation, borehole, h_min=None,
                         domain_radius=None, growth=1.15, fine_margin=0.5,
                         subsample=4):
    """Global grid + explicit operator + per-(tool, depth) task indexing.

    One grid for ALL tools (v2 "variant b"): axial spacing = the finest tool
    h_min, radius = the largest tool auto-R (v1 conventions: h_min from
    tool.default_h_min, R = max(10*span, 5)). Pass a single-tool list for the
    per-tool "variant a".

    Returns a dict with the grid, sigma, A (CSR on free nodes), band metadata,
    and per-tool task arrays in FREE numbering: src/M/N node ids per depth,
    plus the deduplicated source-column table (unique src -> column).
    """
    p = build_global_tasks(tools, depths, formation, borehole, h_min=h_min,
                           domain_radius=domain_radius, growth=growth,
                           fine_margin=fine_margin)
    sigma = sample_sigma_aniso_zmud(p["r_nodes"], p["z_nodes"], formation,
                                    borehole, subsample=subsample)
    A, fid, diag, Tr, Tz = assemble_csr(p["r_nodes"], p["z_nodes"], sigma)
    assert p["n_free"] == A.shape[0] and (fid == p["fid"]).all()
    p.update({"sigma": sigma, "A": A, "diag": diag, "Tr": Tr, "Tz": Tz})
    return p


# ---------------------------------------------------------------- solve ----

def factor_banded(problem):
    """Banded Cholesky of A (upper form). Returns the factor for solves.

    NOTE: scipy's ?pbtrf is the unblocked reference LAPACK routine — measured
    ~180 s for n=2e5, b=111 — so this path is unusable at global size; kept
    only as a tiny-problem cross-check. Use factor_block_thomas instead.
    """
    ab = band_upper_from_csr(problem["A"], problem["bandwidth"])
    return cholesky_banded(ab, lower=False, check_finite=False)


def factor_block_thomas(problem):
    """Global block-Thomas factorization (v1's algorithm, factored ONCE).

    The free operator is block-tridiagonal in z: n_rows = nz-2 diagonal blocks
    D_j (each tridiagonal, m = nr-1), coupled by DIAGONAL blocks
    C_j = diag(-Tz[j+1, :m]) (vertical faces connect equal r indices only).
    Standard block LDL^T: S_1 = D_1, S_{j+1} = D_{j+1} - C_j S_j^{-1} C_j;
    each Schur block S_j is dense SPD and gets a LAPACK dpotrf. This is
    exactly v1's per-window batched factorization (direct.py), but done once
    per SAMPLE on the global grid instead of once per (tool, depth).

    Returns dict with the per-row Cholesky factors (n_rows, m, m; upper) and
    the coupling vectors c (n_rows-1, m).
    """
    diag, Tr, Tz = problem["diag"], problem["Tr"], problem["Tz"]
    nz, nr = diag.shape
    m = nr - 1
    n_rows = nz - 2

    # Free-row slices: interior z rows j=1..nz-2, free r columns i=0..nr-2.
    d_main = diag[1:-1, :m]                      # (n_rows, m)
    d_sub = -Tr[1:-1, :m - 1]                    # (n_rows, m-1) radial coupling
    c = -Tz[1:-1, :m]                            # (n_rows-1, m): row j -> j+1

    factors = np.empty((n_rows, m, m))
    eye = np.eye(m)
    S = np.diag(d_main[0]) + np.diag(d_sub[0], 1) + np.diag(d_sub[0], -1)
    for j in range(n_rows):
        cf, info = dpotrf(S, lower=0, overwrite_a=1)
        if info != 0:
            raise np.linalg.LinAlgError(f"dpotrf failed at row {j}: {info}")
        factors[j] = cf
        if j + 1 < n_rows:
            Sinv, info = dpotrs(cf, eye)         # S_j^{-1}
            cj = c[j]
            S = (np.diag(d_main[j + 1])
                 + np.diag(d_sub[j + 1], 1) + np.diag(d_sub[j + 1], -1)
                 - (cj[:, None] * Sinv) * cj[None, :])
    return {"factors": factors, "c": c, "m": m, "n_rows": n_rows}


def solve_block_thomas(fac, B):
    """Solve A X = B for a block of RHS columns via the stored factors.

    B : (n_free, k) — modified in place is avoided; returns X same shape.
    Forward: v_1 = b_1, v_{j+1} = b_{j+1} - C_j S_j^{-1} v_j (w_j := S_j^{-1}
    v_j is kept for the back sweep); back: x_last = w_last,
    x_j = w_j - S_j^{-1} (C_j x_{j+1}). Per row the work is a dpotrs with k
    RHS — Level-3, threads well.
    """
    factors, c = fac["factors"], fac["c"]
    m, n_rows = fac["m"], fac["n_rows"]
    k = B.shape[1]
    V = B.reshape(n_rows, m, k)
    W = np.empty_like(V)

    w, _ = dpotrs(factors[0], V[0])
    W[0] = w
    for j in range(1, n_rows):
        v = V[j] - c[j - 1][:, None] * W[j - 1]
        W[j], _ = dpotrs(factors[j], v)

    X = np.empty_like(V)
    X[-1] = W[-1]
    for j in range(n_rows - 2, -1, -1):
        y, _ = dpotrs(factors[j], c[j][:, None] * X[j + 1])
        X[j] = W[j] - y
    return X.reshape(n_rows * m, k)


def solve_logs(problem, fac, block=32, n_jobs=16, _solve=None):
    """Solve the deduplicated source columns, assemble logs per tool.

    Columns are processed in blocks of ``block`` RHS; blocks run concurrently
    in a thread pool (LAPACK releases the GIL, the factors are shared
    read-only). Run with BLAS pinned to 1 thread — parallelism lives HERE, in
    the RHS blocks, not inside the tiny 111x111 kernels (64-thread OpenBLAS
    on these blocks is the same oversubscription blow-up as the v1
    TaskManager story: a >10x slowdown, measured).

    Returns logs dict tool -> (n_depths,) Ra.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = problem["n_free"]
    uniq = problem["uniq_src"]
    tasks = problem["tasks"]
    solve = _solve or (lambda B: solve_block_thomas(fac, B))
    Ra = {t: np.empty(len(problem["depths"])) for t in problem["tools"]}

    def run_block(lo):
        hi = min(lo + block, len(uniq))
        B = np.zeros((n, hi - lo))
        B[uniq[lo:hi], np.arange(hi - lo)] = 1.0
        return lo, hi, solve(B)

    blocks = range(0, len(uniq), block)
    with ThreadPoolExecutor(max_workers=max(1, int(n_jobs))) as pool:
        for lo, hi, U in pool.map(run_block, blocks):
            for t in problem["tools"]:
                tk = tasks[t]
                sel = (tk["col"] >= lo) & (tk["col"] < hi)
                if sel.any():
                    c = tk["col"][sel] - lo
                    dU = U[tk["M"][sel], c] - U[tk["N"][sel], c]
                    Ra[t][sel] = tk["K"] * np.abs(dU)
    return Ra
