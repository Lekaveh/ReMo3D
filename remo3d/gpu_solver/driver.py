# -*- coding: utf-8 -*-
"""High-level driver: batched GPU forward modelling for dip=0 logs.

`compute_logs_gpu` mirrors the result contract of Model.compute_synthetic_logs
(logs[tool] = array with [:, 0] = depth, [:, 1] = apparent resistivity) but
runs entirely on the structured-grid JAX solver: one graded depth-relative
grid per tool, sigma sampled per measurement depth, and a single vmap-batched
PCG solve over all depths of a tool.

Restrictions vs the NGSolve pipeline: dip=0 only, normalized single-current-
electrode tools (the default force_single_electrode_configuration=True form).
NGSolve remains the reference implementation and the 3D path.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from ..sensitivity import _load_formation, _load_borehole, _parse_tool
from . import grid as ggrid
from . import operator_fv, operator_fem
from . import sigma_gpu
from . import solve as gsolve
from . import tool as gtool

_BACKEND_MODULES = {"fv": operator_fv, "fem": operator_fem}


def build_tool_problem(tool_name, measurement_depths, formation, borehole,
                       backend="fv", h_min=None, growth=1.15,
                       domain_radius=None, dtype=jnp.float64,
                       precond="mg", data_dependent_foci=False):
    """Grid, geometry, RHS and per-depth sigma stacks for one tool.

    Returns a dict with everything needed to solve and extract the log.
    The grid is depth-relative (z=0 at the current electrode), so all depths
    share geometry and RHS; only sigma varies along the batch axis.

    precond="mg" builds the multigrid hierarchy (level 0 is the solve grid)
    and per-level sigma stacks; "jacobi" keeps the single-level problem.
    """
    from . import mg as gmg

    cfg = gtool.tool_config(tool_name)
    if h_min is None:
        h_min = gtool.default_h_min(cfg)
    if domain_radius is None:
        domain_radius = max(10.0 * cfg["span"], 5.0)

    elec_rel = [0.0, cfg["dz_M"], cfg["dz_N"]]
    # Canonical (data-independent) radial grid so the grid SHAPE depends only
    # on the tool (R, h_min), not on the per-sample invasion radii -> the XLA
    # compile is cached across samples. data_dependent_foci=True restores the
    # old behaviour for comparison.
    r_foci = None
    if data_dependent_foci:
        fz = formation[:, 2]
        r_foci = ggrid.dedup_foci(
            [float(np.median(borehole[:, 1]))]
            + [float(v) for v in fz[~np.isnan(fz)]], min_sep=4.0 * h_min)

    hier = None
    if precond == "mg":
        hier = gmg.build_hierarchy(domain_radius, elec_rel, r_foci, h_min,
                                   growth=growth, backend=backend,
                                   dtype=dtype)
        lv0 = hier["levels"][0]
        r_nodes, z_rel, geom = lv0["r"], lv0["z"], lv0["geom"]
    else:
        r_nodes, z_rel = ggrid.build_grid(domain_radius, 0.0, elec_rel,
                                          r_foci=r_foci, h_min=h_min,
                                          growth=growth)
        geom = _BACKEND_MODULES[backend].build_geometry(r_nodes, z_rel)
        geom = {k: (v.astype(np.dtype(dtype))
                    if np.issubdtype(np.asarray(v).dtype, np.floating) else v)
                for k, v in geom.items()}

    j_C = ggrid.node_index(0.0, z_rel)
    j_M = ggrid.node_index(cfg["dz_M"], z_rel)
    j_N = ggrid.node_index(cfg["dz_N"], z_rel)
    rhs = operator_fv.point_source_rhs(
        (len(z_rel), len(r_nodes)), [((j_C, 0), 1.0)], geom["dirichlet"],
        dtype=dtype)

    depths = np.asarray(measurement_depths, dtype=float)

    def _stacks(levels):
        """Per-level (sigma_r, sigma_z) stacks over all depths.

        Vectorized over depths (sample_sigma_aniso_batch); the per-depth
        Python loop dominated driver setup time. Subcell averaging
        (subsample=4) is only worth its CPU cost on level 0 — the level that
        defines the discrete solution. Coarser levels exist only inside the
        preconditioner, where plain cell-center sampling is plenty; sampling
        them at subsample=4 made grid setup dominate the whole run
        (923 s vs 321 s full-Ex1) for zero accuracy gain.
        """
        z_sims = depths + cfg["depth_shift"]
        muds = np.interp(z_sims, borehole[:, 0], borehole[:, 2])
        return tuple(
            sigma_gpu.sample_sigma_aniso_gpu(
                lv["r"], lv["z"], z_sims, formation, borehole, muds,
                subsample=4 if li == 0 else 1, dtype=dtype)
            for li, lv in enumerate(levels))

    if precond == "mg":
        sigmas_l = _stacks(hier["levels"])
        sigmas = sigmas_l[0]
    else:
        sigmas_l = None
        sigmas = _stacks([{"r": r_nodes, "z": z_rel}])[0]

    return {
        "cfg": cfg, "geom": geom, "rhs": rhs,
        "sigmas": sigmas, "sigmas_l": sigmas_l, "hier": hier,
        "j_M": j_M, "j_N": j_N,
        "r_nodes": r_nodes, "z_rel": z_rel,
        "depths": depths, "backend": backend, "h_min": h_min,
        "domain_radius": float(domain_radius), "precond": precond,
    }


def solve_tool_log(problem, tol=1e-8, maxiter=None, batch_size=None):
    """Batched solve over all depths of one tool -> (depths, Ra) array.

    batch_size limits device memory: depths are processed in chunks of that
    size (None = all at once). Uses the preconditioner chosen at
    build_tool_problem time (multigrid by default).
    """
    sig_r, sig_z = problem["sigmas"]
    n = sig_r.shape[0]
    if batch_size is None:
        batch_size = n

    if problem["precond"] == "mg":
        solver = gsolve.make_mg_solver(problem["hier"], tol=tol,
                                       maxiter=maxiter or 300)
        sigmas_l = problem["sigmas_l"]

        def run_chunk(lo, hi):
            chunk = tuple((sr[lo:hi], sz[lo:hi]) for sr, sz in sigmas_l)
            u, _ = solver(chunk, problem["rhs"])
            return u
    else:
        fn = jax.vmap(lambda s: gsolve.solve_one(
            s, problem["rhs"], problem["geom"], backend=problem["backend"],
            tol=tol, maxiter=maxiter or 20000))

        def run_chunk(lo, hi):
            return fn((sig_r[lo:hi], sig_z[lo:hi]))

    Ra = np.empty(n)
    cfg, j_M, j_N = problem["cfg"], problem["j_M"], problem["j_N"]
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        u = run_chunk(lo, hi)
        Ra[lo:hi] = gtool.apparent_resistivity(np.asarray(u), cfg, j_M, j_N)
    return np.column_stack([problem["depths"], Ra])


def bucket_tools_by_radius(tools, max_ratio=4.0):
    """Group tools so each bucket's auto domain radii span at most max_ratio.

    One shared grid uses min(h_min) x max(R) across its tools, so mixing a
    short tool (small R, fine h) with a long one (large R) forces a grid with
    both a huge extent AND fine spacing -> the node count and the nested-vmap
    XLA graph explode (the full Ex1 set on one grid = 339k nodes, >30 min to
    compile). Bucketing keeps each shared grid homogeneous; every bucket is
    still one grid / one compilation / one batch over all its tools+depths.
    """
    def auto_r(t):
        i = _parse_tool(t)
        return max(10.0 * max(abs(i["z_M"] - i["z_C"]),
                              abs(i["z_N"] - i["z_C"])), 5.0)

    ordered = sorted(tools, key=auto_r)
    buckets, cur, r0 = [], [], None
    for t in ordered:
        r = auto_r(t)
        if cur and r / r0 > max_ratio:
            buckets.append(cur)
            cur, r0 = [], None
        if not cur:
            r0 = r
        cur.append(t)
    if cur:
        buckets.append(cur)
    return buckets


def build_shared_problem(tools, measurement_depths, formation, borehole,
                         backend="fv", h_min=None, growth=1.15,
                         domain_radius=None, dtype=jnp.float64,
                         mg_subsample_coarse=1):
    """One measurement-point-centred grid + MG hierarchy for ALL tools.

    The grid is centred on the measurement point (z_rel = 0 at MP), so sigma
    on a given depth is SHARED by every tool; per-tool data reduces to the
    source node (RHS) and the M/N sample nodes. Electrode positions of all
    tools become z-foci, so each is an exact grid node.

    Sigma stacks are sampled on the GPU (sigma_gpu) for every hierarchy
    level: level 0 with subcell averaging (subsample=4), coarser levels at
    subsample=1 (preconditioner only).

    Note the one physics compromise vs the per-tool path: mud resistivity is
    interpolated at the MP depth for all tools (the NGSolve pipeline uses the
    current-electrode depth per tool). The RM log varies slowly, and the Ex1
    validation shows no visible effect; revisit if a mud log ever has sharp
    gradients.
    """
    from . import mg as gmg

    cfgs = {}
    for t in tools:
        info = _parse_tool(t)
        cfgs[t] = info
    if h_min is None:
        h_min = min(gtool.default_h_min(gtool.tool_config(t)) for t in tools)
    if domain_radius is None:
        domain_radius = max(
            max(10.0 * max(abs(i["z_M"] - i["z_C"]), abs(i["z_N"] - i["z_C"])),
                5.0) for i in cfgs.values())

    elec_all = sorted({0.0} | {round(v, 9) for i in cfgs.values()
                              for v in (i["z_C"], i["z_M"], i["z_N"])})
    # Canonical data-independent radial grid (see build_tool_problem).
    hier = gmg.build_hierarchy(domain_radius, elec_all, None, h_min,
                               growth=growth, backend=backend, dtype=dtype)
    lv0 = hier["levels"][0]
    r_nodes, z_rel, geom = lv0["r"], lv0["z"], lv0["geom"]

    depths = np.asarray(measurement_depths, dtype=float)
    muds = np.interp(depths, borehole[:, 0], borehole[:, 2])
    sigmas_l = tuple(
        sigma_gpu.sample_sigma_aniso_gpu(
            lv["r"], lv["z"], depths, formation, borehole, muds,
            subsample=4 if li == 0 else mg_subsample_coarse, dtype=dtype)
        for li, lv in enumerate(hier["levels"]))

    shape = (len(z_rel), len(r_nodes))
    rhs_stack = jnp.stack([
        operator_fv.point_source_rhs(
            shape, [((ggrid.node_index(cfgs[t]["z_C"], z_rel), 0), 1.0)],
            geom["dirichlet"], dtype=dtype)
        for t in tools])
    probes = {t: (ggrid.node_index(cfgs[t]["z_M"], z_rel),
                  ggrid.node_index(cfgs[t]["z_N"], z_rel),
                  float(cfgs[t]["K"])) for t in tools}

    return {
        "tools": list(tools), "cfgs": cfgs, "probes": probes,
        "hier": hier, "geom": geom, "rhs_stack": rhs_stack,
        "sigmas_l": sigmas_l, "depths": depths,
        "r_nodes": r_nodes, "z_rel": z_rel, "backend": backend,
        "h_min": h_min, "domain_radius": float(domain_radius),
    }


def solve_shared_log(problem, tol=1e-8, maxiter=300, batch_size=32):
    """Solve all (depth x tool) tasks on the shared grid -> logs dict.

    Depth chunks are padded to a fixed size so the nested-vmap solver
    compiles exactly once; the padded tail is discarded.
    """
    solver = gsolve.make_shared_mg_solver(problem["hier"], tol=tol,
                                          maxiter=maxiter or 300)
    sigmas_l = problem["sigmas_l"]
    rhs_stack = problem["rhs_stack"]
    depths = problem["depths"]
    n = len(depths)
    batch_size = min(batch_size or n, n)

    tools = problem["tools"]
    Ra = np.empty((len(tools), n))
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        pad = batch_size - (hi - lo)
        chunk = tuple(
            (jnp.concatenate([sr[lo:hi], sr[hi - 1:hi].repeat(pad, 0)])
             if pad else sr[lo:hi],
             jnp.concatenate([sz[lo:hi], sz[hi - 1:hi].repeat(pad, 0)])
             if pad else sz[lo:hi])
            for sr, sz in sigmas_l)
        u, _ = solver(chunk, rhs_stack)          # (bs, T, nz, nr)
        u = np.asarray(u[: hi - lo])
        for ti, t in enumerate(tools):
            j_M, j_N, K = problem["probes"][t]
            Ra[ti, lo:hi] = K * np.abs(u[:, ti, j_M, 0] - u[:, ti, j_N, 0])

    return {t: np.column_stack([depths, Ra[ti]])
            for ti, t in enumerate(tools)}


def compute_logs_gpu(tools, measurement_depths, formation_model,
                     borehole_model, borehole_geometry_type="diameter",
                     backend="fv", h_min=None, growth=1.15,
                     domain_radius=None, dtype=jnp.float64,
                     tol=1e-8, maxiter=None, batch_size=None,
                     precond="mg", shared_grid=False, global_solver=False,
                     precision="mixed", convention="v2", verbose=False):
    """GPU forward modelling for a list of tools over measurement depths.

    Parameters mirror Model.compute_synthetic_logs where they overlap;
    formation_model / borehole_model accept paths or ndarrays (same loaders).
    domain_radius: None = per-tool auto (10 x span, floor 5 m), a number =
    shared fixed radius, "auto" = same as None. precond: "mg" (default) or
    "jacobi".

    shared_grid=True solves ALL tools on one measurement-point-centred grid
    in a single nested-vmap batch (one hierarchy, one compilation, sigma
    sampled once per depth on the GPU) — the fast path for multi-tool sweeps.

    global_solver=True is the v2 path (RECOMMENDED — see
    GPU_SOLVER_V2_PLAN.md / wiki findings/gpu-solver-v2): ONE global grid
    over the whole logged interval, factor once, all (tool, depth) tasks as
    deduplicated RHS columns. `precision` is "mixed" (fp64 factor recursion,
    fp32 solves; Ra error ~4e-5) or "f64"; `dtype`/`tol`/`precond`/
    `batch_size`/`backend` are ignored on this path.

    convention (global path only):
      "v2" (default) — physical z-varying mud column + far boundary
        (domain_radius max(80*span, 45));
      "cpu" — imitate the CPU pipeline where architecturally possible:
        scalar mud RM(z_sim) per task (mud-split operator + Neumann series
        on one factorization, exact to ~5e-5) and the pipeline's FIXED
        domain radius (simulate_logs default 50; pass domain_radius to
        match a specific run — e.g. the optim_bench references used 40).
        NOT imitable: the pipeline's per-depth z-window truncation — for a
        small R on conductive-mud models its own error reaches tens of
        percent (fresh NGSolve converges to the v2 value as its window
        grows), so residual differences remain exactly where the pipeline
        itself is unconverged.

    Returns
    -------
    logs : dict tool -> ndarray (n_depths, 2) with [:, 0]=depth, [:, 1]=R_a.
    """
    if isinstance(domain_radius, str):
        if domain_radius != "auto":
            raise ValueError('domain_radius must be a number, None, or "auto"')
        domain_radius = None
    formation = _load_formation(list(tools), formation_model)
    borehole = _load_borehole(list(tools), borehole_model,
                              borehole_geometry_type=borehole_geometry_type)

    if global_solver:
        from . import global_op, global_gpu
        from . import tool as gtool
        t0 = time.perf_counter()
        if convention == "cpu" and domain_radius is None:
            domain_radius = 50.0        # simulate_logs' fixed-R default
        problem = global_op.build_global_tasks(
            tools, measurement_depths, formation, borehole,
            h_min=h_min, domain_radius=domain_radius, growth=growth)
        solver = global_gpu.make_solver(
            problem, precision=precision,
            mud=("cpu" if convention == "cpu" else "log"),
            chunk_cols=(96 if convention == "cpu" else None))
        X0 = solver(np.asarray(formation, float)[None],
                    np.asarray(borehole, float)[None])
        ra = global_gpu.extract_logs(problem, X0)[0]
        if verbose:
            nr, nz = len(problem["r_nodes"]), len(problem["z_nodes"])
            print(f"[gpu_solver] global: grid {nr}x{nz} "
                  f"(h_min={problem['h_min']:.4g}, "
                  f"R={problem['domain_radius']:.1f}), "
                  f"{problem['n_tasks']} tasks -> "
                  f"{len(problem['uniq_src'])} RHS | "
                  f"{time.perf_counter() - t0:.1f}s incl. compile")
        depths = problem["depths"]
        return {t: np.column_stack([depths, ra[t]]) for t in tools}

    if shared_grid:
        # One shared grid per radius bucket (a single grid for tools of very
        # different scale is pathological — see bucket_tools_by_radius).
        buckets = (bucket_tools_by_radius(tools)
                   if domain_radius is None else [list(tools)])
        logs = {}
        for bi, bucket in enumerate(buckets):
            t0 = time.perf_counter()
            problem = build_shared_problem(
                bucket, measurement_depths, formation, borehole,
                backend=backend, h_min=h_min, growth=growth,
                domain_radius=domain_radius, dtype=dtype)
            t_setup = time.perf_counter() - t0
            t0 = time.perf_counter()
            logs.update(solve_shared_log(problem, tol=tol, maxiter=maxiter,
                                         batch_size=batch_size or 32))
            t_solve = time.perf_counter() - t0
            if verbose:
                nr, nz = len(problem["r_nodes"]), len(problem["z_rel"])
                print(f"[gpu_solver] bucket {bi + 1}/{len(buckets)} "
                      f"{bucket}: grid {nr}x{nz} "
                      f"(h_min={problem['h_min']:.4g}, R={problem['domain_radius']:.1f}), "
                      f"{len(problem['depths'])} depths | "
                      f"setup {t_setup:.1f}s, solve {t_solve:.1f}s")
        return logs

    logs = {}
    for tool_name in tools:
        t0 = time.perf_counter()
        problem = build_tool_problem(
            tool_name, measurement_depths, formation, borehole,
            backend=backend, h_min=h_min, growth=growth,
            domain_radius=domain_radius, dtype=dtype, precond=precond)
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        logs[tool_name] = solve_tool_log(problem, tol=tol, maxiter=maxiter,
                                         batch_size=batch_size)
        t_solve = time.perf_counter() - t0
        if verbose:
            nr, nz = len(problem["r_nodes"]), len(problem["z_rel"])
            print(f"[gpu_solver] {tool_name}: grid {nr}x{nz}, "
                  f"h_min={problem['h_min']:.4g}, R={problem['domain_radius']:.1f}, "
                  f"{len(problem['depths'])} depths | "
                  f"setup {t_setup:.1f}s, solve {t_solve:.1f}s")
    return logs
