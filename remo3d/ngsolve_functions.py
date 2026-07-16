# -*- coding: utf-8 -*-

import os
import numpy as np
import ngsolve as ngs
import time

#ngs.ngsglobals.msg_level=0

# Ngsolve funtions

DIRECT_SOLVER_DOF_THRESHOLD = 10000

def AddPointSource(f, position, fac, model_dimensionality):
        spc = f.space
        if model_dimensionality==2:
            mp = spc.mesh(0,position)
        elif model_dimensionality==3:
            mp = spc.mesh(0,0,position)
        ei = ngs.ElementId(ngs.VOL, mp.nr)
        fel = spc.GetFE(ei)
        dnums = spc.GetDofNrs(ei)
        shape = fel.CalcShape(*mp.pnt)
        for d,s in zip(dnums, shape):
            f.vec[d] += fac*s


def _count_free_dofs(fes, condense):
    """Return the number of free DOFs when NGSolve exposes the BitArray count."""
    free_dofs = _free_dofs(fes, condense)

    for attr_name in ("NumSet", "numset"):
        attr = getattr(free_dofs, attr_name, None)
        if callable(attr):
            try:
                return int(attr())
            except TypeError:
                pass

    try:
        return int(sum(bool(free_dofs[i]) for i in range(len(free_dofs))))
    except Exception:
        return None


def _free_dofs(fes, condense):
    try:
        return fes.FreeDofs(condense) if condense else fes.FreeDofs()
    except TypeError:
        return fes.FreeDofs()


def _solver_stat(inv, names):
    """Read a solver statistic across NGSolve version-specific spellings."""
    for name in names:
        attr = getattr(inv, name, None)
        if attr is None:
            continue
        try:
            value = attr() if callable(attr) else attr
        except TypeError:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return None


def _solver_metrics(inv, fes, condense):
    return {
        "dofs_total": int(getattr(fes, "ndof", 0)),
        "dofs_free": _count_free_dofs(fes, condense),
        "cg_iterations": _solver_stat(inv, ("GetSteps", "GetIterations", "iterations", "steps")),
        "final_residual_norm": _solver_stat(inv, ("GetResidual", "GetResiduum", "residual", "residuum")),
    }


def _normalize_order_and_metrics(order, return_metrics):
    if type(order) == bool:
        return 3, order
    if type(order) != int or order < 1:
        raise ValueError("The finite element order has to be a positive integer")
    return order, return_metrics


def _condensed_solve(a, inv, f, fes, condense, rhs_vector_factory=None):
    """Single source of truth for static-condensation solve ordering.

    This follows the documented NGSolve sequence: apply
    harmonic_extension_trans to the RHS before the condensed solve, then
    reconstruct internal DOFs with harmonic_extension and inner_solve after the
    solve. Benchmark code in scripts/benchmark_task0.py compares this path with
    condense=False baselines.
    """
    gfu = ngs.GridFunction(fes)
    if condense:
        f.vec.data += a.harmonic_extension_trans * f.vec

    rhs_vec = rhs_vector_factory(f) if rhs_vector_factory is not None else f.vec
    gfu.vec.data = inv * rhs_vec

    if condense:
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec

    return gfu

def AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense, order=3,
                   symmetric=True, direct_solver="auto", return_metrics=False):
    """Assemble the stiffness matrix and reusable solver data once per mesh/sigma pair.

    Optimization toggles (defaults reproduce the current optimized behavior):
      symmetric      -- assemble a symmetric (SPD) bilinear form (Task 1). False
                        gives the original non-symmetric assembly.
      direct_solver  -- "auto": use the sparse-Cholesky direct solver for small 2D
                        systems (Task 5); True: force the direct solver (2D only,
                        requires symmetric=True); False: always use multigrid/CG.
    """
    order, return_metrics = _normalize_order_and_metrics(order, return_metrics)
    if isinstance(direct_solver, np.bool_):
        direct_solver = bool(direct_solver)

    timings = {}
    setup_started = time.perf_counter()
    model_dimensionality = mesh.dim

    fes = ngs.H1(mesh, order=order, dirichlet=dirichlet_boundary, autoupdate=True)

    # #2 p-adaptivity (env REMO3D_PADAPT_RADIUS, meters): keep the base `order` far away
    # but raise it by one near the borehole axis (small radial coord x), where the tool
    # measures and sharp resistivity boundaries drive the low-order error. 0 = off.
    padapt_radius = float(os.environ.get("REMO3D_PADAPT_RADIUS", "0") or 0)
    if padapt_radius > 0 and model_dimensionality == 2:
        raised = 0
        for el in mesh.Elements(ngs.VOL):
            cx = sum(mesh[vtx].point[0] for vtx in el.vertices) / len(el.vertices)
            if cx < padapt_radius:
                fes.SetOrder(ngs.NodeId(ngs.ELEMENT, el.nr), order + 1)
                raised += 1
        if raised:
            fes.UpdateDofTables()

    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = ngs.BilinearForm(fes, symmetric=symmetric, condense=condense)

    if model_dimensionality==2:
        a += 2*np.pi*ngs.grad(u)*ngs.grad(v)*ngs.x*sigma*ngs.dx
    elif model_dimensionality==3:
        a += ngs.grad(u)*ngs.grad(v)*sigma*ngs.dx
    timings["setup"] = time.perf_counter() - setup_started

    if direct_solver == "auto":
        use_direct = (model_dimensionality == 2
                      and fes.ndof < DIRECT_SOLVER_DOF_THRESHOLD
                      and symmetric)
    elif direct_solver is True:
        if not symmetric:
            raise ValueError("direct_solver=True requires symmetric=True (sparsecholesky needs an SPD matrix)")
        use_direct = model_dimensionality == 2
    elif direct_solver is False:
        use_direct = False
    else:
        raise ValueError('direct_solver must be "auto", True, or False')

    c = None
    if not use_direct:
        c = ngs.Preconditioner(a, preconditioner)

    assembly_started = time.perf_counter()
    a.Assemble()
    timings["assembly"] = time.perf_counter() - assembly_started

    factorization_started = time.perf_counter()
    inv = None
    if use_direct:
        inv = a.mat.Inverse(_free_dofs(fes, condense), inverse="sparsecholesky")
    timings["factorization"] = time.perf_counter() - factorization_started

    if return_metrics:
        metrics = {
            "dofs_total": int(getattr(fes, "ndof", 0)),
            "dofs_free": _count_free_dofs(fes, condense),
            "solver_type": "direct" if inv is not None else "cg",
            "timings": timings,
        }
        return fes, a, c, inv, metrics

    return fes, a, c, inv


def SolveRHS(fes, a, c, inv, tool_geometry, source_terms, condense, return_metrics=False):
    """Solve one right-hand side against a pre-assembled system."""
    timings = {}
    rhs_started = time.perf_counter()

    f = ngs.LinearForm(fes)
    f.Assemble()
    model_dimensionality = f.space.mesh.dim

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    timings["rhs"] = time.perf_counter() - rhs_started

    solve_started = time.perf_counter()
    solve_inv = inv
    if solve_inv is None:
        solve_inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
    gfu = _condensed_solve(a, solve_inv, f, fes, condense)
    timings["solve"] = time.perf_counter() - solve_started

    if return_metrics:
        metrics = _solver_metrics(solve_inv, fes, condense)
        metrics["solver_type"] = "direct" if inv is not None else "cg"
        metrics["timings"] = timings
        return fes, gfu, metrics

    return fes, gfu


def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense, order=3,
             symmetric=True, direct_solver="auto", return_metrics=False):
    """Original API preserved for backward compatibility.

    ``symmetric`` and ``direct_solver`` expose the Task 1 / Task 5 optimizations
    for ablation benchmarking; their defaults reproduce the optimized behavior.
    """
    order, return_metrics = _normalize_order_and_metrics(order, return_metrics)

    if return_metrics:
        fes, a, c, inv, assemble_metrics = AssembleSystem(
            mesh,
            sigma,
            dirichlet_boundary,
            preconditioner,
            condense,
            order=order,
            symmetric=symmetric,
            direct_solver=direct_solver,
            return_metrics=True,
        )
        fes, gfu, solve_metrics = SolveRHS(fes, a, c, inv, tool_geometry, source_terms, condense, return_metrics=True)
        metrics = solve_metrics
        metrics["timings"] = {**assemble_metrics["timings"], **solve_metrics["timings"]}
        metrics["dofs_total"] = assemble_metrics["dofs_total"]
        metrics["dofs_free"] = assemble_metrics["dofs_free"]
        metrics["solver_type"] = assemble_metrics["solver_type"]
        return fes, gfu, metrics

    fes, a, c, inv = AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense, order=order,
                                    symmetric=symmetric, direct_solver=direct_solver)
    return SolveRHS(fes, a, c, inv, tool_geometry, source_terms, condense)
