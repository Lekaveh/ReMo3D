# -*- coding: utf-8 -*-

import numpy as np
import ngsolve as ngs
import time

#ngs.ngsglobals.msg_level=0

# Ngsolve funtions

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
    try:
        free_dofs = fes.FreeDofs(condense) if condense else fes.FreeDofs()
    except TypeError:
        free_dofs = fes.FreeDofs()

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

def AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense, return_metrics=False):
    """Assemble the stiffness matrix and preconditioner once per mesh/sigma pair."""
    timings = {}
    setup_started = time.perf_counter()
    model_dimensionality = mesh.dim

    fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = ngs.BilinearForm(fes, symmetric=False, condense=condense)

    if model_dimensionality==2:
        a += 2*np.pi*ngs.grad(u)*ngs.grad(v)*ngs.x*sigma*ngs.dx
    elif model_dimensionality==3:
        a += ngs.grad(u)*ngs.grad(v)*sigma*ngs.dx
    timings["setup"] = time.perf_counter() - setup_started

    assembly_started = time.perf_counter()
    c = ngs.Preconditioner(a, preconditioner)
    a.Assemble()
    timings["assembly"] = time.perf_counter() - assembly_started

    if return_metrics:
        metrics = {
            "dofs_total": int(getattr(fes, "ndof", 0)),
            "dofs_free": _count_free_dofs(fes, condense),
            "timings": timings,
        }
        return fes, a, c, metrics

    return fes, a, c


def SolveRHS(fes, a, c, tool_geometry, source_terms, condense, return_metrics=False):
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
    inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
    gfu = _condensed_solve(a, inv, f, fes, condense)
    timings["solve"] = time.perf_counter() - solve_started

    if return_metrics:
        metrics = _solver_metrics(inv, fes, condense)
        metrics["timings"] = timings
        return fes, gfu, metrics

    return fes, gfu


def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense, return_metrics=False):
    """Original API preserved for backward compatibility."""
    if return_metrics:
        fes, a, c, assemble_metrics = AssembleSystem(
            mesh,
            sigma,
            dirichlet_boundary,
            preconditioner,
            condense,
            return_metrics=True,
        )
        fes, gfu, solve_metrics = SolveRHS(fes, a, c, tool_geometry, source_terms, condense, return_metrics=True)
        metrics = solve_metrics
        metrics["timings"] = {**assemble_metrics["timings"], **solve_metrics["timings"]}
        metrics["dofs_total"] = assemble_metrics["dofs_total"]
        metrics["dofs_free"] = assemble_metrics["dofs_free"]
        return fes, gfu, metrics

    fes, a, c = AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense)
    return SolveRHS(fes, a, c, tool_geometry, source_terms, condense)
