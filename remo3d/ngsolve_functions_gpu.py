# -*- coding: utf-8 -*-

import numpy as np
import ngsolve as ngs
import time
    
ngs.ngsglobals.msg_level=0

from ngsolve_functions import AddPointSource, AssembleSystem, _condensed_solve, _solver_metrics

from ngsolve.ngscuda import *


# Ngsolve gpu funtions

def SolveRHS(fes, a, c, tool_geometry, source_terms, condense, return_metrics=False):
    """GPU-accelerated solve for one RHS against a pre-assembled CPU system."""
    timings = {}
    rhs_started = time.perf_counter()

    f = ngs.LinearForm(fes)
    f.Assemble()
    model_dimensionality = f.space.mesh.dim

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    timings["rhs"] = time.perf_counter() - rhs_started

    device_started = time.perf_counter()
    adev = a.mat.CreateDeviceMatrix()
    cdev = c.mat.CreateDeviceMatrix()
    timings["device_transfer_setup"] = time.perf_counter() - device_started

    solve_started = time.perf_counter()
    inv = ngs.CGSolver(adev, cdev, maxsteps=1000, printrates=False)
    gfu = _condensed_solve(
        a,
        inv,
        f,
        fes,
        condense,
        rhs_vector_factory=lambda assembled_f: assembled_f.vec.CreateDeviceVector(copy=True),
    )
    timings["solve"] = time.perf_counter() - solve_started

    if return_metrics:
        metrics = _solver_metrics(inv, fes, condense)
        metrics["timings"] = timings
        return fes, gfu, metrics

    return fes, gfu


def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense, solve_on="CPU", return_metrics=False):
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
