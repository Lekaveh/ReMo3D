# -*- coding: utf-8 -*-

import numpy as np
import ngsolve as ngs
import time
    
ngs.ngsglobals.msg_level=0

from ngsolve_functions import AddPointSource, _condensed_solve, _solver_metrics

from ngsolve.ngscuda import *


# Ngsolve gpu funtions

def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense, solve_on="CPU", return_metrics=False):

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

    rhs_started = time.perf_counter()
    f = ngs.LinearForm(fes)
    f.Assemble()

    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    timings["rhs"] = time.perf_counter() - rhs_started

    assembly_started = time.perf_counter()
    c = ngs.Preconditioner(a, preconditioner)
    a.Assemble()
    timings["assembly"] = time.perf_counter() - assembly_started

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
