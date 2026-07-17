# -*- coding: utf-8 -*-

# --- Thread pinning (part 1/2): env caps, MUST run before numpy import -------
# Each equation solve runs in its own MPI worker process, so parallelism already
# lives at the process level. `import numpy` alone instantiates a node-sized
# MKL/libiomp5 pool (64 threads on a 64-core node) in EVERY worker. Those
# threads sleep through the solves, but 100 workers x 64 = 6400 clones is the
# ">1000 workers" count seen in ps/htop on HPC. Cap them to
# REMO3D_WORKER_THREADS (default 1); MPI already supplies the parallelism.
#
# NOTE: these env vars do NOT fix the direct-solver blow-up by themselves —
# see part 2 below. Verified 2026-07-17 (WORK_SUMMARY.md B.3).
import os
_worker_threads = os.environ.get("REMO3D_WORKER_THREADS", "1")
for _tvar in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_tvar, _worker_threads)

from mpi4py import MPI

import numpy as np
import ngsolve as ngs

# --- Thread pinning (part 2/2): NGSolve TaskManager — the essential cap ------
# `Inverse(inverse="sparsecholesky")` (the direct solver) spins up NGSolve's own
# TaskManager: one std::thread per core, and it IGNORES OMP_/MKL_NUM_THREADS
# (measured: with both=1 the factorization still ran 64 threads). This is the
# layer that saturated HPC nodes with direct_solver on — cpu_workers x cores
# RUNNING threads — while the CG/multigrid path (which never enters TaskManager)
# looked fine. SetNumThreads is therefore the essential part of the fix, not a
# belt-and-suspenders extra. Bonus: on these small 2D systems the 1-thread
# factorization is ~2.8x faster than the node-wide one (TaskManager sync
# overhead exceeds the work).
try:
    ngs.SetNumThreads(int(_worker_threads))
except Exception:
    pass

import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gmsh_functions as gmf
import netgen_functions as ngf
import ngsolve_functions as ngsf

WORKER_DEBUG = os.environ.get("REMO3D_WORKER_DEBUG", "").lower() in ("1", "true", "yes", "on")

# Supress Netgen teminal output during mesh creation process
ngs.ngsglobals.msg_level = 0

# Connect to main process
try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
except:
    raise ValueError('The worker could not connect to main process')

## Collect and wait for all workers to receive data
solve_on = list()
solve_on = comm.bcast(solve_on, root=0)
# Process-unique mesh-file directory chosen by the manager (see
# initialize_workers in remo3d.py). Overwrite the module-level fallback so every
# worker of this run reads/writes the same isolated directory.
tmp_dir = comm.bcast(None, root=0)
gmf.TMP_DIR = tmp_dir
comm.barrier()

# Import ngsolve functions
if solve_on[rank] == "CPU":
    import ngsolve_functions as ngsf
elif solve_on[rank] == "GPU":
    import ngsolve_functions_gpu as ngsf

## Ask for tasks until receving stop sentinel
for lvl_1_msg in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
    
    # Collect information about shapes of broadcasted arrays
    arrays_shape = list()
    arrays_shape = comm.bcast(arrays_shape, root=0)

    # Prepare empty variables
    formation_parameters = np.empty(arrays_shape[0], dtype='float')
    borehole_geometry = np.empty(arrays_shape[1], dtype='float')
    mud_resistivities = np.empty(arrays_shape[2], dtype='float')
    simulation_depths = np.empty(arrays_shape[3], dtype='float')
    dip = float()
    tools_parameters = dict()
    domain_radius = float()
    mesh_generator = str()
    preconditioner = str()
    condense = bool()
    fe_order = int()
    symmetric = bool()
    reuse_assembly = bool()
    direct_solver = None
    collect_metrics = bool()
    task_list = list()

    # Fill variables with data
    comm.Bcast([formation_parameters, MPI.FLOAT], root=0)
    comm.Bcast([borehole_geometry, MPI.FLOAT], root=0)
    comm.Bcast([mud_resistivities, MPI.FLOAT], root=0)
    comm.Bcast([simulation_depths, MPI.FLOAT], root=0)
    dip = comm.bcast(dip, root=0)
    tools_parameters = comm.bcast(tools_parameters, root=0)
    domain_radius =  comm.bcast(domain_radius, root=0)
    mesh_generator = comm.bcast(mesh_generator, root=0)
    preconditioner = comm.bcast(preconditioner, root=0)
    condense = comm.bcast(condense, root=0)
    fe_order = comm.bcast(fe_order, root=0)
    # New optimization-toggle flags (must match the master's broadcast order in
    # remo3d.py simulate_logs; task_list stays LAST on both sides).
    symmetric = comm.bcast(symmetric, root=0)
    reuse_assembly = comm.bcast(reuse_assembly, root=0)
    direct_solver = comm.bcast(direct_solver, root=0)
    collect_metrics = comm.bcast(collect_metrics, root=0)
    task_list = comm.bcast(task_list, root=0)

    ## Wait for all workers to receive data
    comm.barrier()

    results = list()
    task_metrics = list()
    for lvl_2_msg in iter(lambda: comm.sendrecv(None, dest=0), StopIteration):
        task = task_list[lvl_2_msg]
        try:
            if collect_metrics:
                task_t0 = time.perf_counter()
            depth_index = task[0]
            tool = task[1]
            tool_geometry = tool[0,:]
            source_terms = tool[1,:]

            # #4 per-tool domain (env REMO3D_PER_TOOL_DOMAIN): size the simulation domain
            # to THIS task's tool geometry instead of the global domain, so short tools get
            # small meshes. Falls back to the global domain_radius when unset.
            task_domain_radius = domain_radius
            if os.environ.get("REMO3D_PER_TOOL_DOMAIN", "") not in ("", "0"):
                task_domain_radius = max(10.0 * float(np.max(np.abs(tool_geometry))), 5.0)

            if mesh_generator=="gmsh":
                # Carve out suitable range of data
                local_formation_geometry, local_borehole_geometry, sigma = gmf.SelectGmshDataRange(borehole_geometry, formation_parameters, dip, mud_resistivities[depth_index], simulation_depths[depth_index], task_domain_radius)
                # Create geometry and mesh
                if dip==0:
                    mesh = gmf.ConstructGmsh2dModel(task_domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry, rank, mesh_generator)
                else:
                    mesh = gmf.ConstructGmsh3dModel(task_domain_radius, tool_geometry, source_terms, local_formation_geometry, dip, local_borehole_geometry, rank)
                dirichlet_boundary = 'dirichlet_boundary'
            # Generate mesh using netgen
            elif mesh_generator=="netgen":
                # Carve out suitable range of data
                local_formation_geometry, local_borehole_geometry, sigma = ngf.SelectNetgenDataRange(borehole_geometry, formation_parameters, mud_resistivities[depth_index], simulation_depths[depth_index], task_domain_radius)
                # Create geometry and mesh
                mesh = ngf.ConstructNetgen2dModel(task_domain_radius, tool_geometry, source_terms, local_formation_geometry, local_borehole_geometry)
                dirichlet_boundary = [2]

            # Convert data to ngsolve format
            mesh = ngs.Mesh(mesh)
            sigma = ngs.CoefficientFunction(sigma)

            ## Assemble + solve. reuse_assembly=True (Task 2): assemble the matrix and
            ## preconditioner/factorization once per mesh, reuse across all RHS in the
            ## batch. reuse_assembly=False: re-assemble per RHS (original behavior).
            task_dofs_total = task_dofs_free = task_solver_type = task_cg_iters = None
            task_assembly_time = 0.0
            task_solve_time = 0.0
            n_solves = 0

            if reuse_assembly:
                if collect_metrics:
                    fes, a, c, inv, am = ngsf.AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense,
                                                             order=fe_order, symmetric=symmetric, direct_solver=direct_solver,
                                                             return_metrics=True)
                    task_dofs_total, task_dofs_free = am["dofs_total"], am["dofs_free"]
                    task_solver_type = am["solver_type"]
                    task_assembly_time += sum(am["timings"].values())
                else:
                    fes, a, c, inv = ngsf.AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense,
                                                         order=fe_order, symmetric=symmetric, direct_solver=direct_solver)

            ## Compute measured resistivity
            for modelling_task in task[2]:
                tool = modelling_task[1]
                tool_geometry = tool[0,:]
                source_terms = tool[1,:]

                if not reuse_assembly:
                    if collect_metrics:
                        fes, a, c, inv, am = ngsf.AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense,
                                                                 order=fe_order, symmetric=symmetric, direct_solver=direct_solver,
                                                                 return_metrics=True)
                        task_dofs_total, task_dofs_free = am["dofs_total"], am["dofs_free"]
                        task_solver_type = am["solver_type"]
                        task_assembly_time += sum(am["timings"].values())
                    else:
                        fes, a, c, inv = ngsf.AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense,
                                                             order=fe_order, symmetric=symmetric, direct_solver=direct_solver)

                ## Solve BVP for this right-hand side
                if collect_metrics:
                    fes, gfu, sm = ngsf.SolveRHS(fes, a, c, inv, tool_geometry, source_terms, condense, return_metrics=True)
                    task_solve_time += sum(sm["timings"].values())
                    if sm.get("cg_iterations") is not None:
                        task_cg_iters = sm["cg_iterations"]
                else:
                    fes, gfu = ngsf.SolveRHS(fes, a, c, inv, tool_geometry, source_terms, condense)
                n_solves += 1

                # Compute resistivity values
                for rc_task in modelling_task[2]:
                    depth = rc_task[0]
                    tool = list(tools_parameters.keys())[rc_task[1]]
                    offset = rc_task[2]
                    tool_geometry = tools_parameters[tool][0,:3] + offset
                    source_terms = tools_parameters[tool][1,:3]
                    geometric_factor = tools_parameters[tool][0,3]
                    measuring_electodes = tool_geometry[source_terms==0]

                    if dip==0:
                        if np.shape(measuring_electodes)[0] == 2:
                            result = abs(geometric_factor * (gfu(mesh(0.0, measuring_electodes[1]))-gfu(mesh(0.0, measuring_electodes[0]))))
                        elif np.shape(measuring_electodes)[0] == 1:
                            result = abs(geometric_factor * gfu(mesh(0.0, measuring_electodes[0])))
                    else:
                        if np.shape(measuring_electodes)[0] == 2:
                            result = abs(geometric_factor * (gfu(mesh(0.0, 0.0, measuring_electodes[1]))-gfu(mesh(0.0, 0.0, measuring_electodes[0]))))/2 # division by two because only halfsphere is present within the model
                        elif np.shape(measuring_electodes)[0] == 1:
                            result = abs(geometric_factor * gfu(mesh(0.0, 0.0, measuring_electodes[0])))/2 # division by two because only halfsphere is present within the model

                    # Append result to results
                    results.append([rc_task[0], rc_task[1], result])

            if collect_metrics:
                task_metrics.append({
                    "rank": rank,
                    "task_index": int(lvl_2_msg),
                    "n_solves": int(n_solves),
                    "wall": time.perf_counter() - task_t0,
                    "assembly_time": float(task_assembly_time),
                    "solve_time": float(task_solve_time),
                    "dofs_total": task_dofs_total,
                    "dofs_free": task_dofs_free,
                    "solver_type": task_solver_type,
                    "cg_iterations": task_cg_iters,
                    "reuse_assembly": bool(reuse_assembly),
                })
        except Exception:
            if WORKER_DEBUG:
                raise
            for modelling_task in task[2]:
                for rc_task in modelling_task[2]:
                    results.append([rc_task[0], rc_task[1], np.nan])

    # Report results to master process
    comm.barrier()
    comm.gather(sendobj=results, root=0)

    # Report per-task metrics when requested. This gather is collective and gated
    # by the same broadcast flag on master and workers, so it stays symmetric.
    if collect_metrics:
        comm.gather(sendobj=task_metrics, root=0)

## Shutdown
comm.Disconnect()
