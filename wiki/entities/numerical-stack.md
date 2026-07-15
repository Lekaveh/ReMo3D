---
title: Numerical Stack (Gmsh · Netgen · NGSolve · MPI)
type: entity
tags: [tools, dependencies, fem, mesh, mpi]
sources: [repo-docs]
updated: 2026-07-15
---

# Numerical Stack

*(traces to `[[repo-docs]]` — architecture.md, installation-and-environment.md.)*
The external libraries ReMo3D orchestrates, and each one's role.

| Library | Role in ReMo3D | Used by |
|---|---|---|
| **Gmsh** (OCC) | 2D + **all 3D** geometry (boolean CSG) and meshing; writes `.msh` | [mesh generation](../concepts/mesh-generation.md) |
| **Netgen** | Native **2D** `SplineGeometry` meshing (default 2D); also the target mesh container (`ReadGmsh` maps Gmsh → Netgen) | [mesh generation](../concepts/mesh-generation.md) |
| **NGSolve** | FE space (`H1`), bilinear form, assembly, `CGSolver`, sparse-Cholesky, static condensation | [FEM solver](../concepts/fem-solver.md) |
| **ngsolve.ngscuda** | Optional CUDA device matrices/vectors for the GPU CG solve | [FEM solver](../concepts/fem-solver.md#gpu-path) |
| **mpi4py / MPI** | Master/worker spawn (`COMM_WORLD.Spawn`), broadcast, dynamic dispatch, gather | [parallel execution](../concepts/parallel-execution.md) |
| **numpy / scipy.interpolate** | Geometry bookkeeping; borehole & plot interpolation | model API |
| **matplotlib** | Result plots (formation panel + stacked log tracks via `twiny`) | [io formats](../../docs/io-formats.md) |

## Notes

- **Backend split:** Netgen = default for `dip==0` (fast, direct, no `.msh`
  round-trip); Gmsh = required for `dip!=0` (half-sphere domain, revolved
  borehole, rotated dipping boxes). Gmsh round-trips through `./tmp/fm_<rank>.msh`,
  so slow disks penalize it.
- **GPU is optional:** a failed `import ngsolve.ngscuda` silently downgrades
  `gpu_workers` to 0.
- **`setup.py` under-declares:** it lists only numpy/scipy/matplotlib/mpi4py/gmsh
  — the solve path *also* needs Netgen + NGSolve.
- MPI runtime: MPICH (Linux) / Microsoft MPI (Windows).

## Links

- Package: [ReMo3D](remo3d.md). Method: [forward modeling](../concepts/forward-modeling.md).
- Environment: [`../../docs/installation-and-environment.md`](../../docs/installation-and-environment.md).
