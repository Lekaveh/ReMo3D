---
title: ReMo3D (package)
type: entity
tags: [software, tool, fem]
sources: [repo-docs]
updated: 2026-07-15
---

# ReMo3D

*(seeded from repo — README.md, docs/.)*

Python package for generating **synthetic normal and lateral resistivity logs**
in 2D axisymmetric and 3D dipping earth models. Couples local mesh generation
with finite-element solves and distributed-memory worker processes.

## Stack

| Layer | Tech |
|---|---|
| Language | Python ≥ 3.7 |
| Parallelism | MPI (MPICH on Linux, MS-MPI on Windows) + worker processes |
| Meshing | Gmsh, Netgen |
| FEM solve | NGSolve (optional CUDA GPU workers) |

## Key modules (`../remo3d/`)

- `remo3d.py` — main `Model` API (`compute_synthetic_logs`).
- `ngsolve_functions.py` / `ngsolve_functions_gpu.py` — CPU/GPU solvers.
- `gmsh_functions.py`, `netgen_functions.py` — mesh generation.
- `sensitivity.py` — sensitivity-analysis code *(active on `optim` branch)*.
- `workers/worker.py` — distributed worker.

## Provenance

- Original repo: `github.com/eMWu94/ReMo3D`; this fork: `github.com/Lekaveh/ReMo3D`.
- Funding: National Science Centre, Poland, grant 2020/37/N/ST10/03230.
- License: code GPL v2.1; data CC BY 4.0.
- Rough runtime: 2D model ~15–30 s / 100 points; 3D ~15–30 min (AMD Ryzen 2600
  class).

## Links

- Concepts: [resistivity logging](../concepts/resistivity-logging.md),
  [forward modeling](../concepts/forward-modeling.md),
  [mesh generation](../concepts/mesh-generation.md),
  [FEM solver](../concepts/fem-solver.md),
  [parallel execution](../concepts/parallel-execution.md),
  [performance & accuracy](../concepts/performance-and-accuracy.md).
- Dependencies: [numerical stack](numerical-stack.md).
- Full code docs: [`../../docs/README.md`](../../docs/README.md) (indexed in [[repo-docs]]).
