# Changelog

## v1.3.0

- Converted the main API from a mostly script-style workflow into the `Model`
  class.
- Split worker management into `initialize_workers`, `simulate_logs`, and
  `shutdown_workers` so repeated simulations can reuse the worker pool.
- Made the public workflow better suited to inversion and optimization loops
  where many forward solves are evaluated against changing model parameters.

## v1.2.0

- Standardized the output contract of the Gmsh and Netgen paths.
- Moved conversion into NGSolve mesh objects into the worker so the mesh
  generator modules stay focused on local geometry and mesh construction.
- Simplified the worker-side numerical pipeline by making the mesh backend and
  solver backend agree on a common interface.

## v1.1.0

- Restructured the repository by splitting the original monolithic code into
  `remo3d.py`, `gmsh_functions.py`, `netgen_functions.py`, and
  `ngsolve_functions.py`.
- Added batch mode so adjacent measurement depths can share one local task.
- Added the single-electrode optimization so multiple tools can share a solve
  whenever their current-electrode configuration matches.
- Added `Changelog.md` to track versioned technical changes.

## v1.0.0

- Initial public release of ReMo3D as described in the publication cited below.
- Established the core forward-modeling workflow: tool parsing, local geometry
  construction, finite-element solution of the resistivity problem, and
  synthetic normal and lateral log generation.

Reference:

- Wilkosz, M. (2022). ReMo3D - an open-source Python package for 2D and 3D
  simulation of normal and lateral resistivity logs. Geology, Geophysics and
  Environment, 48(2), 195-211. https://doi.org/10.7494/geol.2022.48.2.195
