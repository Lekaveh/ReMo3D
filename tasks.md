# ReMo3D Documentation Tasks

This task list covers the creation of complete, developer- and mathematician-oriented documentation for the ReMo3D framework. The goal is to enable developers and mathematicians to understand, modify, and extend every aspect of the package.

---

## 1. Project Overview & Architecture

- [x] **1.1** Write a high-level architecture document describing the overall structure of ReMo3D: the `Model` class as orchestrator, the mesh generation layer (Gmsh + Netgen), the FEM solver layer (NGSolve), and the MPI-based parallel execution layer (worker processes).
- [x] **1.2** Create an architecture diagram (text-based or image) showing the data flow: user input -> tool parsing -> model parameter setup -> worker spawning -> mesh generation -> BVP solving -> result gathering -> output/visualization.
- [x] **1.3** Document the module dependency graph: `remo3d.py` depends on `gmsh_functions.py`, `netgen_functions.py`, `ngsolve_functions.py`, `ngsolve_functions_gpu.py`, and `workers/worker.py`. Show which external libraries each module imports.
- [x] **1.4** Write a glossary of domain-specific terms used throughout the codebase (e.g., normal log, lateral log, resistivity, conductivity, formation, filtration zone, undisturbed zone, mud resistivity, caliper, dip angle, geometric factor, electrode configuration, BVP, preconditioner, static condensation).

---

## 2. Mathematical Foundations

- [x] **2.1** Document the governing PDE for electrical resistivity logging: the Laplace/Poisson equation for the electric potential in a heterogeneous conductivity medium.
- [x] **2.2** Derive and document the 2D axisymmetric formulation used in `SolveBVP` (the `2*pi*x` weighting factor in the bilinear form `a += 2*np.pi*grad(u)*grad(v)*x*sigma*dx`) and explain why it arises from the cylindrical symmetry assumption.
- [x] **2.3** Derive and document the full 3D formulation used when `dip != 0` (the standard `grad(u)*grad(v)*sigma*dx` bilinear form).
- [x] **2.4** Document the point source model: how `AddPointSource` injects a Dirac delta source into the FEM linear form by evaluating shape functions at the source point, and the physical meaning (current injection electrode).
- [x] **2.5** Document the boundary conditions: Dirichlet boundary (zero potential at the outer domain boundary) and Neumann boundary (zero normal current flux at symmetry planes / internal interfaces).
- [x] **2.6** Explain the geometric factor computation for each electrode configuration (single-current-electrode with 2 potential electrodes AM/AN or BM/BN; two-current-electrode with 1 potential electrode AN/BN or AM/BM). Derive the formulas `4*pi*AM*AN/(AN-AM)` etc. from the potential of a point source in a homogeneous medium.
- [x] **2.7** Document the resistivity measurement formula: how measured apparent resistivity is computed from the potential difference at measuring electrodes multiplied by the geometric factor, including the `division by two` correction for 3D half-space models.
- [x] **2.8** Document the static condensation technique (`condense=True`): what unknowns are eliminated, how `harmonic_extension`, `harmonic_extension_trans`, and `inner_solve` operators work mathematically, and the performance benefit.
- [x] **2.9** Document the role of the preconditioner choices (`"local"` vs `"multigrid"`) in the CG solver and their impact on convergence for the resistivity problem.
- [x] **2.10** Document the simulation domain geometry: why a circular (2D) / spherical (3D) domain of configurable radius is used, and how the domain radius affects accuracy and computational cost.

---

## 3. Model Class (`remo3d.py`) - Core API

- [x] **3.1** Write comprehensive API reference for the `Model` class constructor `__init__`: parameters, attributes initialized, and the tool parsing flow.
- [x] **3.2** Document the `compute_synthetic_logs` classmethod: its role as a convenience wrapper that orchestrates the full pipeline (init -> set model -> init workers -> simulate -> shutdown).
- [x] **3.3** Document the tool naming convention in detail: the 3-character electrode pattern (e.g., `"B5.7A0.4M"`), how electrode symbols (A, B, M, N) and inter-electrode distances are parsed by `_str2float` and `itertools.groupby`.
- [x] **3.4** Document the `_set_tool_parameters` method: how electrode positions are computed relative to the measurement point, how the geometric factor is calculated for each of the 4 possible electrode configurations (missing A, B, M, or N), and the structure of the returned `tool_parameters` array (rows: geometry + source_terms, columns: 3 electrodes + geometric_factor/depth_shift).
- [x] **3.5** Document the `force_single_electrode_configuration` optimization: how two-current-electrode tools (with both A and B) are converted to equivalent single-current-electrode tools by swapping A/B with M/N, and why this enables the single-electrode computation mode (`sec`).
- [x] **3.6** Document the single-electrode computation mode (`sec`): when all tools share the same current electrode configuration, multiple tools can share one mesh and one BVP solve per simulation depth, drastically reducing computation.
- [x] **3.7** Document `set_model_parameters`: how formation and borehole models are loaded from files or arrays, unit conversion via `conversion_table`, validation checks, and dip angle setup.
- [x] **3.8** Document the formation model data format: the tab-delimited file with columns TOP, BOTTOM, FZ_RADIUS, FZ_VALUE, UZ_VALUE, their units row, and the meaning of NaN for layers without filtration zones.
- [x] **3.9** Document the borehole model data format: the tab-delimited file with columns DEPT, CALM (caliper), RM (mud resistivity), their units row, and how diameter is converted to radius.
- [x] **3.10** Document `_check_model_geometry`: the validation that borehole radius never exceeds the filtration zone radius.
- [x] **3.11** Document `initialize_workers` and `shutdown_workers`: MPI spawning of worker processes, GPU availability detection, and the two-level message-passing protocol (setup sentinel, task dispatch, stop sentinel).
- [x] **3.12** Document `_prepare_simulation_depths_and_tasks`: the batch mode logic, how simulation depths are computed from measurement depths and tool depth shifts, how tasks are grouped into batches, and the nested task structure `[batch_index, combined_tools, batch_modelling_tasks]`.
- [x] **3.13** Document `_add_points_to_borehole`: why additional points are interpolated into sparse borehole geometry for 3D meshing, and the maximal_distance threshold.
- [x] **3.14** Document `simulate_logs`: the main simulation loop including data broadcasting via MPI, the master-worker task dispatch pattern (dynamic load balancing), result gathering, and assembly of final log arrays.
- [x] **3.15** Document `save_results`: the output file format (tab-delimited text), the visualization system (matplotlib-based model cross-section + log tracks), and all optional plotting parameters.

---

## 4. Mesh Generation - Gmsh (`gmsh_functions.py`)

- [x] **4.1** Document `SelectGmshBoreholeDataRange`: how borehole geometry is clipped to the simulation domain, the domain-line intersection algorithm, and the difference between 2D (circular domain) and 3D (box domain) clipping.
- [x] **4.2** Document `SelectGmshFormationDataRange`: how formation layers are selected and trimmed to the active geometry window, how filtration zones outside the domain are removed, and how the top/bottom boundaries are extended slightly beyond the domain.
- [x] **4.3** Document `SelectGmshDataRange`: the combined data selection and conductivity distribution assembly (conversion from resistivity to conductivity).
- [x] **4.4** Document `ReadGmsh`: the custom Gmsh `.msh` file parser (based on Netgen's `ReadGmsh`), how physical groups, nodes, and elements (1D segments, 2D triangles/quads, 3D tetrahedra/hexahedra/prisms/pyramids) are mapped to a Netgen mesh object, including second-order element support.
- [x] **4.5** Document `ConstructGmsh2dModel`: step-by-step construction of the 2D axisymmetric mesh using Gmsh OCC kernel: points at borehole axis, borehole boundary, circular arcs, domain surface, layer splitting via rectangle intersections, filtration zone handling, mesh size fields (linear distance + electrode proximity refinement), boundary classification (Dirichlet vs Neumann), physical group assignment, and conversion to Netgen format.
- [x] **4.6** Document `ConstructGmsh3dModel`: step-by-step construction of the 3D half-space mesh: half-sphere domain, borehole template via revolution, domain splitting into borehole and formation volumes, dipping layer creation via rotated boxes and cylinder intersections, mesh size fields, boundary classification, physical group assignment, and conversion to Netgen format.
- [x] **4.7** Document the mesh size control strategy: the `MathEval` fields `"x + 0.1"` (linear radial refinement) and `"(x^2 + (y+pos)^2)/2 + 0.01"` (electrode proximity refinement), how they are combined via `Min` field, and how they affect solution accuracy.

---

## 5. Mesh Generation - Netgen (`netgen_functions.py`)

- [x] **5.1** Document `SelectNetgenDataRange`: the data selection algorithm for the Netgen 2D mesh generator, including borehole geometry clipping (with domain-line intersection), formation geometry filtering by active geometry window, filtration zone presence checking, zone removal logic, and conductivity distribution assembly.
- [x] **5.2** Document `ConstructNetgen2dModel`: the detailed 2D mesh construction algorithm using Netgen's `SplineGeometry`, including: the point data structure `[index, r, z]`, the line data structure `[index, start-point, end-point, boundary-condition, domain-left, domain-right]`, the construction of points at borehole axis / borehole wall / filtration boundaries / layer boundary endpoints / domain boundary (with circular approximation), and how all lines (vertical at axis, borehole wall, filtration boundaries; horizontal at layer boundaries; circular at domain boundary) are assembled with correct domain indices.
- [x] **5.3** Document the domain boundary approximation: how existing points at the circular domain boundary are converted to polar coordinates, how additional points are interpolated to approximate the circular shape (every 9 degrees), and how they are converted back to Cartesian coordinates.
- [x] **5.4** Document the region indexing scheme: how the borehole (region 1), formation layers (regions 2+), and filtration/undisturbed zones are assigned integer indices that map to entries in the conductivity distribution array.
- [x] **5.5** Compare Gmsh vs Netgen mesh generation: when each is used (`netgen` for 2D by default, `gmsh` for 3D or when explicitly specified), advantages/disadvantages, and output format differences.

---

## 6. FEM Solver (`ngsolve_functions.py`, `ngsolve_functions_gpu.py`)

- [x] **6.1** Document `AddPointSource`: how a Dirac delta source is discretized in the FEM framework by finding the mesh element containing the source point, evaluating basis function shapes, and adding weighted contributions to the load vector.
- [x] **6.2** Document `SolveBVP` (CPU version): the complete FEM solution pipeline: H1 finite element space (order 3), trial/test functions, bilinear form assembly (2D axisymmetric vs 3D), linear form with point sources, preconditioner setup, CG solver (max 1000 iterations), static condensation post-processing.
- [x] **6.3** Document `SolveBVP` (GPU version): differences from the CPU version: `CreateDeviceMatrix`/`CreateDeviceVector` calls, CUDA-accelerated CG solve, and when GPU acceleration provides benefit.
- [x] **6.4** Document the choice of H1 finite element space with order 3: why cubic elements are used, how this affects accuracy near point sources, and the trade-off with computational cost.
- [x] **6.5** Document the solver output: the `fes` (finite element space) and `gfu` (grid function) return values, and how the solution potential is evaluated at arbitrary points using `gfu(mesh(x, y))` or `gfu(mesh(x, y, z))`.

---

## 7. Parallel Execution (`workers/worker.py`)

- [x] **7.1** Document the worker process lifecycle: MPI parent connection, receive `solve_on` configuration, enter outer task loop (level 1: simulation batches), receive broadcasted data, enter inner task loop (level 2: individual mesh+solve tasks), report results, disconnect.
- [x] **7.2** Document the MPI communication protocol in detail: `comm.bcast` for global data (array shapes, formation parameters, borehole geometry, mud resistivities, simulation depths, dip, tools, domain_radius, mesh_generator, preconditioner, condense, task_list), `comm.Bcast` for contiguous arrays, `comm.sendrecv`/`comm.send`/`comm.recv` for task dispatch, `comm.gather` for result collection, `comm.barrier` for synchronization.
- [x] **7.3** Document the two-level task loop: level 1 receives batch configuration data, level 2 processes individual simulation tasks within the batch. Explain the `StopIteration` sentinel pattern for graceful shutdown.
- [x] **7.4** Document the worker's computation pipeline for each task: data range selection -> mesh construction -> NGSolve mesh conversion -> BVP solve for each unique current electrode configuration -> resistivity computation for each measurement -> result collection.
- [x] **7.5** Document error handling: the `try/except` block that catches meshing or solving failures and reports `np.nan` for all affected measurements, and implications for result quality.
- [x] **7.6** Document the dynamic load balancing: how the master process dispatches tasks on-demand to any available worker (via `recv(source=MPI.ANY_SOURCE)`), ensuring even workload distribution.

---

## 8. Input/Output Formats

- [x] **8.1** Document the formation model file format with a complete specification: header row (column names), units row, data rows with tab delimiters. Define each column: TOP (top of layer), BOTTOM (bottom of layer), FZ_RADIUS (filtration zone radius, NaN if absent), FZ_VALUE (filtration zone resistivity), UZ_VALUE (undisturbed zone resistivity).
- [x] **8.2** Document the borehole model file format: header row, units row, data rows. Define each column: DEPT (depth), CALM (caliper - borehole diameter), RM (mud resistivity). Document supported units (M, DM, CM, MM, IN, FT).
- [x] **8.3** Document the output results file format: header with tool names, units row, tab-delimited depth+resistivity columns.
- [x] **8.4** Document the visualization output: the PNG plot structure (formation model panel + log track panels), colorbar, and all configurable visual parameters.

---

## 9. Configuration & Parameters

- [x] **9.1** Create a complete parameter reference table for `compute_synthetic_logs` with: parameter name, type, default value, allowed values, description, and impact on results/performance.
- [x] **9.2** Document the `domain_radius` parameter: physical meaning, how it affects accuracy (larger = more accurate but slower), recommended values for different model geometries, and the 75% warning threshold.
- [x] **9.3** Document the `batch_size` parameter: how batching works (adjacent measurement points share one mesh), the trade-off between batch size and accuracy (larger batches use one average simulation depth), and performance impact.
- [x] **9.4** Document the `mesh_generator` parameter: `"auto"` selection logic, capabilities of each generator, and compatibility with 2D/3D models.
- [x] **9.5** Document the `preconditioner` parameter: `"local"` vs `"multigrid"` options, their mathematical properties, and when to use each.
- [x] **9.6** Document the `condense` parameter: static condensation on/off, performance impact, and numerical equivalence.
- [x] **9.7** Document CPU/GPU worker configuration: how `cpu_workers` and `gpu_workers` interact, minimum requirements, CUDA detection, and performance scaling guidelines.

---

## 10. Developer Guide - Extending the Framework

- [x] **10.1** Write a guide on how to add a new logging tool type: what changes are needed in the tool parser, how to define new electrode configurations beyond 3-electrode setups, and how geometric factors would need to be modified.
- [x] **10.2** Write a guide on how to add a new mesh generator backend: what interface a mesh generator module must implement (data range selection function + mesh construction function), how to register it in `simulate_logs`, and how the output must be compatible with NGSolve.
- [x] **10.3** Write a guide on how to modify the PDE formulation: where the bilinear form is defined in `SolveBVP`, how to change the governing equation (e.g., adding anisotropic conductivity, frequency-dependent terms), and how this affects the point source and boundary conditions.
- [x] **10.4** Write a guide on how to add new boundary condition types: where boundary conditions are set in the mesh generators and solver, how to implement mixed or Robin boundary conditions.
- [x] **10.5** Write a guide on how to modify the solver: changing element order, switching solvers (CG to GMRES, direct solvers), adjusting convergence criteria, and using adaptive mesh refinement.
- [x] **10.6** Write a guide on how to extend the parallel execution model: adding new data to the MPI broadcast, modifying the task structure, implementing fault tolerance, and scaling to cluster environments.
- [x] **10.7** Write a guide on how to add new output/post-processing capabilities: computing derived quantities (e.g., apparent conductivity, formation factor), adding new visualization types, exporting to industry-standard formats (LAS, DLIS).
- [x] **10.8** Write a guide on how to integrate ReMo3D into inversion workflows: using the separated `initialize_workers`/`simulate_logs`/`shutdown_workers` API (as noted in Changelog v1.3.0), passing model parameters programmatically, and interpreting results for objective function evaluation.

---

## 11. Code Walkthrough - Key Algorithms

- [x] **11.1** Write a step-by-step walkthrough of a complete 2D simulation from Example_01: from tool parsing through mesh generation to final resistivity computation, with intermediate data shapes and values.
- [x] **11.2** Write a detailed walkthrough of the Netgen 2D mesh construction algorithm: how points, lines, and regions are systematically built from formation and borehole geometry, with diagrams showing the geometric construction.
- [x] **11.3** Write a detailed walkthrough of the Gmsh 2D mesh construction: how OCC boolean operations (intersect, cut) create the layered formation geometry, and how physical groups map to conductivity regions.
- [x] **11.4** Write a detailed walkthrough of the Gmsh 3D mesh construction: revolution of the borehole template, sphere/cylinder intersections for dipping layers, and the half-space symmetry exploitation.
- [x] **11.5** Write a walkthrough of the task preparation algorithm (`_prepare_simulation_depths_and_tasks`): how simulation depths are computed from measurement depths and tool offsets, how the single-electrode optimization groups tasks, and how batching creates compound tasks.
- [x] **11.6** Write a walkthrough of the master-worker communication protocol: the full lifecycle from `initialize_workers` through setup messages, data broadcast, task dispatch loop, result gathering, to `shutdown_workers`.

---

## 12. Data Structures Reference

- [x] **12.1** Document the `tools_parameters` dictionary: key = tool name string, value = 2x4 numpy array where row 0 = [electrode_z1, electrode_z2, electrode_z3, geometric_factor], row 1 = [source_term1, source_term2, source_term3, depth_shift].
- [x] **12.2** Document the `formation_model` array: shape (n_layers, 5), columns = [top, bottom, fz_radius, fz_resistivity, uz_resistivity], units in meters and ohm-m after conversion.
- [x] **12.3** Document the `borehole_model` array: shape (n_points, 3), columns = [depth, radius, mud_resistivity], units in meters and ohm-m after conversion.
- [x] **12.4** Document the `local_formation_geometry` array (Netgen): shape (n_layers, 5), columns = [top, bottom, fz_radius, region_index_1, region_index_2], and how region indices map to the conductivity distribution list.
- [x] **12.5** Document the `local_formation_geometry` array (Gmsh): shape (n_layers, 3), columns = [top, bottom, fz_radius], and how the separate `formation_resistivity_distribution` array maps to physical groups.
- [x] **12.6** Document the task list structure: list of `[batch_index, combined_tools_array, batch_modelling_tasks]`, where `combined_tools_array` is a 2xN array of electrode positions and source terms, and `batch_modelling_tasks` is a nested list of `[depth_index, combined_tools, modelling_tasks]` with inner modelling_tasks being `[measurement_depth_index, tool_index, simulation_offset]`.
- [x] **12.7** Document the `logs` dictionary: key = tool name string, value = Nx2 numpy array with columns [measurement_depth, apparent_resistivity].
- [x] **12.8** Document the conductivity distribution list (`sigma`): `[1/mud_resistivity, 1/resistivity_region_1, 1/resistivity_region_2, ...]`, ordered by region index, and how it maps to the `CoefficientFunction` used in the bilinear form.

---

## 13. Installation, Dependencies & Environment

- [x] **13.1** Document all dependencies and their roles: `numpy` (numerical computation), `scipy` (interpolation), `matplotlib` (visualization), `mpi4py` (parallel execution), `gmsh` (Gmsh mesh generation), `netgen`/`ngsolve` (Netgen mesh generation + FEM solver), and optional `ngsolve.ngscuda` (GPU acceleration).
- [x] **13.2** Document the MPI runtime requirements: which MPI implementations are supported (MPICH on Linux, MS-MPI on Windows), how to install and configure them, and how `mpiexec` is used to launch the main process.
- [x] **13.3** Document the GPU setup: CUDA requirements, NGSolve CUDA build, how GPU availability is detected, and how to troubleshoot GPU-related issues.
- [x] **13.4** Document platform-specific installation notes for Ubuntu 18.04, Ubuntu 20.04, Windows 10 Pro, and Windows 11 Pro, including known issues and workarounds.

---

## 14. Examples & Tutorials

- [x] **14.1** Write a tutorial based on Example_01: minimal usage with required parameters only, explaining each line of code and the expected output.
- [x] **14.2** Write a tutorial based on Example_02: advanced usage with optional parameters, explaining how each parameter affects the simulation and visualization.
- [x] **14.3** Write a tutorial on creating custom formation models: how to define layers with and without filtration zones, how to set realistic resistivity values, and how layer geometry affects the simulation.
- [x] **14.4** Write a tutorial on creating custom borehole models: how to define varying caliper and mud resistivity, supported units, and common modeling scenarios (washout, mud cake).
- [x] **14.5** Write a tutorial on interpreting simulation results: what the output resistivity logs represent physically, how to compare with real log data, and common artifacts to watch for.
- [x] **14.6** Write a tutorial on the benchmark models: document each benchmark model (BM1, BM2, BM3 with varying dip angles, thin-bedded model), their purpose, and how to use them for validation.
- [x] **14.7** Write a performance tuning guide: how to choose optimal `cpu_workers`, `batch_size`, `domain_radius`, `mesh_generator`, and `preconditioner` for different model sizes and accuracy requirements.

---

## 15. Testing & Validation

- [x] **15.1** Document the existing benchmark models and their expected results: what analytical or reference solutions they validate against.
- [x] **15.2** Write a guide on how to create new validation tests: what reference solutions exist for resistivity logging, how to compare ReMo3D output with analytical solutions for simple models (e.g., homogeneous medium, single layer boundary).
- [x] **15.3** Document known limitations and edge cases: very thin layers, extreme resistivity contrasts, electrodes near domain boundary, high dip angles approaching 90 degrees, and numerical stability considerations.

---

## 16. Changelog & Version History

- [x] **16.1** Expand the changelog with detailed technical descriptions of each version's changes: v1.0.0 (initial release), v1.1.0 (package restructuring + batch mode + single-electrode optimization), v1.2.0 (standardized output + data conversion in worker), v1.3.0 (class-based API + separated worker lifecycle).
- [x] **16.2** Document the evolution of the API across versions and any breaking changes.

---

## 17. Code Quality & Conventions

- [x] **17.1** Document the coding conventions used in the project: variable naming patterns, array indexing conventions, coordinate system conventions (z = depth along borehole axis, r = radial distance; sign conventions for z in local coordinate frames).
- [x] **17.2** Document the coordinate system in detail: in 2D models, x = radial distance, y = depth (z-axis); in 3D models, z = borehole axis; how simulation depths are shifted to local coordinates (centered at the simulation point).
- [x] **17.3** Document error handling patterns: which functions raise `ValueError`, what conditions trigger errors, and the generic `try/except` in the worker that silently converts errors to NaN results.
