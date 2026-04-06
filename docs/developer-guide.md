# Developer Guide - Extending the Framework

## 10.1 Adding a New Logging Tool Type

Current parser assumptions:

- exactly three electrode symbols
- exactly two positive spacings
- symbols drawn from `A`, `B`, `M`, `N`

To extend beyond that:

1. update `set_tools_parameters` and `_set_tool_parameters`
2. decide how the measurement point is defined for the new geometry
3. define source-term layout for the new electrode set
4. derive the new geometric factor
5. update any documentation and examples that assume three electrodes

If the new tool cannot be reduced to the current three-electrode layout, the
`tools` dictionary shape will also need to change.

## 10.2 Adding a New Mesh Generator Backend

A backend compatible with the current worker pipeline needs two functions:

1. data selection function returning local geometry plus conductivity ordering
2. mesh construction function returning a Netgen-compatible mesh object

The worker currently expects the selection stage to produce:

```text
(local_formation_geometry, local_borehole_geometry, sigma)
```

and the mesh stage to return something that can be wrapped as:

```python
mesh = ngs.Mesh(mesh)
```

To register a new backend, add a new branch in `workers/worker.py` and extend
validation in `Model.simulate_logs`.

## 10.3 Modifying the PDE Formulation

The bilinear form is defined in:

- `remo3d/ngsolve_functions.py`
- `remo3d/ngsolve_functions_gpu.py`

Current forms:

- 2D axisymmetric: `2*pi*x*sigma*grad(u).grad(v)`
- 3D: `sigma*grad(u).grad(v)`

To add physics such as anisotropy or frequency dependence:

1. change the bilinear form in both CPU and GPU versions
2. decide whether `sigma` stays scalar or becomes tensor-valued / complex-valued
3. check whether `AddPointSource` is still the right source model
4. revisit the apparent-resistivity post-processing formula if the measured
   quantity changes

## 10.4 Adding New Boundary Condition Types

Boundary conditions are encoded in two places:

1. mesh generators define physical or boundary groups
2. solver setup passes the Dirichlet group name or index into `ngs.H1`

To add mixed or Robin boundaries:

- create a distinct boundary label in the mesh generator
- include the corresponding boundary integral in the bilinear and or linear form
- keep only the true Dirichlet part in the `dirichlet=` argument

## 10.5 Modifying the Solver

Common extension points:

- element order: change `order=3` in both solver modules
- Krylov solver: replace `CGSolver` with another NGSolve solver
- stopping criteria: change `maxsteps` and any convergence controls
- adaptive refinement: add an outer refine-solve loop around the current solve

Whenever the CPU solver is changed, the GPU solver should usually be updated in
parallel so feature parity stays intact.

## 10.6 Extending the Parallel Execution Model

If new data must reach every worker:

1. add its shape to the array-shape broadcast if needed
2. add a `Bcast` or `bcast` call in `Model.simulate_logs`
3. add a matching receive in `workers/worker.py`
4. thread the new data through the per-task logic

If task granularity changes, update both:

- `_prepare_simulation_depths_and_tasks`
- the worker logic that unpacks each task entry

Potential future work:

- worker-side retry logic
- fault-tolerant task resubmission
- cluster-aware MPI launch documentation

## 10.7 Adding New Output or Post-Processing Capabilities

The clean extension points are:

- after result assembly in `Model.simulate_logs`
- inside `Model.save_results`

Examples:

- derived quantities such as apparent conductivity
- new figure layouts or additional tracks
- export to LAS, DLIS, or custom downstream formats

Any new export should preserve the existing `self.logs` contract unless a major
API change is intended.

## 10.8 Using ReMo3D in Inversion Workflows

The class-based API introduced in the current repo structure is already suitable
for inversion-style loops because worker initialization is separated from the
solve call.

Recommended pattern:

```python
model = Model(tools)
model.set_model_parameters(formation_model, borehole_model, dip=dip)
model.initialize_workers(cpu_workers=..., gpu_workers=...)

for candidate_model in candidates:
    model.set_model_parameters(...)
    model.simulate_logs(measurement_depths, ...)
    objective = misfit(model.logs, observed_logs)

model.shutdown_workers()
```

Benefits:

- worker processes are reused across iterations
- the public `self.logs` structure is already convenient for objective-function
  evaluation
- model parameters can be passed programmatically instead of through files
