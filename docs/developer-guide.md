# Developer Guide - Extending the Framework

This guide assumes you already know where the main modules live. If not, start
with [`architecture.md`](architecture.md#13-module-dependency-graph) and then
jump back here.

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
`tools` dictionary shape will also need to change, so also revisit
[`data-structures.md`](data-structures.md#121-tools_parameters).

## 10.2 Adding a New Mesh Generator Backend

A backend compatible with the current worker pipeline needs two functions:

1. data selection function returning local geometry plus conductivity ordering
2. mesh construction function returning a Netgen-compatible mesh object

### Skeleton module

A minimal template looks like this:

```python
# custom_mesh_functions.py
import numpy as np


def SelectDataRange(
    borehole_geometry,
    formation_parameters,
    mud_resistivity,
    simulation_depth,
    domain_radius,
    active_geometry_window=0.995,
):
    """Return local geometry arrays plus conductivity ordering."""
    # Shift to local coordinates.
    local_borehole_geometry = borehole_geometry.copy()
    local_borehole_geometry[:, 0] -= simulation_depth

    # Replace this with your own clipping and region-building logic.
    local_formation_geometry = formation_parameters.copy()
    local_formation_geometry[:, :2] -= simulation_depth

    sigma = [1 / mud_resistivity]
    sigma += list(1 / np.ndarray.flatten(local_formation_geometry[:, 3:5])[~np.isnan(np.ndarray.flatten(local_formation_geometry[:, 3:5]))])

    return local_formation_geometry, local_borehole_geometry[:, :2], sigma


def ConstructModel(
    domain_radius,
    tool_geometry,
    source_terms,
    formation_geometry,
    borehole_geometry,
    file_number=None,
    output_folder_path="./meshfiles",
    output_mode="variable",
):
    """Return a mesh object that ngsolve can wrap as ngs.Mesh(mesh)."""
    # Build your geometry here and return a Netgen-compatible mesh.
    raise NotImplementedError("Implement your custom mesh builder")
```

### Required signatures and return contract

The worker expects the selection stage to return:

```text
(local_formation_geometry, local_borehole_geometry, sigma)
```

and the construction stage to return a mesh object that can be wrapped as:

```python
mesh = ngs.Mesh(mesh)
```

### Exact integration points

#### In `workers/worker.py`

Add a new import near the top:

```python
import custom_mesh_functions as cmf
```

Then add a new backend branch next to the existing `gmsh` and `netgen` branches:

```python
elif mesh_generator == "custom":
    local_formation_geometry, local_borehole_geometry, sigma = cmf.SelectDataRange(
        borehole_geometry,
        formation_parameters,
        mud_resistivities[depth_index],
        simulation_depths[depth_index],
        domain_radius,
    )
    mesh = cmf.ConstructModel(
        domain_radius,
        tool_geometry,
        source_terms,
        local_formation_geometry,
        local_borehole_geometry,
    )
    dirichlet_boundary = [2]  # or whatever your backend uses
```

#### In `remo3d/remo3d.py`

Extend the backend validation in `simulate_logs`:

```python
if mesh_generator == "auto":
    mesh_generator = "netgen" if np.isclose(self.dip_deg, 0) else "gmsh"

if mesh_generator not in ["gmsh", "netgen", "custom"]:
    raise ValueError("Unsupported mesh generator")

if ~np.isclose(self.dip_deg, 0) and mesh_generator not in ["gmsh", "custom"]:
    raise ValueError("The selected mesh generator does not support 3D models")
```

## 10.3 Modifying the PDE Formulation

The bilinear form is defined in:

- `remo3d/ngsolve_functions.py`
- `remo3d/ngsolve_functions_gpu.py`

Current forms:

- 2D axisymmetric: `2*pi*x*sigma*grad(u).grad(v)`
- 3D: `sigma*grad(u).grad(v)`

### Example: anisotropic conductivity

#### Before

```python
if model_dimensionality==2:
    a += 2*np.pi*ngs.grad(u)*ngs.grad(v)*ngs.x*sigma*ngs.dx
elif model_dimensionality==3:
    a += ngs.grad(u)*ngs.grad(v)*sigma*ngs.dx
```

#### After: CPU version

```python
if model_dimensionality == 2:
    sigma_tensor = ngs.CoefficientFunction(
        (sigma_r, 0,
         0, sigma_z),
        dims=(2, 2),
    )
    a += 2*np.pi * ngs.InnerProduct(sigma_tensor * ngs.grad(u), ngs.grad(v)) * ngs.x * ngs.dx
elif model_dimensionality == 3:
    sigma_tensor = ngs.CoefficientFunction(
        (sigma_x, 0, 0,
         0, sigma_y, 0,
         0, 0, sigma_z),
        dims=(3, 3),
    )
    a += ngs.InnerProduct(sigma_tensor * ngs.grad(u), ngs.grad(v)) * ngs.dx
```

#### After: GPU version

The GPU file keeps the same bilinear-form code. Only the matrix assembly and
solve transport differ later in the function, so apply the same bilinear-form
change there as well.

### Checklist when changing the PDE

1. update both CPU and GPU solver files
2. decide whether `sigma` is still scalar or must become tensor-valued
3. confirm that the point-source model still matches the physics
4. revisit the post-processing formula if the measured quantity is no longer a
   pure DC potential difference

For the mathematical background behind the current forms, see
[`mathematical-foundations.md`](mathematical-foundations.md#22-2d-axisymmetric-formulation).

## 10.4 Adding New Boundary Condition Types

Boundary conditions are encoded in two places:

1. mesh generators define physical or boundary groups
2. solver setup passes the Dirichlet group name or index into `ngs.H1`

### Example: adding a Robin boundary condition

#### Step 1: create a new boundary label in the mesh generator

For the Gmsh 2D path, the current classification is just Dirichlet versus
Neumann. To add a Robin boundary on the borehole wall, split the classification:

```python
dirichlet_boundaries = []
neumann_boundaries = []
robin_boundaries = []

for line in lines:
    nodes = gmsh.model.mesh.getNodes(dim=1, tag=line, includeBoundary=True)
    coordinates = nodes[1].reshape((int(np.shape(nodes[1])[0]/3), 3))
    R = np.sqrt(coordinates[:, 0]**2 + coordinates[:, 1]**2)
    if np.allclose(R, domain_radius):
        dirichlet_boundaries.append(line)
    elif line in lines_at_borehole_boundary:
        robin_boundaries.append(line)
    else:
        neumann_boundaries.append(line)

rb = gmsh.model.addPhysicalGroup(1, robin_boundaries, 3)
gmsh.model.setPhysicalName(1, rb, "robin_boundary")
```

For the Netgen path, the equivalent idea is to assign a new boundary-condition
integer to the borehole-wall segments and then expose that label to NGSolve.

#### Step 2: add the Robin term in the solver

```python
alpha = 0.25
if model_dimensionality == 2:
    a += alpha * u * v * ngs.ds("robin_boundary")
elif model_dimensionality == 3:
    a += alpha * u * v * ngs.ds("robin_boundary")
```

#### Step 3: keep only true Dirichlet boundaries in `ngs.H1`

Do not put `robin_boundary` into the `dirichlet=` argument. Robin boundaries are
natural boundary terms added to the bilinear form.

## 10.5 Modifying the Solver

Common extension points:

- element order: change `order=3` in both solver modules
- Krylov solver: replace `CGSolver` with another NGSolve solver
- stopping criteria: change `maxsteps` and add residual checks
- adaptive refinement: add an outer refine-solve loop around the current solve

### Example: direct factorization instead of CG

#### Current CG block

```python
c = ngs.Preconditioner(a, preconditioner)
a.Assemble()
gfu = ngs.GridFunction(fes)

inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
gfu.vec.data = inv * f.vec
```

#### Direct-solver replacement

```python
a.Assemble()
gfu = ngs.GridFunction(fes)

inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
gfu.vec.data = inv * f.vec
```

If your NGSolve build exposes an `ngs.directsolve` helper, this is the same
swap point where it belongs.

### Example: change element order

```python
fes = ngs.H1(mesh, order=4, dirichlet=dirichlet_boundary, autoupdate=True)
```

### Example: add a convergence check for CG

```python
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
gfu.vec.data = inv * f.vec

residual = f.vec.CreateVector()
residual.data = f.vec - a.mat * gfu.vec
residual_norm = ngs.Norm(residual)
if residual_norm > 1e-8:
    print(f"Warning: large residual after solve: {residual_norm}")
```

Whenever the CPU solver is changed, the GPU solver should usually be updated in
parallel so feature parity stays intact.

## 10.6 Extending the Parallel Execution Model

If new data must reach every worker:

1. add it to the master-side broadcast in `simulate_logs`
2. add the matching receive in `workers/worker.py`
3. thread it into the task logic
4. consume it where the solver or mesh backend needs it

### Example: add a new broadcasted solver option

Suppose you want a configurable `solver_maxsteps`.

#### In `remo3d/remo3d.py`

Add the parameter to `simulate_logs` and broadcast it:

```python
def simulate_logs(..., condense=True, solver_maxsteps=1000):
    ...
    self.comm.bcast(solver_maxsteps, root=MPI.ROOT)
```

#### In `workers/worker.py`

Receive it next to the other scalar options:

```python
solver_maxsteps = int()
...
solver_maxsteps = comm.bcast(solver_maxsteps, root=0)
```

and then pass it onward:

```python
fes, gfu = ngsf.SolveBVP(
    mesh,
    sigma,
    tool_geometry,
    source_terms,
    dirichlet_boundary,
    preconditioner,
    condense,
    maxsteps=solver_maxsteps,
)
```

#### In the solver modules

Extend the signature and use the value:

```python
def SolveBVP(..., condense, maxsteps=1000):
    ...
    inv = ngs.CGSolver(a.mat, c.mat, maxsteps=maxsteps)
```

This is the pattern to follow for any new solver or meshing option that must be
visible on every worker.

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

The class-based API is already suitable for inversion-style loops because worker
initialization is separated from the solve call.

### Minimal worked example

```python
from remo3d import Model
import numpy as np


def l2_misfit(simulated_logs, observed_logs):
    misfit = 0.0
    for tool, simulated in simulated_logs.items():
        observed = observed_logs[tool]
        residual = simulated[:, 1] - observed[:, 1]
        misfit += float(np.nansum(residual**2))
    return misfit


tools = ["B5.7A0.4M", "A2.0M0.5N"]
measurement_depths = np.arange(0.0, 10.1, 0.5)

observed_logs = {
    tool: np.vstack([measurement_depths, np.ones_like(measurement_depths) * 10.0]).T
    for tool in tools
}

model = Model(tools)
model.initialize_workers(cpu_workers=2, gpu_workers=0)

try:
    for uz_value in [8.0, 10.0, 12.0]:
        formation_model = np.array([
            [0.0, 4.0, np.nan, np.nan, 5.0],
            [4.0, 7.0, 0.30, 3.0, uz_value],
            [7.0, 12.0, np.nan, np.nan, 8.0],
        ], dtype=float)
        borehole_model = np.array([
            [0.0, 0.10, 1.0],
            [12.0, 0.10, 1.0],
        ], dtype=float)

        model.set_model_parameters(
            formation_model,
            borehole_model,
            borehole_geometry_type="radius",
            dip=0,
        )
        model.simulate_logs(
            measurement_depths,
            domain_radius=20,
            batch_size=3,
            mesh_generator="netgen",
            preconditioner="multigrid",
            condense=True,
        )
        objective = l2_misfit(model.logs, observed_logs)
        print(f"UZ value {uz_value:.1f} -> objective {objective:.6f}")
finally:
    model.shutdown_workers()
```

Why this pattern matters:

- workers are reused across candidate models
- only the model parameters and results change inside the loop
- `model.logs` already has the right structure for a simple objective function

## See Also

- [`architecture.md`](architecture.md#13-module-dependency-graph): where the extension points live in the module tree.
- [`model-api.md`](model-api.md#314-simulate_logs): the public orchestration entry point that feeds most extension work.
- [`solver.md`](solver.md#62-solvebvp-cpu-version): current solver pipeline before you modify it.
- [`parallel-execution.md`](parallel-execution.md#72-mpi-communication-protocol): how new data moves from the master to the workers.

