# FEM Solver

This page describes the implementation details of the finite-element solve. For
the derivation of the weak forms themselves, see
[`mathematical-foundations.md`](mathematical-foundations.md#22-2d-axisymmetric-formulation).

## 6.1 `AddPointSource`

`AddPointSource(f, position, fac, model_dimensionality)` discretizes a Dirac
source by modifying the assembled load vector directly.

Algorithm:

1. map the axial source position into a mesh point
2. fetch the containing volume element
3. get the element finite element and its DOF numbers
4. evaluate the local shape functions at the source point
5. add `fac * shape_value` into the corresponding load-vector entries

For 2D the source location is interpreted as `mesh(0, position)`. For 3D it is
interpreted as `mesh(0, 0, position)`.

## 6.2 CPU Solver Path

The CPU solver path in `ngsolve_functions.py` is split into reusable assembly
and per-RHS solve steps.

`AssembleSystem(...)` runs once per mesh/conductivity pair:

1. infer model dimensionality from `mesh.dim`
2. create an H1 finite-element space with the configured `fe_order`
3. define trial and test functions
4. build a bilinear form with optional static condensation
5. assemble the axisymmetric or 3D stiffness term
6. build the selected preconditioner
7. assemble the bilinear form
8. return `fes, a, c`

`SolveRHS(...)` runs once per source vector:

1. assemble an initially empty linear form
2. inject point sources with `AddPointSource`
3. solve the linear system with CG, up to 1000 steps
4. apply the shared static-condensation helper when `condense=True`
5. return `fes, gfu`

The actual solve call is:

```python
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
gfu = _condensed_solve(a, inv, f, fes, condense)
```

`SolveBVP(...)` remains as a compatibility wrapper that calls
`AssembleSystem(...)` and then `SolveRHS(...)`.

`_condensed_solve` is the single source of truth for the static-condensation
ordering. It applies `harmonic_extension_trans` to the RHS before the condensed
solve, then reconstructs internal DOFs with `harmonic_extension` and
`inner_solve`.

## 6.3 `SolveBVP` GPU Version

The GPU path in `ngsolve_functions_gpu.py` reuses the CPU `AssembleSystem(...)`
implementation so the finite-element space, bilinear form, preconditioner, and
static-condensation operators are built exactly the same way. Its `SolveRHS(...)`
then transfers the assembled matrix/preconditioner to the device for each RHS
solve.

The differences are:

- the sparse matrix is transferred with `CreateDeviceMatrix()`
- the preconditioner matrix is transferred with `CreateDeviceMatrix()`
- the load vector is copied with `CreateDeviceVector(copy=True)`
- CG is run against the device matrices and vector

Core device setup:

```python
adev = a.mat.CreateDeviceMatrix()
cdev = c.mat.CreateDeviceMatrix()
inv = ngs.CGSolver(adev, cdev, maxsteps=1000, printrates=False)
gfu = _condensed_solve(
    a,
    inv,
    f,
    fes,
    condense,
    rhs_vector_factory=lambda assembled_f: assembled_f.vec.CreateDeviceVector(copy=True),
)
```

The FE space, bilinear form, point-source assembly, and static-condensation
reconstruction remain conceptually the same.

## 6.4 H1 Polynomial Order

The solver defaults to the historical cubic basis:

```python
fes = ngs.H1(mesh, order=order, dirichlet=dirichlet_boundary, autoupdate=True)
```

Implications:

- the solution is globally continuous, which matches the electric-potential
  formulation
- cubic basis functions improve spatial resolution compared with low-order
  spaces, especially near current electrodes and strong local gradients
- the price is more DOFs per element and therefore a larger solve cost

The public `fe_order` parameter defaults to `3` to preserve previous behavior.
Lowering it, commonly to `2`, is a speed/accuracy tradeoff that should be
benchmarked against representative `fe_order=3` results.

## 6.5 Solver Output and Evaluation

Both solver variants return:

- `fes`: the finite-element space
- `gfu`: the grid function holding the solved potential

The worker evaluates the solution at arbitrary points by calling the grid
function on a mapped mesh point:

- 2D: `gfu(mesh(0.0, z))`
- 3D: `gfu(mesh(0.0, 0.0, z))`

These point evaluations are the last step before the worker multiplies by the
geometric factor and records the final apparent resistivity.

## 6.6 CG Convergence Behavior

The current code fixes the maximum iteration count at `1000`.

### What happens at the iteration limit

The implementation does not inspect a convergence flag or residual history after
calling `ngs.CGSolver`. That means:

- if CG converges early, the returned iterate is the expected solution
- if CG reaches the `1000`-step cap without converging, the returned iterate is
  still written into `gfu`
- no explicit warning is emitted by the current ReMo3D code path

### How to detect a stalled solve

Practical options are:

- compute a residual norm after the solve, as shown in
  [`developer-guide.md`](developer-guide.md#105-modifying-the-solver)
- temporarily enable solver-rate printing while debugging
- compare the same case under different preconditioners, mesh densities, or
  domain sizes and look for unstable apparent-resistivity outputs

### What non-convergence implies

A non-converged solve does not necessarily fail catastrophically, but it means
reported potentials and apparent resistivities may be contaminated by linear-
solver error instead of only discretization error.

## 6.7 Troubleshooting Solver Failures

Common causes of worker-side solver failures are:

- degenerate or badly shaped mesh elements
- extreme conductivity or resistivity contrasts
- point sources landing on awkward element boundaries or in very small elements
- an outer domain that is too small relative to the tool spacing
- a large batch size that forces one local mesh to represent too much geometry
  variation

### Diagnostic steps

1. rerun the same case with `batch_size=1`
2. increase `domain_radius`
3. compare `preconditioner="local"` versus `"multigrid"`
4. switch between `netgen` and `gmsh` on 2D cases if the mesh backend is
   suspected
5. inspect whether the worker output becomes `NaN`, which indicates that the
   broad worker `try/except` caught the failure
6. set `REMO3D_WORKER_DEBUG=1` to make worker task exceptions re-raise instead
   of being converted to `NaN`
7. if modifying the solver, add residual checks and print the local solve depth,
   current-electrode geometry, and conductivity distribution for the failing task

### Recommended parameter adjustments

- increase `domain_radius` when electrodes are near the boundary
- reduce `batch_size` for thin beds or rapidly varying borehole geometry
- keep the default mesh refinement near current electrodes unless you have a
  clear reason to coarsen it
- use `multigrid` first for larger elliptic problems

## 6.8 Mesh Quality and Solver Performance

Mesh quality and solver behavior are tightly coupled.

### Why local refinement helps

The mesh generators deliberately refine near:

- the borehole axis
- current electrodes
- material interfaces created by invaded zones and layer boundaries

That refinement improves the representation of the strong gradients introduced by
point sources and conductivity jumps.

### Why poor or mismatched refinement hurts

If the mesh is too coarse near the source, the solver can still converge in the
linear-algebra sense while converging to a poor discrete solution. If the mesh is
highly irregular or contains sliver-like features, CG can also converge more
slowly because the assembled system is harder to precondition effectively.

The mesh-side tuning logic is documented in:

- [`mesh-generation.md`](mesh-generation.md#47-mesh-size-control-strategy)
- [`configuration.md`](configuration.md#99-internal-geometry-window-and-meshing-knobs)

The key practical interaction is:

- stronger local refinement near electrodes usually improves physical accuracy
- but more DOFs and stronger coefficient jumps can increase linear-solver cost
- the best setting is therefore problem-dependent rather than universal

## See Also

- [`mathematical-foundations.md`](mathematical-foundations.md#28-static-condensation): mathematical meaning of condensation and the weak forms.
- [`mesh-generation.md`](mesh-generation.md#47-mesh-size-control-strategy): mesh fields and refinement strategy that shape the linear system.
- [`parallel-execution.md`](parallel-execution.md#75-error-handling): how worker-side failures are converted into `NaN` results.
- [`configuration.md`](configuration.md#95-preconditioner): runtime tuning knobs that influence solver behavior.
