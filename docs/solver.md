# FEM Solver

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

## 6.2 `SolveBVP` CPU Version

The CPU solver path in `ngsolve_functions.py` is:

1. infer model dimensionality from `mesh.dim`
2. create an H1 finite-element space with `order=3`
3. define trial and test functions
4. build a bilinear form with optional static condensation
5. assemble the axisymmetric or 3D stiffness term
6. assemble an initially empty linear form
7. inject point sources with `AddPointSource`
8. build the selected preconditioner
9. assemble the bilinear form
10. allocate the grid function
11. solve the linear system with CG, up to 1000 steps
12. reconstruct condensed DOFs if needed
13. return `fes, gfu`

The actual solve call is:

```python
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
gfu.vec.data = inv * f.vec
```

## 6.3 `SolveBVP` GPU Version

The GPU path in `ngsolve_functions_gpu.py` is intentionally close to the CPU
path so the numerical formulation stays identical.

The differences are:

- the sparse matrix is transferred with `CreateDeviceMatrix()`
- the preconditioner matrix is transferred with `CreateDeviceMatrix()`
- the load vector is copied with `CreateDeviceVector(copy=True)`
- CG is run against the device matrices and vector

Core device setup:

```python
adev = a.mat.CreateDeviceMatrix()
cdev = c.mat.CreateDeviceMatrix()
fdev = f.vec.CreateDeviceVector(copy=True)
inv = ngs.CGSolver(adev, cdev, maxsteps=1000, printrates=False)
```

The FE space, bilinear form, point-source assembly, and static-condensation
reconstruction remain conceptually the same.

## 6.4 Why an H1 Space of Order 3

The solver hard-codes:

```python
fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
```

Implications:

- the solution is globally continuous, which matches the electric-potential
  formulation
- cubic basis functions improve spatial resolution compared with low-order
  spaces, especially near current electrodes and strong local gradients
- the price is more DOFs per element and therefore a larger solve cost

This is a sensible compromise for smooth elliptic fields with localized source
singularities that still need accurate potential differences at electrode
locations.

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
