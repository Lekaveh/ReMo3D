# ReMo3D Performance Optimization Tasks (2D Focus)

This document catalogs performance optimization opportunities for the 2D axisymmetric simulation pipeline. Each task includes the current (before) code, the proposed (after) code, and a detailed explanation of what changes and why.

**Important:** Speedup estimates in this document are directional, not guaranteed. Actual gains depend on model complexity, mesh density, hardware, and NGSolve version. Every optimization that changes numerical behavior must be validated against reference results before adoption. See Task 0 (benchmark harness) for the validation framework.

---

## Task 0: Validate Static Condensation and Build Benchmark Harness

**Files:** `remo3d/ngsolve_functions.py`, lines 50-56 and `remo3d/ngsolve_functions_gpu.py`, lines 49-52
**Impact:** Critical (prerequisite for all other tasks) | **Effort:** Medium
**Expected speedup:** None directly; enables safe implementation of all subsequent tasks

### Problem

The current static condensation sequence in `SolveBVP` may not follow NGSolve's intended workflow. In the standard NGSolve static condensation pattern, `harmonic_extension_trans` is applied to the RHS *before* the condensed solve, and `harmonic_extension` + `inner_solve` reconstruct internal DOFs *after* the solve. The current code applies `harmonic_extension_trans` to `f.vec` *after* the solve:

```python
# Current code (ngsolve_functions.py, lines 50-56):
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
gfu.vec.data = inv * f.vec

if condense==True:
    f.vec.data += a.harmonic_extension_trans * f.vec
    gfu.vec.data += a.harmonic_extension * gfu.vec
    gfu.vec.data += a.inner_solve * f.vec
```

The standard NGSolve static condensation workflow is:

```python
# Standard NGSolve pattern:
f.vec.data += a.harmonic_extension_trans * f.vec     # modify RHS before solve
gfu.vec.data = inv * f.vec                            # solve condensed system
gfu.vec.data += a.harmonic_extension * gfu.vec        # reconstruct internal DOFs
gfu.vec.data += a.inner_solve * f.vec                 # reconstruct internal DOFs
```

Before building performance work on the solver path, this ordering must be validated.

### Required actions

1. **Investigate the condensation sequence.** Compare results with `condense=True` vs `condense=False` on a known benchmark model (e.g., homogeneous medium where the analytical solution is known). If results match to solver tolerance, the current ordering is functionally correct (possibly due to NGSolve internal handling). If they diverge, fix the ordering before proceeding.

2. **Define the validated condensation solve sequence.** Once the correct ordering is established (step 1), encode it as a single helper function that all subsequent tasks must call. This prevents later tasks from independently copying a condensation sequence that may be wrong:

    ```python
    def _condensed_solve(a, inv, f, fes, condense):
        """Single source of truth for the static-condensation solve sequence.
        
        The ordering here was validated by Task 0 against condense=False baselines.
        All solver paths (CPU iterative, CPU direct, GPU) must call this function
        rather than inlining the condensation steps.
        """
        gfu = ngs.GridFunction(fes)
        if condense:
            # Validated ordering -- adjust if Task 0 investigation determines
            # that harmonic_extension_trans must precede the solve:
            f.vec.data += a.harmonic_extension_trans * f.vec
        gfu.vec.data = inv * f.vec
        if condense:
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec
        return gfu
    ```

    **The ordering shown above follows the standard NGSolve documentation pattern** (modify RHS before solve, reconstruct after). If Task 0 step 1 finds that the *current* code's ordering (solve first, then modify RHS) also produces correct results, document why and update this helper accordingly. Either way, this helper becomes the single source of truth that Tasks 2 and 5 call.

3. **Build a benchmark harness.** Create a small test suite with:
   - **Reference models:** At least two 2D models with known analytical or published solutions (e.g., homogeneous medium, two-layer model).
   - **Metrics to capture per run:**
     - Wall-clock time breakdown: mesh generation, assembly, solve, evaluation.
     - DOF count (total and free after condensation).
     - CG iteration count and final residual norm.
     - Apparent resistivity values at all measurement points.
   - **Acceptance criteria:** Apparent resistivity values must match the baseline within a defined tolerance (e.g., 0.1% relative error) for any optimization to be accepted.

4. **Address worker exception swallowing.** The current worker code at `worker.py` line 135 catches all exceptions with a bare `except:` and silently replaces results with `NaN`. A benchmark harness that only calls the public `simulate_logs` path would miss solver failures or misattribute them as numerical output. Task 0 must either:
   - Add a `debug=True` mode that re-raises worker exceptions (preferred for development/benchmarking), or
   - Include solver-level unit tests that call `SolveBVP`/`AssembleSystem`/`SolveRHS` directly, bypassing the worker's exception handler, or
   - At minimum, have the benchmark harness check for `NaN` in results and flag them as solver failures rather than valid output.

5. **Establish baselines.** Run the benchmark harness on the current unmodified code to produce reference wall-time and accuracy numbers. All subsequent tasks are validated against these baselines.

### Why this must come first

Every subsequent task in this document changes either the numerical path (Tasks 1, 3, 4, 5) or the solver structure (Tasks 2, 5). Without a validated baseline and automated comparison, there is no way to confirm that an optimization preserves correctness or to measure its actual speedup. The condensation question is especially important because Tasks 2 and 5 propose restructuring the solver path. The `_condensed_solve` helper ensures that the validated sequence is defined once and reused everywhere, rather than each task independently repeating a possibly-wrong condensation ordering.

---

## Task 1: Enable Symmetric Bilinear Form

**File:** `remo3d/ngsolve_functions.py`, line 31 and `remo3d/ngsolve_functions_gpu.py`, line 23
**Impact:** High | **Effort:** Minimal (1 line per file)
**Expected speedup:** Likely 20-40% on assembly and solve phases (must be benchmarked)

### Prerequisite

Task 0 (benchmark harness with baseline results).

### Problem

The bilinear form is declared as non-symmetric, even though the underlying PDE operator is symmetric for both 2D and 3D cases.

### Before

```python
a = ngs.BilinearForm(fes, symmetric=False, condense=condense)
```

### After

```python
a = ngs.BilinearForm(fes, symmetric=True, condense=condense)
```

### Explanation

The 2D axisymmetric weak form is:

```
a(u,v) = integral 2*pi*r * sigma * grad(u) . grad(v) dr dz
```

This is a symmetric bilinear form: `a(u,v) = a(v,u)` because `grad(u).grad(v) = grad(v).grad(u)` and `sigma`, `r`, `2*pi` are scalar coefficients. The same symmetry holds for the 3D form `a(u,v) = integral sigma * grad(u).grad(v) dx`.

Setting `symmetric=True` tells NGSolve that the bilinear form is mathematically symmetric. Depending on the NGSolve version and internal implementation, this may enable:
1. **Upper-triangle-only assembly** of the sparse stiffness matrix, potentially reducing assembly work and memory.
2. **Optimized sparse storage formats** (e.g., symmetric CSR), potentially reducing cache pressure during matrix-vector products in the CG solver.

Whether these optimizations are actually applied depends on NGSolve internals and version. The mathematical argument for symmetry is solid; the performance effect should be measured.

**Clarification on SPD:** `symmetric=True` declares the intended symmetry of the bilinear form to NGSolve. It does not by itself guarantee the assembled matrix is SPD. Positive-definiteness depends on the physics: strictly positive conductivity everywhere, sufficient Dirichlet boundary conditions eliminating the nullspace, and correct condensation. In the ReMo3D context, these conditions are met by construction (conductivity is `1/resistivity > 0`, Dirichlet BC is applied on the full outer boundary). However, this should be verified by checking that the CG solver converges without negative eigenvalue warnings after the change.

**Note on current CG correctness:** CG on the current `symmetric=False` matrix is not strictly incorrect. The assembled matrix is numerically symmetric regardless of the flag; the flag primarily controls NGSolve's internal storage format and assembly strategy. The optimization value is in reduced assembly cost and memory, not in "fixing" CG convergence.

### Applies to

Both `remo3d/ngsolve_functions.py` (line 31) and `remo3d/ngsolve_functions_gpu.py` (line 23).

### Validation

Run benchmark harness. Apparent resistivity values must match baseline within 0.01% (changes should be at floating-point rounding level only). Verify CG iteration counts are equal or lower.

---

## Task 2: Reuse Stiffness Matrix and Preconditioner Within a Batch

**File:** `remo3d/ngsolve_functions.py`, `remo3d/ngsolve_functions_gpu.py`, and `remo3d/workers/worker.py`
**Impact:** High | **Effort:** Medium (~80 lines across 3 files)
**Expected speedup:** Likely 2-4x on the solver phase per batch (must be benchmarked)

### Prerequisites

Task 0 (validated condensation and benchmark baseline).

### Problem

Within a single batch, the mesh and conductivity distribution are identical for all modelling tasks. Yet `SolveBVP` is called independently for each task, and every call reconstructs the H1 space, reassembles the stiffness matrix, and rebuilds the preconditioner from scratch.

### Before

**worker.py (lines 99-110):**
```python
mesh = ngs.Mesh(mesh)
sigma = ngs.CoefficientFunction(sigma)

## Compute measured resistivity
for modelling_task in task[2]:
    tool = modelling_task[1]
    tool_geometry = tool[0,:]
    source_terms = tool[1,:]

    ## Solve BVP
    fes, gfu = ngsf.SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense)
```

**ngsolve_functions.py (SolveBVP, lines 23-57):**
```python
def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense):
    model_dimensionality = mesh.dim
    fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = ngs.BilinearForm(fes, symmetric=False, condense=condense)
    if model_dimensionality==2:
        a += 2*np.pi*ngs.grad(u)*ngs.grad(v)*ngs.x*sigma*ngs.dx
    elif model_dimensionality==3:
        a += ngs.grad(u)*ngs.grad(v)*sigma*ngs.dx
    f = ngs.LinearForm(fes)
    f.Assemble()
    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    c = ngs.Preconditioner(a, preconditioner)
    a.Assemble()
    gfu = ngs.GridFunction(fes)
    inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
    gfu.vec.data = inv * f.vec
    if condense==True:
        f.vec.data += a.harmonic_extension_trans * f.vec
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec
    return fes, gfu
```

### After

**ngsolve_functions.py -- split into two functions, plus backward-compat wrapper:**
```python
def AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense):
    """Assemble the stiffness matrix and preconditioner once per mesh/sigma pair."""
    model_dimensionality = mesh.dim
    fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = ngs.BilinearForm(fes, symmetric=True, condense=condense)
    if model_dimensionality == 2:
        a += 2*np.pi * ngs.grad(u)*ngs.grad(v) * ngs.x * sigma * ngs.dx
    elif model_dimensionality == 3:
        a += ngs.grad(u)*ngs.grad(v) * sigma * ngs.dx
    c = ngs.Preconditioner(a, preconditioner)
    a.Assemble()
    return fes, a, c

def SolveRHS(fes, a, c, tool_geometry, source_terms, condense):
    """Solve for a given right-hand side, reusing the pre-assembled system."""
    model_dimensionality = fes.mesh.dim
    f = ngs.LinearForm(fes)
    f.Assemble()
    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000, tol=1e-8, printrates=False)
    # Delegate condensation solve to the validated helper from Task 0.
    # Do NOT inline condensation steps here.
    gfu = _condensed_solve(a, inv, f, fes, condense)
    return fes, gfu

def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense):
    """Original API preserved for backward compatibility."""
    fes, a, c = AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense)
    return SolveRHS(fes, a, c, tool_geometry, source_terms, condense)
```

**ngsolve_functions_gpu.py -- matching split required:**

The worker at `worker.py` line 32-35 imports either `ngsolve_functions` or `ngsolve_functions_gpu` as `ngsf`. Both modules must expose the same `AssembleSystem`/`SolveRHS` interface, or the worker must branch explicitly. The GPU version of `AssembleSystem` would be identical to the CPU version (assembly is always on CPU). The GPU version of `SolveRHS` would transfer matrices/vectors to the device before solving:

```python
# ngsolve_functions_gpu.py
from ngsolve_functions import AssembleSystem, AddPointSource  # reuse CPU assembly

def SolveRHS(fes, a, c, tool_geometry, source_terms, condense):
    """GPU-accelerated RHS solve, reusing pre-assembled system."""
    model_dimensionality = fes.mesh.dim
    f = ngs.LinearForm(fes)
    f.Assemble()
    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)
    adev = a.mat.CreateDeviceMatrix()
    cdev = c.mat.CreateDeviceMatrix()
    fdev = f.vec.CreateDeviceVector(copy=True)
    inv = ngs.CGSolver(adev, cdev, maxsteps=1000, tol=1e-8, printrates=False)
    # Delegate condensation solve to the validated helper from Task 0.
    # Note: inv operates on device vectors, but _condensed_solve uses CPU
    # vectors for the harmonic extension steps. This CPU/GPU interplay is
    # inherited from the current code and should be profiled separately.
    gfu = _condensed_solve(a, inv, f, fes, condense)
    return fes, gfu

def SolveBVP(mesh, sigma, tool_geometry, source_terms, dirichlet_boundary, preconditioner, condense):
    """Original API preserved for backward compatibility."""
    fes, a, c = AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense)
    return SolveRHS(fes, a, c, tool_geometry, source_terms, condense)
```

**Note on GPU `CreateDeviceMatrix` placement:** In this design, `CreateDeviceMatrix()` is called per RHS solve, which means the matrix is transferred to the GPU for each solve even though it hasn't changed. A further optimization would be to transfer once in `AssembleSystem` for GPU workers, but this requires the assembled-system object to be GPU-aware, adding complexity. The per-RHS transfer is the conservative starting point.

**worker.py -- assemble once, solve many:**
```python
mesh = ngs.Mesh(mesh)
sigma = ngs.CoefficientFunction(sigma)

## Assemble system once for this batch
fes, a, c = ngsf.AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense)

## Compute measured resistivity (reuse assembled system)
for modelling_task in task[2]:
    tool = modelling_task[1]
    tool_geometry = tool[0,:]
    source_terms = tool[1,:]

    ## Solve with new RHS only
    fes, gfu = ngsf.SolveRHS(fes, a, c, tool_geometry, source_terms, condense)
```

### Explanation

In the current code, each call to `SolveBVP` within a batch performs these expensive operations:
1. **H1 space construction** -- allocates DOF tables, element connectivity, boundary markers.
2. **Bilinear form assembly** -- loops over all mesh elements, computes element stiffness matrices, assembles into global sparse matrix.
3. **Preconditioner setup** -- for multigrid, this builds coarse-grid hierarchies and smoothers.

Within a batch, the mesh and conductivity (`sigma`) are identical. Only the source term positions and magnitudes change (different current electrode locations). The stiffness matrix `A` depends solely on the mesh geometry and conductivity, not on the source terms.

By splitting the function, the stiffness matrix and preconditioner are assembled once per batch, and only the cheap right-hand side assembly and CG solve are repeated. For a default batch of 5 depths in SEC mode, this eliminates up to 4 redundant assemblies.

### Effort estimate

This change touches 3 files (`ngsolve_functions.py`, `ngsolve_functions_gpu.py`, `worker.py`) and requires careful testing of both CPU and GPU code paths. Estimated ~80 lines of changes across all files.

### Validation

Run benchmark harness on both CPU-only and GPU worker configurations. Apparent resistivity values must match baseline within tolerance. Wall-time breakdown should show reduced assembly time proportional to batch size.

---

## Task 3: Set Explicit CG Solver Tolerance

**File:** `remo3d/ngsolve_functions.py`, line 50 and `remo3d/ngsolve_functions_gpu.py`, line 45
**Impact:** Medium | **Effort:** Minimal (1 line each)
**Expected speedup:** Potentially 10-30% on solve phase (must be benchmarked)

### Prerequisite

Task 0 (benchmark harness). Changing solver tolerance is a numerical change that must be validated.

### Problem

The CG solver has no explicit tolerance, relying on the NGSolve default and a hard cap of 1000 iterations.

### Before

```python
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
```

### After

```python
inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000, tol=1e-8, printrates=False)
```

### Explanation

- `tol=1e-8` sets a relative residual tolerance. For resistivity logging, measurement accuracy of 4-5 significant digits is more than sufficient, so `1e-8` provides ample margin while allowing early termination once convergence is reached.
- `printrates=False` suppresses per-iteration output in worker processes, which otherwise generates I/O noise when running many parallel solves.

### API compatibility note

The `tol` keyword parameter for `ngs.CGSolver` has been available since NGSolve 6.2 (2019). However, `ngs.CGSolver` is often a C++/pybind11 binding where `inspect.signature` may fail or return incomplete metadata. The safest way to verify compatibility is to run a minimal solver construction in the project's actual NGSolve environment:

```python
import ngsolve as ngs
from ngsolve import Mesh, H1, BilinearForm, grad, CGSolver
# Create a trivial 1-element mesh and test CGSolver accepts tol=
mesh = Mesh(ngsolve.unit_square.GenerateMesh(maxh=1))
fes = H1(mesh, order=1)
u, v = fes.TnT()
a = BilinearForm(fes)
a += grad(u)*grad(v)*ngs.dx
a.Assemble()
try:
    inv = CGSolver(a.mat, a.mat, maxsteps=1, tol=1e-8, printrates=False)
    print("tol= keyword accepted")
except TypeError as e:
    print(f"tol= keyword not supported: {e}")
```

If the installed version uses a different keyword name (e.g., `precision`), adapt accordingly.

### Validation

Run benchmark harness. Compare:
1. CG iteration counts before and after. **Note:** iteration count may increase or decrease depending on whether the current NGSolve default tolerance is tighter or looser than `1e-8`. The relevant acceptance criteria are residual norm and output accuracy, not iteration count.
2. Apparent resistivity values must match baseline within 0.1% relative error.
3. If any measurement point diverges by more than 0.1%, tighten tolerance to `1e-10` and re-test.
4. Record and compare wall-clock time of the solve phase.

---

## Task 4: Expose FE Polynomial Order as a Parameter

**File:** `remo3d/ngsolve_functions.py`, `remo3d/ngsolve_functions_gpu.py`, `remo3d/remo3d.py`, `remo3d/workers/worker.py`
**Impact:** High (user-controlled) | **Effort:** Medium (~40 lines across 5 files)
**Expected speedup:** Fewer DOFs when using order=2 (must be benchmarked for accuracy tradeoff)

### Prerequisite

Task 0 (benchmark harness for accuracy comparison).

### Problem

The finite element polynomial order is hardcoded to 3 (cubic). For some 2D problems, lower order may provide sufficient accuracy at fewer DOFs.

### Before

```python
fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
```

### After

```python
fes = ngs.H1(mesh, order=order, dirichlet=dirichlet_boundary, autoupdate=True)
```

Where `order` is a new parameter propagated through the full call chain:

1. `remo3d.py`: `compute_synthetic_logs(..., fe_order=3, ...)` and `simulate_logs(..., fe_order=3, ...)`
2. `remo3d.py`: MPI broadcast of `fe_order` to workers (new broadcast line after `condense`)
3. `worker.py`: receive `fe_order` from broadcast, pass to solver functions
4. `ngsolve_functions.py`: `AssembleSystem(..., order=3, ...)` (or `SolveBVP` if Task 2 is not yet implemented)
5. `ngsolve_functions_gpu.py`: same change as CPU module

### Effort estimate

This is more than "~15 lines." It touches 5 files, adds a new MPI broadcast variable, updates two public API signatures, and requires documentation updates. Estimated ~40 lines of actual changes.

### Explanation

In a 2D triangular mesh, local DOFs per element scale as `(order+1)*(order+2)/2`:
- Order 2: 6 DOFs per element
- Order 3: 10 DOFs per element

However, the global DOF count after static condensation scales differently. With condensation, only DOFs on element boundaries (edges and vertices) remain in the global system; interior DOFs are eliminated. The condensed system size depends on the mesh connectivity, not just per-element DOF count. The actual speedup from order reduction should be measured, not assumed from local element math.

For point-source problems, the solution has a 1/r singularity. This limits the global H1 convergence rate regardless of polynomial order unless adaptive refinement is used. The current h-refinement via `mesh_size_min` partially addresses this, but the theoretical superconvergence benefits of high order are not fully realized.

**This is a user-controlled tradeoff, not a free optimization.** Order 2 will produce different (slightly less accurate) results. The parameter should default to `order=3` (preserving current behavior) and users should be advised to compare order=2 and order=3 results on their specific models before adopting lower order for production runs.

### Validation

Run benchmark harness with order=2 and order=3. Report:
1. DOF count (total and condensed) for each.
2. Wall-time breakdown for each.
3. Apparent resistivity relative error between order=2 and order=3 at every measurement point.
4. Document the accuracy-vs-speed tradeoff to guide users.

---

## Task 5: Use Direct Solver for Small 2D Systems

**File:** `remo3d/ngsolve_functions.py` (inside `AssembleSystem` if Task 2 is implemented, otherwise in `SolveBVP`)
**Impact:** Medium-High | **Effort:** Low-Medium (~15 lines)
**Expected speedup:** Potentially 2-3x for typical 2D systems (must be benchmarked)

### Prerequisites

Task 0 (benchmark baseline), Task 1 (symmetric form -- required for `sparsecholesky`), Task 2 (matrix reuse -- required for factorization reuse benefit).

### Problem

The iterative CG solver with multigrid preconditioner is designed for large systems (10k+ DOFs). For typical 2D axisymmetric meshes (1k-5k DOFs after static condensation), a direct sparse Cholesky factorization is faster and provides exact solutions.

### Before

```python
c = ngs.Preconditioner(a, preconditioner)
a.Assemble()
gfu = ngs.GridFunction(fes)

inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000)
gfu.vec.data = inv * f.vec
```

### After -- factorization in AssembleSystem (critical design point)

**Note:** This task extends the Task 2 API. `AssembleSystem` gains a fourth return value (`inv`), and `SolveRHS` gains an `inv` parameter. The Task 2 snippets show the pre-Task-5 signatures; the snippets below show the final state after both tasks are applied.

The key benefit of the direct solver is **factorization reuse**: factor once, back-substitute many times. This means the inverse must be computed during assembly, not during each RHS solve. The factorization object is returned as part of the assembled system.

**Critical structural requirement:** In NGSolve, `ngs.Preconditioner(a, ...)` must be registered *before* `a.Assemble()` for multigrid to work correctly. The code below handles this by deciding the solver strategy first, registering the preconditioner if needed, then assembling, then computing the direct inverse if applicable:

```python
def AssembleSystem(mesh, sigma, dirichlet_boundary, preconditioner, condense):
    """Assemble stiffness matrix; compute direct factorization or iterative preconditioner."""
    model_dimensionality = mesh.dim
    fes = ngs.H1(mesh, order=3, dirichlet=dirichlet_boundary, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()
    a = ngs.BilinearForm(fes, symmetric=True, condense=condense)
    if model_dimensionality == 2:
        a += 2*np.pi * ngs.grad(u)*ngs.grad(v) * ngs.x * sigma * ngs.dx
    elif model_dimensionality == 3:
        a += ngs.grad(u)*ngs.grad(v) * sigma * ngs.dx

    # Decide solver strategy before assembly.
    # For small 2D systems, use direct factorization; otherwise iterative.
    use_direct = (fes.ndof < 10000 and model_dimensionality == 2)

    # Register preconditioner BEFORE assembly (required by NGSolve for multigrid).
    # For the direct path, we skip preconditioner registration entirely.
    c = None
    if not use_direct:
        c = ngs.Preconditioner(a, preconditioner)

    # Assemble (preconditioner, if registered, is updated during this call).
    a.Assemble()

    # Compute reusable inverse for the direct path.
    inv = None
    if use_direct:
        free_dofs = fes.FreeDofs(condense) if condense else fes.FreeDofs()
        inv = a.mat.Inverse(free_dofs, inverse="sparsecholesky")

    return fes, a, c, inv

def SolveRHS(fes, a, c, inv, tool_geometry, source_terms, condense):
    """Solve for a given RHS. Uses pre-computed direct inverse or iterative solver."""
    model_dimensionality = fes.mesh.dim
    f = ngs.LinearForm(fes)
    f.Assemble()
    for l in range(np.shape(source_terms)[0]):
        if source_terms[l] != 0.0:
            AddPointSource(f, tool_geometry[l], source_terms[l], model_dimensionality)

    if inv is not None:
        # Direct solve: back-substitution only (fast, reuses factorization)
        solve_inv = inv
    else:
        # Iterative solve: create CG solver using pre-built preconditioner
        solve_inv = ngs.CGSolver(a.mat, c.mat, maxsteps=1000, tol=1e-8, printrates=False)

    # Delegate condensation solve to the validated helper from Task 0.
    # Do NOT inline condensation steps here.
    gfu = _condensed_solve(a, solve_inv, f, fes, condense)
    return fes, gfu
```

### Explanation

For small 2D systems, the direct factorization `L*L^T = A` is computed once in `AssembleSystem`. Each subsequent call to `SolveRHS` performs only forward/backward substitution (O(nnz)), which is extremely fast compared to iterative CG.

The DOF threshold of 10,000 is an initial estimate. The actual crossover point depends on hardware and sparse matrix structure. The benchmark harness (Task 0) should be used to determine the optimal threshold empirically.

**Important:** `sparsecholesky` requires the matrix to be SPD. This requires Task 1 (`symmetric=True`) and valid physics (positive conductivity, sufficient Dirichlet BCs). The 3D case is excluded from the direct solver path (`model_dimensionality == 2` check) because 3D systems are typically too large.

### Validation

Run benchmark harness comparing direct vs. iterative solver for 2D models. Check:
1. Apparent resistivity values match baseline within 0.01% (direct solver should be more accurate).
2. Wall-time reduction per batch.
3. Memory usage (direct solver stores the factorization; verify it fits in the per-worker memory budget).

---

## Task 6: Vectorize Domain Boundary Point Insertion in Netgen Mesh Builder

**File:** `remo3d/netgen_functions.py`, lines 206-221
**Impact:** Low-Medium | **Effort:** Low (~15 lines)
**Expected speedup:** 10-20% on mesh generation phase

### Problem

The loop that inserts additional points along the circular domain boundary uses repeated `np.vstack` calls, which allocate and copy a new array on every iteration.

### Before

```python
starting_index = index_0D
points_at_domain_boundary = existing_points_at_domain_boundary[0,:]
for i in range(np.shape(points_to_add)[0]):
    if points_to_add[i] > 0:
        index = np.array([0, points_to_add[i]+1])
        angle = np.array([existing_points_at_domain_boundary[i,2], existing_points_at_domain_boundary[i+1,2]])
        interpolated_angles = np.interp(np.arange(points_to_add[i])+1, index, angle)
        additional_points = np.vstack([np.arange(points_to_add[i]) + index_0D, np.full(points_to_add[i], domain_radius), interpolated_angles]).T
        points_at_domain_boundary = np.vstack([points_at_domain_boundary, additional_points, existing_points_at_domain_boundary[i+1,:]])
        index_0D += points_to_add[i]
    else:
        points_at_domain_boundary = np.vstack([points_at_domain_boundary, existing_points_at_domain_boundary[i+1,:]])
```

### After

```python
starting_index = index_0D
segments = [existing_points_at_domain_boundary[0:1, :]]
for i in range(np.shape(points_to_add)[0]):
    if points_to_add[i] > 0:
        index = np.array([0, points_to_add[i]+1])
        angle = np.array([existing_points_at_domain_boundary[i,2], existing_points_at_domain_boundary[i+1,2]])
        interpolated_angles = np.interp(np.arange(points_to_add[i])+1, index, angle)
        additional_points = np.vstack([np.arange(points_to_add[i]) + index_0D, np.full(points_to_add[i], domain_radius), interpolated_angles]).T
        segments.append(additional_points)
        index_0D += points_to_add[i]
    segments.append(existing_points_at_domain_boundary[i+1:i+2, :])
points_at_domain_boundary = np.vstack(segments)
```

### Explanation

The original code calls `np.vstack` inside the loop, resulting in O(n^2) total memory copies (each iteration copies all accumulated data plus new data). The collect-then-vstack pattern reduces this to O(n) total copies with a single allocation at the end.

---

## Task 7: Index-Based Point Lookup in Boundary Line Construction

**File:** `remo3d/netgen_functions.py`, lines 247-278
**Impact:** Low | **Effort:** Low (~20 lines)
**Expected speedup:** Minor for typical models, noticeable for models with 50+ layers

### Problem

The boundary line construction loop scans the entire points array for every layer boundary using boolean masking on floating-point equality. This applies to both the line-counting loop (line 248) and the line-construction loop (line 267).

### Before

```python
# Line counting (line 248):
number_of_lines_at_boundary = np.empty_like(boundaries_z)
for i in range(np.shape(number_of_lines_at_boundary)[0]):
    number_of_lines_at_boundary[i] = np.sum(np.all([points[:,2]==boundaries_z[i], points[:,1] > 0], axis=0)) - 1

# Line construction (line 267):
for i in range(np.shape(boundaries_z[1:-1])[0]):
    points_at_ith_boundary = points[np.all([points[:,2]==boundaries_z[1:-1][i], points[:,1] > 0], axis=0), :]
    points_at_ith_boundary = points_at_ith_boundary[np.argsort(points_at_ith_boundary[:,1]),:]
    ...
```

### After

The correct approach is to assign stable integer boundary IDs during point construction, rather than using floating-point values or object identity as hash keys.

**Step 1:** During point construction (earlier in `ConstructNetgen2dModel`), tag each point with its boundary index:

```python
# When creating points, also build a mapping: boundary_index -> [point_row_indices]
# This is populated as points are added to the various boundary depth levels.
# boundaries_z is computed at line 130 as np.sort(np.unique(formation_geometry[:,:2]))
# Each point at a boundary depth is tagged with that boundary's integer index.

boundary_to_point_rows = {i: [] for i in range(len(boundaries_z))}

# ... during point creation, when a point is placed at boundaries_z[k]:
# boundary_to_point_rows[k].append(point_row_index)
```

**Step 2:** Use the pre-built index for both counting and construction:

```python
# Line counting (replaces the loop at line 248):
number_of_lines_at_boundary = np.empty_like(boundaries_z)
for i in range(len(boundaries_z)):
    # Filter to points with r > 0
    rows = [r for r in boundary_to_point_rows[i] if points[r, 1] > 0]
    number_of_lines_at_boundary[i] = max(len(rows) - 1, 0)

# Line construction (replaces the loop at line 267):
for i in range(len(boundaries_z) - 2):  # boundaries_z[1:-1]
    bz_index = i + 1  # offset into boundaries_z
    rows = [r for r in boundary_to_point_rows[bz_index] if points[r, 1] > 0]
    points_at_ith_boundary = points[rows, :]
    points_at_ith_boundary = points_at_ith_boundary[np.argsort(points_at_ith_boundary[:, 1]), :]
    ...
```

### Explanation

The original performs a full-array scan (`points[:,2] == boundaries_z[i]`) for every boundary depth: O(n_points * n_boundaries). The index-based approach makes each lookup O(1).

**Why `id(bz)` does not work:** In the previous version of this document, the index used `id(bz)` as the hash key, where `bz` was a NumPy scalar produced during iteration over `boundaries_z`. NumPy scalar `id()` reflects the *Python object identity* of a temporary, not the identity of the array slot or the numeric value. When later looking up `id(boundaries_z[i])`, a *different* temporary object is created, producing a different `id()`. The lookups would miss silently, making the optimization incorrect.

**Why float-value keys are fragile:** Using the float value itself as a dict key (e.g., `float(boundaries_z[i])`) works when the same computation produced both the point coordinates and `boundaries_z`. But any intermediate arithmetic (e.g., interpolation, clipping to domain boundary) could introduce ULP-level differences, causing silent lookup misses. Rounding to N decimal places risks merging genuinely distinct boundaries in models with very thin beds.

**Why integer boundary IDs are the right design:** The boundary depths in `boundaries_z` come from `np.sort(np.unique(formation_geometry[:,:2]))`. Each boundary has a natural integer index (its position in this sorted array). By tagging points with this index during construction, lookups become integer-keyed and immune to floating-point issues. This requires threading the index through the point-construction code, which is more invasive than a drop-in replacement but structurally correct.

---

## Task 8: Adaptive Domain Radius Based on Tool Geometry

**File:** `remo3d/remo3d.py`, inside `simulate_logs`
**Impact:** Medium | **Effort:** Medium (~30 lines, plus validation)
**Expected speedup:** Potentially significant for short-spaced tools (must be benchmarked per model)

### Prerequisite

Task 0 (benchmark harness -- accuracy validation is critical for this task).

### Problem

The domain radius defaults to 50 meters regardless of tool geometry. For short-spaced logging tools (electrode spacings < 2m), the simulation domain may be larger than necessary.

### Before

```python
# User always passes a numeric domain_radius (default 50)
domain_radius = 50
```

### After -- auto-sizing must happen before electrode validation

The current code at `remo3d.py` lines 766-769 compares electrode positions to `domain_radius` numerically:

```python
domain_radius_alert = False
for tool in self.tools.keys():
    if np.max(np.abs(self.tools[tool][0,:3])) > domain_radius:
        raise ValueError("Some electrodes are locate outside the simulation domain...")
```

If `domain_radius` were a string `"auto"`, this comparison would crash. Auto-sizing must be resolved to a numeric value **before** this validation block:

```python
## Resolve auto domain radius (must precede electrode validation)
if domain_radius == "auto":
    max_electrode_distance = max(
        np.max(np.abs(self.tools[tool][0,:3])) for tool in self.tools.keys()
    )
    domain_radius = max(10 * max_electrode_distance, 5.0)
    print(f"Auto domain radius: {domain_radius:.1f} m (based on max electrode distance {max_electrode_distance:.2f} m)")

## Electrode validation (existing code, now safe for numeric domain_radius)
domain_radius_alert = False
for tool in self.tools.keys():
    if np.max(np.abs(self.tools[tool][0,:3])) > domain_radius:
        raise ValueError("Some electrodes are locate outside the simulation domain...")
```

### Accuracy and scope concerns

`domain_radius` controls far more than just the outer boundary location. It also determines:

1. **Data clipping window:** `SelectNetgenDataRange` and `SelectGmshDataRange` use `domain_radius` to decide which formation layers and borehole segments are included in the local model. A smaller radius may exclude layers that influence the measurement.
2. **Batch extent:** Batched simulation depths are grouped assuming they fit within the domain. Reducing the domain may interact with batch offsets.
3. **Borehole geometry clipping:** Top/bottom borehole points are adjusted to the domain boundary circle.
4. **Invasion zone visibility:** Filtration zones extending far from the borehole may be clipped out with a small domain.

The simple formula `10 * max_electrode_distance` does not account for these dependencies. For example, a tool with 0.5m spacing in a model with invasion zones extending to 3m would get `domain_radius = 5.0m`, which may clip relevant invasion zones differently than the default 50m.

### Recommendation

This task should be implemented as an **opt-in feature** (not a new default) with clear documentation:

- Default remains `domain_radius=50` (no behavior change).
- `domain_radius="auto"` enables auto-sizing with the formula above.
- Documentation must warn that auto-sizing may affect accuracy for models with:
  - Large invasion zones relative to tool spacing.
  - Thin beds near the domain boundary.
  - Large batch sizes (batch offsets push electrodes toward domain edge).
- Users should validate auto-sized results against `domain_radius=50` results on their specific models before adopting.

### Validation

Run benchmark harness with `domain_radius=50` and `domain_radius="auto"` on multiple test models:
1. Homogeneous model (should be nearly identical).
2. Multi-layer model with invasion zones (check for boundary artifacts).
3. Report maximum relative error across all measurement points.
4. Determine if a safety factor larger than 10x is needed for general use.

---

## Task 9: CPU-Only Guidance for 2D Simulations

**File:** `remo3d/remo3d.py` (inside `initialize_workers` or before it)
**Impact:** Low-Medium | **Effort:** Low (~5 lines)
**Expected speedup:** Avoids negative speedup from GPU overhead

### Problem

GPU acceleration is counterproductive for 2D simulations. The GPU solver transfers matrices and vectors to the device for systems that are too small to benefit from GPU parallelism. Currently, the warning would be issued too late if placed in `simulate_logs`, since `initialize_workers` (line 168 in `compute_synthetic_logs`) spawns MPI processes before `simulate_logs` runs.

### Before

No warning exists. Users can set `gpu_workers > 0` for 2D models without feedback.

### After -- place warning/override before worker spawn

The most effective location is inside `initialize_workers` itself, or in `compute_synthetic_logs` between `set_model_parameters` and `initialize_workers`:

```python
# Option A: Inside initialize_workers, which has access to self.dip_deg
def initialize_workers(self, cpu_workers=4, gpu_workers=0):
    # Warn before spawning GPU workers for 2D models
    if self.dip_deg is not None and np.isclose(self.dip_deg, 0) and gpu_workers > 0:
        print("Warning: GPU acceleration is not beneficial for 2D (dip=0) models. "
              "Consider setting gpu_workers=0. Proceeding with GPU workers as requested.")

    ## Check GPU availability (existing code)
    ...
```

### Explanation

For typical 2D axisymmetric meshes (1k-5k DOFs), the CPU-GPU transfer overhead exceeds compute savings. The `CreateDeviceMatrix()` and `CreateDeviceVector()` calls in `ngsolve_functions_gpu.py` involve serialization and PCI-e transfer that dominate the total solve time for small systems.

The warning is advisory, not a hard block, since there may be edge cases where users have very dense 2D meshes that benefit from GPU. But for the common case, `gpu_workers=0` is the right choice for `dip=0`.

---

## Implementation Priority

The recommended order has been revised to prioritize validation before optimization:

| Priority | Task | Description | Effort | Prerequisite |
|----------|------|-------------|--------|--------------|
| **1** | **Task 0** | **Validate condensation + build benchmark harness** | Medium | None |
| 2 | Task 1 | Symmetric bilinear form | 1 line x 2 files | Task 0 |
| 3 | Task 3 | Explicit CG tolerance | 1 line x 2 files | Task 0 |
| 4 | Task 2 | Matrix reuse + GPU parity + backward compat | ~80 lines / 3 files | Task 0 |
| 5 | Task 5 | Direct solver for small 2D (factorization in assembly) | ~15 lines | Tasks 0, 1, 2 |
| 6 | Task 9 | CPU-only warning for 2D (before worker spawn) | ~5 lines | None |
| 7 | Task 6 | Vectorize boundary point insertion | ~15 lines | None |
| 8 | Task 7 | Index-based point lookup | ~20 lines | None |
| 9 | Task 4 | Expose FE order parameter | ~40 lines / 5 files | Task 0 |
| 10 | Task 8 | Adaptive domain radius (opt-in) | ~30 lines + validation | Task 0 |

### Rationale for ordering

1. **Task 0 first:** No optimization should be merged without a validated baseline and automated accuracy comparison. The condensation question must be answered before restructuring the solver path.
2. **Tasks 1 and 3 next:** Minimal-effort changes with no structural risk, but they are numerical changes requiring benchmark validation.
3. **Task 2 before Task 5:** The direct solver's main benefit (factorization reuse) requires the split assembly/solve architecture from Task 2.
4. **Tasks 6-7 can proceed independently:** Pure Python refactoring with no numerical impact; no benchmark needed.
5. **Tasks 4 and 8 last:** These change user-facing behavior and require the most careful validation. Task 4 introduces accuracy tradeoffs. Task 8 interacts with multiple subsystems.

### On speedup estimates

All speedup numbers in this document are directional estimates, not guarantees. Actual performance depends on:
- Model complexity (number of layers, invasion zones).
- Mesh density (controlled by `mesh_size_min`, `mesh_size_max`, `mesh_density`).
- Hardware (CPU cache size, memory bandwidth, GPU if applicable).
- NGSolve version (internal assembly and solver optimizations vary between versions).

The benchmark harness from Task 0 will provide concrete numbers for specific configurations.
