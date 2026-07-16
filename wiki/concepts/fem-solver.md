---
title: FEM Solver
type: concept
tags: [fem, ngsolve, cg, preconditioner, static-condensation, gpu, method]
sources: [repo-docs]
updated: 2026-07-16
---

# FEM Solver

*(traces to `[[repo-docs]]` — solver.md, mathematical-foundations.md,
configuration.md, developer-guide.md.)*

Solves the [forward problem](forward-modeling.md) on the per-task
[mesh](mesh-generation.md) using NGSolve. See
[`../../docs/mathematical-foundations.md`](../../docs/mathematical-foundations.md)
for the weak forms this assembles.

## Discretization

- **H1** continuous finite-element space, default **cubic** order (`fe_order=3`);
  Dirichlet on the outer boundary.
- Bilinear form assembled `symmetric=True`, optional `condense`.
- **Point (Dirac) source** is applied by modifying the assembled load vector
  directly: locate the containing element, evaluate local shape functions at the
  source point, add `fac·φⱼ(xₖ)` to each element DOF. Physically each is a
  current electrode.

## Assemble-once / solve-per-RHS

`AssembleSystem` runs **once per (mesh, conductivity)** — builds the FE space,
form, stiffness term (axisymmetric `2πxσ∇u·∇v` or 3D `σ∇u·∇v`), and the solver
strategy/preconditioner. `SolveRHS` runs **once per source vector**. This split
is what makes [SEC](parallel-execution.md#single-electrode-computation-sec) cheap:
one assembled system serves many measurement solves.

## Direct vs iterative (chosen by size)

| System | Path |
|---|---|
| Small 2D (`ndof < 10000`) | cached **sparse Cholesky** direct inverse (reused across RHS) |
| Larger / 3D | **CG** (`CGSolver`, `maxsteps=1000`) with a preconditioner |

> ✅ **Benchmark finding: the `ndof < 10000` threshold is far too conservative for
> 2D.** Production 2D meshes are ~40k DOF, so `"auto"` always falls back to CG.
> Forcing the direct solver on those 20–45k-DOF systems is **3.48× faster than the
> original *and* essentially exact** (vs CG's iterative error) — the single best
> optimization found. See [optimization benchmark](../findings/optimization-benchmark.md).
> Invoke with `simulate_logs(..., direct_solver=True)` (needs the symmetric SPD
> form, which is the default); or raise `DIRECT_SOLVER_DOF_THRESHOLD`.
> The `direct_solver` kwarg accepts `"auto"` / `True` (force) / `False`.

- **Preconditioner**: `"multigrid"` (default — tens-to-low-hundreds of iters,
  scales with refinement) vs `"local"` (simpler; iters can drift into the
  hundreds on refined/3D meshes). Multigrid is the right default for these
  elliptic problems.
- **Static condensation** (`condense=True`): eliminates element-internal DOFs
  (Schur complement) → ~50–70% smaller global system for order-3 elements;
  reconstructs internals via `harmonic_extension` + `inner_solve`.
  Mathematically equivalent — a *performance* choice, not an accuracy one.

> ⚠️ **CG convergence caveat.** `maxsteps=1000` is hardcoded and **no residual /
> convergence flag is checked** in the production path — a non-converged iterate
> is written to the solution silently, so results can carry linear-solver error
> on top of discretization error. Detect by comparing preconditioners / mesh
> densities / domain radii, or via the offline [Task 0 harness](validation.md).
> The developer guide's `‖f − A·u‖ > 1e-8` residual check is opt-in only.

## GPU path

`ngsolve_functions_gpu.py` reuses the *identical* CPU `AssembleSystem`
(same space/form/preconditioner/condensation); it only moves the matrix +
preconditioner (`CreateDeviceMatrix`) and RHS (`CreateDeviceVector`) to device
and runs CG there. Reuses the CPU factorization when a direct inverse exists. GPU
helps only when the local solve is large enough to amortize transfer/setup — see
[performance & accuracy](performance-and-accuracy.md).

## Output

Grid function evaluated on the borehole axis — 2D `gfu(mesh(0,z))`, 3D
`gfu(mesh(0,0,z))` — then multiplied by the tool
[geometric factor](resistivity-logging.md#geometric-factors) to give apparent
resistivity.

## Links

- [forward modeling](forward-modeling.md) · [mesh generation](mesh-generation.md) · [resistivity logging](resistivity-logging.md).
- What was optimized here (assemble/solve split, direct solver, symmetric form, condensation fix): [optimization changes](../findings/optimization-changes.md).
- Extending the PDE/BCs: [`../../docs/developer-guide.md`](../../docs/developer-guide.md).
