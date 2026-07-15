---
title: Optimization Changes (optim branch)
type: finding
tags: [optimization, performance, solver, optim-branch, changelog]
sources: [repo-docs]
updated: 2026-07-15
---

# Optimization Changes — `optim` branch

*What was changed to speed up ReMo3D, and why.* Reconstructed from git history
(branch `optim`, merge-base `d099f0b` → `63f7694`, May 2026) and the plan in
[`../../performance_optimization_tasks.md`](../../performance_optimization_tasks.md).
All changes target the **2D axisymmetric** solve path and were gated on a
[validation harness](../concepts/validation.md) built first.

## Guiding principle

The plan's non-negotiable rule: **validate before optimizing.** Task 0 built a
benchmark harness and a single validated condensation routine *before* any
speed change, so every later task could be checked for correctness (apparent
resistivity within tolerance) rather than assumed. Speedup figures in the plan
are explicitly *directional, not guaranteed*.

## What landed vs. what didn't

| Task | Change | Why | Status |
|---|---|---|---|
| **0** | Benchmark harness + `_condensed_solve` helper + worker debug mode | Safe baseline for all other work | ✅ done |
| **1** | Bilinear form `symmetric=False → True` | PDE operator is symmetric; enables symmetric storage/assembly **and** unlocks Cholesky (Task 5) | ✅ done |
| **2** | Split `SolveBVP` → `AssembleSystem` + `SolveRHS`; assemble once per batch | Mesh & σ are identical within a batch — stop re-assembling the matrix/preconditioner per source | ✅ done |
| **3** | Explicit CG `tol=1e-8` | Early termination once converged | ❌ **not implemented** (no `tol=` in final code) |
| **4** | Expose `fe_order` through the API + MPI broadcast | Let users trade accuracy for fewer DOFs (order 2 vs 3) | ✅ done |
| **5** | Direct `sparsecholesky` for small 2D systems | For ~1k–5k-DOF 2D meshes, factor-once/back-substitute beats iterative CG | ✅ done |
| **6** | Vectorize Netgen boundary-point insertion | Remove O(n²) `np.vstack` in a loop | ❌ not implemented (`netgen_functions.py` untouched) |
| **7** | Index-based boundary point lookup | Replace O(n·boundaries) float-equality scans | ❌ not implemented |
| **8** | `domain_radius="auto"` sizing | Short-spaced tools don't need a 50 m domain | ✅ done (opt-in) |
| **9** | Warn against GPU for 2D | GPU transfer overhead exceeds gains on small systems | ❌ not implemented |

## The changes that landed, in detail

### Task 2 — assemble once, solve many (the big structural win)
`SolveBVP` was split into
[`AssembleSystem`](../concepts/fem-solver.md#assemble-once--solve-per-rhs) (builds
FE space, symmetric bilinear form, preconditioner/factorization — **once per
mesh/σ**) and `SolveRHS` (RHS + solve — **per source**). The worker now calls
`AssembleSystem` once per batch, then loops `SolveRHS`. Because
[SEC](../concepts/parallel-execution.md#single-electrode-computation-sec) makes a
batch share one mesh, this eliminates up to `batch_size − 1` redundant
assemblies/preconditioner builds per batch. `SolveBVP` is kept as a
backward-compatible wrapper.

### Task 5 — direct solver for small 2D
`AssembleSystem` chooses `use_direct = (dim==2 and ndof < 10000)`
(`DIRECT_SOLVER_DOF_THRESHOLD`). On that path it skips the preconditioner and
computes a **reusable** `a.mat.Inverse(..., inverse="sparsecholesky")`; each
`SolveRHS` is then just back-substitution. Larger/3D systems keep multigrid + CG.
Requires the symmetric form (Task 1) for SPD-ness.

### Task 1 — symmetric bilinear form
One-line change (`symmetric=True`) reflecting the mathematical symmetry of both
weak forms ($2\pi r\sigma\nabla u\cdot\nabla v$ and $\sigma\nabla u\cdot\nabla v$).
Enables symmetric assembly/storage and is a hard prerequisite for `sparsecholesky`.

### Task 0 — harness, condensation correctness, debuggability
- `scripts/benchmark_task0.py` bypasses MPI and calls the solver directly (so
  exceptions surface instead of becoming NaN); pass condition = `condense=True`
  matches `condense=False` within rel-tol `1e-3`. See [validation](../concepts/validation.md).
- **Condensation ordering was corrected.** The old inline code applied
  `harmonic_extension_trans` to the RHS *after* the solve; the new
  `_condensed_solve` (single source of truth for CPU + GPU) applies it *before*
  the condensed solve, matching the documented NGSolve sequence. Behaviour is
  validated by the harness.
- `REMO3D_WORKER_DEBUG=1` re-raises worker exceptions instead of the usual
  [NaN-swallowing](../concepts/parallel-execution.md#fault-tolerance).
- Optional `return_metrics` path records timings, DOF counts, CG iters/residual.

### Task 4 — user-controlled FE order
`fe_order` (default 3, preserving behaviour) is threaded through
`compute_synthetic_logs` → `simulate_logs` → a **new MPI broadcast** → worker →
`AssembleSystem`. Order 2 cuts DOFs/time but is an **accuracy tradeoff**, not
free — see [performance & accuracy](../concepts/performance-and-accuracy.md).

### Task 8 — adaptive domain radius (opt-in)
`domain_radius="auto"` resolves to `max(10 × max_electrode_distance, 5.0)`
**before** electrode-in-domain validation (a string would otherwise crash the
numeric comparison). Default stays `50 m`. Flagged as opt-in because
`domain_radius` also controls data clipping / invasion-zone visibility, so
auto-sizing can change results on models with large invasion or thin beds near
the boundary.

## Not implemented — and why it matters

- **Task 3 (CG `tol`)**: final code still `CGSolver(a.mat, c.mat, maxsteps=1000)`
  with no tolerance — so the [silent no-convergence-check caveat](../concepts/fem-solver.md)
  is unchanged. A candidate quick win still on the table.
- **Tasks 6 & 7 (Netgen meshing)**: `netgen_functions.py` was never modified;
  the O(n²) boundary-point `vstack` and float-equality scans remain. Low impact
  except on many-layer models.
- **Task 9 (2D GPU warning)**: no guard added; users can still set
  `gpu_workers>0` for 2D and silently lose performance.

## Open threads

- No recorded **benchmark numbers** from the harness yet — the "why" is
  documented but the realized speedup is not quantified here. Running Task 0
  baselines vs. optimized would give a real before/after and belongs on this page.
- The [sensitivity analysis](../../remo3d/sensitivity.py) commit (`63f7694`) is
  separate work, not an optimization — it deserves its own findings page.

## Links

- Mechanisms: [FEM solver](../concepts/fem-solver.md), [parallel execution](../concepts/parallel-execution.md), [performance & accuracy](../concepts/performance-and-accuracy.md).
- Validation: [benchmarks](../concepts/validation.md). Plan: [`../../performance_optimization_tasks.md`](../../performance_optimization_tasks.md).
