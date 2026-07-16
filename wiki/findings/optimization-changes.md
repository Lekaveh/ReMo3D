---
title: Optimization Changes (optim branch)
type: finding
tags: [optimization, performance, solver, optim-branch, changelog]
sources: [repo-docs]
updated: 2026-07-16
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
| **6** | Vectorize Netgen boundary-point insertion | Remove O(n²) `np.vstack` in a loop | ✅ done (`8e83c37`, verified mesh-identical to ~1e-14) |
| **7** | Index-based boundary point lookup | Replace O(n·boundaries) float-equality scans | ✅ done (`8e83c37`, `np.unique` grouping; mesh-identical) |
| **8** | `domain_radius="auto"` sizing | Short-spaced tools don't need a 50 m domain | ✅ done (opt-in) — but a **net loss** on suites with a long tool, see benchmark |
| **9** | Warn against GPU for 2D | GPU transfer overhead exceeds gains on small systems | ✅ done (`8e83c37`, advisory in `initialize_workers`) |

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

## Update (2026-07-16): Tasks 6/7/9 landed + four new ideas benchmarked

Tasks **6, 7, 9** were implemented (`8e83c37`); 6 & 7 are pure Netgen-meshing
refactors verified **mesh-identical** (DOF + apparent resistivity to ~1e-14).
Four further exploratory optimizations (`dfa4b74`, env-toggled) were tried and
measured — see [optimization benchmark](optimization-benchmark.md):

- **#1 force the direct solver past the 10k-DOF threshold → 3.48× and exact.**
  The biggest win by far; the threshold is too conservative for 2D.
- **#2 p-adaptivity** (order-2 + order-3 near axis, direct solver) → 3.65×.
- **#3 coarser far-field mesh** (×1.5) → 2.28×.
- **#4 per-tool domain** → net loss (long tool inflates the domain).

**Task 3 (CG `tol`) — dropped for the 2D case.** Typical 2D now uses the direct
solver (Task 5), not CG, so an explicit CG tolerance is moot for the 2D path. It
would only matter for large 2D (above the direct-solver DOF threshold) or 3D
that fall back to CG, where it guards the
[silent no-convergence caveat](../concepts/fem-solver.md) — so it is out of scope
for 2D optimization.

## Open threads

- ~~No recorded benchmark numbers~~ **→ done:** realized before/after (100 samples,
  original vs each optimization) is now quantified in
  [optimization benchmark](optimization-benchmark.md). Headline: forcing the
  direct solver = 3.48× and exact.
- The [sensitivity analysis](../../remo3d/sensitivity.py) commit (`63f7694`) is
  separate work, not an optimization — it deserves its own findings page.

## Links

- Mechanisms: [FEM solver](../concepts/fem-solver.md), [parallel execution](../concepts/parallel-execution.md), [performance & accuracy](../concepts/performance-and-accuracy.md).
- Validation: [benchmarks](../concepts/validation.md). Plan: [`../../performance_optimization_tasks.md`](../../performance_optimization_tasks.md).
