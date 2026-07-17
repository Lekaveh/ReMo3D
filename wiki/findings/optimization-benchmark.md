---
title: Optimization Benchmark — realized speedups
type: finding
tags: [optimization, performance, solver, benchmark, direct-solver, optim-branch, thread-pinning, batching]
sources: [repo-docs]
updated: 2026-07-17
---

# Optimization Benchmark — realized speedups & accuracy

*Fills the open gap flagged in [optimization changes](optimization-changes.md):
the **why** was documented, the **realized before/after** was not.* This page
records measured numbers on the `optim` branch (commits `8e83c37` + `dfa4b74`).

## Method

- **Dataset:** 100 synthetic pseudowells (`scripts/generate_benchmark_data.py`),
  5 normal tools (A0.4M0.1N … A8.0M1.0N), gmsh mesh, domain 40 m, 128 depths.
- **Baseline V1 = original `main`** (all optimizations off). Verified **bit-identical**
  to `main` on a model panel, so speedups are honest before/after.
- **Instrument A** (`scripts/benchmark_optimizations.py`): full MPI pipeline, per-tool
  apparent-resistivity logs + end-to-end wall time, one variant per run.
- **Instrument B** (`scripts/benchmark_solver_phases.py`): single-process per-phase
  timing (assembly / factorization / solve) + DOF/solver-type.
- Merge + summary: `scripts/analyze_benchmark.py` → `benchmark_data/optim_bench/summary.md`.

Production DOF (gmsh, order 3) is **38–55k** (median ~45k) — an order of magnitude
above the `DIRECT_SOLVER_DOF_THRESHOLD = 10000`, so the auto direct solver never
triggers and V1–V6 all run on CG. That single fact drives the headline result.

## Results (100 samples, vs original V1)

Legend: ✓ on · – off · "auto→CG" = the 10k threshold never fires.

| Variant | sym | reuse | cond | order | direct | coarse-far | speedup | max rel-err | mean rel-err | verdict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|--:|--:|--:|:-:|
| V1 baseline (`main`) | – | – | – | 3 | – | – | 1.00× | — | — | base |
| V2 +symmetric (T1) | ✓ | – | – | 3 | – | – | 1.01× | 0 | 0 | ✅ exact |
| V3 +reuse (T2) | ✓ | ✓ | – | 3 | – | – | 1.25× | 0 | 0 | ✅ exact |
| V4 +condense (T0) | ✓ | ✓ | ✓ | 3 | – | – | 2.02× | 1.1e-5 | 6e-8 | ✅ |
| V5 all_on | ✓ | ✓ | ✓ | 3 | auto→CG | – | 2.01× | 1.1e-5 | 6e-8 | ✅ |
| V6 order2 (T4) | ✓ | ✓ | ✓ | **2** | auto→CG | – | 2.79× | **40%** | 0.33% | ⚠️ tradeoff |
| **#1 direct forced** | ✓ | ✓ | ✓ | 3 | **✓ forced** | – | **3.48×** | **7.4e-6** | 2e-8 | ✅✅ **best** |
| **#2 p-adaptivity** | ✓ | ✓ | ✓ | **2 +3 near axis** | **✓ forced** | – | **3.65×** | 11% | 0.09% | ⚠️ fastest |
| #3 coarse-far ×1.5 | ✓ | ✓ | ✓ | 3 | – | **✓** | 2.28× | 1.8% | 0.023% | 🆗 safe-ish |
| #4 per-tool domain | ✓ | ✓ | ✓ | 3 | auto→CG | – | net loss | 4.3% | 0.18% | ❌ |

Reading the cumulative ladder V1→V5: **T1 is timing-neutral, T2 (assemble-once)
buys 1.25×, condensation (T0) another 1.62× → 2.0× total, all exact.** Beyond that
the design space forks into the four ideas below.

## The four explored ideas — method + code + result

### #1 — force the direct solver (the win) ✅✅
**What:** the `sparsecholesky` direct inverse, but forced past the 10k-DOF
threshold. 2D FEM matrices are cheap to factor; on 20–45k-DOF systems a single
factorization + back-substitution beats a full multigrid-CG solve **and** carries
no iterative error.
**Code:** `direct_solver` flag on `AssembleSystem`/`SolveBVP`
(`ngsolve_functions.py`): `"auto"` (threshold), `True` (force, 2D, needs SPD),
`False`. Requires `symmetric=True`.
**Result:** **3.48× and essentially exact (7.4e-6)** — faster *and* more accurate
than every CG variant. The `DIRECT_SOLVER_DOF_THRESHOLD = 10000` is far too
conservative and should be raised or removed for 2D.

**► How to invoke it:**
```python
model.simulate_logs(depths, direct_solver=True)          # symmetric=True is default
# or one-shot:
Model.compute_synthetic_logs(tools, depths, formation, borehole, direct_solver=True)
```
(Or lift the threshold in `ngsolve_functions.py` so `"auto"` picks it up.)

### #2 — p-adaptivity (order-2 base + order-3 near the axis) ⚠️
**What:** cheap order-2 everywhere, order-3 only on elements near the borehole
axis where sharp resistivity boundaries drive the low-order error.
**Code:** `ngsolve_functions.py:AssembleSystem`, env `REMO3D_PADAPT_RADIUS` (m) —
`fes.SetOrder(NodeId(ELEMENT, ...), order+1)` for elements with centroid radius < R.
**Requires the direct solver** — variable p-order produces NaN with CG+multigrid.
**Result:** **3.65× (fastest)**, mean 0.09% (vs uniform order-2's 0.33%), but
worst case 11% (still far below uniform order-2's 40%). Widening the order-3 zone
would trade speed for a smaller tail.

### #3 — coarser far-field mesh 🆗
**What:** grow the radial mesh-size field faster far from the axis (the far field
is smooth). **Code:** `gmsh_functions.py:ConstructGmsh2dModel`, env
`REMO3D_FAR_MESH_FACTOR` — size field `factor*x + 0.1`.
**Result:** factor **1.5** → 2.28×, mean 0.023%, worst 1.8%. Factor 2 coarsens the
near field too and **breaks robustness (NaN)** — the mesh already coarsens as ∝x,
so there is limited safe headroom.

### #4 — per-tool domain ❌
**What:** size the domain to each task's tool (short tools shouldn't pay for a big
domain). **Code:** `workers/worker.py`, env `REMO3D_PER_TOOL_DOMAIN` — per-task
`max(10·max|electrode|, 5)`. **Result: net loss** for this suite — the long
A8.0M1.0N tool forces an ~80 m domain (bigger than the fixed 40 m), inflating
total DOF (median 76k). Same failure mode as `domain_radius="auto"`.

## 2026-07-17 addendum — thread pinning + batch axis (pinned direct)

**Worker thread pinning (the HPC ">1k workers" fix).** Root-cause verified:
`Inverse(inverse="sparsecholesky")` wakes **NGSolve's own TaskManager** (one
std::thread per core, per worker) which **ignores `OMP_/MKL_NUM_THREADS`**;
`import numpy` separately parks a node-sized dormant MKL pool in every worker.
In vivo (24 workers, 64 cores): unpinned direct phase peaked at **571 running /
4407 total threads**; pinned = **24 running / 72 total**. Fix in
`workers/worker.py`: env caps before numpy import **plus `ngs.SetNumThreads(1)`
(the essential part)**. Pinning is also *faster*: factorization ~2.8× (368→134
ms standalone), full pipeline **11.84 → 9.60 s/sample**. CG never enters
TaskManager (pinning-neutral). Details: `WORK_SUMMARY.md` B.3.

**Axis study** (100 samples, pinned direct workers, vs V1; base = symmetric +
reuse + condense + order 3 + domain 40, SEC on, `benchmark_data/optim_bench/summary_axes.md`):

| Config | s/sample | speedup | max rel-err | mean rel-err |
|---|--:|--:|--:|--:|
| CG, batch 5 (== V4) | 20.72 | 1.99× | 1.1e-5 | 6.5e-8 |
| **direct pinned, batch 5** | **9.60** | **4.30×** | **7.4e-6** | 1.6e-8 |
| direct pinned, batch 10 | 8.45 | 4.88× | 10% | 0.51% |
| direct pinned, batch 15 | 5.85 | 7.05× | 11% | 0.47% |

- **Batching is the approximation axis**: batch 5 is solver-exact; batch 10
  jumps to a 10% worst-case tail (sawtooth error over the whole well, worst on
  the short A0.4M0.1N tool — geometric shift from the batch-mean depth; figures:
  `figures/sample_{86,39,98}_baseline_vs_b10.png`). Interesting: batch 10 → 15
  barely worsens the tail (10→11%, mean saturates ~0.5%) while speed jumps
  8.45 → 5.85 s — the error saturates past the initial cliff, so **if ~10% tail
  is acceptable at all, batch 15 dominates batch 10**.
- SEC-off / batch-1 / condense-off cells were dropped from the 100-sample run
  (2-sample smoke, unpinned, indicative only: SEC-off ≈ 2× slower both solvers;
  batch-1 direct 3× slower than batch-5 — 275 tiny factorizations; condense-off
  ≈ +54% on CG but ≈ neutral on direct).
- **len512 validation** (10 samples, 256 depths, 51 m wells, fresh V1+b5
  reference at 40 workers = 72.76 s/sample): direct+pin+b5 = 41.46 s at 20
  workers, 31.38 s at 40 → **2.32× vs V1 at equal (40) workers**, max rel-err
  **3.2e-7** (even tighter than the short wells' 7.4e-6); w20 and w40 logs are
  **bit-identical** (pinned solver fully deterministic). Caveat: worker scaling
  is weak — 20→40 workers bought only 1.32× (parallel efficiency ~66%,
  ~110 tasks/sample → dispatch waves + master serialization), so the direct
  solver's edge narrows at high worker counts where V1's long CG tasks
  parallelize more evenly.

## Recommendations (updated 2026-07-17)

1. **Production config: direct solver + pinned workers + batch 5** — 4.30×,
   solver-exact (7e-6), and safe on HPC (`cpu_workers ≈ cores`, 1 thread each,
   `REMO3D_WORKER_THREADS=1` default). The old ">1k workers" HPC failure is the
   TaskManager oversubscription, fixed by the pin; re-enable `direct_solver`.
2. If a ~10% worst-case tail is tolerable, **batch 15 gives 7.05×** — and
   dominates batch 10 (same tail, much faster). Map batch ∈ (5, 10) if a
   tighter tail at ~5-6× is wanted.
3. p-adaptivity (#2) is worth pursuing if the worst-case tail is tightened.
4. Order-2 (V6) only if 40% boundary error is acceptable — #1 dominates it.
5. Drop per-tool / auto domain for suites containing a long tool.

## Gotchas (for reproducing)

- gmsh writes to a shared `./tmp` → **never run two gmsh benchmark processes
  concurrently** (rmtree race → crash). Run variants sequentially.
- Long runs: launch detached (`setsid nohup`); the harness checkpoints one npz per
  variant, so a killed run resumes from the next variant.

## Links
- Mechanism: [FEM solver — direct vs CG](../concepts/fem-solver.md#direct-vs-iterative-chosen-by-size).
- What changed & why: [optimization changes](optimization-changes.md).
- Cost/fidelity knobs: [performance & accuracy](../concepts/performance-and-accuracy.md).
