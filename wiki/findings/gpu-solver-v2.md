---
title: "GPU solver v2 — Phase 0: global factor-once/solve-many is viable"
type: finding
tags: [gpu, solver, sparse-direct, block-thomas, amortization, accuracy, mud]
sources: [deep-research-gpu-solver, repo-docs]
updated: 2026-07-17
---

# GPU solver v2 — Phase 0 results (E0–E3)

Decision experiments from [`../../GPU_SOLVER_V2_PLAN.md`](../../GPU_SOLVER_V2_PLAN.md),
run 2026-07-17 on the len512 workload (5 normal tools × 256 depths = 1280
BVPs/sample; v1 GPU baseline 7.65 s/sample). Code:
`remo3d/gpu_solver/global_op.py`, `scripts/gpu_v2_global_sizing.py`,
`scripts/gpu_v2_amortize_cpu.py`, `scripts/gpu_v2_accuracy_check.py`.

## E0 — sizing: PASS (memory is a non-issue)

One global (r, z) grid spans all measurement depths (uniform h_min z-plateau
over the electrode range + geometric far-field tails; canonical radial grid).
The operator is assembled explicitly; with z-major numbering it is
block-tridiagonal with half-bandwidth m = nr−1.

| Variant | grid | n_free | nnz(A) | factor size |
|---|---|--:|--:|--:|
| shared, all 5 tools (h=5 mm, R=90) | 112×12315 | 1.37 M | 6.8 M | 1.22 GB |
| per-tool (5 grids, native h/R) | 91…112 × 5.5–10.7k | 0.56–1.03 M each | 2.8–5.1 M | 0.47–0.80 GB |

## E3 — RHS dedup: ×2.41 (source form beats reciprocal form)

On the shared grid the 1280 (tool, depth) tasks collapse to **531 unique
source columns** (tool current-electrode offsets differ by multiples of the
0.2 m depth step for 4 of 5 tools). The reciprocal formulation (source at
M/N) would need 775 columns — for single-current normal tools the source form
wins; reciprocity is already exploited by [SEC](../concepts/parallel-execution.md)
and adds nothing extra here.

## E1 — factor-once/solve-many, CPU control

`scipy`'s banded Cholesky (`?pbtrf`) is the unblocked reference routine
(~180 s at n=2e5, b=111 — unusable). The right CPU control is **the global
block-Thomas**: v1's own algorithm (dense Schur Cholesky per z-row), but
factored **once per sample** instead of once per (tool, depth). Verified
exact vs `spsolve` (5.5e-15).

Per len512 sample (64-core box, BLAS pinned to 1 thread — threaded OpenBLAS
on 111×111 blocks is a >10× slowdown, the TaskManager-oversubscription lesson
again): **build σ+CSR 2.9 s, factor 6.2 s, solve 531 columns 51–66 s → ~75
s/sample** (thread-pool over RHS blocks is GIL-bound; a process pool or the
GPU scan is the fix). The <3.06 s gate is *not* met on CPU — as expected for
a single-box control; the gate targets the GPU implementation. Key
observations for the GPU port: per-sample factor work is ~2×10¹⁰ flops
(≈1/8 of v1's 1280 window factorizations), and the solve stage is
2 sweeps × 12313 rows of (111×111)@(111×531) GEMMs — far better GPU shapes
than v1's per-window batches; sample-batching via `vmap` amortizes the scan
latency. GPU projection ~1–3 s/sample; to be measured (Phase G milestone 4).

## E2 — accuracy: discretization parity; the differences are *conventions*

Three-way arbitration (v2 global vs stored v1 GPU logs vs fresh NGSolve
gmsh forced-direct) on the worst-disagreement points first showed a puzzle:
short tools sided with v1 (v2 "off" by 3–5.5%), long tools sided with v2
(v1 off by up to 6.5%), errors h-independent. Bisection (my assembler ≡
`apply_A` to 1.8e-12; v1 grid+σ through my solver reproduces v1 to 0.06%)
pinned the whole discrepancy on the **σ field**, and then on two modeling
conventions:

1. **Mud column.** The production path (NGSolve *and* v1 GPU) fills the
   entire borehole with **one scalar RM interpolated at the simulation
   depth**. v2 samples the **physical z-varying RM log**. On len512 the RM
   log is noisy — **0.75–0.88, ±8% between adjacent depths** — so near-source
   mud differs by up to ~8% and short-normal Ra moves by up to ~5.5%,
   exactly at the "spike" depths. With RM forced constant, **v2 and v1 agree
   to 0.03–0.09%** and both match NGSolve to ~0.2%. v2's convention is the
   more faithful physics; the production scalar-mud convention is also
   exactly what would make the global operator depth-dependent and break
   factorization reuse. ⚠️ Any future v2-vs-production comparison on noisy-RM
   data must be like-for-like (constant/smooth RM) or it measures the mud
   convention, not the solver.
2. **Boundary truncation.** v1 and NGSolve share the same truncated domain
   (u=0 at R = max(10·span, 5) from the electrode, in z too). Their mutual
   0.4% agreement silently cancels this shared truncation error: at e.g.
   z=5.8 (A0.4, R=5), growing the NGSolve domain moves the reference by
   **+3.6%** — toward v2, whose z-boundary is naturally far. The remaining
   radial truncation knob (`domain_radius`) affects v2 and the references
   alike; the full saturation study is still open (E2 residual).

## Phase G1 — GPU port: gate PASSED, ×5.5 over v1

`global_gpu.py`: the whole per-sample pipeline in one jitted function —
σ (z-varying mud, jnp) → face conductances → block rows → **factor
`lax.scan`** (dense Schur Cholesky per z-row) → **two solve scans** with all
531 RHS columns at once → axis potentials; `vmap` over samples.

Precision (measured on the len512 shared grid, A6000):

| Mode | Correctness | s/sample (B=1) |
|---|---|--:|
| fp64 | 4.9e-11 vs CPU control | 18.3 (factor 7.5 + solve 10.8) |
| pure fp32 | **NaN** — Schur recursion loses SPD at row ~2153 of 12313, Jacobi scaling doesn't save it (v1's 341-row windows were below the cliff) | — |
| **mixed** (fp64 recursion → fp32 factors + fp32 solves, Jacobi-scaled) | Ra error **4.0e-5** | 9.9 |

Both stages are **latency-bound** (fp64 factor is ~17 ms of pure flops in
7.5 s of wall — it's 12313 sequential scan steps), so sample-batching is
almost free: batch wall time stays ~10–11 s while per-sample cost divides.

| Batch B | warm s/sample | vs v1 (7.65 s) | vs CPU pipeline (~40 s) |
|--:|--:|--:|--:|
| 1 | 9.87 | ×0.8 | ×4 |
| 4 | 2.58 | ×3.0 — **gate <3.06 s PASS** | ×15.5 |
| 8 | **1.40** | **×5.5** | **×28.6** |

B is memory-bound by the forward-sweep stack (~3 GB/sample fp32 at k=544);
B=8 uses ~32 GB of the 48 GB card. Compile is paid once (~20 s). Combined
with Phase 0: the deep-research report's central bet is **confirmed
end-to-end** — the ×5.2 "ceiling" of v1 falls to an architecture change on
the same hardware, with no CUDA C++ and no new toolchain.

## Cross-check on the optim_bench workload (100 samples, 5 tools × 128 depths)

`scripts/gpu_v2_optim_bench.py`, run 2026-07-17 against the stored
`benchmark_data/optim_bench/full_pipeline` references (the real MPI
pipeline, batch_size=5).

**Time** (warm, B=10, single A6000; grid 112×7195, 798k DOF, 275 unique RHS
= ×2.33 dedup): **0.64 s/sample** — ×65 vs the V1 CG baseline (41.25 s),
×19 vs Vd forced-direct (11.84 s), ×15 vs the axis-study pinned-direct b5
(9.60 s). Compile 13.9 s once.

**Accuracy vs stored logs**: mean 1.2–1.7 %, max 10–20 % — and the
arbitration (fresh NGSolve at the worst points, matched const-RM
convention) decomposes it exactly as Phase 0 predicts, plus one NEW term:

1. **Mud convention** (dominant broad term; this dataset's RM log is noisy
   ±4 %): e.g. A1.0 s56 z=4.8 — stored 4.386 vs v2 3.888 collapses to NG
   4.130 vs v2 4.145 (**0.36 %**) at const RM.
2. **The reference's own batch_size=5 error**: at A8.0 s24 z=25.4 the
   stored log (2.324) is **15 % off a fresh unbatched same-convention
   NGSolve (2.727)**, which lands next to v2 (2.793). The stored pipeline
   logs are *not* point-exact — the mesh is shared across 5 depths with up
   to (B−1)·dz/2 = 0.4 m geometric offset.
3. Small boundary residual for the longest tool at the well edge (NG cRM
   moves +3.6 % when R is doubled at A8.0 s24 z=25.4).

In matched conventions v2 agrees with fresh NGSolve to **0.3–1.1 %** at
every arbitrated point — discretization-level, consistent with the Phase 0
parity result.

## E2 closed — boundary saturation; adopted conventions

`scripts/gpu_v2_boundary_sweep.py` (optim workload, samples s0 + s24, worst
over 5 tools × 128 depths, error vs R=2880):

| R | 45 | 90 (v1 conv.) | 180 | 360 | 720 | 1440 |
|---|--:|--:|--:|--:|--:|--:|
| worst truncation | 19.6 % | **6.2 %** | 1.6 % | 0.36 % | **0.065 %** | 0.029 % |

Error decays ~×4 per doubling of R (≈1/R²), dominated by A8.0 at the well
edges; far-field nodes are logarithmic — R=90→720 costs only +14 % DOF
(798k→910k) and +0.10 s/sample.

**Adopted conventions (production decisions):**
1. `domain_radius` default = **max(80·span, 45)** (→720 here): truncation
   ≤0.07 %, safely below the ~0.3 % discretization envelope. The v1/NGSolve
   production convention (10·span) carries up to ~6 % at long-tool edge
   points; matched-convention comparisons must pass it explicitly.
2. **z-varying mud column** (the physical RM log): the only convention
   compatible with factorization reuse, and the more faithful physics.
3. Consequence: v2-vs-stored-reference maxima *grow* at A8.0 edges (to
   ~27 %) because v2 got more accurate while the stored refs keep R=90
   truncation + batch_size=5 meshes. **Regression baselines should be
   recomputed** (unbatched, large-R NGSolve — or v2 fp64).

## G2 (partial) — multi-GPU throughput + energy

Two A6000s concurrently (50 samples each, B=10, R=720): **100 samples in
54.4 s wall** including per-process compile (~14 s); warm 0.74 s/sample/GPU
→ combined warm throughput **0.37 s/sample** ≈ ×32 vs Vd forced-direct.
Power during the run: GPU0+1 mean 257 W → **~140 J/sample** (nvidia-smi
0.5 s polling, compile included). The CPU pipeline's J/sample is not yet
measured (order-of-magnitude estimate at 40 s × node power ≫ this).

## G2 — Ex1 validation ladder + driver API

**Full Ex1** (8 tools incl. laterals × 251 depths, invasion blanked, smooth
RM → mud convention moot) on ONE global grid — no radius bucketing, unlike
v1's depth-relative grids (`scripts/gpu_v2_full_ex1.py`): grid 114×8017,
906k DOF, 2008 tasks → 818 RHS (dedup ×2.45), ~19.5 s incl. compile.
vs frozen `Results_1.txt`: **overall max 1.00 % (M1.0A0.1B), per-tool means
0.08–0.25 %** — matches v1's validated envelope (1.05–1.2 %), laterals
included. Matched-R (110) and default-R (880) agree to <0.005 % here, so
the frozen-reference comparison is boundary-insensitive on Ex1.

**Driver API**: `compute_logs_gpu(..., global_solver=True,
precision="mixed"|"f64")` — same result contract as
`Model.compute_synthetic_logs` (`logs[tool] = [depth, Ra]` columns);
verified against the bench path (6.7e-5, fp32 batching noise).
`dtype/tol/precond/batch_size/backend` are ignored on this path; conventions
are v2's (z-varying mud, far boundary).

## G2 — regression baseline recomputed

`benchmark_data/gpu_solver/global_optim_bench_f64.npz`: **v2-fp64 logs for
all 100 optim_bench samples** (adopted conventions: z-varying mud, R=720;
fp64 is 4.9e-11-exact for this operator). Generated with
`gpu_v2_optim_bench.py --dtype f64` at 3.41 s/sample (B=4, one A6000 —
still ×12 vs the CG pipeline). The production mixed-precision path agrees
with this baseline to **worst 1.9e-4, mean 7.7e-6** over 50 samples — that
is the regression gate to test against, replacing the stored pipeline logs
(which carry R=90 truncation + batch=5 mesh sharing).

## CPU-compat mode (`convention="cpu"`) — matching the pipeline where possible

User requirement: v2 must reproduce the CPU pipeline's numbers; where it
can't, provide an imitation mode without giving up the speed class.

**Design.** The scalar-mud-per-depth convention makes the operator
depth-dependent — fatal for factor-once *naively*. But face conductances are
LINEAR in the mud conductivity ν=1/RM for pure-mud cells, so
A(ν) ≈ A_ref + Δν·A₁ + Δν²·A₂ (quadratic fit; caliper-crossing cells'
harmonic mixes captured to O(Δν³)). Each source column's own ν is then
handled by a 3-term **Neumann series** on the single ν_ref factorization.
Two numerical traps found and fixed on the way: (1) residual-based fp32
refinement stalls at ~1e-2 — catastrophic cancellation of b−Ax near the
singular source (the series form removes b entirely); (2) the ΔA·x stencil
sum itself cancels (discrete divergence of a near-harmonic field), so the
ΔA applies run in fp64 while solves stay fp32. Column chunks execute under
`lax.map` (sequential — a python loop lets XLA schedule chunks concurrently
and blow the memory peak).

**Exactness vs its own convention:** worst **5.2e-5** against per-column
scalar-mud factorizations (native precision floor).

**vs stored pipeline logs (100 samples):** means drop to 0.34–1.4 %
(native: 1.2–1.9 %); residual maxima are the references' own batch_size=5
and truncation errors.

**vs fresh unbatched NGSolve** (samples 0 & 56, every 4th depth): sample 0
(normal mud) — short tools ≤1.0 %, A8.0 0.60 %. Sample 56 (RM≈0.15, salt
mud; 31/100 samples have mean RM<0.5) — short tools disagree by up to 40 %,
and the R-sweep proves whose error it is: **NGSolve converges to the compat
value as its window grows** (A0.4 z=24.8: R=5→0.763, R=20→0.469,
R=80→0.4559 vs compat 0.4566). The pipeline's per-depth ±R z-window is
untruncatable-imitable in a factor-once architecture *and* carries up to
~67 % error of its own in the conductive-channel regime — documented as the
one intentional non-match.

**Cost:** 6.90 s/sample warm (B=4, kc=96) — still faster than every CPU
config (Vd 11.84 s), ~10× slower than native v2 (0.64 s); the refinement
temporaries limit B (XLA buffer bloat — optimization TODO). Driver:
`compute_logs_gpu(..., global_solver=True, convention="cpu")`.

## Verdict

**Global path: CONFIRMED** (was: GO pending GPU timing). Memory trivial, RHS
amortization ×2.41 on top of factor-once, discretization at parity with v1,
accuracy differences understood (and favor v2), the GPU implementation beats
the gate by ×2.2, boundary and mud conventions decided and measured, 2-GPU
sharding demonstrated, **Ex1 ladder passed (max 1.00 %), driver API
integrated, and the fp64 regression baseline recomputed**. Remaining:
fresh-NGSolve matched-convention subset for the record, 3rd-GPU sharding
when the card frees up, and the strategic G7 follow-on (adjoint solves on
the same factorization → Fréchet kernels for sensitivity/DOI/inversion).

## Links

- Plan: [`../../GPU_SOLVER_V2_PLAN.md`](../../GPU_SOLVER_V2_PLAN.md) · report that motivated it: [[deep-research-gpu-solver]].
- v1 architecture and the ×5.2 wall: [`../../WORK_SUMMARY.md`](../../WORK_SUMMARY.md).
- Solver concepts: [FEM solver](../concepts/fem-solver.md) · [parallel execution](../concepts/parallel-execution.md) (SEC/reciprocity).
