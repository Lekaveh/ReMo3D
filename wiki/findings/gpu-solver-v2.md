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

## Verdict for the fork

**Global path: GO** (pending the GPU-timing milestone). Memory trivial,
RHS amortization ×2.41 on top of factor-once, discretization at parity with
v1, and the accuracy differences are understood and favor v2's conventions.
Next: JAX `lax.scan` port of the global block-Thomas (reuse `direct.py`
machinery) with `vmap` over samples; then the boundary-saturation study and
the production decision on the mud convention.

## Links

- Plan: [`../../GPU_SOLVER_V2_PLAN.md`](../../GPU_SOLVER_V2_PLAN.md) · report that motivated it: [[deep-research-gpu-solver]].
- v1 architecture and the ×5.2 wall: [`../../WORK_SUMMARY.md`](../../WORK_SUMMARY.md).
- Solver concepts: [FEM solver](../concepts/fem-solver.md) · [parallel execution](../concepts/parallel-execution.md) (SEC/reciprocity).
