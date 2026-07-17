---
title: "Source: Deep research — GPU-first solver for the axisymmetric potential problem (ChatGPT)"
type: source
tags: [source, external, llm-report, gpu, solver, sparse-direct, performance, roadmap]
sources: [deep-research-gpu-solver]
updated: 2026-07-17
---

# Source — Deep research: GPU-first решатель для осесимметрической задачи потенциала

**Slug:** `deep-research-gpu-solver` · **Kind:** LLM deep-research report
(ChatGPT) commissioned by the user · **Ingested:** 2026-07-17 ·
**Raw:** [`raw/2026-07-17-deep-research-gpu-first-axisymmetric-solver.md`](../raw/2026-07-17-deep-research-gpu-first-axisymmetric-solver.md)

> **Citation provenance.** The markdown export in `raw/` lost its references
> (corrupted into `citeturn…` artifacts). The user's PDF export carries the
> real footnotes (1–24); they are restored in [References](#references-restored-from-the-pdf)
> below. The PDF itself was provided in-conversation and is not stored in the repo.

## Context

Commissioned in response to the GPU forward-solver v1 verdict
([`../../WORK_SUMMARY.md`](../../WORK_SUMMARY.md)): the batched block-Thomas
direct solver reached **×5.2** over the forced-direct 32-worker CPU pipeline and
was declared at a "compute-bound" ceiling (batched Cholesky of dense ~103×103
Schur blocks at ~307 GFLOP/s ≈ 0.8 % of A6000 fp32 peak; four kernel-level
attacks failed, CUDA Graphs probe ×0.93 decisive). The report answers: *is that
ceiling real, and what would break it?*

## Key claims

1. **The ×5.2 ceiling is a property of the current combination, not hardware.**
   0.8 % of peak with low memory throughput is, by Nsight taxonomy,
   *latency/parallelism-bound*, not compute-bound; the claim "hard ceiling" is
   premature before per-kernel SOL/occupancy/stall profiling. [1, 6, 7, 21, 22]
2. **Bet #1 — global operator + factor-once/solve-many.** Since conductivity is
   fixed along a pseudowell and only source/receiver positions change, assemble
   one global SPD operator, factorize once, solve many RHS; reciprocity can
   further cut unique RHS. Stack: cuDSS (multi-RHS, refactorization, batching,
   Schur-complement mode) and cuSolverRF for same-sparsity sequences. Declared
   the *only* path with a shot at a regime change (est. 3–15×). [2, 13, 23]
3. **Local fallback = a combination, not a library swap:** singularity
   subtraction `u = G + v` (analytic Green's part near the electrode) [8, 9],
   log-FEM/enrichment near the well [11, 12], radial coordinate mapping — all
   aimed at cutting `n_r`, which enters the local cost as `O(n_z·n_r³)`;
   plus custom **persistent batched kernels** (cuSolverDx device-side
   `POTRF/TRSM/POTRS` with `BatchesPerBlock` tuning [14, 17], MAGMA as the
   small-matrix baseline [4], CUTLASS grouped persistent kernels).
4. **CUDA Graphs are explicitly *not* the answer** — they remove host launch
   overhead only, not intra-kernel serialization (consistent with our measured
   ×0.93). [3, 18, 20]
5. **Mixed precision**: TF32/BF16 GEMM/SYRK Schur updates + FP32 panels +
   iterative refinement, est. 1.2–2.5×. [19]
6. **Multigrid only as *structured geometric* MG** with semicoarsening + line
   smoothers [15]; generic AMG/AMGX is a baseline, not a breakthrough candidate.
7. **HSS/HODLR/BLR** on Schur/frontal blocks is theoretically grounded (low
   off-diagonal numerical rank for elliptic PDEs [5, 10, 16]) but is an
   *escalation* path — low GPU production maturity, high cost.
8. **Decision matrix — three measurable experiments** before committing:
   (a) global operator amortized time < 40 % of current GPU baseline at 16/64/256
   RHS; (b) reciprocity cuts unique RHS ≥ 1.5×; (c) boundary error saturates as
   the global domain grows. Pass (a)+(b) → move the main budget to the global
   branch.
9. **Benchmark hygiene**: log CPU/GPU models, library versions, clocks, ECC,
   power cap; measure **joules per sample**, not just wall time. [24]

Expected-speedup table (the report marks these as *its own estimates, not
promises*): global factorization 3–15×; local GPU-native persistent kernels
1.5–3×; singularity subtraction + smaller `n_r` 1.5–4×; mixed precision
1.2–2.5×; geometric MG 2–8×; HSS/HODLR 2–6×; generic AMG 1–3×.

## References (restored from the PDF)

| # | Reference | Supports |
|---|---|---|
| 1 | [Nsight Compute Documentation (2019.5)](https://docs.nvidia.com/nsight-compute/2019.5/NsightCompute/index.html) | profiling methodology |
| 2, 13 | [cuDSS Functions — NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/functions.html) | cuDSS capabilities (multi-RHS, refactorization, batching, Schur mode) |
| 3, 20 | [CUDA C++ Programming Guide (12.1.1)](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/index.html) | CUDA Graphs, kernel-launch cost |
| 4 | [Fast Cholesky Factorization on GPUs for Batch and Native Modes in MAGMA (ICL)](https://icl.utk.edu/node/1102) | small-matrix batched kernels |
| 5, 10, 16 | [On the Numerical Rank of the Off-Diagonal Blocks of Schur Complements of Discretized Elliptic PDEs (SIAM J. Matrix Anal. Appl.)](https://epubs.siam.org/doi/10.1137/090775932) | low-rank Schur theory → HSS/HODLR |
| 6 | [Nsight Compute Kernel Profiling Guide (2020.1.2)](https://docs.nvidia.com/nsight-compute/2020.1.2/ProfilingGuide/index.html) | SOL/occupancy/stall taxonomy |
| 7 | [Nsight Compute CLI (2020.3)](https://docs.nvidia.com/nsight-compute/2020.3/NsightComputeCli/index.html) | metric collection |
| 8, 9 | [Solving Elliptic Problems with Singular Sources Using Singularity Splitting Deep Ritz Method (SIAM J. Sci. Comput.)](https://epubs.siam.org/doi/10.1137/22M1520840) | singularity subtraction `u = G + v` |
| 11, 12 | [Logarithmic finite element interpolation of flow near wells in phreatic aquifers (Adv. Water Resour., 1985)](https://www.sciencedirect.com/science/article/pii/0309170885900508) | log-FEM/enrichment near wells |
| 14, 17 | [NVIDIA cuSolverDx Documentation](https://docs.nvidia.com/cuda/cusolverdx/) | device-side POTRF/TRSM, `BatchesPerBlock` |
| 15 | [The Improved Robustness of Multigrid Elliptic Solvers Based on Multiple Semicoarsened Grids (SIAM J. Numer. Anal.)](https://epubs.siam.org/doi/10.1137/0730010) | semicoarsening / line smoothers |
| 18 | [CUDA C++ Programming Guide (13.0.0)](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html) | `cp.async` alignment, warp shuffle |
| 19 | [Mixed Precision Block Fused Multiply-Add: Error Analysis and Application to GPU Tensor Cores (SIAM J. Sci. Comput.)](https://epubs.siam.org/doi/10.1137/19M1289546) | mixed precision + refinement |
| 21, 22 | [Nsight Compute CLI (2023.3)](https://docs.nvidia.com/nsight-compute/2023.3/NsightComputeCli/index.html) | metric names/sections, `--query-metrics` |
| 23 | [NVIDIA cuDSS (Preview) overview](https://docs.nvidia.com/cuda/cudss/) | cuDSS scope; 48 GB A6000 feasibility |
| 24 | [On the performance and energy efficiency of sparse linear algebra on GPUs (ICL)](https://icl.utk.edu/node/990) | joules-per-sample benchmarking |

## Assessment

**Strengths.**
- The central move — attack the *architectural assumption* ("each tool position
  = a fresh local factorization") rather than the kernels — is exactly the axis
  v1 never tested. In our workload σ(r,z) is fixed per pseudowell; the 1280
  solves per sample differ only by source position, i.e. textbook
  factor-once/solve-many. And the project *already* exploits reciprocity on the
  CPU path ([SEC](../concepts/parallel-execution.md)), so claim (2)'s RHS
  reduction has an in-repo precedent.
- Internally consistent with our measurements it did not see in detail: its
  CUDA-Graphs skepticism matches the ×0.93 probe; its MG advice (line
  smoothers/semicoarsening) matches WORK_SUMMARY open item A.7; the `n_r³`
  leverage matches our cost model.
- The decision matrix (claim 8) is cheap and falsifiable — days, not weeks.

**Caveats.**
- **Citation base is thin relative to claim strength**: mostly NVIDIA docs
  (several to *outdated* versions — Nsight 2019.5/2020.x) plus a handful of SIAM
  papers. Library-capability claims (cuDSS batching/Schur mode, cuSolverDx
  limits) must be re-verified against current docs before building. Also note
  cuDSS/cuSolverDx CUDA-version requirements vs our nvcc 11.8 toolchain.
- The **speedup table is self-declared guesswork**; do not quote as forecast.
- **Singularity subtraction is overweighted for us**: much of `n_r = 103`
  resolves *material* radii (borehole wall, invasion fronts), which subtraction
  does not remove; the canonical grid already cut radial nodes 334 → 103.
- **Underweights the cost of leaving JAX** for CUDA C++ first-class: loses
  autograd (the unexploited Fréchet-kernel bonus, A.7) and the XLA shape-cache
  story, adds a CUDA maintenance burden.
- It presumes profiling has not been done; v1's negative results (custom kernel,
  Pallas, BCR, Graphs) already rule out several branches it hedges on — though
  our custom kernel was one-block-per-system, *not* the cuSolverDx-style
  multi-batch-per-CTA design it recommends, so the local branch is not fully
  closed.

**Verdict:** adopt the report's decision-driven structure; run the global-path
experiments first. Operationalized in
[`../../GPU_SOLVER_V2_PLAN.md`](../../GPU_SOLVER_V2_PLAN.md).

## Links

- [FEM solver](../concepts/fem-solver.md) — direct-vs-CG, assemble-once/solve-per-RHS (the CPU analogue of factor-once/solve-many).
- [Parallel execution](../concepts/parallel-execution.md) — SEC: the existing reciprocity mechanism.
- [Performance & accuracy](../concepts/performance-and-accuracy.md) — cost/fidelity knobs.
- [Optimization benchmark](../findings/optimization-benchmark.md) — the CPU baseline (forced direct, 3.48×).
- [`../../WORK_SUMMARY.md`](../../WORK_SUMMARY.md) — GPU solver v1: design, validation, the ×5.2 result and the four ruled-out attacks.
