# GPU Solver v2 — Work Plan (global-factorization program)

**Branch:** `gpu-solver-v2` · **Date:** 2026-07-17
**Inputs:** [`WORK_SUMMARY.md`](WORK_SUMMARY.md) (v1: ×5.2 result + wall
analysis) · [`wiki/sources/deep-research-gpu-solver.md`](wiki/sources/deep-research-gpu-solver.md)
(ingested deep-research report; citations restored from the PDF export)

---

## 0. Framing

v1 verdict: the local per-depth batched block-Thomas solver reaches **×5.2**
over the forced-direct 32-worker CPU pipeline (7.65 s vs ~40 s per len512
sample) and stalls on batched Cholesky of dense ~103×103 Schur blocks
(~307 GFLOP/s ≈ 0.8 % of A6000 fp32 peak). Four kernel-level attacks failed.

The report's central observation, which v1 never tested: **σ(r,z) is fixed per
pseudowell — the 1280 solves per sample differ only by source position.** That
is the textbook factor-once/solve-many regime: one global sparse SPD operator,
one factorization, many RHS. The project already exploits the enabling physics
(reciprocity) on the CPU path via SEC.

This plan is **decision-driven**: cheap decisive experiments (Phase 0), a hard
fork, then investment in exactly one branch.

**Workload constants (len512):** interval 51.2 m = 256 layers × 0.2 m;
5 tools × 256 depths = 1280 BVPs/sample; depth step 0.2 m; per-tool
`h_min = min_gap/20` clipped to [2.5, 10] mm; canonical radial nodes
`n_r ≈ 103`. Targets are quoted against v1's **7.65 s/sample**.

---

## 1. Phase 0 — decisive experiments (E0–E3)

> **STATUS (2026-07-17): Phase 0 run — global path GO.** Results in
> [`wiki/findings/gpu-solver-v2.md`](wiki/findings/gpu-solver-v2.md).
> E0 ✅ (shared grid 1.37M DOF, factor 1.22 GB); E3 ✅ (1280→531 columns,
> ×2.41, source form beats reciprocal); E1 ✅ CPU control = 75 s/sample
> (build 2.9 / factor 6.2 / solve 66; scipy `?pbtrf` unusable — global
> block-Thomas used instead; GPU scan port projected 1–3 s/sample, to be
> measured); E2 ✅ discretization parity 0.03–0.09% — the 3–5% deltas were
> the scalar-mud convention (len512 RM log is noisy ±8%) and the shared
> boundary truncation of v1+NGSolve. Residual E2 work: boundary-saturation
> study + production decision on the mud convention.

### E0 — Global operator sizing (feasibility)
- Build a **global canonical (r,z) grid** spanning the full logged interval:
  z-plateau at `h_z` across all measurement depths (+ far-field geometric pads);
  radial grid = existing `grid.canonical_radial_nodes`.
- Assemble the 5-point FV operator (stencil already in
  `remo3d/gpu_solver/operator_fv.py`) into explicit **CSR** (new
  `gpu_solver/global_op.py`).
- Two variants: **(a)** one grid per tool (native `h_min`, ≤5 factorizations
  per sample); **(b)** one shared grid at the finest `h_min` (1 factorization).
- Back-of-envelope to confirm/refute: `h_z = 5 mm` → ~10.3k fine z-nodes;
  with `n_r ≈ 103–150` → **n ≈ 1.1–1.6 M**, `nnz(A) ≈ 5n`. 2D-type fill
  (nested dissection) → factor well under 1 GB — trivially inside 48 GB.
- Deliverable: table of `n`, `nnz(A)`, symbolic-analysis fill and memory
  prediction per variant. **Gate:** factor memory ≪ 10 GB, analysis sane.

### E1 — Factor-once / solve-many timing
- **CPU control first** (no new GPU code needed): CHOLMOD (`scikit-sparse`) or
  MKL PARDISO; factor once, solve 1280 RHS in blocks of 16/64/256/1280.
  This alone answers whether amortization changes the game — and whether the
  CPU becomes competitive again.
- **GPU:** cuDSS — via the official Python bindings if installable, else the C
  API through `jax.ffi` (toolchain proven by `cuda/ffi_smoke.cu`). Measure
  analysis / factorization / blocked multi-RHS solve separately.
- ⚠️ **Pre-check:** cuDSS CUDA-version requirements vs our nvcc 11.8 toolchain
  and the driver on the shared 3×A6000 box; pick binding path accordingly.
- **Gate (report's criterion):** amortized time < 40 % of v1 → **< ~3 s/sample**
  at matched accuracy.

### E2 — Accuracy & boundary error
- Rerun v1's validation ladder (WORK_SUMMARY §A.4) on the global solution:
  homogeneous analytic check; 28-point vs `SolveBVP`; Ex1 subset; len512 subset
  vs **fresh** NGSolve (stored len512 `model_logs` are not a valid reference).
- **Boundary saturation study:** grow R_dom and z-pads until the log error
  saturates; require ≤ v1's envelope (max ~1 %, mean ~0.2–0.3 %).
- Engineering detail to resolve here: electrodes must land on grid nodes for
  *all* depths → `h_z` must divide the 0.2 m depth step and the tool electrode
  offsets (0.2 m is a multiple of every allowed `h_min`, so this should hold —
  verify per tool).
- **Gate:** error saturates at target accuracy with acceptable memory.

### E3 — RHS-reduction audit (dedup + reciprocity)
- Count **unique current-electrode z-nodes** across 5 tools × 256 depths on the
  shared grid: offsets that differ by multiples of the depth step collapse.
  Expected: ~300 unique nodes instead of 1280 solves (≈4×), before reciprocity.
- Reciprocity accounting (SEC's principle on the global operator): one solve
  per unique electrode node; any (tool, depth) measurement = linear combination
  of nodal samples of those solutions.
- **Gate (report):** ≥1.5× RHS cut; expectation here is materially higher.

**Fork decision** after E1+E2 (E3 sharpens the win but is not gating):
pass → **Phase G**; fail → **Phase L**. Either way, record numbers and the
decision in a new `wiki/findings/gpu-solver-v2.md`.

---

## 2. Phase G — Global path v1 (main branch if the fork passes)

| # | Work item | Notes |
|---|---|---|
| G1 | cuDSS integration | analysis once per grid shape; factorization once per (sample, grid); use the **refactorization** path across samples (same sparsity, new values); blocked multi-RHS sized to memory |
| G2 | Driver backend | extend `gpu_solver/driver.py`: σ once per sample (`sigma_gpu`), CSR values assembled on GPU, solve, sample electrode nodes, assemble logs via existing `tool.py` geometric factors + E3 reciprocity table |
| G3 | Pipelining | overlap assembly of sample *k+1* with solves of sample *k*; optional multi-GPU by samples (3×A6000, subject to colleagues' usage) |
| G4 | Precision policy | fp64 factor first (correctness), then fp32 + iterative refinement — remember `bcr.py`'s fp32-conditioning lesson |
| G5 | Benchmarks | len512 100-sample suite; cold/warm; log library versions, clocks, ECC; **joules/sample** via NVML |
| G6 | Validation gates | extend `scripts/validate_gpu_solver.py`; regression thresholds from E2 |
| G7 | **Adjoint bonus** | the same factorization serves adjoint solves → near-free Fréchet kernels for sensitivity/DOI/inversion — strategic tie-in to `remo3d/sensitivity.py` |

**Target:** ×3–10 over v1 (report's 3–15× discounted), i.e. **~1–2.5 s per
len512 sample**. Robustness bonus of v1 (no mesher to fail on noisy models)
is retained — there is still no mesh.

---

## 3. Phase L — Local path v2 (only if the fork fails)

Ordered by the report's do-not-invert priority: math first, kernels second.

| # | Work item | Notes |
|---|---|---|
| L0 | Nsight diagnosis | profile v1 hot kernels with the report's `ncu` section/metric set; classify latency- vs math-pipe-bound |
| L1 | Microbenchmarks | MAGMA + cuSolverDx batched `potrf/trsm/syrk`, n ∈ {64…128}, B ∈ {256…8192}, vs the 307 GFLOP/s cuSOLVER baseline → measure the *real* kernel headroom (expect ≤2–3×) |
| L2 | `n_r` reduction | (a) radial-mapping audit vs the already-graded grid (marginal?); (b) singularity subtraction `u = G + v` near the electrode — caveat: material-radius nodes (borehole wall, invasion) remain; (c) enrichment. Target `n_r` 103 → ~80 ⇒ ≈2.1× on the cubic term |
| L3 | Persistent batched direct | cuSolverDx device-side factorization, `BatchesPerBlock` tuning, size bucketization; integrate via `jax.ffi` |
| L4 | Mixed precision | TF32 Schur updates + refinement |

**Realistic compound target:** ×1.5–3 over v1.

---

## 4. Out of scope (escalation only)

- HSS/HODLR/BLR Schur compression; structured MG with line smoothers
  (v1 open item A.7) — only if both G and L stall.
- Generic AMG/AMGX baseline — skip (report agrees it is not a breakthrough
  candidate here).

## 5. Risks

| Risk | Phase | Mitigation |
|---|---|---|
| cuDSS unavailable for our CUDA 11.8 toolchain | E1 | check first; fall back to CPU CHOLMOD/PARDISO for the decision experiment — the *architecture* question is solver-agnostic |
| Fill-in/memory blow on the global factor | E0 | symbolic analysis before numeric; per-tool grids (variant a) |
| Blocked multi-RHS triangular solves memory-bound on GPU | E1 | tune block size; CPU may win — accept it, the goal is throughput not GPU pride |
| Electrode-on-node constraint forces too-fine `h_z` | E2 | snap depths/offsets; distributed-delta RHS (validate against ladder) |
| Global-domain boundary error | E2 | saturation study; far-field pads are geometric, hence cheap |
| fp32 conditioning of the global factor | G4 | fp64 default; Jacobi scaling + refinement as in v1 |

## 6. Milestones

| # | Milestone | Exit artifact |
|---|---|---|
| 1 | E0+E1 feasibility | sizing table + amortized-timing table (CPU and, if bindings allow, GPU) |
| 2 | E2+E3 | validation table + unique-RHS count |
| 3 | **Fork decision** | `wiki/findings/gpu-solver-v2.md` with go/no-go and numbers |
| 4 | G1–G2 (or L0–L2) | end-to-end len512 run vs 7.65 s/sample |
| 5 | G3–G5 (or L3–L4) | 100-sample benchmark + energy + versions |
| 6 | G6–G7 | regression gates + adjoint/Fréchet demo |

## 7. Immediate next actions

Phase 0 items 1–4 are DONE (see status above; scripts exist under
`scripts/gpu_v2_*.py`).

> **G1 DONE (2026-07-17): gate PASSED ×2.2.** `global_gpu.py` — jitted
> σ→assembly→factor-scan→solve-scans pipeline, vmap over samples; mixed
> precision (fp64 Schur recursion → fp32 factors/solves; pure fp32 NaNs at
> row ~2153; Ra error 4e-5). **1.40 s/sample @ B=8** (2.58 @ B=4, 9.87 @
> B=1) = ×5.5 over v1, ×28.6 over the CPU pipeline. Both scans are
> latency-bound → B is the lever, bounded by the ~3 GB/sample fp32 forward
> stack. Bench: `scripts/gpu_v2_global_gpu_bench.py`.

Next, in order:

1. E2 residual: boundary-saturation study (`domain_radius` sweep of the
   global grid vs NGSolve at matched R, const-RM data) + decide the
   production mud convention (v2's z-varying column is the more faithful
   physics and the only one compatible with factorization reuse).
2. G2 hardening: full validation ladder (Ex1 + fresh-NGSolve subset),
   100-sample benchmark with energy/sample, multi-GPU sharding by sample
   batches, driver API integration.
3. Optional tuning: larger B via k-chunked forward stack; scan `unroll`;
   per-tool grids to shrink m. cuDSS stays a comparison path only — the
   scan port needed no new toolchain.
