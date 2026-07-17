# ReMo3D — Work Summary: GPU forward solver + concurrency/correctness fixes

**Branch of origin:** `gpu-solver` (18 commits on top of `optim`, base `51b461d`)
**Period:** 2026-07-15 … 2026-07-17
**Scope of this document:** everything done on the `gpu-solver` branch — the
full-GPU 2D forward-solver investigation (JAX + CUDA), its results and verdict,
plus the general-purpose concurrency/correctness fixes that came out of it.

> This is a self-contained report. Deeper, cross-linked detail lives in the
> research wiki on the `gpu-solver` branch:
> `wiki/findings/gpu-solver.md` and `wiki/findings/cpu-vs-gpu-compute-pipeline.md`.
> Those wiki pages do **not** exist on `optim`, which is one reason this summary
> is carried here.

---

## 0. TL;DR

- Built a **full-GPU alternative** to the NGSolve/Netgen CPU pipeline for the
  `dip=0` (2D axisymmetric) DC-resistivity forward problem: a structured graded
  `(r,z)` grid + matrix-free operators + a batched direct solver, all in JAX,
  solving **all depths of a tool in one `vmap`-batched GPU call**.
- **Validated to sub-percent** against NGSolve: full Ex1 max **1.05–1.2 %**,
  len512 worst point **0.4 %** vs a fresh NGSolve solve.
- **Throughput: ×5.2** over the (already optimised, forced-direct) 32-worker CPU
  pipeline on the len512 workload — **7.65 s/sample vs ~40 s/sample** — and it is
  **robust on noisy 256-layer models where the CPU mesher returns all-NaN**.
- **The ×10 target was decisively ruled out.** Exact FEM forward modelling here
  is **compute-bound** on a batched Cholesky of dense ~103×103 Schur blocks that
  runs at ~307 GFLOP/s (~0.8 % of the A6000's fp32 peak). Four independent
  attempts to beat it (Pallas, partial cyclic reduction, a custom CUDA kernel,
  CUDA Graphs) all failed for the same root cause: **there is no launch/latency
  overhead to remove — it is raw small-block compute.** ×5.2 is at the practical
  ceiling for this method class on this hardware.
- Along the way, three **general-purpose fixes** were produced (independent of
  GPU work): a **per-run temp-dir isolation** (committed, `c8c4959`), a
  **critical `MPI.FLOAT`→`MPI.DOUBLE` datatype bug fix**, and a **worker
  thread-pinning fix** for the direct-solver oversubscription blow-up on HPC.

---

## Part A — Full-GPU 2D forward solver (branch `gpu-solver`)

### A.1 Motivation

The production path (`remo3d/ngsolve_functions.py` + Netgen/Gmsh) re-meshes and
re-assembles **per depth on the CPU**, then does one sparse linear solve. Two
asks drove this experiment:

1. Move the **entire** FEM pipeline to the GPU — not just the linear solve
   (which `ngsolve_functions_gpu.py` already offloads), but mesh + assembly too.
2. Solve **N depths at once** in a single batched GPU call.

Unstructured body-fitted meshes make both impossible: mesh topology differs per
depth, so nothing can be `vmap`-batched, and Netgen/Gmsh meshing is a serial,
CPU-only geometric algorithm with no GPU version.

**The architectural bet:** replace the body-fitted mesh with a **fixed graded
rectilinear `(r,z)` grid + matrix-free operators**. For `dip=0` all material
boundaries are axis-aligned (beds ⊥ z; borehole wall and invasion front ⊥ r),
so a graded rectilinear grid is effectively body-fitted — little accuracy is
lost. This **eliminates mesh generation and assembly entirely**, and makes every
depth of a tool share one grid topology, so the only batch axis is `σ(r,z)`.

### A.2 Design

| Piece | Choice |
|---|---|
| Grid | data-independent graded 1D node sets; electrodes and material radii are exact/near-exact nodes; `h_min` per tool = `min_gap/20` clipped to `[2.5 mm, 10 mm]` |
| Discretisation | two interchangeable **matrix-free** backends: vertex-centred FV (5-point stencil) and Q1 FEM (9-point, exact 2×2 Gauss of `2πrσ∇u·∇v`) |
| σ sampling | per-cell, **subcell-averaged (4×4)**, **anisotropic**: series-harmonic along the flux, parallel-arithmetic across → direction-dependent `(σ_r, σ_z)` (transmissibility upscaling) |
| Solver (final) | **batched direct block-Thomas** (block-tridiagonal in z, dense Schur Cholesky per z-row); earlier: geometric-multigrid PCG |
| Batching | `vmap` over depths of one tool (shared grid/RHS), chunked (≈32–256) so the worst batch element doesn't stall the rest |
| Precision | fp32 production / fp64 validation (drift ≤ 3e-5 on R_a) |
| Server hygiene | `XLA_PYTHON_CLIENT_PREALLOCATE=false` (shared 3× A6000 host) |

The governing PDE (axisymmetric, after integrating out azimuth):

```
-∇·(2πr · σ(r,z) · ∇u) = I·δ(r=0, z=z_A),   u=0 on the outer boundary R_dom.
```

Output is apparent resistivity `R_a = K·|u(z_M) − u(z_N)|`, `K` the tool's
analytic geometric factor. A full log = this BVP at every depth (e.g. 256) for
every tool (e.g. 5) → **1280 independent BVPs per pseudowell**.

### A.3 What was built, stage by stage

**Stage 1 — Structured-grid skeleton + validation** (`baddb9e`, `05667dc`,
`5887e75`, `2b46aed`)
- FV and Q1 matrix-free backends (`operator_fv.py`, `operator_fem.py`) on graded
  `(r,z)` grids (`grid.py`).
- Geometric-multigrid PCG (`mg.py`): non-nested graded levels (`h×2^l`), σ
  re-sampled per level, bilinear tensor-product prolongation, restriction as the
  exact transpose via `jax.linear_transpose`, damped-Jacobi smoother.
- Anisotropic subcell σ upscaling — the key to sub-percent accuracy without a
  body-fitted mesh.
- Per-tool `h_min` heuristic + batched driver API.

**Stage 2 — GPU σ-sampling + all-tools-at-once** (`c6d655b`)
- `sigma_gpu.py`: jnp port of the σ-sampler (`searchsorted`/`interp`/`where`),
  **bit-identical** to the numpy path (≤6.7e-16) and **×29–348 faster**; σ stacks
  are born device-resident. Full-Ex1 per-tool wall time 666 s → **531 s**.
- Shared-grid mode (`driver.build_shared_problem` / `solve_shared_log`,
  `--shared`): centre the grid on the **measurement point** so σ(r,z) is shared
  by every tool at a depth; tools differ only by RHS + sample nodes. One grid,
  one MG hierarchy, one compile, nested `vmap` (depths × tools).
  Radius-bucketing keeps each shared grid homogeneous.
- **Discovery:** the real bottleneck became **JIT compilation, not compute** —
  ~30–50 s per distinct grid shape, 0.3 s on a shape cache-hit.

**Stage 3 — Canonical grids (compilation amortised)** (traced + fixed)
- Per-sample recompilation was traced to **data-dependent grid shapes**: radial
  foci included the formation's invasion radii, so every noisy len512 sample
  shifted the node count and forced a fresh XLA compile.
- Fix (`grid.canonical_radial_nodes`, driver `r_foci=None`): a
  **data-independent** radial grid — uniform `h_min` plateau to `r_fine=0.6 m`
  then geometric growth to R. Grid shape now depends only on the tool → **same
  tool, every sample → identical shape → XLA cache hit** (compile paid once per
  tool per session). Bonus: canonical grids are *smaller* on noisy models
  (len512 A2.0 radial nodes 334 → 103), so solves are faster too.

**Stage 4 — Direct block-Thomas solver + the 10× wall** (`827de01`, `8f56252`,
`ae382c8`, `50cae58`, and the negative-result probes below)
- MG-PCG was profiled and found **not** latency-bound: 101–249 Jacobi-smoothed
  iterations, and the `vmap` batch waits for its worst member (efficiency
  1.0→0.5 by B=1024).
- Replaced it with a **direct block-tridiagonal solver** (`direct.py`): the FV
  operator is block-tridiagonal in z (nz≈341 blocks, each nr≈103 tridiagonal),
  solved by **block-Thomas** with per-row **Cholesky of the dense Schur
  complements**, symmetric Dirichlet elimination, Jacobi scaling, and one
  matrix-free iterative-refinement step (fp32-stable: 5e-6 vs fp64).

### A.4 Validation ladder (vs NGSolve)

| Check | Result |
|---|---|
| Homogeneous medium (R_a = ρ analytically) | FV 3–10e-4, FEM ~2e-3 |
| 28 (tool, depth) points vs `SolveBVP` (order 3) | worst **0.84 %**, median ~0.2–0.3 % (after per-tool `h_min` fixed a short-spacing outlier 2.1 % → 0.086 %) |
| Layer-boundary spikes (electrode near a bed) | subcell aniso upscaling cut worst 4.1 % → **0.08 %** |
| Full Ex1 (8 tools × 251 depths) vs frozen `Results_1.txt` | overall max **1.2 %** (per-tool grid), **1.05 %** (shared grid); per-tool mean 0.18–0.31 % |
| Direct solver vs MG | **1.3e-4** |
| len512 worst point vs a **fresh** NGSolve solve | **0.4 %** |

Note: the repo's 1e-4 gate is an NGSolve-vs-NGSolve regression bound; two
*different* discretisations legitimately differ at the sub-percent level.
(The stored len512 `model_logs` are **not** a valid accuracy reference —
formation re-noised after the logs were computed — so a fresh NGSolve solve is
the correct reference.)

### A.5 Performance

**len512** (5 tools × 256 depths = 1280 solves/sample, fp32, single A6000):

| Solver | Warm wall/sample | vs CPU |
|---|---|---|
| MG-PCG (canonical grids) | 87 s | slower than CPU |
| **Direct block-Thomas** | **7.65 s** | **×5.2** vs the forced-direct 32-worker CPU (~40 s) |

- Compile (~200 s) is paid once per tool per session, then amortises to ~0 over
  many samples — which is exactly the workload the GPU path targets.
- On a **single** sample, compilation dominates and the GPU does not beat the
  20-core CPU pipeline. The GPU win requires **many samples sharing tool shapes**.
- **Robustness bonus:** on the noisy 256-layer len512 models the real 32-worker
  CPU pipeline returns **all-NaN** (Netgen `SplineGeometry` meshing fails); the
  GPU path has **no mesh to fail** and produces valid results.

### A.6 The ×10 wall — four approaches, all ruled out

The direct solve's cost is `≈ nz · O(nr³)` per system ≈ 0.5 TFLOP per model, and
the right parallel axis is a **batched Cholesky across systems**. Measured on the
A6000 (fp32): a batched Cholesky of B=256 matrices of nr=103 takes **0.303 ms**
= **307 GFLOP/s = ~0.8 % of peak**. A 103×103 factorisation is simply too small
to fill the GPU (limited parallelism, long 103-step dependency chain, low
arithmetic intensity). This is a **hardware property**, not overhead:

| Attempt | File | Result | Root cause |
|---|---|---|---|
| Iterative MG-PCG | `mg.py` | 87 s (×0.5) | 101–249 iters; batch waits for worst member |
| Pallas fused Thomas | `pallas_thomas.py` | ~50× worse + fp32-NaN | dense in-block Cholesky is sequential/small; Triton can't express O(nr³) |
| Partial cyclic reduction | `bcr.py` | fp32-NaN, no speedup | Schur recursion ill-conditions reduced blocks in fp32 |
| Custom block-per-system CUDA kernel | `cuda/block_thomas.cu` | ×6 **worse** | one thread-block can't parallelise the O(nr³) dense Schur update — wrong axis |
| **CUDA Graphs on the cuSOLVER scan** | `cuda/graph_probe.cu` | **×0.93 (decisive)** | capturing/replaying 341 batched-`potrf` calls gave **no** speedup (0.303 → 0.325 ms/step) — **nothing to remove**; the 0.3 ms is real kernel time |

The CUDA FFI toolchain itself was proven viable (`cuda/ffi_smoke.cu`: nvcc 11.8
→ `.so` → `jax.ffi` runs on JAX's stream), so the negative results are about the
algorithm/hardware, not tooling. **Verdict:** exact FEM forward modelling is
compute-bound on batched small-block factorisation here; ×5.2 is near the
practical ceiling. Going past it needs a **cheaper method class** (reduced-order
model / neural surrogate) or **newer hardware**, not a faster execution of the
same FLOPs.

### A.7 Open items (carried on `gpu-solver`)

- **Decouple vertical extent from R** — `build_grid` ties the z-domain to
  ±domain_radius; for long tools most z-nodes are far-field waste.
- **Grid-shape determinism** — `graded_1d` occasionally yields N vs N±1 nodes for
  near-identical tools, defeating the shape cache; snap to canonical shapes.
- **Line/semicoarsening smoother** would cut MG iterations (Jacobi is weak on the
  graded axisymmetric stencil).
- **Multi-GPU sharding** over the batch axis (3× A6000) is mechanical
  (`jax.sharding`) but limited while colleagues occupy the other cards.
- **Differentiability bonus (unexploited):** the solver is differentiable
  (`jax.grad` through `solve`) — a route to fast Fréchet kernels for
  sensitivity/DOI and inversion.

### A.8 Code map (all on `gpu-solver`)

```
remo3d/gpu_solver/
  grid.py            graded/canonical (r,z) grids, electrode/material nodes
  operator_fv.py     matrix-free 5-point finite-volume operator
  operator_fem.py    matrix-free 9-point Q1 finite-element operator
  sigma_gpu.py       GPU anisotropic subcell σ-sampler (jnp)
  mg.py              geometric-multigrid PCG preconditioner
  direct.py          batched direct block-Thomas solver (the final path)
  solve.py, driver.py, tool.py   batched driver API + tool geometry
  pallas_thomas.py   NEGATIVE RESULT — Triton/Pallas fused Thomas
  bcr.py             NEGATIVE RESULT — partial cyclic reduction
  cuda_thomas.py     jax.ffi wrapper for the CUDA path
  cuda/
    ffi_smoke.cu     proven-viable FFI smoke test
    block_thomas.cu  NEGATIVE RESULT — custom block-per-system kernel
    graph_probe.cu   DECISIVE — CUDA-Graphs probe (×0.93)
scripts/
  validate_gpu_solver.py, benchmark_gpu_batch.py, gpu_full_ex1.py,
  gpu_len512_amortize.py, test_sigma_gpu.py
  gpu_smooth_noise_bench.py, gpu_smooth_noise_bench_batched.py  (uncommitted WIP)
```

---

## Part B — General-purpose concurrency/correctness fixes

Three fixes fell out of the GPU work but are **independent of it** and belong in
the production/optimisation line. **Only `c8c4959` is being carried to `optim`
now (committed); the other two remain as uncommitted WIP on `gpu-solver`.**

### B.1 Per-run temp-dir isolation — `c8c4959` (COMMITTED → carried to `optim`)

**Problem:** gmsh mesh files were written to a hardcoded `./tmp` in the working
directory. Concurrent runs sharing a CWD (or an NFS mount) collided on
rank-numbered mesh files and **raced on cleanup** — `shutil.rmtree("./tmp")`
could delete another run's files or crash.

**Fix:** a per-invocation directory under the system temp dir
(`remo3d_tmp_<user>_<host>_<pid>_<uuid8>`). The manager generates it once in
`initialize_workers()` and broadcasts it to the spawned workers (an extra `bcast`
right after `solve_on`, matched worker-side); the worker overrides the
module-level `gmsh_functions.TMP_DIR` fallback. Cleanup uses
`rmtree(..., ignore_errors=True)`. Files touched: `remo3d/gmsh_functions.py`,
`remo3d/remo3d.py`, `remo3d/workers/worker.py`. Verified end-to-end with a gmsh
forward run under `mpiexec -n 1`.

### B.2 Critical `MPI.FLOAT` → `MPI.DOUBLE` datatype bug (uncommitted WIP)

**Problem (long-standing):** the manager broadcast the float64 arrays
(`formation_model`, `borehole_geometry`, `mud_resistivities`,
`simulation_depths`) with **`MPI.FLOAT`** while the worker receive buffers are
float64 (`dtype='float'`). The datatype mismatch **reinterprets the float64
payload and delivers strided-garbage geometry** to the workers (worker row *k* ←
manager row *5k*), which then **crashes gmsh meshing → NaN logs**.

**Fix:** use `MPI.DOUBLE` on both the send (`remo3d/remo3d.py`) and receive
(`remo3d/workers/worker.py`) sides, and cast the send buffers to contiguous
float64 (`np.ascontiguousarray(..., dtype=np.float64)`) so the datatype always
matches regardless of the upstream array dtype. This is a **correctness** fix.

### B.3 Worker thread-pinning — direct-solver oversubscription (uncommitted WIP)

**Problem:** each equation solve runs in its own MPI worker process, so
parallelism already lives at the process level. Left unpinned, the direct
(sparse-Cholesky) solver's dense-block LAPACK calls (MKL/OpenBLAS) spin up a
thread pool sized to the **whole node's core count in every worker** —
`cpu_workers × cores` threads (e.g. 100 × 64 ≈ 6400). That is the observed
“>1000 workers” blow-up on HPC; it only surfaced with `direct_solver` on because
the iterative CG/multigrid path never hits dense BLAS.

**Fix (in `remo3d/workers/worker.py`, before numpy/ngsolve import):** pin the
thread-pool env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`) to one
thread per worker (override via `REMO3D_WORKER_THREADS`), plus a
belt-and-suspenders `ngs.SetNumThreads(...)`. MPI supplies the parallelism.

---

## Part C — CPU-side baseline (context for the ×5.2)

The `optim` branch (shared history, base of `gpu-solver`) already carries the CPU
optimisations the GPU is measured against. Benchmarked on 100 synthetic
pseudowells (`scripts/benchmark_optimizations.py`), cumulative vs original `main`:

| Config | Speedup | Accuracy |
|---|--:|---|
| +symmetric assembly | 1.01× | exact |
| +assemble-once reuse | 1.25× | exact |
| +static condensation | 2.02× | ~1e-5 |
| **force direct solver** | **3.48×** | **7.4e-6 (exact + fastest)** |

**Headline:** forcing the sparse-Cholesky **direct solver** (`direct_solver=True`)
is **3.48× faster than the original and more accurate**; the `ndof < 10000`
auto-threshold is far too conservative for these ~40–150 k-DOF 2-D systems. This
forced-direct 32-worker pipeline (~40 s/sample on len512) is the **CPU baseline**
the GPU path's ×5.2 is measured against. Full detail:
`wiki/findings/optimization-benchmark.md`.

---

## Part D — What moves to `optim`, what stays

| Item | Destination |
|---|---|
| **This document** (`WORK_SUMMARY.md`) | → `optim` |
| **`c8c4959`** (per-run temp-dir isolation) | → `optim` (cherry-picked) |
| GPU solver code (`remo3d/gpu_solver/`, `cuda/`, GPU scripts) | stays on `gpu-solver` |
| GPU wiki findings (`gpu-solver.md`, `cpu-vs-gpu-compute-pipeline.md`) | stays on `gpu-solver` |
| **WIP**: `MPI.FLOAT`→`DOUBLE` fix + worker thread-pinning | stays on `gpu-solver` (uncommitted) — strong candidates to land on `optim` next, but out of scope for this move |

---

## References (on the `gpu-solver` branch)

- `wiki/findings/gpu-solver.md` — full GPU-solver finding (all stages, numbers).
- `wiki/findings/cpu-vs-gpu-compute-pipeline.md` — stage-by-stage anatomy of both
  pipelines with the governing math, FLOP counts, and the compute-bound wall.
- `wiki/findings/optimization-benchmark.md` — realised CPU speedups.
- `wiki/log.md` — chronological record of the work.
