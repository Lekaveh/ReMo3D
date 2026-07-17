# Wiki Log

Append-only, chronological. Newest at the bottom. Entry kinds: `scaffold`,
`ingest`, `query`, `lint`. Grep with `grep '^## \[' log.md | tail`.

## [2026-07-15] scaffold | ReMo3D research wiki created
- Created `wiki/` with schema (`CLAUDE.md`), `index.md`, `log.md`, `overview.md`.
- Dirs: `raw/` (+`assets/`), `sources/`, `concepts/`, `entities/`, `findings/`.
- Seeded from repo (README + docs/): `concepts/resistivity-logging.md`,
  `concepts/forward-modeling.md`, `entities/remo3d.md`.
- No external sources ingested yet.

## [2026-07-15] ingest | repo-docs (docs/ code documentation, 16 pages)
- Read all 16 `docs/*.md` (via 3 parallel extraction agents) for research/method content.
- New source page: `sources/repo-docs.md` (catalog + `[[repo-docs]]` slug).
- New concept pages: `mesh-generation.md`, `fem-solver.md`, `parallel-execution.md`,
  `performance-and-accuracy.md`, `validation.md`.
- Expanded concepts: `resistivity-logging.md` (tool naming, geometric factors, effects),
  `forward-modeling.md` (stage links, now the method hub).
- New entity: `entities/numerical-stack.md` (Gmsh/Netgen/NGSolve/MPI roles). Expanded `remo3d.md`.
- Updated `index.md`, `overview.md` (key-facts + thesis). All method pages cite `[[repo-docs]]`,
  linking out to `docs/` rather than duplicating.
- pages touched: 12. Approach: distil not mirror — wiki holds the *why/what*, docs hold the *how*.

## [2026-07-15] ingest | git history — optim-branch optimization changes
- Reviewed branch `optim` (d099f0b..63f7694) + performance_optimization_tasks.md; diffed solver/worker/remo3d source.
- New finding page: findings/optimization-changes.md — what changed & why, verified against actual code.
- Verified landed: Tasks 0,1,2,4,5,8. Verified NOT landed: Tasks 3,6,7,9 (evidence: no `tol=`, netgen_functions untouched, no 2D-GPU warning).
- Cross-linked from index/overview and concepts/fem-solver + performance-and-accuracy.
- pages touched: 6. Open thread noted: realized speedups not yet benchmarked.

## [2026-07-15] query | DOI table per tool (90/95/99%)
- Computed radial + vertical (up/down) DOI for 12 tools from the analytical Born kernel (homogeneous bg).
- Robust routine: fixed cell size scaled to tool, grow domain until 99% converges, read all fractions off one cumulative curve (monotonic).
- Validated by reciprocity (A2.0M0.5N≡N0.5M2.0A, A4.0M0.5N≡M4.0A0.5B) and r90≈AM for normals.
- Wrote findings/doi-table.md + machine-readable notebooks/doi_sensitivity_table.csv. Indexed.

## [2026-07-15] fix | _measure_doi grid-resolution bug (DOI mis-ordering)
- plot_sensitivity_doi's _measure_doi fixed the point count while growing the domain -> coarse grid on high-fraction tails -> A2.0 reported larger 92% DOI than A4.0.
- Fix: fixed cell size scaled to tool + converge on requested fraction (same logic as the DOI table routine). r_eff now strictly increases with tool length at all fractions; ~1s/call.
- Touched: remo3d/sensitivity.py::_measure_doi; note updated in findings/doi-table.md.

## [2026-07-15] update | DOI optimization to-do: drop Task 3 for 2D
- findings/optimization-changes.md: removed Task 3 (explicit CG tol) from the 2D to-do list — moot for 2D since the direct solver (Task 5) replaced CG there; kept 6/7 (mesh gen, the 2D bottleneck) and 9 (guardrail).

## [2026-07-16] finding | Optimization benchmark — realized speedups + direct-solver win
- New page findings/optimization-benchmark.md: 100-sample before/after (original V1 vs each optimization), method descriptions, code changes, flag table.
- Headline: forcing the direct solver (direct_solver=True) = 3.48x AND exact; the ndof<10000 auto-threshold is far too conservative for 2D (~40k DOF). Marked as the best solver + how to invoke.
- Also benchmarked: #2 p-adaptivity 3.65x (needs direct solver), #3 coarse-far mesh 2.28x, #4 per-tool domain net loss.
- Updated: findings/optimization-changes.md (Tasks 6/7/9 now done; benchmark-gap closed), concepts/fem-solver.md (direct-vs-CG finding + how to force), overview.md (open thread resolved), index.md.

## [2026-07-17] query | Axis study (SEC/batch/condense × solver) + thread-pinning root cause
- Ran the axis benchmark (100 samples): CG base 20.72s; pinned direct b5 9.60s
  (4.30×, exact), b10 8.45s (4.88×, 10% tail), b15 5.85s (7.05×, 11% tail —
  error saturates past b10, b15 dominates b10).
- Root-caused the HPC ">1k workers" blow-up: NGSolve TaskManager per-core
  threads in every worker under `sparsecholesky` (ignores OMP/MKL env);
  verified in vivo (571 running / 4407 total → 24 / 72 after pin) and fixed in
  `workers/worker.py` (env caps + `ngs.SetNumThreads(1)`); pinning itself is
  ~25% faster end-to-end.
- pages touched: findings/optimization-benchmark.md

## [2026-07-17] ingest | Deep research: GPU-first axisymmetric solver (ChatGPT)
- User-commissioned LLM deep-research report on breaking the GPU solver's ×5.2
  ceiling. Raw md export had corrupted citations (`citeturn…`); the reference
  list (24 footnotes) was restored from the user's PDF export into the source page.
- Raw renamed: `raw/deep-research-report.md` →
  `raw/2026-07-17-deep-research-gpu-first-axisymmetric-solver.md`.
- Key take: the ×5.2 wall is architectural, not hardware — bet #1 is a global
  operator + factor-once/solve-many (cuDSS, reciprocity RHS cut); local
  persistent-kernel path is the fallback. Decision matrix of 3 experiments.
- pages touched: sources/deep-research-gpu-solver.md (new),
  concepts/fem-solver.md, concepts/parallel-execution.md, overview.md, index.md.
- follow-up artifact: `../GPU_SOLVER_V2_PLAN.md` (expanded work plan, repo root).

## [2026-07-17] finding | GPU solver v2 Phase 0 — global path GO
- Ran plan experiments E0–E3 (global_op.py + 3 scripts): sizing 1.37M DOF /
  1.22 GB factor (shared grid 112×12315); 1280 tasks -> 531 unique RHS (×2.41);
  CPU block-Thomas control 75 s/sample (build 2.9 / factor 6.2 / solve 66).
- Accuracy detective story: v2-vs-v1 spikes up to 5.5% were NOT discretization
  (h-independent; assembler ≡ apply_A to 1e-12; parity 0.03–0.09% at const RM).
  Root causes: (1) production scalar-mud-per-solve convention vs v2's physical
  z-varying RM column — len512 RM log is noisy ±8%, moves short normals ~5%;
  (2) shared boundary truncation in v1+NGSolve (grows to 3.6% at R=5 points).
  Fresh NGSolve confirmed both directions.
- scipy ?pbtrf is reference-unblocked (~180 s @ n=2e5) — banded path dead;
  64-thread OpenBLAS on 111×111 blocks >10× slower (pin to 1, again).
- pages touched: findings/gpu-solver-v2.md (new), index.md, overview.md;
  repo: remo3d/gpu_solver/global_op.py, scripts/gpu_v2_*.py, v1 core ported.

## [2026-07-17] finding | GPU solver v2 G1 — gate PASSED, x5.5 over v1
- global_gpu.py: whole pipeline (sigma zmud -> assembly -> factor lax.scan ->
  2 solve scans, 531 RHS at once) in one jit, vmap over samples.
- Precision: fp64 exact (4.9e-11 vs CPU control); pure fp32 recursion NaNs at
  row ~2153/12313 (SPD loss; Jacobi scaling insufficient; v1's 341-row windows
  were below the cliff); MIXED (fp64 recursion -> fp32 factors+solves) gives
  Ra err 4e-5.
- Throughput (A6000, warm): B=1 9.87 s/sample; B=4 2.58 (gate <3.06 PASS);
  B=8 1.40 -> x5.5 vs v1 GPU, x28.6 vs CPU pipeline. Latency-bound scans ->
  batch almost free; B memory-bound (~3 GB/sample fp32 forward stack).
- pages touched: findings/gpu-solver-v2.md (G1 section + verdict), index.md,
  overview.md; repo: remo3d/gpu_solver/global_gpu.py,
  scripts/gpu_v2_global_gpu_bench.py, global_op.py refactor (build_global_tasks).

## [2026-07-17] query | v2 GPU on optim_bench — 0.64 s/sample, x65/x19/x15
- Ran the global mixed-precision solver on the 100-sample optim_bench workload
  (5 tools x 128 depths): warm 0.64 s/sample (B=10), grid 112x7195, 798k DOF,
  275 unique RHS (x2.33). vs V1 CG 41.25s = x65; Vd forced-direct 11.84s = x19;
  axis pinned b5 9.60s = x15.
- Accuracy vs stored pipeline logs: mean 1.2-1.7%, max 10-20%. Arbitrated worst
  points (fresh NGSolve, const-RM): matched-convention agreement 0.3-1.1%.
  Decomposition: mud convention (noisy RM +-4%) + the stored references' OWN
  batch_size=5 error (up to 15% at A8.0 s24 z=25.4 vs fresh unbatched NGSolve)
  + small boundary residual at the well edge for A8.0.
- pages touched: findings/gpu-solver-v2.md (optim_bench section);
  repo: scripts/gpu_v2_optim_bench.py.

## [2026-07-17] finding | E2 closed (boundary) + conventions + 2-GPU/energy
- Boundary sweep (gpu_v2_boundary_sweep.py): truncation ~1/R^2; v1 convention
  10*span (R=90) = 6.2% worst (A8.0 edges); adopted default max(80*span,45)
  -> 0.065% for +14% DOF. global_op default changed accordingly.
- Adopted production conventions: z-varying mud column; large-R boundary.
  Stored optim_bench refs (R=90 + batch=5) now diverge more at A8.0 edges
  (27% max) BECAUSE v2 improved -> regression baselines need recompute.
- 2x A6000: 100 samples in 54.4s wall (0.74 s/sample/GPU warm at R=720;
  combined 0.37 s/sample = x32 vs Vd); energy ~140 J/sample (257 W mean).
- pages touched: findings/gpu-solver-v2.md (E2/G2 sections, verdict), plan.

## [2026-07-17] finding | G2: Ex1 ladder passed + driver API integrated
- Full Ex1 (8 tools incl. laterals x 251 depths) on ONE global grid (906k DOF,
  2008 tasks -> 818 RHS, x2.45): overall max 1.00% vs frozen Results_1.txt,
  per-tool means 0.08-0.25% — matches v1's validated envelope; laterals fine;
  matched-R vs default-R < 0.005% apart (Ex1 boundary-insensitive).
- Driver: compute_logs_gpu(..., global_solver=True, precision="mixed"|"f64")
  returns the standard logs contract; verified vs bench path (6.7e-5).
- v2-fp64 regression-baseline recompute (100 optim samples) launched.
- pages touched: findings/gpu-solver-v2.md (G2 section, verdict).

## [2026-07-17] finding | G2: fp64 regression baseline recomputed
- benchmark_data/gpu_solver/global_optim_bench_f64.npz: v2-fp64 logs for all
  100 optim_bench samples (adopted conventions), 3.41 s/sample B=4 (x12 vs CG
  pipeline even in fp64). Mixed path agrees to worst 1.9e-4 / mean 7.7e-6 over
  50 samples -> this file replaces the stored pipeline logs as the regression
  gate (those carry R=90 truncation + batch=5 sharing).
- pages touched: findings/gpu-solver-v2.md (baseline section, verdict).

## [2026-07-17] finding | CPU-compat mode: scalar mud imitated, z-window can't be
- convention="cpu": mud-split operator A(nu)=A_ref+dnu*A1+dnu^2*A2 + 3-term
  Neumann series on ONE factorization; exact to 5.2e-5 vs per-column
  scalar-mud factorizations. Two fp32 traps fixed: residual cancellation at
  the source (series form) and stencil-sum cancellation in dA*x (fp64 apply).
- vs stored refs: means 0.34-1.4% (native 1.2-1.9%). vs fresh unbatched
  NGSolve: sample 0 <=1.0%; sample 56 (salt mud RM~0.15) short tools up to
  40% — NGSolve's OWN window truncation: R-sweep converges to the compat
  value (0.763@R5 -> 0.456@R80 vs 0.457). 31/100 samples are conductive.
- The per-depth z-window is the one CPU convention factor-once cannot (and
  should not) imitate. Cost: 6.90 s/sample B=4 (chunked lax.map);
  native 0.64 s. ngsolve_protocol.npz saved as the clean reference subset.
- pages touched: findings/gpu-solver-v2.md (compat section); driver
  convention arg; scripts gpu_v2_ngsolve_protocol/compat_report.

## [2026-07-17] fix | Reference conventions corrected (user-caught): fixed R=40
- User flagged the batching attribution; verification showed the stored
  full_pipeline refs ran with FIXED domain_radius=40 (harness flag; production
  simulate_logs default is fixed 50) — NOT per-tool max(10*span,5). The "15%
  reference batching error" and the "40% s56" protocol numbers were OUR
  R-convention mismatches; fresh unbatched NG@R40 sits 0.3% from the stored
  A8.0 point, and the stored conductive-channel point was near-converged.
- Compat mode switched to fixed R (bench 40 / driver default 50). Rerun of all
  100 samples: overall mean 0.449% vs stored refs; residual maxima 4.4-9.4%
  are the stored files' OWN batch_size=5 error (compat matches unbatched
  NG@R40 to 0.19-0.32% at the three worst points; detached logs record CLI
  b=5 — a batch-1 baseline run, per the user, would be the cleaner target).
- pages touched: findings/gpu-solver-v2.md (cross-check item 2/3 rewritten,
  compat section updated); driver/bench/global_gpu R handling.
