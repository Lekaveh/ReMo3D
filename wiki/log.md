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
