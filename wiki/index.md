# Wiki Index

Content catalog for the ReMo3D research wiki. Read [`overview.md`](overview.md)
first for the synthesis, then drill into pages below. Updated on every ingest.

## Overview
- [overview.md](overview.md) — the evolving synthesis / thesis. Start here.

## Concepts
- [resistivity-logging.md](concepts/resistivity-logging.md) — normal & lateral logs, tool naming, apparent resistivity & geometric factors, log-shaping effects.
- [forward-modeling.md](concepts/forward-modeling.md) — **method hub**: governing PDE, weak forms, geometry→mesh→solve→log pipeline.
- [mesh-generation.md](concepts/mesh-generation.md) — domain truncation, refinement fields, region indexing, Netgen vs Gmsh, 3D CSG.
- [fem-solver.md](concepts/fem-solver.md) — H1 order-3, assemble/solve split, direct vs CG, preconditioners, static condensation, GPU, CG caveat.
- [parallel-execution.md](concepts/parallel-execution.md) — MPI master/worker, dynamic dispatch, SEC (exact) & batching (approximate).
- [performance-and-accuracy.md](concepts/performance-and-accuracy.md) — **reference table** of every cost/fidelity knob; wall-time split; CPU/GPU decision.
- [validation.md](concepts/validation.md) — Task 0 harness, benchmarks BM1–BM3, thin-bed model, recommended tests.

## Entities
- [remo3d.md](entities/remo3d.md) — the package: purpose, stack, provenance.
- [numerical-stack.md](entities/numerical-stack.md) — Gmsh, Netgen, NGSolve, MPI and each one's role.

## Findings
- [optimization-changes.md](findings/optimization-changes.md) — what changed on the `optim` branch to speed up the 2D solve path, and why (Tasks 0–9; all landed as of 2026-07-16).
- [optimization-benchmark.md](findings/optimization-benchmark.md) — **realized** speedups (100 samples, original vs each optimization); forcing the direct solver = 3.48× and exact; four explored ideas + how to invoke the direct solver.
- [doi-table.md](findings/doi-table.md) — per-tool radial & vertical depth-of-investigation at 90/95/99% (12 tools; homogeneous background).
- [gpu-solver-v2.md](findings/gpu-solver-v2.md) — **Phase 0 of the global factor-once/solve-many path: GO.** Sizing trivial (1.37M DOF, 1.2 GB factor), ×2.41 RHS dedup, CPU control 75 s/sample, discretization parity with v1 (0.03–0.09%); the 3–5% "errors" were the mud-column + boundary-truncation conventions.

## Sources
- [repo-docs.md](sources/repo-docs.md) — `[[repo-docs]]` the 16-page `docs/` code documentation set (ingested 2026-07-15).
- [deep-research-gpu-solver.md](sources/deep-research-gpu-solver.md) — `[[deep-research-gpu-solver]]` ChatGPT deep-research report on breaking the GPU solver's ×5.2 ceiling; bet #1 = global factorization + many RHS; citations restored from the PDF (ingested 2026-07-17).
