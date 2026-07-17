---
title: Overview & Synthesis
type: overview
tags: [synthesis, thesis]
sources: [repo-docs, deep-research-gpu-solver]
updated: 2026-07-17
---

# ReMo3D Research Wiki — Overview

The one page to read first. It states what this wiki is building toward and
links out to the current best pages. The method layer is now populated from the
repository documentation ([[repo-docs]]); it will grow a real *thesis* as
external literature and results are added.

## What ReMo3D is

[ReMo3D](entities/remo3d.md) is a Python package for generating synthetic
**normal and lateral resistivity logs** in 2D axisymmetric and 3D dipping
earth models, by coupling local mesh generation with finite-element solves and
distributed-memory workers. *(seeded from repo — README.md)*

## What this wiki is for

Accumulating the *research-level* knowledge around the project, distinct from
the code documentation in [`../docs/`](../docs/README.md) (which is indexed and
distilled here via [[repo-docs]]):

- **Physics** — normal/lateral logging, apparent resistivity & geometric
  factors, and the borehole/invasion/dip effects that shape a log:
  [resistivity logging](concepts/resistivity-logging.md).
- **Method** — the [forward-modeling](concepts/forward-modeling.md) hub, feeding
  [mesh generation](concepts/mesh-generation.md), the
  [FEM solver](concepts/fem-solver.md), and
  [parallel execution](concepts/parallel-execution.md).
- **Accuracy vs cost** — the knob reference in
  [performance & accuracy](concepts/performance-and-accuracy.md) and how
  correctness is checked in [validation](concepts/validation.md).
- **Findings** — our own work: [optimization changes](findings/optimization-changes.md)
  made on the `optim` branch, plus sensitivity / benchmark results as they land.
- **Literature** — summaries of external papers/reports in `sources/`
  (first external source: [[deep-research-gpu-solver]], an LLM deep-research
  report — peer-reviewed literature still wanted).

## Key facts worth surfacing

- **SEC is exact, batching is approximate.** SEC reuses one solve across N tools
  via reciprocity with a recomputed geometric factor; batching evaluates on a
  shared mesh with a bounded geometric offset `(B−1)·dz/2`. See
  [parallel execution](concepts/parallel-execution.md).
- **The CG solve has no convergence guard in production** (`maxsteps=1000`, no
  residual check) — an under-converged iterate is written silently. Offline
  [Task 0](concepts/validation.md) is the guard. See [FEM solver](concepts/fem-solver.md).
- **`domain_radius` is the dominant 3D cost lever** (~r³), and truncation error
  grows if it is too small — the central accuracy/cost tension.

## Open threads

- The `optim` branch's solver optimizations are now **benchmarked** (100 samples,
  original vs each): [optimization benchmark](findings/optimization-benchmark.md).
  Headline — **forcing the direct solver is 3.48× faster than the original *and*
  more accurate**; its `ndof < 10000` auto-threshold is far too conservative for
  2D (production ~40k DOF). Best single lever found.
- Sensitivity analysis is active on the `optim` branch
  (`remo3d/sensitivity.py`, `Sensitivity.ipynb`) — no [findings/](findings/) page
  captures it yet.
- **GPU forward solver v2 is the active program.** v1 (structured-grid JAX
  block-Thomas) reached **×5.2** and hit a small-block factorization wall
  ([`../WORK_SUMMARY.md`](../WORK_SUMMARY.md)). The ingested
  [[deep-research-gpu-solver]] report argued the wall is architectural, not
  hardware — and the Phase 0 experiments **confirmed the global
  factor-once/solve-many path: GO**
  ([findings/gpu-solver-v2.md](findings/gpu-solver-v2.md)). Next: the JAX
  scan port + boundary study per
  [`../GPU_SOLVER_V2_PLAN.md`](../GPU_SOLVER_V2_PLAN.md).
- Only one **external** source ingested so far ([[deep-research-gpu-solver]],
  LLM-generated) — physics/method pages still cite mostly [[repo-docs]] and want
  peer-reviewed references.

## Current thesis

_Too early for a literature-grounded thesis. The method layer is complete and
consistent with the codebase; the next step is external sources and our own
sensitivity results._
