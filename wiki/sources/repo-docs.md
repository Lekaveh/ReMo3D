---
title: "Source: ReMo3D code documentation (docs/)"
type: source
tags: [source, internal, documentation]
sources: [repo-docs]
updated: 2026-07-15
---

# Source — `docs/` code documentation

**Slug:** `repo-docs` · **Kind:** internal repository documentation · **Ingested:** 2026-07-15

The 16-page code-documentation set at [`../../docs/`](../../docs/README.md),
written against the repository as it exists on the `optim` branch. It documents
the *implementation*; this wiki distils its *research/method* content and links
back to it. Cite as `[[repo-docs]]` and, where precision matters, link the exact
doc + section.

## What each doc holds (catalog)

| Doc | Research-relevant content |
|---|---|
| [architecture.md](../../docs/architecture.md) | Master/worker split, data flow, backend responsibility (Netgen 2D default, Gmsh 2D+3D), SEC & batch modes |
| [mathematical-foundations.md](../../docs/mathematical-foundations.md) | Governing PDE, 2D axisymmetric vs 3D weak forms, point source, BCs, geometric factors, static condensation, preconditioners |
| [model-api.md](../../docs/model-api.md) | `compute_synthetic_logs` workflow, tool naming, geometric-factor branch, depth shift, dynamic dispatch |
| [mesh-generation.md](../../docs/mesh-generation.md) | Domain truncation (circle / half-sphere), refinement fields, region indexing, Netgen vs Gmsh geometry |
| [solver.md](../../docs/solver.md) | H1 order-3, assembly/solve split, direct vs CG, static condensation, GPU path, CG-convergence caveat |
| [parallel-execution.md](../../docs/parallel-execution.md) | Worker lifecycle, MPI protocol, dynamic pull scheduling, per-task pipeline, NaN error policy |
| [performance-guide.md](../../docs/performance-guide.md) | Every accuracy-vs-cost knob, wall-time split 2D vs 3D, CPU/GPU decision table |
| [configuration.md](../../docs/configuration.md) | Full parameter reference, hidden meshing knobs |
| [data-structures.md](../../docs/data-structures.md) | `tools_parameters`, formation/borehole/`sigma` arrays, nested task list |
| [io-formats.md](../../docs/io-formats.md) | Formation/borehole input formats, output `.txt`/plot, units |
| [testing-and-validation.md](../../docs/testing-and-validation.md) | Benchmarks BM1–BM3, thin-bed model, Task 0 harness & tolerances |
| [developer-guide.md](../../docs/developer-guide.md) | Weak forms in code, how to extend the PDE/BCs/solver |
| [walkthroughs.md](../../docs/walkthroughs.md) | Numerically traced reference example (`B5.7A0.4M`) |
| [examples-and-tutorials.md](../../docs/examples-and-tutorials.md) | Runnable examples, tuning recipes |
| [installation-and-environment.md](../../docs/installation-and-environment.md) | Full numerical stack, GPU/MPI environment |
| [history-and-conventions.md](../../docs/history-and-conventions.md) | Feature evolution (batch/SEC/Model class), coordinate conventions |

## Assessment

Authoritative for *what the code does today*. It is not external literature —
physical/method claims traced only to `[[repo-docs]]` still want a peer-reviewed
citation (tracked as open questions on the concept pages).
