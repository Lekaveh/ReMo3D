---
title: Validation & Benchmarks
type: concept
tags: [validation, benchmarks, convergence, methodology]
sources: [repo-docs]
updated: 2026-07-15
---

# Validation & Benchmarks

*(traces to `[[repo-docs]]` — testing-and-validation.md, walkthroughs.md.)* How
correctness and accuracy are checked, and the reference cases to anchor new work
(including [sensitivity](../findings/) and [performance](performance-and-accuracy.md) studies).

## Task 0 harness (`scripts/benchmark_task0.py`)

Bypasses MPI and calls Netgen/NGSolve directly, so **solver exceptions surface
instead of becoming NaN** (unlike the [worker path](parallel-execution.md#fault-tolerance)).
Two reference cases:

- `homogeneous_10ohmm` — compared against the **analytical** homogeneous value.
- `two_layer_10_100ohmm` — non-condensed solve as numerical baseline.

**Pass condition:** `condense=True` matches `condense=False` within relative
tolerance **`1e-3`** (default); any NaN/Inf = solver failure. Records
mesh-gen/assembly/solve/eval timings, total & free DOF counts, and CG
iters/residuals when exposed. This is the offline guard against the
[silent CG-convergence caveat](fem-solver.md).

## Benchmark models

| Name | Setup | Validates |
|---|---|---|
| **BM1** | 9 layers 0–60 m, alternating 10/100 Ω·m, 200 mm hole, 1 Ω·m mud, no invasion | clean layer-boundary response; thin resistive beds |
| **BM2** | 7 layers, invaded intervals (radii 0.2/0.35/0.5 m, 5 Ω·m filtrate) | filtration-zone region numbering & conductivity ordering |
| **BM3** | 3-layer (10/100/10 Ω·m) at dip 0/15/30/45/60° | 2D→3D transition, Gmsh 3D weak form (boundaries widen with dip) |
| **Thin-bed** | perturbed boundaries near 0.25 m step; 4 log sets | isolates boundary effect vs depth-shift error (Logs 1–4) |

The traced reference example (`B5.7A0.4M` at 10.0 m → `ρₐ = 7.4759 Ω·m`) from
[walkthroughs](../../docs/walkthroughs.md) is a good single-point regression anchor.

## Recommended new tests

Homogeneous medium (vs analytic); single sharp boundary; single cylindrical
invasion; **`domain_radius` sweep** (boundary convergence); **`batch_size=1` vs
larger** (quantify batching error). These double as sensitivity experiments —
file results in [findings/](../findings/).

## Known limitations

Benchmarks are regression/qualitative references, **not analytical truth
tables**. Worker `try/except` → NaN can mask the first failure; extreme
resistivity contrasts stress meshing/convergence; dip = 90° is rejected.

## Links

- [performance & accuracy](performance-and-accuracy.md) · [FEM solver](fem-solver.md) · [findings/](../findings/).
- Code: [`../../docs/testing-and-validation.md`](../../docs/testing-and-validation.md).
