---
title: Performance & Accuracy Tradeoffs
type: concept
tags: [performance, accuracy, tuning, reference, method]
sources: [repo-docs]
updated: 2026-07-15
---

# Performance & Accuracy Tradeoffs

*(traces to `[[repo-docs]]` — performance-guide.md, configuration.md,
solver.md.)* The reference page for the knobs that trade cost against fidelity —
the levers a [sensitivity study](../findings/) would perturb.

## The knobs

| Knob | Faster ⟶ | Cost / accuracy effect | Exact? |
|---|---|---|---|
| **SEC** (`force_single_electrode_configuration`) | on | 1 solve reused across N tools; ×N fewer FE solves | **Exact** (reciprocity + recomputed K) |
| **batch_size** (default 5) | larger | 1 mesh per batch-mean depth; error `(B−1)·dz/2` geometric offset | Approximate |
| **domain_radius** (default 50 m; `"auto"`=`max(10·max_spacing, 5)`) | smaller | truncation error if too small; cost ~r² (2D) / ~r³ (3D) — top 3D lever | Approximate |
| **fe_order** (default 3) | lower (2) | fewer DOFs/time; coarser near-electrode resolution | Approximate |
| **condense** (default on) | on | ~50–70% smaller system, extra local elimination | **Exact** |
| **preconditioner** | `multigrid` | multigrid: fewer iters, scales; local: cheaper setup, more iters | Exact (both converge) |
| **mesh_generator** | `netgen` (2D) | Netgen ~2× faster 2D meshing (no OCC / `.msh` round-trip); Gmsh required for 3D | Exact |
| **cpu/gpu_workers** | more CPU | GPU accelerates only the CG solve; helps large 3D, not small 2D | Exact |

**Guardrails**: master warns if any electrode > `0.75·domain_radius`, aborts if
outside the domain; recommended `domain_radius ≳ 5–10 × largest electrode
spacing`.

## Where wall-time goes

| | Meshing | CG solve | Assembly |
|---|---|---|---|
| **2D** | 35–55% | 20–40% | 10–20% |
| **3D** | 25–45% | 35–55% | — |

Reference (README / AMD Ryzen 2600, 1 tool, 100 points): **2D ~15–30 s**,
**3D ~15–30 min**. Per-worker memory: 2D 20–150 MB, 3D 0.3–4+ GB — memory, not
core count, usually bottlenecks 3D.

## CPU vs GPU decision (by DOF count)

| DOFs | Use |
|---|---|
| < 50k | CPU |
| 50k–200k | benchmark both |
| > few-hundred-k | GPU |

GPU keeps meshing / FE-space / assembly on CPU; only the CG solve moves to
device, so it pays off only once the solve dominates.

## Tuning method

Change **one** parameter at a time and validate against a conservative reference
run (`batch_size=1`, generous `domain_radius`, `multigrid`). Watch for NaNs,
boundary artifacts, and material-response shifts outside tolerance — see
[validation](validation.md). File any quantified result in [findings/](../findings/).

## Links

- Mechanisms: [parallel execution](parallel-execution.md) (SEC/batching), [FEM solver](fem-solver.md), [mesh generation](mesh-generation.md).
- History: [optimization changes](../findings/optimization-changes.md) — which of these knobs/paths were added on the `optim` branch, and why.
- Code: [`../../docs/performance-guide.md`](../../docs/performance-guide.md).
