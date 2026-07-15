---
title: Forward Modeling Pipeline
type: concept
tags: [fem, ngsolve, mesh, forward-problem, method]
sources: [repo-docs]
updated: 2026-07-15
---

# Forward Modeling Pipeline

*(traces to `[[repo-docs]]` — mathematical-foundations.md, architecture.md; not
literature.)* This is the **method hub**: geometry → mesh → solve → log.

The **forward problem**: given a resistivity model (geometry + $\rho$ per
region) and a tool definition, compute the synthetic apparent-resistivity log.
ReMo3D solves the stationary current-flow PDE

$$-\nabla\cdot(\sigma\nabla u) = \sum_k I_k\,\delta(\mathbf{x}-\mathbf{x}_k),\qquad \sigma = 1/\rho$$

with $u=0$ on the far (Dirichlet) outer boundary and homogeneous Neumann on
symmetry/remaining boundaries. Point sources are the current electrodes,
injected via local basis-function evaluation.

## Pipeline

```mermaid
flowchart LR
  M[Model: geometry + rho] --> G[Mesh generation]
  T[Tool + electrode positions] --> G
  G --> S[FEM assembly + CG solve]
  S --> P[Evaluate potential at M/N]
  P --> R[rho_a = K * |dV|  -> synthetic log]
```

## Key numerical choices (accuracy vs. cost)

| Choice | 2D axisymmetric (`dip==0`) | 3D (`dip!=0`) |
|---|---|---|
| Domain | circular cross-section | half-sphere (symmetry plane) |
| Weak form | $\int 2\pi r\,\sigma\,\nabla u\cdot\nabla v\,dr\,dz$ | $\int \sigma\,\nabla u\cdot\nabla v\,dV$ |
| Cost | ~15–30 s / 100 points | ~15–30 min / 100 points |
| Note | — | worker divides response by 2 (half-space) |

Other knobs that trade accuracy for speed:

- **`domain_radius`** — larger = boundary farther, better truncation of the
  unbounded problem, higher cost. Master warns beyond `0.75*radius`.
- **Static condensation** (`condense=True`) — eliminate element-internal DOFs
  (Schur complement) before the CG solve.
- **Preconditioner** — `"multigrid"` (default) vs `"local"`.
- **Mesh refinement** near electrodes and material interfaces.

All of these are the levers a **sensitivity analysis** would perturb — see the
[performance & accuracy](performance-and-accuracy.md) reference and
[findings/](../findings/) once studies are recorded.

## Stages in detail

- **Mesh** — local domain truncation + refinement: [mesh generation](mesh-generation.md).
- **Solve** — H1 order-3 FEM, direct/CG, condensation, GPU: [FEM solver](fem-solver.md).
- **Distribute** — MPI master/worker, SEC & batching: [parallel execution](parallel-execution.md).
- **Post-process** — geometric factor → apparent resistivity: [resistivity logging](resistivity-logging.md#apparent-resistivity--geometric-factors).

## Links

- Physics: [resistivity logging](resistivity-logging.md). Package: [ReMo3D](../entities/remo3d.md).
- Tools: [numerical stack](../entities/numerical-stack.md). Validation: [benchmarks](validation.md).
- Code/theory: [`../../docs/mathematical-foundations.md`](../../docs/mathematical-foundations.md),
  [`../../docs/solver.md`](../../docs/solver.md),
  [`../../docs/mesh-generation.md`](../../docs/mesh-generation.md),
  [`../../docs/performance-guide.md`](../../docs/performance-guide.md).
