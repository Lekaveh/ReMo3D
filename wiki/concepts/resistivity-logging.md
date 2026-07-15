---
title: Resistivity Logging (Normal & Lateral)
type: concept
tags: [physics, well-logging, resistivity, apparent-resistivity]
sources: [repo-docs]
updated: 2026-07-15
---

# Resistivity Logging

*(physics traces to `[[repo-docs]]` — mathematical-foundations.md,
model-api.md; still wants a peer-reviewed citation, see open questions.)*

Resistivity logging measures formation resistivity around a borehole by injecting
current through **current electrodes** (`A`, `B`) and measuring potential at
**measurement electrodes** (`M`, `N`). ReMo3D simulates the two classic galvanic
tool families — **normal** and **lateral** logs.

## Tool naming

A tool is 3 electrode symbols from `{A,B,M,N}` plus 2 positive spacings, written
top-to-bottom (e.g. `B5.7A0.4M`). The measurement point sits halfway within the
closer adjacent electrode pair (equal spacings are rejected). `A`/`B` are current
electrodes, `M`/`N` measurement electrodes; the missing symbol selects the
[geometric-factor](#geometric-factors) branch. See
[`../../docs/model-api.md`](../../docs/model-api.md#33).

## Apparent resistivity & geometric factors

The reported quantity is **apparent resistivity** $\rho_a$, not true resistivity.
Each solve evaluates the potential at the measurement electrodes and multiplies
by a spacing-only **geometric factor** $K$:

$$\rho_a = K\,|u(M) - u(N)| \quad\text{or}\quad \rho_a = K\,|u(M)|$$

$K$ derives from the homogeneous point-source potential $V(r)=\rho/(4\pi r)$ and
branches on the absent electrode:

| Missing | $K$ |
|---|---|
| A | $4\pi\,BM\,BN/(BN-BM)$ |
| B | $4\pi\,AM\,AN/(AN-AM)$ |
| M | $4\pi\,AN\,BN/(AN-BN)$ |
| N | $4\pi\,AM\,BM/(BM-AM)$ |

The 3D worker divides the response by 2 (half-space mesh vs full-space $K$). Full
derivation: [`../../docs/mathematical-foundations.md`](../../docs/mathematical-foundations.md#26-geometric-factors).
The `ABMN→MNAB` reciprocity rewrite that enables
[SEC](parallel-execution.md#single-electrode-computation-sec) recomputes $K$
exactly.

## Formation model

Layered rock with, per layer, an optional **filtration (invaded) zone**
(`FZ_RADIUS`, `FZ_VALUE`) around an **undisturbed zone** (`UZ_VALUE`); the
borehole holds mud of resistivity `RM`. The solver works in conductivity
$\sigma = 1/\rho$.

## Effects that shape a log

- **Borehole effect** — conductive/resistive mud column near the tool.
- **Invasion** — drilling fluid displacing formation fluid radially; motivates
  the 2D **axisymmetric** model and the filtration zone.
- **Bed-thickness / shoulder-bed effect** — thin beds blur into neighbors.
- **Dip** — non-perpendicular beds break symmetry, forcing a full 3D model
  (`dip != 0`); bed responses widen with dip (see [BM3](validation.md)).
- **Depth of investigation** — longer electrode spacings read deeper; large
  tool-to-tool differences at one depth indicate contrasting DOI.

## Links

- Method: [forward modeling](forward-modeling.md), [FEM solver](fem-solver.md).
- Package: [ReMo3D](../entities/remo3d.md). Validation: [benchmarks](validation.md).
- Code/theory: [`../../docs/mathematical-foundations.md`](../../docs/mathematical-foundations.md), [`../../docs/model-api.md`](../../docs/model-api.md).

## Open questions / to source

- Peer-reviewed reference for the specific normal/lateral tool geometries.
- Standard invasion-model parameterization in the literature (vs the FZ/UZ scheme here).
