---
title: Mesh Generation
type: concept
tags: [mesh, fem, gmsh, netgen, refinement, method]
sources: [repo-docs]
updated: 2026-07-15
---

# Mesh Generation

*(traces to `[[repo-docs]]` — mesh-generation.md, walkthroughs.md,
configuration.md; no external literature yet.)*

Each worker builds a **local** mesh per task: the borehole/formation geometry is
clipped to a truncation domain centred on the current simulation depth (depths
shifted so that depth = 0), meshed, then handed to the [FEM solver](fem-solver.md).

## Truncation domain

The unbounded resistivity problem is cut to a finite domain with a far
Dirichlet boundary (`u=0`), see [forward modeling](forward-modeling.md):

| | 2D (`dip==0`) | 3D (`dip!=0`) |
|---|---|---|
| Domain | circle in local `(radius, depth)` plane | half-sphere (`addSphere`, `angle3=π`) |
| Borehole | 2D template | 2D template **revolved through π**, boolean-intersected with sphere |
| Backend | **Netgen** (default) | **Gmsh** (required) |

`active_geometry_radius = domain_radius × active_geometry_window` (window
0.999 Netgen / 0.99 Gmsh) deliberately shrinks the selection slightly to avoid
sliver regions at the outer edge; top/bottom layers are extended by ×1.01.

## Refinement strategy (the key accuracy/cost knob)

Mesh size is a `Min` over a background field plus one field per current
electrode — **finest at electrodes, moderate near the borehole, coarse far
field**:

- Background: 2D `x + 0.1`; 3D `√(x²+y²) + 0.1` (finer toward the axis).
- Per source electrode: `(x² + (y+pos)²)/2 + 0.01` (finest at the electrode).
- Netgen hardcoded: `mesh_size_min=0.001`, `mesh_size_max=10`, moderate density.
- Gmsh algorithms: `6` (2D), `5` (3D) — changes must be benchmark-validated.

Mesh quality feeds back into the solver: too-coarse near the source converges in
linear-algebra terms but to a poor *discrete* solution; slivers make the system
harder to precondition and slow CG. Optimal refinement is problem-dependent.

## Region indexing → conductivity

Regions map to entries of the [`sigma` list](#): region 1 = borehole mud, 2+ =
formation layers top-to-bottom, **one region per plain layer, two per layer with
a filtration (invaded) zone**. Netgen encodes region indices explicitly in its
`SplineGeometry` line records (`[leftdomain, rightdomain]`); Gmsh keeps region
numbering implicit in surface/volume creation order with a separate resistivity
list. The Gmsh `.msh` is always parsed back into a Netgen mesh object via a
modified `ReadGmsh` (physical groups → BC names / material indices).

## 3D construction sequence (Gmsh OCC CSG)

Build 2D borehole template → revolve by π → half-sphere domain → intersect for
borehole volume → cut borehole from sphere for formation → **rotate one box per
layer by the dip angle**, intersect with formation → split invaded layers with an
axis cylinder (inner = invaded, outer = undisturbed) → read mesh back into Netgen.

## Links

- Method hub: [forward modeling](forward-modeling.md) · Solver: [FEM solver](fem-solver.md).
- Tradeoffs: [performance & accuracy](performance-and-accuracy.md).
- Tools: [numerical stack](../entities/numerical-stack.md).
- Code: [`../../docs/mesh-generation.md`](../../docs/mesh-generation.md), [`../../docs/walkthroughs.md`](../../docs/walkthroughs.md).
