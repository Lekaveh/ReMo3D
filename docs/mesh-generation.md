# Mesh Generation

## 4. Mesh Generation - Gmsh (`gmsh_functions.py`)

## 4.1 `SelectGmshBoreholeDataRange`

Purpose:

- clip the borehole polyline to the active simulation window
- shift the local coordinate system so the simulation point is at depth `0`
- guarantee that the local borehole line reaches the domain boundary

### 2D clipping (`dip == 0`)

The active domain is a circle in the local `(depth, radius)` plane. The helper
`domain_line_intersection` computes where a borehole segment intersects the
circular boundary when one endpoint lies outside the domain.

### 3D clipping (`dip != 0`)

The borehole is clipped by depth extent rather than circular distance because
the later 3D geometry is built inside a half-sphere and the borehole axis is the
main reference line. The top and bottom points are extended or interpolated to
`-domain_radius` and `+domain_radius` in local depth coordinates.

## 4.2 `SelectGmshFormationDataRange`

Purpose:

- shift formation boundaries into local coordinates
- select only the layers that can affect the local solve
- remove filtration zones that do not enter the active geometry window
- slightly extend the top and bottom boundaries beyond the solve domain

The active geometry window is:

```text
active_geometry_radius = domain_radius * active_geometry_window
```

This avoids tiny sliver regions at the outer edge of the domain.

For `dip == 0`, layer relevance is tested by radial distance in the local 2D
cross-section. For `dip != 0`, it is tested by the point-to-plane distance to
the rotated layer interfaces.

If a filtration zone never enters the active window, the function removes it by:

- setting `FZ_RADIUS` and `FZ_VALUE` to `NaN`
- copying `UZ_VALUE` into the remaining resistivity slot

The first layer top and last layer bottom are then extended slightly past the
solve boundary using a factor of `1.01`.

## 4.3 `SelectGmshDataRange`

This is the Gmsh-side convenience wrapper:

1. call `SelectGmshBoreholeDataRange`
2. call `SelectGmshFormationDataRange`
3. convert resistivity to conductivity

Returned tuple:

```text
(local_formation_geometry, local_borehole_geometry, sigma)
```

with:

```text
sigma = [1 / mud_resistivity, 1 / rho_region_1, 1 / rho_region_2, ...]
```

## 4.4 `ReadGmsh`

`ReadGmsh` is a modified copy of Netgen's Python-side Gmsh reader. It:

- parses physical names
- parses nodes
- parses elements
- maps Gmsh physical groups to Netgen boundary-condition names and material
  indices
- creates Netgen `Element1D`, `Element2D`, and `Element3D` entities

Supported element families include first- and second-order variants of:

- segments
- triangles
- quadrilaterals
- tetrahedra
- hexahedra
- prisms
- pyramids

That allows the Gmsh constructors to return `.msh` files while the rest of the
pipeline still works with Netgen/NGSolve mesh objects.

## 4.5 `ConstructGmsh2dModel`

The 2D Gmsh constructor builds an axisymmetric meridional cross-section.

High-level sequence:

1. initialize Gmsh and create a temporary model
2. add a center point used to define circular arcs
3. add points on the borehole axis at domain bottom, electrode depths, and
   domain top
4. add points along the borehole wall polyline
5. create line segments on the borehole axis and on the borehole wall
6. create circular arcs for the top and bottom caps and the outer domain arc
7. create the borehole surface and the surrounding domain surface
8. split the formation into layers by intersecting the domain with rectangles
9. if a layer has a filtration zone, intersect separate rectangles for the
   filtration and undisturbed subregions
10. synchronize, define mesh-size fields, and generate the 2D mesh
11. classify outer circular edges as Dirichlet and everything else as Neumann
12. add physical groups for each conductivity region
13. write the `.msh` file and immediately read it back into Netgen format

### Boundary classification

A line is tagged as `dirichlet_boundary` when all of its nodes satisfy
`R == domain_radius`. All other line entities are grouped into the Neumann set.

## 4.6 `ConstructGmsh3dModel`

The 3D Gmsh constructor builds a half-space model for dipping layers.

High-level sequence:

1. initialize Gmsh and create a temporary model
2. place points on the borehole axis at domain limits and electrode depths
3. place points on the borehole wall polyline
4. create a 2D borehole template and revolve it by `pi` around the borehole
   axis to obtain a half-borehole volume
5. create a half-sphere domain with `addSphere(..., angle3=np.pi)`
6. intersect the revolved borehole with the sphere to get the borehole volume
7. cut the borehole volume out of the half-sphere to get the formation volume
8. create one large rotated box per layer to represent the dipping slab
9. intersect each box with the formation volume
10. if the layer has a filtration zone, intersect again with a cylinder and cut
    the outer part to separate invaded and undisturbed regions
11. synchronize, define mesh-size fields, and generate the 3D mesh
12. classify outer spherical surfaces as Dirichlet and all remaining surfaces as
    Neumann
13. assign volume physical groups and convert the `.msh` file back to Netgen

## 4.7 Mesh Size Control Strategy

Both Gmsh constructors use a background field based on the minimum of multiple
`MathEval` fields.

### Background refinement

- 2D: `x + 0.1`
- 3D: `(x^2 + y^2)^0.5 + 0.1`

This makes elements finer near the borehole axis and progressively coarser away
from it.

### Source refinement

For each current electrode, the code adds:

```text
(x^2 + (y+pos)^2)/2 + 0.01
```

and then uses a `Min` field across the background field and all source fields.

Effect:

- finest mesh close to current electrodes
- moderate refinement near the borehole
- coarser mesh far from the physics that dominate the measurement

## 5. Mesh Generation - Netgen (`netgen_functions.py`)

## 5.1 `SelectNetgenDataRange`

Purpose:

- clip borehole geometry to the local circular solve window
- shift all depths so the simulation depth is `0`
- select only formation intervals that intersect the active window
- remove filtration zones that remain completely outside the active window
- assemble the conductivity distribution in region order

The borehole clipping logic is similar to the 2D Gmsh path, including explicit
intersection with the circular domain boundary when a segment crosses it.

The formation-selection logic also uses an active-geometry window slightly
smaller than the full domain radius to avoid thin wedges at the outer edge.

## 5.2 `ConstructNetgen2dModel`

The Netgen path uses `SplineGeometry` and an explicit point-line-region data
model.

### Point data structure

Every point is stored as:

```text
[index, r, z]
```

Points are created for:

- the borehole axis
- the borehole wall polyline
- filtration-zone boundaries
- layer-boundary endpoints
- the circular domain boundary

### Line data structure

Every line is stored as:

```text
[index, start-point, end-point, boundary-condition, domain-left, domain-right]
```

Lines are then created for:

- vertical segments on the borehole axis
- vertical segments on the borehole wall
- vertical filtration-boundary segments
- horizontal layer-boundary segments
- circular outer-boundary segments

The region numbers in `domain-left` and `domain-right` determine which entry in
`sigma` is assigned to the resulting subdomain.

## 5.3 Domain Boundary Approximation

The outer domain is circular, but `SplineGeometry` is assembled from points and
line segments.

Strategy:

1. collect all existing points already on the circular boundary
2. convert them to polar angle coordinates
3. insert additional boundary points whenever the angular gap exceeds 9 degrees
4. convert those points back to Cartesian `(r, z)` coordinates
5. connect the resulting ordered list with straight segments

This gives a polygonal approximation of the circle that is dense enough for the
intended local solves.

## 5.4 Region Indexing Scheme

Netgen uses explicit integer region numbers.

The indexing pattern is:

- region `1`: borehole mud
- regions `2+`: formation regions in top-to-bottom order
- each layer contributes one region if it has no filtration zone
- each layer contributes two regions if it has a filtration zone

These region indices map directly to the conductivity list returned by
`SelectNetgenDataRange`.

## 5.5 Gmsh vs Netgen

| Aspect | Netgen | Gmsh |
| --- | --- | --- |
| Supported dimensions | 2D only in this repo | 2D and 3D |
| Default selection | default for `dip == 0` | required for `dip != 0` |
| Geometry style | explicit points, lines, and region numbers | OCC constructive solid geometry and boolean operations |
| Output handoff | native Netgen mesh | `.msh` file parsed back into Netgen mesh |
| Main strength | simple and fast 2D axisymmetric meshing | flexible boolean geometry, especially in 3D |

Use Netgen for the default axisymmetric workflow. Use Gmsh when the geometry is
3D or when the Gmsh boolean construction is explicitly desired.
## See Also
- [walkthroughs.md](walkthroughs.md#112-netgen-2d-mesh-construction-walkthrough): worked geometry traces for the same algorithms.
- [configuration.md](configuration.md#99-internal-geometry-window-and-meshing-knobs): hardcoded meshing defaults and developer tuning notes.
- [solver.md](solver.md#68-mesh-quality-and-solver-performance): why mesh choices affect linear-solver behavior.
