# Code Walkthroughs - Key Algorithms

These walkthroughs complement the structural descriptions in
[`mesh-generation.md`](mesh-generation.md), the orchestration details in
[`model-api.md`](model-api.md), and the array-layout reference in
[`data-structures.md`](data-structures.md).

## 11.1 Complete 2D Simulation Walkthrough (Example_01)

Example_01 uses:

- 6 tools
- measurement depths from `0` to `25` m with a `0.1` m step, so
  `measurement_depths.shape == (251,)`
- formation array with 7 layers
- borehole array with 251 samples
- `dip=0`, so the axisymmetric path is selected

### High-level execution path

1. `Model.compute_synthetic_logs` creates the `Model` instance.
2. Tool strings are parsed into `self.tools` entries.
3. Because the default `force_single_electrode_configuration=True` rewrites
   compatible dual-current tools, the tool suite can enter SEC mode.
4. Formation and borehole files are loaded, unit rows are read, and geometry is
   converted to meters.
5. Workers are spawned.
6. `simulate_logs` chooses `netgen` because `dip == 0` and `mesh_generator` is
   `auto` by default.
7. Measurement depths are shifted by each tool's `depth_shift` to obtain the
   simulation depths.
8. Simulation depths are batched and converted into the nested task list.
9. The master broadcasts the model and tasks.
10. Each worker clips a local circular window around the requested simulation
    depth.
11. The worker constructs a 2D Netgen mesh, solves the weighted axisymmetric
    problem, evaluates potentials at the measuring electrodes, and multiplies by
    the geometric factor.
12. The master gathers all results and stores one `(251, 2)` log array per tool.
13. `save_results` writes `Results_1.txt` and `Results_plot.png`.

### Concrete tool-parameter example

For the Example_01 tool `B5.7A0.4M`, the default single-electrode rewrite turns
it into the equivalent single-current form `N5.7M0.4A` before the geometry is
packed.

The resulting `tool_parameters` array is:

```text
[[-6.1    -0.4     0.0     5.3793]
 [ 0.0     0.0     1.0     0.2   ]]
```

Interpretation:

- measuring electrodes at local depths `-6.1 m` and `-0.4 m`
- current electrode at local depth `0 m`
- geometric factor `K = 5.3793`
- depth shift `+0.2 m`, so `simulation_depth = measurement_depth + 0.2`

### Small Example_01 subset for array shapes

Take only the first three measurement depths and the first two tools:

- measurement depths: `[0.0, 0.1, 0.2]`
- tool 0 simulation depths: `[0.2, 0.3, 0.4]`
- tool 1 simulation depths: `[0.81, 0.91, 1.01]`

With `batch_size=2`, the unique simulation depths reshape into:

```text
simulation_depths =
[[0.20, 0.30],
 [0.40, 0.81],
 [0.91, 1.01]]
```

so, for this didactic subset:

- `simulation_depths.shape == (3, 2)`
- `len(task_list) == 3`

### Trace of one worker cycle with actual numbers

Take the saved Example_01 output at measurement depth `10.0 m` for
`B5.7A0.4M`.

1. The tool depth shift gives `simulation_depth = 10.2 m`.
2. The borehole file gives `CALM = 243.9279 mm` at `10.2 m`, so the local
   borehole radius is `0.12196 m`.
3. The borehole file gives mud resistivity `RM = 1.1204 ohm-m` at `10.2 m`.
4. The worker solves for one source at local depth `0.0 m`.
5. It evaluates the solution at `(r, z) = (0.0, -6.1)` and `(0.0, -0.4)`.
6. It computes:

```text
rho_a = 5.3793 * |u(0, -0.4) - u(0, -6.1)|
```

7. The stored Example_01 output gives `rho_a = 7.4759 ohm-m` at `10.0 m`.
8. That implies an evaluated potential drop of approximately:

```text
|u(0, -0.4) - u(0, -6.1)| = 7.4759 / 5.3793 = 1.3898
```

The exact potential values depend on the mesh and solve, but the trace shows how
one local solve maps numerically into one reported log sample.

## 11.2 Netgen 2D Mesh Construction Walkthrough

The Netgen constructor is the densest geometry-building routine in the codebase.
It assembles a full 2D region map from point lists and line lists rather than
from boolean solids.

### Spatial layout of the point families

For a two-layer example with one invaded lower layer, the point families are
laid out conceptually like this:

```text
                         increasing radial distance r

axis r=0        borehole wall r=rb(z)     filtration wall r=rfz     domain boundary r=R

A0 o            B0 o                                               D0 o
   |               |                                                   \
A1 o------------B1 o-----------------------------F1 o-------------------D1 o
   |               |                             |                       |
A2 o------------B2 o-----------------------------F2 o-------------------D2 o
   |               |                                                     /
A3 o            B3 o                                               D3 o

z = z_top       z = layer top / bottom intersections               z = domain-circle intersections
```

Point meaning:

- `A*`: points on the borehole axis
- `B*`: points on the borehole wall, including intersections with layer
  boundaries
- `F*`: points on the filtration boundary for invaded layers only
- `D*`: points on the outer circular boundary, including interpolated points

### How the line families are assembled

The constructor then builds five line families.

1. Axis lines: `A0-A1`, `A1-A2`, `A2-A3`
2. Borehole-wall lines: `B0-B1`, `B1-B2`, `B2-B3`
3. Filtration-boundary lines: `F1-F2`
4. Layer-boundary lines: `A1-B1-F1-D1` and `A2-B2-F2-D2`, split into as many
   segments as the boundary requires
5. Domain-boundary polygon segments connecting `D*`

### Region-index assignment

Netgen stores domains explicitly on each oriented line as
`[leftdomain, rightdomain]`.

For the simple two-layer example above, the regions are typically:

- region `1`: borehole mud
- region `2`: upper clean formation layer
- region `3`: lower filtration zone
- region `4`: lower undisturbed zone

The important idea is that every line segment encodes which region is on its
left and right. That is why the bookkeeping is so detailed.

### Worked trace of the bookkeeping

A simplified trace of the constructor is:

1. create axis points from `[-domain_radius, tool_geometry..., +domain_radius]`
2. interpolate borehole-wall points at every layer boundary depth
3. add filtration points only for rows where `FZ_RADIUS` is not `NaN`
4. add layer-end points on the outer circle
5. interpolate extra circle points every 9 degrees where the angular gap is too
   large
6. add vertical line lists first, because they define the borehole and invasion
   partitions
7. add horizontal layer-boundary segments next, because they split vertical
   stacks into distinct regions
8. add outer-boundary segments last, because they close the domain polygon
9. attach the smallest mesh size to the source-point indices
10. pass the complete point and line lists into `SplineGeometry`

### Why this algorithm is complex

This route is harder to read than the Gmsh OCC path, but it exposes the exact
region numbering used later by the conductivity list. That direct control is why
`ConstructNetgen2dModel` is the default 2D backend.

## 11.3 Gmsh 2D Mesh Construction Walkthrough

The Gmsh 2D path uses OCC boolean operations instead of explicit region-numbered
line bookkeeping.

### Concrete worked example

Take a two-layer model:

- layer 1: clean
- layer 2: invaded, with one filtration zone

The constructor creates these core surfaces first:

- domain surface: tag `3`
- borehole surface: tag `4`

Then it walks through the layer stack.

#### Step 1: top clean layer

Because the first layer has no filtration zone, the code creates one rectangle
(tag `1`) and intersects it with the domain surface:

```text
layer_template tag 1
intersect([(2, 1)], [(2, 3)], tag=5)
```

Result:

- surface `5`: clean upper layer clipped to the circular domain

#### Step 2: lower invaded layer

For the second layer, the code creates two rectangles:

- inner filtration rectangle: temporary tag `1`
- outer undisturbed rectangle: temporary tag `2`

and intersects both with the domain surface:

```text
intersect([(2, 1)], [(2, 3)], tag=6)   -> filtration surface
intersect([(2, 2)], [(2, 3)], tag=7)   -> undisturbed surface
```

Results:

- surface `6`: invaded lower zone
- surface `7`: undisturbed lower zone

#### Final physical-group mapping

After boolean cleanup, the physical-group list is effectively:

```text
4 -> borehole
5 -> upper clean layer
6 -> lower filtration zone
7 -> lower undisturbed zone
```

That is the 2D OCC equivalent of the explicit Netgen region numbering.

### Geometry flow summary

1. build the axis and borehole-wall curves
2. close the top and bottom with circular arcs
3. create the borehole and surrounding domain surfaces
4. intersect one or two rectangles per layer with the domain surface
5. delete duplicates and synchronize
6. apply background and source-proximity mesh fields
7. generate the 2D mesh, classify boundaries, write `.msh`, and read it back

## 11.4 Gmsh 3D Mesh Construction Walkthrough

The 3D path is the most geometry-heavy part of the code.

### Shape diagram 1: half-sphere domain and borehole axis

```text
                  z+
                  ^
                  |
             .-----------.
          .-'             '-.
        .'                     '.
       /                         \
      |            o axis         |   half-sphere domain
      |            |              |
       \           |             /
        '.         |           .'
          '-.      |        .-'
              '----+------'
                   |
                   +------> x
```

### Shape diagram 2: revolved borehole template

The constructor first builds a 2D borehole template in the `x-z` plane and then
revolves it by `pi` around the borehole axis:

```text
2D template in x-z plane          revolution by pi            3D result

axis + wall profile        --->   sweep around z-axis   --->  half-borehole volume
```

### Shape diagram 3: rotated dipping layer box

Each dipping layer is created as a large box and rotated around the `y` axis:

```text
before rotation                    after rotation by dip

+-------------+                    +-------------+
|             |                    |\            |
|   layer     |      ---->         | \  layer    |
|             |                    |  \          |
+-------------+                    +---\---------+
```

That rotated slab is then intersected with the half-sphere-minus-borehole
formation volume.

### Shape diagram 4: filtration-zone cylinder split

For invaded dipping layers, the rotated slab is split again by a cylinder around
the borehole axis:

```text
cross-section view

      outer layer volume
   +-----------------------+
   |     filtration         |
   |      cylinder          |
   |        ||              |
   |        ||              |
   +-----------------------+
```

The inner cylinder intersection becomes the invaded zone and the cut remainder
becomes the undisturbed zone.

### Full 3D operation sequence

1. create the 2D borehole template in the `x-z` plane
2. revolve it by `pi` to obtain a half-borehole volume
3. create a half-sphere domain
4. intersect borehole and sphere to get the borehole volume
5. cut the borehole from the sphere to get the formation volume
6. create one rotated box per layer and intersect it with the formation volume
7. for invaded layers, split the rotated slab with a cylinder around the axis
8. apply mesh fields, generate the 3D mesh, classify boundaries, and assign
   physical groups
9. read the Gmsh output back into Netgen format

## 11.5 `_prepare_simulation_depths_and_tasks` Walkthrough

This preparation algorithm solves two problems at once:

- which local solve depths are needed
- how several requested measurements can share one solve

### Worked example with 3 tools and 10 measurement depths

Take:

- measurement depths: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- tools:
  - tool index `0`: `A1.0M2.0N`, `depth_shift = -0.5`
  - tool index `1`: `M1.0A2.0N`, `depth_shift = 0.5`
  - tool index `2`: `A2.0M1.0N`, `depth_shift = -2.5`
- `batch_size = 4`

Their canonical `tool_parameters` arrays are:

```text
tool 0:
[[ 0.0, 1.0, 3.0, 18.8496],
 [ 1.0, 0.0, 0.0, -0.5   ]]

tool 1:
[[-1.0, 0.0, 2.0, 25.1327],
 [ 0.0, 1.0, 0.0,  0.5   ]]

tool 2:
[[ 0.0, 2.0, 3.0, 75.3982],
 [ 1.0, 0.0, 0.0, -2.5   ]]
```

### Step 1: compute simulation depths

```text
tool 0 -> [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
tool 1 -> [ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
tool 2 -> [-2.5,-1.5,-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
```

Because SEC mode is active, the code merges identical depths across tools:

```text
unique simulation depths =
[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
```

### Step 2: batch and compute offsets

With `batch_size = 4`:

```text
simulation_depths =
[[-2.5, -1.5, -0.5,  0.5],
 [ 1.5,  2.5,  3.5,  4.5],
 [ 5.5,  6.5,  7.5,  8.5],
 [ 9.5,  nan,  nan,  nan]]

combined_simulation_depths = [-1.0, 3.0, 7.0, 9.5]

simulation_offsets =
[[-1.5, -0.5,  0.5,  1.5],
 [-1.5, -0.5,  0.5,  1.5],
 [-1.5, -0.5,  0.5,  1.5],
 [ 0.0,  nan,  nan,  nan]]
```

### Step 3: build the nested task structure

The first batch becomes:

```text
[
  0,
  [[-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
   [ 1.0,  1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
  [
    [0,
     [[-1.5, 0.5, 1.5],
      [ 1.0, 0.0, 0.0]],
     [[0, 2, -1.5]]],

    [1,
     [[-0.5, 1.5, 2.5],
      [ 1.0, 0.0, 0.0]],
     [[1, 2, -0.5]]],

    [2,
     [[0.5, 1.5, 2.5, 3.5],
      [1.0, 0.0, 0.0, 0.0]],
     [[0, 0, 0.5], [2, 2, 0.5]]],

    [3,
     [[0.5, 1.5, 2.5, 3.5, 4.5],
      [0.0, 1.0, 0.0, 0.0, 0.0]],
     [[1, 0, 1.5], [0, 1, 1.5], [3, 2, 1.5]]]
  ]
]
```

Interpretation:

- outer index `0`: batch number
- `batch_combined_tools`: every electrode needed anywhere in the batch
- inner entries `0..3`: one shared solve depth inside that batch
- innermost tuples: `[measurement_depth_index, tool_index, simulation_offset]`

This is the exact idea behind the nested list described in
[`data-structures.md`](data-structures.md#126-task-list-structure).

## 11.6 Master-Worker Communication Walkthrough

1. `initialize_workers` spawns the worker processes.
2. The master broadcasts `solve_on` so each worker knows whether to import the
   CPU or GPU solve module.
3. `simulate_logs` computes task metadata and broadcasts the shared arrays and
   options.
4. Workers wait at a barrier until every worker has the same configuration.
5. The master enters a `recv(MPI.ANY_SOURCE)` loop.
6. Each worker requests one task index with `sendrecv(None, dest=0)`.
7. The master replies with a task index until all tasks are exhausted.
8. The master then replies with one `StopIteration` per worker.
9. Workers gather their local result lists.
10. On final shutdown, the outer loop receives `StopIteration` and each worker
    disconnects.

The communication-specific details are documented in more formal terms in
[`parallel-execution.md`](parallel-execution.md#72-mpi-communication-protocol).

## See Also

- [`mesh-generation.md`](mesh-generation.md#4-mesh-generation---gmsh-gmsh_functionspy): geometry-construction details behind the walkthroughs.
- [`model-api.md`](model-api.md#312-_prepare_simulation_depths_and_tasks): orchestration logic that prepares the nested task structure.
- [`data-structures.md`](data-structures.md#126-task-list-structure): exact meanings of the task arrays and nested lists.
- [`parallel-execution.md`](parallel-execution.md#76-dynamic-load-balancing): how the prepared tasks are dispatched at runtime.
