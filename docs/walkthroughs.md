# Code Walkthroughs - Key Algorithms

## 11.1 Complete 2D Simulation Walkthrough (Example_01)

Example_01 uses:

- 6 tools
- measurement depths from `0` to `25` m with a `0.1` m step
- formation array with 7 layers
- borehole array with 251 samples
- `dip=0`, so the axisymmetric path is selected

Step-by-step:

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

## 11.2 Netgen 2D Mesh Construction Walkthrough

The Netgen constructor follows a geometric bookkeeping approach.

1. Build all points at the borehole axis.
2. Add all points along the borehole wall, including intersections with layer
   boundaries.
3. Add points at filtration-zone interfaces.
4. Add layer-boundary endpoints on the outer domain.
5. Add extra points along the circular boundary every 9 degrees where needed.
6. Connect points with vertical, horizontal, and outer-boundary line segments.
7. Assign left and right domain indices to each segment.
8. Feed the resulting lists into `SplineGeometry`.
9. Mark current-electrode points with the smallest mesh size.
10. Generate the final 2D mesh.

Conceptually the geometry looks like:

```text
axis | borehole wall | filtration wall | outer circle
```

with region numbers changing whenever a line crosses a layer boundary or a
filtration boundary.

## 11.3 Gmsh 2D Mesh Construction Walkthrough

The Gmsh 2D path uses OCC boolean operations instead of explicit region-numbered
line bookkeeping.

1. Build the borehole axis and borehole wall curves.
2. Close the top and bottom with circular arcs.
3. Create the borehole and surrounding domain surfaces.
4. For each layer, create a rectangle and intersect it with the domain surface.
5. For invaded layers, split the layer into inner and outer rectangles and
   intersect both with the domain.
6. Remove duplicate entities and synchronize.
7. Apply background and source-proximity mesh fields.
8. Generate the 2D mesh, classify boundaries, and write the `.msh` file.
9. Read the `.msh` back into a Netgen mesh object.

## 11.4 Gmsh 3D Mesh Construction Walkthrough

The 3D path is the most geometry-heavy part of the code.

1. Create a 2D borehole template in the `x-z` plane.
2. Revolve it by `pi` around the borehole axis to obtain a half-borehole volume.
3. Create a half-sphere domain.
4. Intersect the revolved borehole with the sphere to get the borehole volume.
5. Cut the borehole out of the sphere to get the formation volume.
6. For each dipping layer, create a large box, rotate it by the dip angle, and
   intersect it with the formation volume.
7. For invaded layers, split the rotated slab again with a cylinder.
8. Apply the mesh fields, generate the 3D mesh, classify boundaries, and assign
   volume groups.
9. Read the Gmsh file back into Netgen format.

## 11.5 `_prepare_simulation_depths_and_tasks` Walkthrough

The preparation algorithm solves two problems at once:

- which local solve depths are needed
- how several requested measurements can share one solve

High-level steps:

1. add each tool's `depth_shift` to the measurement depths
2. merge identical simulation depths when SEC mode is active
3. pad and reshape simulation depths into batches
4. compute one representative depth per batch
5. store per-measurement offsets from that representative depth
6. build nested task entries that remember how to map each solve result back to
   `(measurement_depth_index, tool_index)` pairs

The key data structure is:

```text
[batch_index, batch_combined_tools, batch_modelling_tasks]
```

where `batch_combined_tools` is the union of all electrodes needed anywhere in
that batch.

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
