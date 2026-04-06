# Model Class (`remo3d.py`) API

## 3.1 `Model.__init__`

Signature:

```python
Model(tools, force_single_electrode_configuration=True)
```

Purpose:

- parse and normalize the logging-tool configuration
- decide whether single-electrode computation mode (`sec`) is possible
- initialize model, worker, and output attributes

Initialized attributes:

| Attribute | Meaning |
| --- | --- |
| `self.tools` | dictionary mapping tool name to a `2x4` parameter array |
| `self.sec` | `True` when every tool can be handled as a single-current-electrode tool |
| `self.formation_model` | formation array after loading/conversion |
| `self.borehole_model` | borehole array after loading/conversion |
| `self.dip_deg`, `self.dip_rad` | dip angle in degrees and radians |
| `self.cpu_workers`, `self.gpu_workers` | worker counts after initialization |
| `self.comm` | MPI intercommunicator returned by `MPI.COMM_WORLD.Spawn` |
| `self.logs` | final log dictionary after simulation |

## 3.2 `Model.compute_synthetic_logs`

`compute_synthetic_logs` is the convenience wrapper that drives the full
workflow:

```python
model = Model.compute_synthetic_logs(...)
```

Internal sequence:

1. `model = cls(tools, ...)`
2. `model.set_model_parameters(...)`
3. `model.initialize_workers(...)`
4. `model.simulate_logs(...)`
5. `model.shutdown_workers()`
6. return the populated `model`

This classmethod is the one-shot API. For repeated simulations inside an
inversion loop, the lower-level worker lifecycle can be used directly.

## 3.3 Tool Naming Convention

Every tool string contains:

- three electrode symbols chosen from `A`, `B`, `M`, `N`
- two positive numeric spacings between consecutive electrodes
- top-to-bottom electrode order

Examples:

- `B5.7A0.4M`
- `A2.0M0.5N`
- `N0.5M2.0A`

Parsing logic:

```python
[''.join(group) for _, group in itertools.groupby(tool, str.isalpha)]
```

This splits `B5.7A0.4M` into:

```text
["B", "5.7", "A", "0.4", "M"]
```

Then `_str2float` converts numeric tokens to floats while keeping the electrode
symbols as strings.

### Measurement-point placement

The code assumes the measurement point lies halfway between the closer pair of
adjacent electrodes:

- if `distances[0] < distances[1]`, use `distances[0] / 2`
- if `distances[0] > distances[1]`, use `distances[0] + distances[1] / 2`

Equal spacings are rejected as invalid in the current implementation.

## 3.4 `_set_tool_parameters`

`_set_tool_parameters(tool, electrodes, distances)` builds the canonical tool
array:

```text
[[z1, z2, z3, geometric_factor],
 [s1, s2, s3, depth_shift]]
```

where:

- `z1:z3` are the sorted electrode depths relative to the simulation point
- `s1:s3` are the source terms aligned with the sorted electrodes
- `geometric_factor` converts potential to apparent resistivity
- `depth_shift` maps measurement depth to simulation depth

### Geometric-factor branch selection

The code infers the tool family by checking which electrode is missing from the
three-electrode string:

- missing `A`: current at `B`, measurements at `M` and `N`
- missing `B`: current at `A`, measurements at `M` and `N`
- missing `M`: currents at `A` and `B`, measurement at `N`
- missing `N`: currents at `A` and `B`, measurement at `M`

### Public dictionary structure

The public `self.tools` attribute is:

```python
{
    "tool-name": np.ndarray(shape=(2, 4)),
    ...
}
```

The first three columns always refer to sorted electrode positions and aligned
source terms. The fourth column stores metadata:

- row `0`, column `3`: geometric factor
- row `1`, column `3`: depth shift

## 3.5 `force_single_electrode_configuration`

When `force_single_electrode_configuration=True`, any tool containing both `A`
and `B` is rewritten by:

```python
tool.translate(str.maketrans("ABMN", "MNAB"))
```

This swaps current and potential roles to replace a two-current-electrode solve
with an equivalent single-current-electrode form for the normalized potential
problem solved by the code.

## 3.6 Single-Electrode Computation Mode (`sec`)

`self.sec` is `True` only if every tool has a nonzero sum of source terms in the
first three columns. That means every tool effectively has one active current
electrode. In this mode:

- multiple tools can share the same solve when they use the same current-source
  location
- the worker computes one PDE solution and many measurements from it

This is the main throughput optimization for mixed normal/lateral tool suites.

## 3.7 `set_model_parameters`

Purpose:

- accept either file paths or already-loaded arrays
- apply unit conversion and validation
- set dip angle
- verify borehole vs filtration-zone geometry consistency

Flow:

1. if `formation_model` is a string, call `load_formation_parameters`
2. if it is an array, call `set_formation_parameters`
3. do the same for the borehole model
4. call `set_dip`
5. call `_check_model_geometry`

## 3.8 Formation Model Data Format

Formation files are tab-delimited text files with:

1. a header row
2. a units row
3. one row per layer

Columns:

| Column | Meaning |
| --- | --- |
| `TOP` | top depth of the layer |
| `BOTTOM` | bottom depth of the layer |
| `FZ_RADIUS` | filtration-zone radius; `NaN` means no filtration zone |
| `FZ_VALUE` | filtration-zone resistivity |
| `UZ_VALUE` | undisturbed-zone resistivity |

After conversion the in-memory array shape is `(n_layers, 5)` with geometry in
meters and resistivity in ohm-meters.

## 3.9 Borehole Model Data Format

Borehole files are also tab-delimited with:

1. a header row
2. a units row
3. one row per sampled depth

Columns:

| Column | Meaning |
| --- | --- |
| `DEPT` | depth |
| `CALM` | caliper, interpreted as diameter unless `borehole_geometry_type='radius'` |
| `RM` | mud resistivity |

In memory the borehole array is shape `(n_points, 3)`:

```text
[depth, radius, mud_resistivity]
```

If the input is diameter-based, column 2 is divided by two during setup.

## 3.10 `_check_model_geometry`

This validation checks that the borehole radius never reaches or exceeds the
filtration-zone radius in any layer interval covered by the borehole samples.

Interpretation:

- the invaded zone must surround the borehole if it exists
- a missing filtration zone is represented by `NaN`

## 3.11 `initialize_workers` and `shutdown_workers`

### `initialize_workers`

Responsibilities:

- validate worker counts
- probe CUDA support if `gpu_workers > 0`
- spawn `remo3d/workers/worker.py` through MPI
- broadcast the per-worker execution target list, for example
  `['CPU', 'CPU', 'GPU']`

Notes:

- the master process is not included in `cpu_workers`
- a failed CUDA import downgrades `gpu_workers` to `0`
- workers wait at a barrier after receiving `solve_on`

### `shutdown_workers`

The master sends one `StopIteration` sentinel per worker through the same
request/reply loop used for tasks. The workers then break their outer loop and
disconnect.

## 3.12 `_prepare_simulation_depths_and_tasks`

This method transforms measurement depths into worker tasks.

### Step 1: compute tool-specific simulation depths

For each tool:

```python
measurement_depths + self.tools[tool][1, 3]
```

The fourth value in row 1 is the tool-specific depth shift.

### Step 2: combine depths

- `sec=True`: merge identical simulation depths across all tools
- `sec=False`: keep each tool/depth pair separate

### Step 3: batch adjacent depths

The simulation-depth array is padded with `NaN`, reshaped to
`(number_of_batches, batch_size)`, and reduced to one representative simulation
depth per batch:

```python
combined_simulation_depths = np.round(np.nanmean(simulation_depths, axis=1), 4)
```

Offsets from the batch-center depth are stored separately.

### Step 4: build nested task objects

Returned structure:

```text
[
  [batch_index, batch_combined_tools, [
    [simulation_depth_index, combined_tools, [
      [measurement_depth_index, tool_index, simulation_offset],
      ...
    ]],
    ...
  ]],
  ...
]
```

Interpretation:

- outer list item: one worker task, typically one batch
- `batch_combined_tools`: union of all electrodes needed anywhere in the batch
- inner `combined_tools`: electrodes needed for one shared solve depth
- innermost entries: how to map one solve result back into final log samples

## 3.13 `_add_points_to_borehole`

The 3D mesher is more sensitive to sparse borehole polylines than the 2D code.
To reduce meshing failures, `_add_points_to_borehole` inserts interpolated
points whenever two borehole samples are more than `0.15 m` apart.

Interpolation rules:

- depth: linear `np.linspace`
- radius: `scipy.interpolate.interp1d(..., kind='linear')`
- mud resistivity: same linear interpolation

## 3.14 `simulate_logs`

`simulate_logs` is the main master-side execution loop.

### Input preparation

- validate `domain_radius` against every electrode offset
- choose the mesh backend:
  - `auto -> netgen` for `dip == 0`
  - `auto -> gmsh` for `dip != 0`
- create `./tmp` for Gmsh-generated meshes
- densify the borehole path for 3D runs

### Broadcast phase

The master sends:

- array shapes via `comm.bcast`
- contiguous numeric arrays via `comm.Bcast`
- scalar settings and Python containers via `comm.bcast`

Broadcast payload:

- `formation_model`
- `borehole_geometry`
- `mud_resistivities`
- `simulation_depths`
- `dip_rad`
- `tools`
- `domain_radius`
- `mesh_generator`
- `preconditioner`
- `condense`
- `task_list`

### Dynamic dispatch phase

Workers repeatedly request work. The master responds to whichever worker becomes
free first:

```python
self.comm.recv(source=MPI.ANY_SOURCE, status=status)
self.comm.send(obj=msg, dest=status.Get_source())
```

This provides load balancing automatically when some local meshes or solves are
more expensive than others.

### Gather and assembly phase

The gathered triplets:

```text
[measurement_depth_index, tool_index, result]
```

are assembled into a dense 2D array and then converted into the public
structure:

```python
{
    tool_name: np.vstack([measurement_depths, apparent_resistivity]).T
}
```

## 3.15 `save_results`

`save_results(...)` performs two jobs:

1. export logs as tab-delimited text files
2. render a combined formation/log figure with Matplotlib

### Text export

Logs are grouped into one output file when they share the same measurement-depth
vector. Each file contains:

- first column: `DEPTH`
- remaining columns: one or more tool names
- units row: `M` followed by `OHMM`

### Plot export

The figure contains:

- one formation-model panel
- one or more log tracks
- a shared resistivity colorbar

Key plotting parameters:

- `plot_layout`
- `plot_depth_lim`
- `plot_aspect_ratio`
- `model_rad_lim`
- `model_res_lim`
- `logs_res_lim`
- `logs_at_nan`
- `logs_interpolation_factor`
- `logs_colours`

### Practical caveats

- `logs_interpolation_factor > 1` overwrites `self.logs` with interpolated
  curves for plotting.
- The method saves PNG output only when `output_folder` is provided.

## Sharp Edge in the Public Signature

Two parameters appear in `compute_synthetic_logs`:

- `formation_units`
- `borehole_units`

but the current implementation does not forward them into
`set_model_parameters`. File-based loading still uses the units row stored in
the file, and direct-array setup still works if arrays are already in meters,
but the wrapper signature currently advertises more configurability than it
actually applies.
