# Data Structures Reference

## 12.1 `tools_parameters`

Public location: `Model.tools`

Type:

```python
dict[str, np.ndarray]
```

Value shape:

```text
(2, 4)
```

Layout:

```text
row 0 = [electrode_z_1, electrode_z_2, electrode_z_3, geometric_factor]
row 1 = [source_term_1, source_term_2, source_term_3, depth_shift]
```

Interpretation:

- column order is sorted by electrode depth
- `source_term == 0` means a measuring electrode
- `source_term != 0` means a current electrode
- `depth_shift` maps measurement depth to simulation depth

## 12.2 `formation_model`

Location: `Model.formation_model`

Shape:

```text
(n_layers, 5)
```

Columns:

| Index | Name | Units after conversion |
| --- | --- | --- |
| `0` | top depth | meters |
| `1` | bottom depth | meters |
| `2` | filtration-zone radius | meters |
| `3` | filtration-zone resistivity | ohm-m |
| `4` | undisturbed-zone resistivity | ohm-m |

Notes:

- `NaN` in column `2` means no filtration zone
- layers must be contiguous: each row's bottom must equal the next row's top

## 12.3 `borehole_model`

Location: `Model.borehole_model`

Shape:

```text
(n_points, 3)
```

Columns:

| Index | Name | Units after conversion |
| --- | --- | --- |
| `0` | depth | meters |
| `1` | radius | meters |
| `2` | mud resistivity | ohm-m |

## 12.4 `local_formation_geometry` for Netgen

Produced by: `SelectNetgenDataRange`

Shape:

```text
(n_local_layers, 5)
```

Columns:

| Index | Meaning |
| --- | --- |
| `0` | local top depth relative to the simulation point |
| `1` | local bottom depth relative to the simulation point |
| `2` | filtration-zone radius, or `NaN` |
| `3` | region index adjacent to the borehole / inside filtration radius |
| `4` | region index outside the filtration radius |

The region indices are integers that index into the conductivity distribution
returned with the same function.

## 12.5 `local_formation_geometry` for Gmsh

Produced by: `SelectGmshFormationDataRange`

Shape:

```text
(n_local_layers, 3)
```

Columns:

| Index | Meaning |
| --- | --- |
| `0` | local top depth relative to the simulation point |
| `1` | local bottom depth relative to the simulation point |
| `2` | filtration-zone radius, or `NaN` |

Unlike the Netgen path, the Gmsh path keeps region numbering implicit in the
order of generated surfaces or volumes and in the separate resistivity list.

## 12.6 Task List Structure

Produced by: `Model._prepare_simulation_depths_and_tasks`

Top-level structure:

```text
[
  [batch_index, batch_combined_tools, batch_modelling_tasks],
  ...
]
```

### `batch_combined_tools`

Shape:

```text
(2, n_batch_electrodes)
```

Rows:

- row `0`: electrode positions
- row `1`: source terms, `0` for potential electrodes and `1` for current
  electrodes at this batch-union level

### `batch_modelling_tasks`

Structure:

```text
[
  [simulation_depth_index, combined_tools, modelling_tasks],
  ...
]
```

### `combined_tools`

Again a `2 x n` array:

- row `0`: electrode positions for one shared solve
- row `1`: source terms for that solve

### `modelling_tasks`

Structure:

```text
[
  [measurement_depth_index, tool_index, simulation_offset],
  ...
]
```

## 12.7 `logs`

Location: `Model.logs`

Type:

```python
dict[str, np.ndarray]
```

Value shape:

```text
(n_measurements, 2)
```

Columns:

| Column | Meaning |
| --- | --- |
| `0` | measurement depth |
| `1` | apparent resistivity |

Each key is the original tool name string supplied by the user.

## 12.8 Conductivity Distribution `sigma`

Generated in:

- `SelectGmshDataRange`
- `SelectNetgenDataRange`

Structure:

```text
[1 / mud_resistivity, 1 / rho_region_1, 1 / rho_region_2, ...]
```

The ordering follows region-number order in the mesh generator:

- index `0`: borehole mud conductivity
- later indices: formation regions in the order they are created

The worker wraps this list as:

```python
sigma = ngs.CoefficientFunction(sigma)
```

so the FEM bilinear form can evaluate the correct piecewise-constant
conductivity per region.
## See Also
- [model-api.md](model-api.md#312-_prepare_simulation_depths_and_tasks): where the nested task structures are built.
- [walkthroughs.md](walkthroughs.md#115-_prepare_simulation_depths_and_tasks-walkthrough): worked example showing these arrays with real indices.
- [parallel-execution.md](parallel-execution.md#74-per-task-worker-pipeline): how the worker consumes the documented structures.
