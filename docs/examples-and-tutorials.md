# Examples and Tutorials

This page is the usage-oriented companion to
[`model-api.md`](model-api.md#32-modelcompute_synthetic_logs) and the benchmark
notes in [`testing-and-validation.md`](testing-and-validation.md#151-existing-benchmark-models-and-expected-behavior).

## 14.1 Tutorial from Example_01

Example_01 is the minimal usage path.

### Full source code

```python
"""
   Example 1
   The script presents a basic use of the package.
   Only required parameters are used.

   How to run:
   mpiexec python3 Example_01.py
"""

from remo3d import Model
import numpy as np

# Specify input data
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]
formation_model_file = "./Input/Formation.txt"
borehole_model_file = "./Input/Borehole.txt"
measurement_depths = np.arange(0, 25.1, 0.1)

# Create model and simulate logs
model = Model.compute_synthetic_logs(tools, measurement_depths, formation_model_file, borehole_model_file)

# Save results
model.save_results(output_folder="./Output")
```

### Commentary

- The module import exposes the single public class used by the examples:
  `Model`.
- The `tools` list mixes short- and long-spacing devices so the example shows a
  realistic multi-tool run.
- `formation_model_file` and `borehole_model_file` point to the tab-delimited
  input files documented in [`io-formats.md`](io-formats.md#81-formation-model-file-format).
- `measurement_depths = np.arange(0, 25.1, 0.1)` creates 251 samples from
  `0.0 m` to `25.0 m`.
- `Model.compute_synthetic_logs(...)` runs the full one-shot workflow:
  initialize, load, spawn workers, simulate, and shut down workers.
- `save_results(output_folder="./Output")` writes one or more `Results_#.txt`
  files and `Results_plot.png`.

### Expected outputs

The example output directory already included in the repository shows the naming
pattern:

- `Results_1.txt`
- `Results_plot.png`

## 14.2 Tutorial from Example_02

Example_02 demonstrates the optional parameters.

### Full source code

```python
"""
   Example 2
   The script presents a more advanced use of the package.
   Required and optional parameters are used.

   How to run:
   mpiexec python3 Example_02.py
"""

from remo3d import Model
import numpy as np

# Specify input data
tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]
formation_model_file = "./Input/Formation.txt"
borehole_model_file = "./Input/Borehole.txt"
measurement_depths = np.arange(0, 25.1, 0.1)

# Create model and simulate logs
model = Model.compute_synthetic_logs(tools, measurement_depths, formation_model_file, borehole_model_file, borehole_geometry_type='diameter', dip=0,
                                     cpu_workers=11, gpu_workers=0, mesh_generator="netgen", domain_radius=25, batch_size=10)

# Save results
model.save_results(output_folder="./Output",
               plot_layout=[["B5.7A0.4M", "B4.48A1.62M"], ["M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]],
               plot_depth_lim=[0, 25], plot_aspect_ratio=1.25,
               model_rad_lim=[-1, 1], model_res_lim=[0, 20],
               logs_colours = [["red", "blue"], ["green", "orange", "purple", "deepskyblue"]],
               logs_res_lim=[0, 30], logs_at_nan="break")
```

### Commentary on the optional parameters

- `borehole_geometry_type='diameter'` makes the meaning of the `CALM` column
  explicit.
- `dip=0` keeps the example on the 2D axisymmetric path.
- `cpu_workers=11` shows an explicitly parallel run instead of relying on the
  default worker count.
- `mesh_generator="netgen"` forces the 2D Netgen path instead of relying on the
  `auto` selector.
- `domain_radius=25` trades some truncation safety for a smaller local domain.
- `batch_size=10` is much more aggressive than the default and is meant to show
  a performance-oriented configuration.
- `plot_layout` groups the six tools into two log tracks.
- `plot_depth_lim`, `plot_aspect_ratio`, `model_rad_lim`, and `model_res_lim`
  reshape the model panel.
- `logs_colours` and `logs_res_lim` explicitly control log-track appearance.
- `logs_at_nan="break"` keeps any missing-value segments visually separated.

Compared with Example_01, Example_02 is mainly about performance tuning and plot
customization.

## 14.3 Quick Start with Inline Arrays

This is the smallest self-contained example that avoids external model files.
Run it under `mpiexec` in an environment where the numerical dependencies are
installed.

```python
from remo3d import Model
import numpy as np

tools = ["A1.0M2.0N"]
measurement_depths = np.arange(0.0, 6.1, 0.5)

formation_model = np.array([
    [0.0, 2.0, np.nan, np.nan, 5.0],
    [2.0, 4.0, 0.30, 3.0, 20.0],
    [4.0, 6.5, np.nan, np.nan, 8.0],
], dtype=float)

borehole_model = np.array([
    [0.0, 0.10, 1.0],
    [6.5, 0.10, 1.0],
], dtype=float)

model = Model.compute_synthetic_logs(
    tools,
    measurement_depths,
    formation_model,
    borehole_model,
    borehole_geometry_type="radius",
    cpu_workers=2,
    gpu_workers=0,
    domain_radius=15,
    batch_size=3,
    mesh_generator="netgen",
)

model.save_results(output_folder="./Output")
```

Why this is useful:

- no external text files are required
- the formation and borehole arrays directly match the in-memory layouts in
  [`data-structures.md`](data-structures.md#122-formation_model)
- it is the fastest way to sanity-check that the runtime stack is working

## 14.4 Creating Custom Formation Models

Guidelines:

1. make layers contiguous: each `BOTTOM` must equal the next `TOP`
2. use `NaN` in `FZ_RADIUS` and `FZ_VALUE` for layers without invasion
3. keep geometry units consistent with the units row
4. choose `FZ_VALUE` and `UZ_VALUE` as resistivity values in ohm-meters

Modeling effects:

- thin layers create sharp depth responses and can demand smaller batches
- larger filtration radii increase the near-well influence on the tools
- stronger resistivity contrasts increase the apparent-log contrast and can make
  meshing or convergence more sensitive

## 14.5 Creating Custom Borehole Models

Guidelines:

1. provide at least two depth samples
2. keep depth strictly increasing
3. store `CALM` as diameter unless you pass `borehole_geometry_type='radius'`
4. provide mud resistivity in `OHMM`

Common scenarios:

- washout: locally larger `CALM`
- stable hole: nearly constant `CALM`
- mud-property variation: smoothly varying `RM`

## 14.6 Interpreting Simulation Results

Each log in `model.logs` is an apparent-resistivity curve sampled at the
requested measurement depths.

Interpretation tips:

- compare peaks and troughs with formation boundaries and invaded zones
- strong tool-to-tool differences often indicate different investigation depths
- abrupt artifacts near the top or bottom of the simulated interval can be
  boundary effects
- missing values (`NaN`) indicate that a worker-side task failed and was caught
  by the worker fallback logic

## 14.7 Benchmark Models

### Benchmark model 1

- alternating clean `10/100 ohm-m` layering
- no invasion zones
- constant `200 mm` borehole and `1 ohm-m` mud
- useful for validating clean layer-boundary response

### Benchmark model 2

- alternating background layers with three invaded intervals
- filtration-zone radii grow from `0.2 m` to `0.5 m`
- useful for validating invaded-zone region assembly and apparent-response shape

### Benchmark model 3

- three-layer model repeated at dip angles `0`, `15`, `30`, `45`, and `60`
- constant borehole and mud properties
- useful for validating the 3D dipping workflow

### Thin-bedded model

- two formation realizations
- three borehole mud variants (`0.2`, `0.35`, `0.5 ohm-m`)
- four log groups showing boundary and depth-shift effects
- one depth-shift table showing assigned versus true measurement depths

For the full benchmark-file descriptions, see
[`testing-and-validation.md`](testing-and-validation.md#151-existing-benchmark-models-and-expected-behavior).

## 14.8 Performance Tuning Guide

Main levers:

- `cpu_workers`: increase until the machine saturates on memory or cores
- `batch_size`: increase for smoother models, reduce for fast local variation
- `domain_radius`: increase for accuracy, reduce for speed
- `mesh_generator`: use `netgen` for routine 2D work and `gmsh` for 3D
- `preconditioner`: keep `multigrid` unless a smaller problem favors `local`

Practical rule set:

- start with default settings
- change one performance parameter at a time
- validate a tuned configuration against a smaller or more conservative run
  before using it as a production baseline

## See Also

- [`model-api.md`](model-api.md#315-save_results): output structure and plotting options used by the tutorials.
- [`io-formats.md`](io-formats.md#83-output-results-file-format): exact text-file layout produced by the examples.
- [`configuration.md`](configuration.md#91-compute_synthetic_logs-parameter-reference): full parameter reference for the optional arguments shown above.
- [`testing-and-validation.md`](testing-and-validation.md#151-existing-benchmark-models-and-expected-behavior): benchmark-specific validation intent behind the example assets.
