# Examples and Tutorials

## 14.1 Tutorial from Example_01

Example_01 is the minimal usage path:

```python
from remo3d import Model
import numpy as np

tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"]
formation_model_file = "./Input/Formation.txt"
borehole_model_file = "./Input/Borehole.txt"
measurement_depths = np.arange(0, 25.1, 0.1)

model = Model.compute_synthetic_logs(tools, measurement_depths, formation_model_file, borehole_model_file)
model.save_results(output_folder="./Output")
```

Line-by-line intent:

- import `Model`
- define tool geometry strings
- point to the formation and borehole files
- define the depth sampling
- run the full one-shot workflow
- save the text output and figure

## 14.2 Tutorial from Example_02

Example_02 demonstrates the optional parameters:

- explicit `borehole_geometry_type='diameter'`
- explicit `dip=0`
- explicit worker counts
- explicit mesh backend
- explicit `domain_radius` and `batch_size`
- custom plot layout, axis limits, and colors

Use this pattern when you want to control accuracy/performance tradeoffs and the
visual presentation instead of relying on defaults.

## 14.3 Creating Custom Formation Models

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

## 14.4 Creating Custom Borehole Models

Guidelines:

1. provide at least two depth samples
2. keep depth strictly increasing
3. store `CALM` as diameter unless you pass `borehole_geometry_type='radius'`
4. provide mud resistivity in `OHMM`

Common scenarios:

- washout: locally larger `CALM`
- stable hole: nearly constant `CALM`
- mud-property variation: smoothly varying `RM`

## 14.5 Interpreting Simulation Results

Each log in `model.logs` is an apparent-resistivity curve sampled at the
requested measurement depths.

Interpretation tips:

- compare peaks and troughs with formation boundaries and invaded zones
- strong tool-to-tool differences often indicate different investigation depths
- abrupt artifacts near the top or bottom of the simulated interval can be
  boundary effects
- missing values (`NaN`) indicate that a worker-side task failed and was caught
  by the worker fallback logic

## 14.6 Benchmark Models

### Benchmark model 1

Files:

- `Examples/Benchmark models/Benchmark model 1/Formation_BM1.txt`
- `Examples/Benchmark models/Benchmark model 1/Borehole_BM1.txt`

Purpose:

- layered 10/100 ohm-m formation without filtration zones
- constant-diameter borehole
- useful baseline for checking layer-boundary responses

### Benchmark model 2

Files:

- `Examples/Benchmark models/Benchmark model 2/Formation_BM2.txt`
- `Examples/Benchmark models/Benchmark model 2/Borehole_BM2.txt`

Purpose:

- alternating clean and invaded layers
- increasing filtration-zone radii
- useful for checking invaded-zone handling

### Benchmark model 3

Files:

- `Formation_BM3_00.txt`
- `Formation_BM3_15.txt`
- `Formation_BM3_30.txt`
- `Formation_BM3_45.txt`
- `Formation_BM3_60.txt`

Purpose:

- same basic layer stack at multiple dip angles
- useful for checking the 3D dipping path

### Thin-bedded model

The `Thin-bedded model` directory contains:

- two formation realizations
- three borehole mud cases
- four log sets documenting boundary effects and depth misalignment

Its README explains the synthetic-model generation process and the meaning of
`Logs 1` through `Logs 4`.

## 14.7 Performance Tuning Guide

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
