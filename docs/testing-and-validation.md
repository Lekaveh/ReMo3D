# Testing and Validation

## 15.1 Existing Benchmark Models and Expected Behavior

The repository ships benchmark inputs and, in some cases, sample outputs. The
benchmarks are best treated as regression references and qualitative validation
cases rather than full analytical truth tables.

### Benchmark model 1

Expected behavior:

- symmetric layer-boundary response in a non-invaded layered medium
- sensitivity mainly to the vertical placement of the tool relative to the
  10/100 ohm-m interfaces

### Benchmark model 2

Expected behavior:

- clear distinction between invaded and undisturbed zones
- sensitivity to the filtration-zone radius and resistivity

### Benchmark model 3

Expected behavior:

- matches the undipped behavior at `0` degrees
- increasingly 3D responses as the dip angle rises through `15`, `30`, `45`,
  and `60` degrees

### Thin-bedded model

Expected behavior:

- strong sensitivity to thin layering
- visible impact of depth misalignment and boundary effects, as documented in
  `Logs 1` through `Logs 4`

The repository does not include a separate analytical-results table, so the
sample benchmark inputs and stored result files act as the main in-repo
reference set.

## 15.2 Creating New Validation Tests

Recommended workflow:

1. start from a geometry simple enough to reason about
2. keep one effect active at a time
3. compare ReMo3D output against either an analytical solution or a trusted
   reference run
4. store the input files and expected outputs together

Good validation targets:

- homogeneous medium
- one sharp layer boundary
- one invaded layer with a simple cylindrical filtration zone
- the same model at several `domain_radius` values to confirm boundary
  convergence
- the same 2D model with `batch_size=1` and a larger batch size to quantify
  batching error

## 15.3 Known Limitations and Edge Cases

The current code has several practical limits worth documenting:

- very thin layers can create small geometric features and local averaging error
  when batch sizes are large
- extreme resistivity contrasts can stress meshing quality and solver
  convergence
- electrodes close to the outer boundary can create truncation artifacts; the
  code warns once any electrode exceeds `0.75 * domain_radius`
- high dip angles approaching `90` degrees are rejected by `set_dip`
- the worker catches all task exceptions and converts them to `NaN`, which keeps
  the run alive but can hide the first failure mode
- 3D runs are much more expensive than 2D axisymmetric runs and may require a
  denser borehole path, larger memory budget, and more conservative settings
