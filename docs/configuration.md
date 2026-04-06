# Configuration and Parameters

This page covers both the public runtime knobs and the important internal mesh
controls that developers may need to adjust. For the implementation details
behind the meshing defaults, see [`mesh-generation.md`](mesh-generation.md).

## 9.1 `compute_synthetic_logs` Parameter Reference

| Parameter | Type | Default | Allowed values | Meaning | Effect on results / performance |
| --- | --- | --- | --- | --- | --- |
| `tools` | `list[str]` | none | valid 3-electrode tool strings | Logging tools to simulate | Changes tool geometry, depth shift, geometric factor, and whether SEC mode is available |
| `measurement_depths` | `np.ndarray` | none | strictly increasing depths | Final log sample depths | More points increase cost roughly proportionally |
| `formation_model` | `str` or `np.ndarray` | none | valid file path or `(n, 5)` array | Formation geometry/resistivity model | Governs all formation contrasts |
| `borehole_model` | `str` or `np.ndarray` | none | valid file path or `(n, 3)` array | Borehole geometry and mud resistivity | Governs borehole shape and mud conductivity |
| `force_single_electrode_configuration` | `bool` | `True` | `True`, `False` | Rewrites compatible dual-current tools into single-current equivalents | Often unlocks SEC mode and reduces solve count substantially |
| `formation_units` | `list[str]` | `["M", "M", "M"]` | entries from `M`, `DM`, `CM`, `MM`, `IN`, `FT` | Intended geometry-unit specification for formation arrays | Present in the public signature, but not currently forwarded by `compute_synthetic_logs` |
| `borehole_geometry_type` | `str` | `"diameter"` | `"diameter"`, `"radius"` | Interpretation of borehole geometry column 2 | Affects radius conversion and validity checks |
| `borehole_units` | `list[str]` | `["M", "M"]` | entries from `M`, `DM`, `CM`, `MM`, `IN`, `FT` | Intended unit specification for borehole arrays | Present in the public signature, but not currently forwarded by `compute_synthetic_logs` |
| `dip` | `float` | `0` | `0 <= dip < 90` | Layer dip angle in degrees | `0` enables 2D axisymmetry; nonzero dip forces 3D Gmsh and is much more expensive |
| `cpu_workers` | `int` | `4` | `>= 1` | Number of CPU worker processes | More workers improve throughput until meshing/solve cost or memory dominates |
| `gpu_workers` | `int` | `0` | `>= 0` | Number of GPU worker processes | Can accelerate large solves if CUDA-enabled NGSolve is available |
| `domain_radius` | `float` | `50` | positive and larger than electrode offsets | Radius of the local computational domain | Larger is usually more accurate and slower |
| `batch_size` | `int` | `5` | positive integer | Number of adjacent simulation depths grouped into one batch | Larger batches reduce meshing cost but use one representative depth per batch |
| `mesh_generator` | `str` | `"auto"` | `"auto"`, `"netgen"`, `"gmsh"` | Mesh backend selection | `netgen` is only for 2D; `gmsh` is required for 3D |
| `preconditioner` | `str` | `"multigrid"` | `"local"`, `"multigrid"` | NGSolve preconditioner name | Affects CG convergence and setup cost |
| `condense` | `bool` | `True` | `True`, `False` | Toggle static condensation | Often reduces solve cost for order-3 FE spaces |

## 9.2 `domain_radius`

Physical meaning:

- truncation radius for the otherwise unbounded resistivity problem
- radius used to clip local borehole and formation data before meshing

Master-side checks:

- if any electrode offset is outside the domain, the simulation aborts
- if any electrode offset exceeds `0.75 * domain_radius`, a warning is printed

Practical guidance:

- use the smallest radius that still keeps the Dirichlet boundary far from all
  current electrodes and the main resistivity contrasts
- increase the radius for long-spacing tools, large invaded zones, or strong
  contrasts
- expect both meshing and FEM cost to grow as the local domain grows

## 9.3 `batch_size`

Batching groups adjacent simulation depths into one local model and one worker
task. The representative solve depth is the average of the batched simulation
depths, and the per-measurement correction is applied through
`simulation_offset`.

Tradeoff:

- larger `batch_size`: fewer meshes and less setup overhead
- smaller `batch_size`: less geometric averaging and more fidelity to fast depth
  variations

## 9.4 `mesh_generator`

Selection logic in the current implementation:

```python
if mesh_generator == "auto":
    mesh_generator = "netgen" if np.isclose(self.dip_deg, 0) else "gmsh"
```

Capability matrix:

| Backend | 2D | 3D | Typical use |
| --- | --- | --- | --- |
| `netgen` | yes | no | default 2D axisymmetric path |
| `gmsh` | yes | yes | explicit 2D Gmsh runs and all 3D dipping runs |

The master rejects `mesh_generator != "gmsh"` when `dip != 0`.

## 9.5 `preconditioner`

Available names:

- `"local"`
- `"multigrid"`

Rule of thumb:

- choose `"multigrid"` for larger or repeated elliptic solves
- choose `"local"` when setup simplicity matters more than asymptotic solve
  speed

## 9.6 `condense`

`condense=True` enables static condensation in the bilinear form:

```python
a = ngs.BilinearForm(fes, symmetric=False, condense=condense)
```

Implications:

- smaller global linear system
- extra local elimination and reconstruction work
- no change in the mathematical solution, only in how it is assembled and
  solved

## 9.7 CPU and GPU Worker Configuration

### How workers are assigned

The master creates:

```python
solve_on = ["CPU"] * cpu_workers + ["GPU"] * gpu_workers
```

and broadcasts that list to the worker ranks. Each worker imports the CPU or
GPU solve module accordingly.

### CUDA detection

If `gpu_workers > 0`, `initialize_workers` tries:

```python
import ngsolve.ngscuda
```

If that fails, the code prints a warning and resets `gpu_workers` to zero.

### Practical scaling guidance

- CPU workers scale well when many independent batches exist.
- GPU workers help only when local solves are large enough to amortize device
  setup and data movement.
- Mixed CPU and GPU configurations are supported by the worker protocol.
- Memory use grows with worker count because every worker receives the broadcast
  model data and creates its own local meshes and FE objects.

## 9.8 Internal Geometry Window and Meshing Knobs

These are not exposed as public API parameters today, but they are important
internal controls if you are modifying the mesh backends.

### `active_geometry_window`

Where it appears:

- `SelectNetgenDataRange(..., active_geometry_window=0.999)`
- `SelectGmshFormationDataRange(..., active_geometry_window=0.99)`
- `SelectGmshDataRange(..., active_geometry_window=0.99)`

Role:

- slightly shrinks the effective geometry-selection radius relative to the full
  `domain_radius`
- prevents extremely thin slivers and tiny wedges at the edge of the local
  domain
- decides whether marginal formation features are kept or dropped from the local
  mesh

Practical effect:

- larger values closer to `1.0` keep more edge geometry but risk tiny edge
  regions
- smaller values are more conservative and can remove geometry that would only
  barely touch the domain

When to adjust it:

- if a custom model produces many edge slivers during meshing
- if a feature near the domain edge is physically important and is being clipped
  too aggressively
- when experimenting with new mesh backends or more pathological geometries

### Hardcoded Netgen mesh controls

The Netgen builder currently hardcodes:

```text
mesh_size_min = 0.001
mesh_size_max = 10
mesh_density = "moderate"
```

Meaning:

- `mesh_size_min`: minimum local size near current-electrode source points
- `mesh_size_max`: upper bound for generated element size
- `mesh_density`: preset passed into `SplineGeometry.GenerateMesh`

When to adjust them:

- reduce `mesh_size_min` if the source zone is still too coarse
- reduce `mesh_size_max` if the far field is too coarse for your accuracy needs
- increase `mesh_size_max` only when you have evidence the far field is over-
  resolved and runtime matters more than conservative accuracy

### Hardcoded Gmsh algorithm choices

The Gmsh builders currently hardcode:

- algorithm `6` for 2D meshes
- algorithm `5` for 3D meshes

These are selected through:

```python
gmsh.option.setNumber("Mesh.Algorithm", 6)  # 2D
gmsh.option.setNumber("Mesh.Algorithm", 5)  # 3D
```

When to adjust them:

- if a geometry repeatedly fails in one meshing algorithm but succeeds in
  another
- if you are benchmarking alternative Gmsh meshing strategies for speed or
  robustness
- when you introduce new geometry types that behave poorly under the current
  defaults

Any change here should be validated against the benchmark models before being
kept as a new default.

## See Also

- [`mesh-generation.md`](mesh-generation.md#41-selectgmshboreholedatarange): where the internal geometry-window and mesh-size decisions are applied.
- [`solver.md`](solver.md#66-cg-convergence-behavior): how these runtime settings ultimately affect solve behavior.
- [`model-api.md`](model-api.md#314-simulate_logs): the master-side method that consumes the public parameters.
- [`testing-and-validation.md`](testing-and-validation.md#152-creating-new-validation-tests): how to validate any tuning change against reference cases.
