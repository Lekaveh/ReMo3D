# Changelog, Version History, and Conventions

## 16.1 Expanded Changelog

### v1.0.0

- initial release described in the publication referenced in `Changelog.md`
- established the core workflow: tool parsing, local meshing, FEM solve, and
  synthetic resistivity-log generation

### v1.1.0

- split the code into `remo3d.py`, `gmsh_functions.py`, `netgen_functions.py`,
  and `ngsolve_functions.py`
- added batch mode so adjacent simulation depths can share one local task
- added the single-electrode optimization so compatible tool suites can share
  one solve per current-electrode depth
- added the changelog file itself

### v1.2.0

- standardized the output format of the Gmsh and Netgen paths
- moved conversion into NGSolve format into the worker, keeping the mesh
  generators focused on geometry and mesh construction

### v1.3.0

- converted the main API into the `Model` class
- separated worker initialization, simulation, and shutdown so repeated solves
  fit inversion-style workflows better

## 16.2 API Evolution and Breaking-Change Notes

Main evolution path:

- early versions centered on a more script-like flow
- the current repository centers on the `Model` class and its helper methods
- worker lifecycle control is now explicit rather than hidden in one function

Important practical changes across versions:

- the modular file split in v1.1.0 changed where developers modify meshing and
  solver behavior
- the class-based API in v1.3.0 made long-lived model objects possible
- the separate worker lifecycle in v1.3.0 is the key change for inversion loops

### Metadata note

There is a version-metadata mismatch in the current repository:

- `setup.py` reports `1.4.0`
- `remo3d/__init__.py` reports `1.1.0`
- `Changelog.md` currently stops at `1.3.0`

That mismatch is worth resolving separately from the documentation pass.

## 17.1 Coding Conventions

Patterns used in the codebase:

- arrays carry most structured geometry data instead of custom classes
- depth is the primary ordering dimension and is usually stored in ascending
  order
- region-dependent conductivity is represented as a simple ordered list
- `ValueError` is the main validation exception type in the public API

Naming patterns:

- `*_geometry` for geometric arrays
- `*_parameters` for arrays that still carry both geometry and property values
- `sigma` for conductivity
- `tool_geometry` and `source_terms` for the per-solve tool representation

## 17.2 Coordinate System Conventions

### 2D path

The axisymmetric solver and mesh generators use:

- `x`: radial distance from the borehole axis
- `y`: depth along the borehole axis

Sources and measurement points are evaluated on the axis with `x = 0`.

### 3D path

The 3D worker evaluates sources and measurement points as:

```python
mesh(0.0, 0.0, z)
```

so the borehole axis is the `z` axis in the solve stage.

### Local-depth convention

Before local meshing, borehole and formation data are shifted so the current
simulation depth is `0`. That means most local geometry arrays are stored in a
solve-centered coordinate frame rather than in absolute model depth.

## 17.3 Error Handling Patterns

Validation errors in the public API mainly raise `ValueError`, for example when:

- tool strings are malformed
- formation geometry is not contiguous
- borehole depths are not strictly increasing
- resistivities are non-positive
- dip is outside `[0, 90)`
- unsupported mesh-generator combinations are requested

The worker follows a different policy:

- broad `try/except`
- no re-raise
- affected measurements converted to `np.nan`

That split makes the API strict before the solve starts but fault-tolerant once
parallel task execution is underway.
