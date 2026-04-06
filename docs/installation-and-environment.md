# Installation, Dependencies, and Environment

## 13.1 Dependency Roles

The repository directly depends on or imports:

- `numpy`: array math and geometry bookkeeping
- `scipy.interpolate`: borehole interpolation and plotting interpolation
- `matplotlib`: output plots
- `mpi4py`: worker spawning and communication
- `gmsh`: 2D and 3D OCC-based mesh generation
- `netgen`: native 2D mesh generation and mesh container types
- `ngsolve`: finite-element assembly, grid functions, CG solver
- `ngsolve.ngscuda`: optional CUDA acceleration for the GPU solver

`setup.py` lists:

- `numpy`
- `scipy`
- `matplotlib`
- `mpi4py`
- `gmsh`

but the runtime also requires a working Netgen/NGSolve installation for the main
numerical path.

## 13.2 MPI Runtime Requirements

The README recommends:

- MPICH on Linux
- Microsoft MPI on Windows

The code uses `MPI.COMM_WORLD.Spawn`, so the environment must support process
spawning from Python through the selected MPI runtime.

Typical launch pattern for examples:

```text
mpiexec python Examples/Example_01/Example_01.py
```

Practical requirements:

- the same Python environment must be visible to the spawned workers
- `mpi4py` must be built against the MPI runtime available on the machine
- spawned worker processes must be able to import the local `remo3d` package

## 13.3 GPU Setup

GPU support is optional and depends on two things:

1. a CUDA-capable device
2. an NGSolve build that exposes `ngsolve.ngscuda`

Detection in the code is simple:

```python
try:
    import ngsolve.ngscuda
except:
    gpu_workers = 0
```

If that import fails, the run falls back to CPU workers only.

Troubleshooting checklist:

- verify that `ngsolve.ngscuda` imports in the target Python environment
- verify that the CUDA driver stack is available on the machine
- start with `gpu_workers=1` and confirm the worker imports the GPU solver path

## 13.4 Platform Notes

The README states that the package was tested on:

- Ubuntu 18.04
- Ubuntu 20.04
- Windows 10 Pro
- Windows 11 Pro

### Linux notes

- use an MPI runtime compatible with `mpi4py`
- the README uses `pip3 install ...`
- MPICH is the recommended MPI runtime in the project docs

### Windows notes

- use Microsoft MPI according to the README
- the README uses `pip install ...`
- spawned worker processes rely on the current Python executable and local file
  paths resolving correctly on the machine

### Cross-platform caveats

- the numerical stack is heavier than what `setup.py` alone suggests because the
  solver path also needs Netgen and NGSolve
- GPU acceleration adds another layer of platform-specific setup and is best
  treated as optional rather than assumed
## See Also
- [README.md](README.md): top-level project summary and entry points.
- [examples-and-tutorials.md](examples-and-tutorials.md#143-quick-start-with-inline-arrays): quick way to validate the environment once dependencies are installed.
- [parallel-execution.md](parallel-execution.md#72-mpi-communication-protocol): MPI behavior that depends on the runtime environment.
