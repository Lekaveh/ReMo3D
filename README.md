# ReMo3D

ReMo3D is a Python package for generating synthetic normal and lateral
resistivity logs in 2D axisymmetric and 3D dipping models. The workflow couples
local mesh generation with finite-element solves and distributed-memory worker
processes.

## Documentation

The main documentation entry point is [`docs/README.md`](docs/README.md).

Key pages:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/mathematical-foundations.md`](docs/mathematical-foundations.md)
- [`docs/model-api.md`](docs/model-api.md)
- [`docs/mesh-generation.md`](docs/mesh-generation.md)
- [`docs/parallel-execution.md`](docs/parallel-execution.md)
- [`docs/examples-and-tutorials.md`](docs/examples-and-tutorials.md)

## Installation

The project README and source code indicate the following runtime stack:

- Python 3.7 or newer
- MPI runtime
- Gmsh
- Netgen / NGSolve
- optional NGSolve CUDA support for GPU workers

The original project README recommends:

- MPICH on Linux
- Microsoft MPI on Windows

Example install commands from the original project notes:

### Linux

```text
pip3 install git+https://github.com/eMWu94/ReMo3D.git
```

### Windows

```text
pip install git+https://github.com/eMWu94/ReMo3D.git
```

## Expected Computation Times

The original project notes state that simulating 100 measurement points for a
single logging tool on a moderate 2D model takes roughly 15 to 30 seconds on an
AMD Ryzen 2600 class CPU, while a moderate 3D model takes roughly 15 to 30
minutes.

## Licensing Information

- Code: GNU General Public License v2.1
- Repository data and supporting materials: CC BY 4.0

## Funding

The research was funded by the National Science Centre, Poland, grant number
2020/37/N/ST10/03230.
