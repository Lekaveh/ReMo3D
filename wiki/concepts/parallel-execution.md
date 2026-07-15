---
title: Parallel Execution (Master/Worker)
type: concept
tags: [mpi, parallel, sec, batching, load-balancing, method]
sources: [repo-docs]
updated: 2026-07-15
---

# Parallel Execution

*(traces to `[[repo-docs]]` — parallel-execution.md, architecture.md,
performance-guide.md, model-api.md.)*

ReMo3D distributes the many independent per-depth solves across MPI worker
processes spawned by the master (`MPI.COMM_WORLD.Spawn`). The master validates,
batches, and gathers; workers do geometry clipping, [meshing](mesh-generation.md),
[solving](fem-solver.md), and apparent-resistivity evaluation.

## Master/worker protocol

- **Global setup**: `comm.bcast(solve_on)` (CPU vs GPU per rank) + `barrier`.
- **Per-`simulate_logs` config broadcast**: array shapes first, then raw arrays
  (`Bcast`), then scalars/containers (`bcast`) — formation, borehole, mud
  resistivities, simulation depths, dip, `tools_parameters`, `domain_radius`,
  `mesh_generator`, `preconditioner`, `condense`, `fe_order`, `task_list`.
- **Result collection**: `comm.gather(root=0)` after the inner task loop.
- Two nested lifecycles: outer per `simulate_logs` call, inner per task; a shared
  `StopIteration` sentinel ends the inner batch and, at the outer level, the
  worker process.

## Dynamic load balancing

The master does **not** statically pre-partition tasks. A free worker calls
`sendrecv(dest=0)`; the master `recv(source=MPI.ANY_SOURCE)` then
`send(dest=Get_source())` the next task index. Because per-task cost varies
enormously (3D ≫ 2D, conductivity contrasts, batch contents), pull-based dispatch
beats equal static partitioning by reducing tail effects.

## Two reuse optimizations

### Single-Electrode Computation (SEC)
One BVP solve per current-electrode depth is **reused** across every compatible
tool/measurement, cutting FE solve count up to ×N for N compatible tools — the
highest-impact optimization. `force_single_electrode_configuration=True`
(default) rewrites 3-electrode dual-current tools via the role swap
`ABMN → MNAB`. This rewrite is **exact** — it rests on electrode reciprocity and
a recomputed [geometric factor](resistivity-logging.md#geometric-factors), not
numerical fitting. Enabled only when every tool has one active current electrode.

### Batching
Adjacent simulation depths are grouped into one task so a single mesh is reused;
each measurement is evaluated on the batch mesh at its own `simulation_offset`
from the batch-mean depth. Unlike SEC this is an **approximate** geometric shift:
for even spacing `dz` and batch `B`, the max offset is `|Δz|ₘₐₓ = (B−1)·dz/2`.

> SEC is exact; batching trades fidelity for speed. Keep this distinction — it
> matters for [validation](validation.md) and [sensitivity](../findings/).

## Fault tolerance

Each task is wrapped in a broad `try/except`; any failure (clipping, meshing,
conversion, solve, evaluation) yields `np.nan` for that task's measurements, so a
run completes despite partial failures — but can mask the first root cause.
`REMO3D_WORKER_DEBUG=1` re-raises instead.

## Links

- [performance & accuracy](performance-and-accuracy.md) (CPU/GPU knobs) · [FEM solver](fem-solver.md) · [mesh generation](mesh-generation.md).
- Code: [`../../docs/parallel-execution.md`](../../docs/parallel-execution.md).
