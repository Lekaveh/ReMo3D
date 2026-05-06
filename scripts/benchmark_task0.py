"""Task 0 benchmark harness for static-condensation validation.

The harness runs small 2D Netgen/NGSolve models directly, bypassing MPI
workers so solver exceptions surface immediately. For each case it compares the
validated condensed solve sequence with a non-condensed baseline and records
timings, DOF counts, solver stats, and apparent resistivity values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REMO3D_DIR = ROOT / "remo3d"
if str(REMO3D_DIR) not in sys.path:
    sys.path.insert(0, str(REMO3D_DIR))

try:
    import numpy as np
    import ngsolve as ngs
    import netgen_functions as ngf
    import ngsolve_functions as ngsf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Task 0 benchmarks require NumPy and Netgen/NGSolve in the active Python "
        f"environment. Missing module: {exc.name}"
    ) from exc


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    description: str
    formation_parameters: List[List[float]]
    borehole_model: List[List[float]]
    simulation_depth: float
    domain_radius: float
    tool_geometry: List[float]
    source_terms: List[float]
    geometric_factor: float
    expected_apparent: Optional[float] = None


@dataclass
class RunResult:
    case: str
    condense: bool
    apparent_resistivity: List[float]
    timings: Dict[str, float]
    dofs_total: Optional[int]
    dofs_free: Optional[int]
    cg_iterations: Optional[float]
    final_residual_norm: Optional[float]


def reference_cases() -> List[BenchmarkCase]:
    """Return small 2D reference models for Task 0 validation."""
    spacing = 0.5
    tool_geometry = [0.0, spacing]
    source_terms = [1.0, 0.0]
    geometric_factor = 4.0 * math.pi * spacing

    return [
        BenchmarkCase(
            name="homogeneous_10ohmm",
            description=(
                "Homogeneous medium with mud and formation both at 10 ohm-m. "
                "The apparent resistivity should be close to 10 ohm-m away "
                "from the finite outer boundary."
            ),
            formation_parameters=[[0.0, 40.0, math.nan, math.nan, 10.0]],
            borehole_model=[[0.0, 0.10, 10.0], [40.0, 0.10, 10.0]],
            simulation_depth=20.0,
            domain_radius=10.0,
            tool_geometry=tool_geometry,
            source_terms=source_terms,
            geometric_factor=geometric_factor,
            expected_apparent=10.0,
        ),
        BenchmarkCase(
            name="two_layer_10_100ohmm",
            description=(
                "Two-layer model with a 10/100 ohm-m boundary near the tool. "
                "This is a regression reference for layer-boundary response; "
                "the non-condensed solve is the numerical baseline."
            ),
            formation_parameters=[
                [0.0, 20.0, math.nan, math.nan, 10.0],
                [20.0, 40.0, math.nan, math.nan, 100.0],
            ],
            borehole_model=[[0.0, 0.10, 10.0], [40.0, 0.10, 10.0]],
            simulation_depth=19.75,
            domain_radius=10.0,
            tool_geometry=tool_geometry,
            source_terms=source_terms,
            geometric_factor=geometric_factor,
            expected_apparent=None,
        ),
    ]


def _array(data: Iterable[Iterable[float]]) -> np.ndarray:
    return np.asarray(list(data), dtype=float)


def run_case(case: BenchmarkCase, condense: bool, preconditioner: str, fe_order: int) -> RunResult:
    formation_parameters = _array(case.formation_parameters)
    borehole_model = _array(case.borehole_model)
    tool_geometry = np.asarray(case.tool_geometry, dtype=float)
    source_terms = np.asarray(case.source_terms, dtype=float)

    timings: Dict[str, float] = {}

    start = time.perf_counter()
    mud_resistivity = float(np.interp(case.simulation_depth, borehole_model[:, 0], borehole_model[:, 2]))
    local_formation, local_borehole, sigma = ngf.SelectNetgenDataRange(
        borehole_model[:, :2],
        formation_parameters,
        mud_resistivity,
        case.simulation_depth,
        case.domain_radius,
    )
    mesh_data = ngf.ConstructNetgen2dModel(
        case.domain_radius,
        tool_geometry,
        source_terms,
        local_formation,
        local_borehole,
    )
    timings["mesh_generation"] = time.perf_counter() - start

    start = time.perf_counter()
    mesh = ngs.Mesh(mesh_data)
    sigma_cf = ngs.CoefficientFunction(sigma)
    timings["ngsolve_conversion"] = time.perf_counter() - start

    fes, gfu, solver_metrics = ngsf.SolveBVP(
        mesh,
        sigma_cf,
        tool_geometry,
        source_terms,
        [2],
        preconditioner,
        condense,
        order=fe_order,
        return_metrics=True,
    )
    timings.update(solver_metrics.get("timings", {}))

    start = time.perf_counter()
    measuring_electrodes = tool_geometry[source_terms == 0.0]
    apparent = [
        float(abs(case.geometric_factor * gfu(mesh(0.0, float(electrode_z)))))
        for electrode_z in measuring_electrodes
    ]
    timings["evaluation"] = time.perf_counter() - start

    return RunResult(
        case=case.name,
        condense=condense,
        apparent_resistivity=apparent,
        timings=timings,
        dofs_total=solver_metrics.get("dofs_total"),
        dofs_free=solver_metrics.get("dofs_free"),
        cg_iterations=solver_metrics.get("cg_iterations"),
        final_residual_norm=solver_metrics.get("final_residual_norm"),
    )


def _relative_error(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(baseline), 1.0e-30)
    return np.abs(candidate - baseline) / scale


def run_benchmarks(tolerance: float, preconditioner: str, fe_order: int) -> Dict:
    cases = reference_cases()
    results: dict = {
        "tolerance_relative": tolerance,
        "preconditioner": preconditioner,
        "fe_order": fe_order,
        "cases": [],
        "passed": True,
    }

    for case in cases:
        baseline = run_case(case, condense=False, preconditioner=preconditioner, fe_order=fe_order)
        condensed = run_case(case, condense=True, preconditioner=preconditioner, fe_order=fe_order)

        baseline_values = np.asarray(baseline.apparent_resistivity)
        condensed_values = np.asarray(condensed.apparent_resistivity)
        if not np.all(np.isfinite(baseline_values)):
            raise RuntimeError(f"{case.name}: non-condensed baseline produced NaN or Inf")
        if not np.all(np.isfinite(condensed_values)):
            raise RuntimeError(f"{case.name}: condensed solve produced NaN or Inf")

        rel_error = _relative_error(condensed_values, baseline_values)
        max_rel_error = float(np.max(rel_error)) if rel_error.size else 0.0
        case_passed = max_rel_error <= tolerance

        expected_error = None
        if case.expected_apparent is not None:
            expected = np.full_like(condensed_values, case.expected_apparent, dtype=float)
            expected_error = float(np.max(_relative_error(condensed_values, expected)))

        results["cases"].append(
            {
                "name": case.name,
                "description": case.description,
                "passed": case_passed,
                "max_relative_error_condensed_vs_baseline": max_rel_error,
                "max_relative_error_vs_expected": expected_error,
                "baseline_condense_false": asdict(baseline),
                "candidate_condense_true": asdict(condensed),
            }
        )
        results["passed"] = bool(results["passed"] and case_passed)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-3,
        help="Relative tolerance for condensed vs non-condensed apparent resistivity.",
    )
    parser.add_argument(
        "--preconditioner",
        default="multigrid",
        help="NGSolve preconditioner name passed to SolveBVP.",
    )
    parser.add_argument(
        "--fe-order",
        type=int,
        default=3,
        help="Polynomial order of the H1 finite element space.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for baseline and timing data.",
    )
    args = parser.parse_args()

    if args.fe_order < 1:
        parser.error("--fe-order must be a positive integer")

    results = run_benchmarks(args.tolerance, args.preconditioner, args.fe_order)
    text = json.dumps(results, indent=2)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + os.linesep, encoding="utf-8")

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
