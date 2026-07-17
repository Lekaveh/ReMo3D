# -*- coding: utf-8 -*-
"""Tool geometry and apparent-resistivity extraction for the GPU solver.

Reuses remo3d.sensitivity._parse_tool for the electrode layout and geometric
factor, then re-expresses everything relative to the CURRENT electrode,
because the solver centres its domain on the current electrode depth — the
same convention as the NGSolve pipeline (remo3d.py `depth_shift`, which makes
the grid identical for every measurement depth of a tool, enabling vmap).

Conventions (identical to the normalized single-current-electrode form used
by Model.set_tools_parameters with force_single_electrode_configuration=True):

  * exactly one current electrode C (A or B) with unit source weight +1;
  * two measuring electrodes M, N on the axis;
  * R_a = K * |U(z_M) - U(z_N)|  with  K = |4*pi*CM*CN / (CN - CM)|
    (worker.py:186-190).
"""

import numpy as np

from ..sensitivity import _parse_tool
from .grid import node_index


def tool_config(tool):
    """Parse ``tool`` into current-electrode-relative geometry.

    Returns dict:
      dz_M, dz_N   : measuring-electrode z offsets from the current electrode
      K            : geometric factor
      depth_shift  : simulation_depth = measurement_depth + depth_shift
                     (offset of the current electrode from the measurement point)
      span         : max |electrode offset| (for domain sizing)
    """
    info = _parse_tool(tool)
    dz_M = info["z_M"] - info["z_C"]
    dz_N = info["z_N"] - info["z_C"]
    offsets = sorted([0.0, dz_M, dz_N])
    min_gap = min(b - a for a, b in zip(offsets, offsets[1:]))
    return {
        "dz_M": float(dz_M),
        "dz_N": float(dz_N),
        "K": float(info["K"]),
        "depth_shift": float(info["z_C"]),
        "span": float(max(abs(dz_M), abs(dz_N))),
        "min_gap": float(min_gap),
    }


def default_h_min(cfg, coarse=0.01, fine=0.0025, gap_fraction=20.0):
    """Grid spacing heuristic tied to the tool's closest electrode pair.

    The apparent resistivity is K * (U_M - U_N); when M and N sit close
    together (e.g. the 0.1 m pair of normal tools) the potential difference is
    small and its discretization error is controlled by the resolution
    BETWEEN the electrodes. min_gap/gap_fraction puts ~gap_fraction cells
    across the closest pair, clipped to [fine, coarse]. Validated on Ex1:
    M1.0A0.1B at 23 m goes from 2.1% error at h=0.01 to <0.2% at the
    heuristic h=0.005 (see benchmark_data/gpu_solver/gpu_validation_ex1.csv).
    """
    return float(np.clip(cfg["min_gap"] / gap_fraction, fine, coarse))


def electrode_nodes(cfg, z_center, z_nodes):
    """Node indices on the axis for source and measuring electrodes.

    z_center is the simulation depth (absolute z of the current electrode).
    All three coordinates must be exact grid nodes — build_grid takes them as
    foci, so this only fails on a mis-built grid.
    """
    j_C = node_index(z_center, z_nodes)
    j_M = node_index(z_center + cfg["dz_M"], z_nodes)
    j_N = node_index(z_center + cfg["dz_N"], z_nodes)
    return j_C, j_M, j_N


def apparent_resistivity(u, cfg, j_M, j_N):
    """R_a = K * |U_M - U_N| sampled on the axis column (i = 0)."""
    dU = u[..., j_M, 0] - u[..., j_N, 0]
    return cfg["K"] * np.abs(np.asarray(dU))
