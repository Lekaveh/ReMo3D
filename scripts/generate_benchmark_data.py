"""Synthetic pseudowell + resistivity-log generator for benchmarking.

This is a reorganized, runnable version of the original ``data_generation``
prototype. It builds random layered formations with a Markov-chain facies
model, decorates each layer with formation/invasion/borehole properties, runs
ReMo3D's forward solver to synthesize apparent-resistivity logs for a suite of
tools, and saves each realization as an ``.npz`` file (optionally with a
figure).

Pipeline per sample
-------------------
1. ``simulate_pseudowell_grid``   -> layered facies column (shale/sand) on a DZ grid
2. borehole track (CALM, RM)      -> smoothed caliper + mud-resistivity logs
3. ``add_formation_properties``   -> per-layer resistivities + invasion zones
4. ``upsample_formation_properties`` (optional, ``use_blocks``) -> per-cell noise
5. ``Model.simulate_logs``        -> apparent-resistivity logs for each tool
6. save ``sample_<n>.npz`` (+ optional formation/log figure)

Usage
-----
    # from the repo root, with the ReMo3D conda env active:
    python scripts/generate_benchmark_data.py --n-samples 10 --plot-count 2

Run ``--help`` for all options. The ReMo3D package is loaded from this repo's
source folder (v1.4.0), never the copy installed in the environment.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Load the repo's remo3d source (v1.4.0), not any installed copy -----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from remo3d_loader import load_remo3d  # noqa: E402

remo3d = load_remo3d("folder", quiet=True)
Model = remo3d.Model

DEFAULT_TOOLS = ["A0.4M0.1N", "A1.0M0.1N", "A2.0M0.5N", "A4.0M0.5N", "A8.0M1.0N"]
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark_data"


def _ensure_mpi_launcher():
    """Re-exec under ``mpiexec -n 1`` if we are not already under a process manager.

    ``Model.initialize_workers`` calls ``MPI.COMM_WORLD.Spawn``. Launched as a
    bare ``python script.py`` (MPI singleton init), that Spawn deadlocks on this
    host — its transient ``mpiexec`` helper dies and the parent hangs forever in
    ``accept()``. Launching under a real ``mpiexec -n 1`` gives Spawn a working
    process manager. This guard makes ``python scripts/generate_benchmark_data.py``
    behave the same as ``mpiexec -n 1 python scripts/generate_benchmark_data.py``.
    """
    if os.environ.get("REMO3D_MPI_RELAUNCHED") == "1":
        return  # already relaunched by us
    if os.environ.get("PMI_RANK") is not None or os.environ.get("PMI_SIZE") is not None:
        return  # already under mpiexec/hydra

    import shutil

    mpiexec = Path(sys.executable).with_name("mpiexec")
    mpiexec_path = str(mpiexec) if mpiexec.exists() else shutil.which("mpiexec")
    if not mpiexec_path:
        print("[warn] mpiexec not found; running singleton (Spawn may hang).", flush=True)
        return

    env = dict(os.environ, REMO3D_MPI_RELAUNCHED="1")
    argv = [mpiexec_path, "-n", "1", sys.executable, *sys.argv]
    print(f"[main] relaunching under: {' '.join(argv[:4])} ...", flush=True)
    os.execve(mpiexec_path, argv, env)


# ==========================================================================
# Formation sequence simulation
# ==========================================================================


def draw_cells(rng, lam, DZ, eps):
    """Sample a layer thickness and convert it to a positive cell count."""
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"lam must be positive and finite, got {lam!r}")
    if not np.isfinite(DZ) or DZ <= 0:
        raise ValueError(f"DZ must be positive and finite, got {DZ!r}")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")

    thickness = max(float(rng.exponential(scale=lam)), float(eps))
    n_cells = max(1, math.ceil(thickness / DZ))
    return n_cells, n_cells * DZ


def enforce_last_shale(fac_idx, mic_idx, facies):
    """Return copies of the facies and microlayer arrays with a shale bottom interval."""
    fac_idx = np.asarray(fac_idx, dtype=np.int16).copy()
    mic_idx = np.asarray(mic_idx, dtype=np.int32).copy()

    if fac_idx.ndim != 1 or mic_idx.ndim != 1:
        raise ValueError("fac_idx and mic_idx must be 1D arrays")
    if fac_idx.shape != mic_idx.shape:
        raise ValueError("fac_idx and mic_idx must have the same shape")
    if fac_idx.size == 0:
        return fac_idx, mic_idx

    facies = list(facies)
    fac2code = {f: i for i, f in enumerate(facies)}
    if "shale" not in fac2code:
        raise ValueError("facies must contain 'shale'")
    if "sand" not in fac2code:
        raise ValueError("facies must contain 'sand'")

    shale_code = fac2code["shale"]
    sand_code = fac2code["sand"]

    last_mic = int(mic_idx[-1])
    if last_mic < 0:
        return fac_idx, mic_idx

    mask_last = mic_idx == last_mic
    n_last = int(mask_last.sum())
    if n_last == 0:
        return fac_idx, mic_idx

    last_code = int(fac_idx[mask_last][0])
    if last_code == shale_code:
        return fac_idx, mic_idx

    if last_code != sand_code:
        fac_idx[mask_last] = shale_code
        return fac_idx, mic_idx

    n_shale = max(1, math.ceil(0.25 * n_last))
    n_sand_remain = n_last - n_shale
    if n_sand_remain < math.ceil(0.5 * n_last):
        fac_idx[mask_last] = shale_code
        return fac_idx, mic_idx

    last_indices = np.flatnonzero(mask_last)
    shale_tail_idx = last_indices[-n_shale:]
    fac_idx[shale_tail_idx] = shale_code
    mic_idx[shale_tail_idx] = last_mic + 1
    return fac_idx, mic_idx


def build_interval_table_from_grid(depth, fac_idx, mic_idx, facies, DZ):
    """Rebuild an interval table from the cell-based representation."""
    depth = np.asarray(depth, dtype=float)
    fac_idx = np.asarray(fac_idx, dtype=np.int16)
    mic_idx = np.asarray(mic_idx, dtype=np.int32)

    if depth.ndim != 1:
        raise ValueError("depth must be a 1D array")
    if not (len(depth) == len(fac_idx) == len(mic_idx)):
        raise ValueError("depth, fac_idx, and mic_idx must have the same length")
    if len(depth) == 0:
        return pd.DataFrame(
            columns=["TOP", "BOTTOM", "microlayer", "thickness", "microlayer_id"]
        )
    if not np.isfinite(DZ) or DZ <= 0:
        raise ValueError(f"DZ must be positive and finite, got {DZ!r}")

    rows = []
    i = 0
    while i < len(depth):
        cur_f = int(fac_idx[i])
        cur_m = int(mic_idx[i])
        if cur_f < 0 or cur_f >= len(facies):
            raise ValueError(f"Invalid facies code at cell {i}: {cur_f}")
        if cur_m < 0:
            raise ValueError(f"Invalid microlayer id at cell {i}: {cur_m}")

        j = i + 1
        while j < len(depth) and fac_idx[j] == cur_f and mic_idx[j] == cur_m:
            j += 1

        top_i = float(depth[i])
        bottom_i = float(depth[j - 1] + DZ)
        rows.append((top_i, bottom_i, facies[cur_f], bottom_i - top_i, cur_m))
        i = j

    return pd.DataFrame(
        rows,
        columns=["TOP", "BOTTOM", "microlayer", "thickness", "microlayer_id"],
    )


def simulate_pseudowell_grid(
    P_dict: pd.DataFrame,
    lam_d: Mapping[str, float],
    facies: Sequence[str],
    top: float,
    base: float,
    DZ: float = 1.0,
    eps: float = 0.5,
    seed=None,
    initial_state: str = "shale",
    enforce_shale: bool = True,
):
    """Simulate a layered facies column on a regular DZ grid via a Markov chain."""
    if not np.isfinite(top) or not np.isfinite(base) or base <= top:
        raise ValueError(f"Expected base > top with finite values, got top={top!r}, base={base!r}")
    if not np.isfinite(DZ) or DZ <= 0:
        raise ValueError(f"DZ must be positive and finite, got {DZ!r}")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")

    facies = list(facies)
    if not facies:
        raise ValueError("facies must be a non-empty sequence")
    if len(set(facies)) != len(facies):
        raise ValueError("facies entries must be unique")
    if initial_state not in facies:
        raise ValueError(f"initial_state must be one of {facies}, got {initial_state!r}")

    missing_lam = [f for f in facies if f not in lam_d]
    if missing_lam:
        raise ValueError(f"lam_d is missing facies keys: {missing_lam}")
    for fac in facies:
        lam = lam_d[fac]
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(f"lam_d[{fac!r}] must be positive and finite, got {lam!r}")

    try:
        P_dict = P_dict.loc[facies, facies].copy()
    except KeyError as exc:
        raise ValueError("P_dict must contain all facies as both index and columns") from exc

    probs = P_dict.to_numpy(dtype=float)
    if not np.all(np.isfinite(probs)):
        raise ValueError("Transition matrix must be finite")
    if np.any(probs < 0):
        raise ValueError("Transition matrix cannot contain negative probabilities")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each row of the transition matrix must sum to 1")

    grid_len_float = (base - top) / DZ
    if not np.isclose(grid_len_float, round(grid_len_float), atol=1e-10):
        raise ValueError("(base - top) must be divisible by DZ")

    rng = np.random.default_rng(seed)
    grid_len = int(round(grid_len_float))
    fac2code = {f: i for i, f in enumerate(facies)}

    depth = top + np.arange(grid_len, dtype=float) * DZ
    fac_idx = np.full(grid_len, -1, dtype=np.int16)
    mic_idx = np.full(grid_len, -1, dtype=np.int32)

    j = 0
    mic_id = 0
    rem_cells = grid_len
    state = initial_state

    while rem_cells > 0:
        n_cells, _ = draw_cells(rng, lam_d[state], DZ, eps)
        n_cells = min(n_cells, rem_cells)

        fac_code = fac2code[state]
        sl = slice(j, j + n_cells)
        fac_idx[sl] = fac_code
        mic_idx[sl] = mic_id

        j += n_cells
        rem_cells -= n_cells
        mic_id += 1

        if rem_cells == 0:
            break

        state_probs = P_dict.loc[state, facies].to_numpy(dtype=float)
        state = rng.choice(facies, p=state_probs)

    if enforce_shale:
        fac_idx, mic_idx = enforce_last_shale(fac_idx=fac_idx, mic_idx=mic_idx, facies=facies)

    df_int = build_interval_table_from_grid(
        depth=depth,
        fac_idx=fac_idx,
        mic_idx=mic_idx,
        facies=facies,
        DZ=DZ,
    )
    df_grid = pd.DataFrame(
        {
            "TOP": depth,
            "BOTTOM": depth + DZ,
            "microlayer": [facies[i] for i in fac_idx],
            "thickness": DZ,
            "microlayer_id": mic_idx,
        }
    )
    return df_int, depth, df_grid


# ==========================================================================
# Formation properties (resistivities, invasion zones, borehole coupling)
# ==========================================================================


def _validate_uniform_bounds(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must be a pair of (low, high)")
    low, high = map(float, bounds)
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise ValueError(f"{name} must contain finite values with high >= low, got {bounds!r}")
    return low, high


def loguniform(rng: np.random.Generator, low: float, high: float, size=None) -> np.ndarray:
    """Draw log-uniformly from [low, high]; requires finite 0 < low <= high (sub-unity bounds like 0.5 are fine)."""
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high < low:
        raise ValueError(f"loguniform requires finite 0 < low <= high, got {(low, high)!r}")
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def add_formation_properties(
    df_form: pd.DataFrame,
    df_borehole: pd.DataFrame,
    uniform_shale_form_res: tuple[float, float],
    uniform_sand_form_res: tuple[float, float],
    uniform_inv_radius_delta: tuple[float, float],
    uniform_inv_res: tuple[float, float],
    inv_prob_sand: float = 0.5,
    seed=None,
) -> pd.DataFrame:
    """Attach formation, invasion, and borehole-linked properties to each layer.

    Resistivities (rtuz, rtfz) are drawn log-uniformly within their bounds;
    the invasion radius delta stays uniform. All resistivity lower bounds are
    floored at the well's maximum mud resistivity (RM), so neither the formation
    nor the invaded zone can be less resistive than the mud.
    """
    if not 0.0 <= float(inv_prob_sand) <= 1.0:
        raise ValueError(f"inv_prob_sand must be in [0, 1], got {inv_prob_sand!r}")

    shale_bounds = _validate_uniform_bounds("uniform_shale_form_res", uniform_shale_form_res)
    sand_bounds = _validate_uniform_bounds("uniform_sand_form_res", uniform_sand_form_res)
    inv_radius_bounds = _validate_uniform_bounds("uniform_inv_radius_delta", uniform_inv_radius_delta)
    inv_res_bounds = _validate_uniform_bounds("uniform_inv_res", uniform_inv_res)

    required_form_cols = {"TOP", "BOTTOM", "microlayer"}
    missing_form = required_form_cols - set(df_form.columns)
    if missing_form:
        raise ValueError(f"df_form is missing columns: {sorted(missing_form)}")

    required_bh_cols = {"DEPT", "CALM", "RM"}
    missing_bh = required_bh_cols - set(df_borehole.columns)
    if missing_bh:
        raise ValueError(f"df_borehole is missing columns: {sorted(missing_bh)}")

    df_out = df_form.copy().reset_index(drop=True)
    if df_out.empty:
        return df_out.assign(
            mid_depth=pd.Series(dtype=float),
            borehole_radius=pd.Series(dtype=float),
            borehole_radius_max=pd.Series(dtype=float),
            rm_mean=pd.Series(dtype=float),
            has_inv_zone=pd.Series(dtype=bool),
            inv_zone_radius_delta=pd.Series(dtype=float),
            rdfz=pd.Series(dtype=float),
            rtfz=pd.Series(dtype=float),
            rtuz=pd.Series(dtype=float),
        )

    top_vals = df_out["TOP"].to_numpy(dtype=float)
    bottom_vals = df_out["BOTTOM"].to_numpy(dtype=float)
    if not np.all(np.isfinite(top_vals)) or not np.all(np.isfinite(bottom_vals)):
        raise ValueError("TOP and BOTTOM must be finite")
    if np.any(bottom_vals <= top_vals):
        raise ValueError("Each layer must satisfy BOTTOM > TOP")

    df_bh = df_borehole.sort_values("DEPT").reset_index(drop=True).copy()
    if df_bh.empty:
        raise ValueError("df_borehole must contain at least one sample")

    dept = df_bh["DEPT"].to_numpy(dtype=float)
    calm = df_bh["CALM"].to_numpy(dtype=float)
    rm = df_bh["RM"].to_numpy(dtype=float)
    if not np.all(np.isfinite(dept)) or not np.all(np.isfinite(calm)) or not np.all(np.isfinite(rm)):
        raise ValueError("DEPT, CALM, and RM must be finite")

    rng_master = np.random.default_rng(seed)
    child_seeds = rng_master.integers(0, np.iinfo(np.uint64).max, size=5, dtype=np.uint64)
    rng_inv_flag = np.random.default_rng(int(child_seeds[0]))
    rng_shale_rtuz = np.random.default_rng(int(child_seeds[1]))
    rng_sand_rtuz = np.random.default_rng(int(child_seeds[2]))
    rng_inv_delta = np.random.default_rng(int(child_seeds[3]))
    rng_inv_res = np.random.default_rng(int(child_seeds[4]))

    def interval_mean_or_nearest(values: np.ndarray, top: float, bottom: float) -> float:
        mask = (dept >= top) & (dept < bottom)
        if np.any(mask):
            return float(np.mean(values[mask]))
        mid = 0.5 * (top + bottom)
        idx = int(np.argmin(np.abs(dept - mid)))
        return float(values[idx])

    def interval_max_or_nearest(values: np.ndarray, top: float, bottom: float) -> float:
        mask = (dept >= top) & (dept < bottom)
        if np.any(mask):
            return float(np.max(values[mask]))
        mid = 0.5 * (top + bottom)
        idx = int(np.argmin(np.abs(dept - mid)))
        return float(values[idx])

    facies = df_out["microlayer"].astype(str).str.strip().str.lower().to_numpy()
    shale_mask = facies == "shale"
    sand_mask = facies == "sand"
    unsupported = ~(shale_mask | sand_mask)
    if np.any(unsupported):
        bad = sorted(set(facies[unsupported]))
        raise ValueError(f"Unsupported facies: {bad!r}")

    n_layers = len(df_out)
    mid_depths = 0.5 * (top_vals + bottom_vals)
    borehole_radius = np.empty(n_layers, dtype=float)
    borehole_radius_max = np.empty(n_layers, dtype=float)
    mud_resistivity_mean = np.empty(n_layers, dtype=float)
    for i, (top_i, bottom_i) in enumerate(zip(top_vals, bottom_vals)):
        borehole_radius[i] = interval_mean_or_nearest(calm, float(top_i), float(bottom_i))
        borehole_radius_max[i] = interval_max_or_nearest(calm, float(top_i), float(bottom_i))
        mud_resistivity_mean[i] = interval_mean_or_nearest(rm, float(top_i), float(bottom_i))

    has_inv_zone = np.zeros(n_layers, dtype=bool)
    if np.any(sand_mask):
        has_inv_zone[sand_mask] = rng_inv_flag.random(np.count_nonzero(sand_mask)) < float(inv_prob_sand)

    shale_bounds = (max(shale_bounds[0], rm.max()), shale_bounds[1])
    sand_bounds = (max(sand_bounds[0], rm.max()), sand_bounds[1])
    inv_res_bounds = (max(inv_res_bounds[0], rm.max()), inv_res_bounds[1])

    formation_resistivity = np.full(n_layers, np.nan, dtype=float)
    if np.any(shale_mask):
        formation_resistivity[shale_mask] = loguniform(rng_shale_rtuz, *shale_bounds, size=np.count_nonzero(shale_mask))
    if np.any(sand_mask):
        formation_resistivity[sand_mask] = loguniform(rng_sand_rtuz, *sand_bounds, size=np.count_nonzero(sand_mask))

    inv_zone_radius_delta = np.full(n_layers, np.nan, dtype=float)
    inv_zone_resistivity = np.full(n_layers, np.nan, dtype=float)
    inv_zone_radius = np.full(n_layers, np.nan, dtype=float)

    invaded_mask = sand_mask & has_inv_zone
    if np.any(invaded_mask):
        inv_zone_radius_delta[invaded_mask] = rng_inv_delta.uniform(*inv_radius_bounds, size=np.count_nonzero(invaded_mask))
        inv_zone_resistivity[invaded_mask] = loguniform(rng_inv_res, *inv_res_bounds, size=np.count_nonzero(invaded_mask))
        inv_zone_radius[invaded_mask] = borehole_radius_max[invaded_mask] + inv_zone_radius_delta[invaded_mask]

    df_out["mid_depth"] = mid_depths
    df_out["borehole_radius"] = borehole_radius
    df_out["borehole_radius_max"] = borehole_radius_max
    df_out["rm_mean"] = mud_resistivity_mean
    df_out["has_inv_zone"] = has_inv_zone
    df_out["inv_zone_radius_delta"] = inv_zone_radius_delta
    df_out["rdfz"] = inv_zone_radius
    df_out["rtfz"] = inv_zone_resistivity
    df_out["rtuz"] = formation_resistivity
    return df_out


def _apply_smoothed_relative_noise(
    values: np.ndarray,
    *,
    noise_fraction: float,
    kernel_size: int,
    rng: np.random.Generator,
    clip_min: float | None = None,
    clip_max: float | None = None,
    group_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Apply depth-smoothed Gaussian noise with relative scale to finite values.

    If group_ids is provided, smoothing is applied independently within each group,
    which prevents cross-layer leakage after upsampling.
    """
    if noise_fraction < 0.0:
        raise ValueError(f"noise_fraction must be non-negative, got {noise_fraction!r}")

    kernel_size = int(kernel_size)
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size!r}")
    if kernel_size % 2 == 0:
        kernel_size += 1

    values = np.asarray(values, dtype=float)
    noisy = values.copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        return noisy

    if group_ids is None:
        group_ids = np.zeros(values.shape[0], dtype=np.int64)
    else:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != values.shape:
            raise ValueError(f"group_ids must match values shape, got {group_ids.shape} vs {values.shape}")

    kernel = np.ones(kernel_size, dtype=float)
    kernel /= kernel.sum()

    for group in np.unique(group_ids[finite]):
        group_mask = finite & (group_ids == group)
        group_values = values[group_mask]
        if group_values.size == 0:
            continue

        scale = np.abs(group_values) * float(noise_fraction)
        group_noise = rng.normal(loc=0.0, scale=1.0, size=group_values.size)

        if group_values.size > 1:
            pad = kernel_size // 2
            padded = np.pad(group_noise, pad_width=pad, mode="edge")
            smooth_noise = np.convolve(padded, kernel, mode="valid")
            # rescale by the kernel's exact attenuation; the group's empirical std
            # is heavy-tailed for 2-3 cell groups (uncentered common mode / tiny spread)
            smooth_noise /= np.sqrt(np.sum(kernel ** 2))
        else:
            smooth_noise = group_noise

        noisy[group_mask] = group_values + smooth_noise * scale
        if clip_min is not None:
            noisy[group_mask] = np.clip(noisy[group_mask], clip_min, clip_max)

    return noisy


def upsample_formation_properties(
    df_form: pd.DataFrame,
    df_borehole: pd.DataFrame,
    DZ: float,
    *,
    rdfz_noise_fraction: float = 0.05,
    rtfz_noise_fraction: float = 0.05,
    rtuz_noise_fraction: float = 0.05,
    rdfz_kernel_size: int = 3,
    rtfz_kernel_size: int = 3,
    rtuz_kernel_size: int = 3,
    min_inv_delta: float = 0.05,
    rdfz_bounds=(None, None),
    rtuz_bounds=(0.8, 300),
    rtfz_bounds=(0.5, 300),
    seed=None,
) -> pd.DataFrame:
    """Upsample layers to DZ, keep initial properties, and add property-specific smoothed noise."""
    dz = float(DZ)
    if not np.isfinite(dz) or dz <= 0.0:
        raise ValueError(f"DZ must be positive and finite, got {DZ!r}")
    if min_inv_delta < 0.0:
        raise ValueError(f"min_inv_delta must be non-negative, got {min_inv_delta!r}")

    property_cols = ["rdfz", "rtfz", "rtuz"]
    required_form_cols = {"TOP", "BOTTOM", *property_cols}
    missing_form = required_form_cols - set(df_form.columns)
    if missing_form:
        raise ValueError(f"df_form is missing columns: {sorted(missing_form)}")

    required_bh_cols = {"DEPT", "CALM"}
    missing_bh = required_bh_cols - set(df_borehole.columns)
    if missing_bh:
        raise ValueError(f"df_borehole is missing columns: {sorted(missing_bh)}")

    df_src = df_form.copy().reset_index(drop=True)
    if df_src.empty:
        out = df_src.copy()
        for col in property_cols:
            out[f"{col}_init"] = pd.Series(dtype=float)
        return out

    df_bh = df_borehole.copy().sort_values("DEPT")
    agg_cols = ["CALM"] + (["RM"] if "RM" in df_bh.columns else [])
    df_bh = df_bh.groupby("DEPT", as_index=False)[agg_cols].mean()
    if df_bh.empty:
        raise ValueError("df_borehole must contain at least one sample")

    dept = df_bh["DEPT"].to_numpy(dtype=float)
    calm = df_bh["CALM"].to_numpy(dtype=float)
    if not np.all(np.isfinite(dept)) or not np.all(np.isfinite(calm)):
        raise ValueError("DEPT and CALM must be finite")
    rm = df_bh["RM"].to_numpy(dtype=float) if "RM" in df_bh.columns else None
    if rm is not None and not np.all(np.isfinite(rm)):
        raise ValueError("RM must be finite when provided")

    # resistivity clip floors may not undercut the mud: noise added below must not
    # push rtfz/rtuz beneath RM even when the static bounds allow it
    mud_floor = float(rm.max()) if rm is not None else None

    def _with_mud_floor(clip_min):
        if mud_floor is None:
            return clip_min
        if clip_min is None:
            return mud_floor
        return max(float(clip_min), mud_floor)

    records: list[dict] = []
    for layer_idx, row in df_src.iterrows():
        top_i = float(row["TOP"])
        bottom_i = float(row["BOTTOM"])
        if not np.isfinite(top_i) or not np.isfinite(bottom_i) or bottom_i <= top_i:
            raise ValueError("Each layer must have finite TOP and BOTTOM with BOTTOM > TOP")

        n_cells = max(1, int(np.ceil((bottom_i - top_i) / dz - 1e-12)))
        for cell_idx in range(n_cells):
            cell_top = top_i + cell_idx * dz
            cell_bottom = min(top_i + (cell_idx + 1) * dz, bottom_i)
            if cell_bottom <= cell_top:
                continue

            mid_i = 0.5 * (cell_top + cell_bottom)
            rec = row.to_dict()
            rec["parent_layer_index"] = int(layer_idx)
            rec["TOP"] = cell_top
            rec["BOTTOM"] = cell_bottom
            rec["thickness"] = cell_bottom - cell_top
            rec["mid_depth"] = mid_i
            rec["borehole_radius"] = float(np.interp(mid_i, dept, calm))
            if rm is not None:
                rec["rm_mean"] = float(np.interp(mid_i, dept, rm))
            records.append(rec)

    df_up = pd.DataFrame.from_records(records)
    parent_layer_index = df_up["parent_layer_index"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    noise_specs = {
        "rdfz": {"noise_fraction": float(rdfz_noise_fraction), "kernel_size": int(rdfz_kernel_size), "clip_min": rdfz_bounds[0], "clip_max": rdfz_bounds[1]},
        "rtfz": {"noise_fraction": float(rtfz_noise_fraction), "kernel_size": int(rtfz_kernel_size), "clip_min": _with_mud_floor(rtfz_bounds[0]), "clip_max": rtfz_bounds[1]},
        "rtuz": {"noise_fraction": float(rtuz_noise_fraction), "kernel_size": int(rtuz_kernel_size), "clip_min": _with_mud_floor(rtuz_bounds[0]), "clip_max": rtuz_bounds[1]},
    }
    for col in property_cols:
        init = df_up[col].to_numpy(dtype=float)
        spec = noise_specs[col]
        noisy = _apply_smoothed_relative_noise(
            init,
            noise_fraction=spec["noise_fraction"],
            kernel_size=spec["kernel_size"],
            rng=rng,
            clip_min=spec["clip_min"],
            clip_max=spec["clip_max"],
            group_ids=parent_layer_index,
        )
        df_up[f"{col}_init"] = init
        df_up[col] = noisy

    rdfz = df_up["rdfz"].to_numpy(dtype=float).copy()
    borehole_radius = df_up["borehole_radius"].to_numpy(dtype=float).copy()
    finite_rdfz = np.isfinite(rdfz)
    if np.any(finite_rdfz):
        rdfz[finite_rdfz] = np.maximum(
            rdfz[finite_rdfz],
            borehole_radius[finite_rdfz] + float(min_inv_delta),
        )
        df_up["rdfz"] = rdfz
        if "inv_zone_radius_delta" in df_up.columns:
            df_up["inv_zone_radius_delta"] = np.where(
                finite_rdfz,
                df_up["rdfz"].to_numpy(dtype=float) - borehole_radius,
                np.nan,
            )

    return df_up


# ==========================================================================
# Plotting
# ==========================================================================


def _edges_from_samples(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("z must be a 1D array")
    if z.size == 0:
        raise ValueError("z must contain at least one sample")
    if z.size == 1:
        return np.array([z[0], z[0] + 1.0], dtype=float)

    edges = np.empty(z.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    return edges


def plot_formation_cylindrical_cut(
    ax,
    layers: np.ndarray,
    *,
    borehole: np.ndarray,
    r_max: float | None,
    symmetric: bool = True,
    cmap_name: str = "viridis",
    add_colorbar: bool = True,
    cax=None,
    colorbar_kwargs: dict | None = None,
    title: str = "Formation model",
    r_vmax: float = 20,
    r_vmin: float = 0,
    log_scale: bool = False,
):
    """Plot a radial-depth cut of the formation, invasion zone, and borehole."""
    if ax is None:
        raise ValueError("ax must be provided")

    layers = np.asarray(layers, dtype=float)
    if layers.ndim != 2 or layers.shape[1] != 5:
        raise ValueError("layers must be shape (N, 5): [top, bottom, r_inv, rho_inv, rho_out]")

    top = layers[:, 0]
    bot = layers[:, 1]
    r_inv = layers[:, 2]
    rho_in = layers[:, 3]
    rho_out = layers[:, 4]
    y_min, y_max = float(np.min(top)), float(np.max(bot))

    max_r = 0.0
    if np.any(np.isfinite(r_inv)):
        max_r = max(max_r, float(np.nanmax(r_inv)))

    bh = None
    if borehole is not None:
        bh = np.asarray(borehole, dtype=float)
        if bh.ndim != 2 or bh.shape[1] != 3:
            raise ValueError("borehole must be shape (M, 3): [depth, radius, rho_bh]")
        if bh.size:
            max_r = max(max_r, float(np.nanmax(bh[:, 1])))

    if r_max is None:
        r_max = max(0.5, 1.25 * max_r)
    elif not np.isfinite(r_max) or r_max <= 0:
        raise ValueError(f"r_max must be positive and finite when provided, got {r_max!r}")

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=r_vmin, vmax=r_vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    if symmetric:
        x0, x1 = -r_max, r_max
    else:
        x0, x1 = 0.0, r_max

    for t, b, rinv, rxo, rt in zip(top, bot, r_inv, rho_in, rho_out):
        h = b - t
        if h <= 0:
            continue

        ax.add_patch(Rectangle((x0, t), x1 - x0, h, facecolor=sm.to_rgba(rt), edgecolor="none"))

        if np.isfinite(rinv) and np.isfinite(rxo) and rinv > 0:
            if symmetric:
                ix0, iw = -rinv, 2.0 * rinv
            else:
                ix0, iw = 0.0, rinv
            ax.add_patch(Rectangle((ix0, t), iw, h, facecolor=sm.to_rgba(rxo), edgecolor="none"))

    if bh is not None and bh.size:
        z = bh[:, 0]
        r = bh[:, 1]
        rb = bh[:, 2]
        mask = np.isfinite(z) & np.isfinite(r) & np.isfinite(rb)
        z, r, rb = z[mask], r[mask], rb[mask]

        if z.size:
            order = np.argsort(z)
            z, r, rb = z[order], r[order], rb[order]
            z_edges = _edges_from_samples(z)

            keep = (z_edges[1:] > y_min) & (z_edges[:-1] < y_max)
            if np.any(keep):
                i0 = int(np.argmax(keep))
                i1 = int(np.where(keep)[0][-1])

                z_edges = z_edges[i0 : i1 + 2].copy()
                rb_cell = rb[i0 : i1 + 1].copy()
                z_edges[0] = max(z_edges[0], y_min)
                z_edges[-1] = min(z_edges[-1], y_max)

                r_edge = np.interp(z_edges, z, r, left=r[0], right=r[-1])
                if symmetric:
                    X = np.column_stack([-r_edge, r_edge])
                else:
                    X = np.column_stack([np.zeros_like(r_edge), r_edge])

                Y = np.column_stack([z_edges, z_edges])
                C = rb_cell.reshape(-1, 1)
                ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm, shading="flat", edgecolors="none")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Radial distance [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(title)

    cbar = None
    if add_colorbar:
        colorbar_kwargs = {} if colorbar_kwargs is None else dict(colorbar_kwargs)
        cbar = ax.figure.colorbar(sm, ax=ax, cax=cax, **colorbar_kwargs)
        if not log_scale:
            cbar.set_label("Resistivity [ohm·m]")
        else:
            cbar.set_label("Resistivity [ohm·m] (log scale)")
            cbar.set_ticks(np.geomspace(r_vmin, r_vmax, num=10))
            cbar.set_ticklabels([f"{t:.2g}" for t in np.geomspace(r_vmin, r_vmax, num=10)])

    return sm, cbar


def _plot_sample(formation_model, borehole_model, logs, all_tools, fig_path, sample_num):
    """Render a formation cross-section next to the synthesized tool logs and save it."""
    formation_model_plot = formation_model.copy()
    formation_model_plot[:, 3] = log_resistivity(formation_model_plot[:, 3])
    formation_model_plot[:, 4] = log_resistivity(formation_model_plot[:, 4])
    borehole_model_plot = borehole_model.copy()
    borehole_model_plot[:, 2:] = log_resistivity(borehole_model_plot[:, 2:])

    fig, ax = plt.subplots(ncols=2, figsize=(2 * 3, 10), dpi=150, sharey=True, constrained_layout=True)
    sm, _ = plot_formation_cylindrical_cut(
        ax=ax[0],
        layers=formation_model_plot,
        borehole=borehole_model_plot,
        r_max=1.0,
        symmetric=False,
        cmap_name="viridis",
        add_colorbar=False,
        title="Simulated formation",
        r_vmin=0.0,
        r_vmax=6.0,
        log_scale=True,
    )

    for tool in all_tools:
        res = logs[tool]
        ax[1].plot(res[:, 1], res[:, 0], label=f"{tool}")

    ax[1].set_xlim(1, 200)
    ax[1].set_xscale("log")
    ax[1].grid(which="minor", linestyle="-", linewidth="0.5", color="gray")
    ax[1].grid(which="major", linestyle="-", linewidth="0.5", color="gray")
    ax[1].xaxis.set_major_locator(mticker.FixedLocator([1, 10, 50, 100, 200]))
    ax[1].xaxis.set_major_formatter(mticker.FixedFormatter(["1", "10", "50", "100", "200"]))
    ax[1].set_xlabel("[ohm·m]")
    ax[1].set_title(f"Apparent resistivity (sample {sample_num})")
    ax[1].legend()

    cbar = fig.colorbar(sm, ax=[ax[0]], orientation="horizontal", pad=0.08, fraction=0.05)
    cbar.set_label("Log Resistivity [ohm·m]")

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Normalization helpers (kept for downstream benchmark preprocessing)
# ==========================================================================


def global_normalize(arr: np.ndarray, phys_bounds: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr_norm = np.full_like(arr, np.nan, dtype=float)

    phys_min, phys_max = map(float, phys_bounds)
    if not np.isfinite(phys_min) or not np.isfinite(phys_max) or phys_max <= phys_min:
        raise ValueError(f"phys_bounds must be finite with high > low, got {phys_bounds!r}")

    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr_norm

    arr_norm[finite] = (arr[finite] - phys_min) / (phys_max - phys_min)
    return arr_norm


def log_resistivity(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)

    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    out[positive] = np.log(arr[positive])
    return out


def log_rad(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)

    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    out[positive] = np.log1p(arr[positive])
    return out


# ==========================================================================
# Single-sample generation
# ==========================================================================


def generate_synthetic_formation_and_logs(
    model_iter,
    seed=None,
    top=0.0,
    base=25.6,
    DZ=0.2,
    eps=0.2,
    transition_probs=None,
    lambda_space=None,
    initial_state="shale",
    enforce_shale=False,
    use_blocks=True,
    cali_mean_bounds=(0.07, 0.3),
    cali_noise_fraction=0.07,
    rm_mean_bounds=(0.1, 1.5),
    rm_noise_fraction=0.05,
    uniform_shale_form_res=(0.8, 8.0),
    uniform_sand_form_res=(0.8, 200.0),
    uniform_inv_radius_delta=(0.05, 0.4),
    uniform_inv_res=(0.5, 200.0),
    inv_prob_sand=0.8,
    rdfz_noise_fraction=0.01,
    rtfz_noise_fraction=0.05,
    rtuz_noise_fraction=0.05,
    rdfz_kernel_size=5,
    rtfz_kernel_size=3,
    rtuz_kernel_size=3,
    min_inv_delta=0.05,
    output_root=DEFAULT_OUTPUT_ROOT,
    folder_name="smooth_noise",
    sample_num=0,
    domain_radius=40,
    batch_size=5,
    mesh_generator="gmsh",
    all_tools=None,
    plot_fig=False,
):
    """Generate a random pseudowell, simulate its logs, and save raw model arrays to npz."""
    if all_tools is None:
        all_tools = list(DEFAULT_TOOLS)

    rng_master = np.random.default_rng(seed)
    child_seeds = rng_master.integers(0, np.iinfo(np.uint64).max, size=6, dtype=np.uint64)
    rng_lambda = np.random.default_rng(int(child_seeds[0]))
    rng_borehole = np.random.default_rng(int(child_seeds[1]))
    pseudowell_seed = int(child_seeds[2])
    property_seed = int(child_seeds[3])
    upsample_seed = int(child_seeds[4])
    rng_state = np.random.default_rng(int(child_seeds[5]))
    if initial_state is None:
        initial_state = str(rng_state.choice(["shale", "sand"]))
    if transition_probs is None:
        transition_probs = pd.DataFrame(
            [
                [0.0, 1.0],
                [0.6, 0.4],
            ],
            index=["shale", "sand"],
            columns=["shale", "sand"],
            dtype=float,
        )
    if lambda_space is None:
        shale_lam = rng_lambda.uniform(2.0, 7.0)
        sand_lam = rng_lambda.uniform(3.0, 7.0)
        lambda_space = {
            "shale": shale_lam,
            "sand": sand_lam,
        }

    df_form_init, depth_grid, df_grid = simulate_pseudowell_grid(
        P_dict=transition_probs,
        lam_d=lambda_space,
        facies=["shale", "sand"],
        top=top,
        base=base,
        DZ=DZ,
        eps=eps,
        seed=pseudowell_seed,
        initial_state=initial_state,
        enforce_shale=enforce_shale,
    )

    depth_axis = depth_grid.copy()  # borehole on the exact measurement grid (top + n*DZ); linspace drifted by DZ*n/(n-1)

    kernel_size = 3
    pad = kernel_size // 2
    kernel = np.ones(kernel_size, dtype=float)
    kernel /= kernel.sum()

    cali_mean = rng_borehole.uniform(*cali_mean_bounds)
    cali_noise = rng_borehole.normal(scale=cali_mean * cali_noise_fraction, size=depth_grid.size)
    padded = np.pad(cali_noise, pad_width=pad, mode="edge")
    cali_noise = np.convolve(padded, kernel, mode="valid")
    cali = np.clip(cali_mean + cali_noise, *cali_mean_bounds)

    rm_mean = rng_borehole.uniform(*rm_mean_bounds)
    rm_noise = rng_borehole.normal(scale=rm_mean * rm_noise_fraction, size=depth_grid.size)
    padded = np.pad(rm_noise, pad_width=pad, mode="edge")
    rm_noise = np.convolve(padded, kernel, mode="valid")
    rm = np.clip(rm_mean + rm_noise, *rm_mean_bounds)

    df_borehole = pd.DataFrame(
        {
            "DEPT": depth_axis,
            "CALM": cali,
            "RM": rm,
        }
    )

    if not use_blocks:
        df_form = add_formation_properties(
            df_form=df_grid,
            df_borehole=df_borehole,
            uniform_shale_form_res=uniform_shale_form_res,
            uniform_sand_form_res=uniform_sand_form_res,
            uniform_inv_radius_delta=uniform_inv_radius_delta,
            uniform_inv_res=uniform_inv_res,
            inv_prob_sand=inv_prob_sand,
            seed=property_seed,
        )
        df_form_upsampled = df_form.copy()

    else:
        df_form = add_formation_properties(
            df_form=df_form_init,
            df_borehole=df_borehole,
            uniform_shale_form_res=uniform_shale_form_res,
            uniform_sand_form_res=uniform_sand_form_res,
            uniform_inv_radius_delta=uniform_inv_radius_delta,
            uniform_inv_res=uniform_inv_res,
            inv_prob_sand=inv_prob_sand,
            seed=property_seed,
        )

        df_form_upsampled = upsample_formation_properties(
            df_form=df_form,
            df_borehole=df_borehole,
            DZ=DZ,
            rdfz_noise_fraction=rdfz_noise_fraction,
            rtfz_noise_fraction=rtfz_noise_fraction,
            rtuz_noise_fraction=rtuz_noise_fraction,
            rdfz_kernel_size=rdfz_kernel_size,
            rtfz_kernel_size=rtfz_kernel_size,
            rtuz_kernel_size=rtuz_kernel_size,
            seed=upsample_seed,
            min_inv_delta=min_inv_delta,
            rdfz_bounds=(0, 0.9),
            rtuz_bounds=(uniform_shale_form_res[0], uniform_sand_form_res[1]),
            rtfz_bounds=uniform_inv_res,
        )

    formation_model = np.round(np.array(df_form_upsampled[["TOP", "BOTTOM", "rdfz", "rtfz", "rtuz"]], dtype="float64"), 4)
    borehole_model = np.round(np.array(df_borehole[["DEPT", "CALM", "RM"]], dtype="float64"), 4)

    # ---------------- Simulate logs ----------------
    measurement_depths = np.arange(top, base, DZ, dtype=float)

    try:
        model_iter.set_model_parameters(
            formation_model.copy(),
            borehole_model.copy(),
            borehole_geometry_type="radius",
            dip=0,
        )

        model_iter.simulate_logs(
            measurement_depths.copy(),
            domain_radius=domain_radius,
            batch_size=batch_size,
            mesh_generator=mesh_generator,
        )

        for tool in all_tools:
            if not np.all(np.isfinite(model_iter.logs[tool])):
                raise ValueError(f"Non-finite values found in logs for tool {tool!r}")

    except Exception as e:
        raise RuntimeError(f"Error during synthetic log computation: {e!r}") from e

    expected_len = int(round((base - top) / DZ))
    if formation_model.shape[0] != expected_len:
        raise ValueError(f"formation_model must have {expected_len} rows, got {formation_model.shape[0]}")

    total_path_data = os.path.join(output_root, folder_name, "data/")
    os.makedirs(total_path_data, exist_ok=True)

    np.savez(
        os.path.join(total_path_data, f"sample_{sample_num}.npz"),
        formation_model=formation_model,
        borehole_model=borehole_model,
        model_logs=model_iter.logs,
    )

    if plot_fig:
        fig_path = os.path.join(output_root, folder_name, "figures", f"sample_{sample_num}_formation.png")
        _plot_sample(formation_model, borehole_model, model_iter.logs, all_tools, fig_path, sample_num)

    return formation_model, borehole_model


# ==========================================================================
# Dataset driver
# ==========================================================================


def generate_data(
    tools=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    folder_name="smooth_noise",
    plot_count=0,
    sample_start=0,
    sample_end=10,
    cpu_workers=16,
    use_blocks=True,
    initial_state=None,
    transition_probs=None,
    lambda_space=None,
    seed=None,
    base=25.6,
    dz=0.2,
    top=0.0,
):
    """Generate ``sample_end - sample_start`` realizations, plotting the first ``plot_count``.

    A single ReMo3D ``Model`` with a persistent MPI worker pool is reused across
    all samples. Failed samples are logged and skipped without advancing the
    sample counter, matching the original behavior.
    """
    if tools is None:
        tools = list(DEFAULT_TOOLS)

    target_samples = sample_end
    i_sample = sample_start

    model = Model(tools=tools)
    model.initialize_workers(cpu_workers=cpu_workers, gpu_workers=0)

    try:
        while i_sample < target_samples:
            # Per-index seed: depends only on (seed, i_sample), never on the
            # sample_start/iteration order. So appending a range (e.g. 10..99)
            # yields fresh, distinct, reproducible samples that never collide
            # with an earlier range (0..9).
            sample_seed = (
                None
                if seed is None
                else int(np.random.SeedSequence([int(seed), int(i_sample)]).generate_state(1)[0])
            )
            try:
                generate_synthetic_formation_and_logs(
                    model_iter=model,
                    seed=sample_seed,
                    sample_num=i_sample,
                    all_tools=tools,
                    initial_state=initial_state,
                    transition_probs=transition_probs,
                    lambda_space=lambda_space,
                    inv_prob_sand=0.8,
                    use_blocks=use_blocks,
                    enforce_shale=False,
                    output_root=output_root,
                    folder_name=folder_name,
                    eps=0.4,
                    top=top,
                    base=base,
                    DZ=dz,
                    plot_fig=(i_sample - sample_start) < plot_count,
                )
                print(f"[generate_data] sample {i_sample} OK")
                i_sample += 1

            except Exception as e:
                print(f"Error during synthetic data generation for sample {i_sample}: {e!r}")
    finally:
        try:
            model.shutdown_workers()
        except Exception as e:
            print(f"[generate_data] worker shutdown warning: {e!r}")


# ==========================================================================
# Stage configurations
# ==========================================================================


def _default_transition_probs():
    return pd.DataFrame(
        [
            [0.0, 1.0],
            [0.6, 0.4],
        ],
        index=["shale", "sand"],
        columns=["shale", "sand"],
        dtype=float,
    )


def build_stages(sample_end=10, cpu_workers=16, plot_count=0, output_root=DEFAULT_OUTPUT_ROOT, sample_start=0,
                 base=25.6, dz=0.2):
    """Return the stage-configuration dict, parameterized for a benchmark-sized run."""
    common = dict(
        tools=list(DEFAULT_TOOLS),
        output_root=output_root,
        plot_count=plot_count,
        sample_start=sample_start,
        sample_end=sample_end,
        cpu_workers=cpu_workers,
        base=base,
        dz=dz,
    )
    return {
        "smooth_noise": {
            **common,
            "folder_name": "smooth_noise",
            "use_blocks": True,
            "initial_state": None,
            "transition_probs": _default_transition_probs(),
            "lambda_space": None,
        },
        "large_noise": {
            **common,
            "folder_name": "large_noise",
            "use_blocks": False,
            "initial_state": None,
            "transition_probs": _default_transition_probs(),
            "lambda_space": None,
        },
        "unphysical_noise": {
            **common,
            "folder_name": "unphysical_noise",
            "use_blocks": False,
            "initial_state": "sand",
            "transition_probs": pd.DataFrame(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                index=["shale", "sand"],
                columns=["shale", "sand"],
                dtype=float,
            ),
            "lambda_space": {"shale": 1, "sand": 20},
        },
    }


# ==========================================================================
# CLI
# ==========================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="smooth_noise", choices=["smooth_noise", "large_noise", "unphysical_noise"],
                        help="Which noise/formation configuration to generate. Default: smooth_noise.")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of samples to generate (a count). Default: 10.")
    parser.add_argument("--sample-start", type=int, default=0,
                        help="First sample index to write (enables appending new distinct "
                             "samples without overwriting existing ones). Default: 0.")
    parser.add_argument("--plot-count", type=int, default=2,
                        help="Number of leading samples to also render as figures. Default: 2.")
    parser.add_argument("--cpu-workers", type=int, default=16,
                        help="MPI CPU workers for the forward solver. Default: 16.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Root output directory. Default: {DEFAULT_OUTPUT_ROOT}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base RNG seed for reproducible sampling. Default: nondeterministic.")
    parser.add_argument("--base", type=float, default=25.6,
                        help="Well/formation length in metres (top=0). (base-top) must be divisible by --dz. Default: 25.6.")
    parser.add_argument("--dz", type=float, default=0.2,
                        help="Depth step / cell size in metres. Default: 0.2.")
    return parser.parse_args(argv)


def main(argv=None):
    _ensure_mpi_launcher()
    args = parse_args(argv)
    if args.n_samples < 1:
        raise SystemExit("--n-samples must be >= 1")
    if args.plot_count < 0:
        raise SystemExit("--plot-count must be >= 0")
    if args.cpu_workers < 1:
        raise SystemExit("--cpu-workers must be >= 1")
    if args.sample_start < 0:
        raise SystemExit("--sample-start must be >= 0")

    stages = build_stages(
        sample_end=args.sample_start + args.n_samples,
        cpu_workers=args.cpu_workers,
        base=args.base,
        dz=args.dz,
        plot_count=min(args.plot_count, args.n_samples),
        output_root=args.output_root,
        sample_start=args.sample_start,
    )
    config = stages[args.stage]

    print(f"[main] stage={args.stage!r}  samples=[{args.sample_start}, {args.sample_start + args.n_samples})  "
          f"plot_count={config['plot_count']}  cpu_workers={args.cpu_workers}")
    print(f"[main] output -> {Path(args.output_root) / config['folder_name']}")

    generate_data(seed=args.seed, **config)
    print("[main] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
