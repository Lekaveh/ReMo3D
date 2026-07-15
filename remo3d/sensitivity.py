# -*- coding: utf-8 -*-
"""
Sensitivity (impact) analysis for ReMo3D logging tools.

This module computes 2D (r, z) Fréchet sensitivity kernels — i.e. how much each
area of the formation contributes to a given apparent-resistivity measurement.

Two methods are available:

* ``analytical_sensitivity`` — closed-form Born kernel for a homogeneous
  background (uses full-space Green's functions ``1/(4π|x|)``). Fast, suitable
  for interactive use.
* ``perturbation_sensitivity`` — finite-difference kernel computed by calling
  ``Model.compute_synthetic_logs`` once per perturbed cell. Slow, intended for
  verification on coarse grids only.

``plot_sensitivity`` dispatches to either method and returns a matplotlib
``Axes`` with a symmetric-log contour and optional formation outline.

The integrand convention is

    ∫∫ S(r, z) dr dz  ≈  R_a   (for a homogeneous background ρ₀ = R_a),

i.e. the kernel is multiplied by the axisymmetric volume Jacobian ``2πr`` so
that the value at a (r, z) point represents the contribution per unit area in
the cross-section to the apparent resistivity.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.collections import PatchCollection
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.patches import Rectangle

from .remo3d import Model


# ---------------------------------------------------------------------------
# Tool string parsing
# ---------------------------------------------------------------------------

def _parse_tool(tool):
    """
    Parse a tool string (e.g. "N2.0M0.5A" or "B5.7A0.4M") into a dict with the
    electrode positions (relative to the tool measurement point), measurement
    polarities, and geometric factor K.

    Applies the same ``force_single_electrode_configuration`` swap used by
    ``Model.set_tools_parameters`` so that A↔M and B↔N are exchanged when both
    A and B are present. By reciprocity this produces the same apparent
    resistivity and the same Fréchet kernel as the original 4-electrode array.

    Returns
    -------
    dict with keys:
        z_C: float
            Axial offset of the (single) current electrode from the
            measurement point, in metres.
        z_M, z_N: float
            Axial offsets of the M and N (measuring) electrodes.
        K: float
            Geometric factor (always positive).
        m_M, m_N: int
            Measurement polarities. By the convention ΔU = U_M − U_N we use
            m_M = +1 and m_N = −1.
    """
    if not isinstance(tool, str) or len(tool) == 0:
        raise ValueError("tool must be a non-empty string")

    if "A" in tool and "B" in tool:
        tool_eff = tool.translate(str.maketrans("ABMN", "MNAB"))
    else:
        tool_eff = tool

    groups = [''.join(g) for _, g in itertools.groupby(tool_eff, str.isalpha)]
    letters = [g for g in groups if g.isalpha()]
    numbers = [float(g) for g in groups if not g.isalpha()]

    if len(letters) != 3 or len(numbers) != 2 or min(numbers) <= 0:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))

    valid = list(itertools.permutations(["A", "B", "M", "N"], 3))
    if tuple(letters) not in valid:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))

    positions = np.array([0.0, numbers[0], numbers[0] + numbers[1]])
    if numbers[0] < numbers[1]:
        z_mp = numbers[0] / 2
    elif numbers[0] > numbers[1]:
        z_mp = numbers[0] + numbers[1] / 2
    else:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))

    pos = {letters[i]: positions[i] - z_mp for i in range(3)}

    if "A" in pos:
        z_C = pos["A"]
    else:
        z_C = pos["B"]

    if "M" not in pos or "N" not in pos:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))
    z_M = pos["M"]
    z_N = pos["N"]

    CM = abs(z_C - z_M)
    CN = abs(z_C - z_N)
    if CM == CN:
        raise ValueError("{} logging tool specification is uncorrect (CM == CN)".format(tool))
    K = abs(4 * np.pi * CM * CN / (CN - CM))

    return {"z_C": z_C, "z_M": z_M, "z_N": z_N, "K": K, "m_M": 1, "m_N": -1}


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _load_formation(tools, formation_model, formation_units=("M", "M", "M")):
    """Accept a path or ndarray and return a metres-converted (N, 5) array."""
    helper = Model(tools)
    if isinstance(formation_model, str):
        return helper.load_formation_parameters(formation_model)
    arr = np.atleast_2d(np.asarray(formation_model, dtype=float)).copy()
    return helper.set_formation_parameters(arr, list(formation_units))


def _load_borehole(tools, borehole_model, borehole_geometry_type="diameter",
                   borehole_units=("M", "M")):
    """Accept a path or ndarray and return a metres-converted (N, 3) array."""
    helper = Model(tools)
    if isinstance(borehole_model, str):
        return helper.load_borehole_parameters(
            borehole_model,
            borehole_geometry_type=borehole_geometry_type
        )
    arr = np.atleast_2d(np.asarray(borehole_model, dtype=float)).copy()
    return helper.set_borehole_parameters(
        arr,
        borehole_geometry_type=borehole_geometry_type,
        borehole_units=list(borehole_units)
    )


def _resistivity_at_depth(depth, formation):
    """Return the UZ resistivity of the layer that contains ``depth``."""
    for row in formation:
        if row[0] <= depth <= row[1]:
            return float(row[4])
    # depth outside formation — fall back to nearest layer
    if depth < formation[0, 0]:
        return float(formation[0, 4])
    return float(formation[-1, 4])


def _default_grids(formation, borehole, depth, n_r, n_z, r_lim=None, z_lim=None):
    """Pick default grid extents matching save_results conventions."""
    if r_lim is None:
        if np.all(np.isnan(formation[:, 2])):
            r_max = 10 * np.nanmax(borehole[:, 1])
        else:
            r_max = 2 * np.nanmax(formation[:, 2])
        r_lim = (-r_max, r_max)
    if z_lim is None:
        z_lim = (float(np.nanmin(formation[:, :2])), float(np.nanmax(formation[:, :2])))
    r_grid = np.linspace(r_lim[0], r_lim[1], n_r)
    z_grid = np.linspace(z_lim[0], z_lim[1], n_z)
    return r_grid, z_grid, r_lim, z_lim


# ---------------------------------------------------------------------------
# Analytical Born sensitivity
# ---------------------------------------------------------------------------

def _kernel_pair(R, Z, e_z, f_z):
    """
    Inner product ∇G(P − E)·∇G(P − F) / G_factor² for full-space Green's
    function G = 1/(4π|x|), evaluated on an (r, z) grid with E = (0, e_z) and
    F = (0, f_z) on the borehole axis.

    Returns ((P−E)·(P−F)) / (|P−E|³ |P−F|³).  The 1/(4π)² factor is applied by
    the caller.
    """
    dE_r = R
    dE_z = Z - e_z
    dF_r = R
    dF_z = Z - f_z
    rE = np.sqrt(dE_r ** 2 + dE_z ** 2)
    rF = np.sqrt(dF_r ** 2 + dF_z ** 2)
    dot = dE_r * dF_r + dE_z * dF_z
    with np.errstate(divide='ignore', invalid='ignore'):
        return dot / (rE ** 3 * rF ** 3)


def analytical_sensitivity(tool, depth, formation_model, borehole_model,
                           r_grid=None, z_grid=None,
                           n_r=120, n_z=120, r_lim=None, z_lim=None,
                           rho_background=None, epsilon=0.01,
                           normalize=None,
                           formation_units=("M", "M", "M"),
                           borehole_units=("M", "M"),
                           borehole_geometry_type="diameter"):
    """
    Compute the analytical Born sensitivity kernel S(r, z) for ``tool`` at the
    given measurement ``depth``.

    The kernel is the integrand of the volume integral against ln ρ, i.e.

        S(r, z) = ∂R_a / ∂(ln ρ)  (per unit (r, z) area, axisymmetric)

    so that ∫∫ S(r, z) dr dz ≈ R_a^homogeneous = ρ₀ when integrated over a
    sufficiently large half-plane (r ≥ 0).

    Parameters
    ----------
    tool : str
        Tool name in the standard form, e.g. "N2.0M0.5A".
    depth : float
        Measurement point depth in metres (matches Model.compute_synthetic_logs).
    formation_model, borehole_model : str or ndarray
        Either a path to the corresponding text file (same format as
        Forward.ipynb) or a pre-loaded numpy array.
    r_grid, z_grid : ndarray, optional
        Explicit grids; otherwise built from ``r_lim``, ``z_lim``, ``n_r``,
        ``n_z`` (defaults match save_results extents).
    rho_background : float, optional
        Background ρ₀ used in the Born formula. Defaults to the UZ resistivity
        of the layer containing ``depth``.
    epsilon : float
        Radius in metres around each electrode where the kernel is masked to
        NaN to avoid the 1/r² singularity.
    normalize : {None, 'percent', 'log'}
        ``None`` returns the kernel in Ω·m. ``'percent'`` divides by ρ₀ and
        multiplies by 100. ``'log'`` returns sign(S) · log10(1 + |S|/ρ₀ · 100).

    Returns
    -------
    S : ndarray, shape (len(z_grid), len(r_grid))
    r_grid : ndarray
    z_grid : ndarray
    """
    formation = _load_formation([tool], formation_model, formation_units)
    borehole = _load_borehole([tool], borehole_model,
                              borehole_geometry_type=borehole_geometry_type,
                              borehole_units=borehole_units)

    if r_grid is None or z_grid is None:
        r_grid_def, z_grid_def, _, _ = _default_grids(formation, borehole, depth,
                                                     n_r, n_z, r_lim, z_lim)
        if r_grid is None:
            r_grid = r_grid_def
        if z_grid is None:
            z_grid = z_grid_def

    info = _parse_tool(tool)
    z_C = depth + info["z_C"]
    z_M = depth + info["z_M"]
    z_N = depth + info["z_N"]
    K = info["K"]

    if rho_background is None:
        rho_background = _resistivity_at_depth(depth, formation)

    R, Z = np.meshgrid(r_grid, z_grid)
    # use |R| because the kernel is axisymmetric (E,F on the borehole axis);
    # mirroring r → −r reproduces the same value at the same physical radius.
    Rabs = np.abs(R)

    K_CM = _kernel_pair(Rabs, Z, z_C, z_M)
    K_CN = _kernel_pair(Rabs, Z, z_C, z_N)
    inv16pi2 = 1.0 / (16.0 * np.pi ** 2)

    # Integrand form: multiply by axisymmetric volume Jacobian 2π r.
    S = K * rho_background * (K_CM - K_CN) * inv16pi2 * (2.0 * np.pi * Rabs)

    # Mask out near-electrode singularity region.
    d_C = np.sqrt(Rabs ** 2 + (Z - z_C) ** 2)
    d_M = np.sqrt(Rabs ** 2 + (Z - z_M) ** 2)
    d_N = np.sqrt(Rabs ** 2 + (Z - z_N) ** 2)
    mask = (d_C < epsilon) | (d_M < epsilon) | (d_N < epsilon)
    S = np.where(mask, np.nan, S)

    if normalize == "percent":
        S = S / rho_background * 100.0
    elif normalize == "log":
        scale = np.abs(S) / rho_background * 100.0
        S = np.sign(S) * np.log10(1.0 + scale)
    elif normalize is not None:
        raise ValueError("normalize must be None, 'percent', or 'log'")

    return S, r_grid, z_grid


# ---------------------------------------------------------------------------
# Perturbation (finite-difference) sensitivity
# ---------------------------------------------------------------------------

def _insert_perturbed_layer(formation, z_lo, z_hi, rho_factor, r_cell_lo, r_cell_hi):
    """
    Build a formation_model where the layer slab [z_lo, z_hi] has its
    resistivity scaled by (1 + rho_factor). For cells that straddle the
    borehole axis (r_cell_lo < 0 < r_cell_hi) we scale the UZ resistivity of
    the slab; for off-axis cells we scale the FZ value if a filtration zone
    already exists and overlaps the cell, otherwise we treat the cell as part
    of the UZ.

    This is a coarse approximation that respects the (Top, Bottom, FZ_radius,
    FZ_value, UZ_value) parameterization without modifying the FEM solver. It
    is good enough to validate the analytical kernel on a smooth medium.
    """
    f = formation.copy()
    # find layers overlapping [z_lo, z_hi]
    rows_new = []
    for i in range(f.shape[0]):
        top, bot, fz_r, fz_v, uz_v = f[i]
        if bot <= z_lo or top >= z_hi:
            rows_new.append(f[i])
            continue
        # split the layer into [top, max(top, z_lo)] | [overlap] | [min(bot, z_hi), bot]
        a = max(top, z_lo)
        b = min(bot, z_hi)
        if top < a:
            rows_new.append(np.array([top, a, fz_r, fz_v, uz_v]))
        # perturbed slab
        uz_new = uz_v * (1.0 + rho_factor)
        fz_v_new = fz_v
        if not np.isnan(fz_r) and not np.isnan(fz_v):
            # if the cell is inside the FZ (|r| < fz_r) and on either side
            if r_cell_hi <= fz_r and r_cell_lo >= -fz_r:
                fz_v_new = fz_v * (1.0 + rho_factor)
        rows_new.append(np.array([a, b, fz_r, fz_v_new, uz_new]))
        if b < bot:
            rows_new.append(np.array([b, bot, fz_r, fz_v, uz_v]))
    return np.vstack(rows_new)


def perturbation_sensitivity(tool, depth, formation_model, borehole_model,
                             r_grid, z_grid, perturbation=0.1,
                             cpu_workers=2, gpu_workers=0,
                             domain_radius=50, batch_size=1,
                             mesh_generator="netgen",
                             formation_units=("M", "M", "M"),
                             borehole_units=("M", "M"),
                             borehole_geometry_type="diameter",
                             dip=0,
                             verbose=True):
    """
    Compute a finite-difference Fréchet kernel by perturbing the formation
    resistivity inside each (r, z) cell and re-running the forward model.

    .. warning::
        Each cell costs one full FEM forward solve. Even with a coarse grid
        (8×16 = 128 cells), expect several minutes to hours depending on
        hardware. Use ``analytical_sensitivity`` for interactive work and call
        this function only for verification on small grids.

    Parameters
    ----------
    r_grid, z_grid : ndarray
        Edge-centred grid (cell *centres* lie at these points; cell widths
        derived from the spacing). Total cells = len(r_grid) × len(z_grid).
    perturbation : float
        Fractional perturbation of ρ inside each cell. The finite-difference
        is computed as ((R_a^+ − R_a^0) / log(1 + perturbation)) so that the
        result is comparable to the analytical ∂R_a/∂(ln ρ).

    Returns
    -------
    S : ndarray, shape (len(z_grid), len(r_grid))
    """
    formation = _load_formation([tool], formation_model, formation_units)
    borehole = _load_borehole([tool], borehole_model,
                              borehole_geometry_type=borehole_geometry_type,
                              borehole_units=borehole_units)

    n_cells = len(r_grid) * len(z_grid)
    if n_cells > 256:
        warnings.warn(
            "perturbation_sensitivity will run {} forward solves; "
            "this is very expensive. Consider a coarser grid.".format(n_cells))

    dr = np.diff(r_grid).mean() if len(r_grid) > 1 else 1.0
    dz = np.diff(z_grid).mean() if len(z_grid) > 1 else 1.0
    log_pert = np.log(1.0 + perturbation)

    # Baseline forward
    if verbose:
        print("Baseline forward solve ...")
    baseline = Model.compute_synthetic_logs(
        [tool], np.array([depth]),
        formation.copy(), borehole.copy(),
        cpu_workers=cpu_workers, gpu_workers=gpu_workers,
        domain_radius=domain_radius, batch_size=batch_size,
        mesh_generator=mesh_generator, dip=dip)
    R_a_0 = baseline.logs[tool][0, 1]

    S = np.full((len(z_grid), len(r_grid)), np.nan)
    cell_area = dr * dz
    count = 0
    for j, z_c in enumerate(z_grid):
        for i, r_c in enumerate(r_grid):
            count += 1
            r_lo, r_hi = r_c - dr / 2, r_c + dr / 2
            z_lo, z_hi = z_c - dz / 2, z_c + dz / 2
            if verbose:
                print("Cell {}/{}  (r={:.2f}, z={:.2f})".format(count, n_cells, r_c, z_c))
            try:
                f_pert = _insert_perturbed_layer(formation, z_lo, z_hi,
                                                 perturbation, r_lo, r_hi)
                pert = Model.compute_synthetic_logs(
                    [tool], np.array([depth]),
                    f_pert, borehole.copy(),
                    cpu_workers=cpu_workers, gpu_workers=gpu_workers,
                    domain_radius=domain_radius, batch_size=batch_size,
                    mesh_generator=mesh_generator, dip=dip)
                R_a_p = pert.logs[tool][0, 1]
            except Exception as exc:  # noqa: BLE001
                warnings.warn("Perturbation at (r={:.2f}, z={:.2f}) failed: {}".format(r_c, z_c, exc))
                continue
            # Convert delta R_a per cell to integrand-form ∂R_a/∂(ln ρ) per (r,z) area.
            S[j, i] = (R_a_p - R_a_0) / log_pert / cell_area
    return S


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_formation_outline(ax, formation, borehole, dip, model_rad_lim, depth_lim):
    """Overlay layer/FZ/borehole boundaries on top of a sensitivity contour."""
    patches, _ = Model._build_model_patches(formation.copy(), borehole.copy(), dip, model_rad_lim)
    collection = PatchCollection(patches, facecolor='none', edgecolor='black',
                                 linewidths=0.8)
    ax.add_collection(collection)
    ax.plot([0, 0], depth_lim, color='black', linewidth=0.8)


def _electrode_positions(tool):
    """
    Return ``[(label, z_offset), ...]`` for each electrode literally present in
    the tool string (in order: top → bottom along the tool), with z_offset
    measured from the tool measurement point (positive = below the MP).

    Unlike ``_parse_tool``, this does NOT apply the A↔M / B↔N reciprocity swap —
    it returns the original electrode labels so they can be annotated on a plot.
    """
    groups = [''.join(g) for _, g in itertools.groupby(tool, str.isalpha)]
    letters = [g for g in groups if g.isalpha()]
    numbers = [float(g) for g in groups if not g.isalpha()]
    if len(letters) != 3 or len(numbers) != 2:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))
    positions = np.array([0.0, numbers[0], numbers[0] + numbers[1]])
    if numbers[0] < numbers[1]:
        z_mp = numbers[0] / 2
    elif numbers[0] > numbers[1]:
        z_mp = numbers[0] + numbers[1] / 2
    else:
        raise ValueError("{} logging tool specification is uncorrect".format(tool))
    return [(letters[i], float(positions[i] - z_mp)) for i in range(3)]


def _draw_electrodes(ax, tool, depth):
    """Mark electrode positions on the borehole axis with their letter labels."""
    color_map = {"A": "#d62728", "B": "#d62728", "M": "#1f77b4", "N": "#1f77b4"}
    for label, z_off in _electrode_positions(tool):
        z_e = depth + z_off
        ax.plot(0, z_e, marker="o", markersize=7,
                markerfacecolor=color_map.get(label, "black"),
                markeredgecolor="white", markeredgewidth=1.2,
                linestyle="None", zorder=10)
        ax.annotate(label, xy=(0, z_e), xytext=(7, 0),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    color=color_map.get(label, "black"),
                    va="center", ha="left", zorder=11)


def _cumulative_extent(masses, distances, fraction):
    """Smallest distance from the origin capturing ``fraction`` of ``masses``.

    ``masses`` and ``distances`` are 1D arrays over the same coordinate samples.
    The samples are ordered by increasing distance and the cumulative mass
    (normalised by its own total) is thresholded at ``fraction``.
    """
    masses = np.asarray(masses, dtype=float)
    distances = np.asarray(distances, dtype=float)
    if masses.size == 0:
        return 0.0
    total = masses.sum()
    if total <= 0:
        return float(np.max(distances))
    order = np.argsort(distances)
    cum = np.cumsum(masses[order]) / total
    idx = min(int(np.searchsorted(cum, fraction)), len(order) - 1)
    return float(distances[order][idx])


def _measure_doi(tool, depth, formation, borehole, fraction,
                 rho_background, epsilon, n_r, n_z,
                 max_passes=12, grow=1.8, edge_frac=0.6, rel_tol=0.02):
    """Measure the directional DOI on a domain that adapts to the tool.

    The default plotting grid ties the radial half-width to the borehole radius
    (``10 × r_borehole``), which truncates the radial sensitivity of longer
    tools and makes ``r_eff`` come out roughly the same for every tool. Here the
    domain is instead seeded from the electrode span and each half-width is
    enlarged whenever its extent reaches the boundary (``edge_frac``) or is still
    drifting between passes (``rel_tol``), until all three extents sit inside the
    domain and have stabilised. A final pass on a grid confined to the converged
    box gives sharp, resolution-independent values.

    Returns
    -------
    r_eff, dz_up, dz_down : float
    """
    span = max((abs(z) for _, z in _electrode_positions(tool)), default=1.0)
    R = max(2.0 * span, 1.0)          # radial half-width
    Zu = Zd = max(2.0 * span, 1.0)    # vertical half-widths (up / down)

    prev = None
    r_eff = dz_up = dz_down = 0.0
    for _ in range(max_passes):
        rf = np.linspace(-R, R, n_r)
        zf = np.linspace(depth - Zu, depth + Zd, n_z)
        S, rf, zf = analytical_sensitivity(
            tool, depth, formation, borehole,
            r_grid=rf, z_grid=zf,
            rho_background=rho_background, epsilon=epsilon)
        r_eff, dz_up, dz_down = _effective_extent(S, rf, zf, depth, fraction=fraction)

        grew = False
        if r_eff > edge_frac * R:
            R *= grow; grew = True
        if dz_up > edge_frac * Zu:
            Zu *= grow; grew = True
        if dz_down > edge_frac * Zd:
            Zd *= grow; grew = True
        if prev is not None and not grew:
            moved = max(abs(c - p) / max(c, 1e-9)
                        for c, p in zip((r_eff, dz_up, dz_down), prev))
            if moved > rel_tol:            # tail not fully captured yet
                R *= grow; Zu *= grow; Zd *= grow; grew = True
        prev = (r_eff, dz_up, dz_down)
        if not grew:
            break

    # The converged domain contains essentially all the sensitivity, so these
    # extents are unbiased; the domain scales with the tool, so the fixed grid
    # keeps a roughly tool-independent (~2 %) resolution.
    return r_eff, dz_up, dz_down


def _effective_extent(S, r_grid, z_grid, depth, fraction=0.9):
    """
    Effective extent of investigation from a sensitivity kernel, computed
    **independently for each direction** so an asymmetric kernel is not forced
    into a single symmetric box:

    * ``r_eff``   — radial half-extent. The kernel is axisymmetric (symmetric in
      r), so a single value describes both ``+r`` and ``−r``.
    * ``dz_up``   — vertical extent *above* the measurement point (``z < depth``).
    * ``dz_down`` — vertical extent *below* the measurement point (``z > depth``).

    Each extent is the smallest distance from its origin (borehole axis for r,
    measurement depth for z) at which the accumulated ``|S|`` along that
    direction reaches ``fraction`` of that direction's own total. Splitting the
    vertical axis into up/down captures the asymmetry of lateral tools, which a
    single ``|z − depth|`` accumulation would average away.

    Returns
    -------
    r_eff, dz_up, dz_down : float
    """
    Sabs = np.where(np.isfinite(S), np.abs(S), 0.0)

    # Radial (symmetric): collapse over z, accumulate over |r|.
    r_eff = _cumulative_extent(Sabs.sum(axis=0), np.abs(r_grid), fraction)

    # Vertical: split into the upper and lower half relative to the measurement
    # depth and treat each side on its own. The centre row (z == depth) belongs
    # to both sides at distance 0 and does not affect the extent.
    row_sums = Sabs.sum(axis=1)
    dz = z_grid - depth
    upper = dz <= 0.0
    lower = dz >= 0.0
    dz_up = _cumulative_extent(row_sums[upper], np.abs(dz[upper]), fraction)
    dz_down = _cumulative_extent(row_sums[lower], np.abs(dz[lower]), fraction)

    return r_eff, dz_up, dz_down


def _draw_doi(ax, depth, r_eff, dz_up, dz_down, fraction, color="0.1"):
    """Overlay the effective DOI on a sensitivity plot in both directions.

    Draws the radial reach as vertical lines at ``r = ±r_eff``, the vertical
    reach as horizontal lines at ``depth − dz_up`` and ``depth + dz_down``, and
    the enclosing box. Each extent is annotated with its value.
    """
    z_up = depth - dz_up
    z_down = depth + dz_down
    line_kw = dict(color=color, lw=1.2, ls="--", alpha=0.9, zorder=5)

    # Radial reach (symmetric ±r_eff) and vertical reach (asymmetric up/down).
    ax.axvline(+r_eff, **line_kw)
    ax.axvline(-r_eff, **line_kw)
    ax.axhline(z_up, **line_kw)
    ax.axhline(z_down, **line_kw)

    # Enclosing DOI box.
    ax.add_patch(Rectangle((-r_eff, z_up), 2 * r_eff, dz_up + dz_down,
                           fill=False, edgecolor=color, lw=1.4, ls="--",
                           zorder=6))

    # Measurement point.
    ax.plot(0.0, depth, marker="o", ms=4, color=color, zorder=7)

    txt_kw = dict(color=color, fontsize=8, zorder=7,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.annotate("r = {:.2f} m".format(r_eff), xy=(r_eff, depth),
                ha="right", va="bottom", **txt_kw)
    ax.annotate("↑ {:.2f} m".format(dz_up), xy=(0.0, z_up),
                ha="center", va="bottom", **txt_kw)
    ax.annotate("↓ {:.2f} m".format(dz_down), xy=(0.0, z_down),
                ha="center", va="top", **txt_kw)


def plot_sensitivity_doi(tool, depth, formation_model, borehole_model,
                         fraction=0.9, n_r=160, n_z=160,
                         rho_background=None, epsilon=0.01,
                         formation_units=("M", "M", "M"),
                         borehole_units=("M", "M"),
                         borehole_geometry_type="diameter",
                         draw_doi=True, doi_margin=1.15, doi_color="0.1",
                         refine=True,
                         **plot_kwargs):
    """
    Plot the sensitivity kernel for ``tool`` at ``depth`` with the depth of
    investigation (DOI) computed and drawn **independently in each direction**:
    the radial reach ``r_eff`` and the vertical reach split into ``dz_up``
    (above the measurement point) and ``dz_down`` (below it).

    Internally calls ``analytical_sensitivity`` once on a default grid, derives
    the three extents via ``_effective_extent`` (cumulative ``|S|`` threshold),
    frames the plot around the DOI box (enlarged by ``doi_margin`` so the box is
    visible), then overlays the box and per-direction markers via ``_draw_doi``.

    Parameters
    ----------
    fraction : float, optional
        Cumulative |S| threshold defining each directional extent (default 0.9).
    draw_doi : bool, optional
        Overlay the DOI box and directional markers (default True).
    doi_margin : float, optional
        Axis limits are set to the DOI extent times this factor so the box is
        not flush against the frame (default 1.15).
    doi_color : optional
        Colour of the DOI overlay.
    refine : bool, optional
        Measure the DOI on a domain that adapts to the tool (default True) via
        ``_measure_doi``, instead of the default plotting grid whose radial
        half-width is tied to the borehole radius. The default grid truncates
        the radial sensitivity of longer tools, making ``r_eff`` come out nearly
        identical for every tool; the adaptive domain lets it scale with the
        tool as expected. Set False for the legacy single-pass behaviour.
    plot_kwargs : dict
        Forwarded to ``plot_sensitivity`` (e.g. ``method``, ``log_scale``,
        ``cmap``, ``ax``, ``title``, ``colorbar``, ``vmax`` …). ``r_lim`` and
        ``z_lim`` here override the auto-computed extent.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    formation = _load_formation([tool], formation_model, formation_units)
    borehole = _load_borehole([tool], borehole_model,
                              borehole_geometry_type=borehole_geometry_type,
                              borehole_units=borehole_units)

    if refine:
        # Measure the DOI on a domain that adapts to the tool, so the radial
        # extent is driven by physics rather than truncated by the default grid
        # (which is tied to the borehole radius and made r_eff look identical
        # for every tool).
        r_eff, dz_up, dz_down = _measure_doi(
            tool, depth, formation, borehole, fraction,
            rho_background, epsilon, n_r, n_z)
    else:
        # Legacy single pass on the default (borehole-radius-tied) grid.
        r_grid, z_grid, _, _ = _default_grids(formation, borehole, depth, n_r, n_z)
        S, r_grid, z_grid = analytical_sensitivity(
            tool, depth, formation, borehole,
            r_grid=r_grid, z_grid=z_grid,
            rho_background=rho_background, epsilon=epsilon)
        r_eff, dz_up, dz_down = _effective_extent(S, r_grid, z_grid, depth,
                                                  fraction=fraction)

    # Frame around the (generally non-square, vertically asymmetric) DOI box,
    # enlarged slightly so the overlaid box sits inside the plot.
    r_span = r_eff * doi_margin
    plot_kwargs.setdefault("r_lim", (-r_span, r_span))
    plot_kwargs.setdefault("z_lim", (depth - dz_up * doi_margin,
                                     depth + dz_down * doi_margin))
    plot_kwargs.setdefault("n_r", n_r)
    plot_kwargs.setdefault("n_z", n_z)
    plot_kwargs.setdefault("rho_background", rho_background)
    plot_kwargs.setdefault("epsilon", epsilon)
    plot_kwargs.setdefault("formation_units", formation_units)
    plot_kwargs.setdefault("borehole_units", borehole_units)
    plot_kwargs.setdefault("borehole_geometry_type", borehole_geometry_type)
    if "title" not in plot_kwargs:
        plot_kwargs["title"] = (
            "Sensitivity — {}  at z = {:.2f} m\n"
            "DOI ({:.0%}): r = {:.2f} m,  ↑ {:.2f} m / ↓ {:.2f} m"
        ).format(tool, depth, fraction, r_eff, dz_up, dz_down)

    ax = plot_sensitivity(tool, depth, formation_model, borehole_model,
                          **plot_kwargs)
    if draw_doi:
        _draw_doi(ax, depth, r_eff, dz_up, dz_down, fraction, color=doi_color)
    return ax


def plot_sensitivity(tool, depth, formation_model, borehole_model,
                     method="born",
                     r_lim=None, z_lim=None, n_r=160, n_z=160,
                     overlay_formation=True, log_scale=True,
                     cmap="RdBu_r", levels=21, ax=None,
                     rho_background=None, epsilon=0.01,
                     formation_units=("M", "M", "M"),
                     borehole_units=("M", "M"),
                     borehole_geometry_type="diameter",
                     dip=0,
                     vmax=None,
                     title=None,
                     colorbar=True,
                     **method_kwargs):
    """
    Plot a 2D sensitivity contour for ``tool`` at ``depth`` over the (r, z)
    cross-section of the formation model.

    Parameters
    ----------
    method : {'born', 'perturbation'}
        ``'born'`` uses ``analytical_sensitivity`` (fast). ``'perturbation'``
        uses ``perturbation_sensitivity`` (very slow — pass a small custom
        ``r_grid``/``z_grid`` via ``method_kwargs``).
    overlay_formation : bool
        If True, overlay layer / filtration-zone / borehole boundaries on top
        of the contour.
    log_scale : bool
        Use a symmetric-log normalisation centred on zero.
    cmap : str
        Colormap. ``RdBu_r`` makes positive sensitivity red and negative blue.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    formation = _load_formation([tool], formation_model, formation_units)
    borehole = _load_borehole([tool], borehole_model,
                              borehole_geometry_type=borehole_geometry_type,
                              borehole_units=borehole_units)

    r_grid, z_grid, r_lim, z_lim = _default_grids(formation, borehole, depth,
                                                  n_r, n_z, r_lim, z_lim)

    if method == "born":
        S, r_grid, z_grid = analytical_sensitivity(
            tool, depth, formation, borehole,
            r_grid=r_grid, z_grid=z_grid,
            rho_background=rho_background, epsilon=epsilon,
            **method_kwargs)
    elif method == "perturbation":
        S = perturbation_sensitivity(
            tool, depth, formation, borehole,
            r_grid=r_grid, z_grid=z_grid,
            dip=dip,
            **method_kwargs)
    else:
        raise ValueError("method must be 'born' or 'perturbation'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8), facecolor="white")
    else:
        fig = ax.figure

    finite = S[np.isfinite(S)]
    if vmax is None:
        if finite.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.nanpercentile(np.abs(finite), 99))
            if vmax == 0:
                vmax = float(np.nanmax(np.abs(finite))) or 1.0

    if log_scale:
        linthresh = max(vmax * 1e-3, 1e-12)
        norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = Normalize(vmin=-vmax, vmax=vmax)

    R, Z = np.meshgrid(r_grid, z_grid)
    mesh = ax.pcolormesh(R, Z, S, norm=norm, cmap=cmap, shading="auto")

    if overlay_formation:
        _draw_formation_outline(ax, formation, borehole, dip, r_lim, z_lim)

    _draw_electrodes(ax, tool, depth)

    ax.set_xlim(r_lim)
    ax.set_ylim(z_lim)
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.set_xlabel("Radial distance [m]", labelpad=10)
    ax.set_ylabel("Depth [m]", labelpad=10)
    if title is None:
        title = "Sensitivity — {}  at z = {:.2f} m\n({})".format(tool, depth, method)
    ax.set_title(title)
    ticks = ax.get_xticks()
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.set_xticklabels(["{:.2f}".format(abs(t)) for t in ticks])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    if colorbar:
        cbar = fig.colorbar(mesh, ax=ax, location="bottom", pad=0.08,
                            shrink=0.85, label="∂R_a / ∂(ln ρ)  [Ω·m]")
        cbar.ax.minorticks_on()

    return ax


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(tool="N0.4M0.1A", depth=5.0, rho=10.0,
              r_max=50.0, z_half_range=50.0, n_r=400, n_z=400,
              verbose=True):
    """
    Validate the analytical kernel against its expected integral.

    For a homogeneous half-space with ρ = ρ₀,

        ∫∫_{r ≥ 0} S(r, z) dr dz ≈ ρ₀

    Returns the relative error of the integral.
    """
    formation = np.array([[depth - 50.0, depth + 50.0, np.nan, np.nan, rho]])
    borehole = np.array([[depth - 50.0, 0.1, rho], [depth + 50.0, 0.1, rho]])

    r_grid = np.linspace(0.0, r_max, n_r)
    z_grid = np.linspace(depth - z_half_range, depth + z_half_range, n_z)

    S, _, _ = analytical_sensitivity(tool, depth, formation, borehole,
                                     r_grid=r_grid, z_grid=z_grid,
                                     rho_background=rho)

    # Integrate over r ≥ 0 only (S is mirrored in r so we use a one-sided grid).
    dr = r_grid[1] - r_grid[0]
    dz = z_grid[1] - z_grid[0]
    integral = np.nansum(S) * dr * dz
    rel_err = abs(integral - rho) / rho
    if verbose:
        print("Selftest tool={} depth={} ρ₀={}".format(tool, depth, rho))
        print("  ∫∫ S dr dz = {:.4f}  (expected {:.4f}, rel. err {:.3%})".format(
            integral, rho, rel_err))
    return rel_err
