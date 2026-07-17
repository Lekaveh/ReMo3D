# -*- coding: utf-8 -*-
"""Structured graded (r, z) grid + piecewise-constant sigma sampling.

JAX-independent foundation of the GPU solver (pure numpy, testable without a
GPU). Builds a graded rectilinear grid on the axisymmetric half-plane
(r >= 0, z) and samples the piecewise-constant conductivity sigma(r, z) from
the ReMo3D formation and borehole models. Because the dip=0 geometry has
axis-aligned material boundaries (layer boundaries at const z, borehole wall /
invasion front at const r), a graded rectilinear grid is effectively
body-fitted.

Physics conventions mirror the NGSolve path (netgen_functions.SelectNetgenDataRange,
workers/worker.py):

  * mud region : r <  r_bh(z)                -> sigma = 1 / mud_resistivity
  * FZ region  : r_bh(z) <= r < FZ_RADIUS    -> sigma = 1 / FZ_VALUE (invaded layers)
  * UZ region  : otherwise                   -> sigma = 1 / UZ_VALUE

mud_resistivity is a single scalar per solve (RM interpolated at the simulation
depth, remo3d.py:883); r_bh(z) is the depth-dependent caliper radius profile.

Sigma is sampled at CELL CENTERS: sigma[j, i] is the conductivity of the cell
between z_nodes[j], z_nodes[j+1] and r_nodes[i], r_nodes[i+1] — shape
(nz-1, nr-1). Both discretization backends (FV and Q1 FEM) consume
cell-centered sigma, which represents piecewise-constant media exactly when
material boundaries lie on grid lines.
"""

import numpy as np


def graded_1d(focus, lo, hi, h_min, growth=1.15, h_max=None):
    """Sorted 1D node set graded around ``focus`` coordinates.

    Spacing is ~``h_min`` next to each focus point and grows geometrically by
    ``growth`` away from it, capped at ``h_max``. Focus points inside [lo, hi]
    and both endpoints are guaranteed to be nodes (so electrodes and material
    boundaries land exactly on grid lines).
    """
    if h_max is None:
        h_max = (hi - lo) / 8.0
    foci = sorted({float(f) for f in focus if lo <= f <= hi} | {float(lo), float(hi)})

    nodes = set(foci)
    for f in foci:
        for direction in (-1.0, +1.0):
            x, h = f, h_min
            while True:
                x = x + direction * h
                if x <= lo or x >= hi:
                    break
                nodes.add(x)
                h = min(h * growth, h_max)

    nodes = np.array(sorted(nodes), dtype=float)
    # Merge nodes closer than 0.25*h_min (overlapping refinement fans create
    # near-degenerate cells), then snap the nearest survivor back onto each
    # focus so foci stay exact.
    keep = [nodes[0]]
    for x in nodes[1:]:
        if x - keep[-1] >= 0.25 * h_min:
            keep.append(x)
    keep = np.array(keep, dtype=float)
    for f in foci:
        keep[int(np.argmin(np.abs(keep - f)))] = f
    return np.unique(keep)


def canonical_radial_nodes(domain_radius, h_min, r_fine=0.6, growth=1.15,
                           h_max=None):
    """Data-independent radial grid: fine plateau near the axis, then growth.

    Uniform ``h_min`` spacing from r=0 out to ``r_fine`` (covers the borehole
    wall and any invasion front), then geometric growth (factor ``growth``,
    capped at ``h_max``) to ``domain_radius``. Because it depends only on
    (domain_radius, h_min, r_fine, growth) — NOT on the formation's invasion
    radii or the caliper — the grid SHAPE is identical for a given tool across
    every sample. That lets the XLA compilation cache hit across samples
    (otherwise per-sample noise shifts the invasion foci and forces a recompile
    every sample; see wiki gpu-solver "canonical shapes"). Material boundaries
    that fall between nodes are represented by the anisotropic subcell sigma
    averaging (sample_sigma_aniso), so an exact node on the invasion front is
    not needed as long as the near-axis region is finely resolved — which the
    plateau guarantees.
    """
    R = float(domain_radius)
    r_fine = min(float(r_fine), R)
    if h_max is None:
        h_max = R / 8.0
    n_fine = max(int(round(r_fine / h_min)), 1)
    nodes = [i * h_min for i in range(n_fine + 1)]      # 0 .. r_fine (uniform)
    x, h = nodes[-1], h_min
    while x < R:
        h = min(h * growth, h_max)
        x = x + h
        if x >= R:
            break
        nodes.append(x)
    nodes.append(R)
    return np.array(nodes, dtype=float)


def build_grid(domain_radius, z_center, electrode_z, r_foci=None,
               h_min=1e-2, growth=1.15, h_max=None, r_fine=0.6):
    """Graded (r, z) grid for one tool solve.

    Parameters
    ----------
    domain_radius : float
        Outer radius R of the half-disk domain in the NGSolve solver. Here the
        domain is the rectangle [0, R] x [z_center - R, z_center + R]; the
        Dirichlet boundary sits at r = R and z = z_center +- R. (A rectangle
        circumscribes the NGSolve half-disk; both are "far away", and the
        difference is absorbed by the boundary-distance convergence check.)
    z_center : float
        Global z the domain is centred on (simulation depth: the current
        electrode's absolute z).
    electrode_z : sequence of float
        Global z of all tool electrodes — axial refinement foci; each becomes
        a grid node.
    r_foci : sequence of float, or None
        Radial refinement anchors (borehole radius, invasion radii). Passing
        None (default) uses the DATA-INDEPENDENT canonical_radial_nodes grid
        (recommended — keeps the grid shape constant across samples so XLA
        compilation is cached). Passing an explicit list restores the old
        data-dependent graded grid (kept for comparison).
    h_min, growth, h_max : float
        Grading parameters.
    r_fine : float
        Fine-plateau radius for the canonical radial grid (see
        canonical_radial_nodes).

    Returns
    -------
    r_nodes, z_nodes : ndarray
        Strictly increasing node coordinates; r_nodes[0] == 0.
    """
    # The fine h_min is a Z requirement (closest electrode pair, e.g. the
    # 0.1 m M-N gap needs ~5 mm axial cells). Radially the only feature is
    # the borehole wall / invasion front, where 10 mm is enough (validated on
    # Ex1); a finer radial plateau just cubes the direct-solver block cost.
    h_min_r = max(h_min, 1e-2)
    if r_foci is None:
        r_nodes = canonical_radial_nodes(domain_radius, h_min_r, r_fine=r_fine,
                                         growth=growth, h_max=h_max)
    else:
        r_nodes = graded_1d([0.0] + list(r_foci), 0.0, float(domain_radius),
                            h_min_r, growth=growth, h_max=h_max)
    z_nodes = graded_1d(electrode_z,
                        z_center - float(domain_radius),
                        z_center + float(domain_radius),
                        h_min, growth=growth, h_max=h_max)
    return r_nodes, z_nodes


def cell_centers(nodes):
    """Midpoints of consecutive nodes (length len(nodes) - 1)."""
    return 0.5 * (nodes[:-1] + nodes[1:])


def _sigma_at_points(R, Z, formation, borehole, mud_resistivity):
    """Pointwise sigma lookup on arbitrary (R, Z) arrays (broadcastable).

    mud_resistivity may be a scalar or an array broadcastable against R/Z
    (e.g. a per-depth column when a leading batch axis is present)."""
    # Layer index per point: first layer with BOTTOM > z (layers are contiguous
    # ascending; z outside the table clamps to the first/last layer, matching
    # _resistivity_at_depth's fallback).
    bots = np.asarray(formation[:, 1], dtype=float)
    layer = np.searchsorted(bots, Z, side="right")
    layer = np.clip(layer, 0, len(formation) - 1)

    fz_radius = formation[layer, 2]
    fz_value = formation[layer, 3]
    uz_value = formation[layer, 4]

    # Formation: invaded annulus (r < FZ_RADIUS in layers that have one), else UZ.
    in_fz = ~np.isnan(fz_radius) & (R < fz_radius)
    res = np.where(in_fz, fz_value, uz_value)

    # Mud column: r < r_bh(z), caliper radius interpolated to the point z.
    r_bh = np.interp(Z, borehole[:, 0], borehole[:, 1])
    res = np.where(R < r_bh, mud_resistivity, res)

    return 1.0 / res


def sample_sigma(r_nodes, z_nodes, formation, borehole, mud_resistivity,
                 subsample=4):
    """Cell sigma, shape (nz-1, nr-1), volume-averaged over each cell.

    With subsample=1 this is the sigma at the cell center. With subsample=s
    each cell is sampled on an s x s sub-grid and averaged with the
    axisymmetric volume weight (proportional to r), i.e. an arithmetic
    (parallel) conductivity mix. Material boundaries that fall inside a cell
    then contribute fractionally instead of snapping to the nearest cell
    boundary — without this, layer boundaries shift by up to half a cell as
    the depth-relative grid slides over the formation, which showed up as
    1-4% apparent-resistivity spikes at depths where a measuring electrode
    sits near a layer boundary (Ex1, e.g. A2.0M0.5N at 20.9 m).

    Parameters
    ----------
    r_nodes, z_nodes : ndarray
        Grid nodes (global z).
    formation : (N, 5) ndarray
        [TOP, BOTTOM, FZ_RADIUS, FZ_VALUE, UZ_VALUE] in metres / Ohm.m,
        contiguous ascending layers (as returned by Model.set_formation_parameters).
        NaN FZ columns = no invasion.
    borehole : (M, 3) ndarray
        [DEPT, radius_m, RM] (as returned by Model.set_borehole_parameters).
    mud_resistivity : float
        Mud resistivity (Ohm.m) for this simulation depth.
    subsample : int
        Sub-points per axis for the in-cell average (1 = cell-center lookup).

    Returns
    -------
    sigma : ndarray (nz-1, nr-1), conductivity in S/m; sigma[j, i] belongs to
        the cell (z_nodes[j:j+2]) x (r_nodes[i:i+2]).
    """
    r = np.asarray(r_nodes, dtype=float)
    z = np.asarray(z_nodes, dtype=float)
    s = int(subsample)
    if s <= 1:
        R, Z = np.meshgrid(cell_centers(r), cell_centers(z))
        return _sigma_at_points(R, Z, formation, borehole, mud_resistivity)

    frac = (np.arange(s) + 0.5) / s
    # Sub-points: (nz-1, s) in z, (nr-1, s) in r.
    z_sub = z[:-1, None] + np.diff(z)[:, None] * frac[None, :]
    r_sub = r[:-1, None] + np.diff(r)[:, None] * frac[None, :]
    # Volume weight of a sub-ring ~ its radius (2*pi*r * dr * dz, with dr, dz
    # constant inside a cell); normalize per cell. Guard the axis cell where
    # all radii ~ 0.
    wr = r_sub / np.maximum(r_sub.sum(axis=1, keepdims=True), 1e-300)

    # Broadcast to (nz-1, s_z, nr-1, s_r).
    R = r_sub[None, None, :, :]
    Z = z_sub[:, :, None, None]
    sig = _sigma_at_points(np.broadcast_to(R, (len(z) - 1, s, len(r) - 1, s)),
                           np.broadcast_to(Z, (len(z) - 1, s, len(r) - 1, s)),
                           formation, borehole, mud_resistivity)
    # Average: uniform in z, radius-weighted in r.
    return (sig * wr[None, None, :, :]).sum(axis=3).mean(axis=1)


def sample_sigma_aniso(r_nodes, z_nodes, formation, borehole, mud_resistivity,
                       subsample=4):
    """Direction-dependent cell conductivities (sigma_r, sigma_z).

    Transmissibility upscaling for a cell crossed by material boundaries
    (layered media are the common case here — horizontal boundaries):

      * sigma_z (vertical flux): sub-columns act in SERIES along z ->
        harmonic mean over z sub-points, then the columns combine in
        PARALLEL across r -> radius-weighted arithmetic mean.
      * sigma_r (radial flux): sub-rows act in SERIES along r -> harmonic
        mean over r sub-points (radius-weighted, since the sub-ring
        resistance ~ dr / (sigma * r)), then rows combine in PARALLEL
        across z -> arithmetic mean.

    Exact for boundaries aligned with the cell axes in the thin-cell limit;
    reduces the residual smearing bias of the plain volume average of
    sample_sigma. Returns two (nz-1, nr-1) arrays.
    """
    r = np.asarray(r_nodes, dtype=float)
    z = np.asarray(z_nodes, dtype=float)
    s = int(subsample)
    if s <= 1:
        sig = sample_sigma(r, z, formation, borehole, mud_resistivity,
                           subsample=1)
        return sig, sig

    frac = (np.arange(s) + 0.5) / s
    z_sub = z[:-1, None] + np.diff(z)[:, None] * frac[None, :]
    r_sub = r[:-1, None] + np.diff(r)[:, None] * frac[None, :]
    wr = r_sub / np.maximum(r_sub.sum(axis=1, keepdims=True), 1e-300)

    shape = (len(z) - 1, s, len(r) - 1, s)   # (jz, sz, ir, sr)
    R = np.broadcast_to(r_sub[None, None, :, :], shape)
    Z = np.broadcast_to(z_sub[:, :, None, None], shape)
    sig = _sigma_at_points(R, Z, formation, borehole, mud_resistivity)

    # sigma_z: harmonic over sz (series), then radius-weighted arithmetic
    # over sr (parallel columns).
    harm_z = 1.0 / np.mean(1.0 / sig, axis=1)                  # (jz, ir, sr)
    sigma_z = (harm_z * wr[None, :, :]).sum(axis=2)            # (jz, ir)

    # sigma_r: harmonic over sr with 1/r-weighted resistance (series rings),
    # then arithmetic over sz (parallel rows). Sub-ring resistance ~
    # dr/(sigma*r): weight 1/r normalized per cell.
    inv_r = 1.0 / np.maximum(R, 1e-300)
    w_res = inv_r / inv_r.sum(axis=3, keepdims=True)           # (jz, sz, ir, sr)
    harm_r = 1.0 / (w_res / sig).sum(axis=3)                   # (jz, sz, ir)
    sigma_r = harm_r.mean(axis=1)                              # (jz, ir)

    return sigma_r, sigma_z


def sample_sigma_aniso_batch(r_nodes, z_nodes_rel, z_sims, formation,
                             borehole, muds, subsample=4, chunk=2):
    """Vectorized sample_sigma_aniso over a stack of simulation depths.

    The depth-relative grid is shared; each depth d shifts the z coordinates
    by z_sims[d] and carries its own mud resistivity muds[d]. Results are
    bit-identical to calling sample_sigma_aniso per depth.

    chunk trades Python overhead against cache locality; the work is
    memory-bound numpy, so small chunks win (chunk=2 measured fastest; large
    chunks spill L3 and run ~1.5x slower). CPU sampling remains the setup
    bottleneck overall — porting this lookup to jnp (GPU) is the known next
    step if setup time matters more.

    Returns (sig_r, sig_z) with shape (D, nz-1, nr-1).
    """
    r = np.asarray(r_nodes, dtype=float)
    z = np.asarray(z_nodes_rel, dtype=float)
    z_sims = np.asarray(z_sims, dtype=float)
    muds = np.asarray(muds, dtype=float)
    s = int(subsample)
    D = len(z_sims)
    nzc, nrc = len(z) - 1, len(r) - 1

    frac = (np.arange(s) + 0.5) / s
    z_sub = z[:-1, None] + np.diff(z)[:, None] * frac[None, :]   # (nzc, s)
    r_sub = r[:-1, None] + np.diff(r)[:, None] * frac[None, :]   # (nrc, s)
    wr = r_sub / np.maximum(r_sub.sum(axis=1, keepdims=True), 1e-300)

    sig_r = np.empty((D, nzc, nrc))
    sig_z = np.empty((D, nzc, nrc))
    shape = (nzc, s, nrc, s)
    R1 = np.broadcast_to(r_sub[None, None, :, :], shape)
    inv_r = 1.0 / np.maximum(R1, 1e-300)
    w_res = inv_r / inv_r.sum(axis=3, keepdims=True)

    for lo in range(0, D, int(chunk)):
        hi = min(lo + int(chunk), D)
        d = hi - lo
        cshape = (d,) + shape
        R = np.broadcast_to(R1[None], cshape)
        Z = np.broadcast_to(
            (z_sub[None, :, :, None, None]
             + z_sims[lo:hi, None, None, None, None]), cshape)
        mud = muds[lo:hi, None, None, None, None]
        sig = _sigma_at_points(R, Z, formation, borehole, mud)
        if s == 1:
            flat = sig[:, :, 0, :, 0]
            sig_r[lo:hi] = flat
            sig_z[lo:hi] = flat
            continue
        # axes of sig: (d, nzc, s_z, nrc, s_r)
        harm_z = 1.0 / np.mean(1.0 / sig, axis=2)                # (d, nzc, nrc, s_r)
        sig_z[lo:hi] = (harm_z * wr[None, None, :, :]).sum(axis=3)
        harm_r = 1.0 / (w_res[None] / sig).sum(axis=4)           # (d, nzc, s_z, nrc)
        sig_r[lo:hi] = harm_r.mean(axis=2)

    return sig_r, sig_z


def dedup_foci(values, min_sep):
    """Sorted focus list with neighbours closer than ``min_sep`` merged.

    Noisy per-cell formation models carry hundreds of near-duplicate invasion
    radii; refining the grid around each would blow the radial node count
    (observed: 910 r-nodes on a len512 sample). Foci are only refinement
    anchors — sub-cell sigma averaging represents the boundaries in between —
    so keeping one per ``min_sep`` (a few cells) loses nothing.
    """
    out = []
    for v in sorted(float(v) for v in values):
        if not out or v - out[-1] >= min_sep:
            out.append(v)
    return out


def node_index(coord, nodes, tol=1e-9):
    """Index of ``coord`` in ``nodes``; it must be an exact grid node."""
    nodes = np.asarray(nodes)
    j = int(np.argmin(np.abs(nodes - coord)))
    if abs(float(nodes[j]) - float(coord)) > tol:
        raise ValueError(
            "coordinate {:.9f} is not a grid node (nearest {:.9f}); "
            "add it to the grid foci".format(float(coord), float(nodes[j])))
    return j
