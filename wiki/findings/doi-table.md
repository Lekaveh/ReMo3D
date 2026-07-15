---
title: DOI Table — Sensitivity Radius & Depth per Tool
type: finding
tags: [doi, sensitivity, depth-of-investigation, tools, reference]
sources: []
updated: 2026-07-15
---

# DOI Table — radial & vertical extent of sensitivity per tool

Effective **depth of investigation** for each logging tool, computed from the
analytical Born [sensitivity kernel](../concepts/resistivity-logging.md) in a
**homogeneous background**. Each value is the smallest distance from the origin
(borehole axis for radius; measurement point for depth) capturing the given
fraction of that direction's total ``|S|`` — measured independently per
direction (see [method](#method)).

In a homogeneous medium the normalised extent is **geometry-only**: independent
of measurement depth and of ρ₀, so this table is a property of the tool alone.

All values in **metres**. `r` = radial reach (symmetric ±r); `↑` = vertical
reach above the measurement point; `↓` = below. `span` = max electrode offset
from the measurement point.

| Tool | span | r₉₀ | ↑₉₀ | ↓₉₀ | r₉₅ | ↑₉₅ | ↓₉₅ | r₉₉ | ↑₉₉ | ↓₉₉ |
|------|-----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| `A0.4M0.1N` | 0.45 | 0.43 | 0.55 | 0.38 | 0.61 | 0.69 | 0.63 | 1.33 | 1.27 | 1.67 |
| `A1.0M0.1N` | 1.05 | 0.91 | 1.25 | 0.64 | 1.30 | 1.54 | 1.12 | 2.80 | 2.80 | 3.11 |
| `A2.0M0.5N` | 2.25 | 2.12 | 2.74 | 1.78 | 2.99 | 3.41 | 2.99 | 6.45 | 6.22 | 7.94 |
| `A4.0M0.5N` | 4.25 | 3.74 | 5.07 | 2.68 | 5.33 | 6.29 | 4.70 | 11.44 | 11.38 | 12.92 |
| `A8.0M1.0N` | 8.50 | 7.48 | 10.14 | 5.36 | 10.67 | 12.58 | 9.39 | 22.87 | 22.77 | 25.84 |
| `B5.7A0.4M` | 5.90 | 1.57 | 1.27 | 1.01 | 2.49 | 2.56 | 1.68 | 6.00 | 7.54 | 4.81 |
| `B4.48A1.62M` | 5.29 | 3.65 | 4.27 | 2.83 | 5.22 | 6.13 | 4.36 | 11.46 | 11.99 | 10.75 |
| `M1.0A0.1B` | 1.05 | 0.91 | 1.25 | 0.64 | 1.30 | 1.54 | 1.12 | 2.80 | 2.80 | 3.11 |
| `N0.5M2.0A` | 2.25 | 2.12 | 1.78 | 2.74 | 2.99 | 2.99 | 3.41 | 6.45 | 7.94 | 6.22 |
| `M4.0A0.5B` | 4.25 | 3.74 | 5.07 | 2.68 | 5.33 | 6.29 | 4.70 | 11.44 | 11.38 | 12.92 |
| `N2.0M0.5A` | 2.25 | 1.29 | 1.47 | 0.96 | 1.89 | 2.38 | 1.50 | 4.19 | 4.54 | 3.82 |
| `N11.0M0.5A` | 11.25 | 2.22 | 1.65 | 1.37 | 3.69 | 3.41 | 2.43 | 9.39 | 12.63 | 7.28 |

Machine-readable copy: [`../../notebooks/doi_sensitivity_table.csv`](../../notebooks/doi_sensitivity_table.csv).

## What the numbers say

- **Normal tools: `r₉₀ ≈ AM spacing`.** A0.4M0.1N→0.43, A1.0→0.91, A2.0→2.12,
  A4.0→3.74, A8.0→7.48 m — the classic result that a normal tool's radial DOI
  is about its electrode spacing.
- **Reciprocity holds** (a correctness check): `A2.0M0.5N` ≡ `N0.5M2.0A` (same
  `r`, up/down mirrored), and `A4.0M0.5N` ≡ `M4.0A0.5B` (identical). The A↔M /
  B↔N swap does not change the sensitivity, only mirrors it vertically.
- **Lateral tools read shallower than their length.** `B5.7A0.4M` spans 5.9 m
  but r₉₀ = 1.57 m — the response is dominated by the close `AM = 0.4 m` pair,
  not the full B–A separation.
- **Vertical asymmetry is real:** normal `A…N` tools reach further **up** (the
  current electrode A sits above); the reciprocal `N…A` tools reach further
  **down**.

## Caveats

- The high-fraction extents (esp. **99%**) have **heavy tails** — the Born
  kernel in an unbounded homogeneous medium decays slowly, so r₉₉/↑₉₉/↓₉₉ are
  several times larger than the 90% box and are more sensitive to the model
  (no bed boundaries or invasion to truncate the tail). Treat 90% as the robust
  DOI and 95/99% as indicative of tail reach.
- Homogeneous-background Born approximation: in a real layered/invaded model the
  effective DOI is smaller and depth-dependent. See
  [validation](../concepts/validation.md) / [resistivity logging](../concepts/resistivity-logging.md).

## Method

Computed via `remo3d.sensitivity` (folder source, see [[remo3d-research-wiki]]).
For each tool: grow an (r, z) domain — seeded from the electrode span, with a
**fixed cell size scaled to the tool** — until the 99% extent converges (stable
< 2 % and well inside the domain), then read all fractions off the *same*
cumulative-`|S|` curve so they are monotonic by construction. Fixing the cell
size (not the point count) avoids the resolution artefact that otherwise
corrupts the far-tail (99%) values. Directions are measured independently
(radial; vertical split into up/down) — see
[`plot_sensitivity_doi` / `_measure_doi`](../../remo3d/sensitivity.py).

> ⚠️ Note: the interactive `plot_sensitivity_doi` framing uses `_measure_doi`,
> which is tuned for the 90% box; its per-tool 90% values agree with this table
> to ~10%. This table's routine is the reference for multi-fraction DOI.

## Links

- [resistivity logging](../concepts/resistivity-logging.md) (geometric factors, DOI),
  [forward modeling](../concepts/forward-modeling.md), [validation](../concepts/validation.md).
