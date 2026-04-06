# Input and Output Formats

## 8.1 Formation Model File Format

Formation files are tab-delimited text files with:

1. header row
2. units row
3. data rows

Expected columns:

| Column | Meaning |
| --- | --- |
| `TOP` | top depth of the layer |
| `BOTTOM` | bottom depth of the layer |
| `FZ_RADIUS` | filtration-zone radius; `NaN` if absent |
| `FZ_VALUE` | filtration-zone resistivity |
| `UZ_VALUE` | undisturbed-zone resistivity |

Example:

```text
TOP    BOTTOM    FZ_RADIUS    FZ_VALUE    UZ_VALUE
M      M         M            OHMM        OHMM
0      3.05      NaN          NaN         5
3.05   8.35      0.3          3           18
```

Supported geometry units are the keys of `Model.conversion_table`:

- `M`
- `DM`
- `CM`
- `MM`
- `IN`
- `FT`

After loading, geometry is stored in meters and resistivity in ohm-meters.

## 8.2 Borehole Model File Format

Borehole files are also tab-delimited with:

1. header row
2. units row
3. data rows

Expected columns:

| Column | Meaning |
| --- | --- |
| `DEPT` | depth |
| `CALM` | caliper; diameter unless `borehole_geometry_type='radius'` |
| `RM` | mud resistivity |

Example:

```text
DEPT    CALM      RM
M       MM        OHMM
0.0000  236.0941  1.1000
0.1000  243.8112  1.1002
```

Supported borehole geometry units are again:

- `M`
- `DM`
- `CM`
- `MM`
- `IN`
- `FT`

Internally the code stores borehole geometry as radius, not diameter.

## 8.3 Output Results File Format

`save_results` writes one or more text files named `Results_#.txt`.

Structure:

1. first row: column names
2. second row: units
3. following rows: tab-delimited numeric data

Example header:

```text
DEPTH    B5.7A0.4M    B4.48A1.62M
M        OHMM         OHMM
```

Rows then contain:

- first column: measurement depth
- remaining columns: apparent resistivity for each saved tool

Logs are grouped into the same file only when they share the same depth vector.

## 8.4 Visualization Output

`save_results` also produces `Results_plot.png` when `output_folder` is given.

Plot structure:

- left panel: formation model cross-section
- right panels: one or more log tracks
- bottom colorbar: model resistivity scale

### Formation panel

The formation plot includes:

- layered polygons for undisturbed zones
- extra polygons for filtration zones where present
- a filled borehole polygon
- the borehole axis
- the dip angle in the title

### Log panels

Each log track uses a top x-axis. Multiple logs on the same track are stacked by
creating multiple `twiny()` axes offset outward.

Configurable visual parameters:

- `plot_layout`
- `plot_depth_lim`
- `plot_aspect_ratio`
- `model_rad_lim`
- `model_res_lim`
- `logs_res_lim`
- `logs_at_nan`
- `logs_interpolation_factor`
- `logs_colours`

The output format is intentionally simple and human-readable rather than LAS or
DLIS.
## See Also
- [model-api.md](model-api.md#315-save_results): the method that reads and writes these formats.
- [examples-and-tutorials.md](examples-and-tutorials.md#141-tutorial-from-example_01): example usage that produces the documented output files.
- [data-structures.md](data-structures.md#127-logs): in-memory form of the same output data.
