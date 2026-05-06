import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from remo3d import Model

ROOT = Path(__file__).resolve().parents[1]
REMO3D_DIR = ROOT / "remo3d"
if str(REMO3D_DIR) not in sys.path:
    sys.path.insert(0, str(REMO3D_DIR))
    
# Specify input data
#all_tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B"] # logging tools
# all_tools = ['N2.0M0.5A', 'N11.0M0.5A']
all_tools = ['N2.0M0.5A']
# all_tools = ["B5.7A0.4M", "B4.48A1.62M", "M1.0A0.1B", "A2.0M0.5N", "N0.5M2.0A", "M4.0A0.5B", 'N2.0M0.5A', 'N11.0M0.5A']
#formation_model_file = "./Input/Ex1/Formation.txt" # path to file with formation parameters
formation_model_file = np.loadtxt("../notebooks/Input/Ex1/Formation.txt", skiprows=2)
formation_model_file[:, 2:4] = np.nan
borehole_model_file = "../notebooks/Input/Ex1/Borehole.txt" # path to file with borehole parameters
measurement_depths = np.arange(0, 25.1, 0.1) # measurement points

# Create model and simulate logs
model = Model.compute_synthetic_logs(all_tools, measurement_depths, formation_model_file, borehole_model_file, borehole_geometry_type='diameter', dip=0,
                                     cpu_workers=20, gpu_workers=0, domain_radius=50, batch_size=5, mesh_generator='netgen')

from pathlib import Path

# Compare the latest forward run against the validation baseline.
def load_results_txt(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} does not contain the expected header, units, and data rows.")

    columns = lines[0].split("\t")
    units = lines[1].split("\t")
    data = np.loadtxt(path, skiprows=2)
    data = np.atleast_2d(data)
    return columns, units, data


output_root = Path("../notebooks/Output")
validation_path = Path("../notebooks/validation/Results_1.txt")

output_dirs = list(output_root.glob("Results_*"))
if not output_dirs:
    raise FileNotFoundError(f"No result folders were found under {output_root.resolve()}.")
if not validation_path.exists():
    raise FileNotFoundError(f"Validation file was not found at {validation_path.resolve()}.")

latest_output_dir = max(output_dirs, key=lambda path: (path.stat().st_mtime_ns, path.name))
forward_result_path = latest_output_dir / "Results_1.txt"

forward_columns, forward_units, forward_data = load_results_txt(forward_result_path)
validation_columns, validation_units, validation_data = load_results_txt(validation_path)

print(f"Forward result:    {forward_result_path}")
print(f"Validation result: {validation_path}")

validation_column_to_idx = {
    column_name: idx for idx, column_name in enumerate(validation_columns)
}

missing_in_validation = [
    column_name for column_name in forward_columns
    if column_name not in validation_column_to_idx
]

if missing_in_validation:
    raise ValueError(
        "Validation is missing columns that are present in forward: "
        + ", ".join(missing_in_validation)
    )

validation_indices = [
    validation_column_to_idx[column_name]
    for column_name in forward_columns
]

validation_data = validation_data[:, validation_indices]
validation_units = [validation_units[idx] for idx in validation_indices]

if forward_units != validation_units:
    print("Unit mismatch for compared columns:")
    print("  forward   :", forward_units)
    print("  validation:", validation_units)
elif forward_data.shape != validation_data.shape:
    print(f"Shape mismatch: forward {forward_data.shape}, validation {validation_data.shape}")
else:
    diff = forward_data - validation_data
    abs_diff = np.abs(diff)
    max_abs_diff = np.nanmax(abs_diff)
    column_max_abs_diff = np.nanmax(abs_diff, axis=0)
    tolerance = 1e-4
    matches = np.allclose(
        forward_data,
        validation_data,
        atol=tolerance,
        rtol=tolerance,
        equal_nan=True,
    )

    print(f"Match within tolerance {tolerance}: {matches}")
    print(f"Max abs diff: {max_abs_diff:.6g}")
    print("Max abs diff by column:")

    for column_name, column_diff in zip(forward_columns, column_max_abs_diff):
        print(f"  {column_name}: {column_diff:.6g}")

    if not matches:
        row_idx, col_idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
        print("Largest difference:")
        print(f"  row index: {row_idx}")
        print(f"  column: {forward_columns[col_idx]}")
        print(f"  forward value: {forward_data[row_idx, col_idx]:.6g}")
        print(f"  validation value: {validation_data[row_idx, col_idx]:.6g}")
        print(f"  abs diff: {abs_diff[row_idx, col_idx]:.6g}")