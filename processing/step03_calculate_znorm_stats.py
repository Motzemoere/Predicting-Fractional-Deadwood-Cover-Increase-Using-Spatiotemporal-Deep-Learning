"""
Calculate z-normalization statistics across all data cubes.
Computes mean and standard deviation for feature normalization during model training.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import os
import csv
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

from utils.parall import paral

CUBES_ROOT = PATHS.cubes
LOG_FILE = PATHS.logs / "03_calculate_znorm_stats.csv"
OUTPUT_FILE = PATHS.meta_data_dir / "znorm_stats.json"

# Processing constants
SAMPLE_SIZE = None  # Set to an integer to sample cubes, or None for all
NUM_WORKERS = 16 
# Variables to skip (Categorical, Targets, or already -1 to 1)
SKIP_VARS = ["forest_type", "sin_aspect", "cos_aspect", "deadwood_forest", "sentle"]

def log_event(cube_id, status, message=""):
    """Logs the status of a cube processing task."""
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "cube_id", "status", "message"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cube_id, status, message])

def get_local_stats(group_name, data, dims_to_reduce):
    """Computes running stats for a single DataArray."""
    count = data.count(dim=dims_to_reduce).values
    sum_val = data.sum(dim=dims_to_reduce, skipna=True).values
    sq_sum = (data**2).sum(dim=dims_to_reduce, skipna=True).values

    # The first dimension is 'stacked' dimension
    main_dim = data.dims[0]
    labels = [str(x) for x in data.coords[main_dim].values] if main_dim in data.coords else [group_name]

    return {
        "sum": sum_val.astype(np.float64),
        "sq": sq_sum.astype(np.float64),
        "count": count.astype(np.float64),
        "coords": labels
    }

def compute_cube_stats(zarr_path: Path):
    """Worker function: Processes one cube and returns a stats dictionary."""
    cube_id = zarr_path.stem
    cube_stats = {}
    
    try:
        # High-Res Group (contains ch, terrain, deadwood_forest, sentle)
        with xr.open_zarr(zarr_path, group="high_res") as ds:
            if 'ch' in ds: 
                cube_stats['canopy_height'] = get_local_stats('canopy_height', ds.ch, ['y', 'x'])
            if 'terrain' in ds: 
                cube_stats['terrain'] = get_local_stats('terrain', ds.terrain, ['y', 'x'])

        # ERA5
        with xr.open_zarr(zarr_path, group="era5") as ds:
            # Dimension names are lat, lon, time
            cube_stats['era5'] = get_local_stats('era5', ds.era5, ['lat', 'lon', 'time'])

        # WorldClim
        with xr.open_zarr(zarr_path, group="worldclim") as ds:
            cube_stats['worldclim'] = get_local_stats('worldclim', ds.worldclim, ['y', 'x'])

        # SoilGrids
        with xr.open_zarr(zarr_path, group="soilgrids") as ds:
            cube_stats['soilgrids'] = get_local_stats('soilgrids', ds.soilgrids, ['y', 'x'])

        # Stand Age
        with xr.open_zarr(zarr_path, group="stand_age") as ds:
            # Stand age uses latitude/longitude
            cube_stats['stand_age'] = get_local_stats('stand_age', ds.stand_age, ['latitude', 'longitude'])

        log_event(cube_id, "Success")
        return cube_stats

    except Exception as e:
        log_event(cube_id, "Fail", str(e))
        return None

def main():
    zarr_stores = list(CUBES_ROOT.glob("*.zarr"))

    print(f"Logging to: {LOG_FILE}")
    print(f"Calculating stats for {len(zarr_stores)} cubes...")

    results = paral(
        compute_cube_stats, 
        iters=[zarr_stores], 
        num_cores=NUM_WORKERS, 
        return_as="generator"
    )

    # --- REDUCE STEP ---
    global_stats = {}
    success_count = 0
    
    for cube_res in results:
        if cube_res is None: continue
        success_count += 1
        
        for group, stats in cube_res.items():
            if group not in global_stats:
                global_stats[group] = stats
            else:
                global_stats[group]["sum"] += stats["sum"]
                global_stats[group]["sq"] += stats["sq"]
                global_stats[group]["count"] += stats["count"]

    if not global_stats:
        print("Error: No stats were successfully calculated. Check the log.")
        return

    # --- FINAL CALCULATION ---
    print(f"Aggregation complete ({success_count} cubes). Computing final Z-scores...")
    final_output = {}
    
    for group_name, stats in global_stats.items():
        count = np.where(stats['count'] == 0, 1.0, stats['count'])
        mean = stats['sum'] / count
        # variance = E[X^2] - (E[X])^2
        variance = np.maximum((stats['sq'] / count) - (mean ** 2), 0)
        std = np.sqrt(variance)

        group_dict = {}
        labels = stats['coords']
        
        if isinstance(mean, np.ndarray) and mean.ndim > 0:
            for i, var_label in enumerate(labels):
                if var_label in SKIP_VARS: continue
                group_dict[var_label] = {"mean": float(mean[i]), "std": float(std[i])}
        else:
            group_dict = {"mean": float(mean), "std": float(std)}
        
        final_output[group_name] = group_dict

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Success! Final stats saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()