"""
Calculate mean statistics for Sentinel and deadwood data.
Computes global mean values across all cubes for baseline initialization in Integrated Gradients.
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
OUTPUT_FILE = PATHS.meta_data_dir / "mean_sentinel_deadwood_stats.json"

NUM_WORKERS = 16

def get_local_stats_deadwood(data_array):
    # Compute sum and count of non-NaN pixels per channel
    # dims_to_reduce: ['x', 'y', 'time_year']
    valid_count = data_array.notnull().sum(dim=['x', 'y', 'time_year']).values
    total_sum = data_array.sum(dim=['x', 'y', 'time_year'], skipna=True).values
    
    return {
        "sum": total_sum,
        "count": valid_count
    }

def get_local_stats_sentle(data_array):
    # 1. Compute valid count (this usually won't overflow as it's an int)
    valid_count = data_array.notnull().sum(dim=['x', 'y', 'time']).values
    
    # 2. Compute total sum - FORCE dtype to float64 to prevent overflow
    # We use .astype(np.float64) before the sum to ensure the accumulator is large enough
    total_sum = data_array.astype(np.float64).sum(dim=['x', 'y', 'time'], skipna=True).values
    
    return {
        "sum": total_sum,
        "count": valid_count
    }

def compute_cube_stats(zarr_path: Path):
    cube_id = zarr_path.stem
    try:
        with xr.open_zarr(zarr_path, group="high_res") as ds:
            sentle_stats = get_local_stats_sentle(ds['sentle'])
            deadwood_stats = get_local_stats_deadwood(ds['deadwood_forest'])

            return {
                "sentle": sentle_stats,
                "deadwood_forest": deadwood_stats
            }
    except Exception as e:
        print(f"Error processing cube {cube_id}: {e}")
        return None

def main():
    zarr_stores = list(CUBES_ROOT.glob("*.zarr"))
    print(f"Calculating stats for {len(zarr_stores)} cubes...")
    results = paral(
        compute_cube_stats, 
        iters=[zarr_stores], 
        num_cores=NUM_WORKERS, 
        return_as="list"
    )

    # Aggregation
    global_sum_dw = np.zeros(4, dtype=np.float64)
    global_count_dw = np.zeros(4, dtype=np.float64)
    global_sum_sent = np.zeros(12, dtype=np.float64)
    global_count_sent = np.zeros(12, dtype=np.float64)
    processed_count = 0

    # 2. Loop through the results list
    for cube_result in results:
        if cube_result is None:
            continue   
        processed_count += 1
        # Aggregate Deadwood
        dw = cube_result.get("deadwood_forest")
        if dw:
            global_sum_dw += dw["sum"]
            global_count_dw += dw["count"]   
        # Aggregate Sentinel
        sent = cube_result.get("sentle")
        if sent:
            global_sum_sent += sent["sum"]
            global_count_sent += sent["count"]

    # Calculate Final Means
    with np.errstate(divide='ignore', invalid='ignore'):
        final_mean_deadwood = np.nan_to_num(global_sum_dw / global_count_dw)
        final_mean_sentinel = np.nan_to_num(global_sum_sent / global_count_sent)

    print(f"\n--- Processing Complete ---")
    print(f"Cubes successfully aggregated: {processed_count}/{len(zarr_stores)}")
    print(f"Global Deadwood Mean: {final_mean_deadwood}")
    print(f"Global Sentinel Mean: {final_mean_sentinel}")

    # Save to JSON
    stats_to_save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "cubes_count": processed_count
        },
        "deadwood_forest": final_mean_deadwood.tolist(),
        "sentinel": final_mean_sentinel.tolist()
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(stats_to_save, f, indent=4)
        
if __name__ == "__main__":
    main()