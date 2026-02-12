"""
Create pixel-level metadata table from data cubes.
Extracts forest/deadwood statistics and target variables for training dataset creation.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import duckdb
import time
import os
import itertools
import sys
import csv
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS
from utils.parall import paral

####################
# Path Configuration (from centralized config)
####################
CUBES_PATH = PATHS.cubes
DUCKDB_PATH = PATHS.meta_data_dir / "pixel_meta_table.duckdb"
PARQUET_TEMP_DIR = PATHS.meta_data_dir / "tmp_parquet_extraction"
LOG_FILE = PATHS.logs / "04_create_training_meta_table.csv"

# Processing constants
THRESHOLD = 0.1  # Minimum fraction to consider as forest/deadwood
YEARS = [2020, 2021, 2022, 2023, 2024]  # Years for which we have data
BIN_WIDTH = 0.1  # Standard bin width for classification
MEMORY_LIMIT = "16GB"
PRAGMA_TEMP_DIR = "/mnt/ssds/mp426/tmp"
NUM_CORES = 16

####################
# Logging Functions (consistent with other scripts)
####################
def setup_logger(log_file):
    """Initialize log file with header if it doesn't exist."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("timestamp,cube_id,status,num_samples,error_msg\n")
    return log_file

def log_result(log_file, cube_id, status, num_samples=0, error_msg=""):
    """Log a single result to the CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_msg = str(error_msg).replace(",", ";").replace("\n", " ")
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{cube_id},{status},{num_samples},{clean_msg}\n")

def get_successful_cubes(log_file):
    """Read log file and return set of successfully processed cube_ids."""
    try:
        if not log_file.exists():
            return set()
        df = pd.read_csv(log_file)
        if df.empty:
            return set()
        # Consider SUCCESS and NO_DATA as completed (no need to retry)
        completed = df[df['status'].isin(['SUCCESS', 'NO_DATA'])]['cube_id'].astype(str).tolist()
        return set(completed)
    except Exception:
        return set()

def process_cube_worker(cube_id, years, threshold, bin_width):
    """
    Processes a single cube. 
    Saves to parquet on success. 
    Saves an empty '.nodata' file if no forest pixels are found.
    """
    cube_zarr = CUBES_PATH / f"{cube_id}.zarr"
    out_parquet = PARQUET_TEMP_DIR / f"{cube_id}.parquet"
    no_data_marker = PARQUET_TEMP_DIR / f"{cube_id}.nodata"
    
    # 1. Faster Resume Logic: If parquet or marker exists, we are done.
    if out_parquet.exists():
        return {"cube_id": cube_id, "status": "ALREADY_DONE", "num_samples": 0}
    if no_data_marker.exists():
        return {"cube_id": cube_id, "status": "ALREADY_DONE_NO_DATA", "num_samples": 0}

    try:
        # chunks=None avoids Dask overhead when reading local Zarr in parallel processes
        with xr.open_zarr(cube_zarr, group="high_res", chunks=None) as ds:
            data = ds["deadwood_forest"]
            y_coords = ds.y.values.astype(np.float32)
            x_coords = ds.x.values.astype(np.float32)
            time_year_coords = pd.to_datetime(ds.time_year.values)
            time_unix = ds.time_unix.values 
            year_unix = ds.time_year_unix.values 
            H, W = len(y_coords), len(x_coords)
            
            all_dfs = []
            for year in years:
                year_indices = np.where(time_year_coords.year == year)[0]
                if len(year_indices) == 0: continue
                year_idx = year_indices[0]
                next_idx = year_idx + 1
                if next_idx >= len(time_year_coords): continue

                # Load only required time slices into memory once per year
                # d_f slices: 0:dead_now, 1:forest_now, 2:dead_next, 3:forest_next
                year_data = data.isel(time_year=[year_idx, next_idx]).values
                
                # Fast mask
                # structure: [channel, height, width, year]
                #mask = (year_data[0, :, :, 0] > threshold) | (year_data[1, :, :, 0] > threshold)
                mask = year_data[1, :, :, 0] > threshold  # Only forest presence needed

                y_idx, x_idx = np.nonzero(mask)
                if len(y_idx) == 0: continue

                target_year_data = year_data[:, :, :, 1]
                t_d = target_year_data[2, y_idx, x_idx] # deadwood increase
                t_f = target_year_data[3, y_idx, x_idx] # forest increase
                t_sum = t_d + t_f
                
                # Fast vectorized binning
                bins = np.clip(np.floor((t_sum + 1e-9) / bin_width), 0, int(1.0/bin_width)).astype(np.int16)

                n = len(y_idx)
                all_dfs.append(pd.DataFrame({
                    "pixel_key": [f"{cube_id}_{year}_{i}" for i in range(n)],
                    "cube_id": np.full(n, int(cube_id), np.int32),
                    "year": np.full(n, year, np.int16),
                    "year_unix": np.full(n, year_unix[year_idx], np.int64),
                    "year_idx": np.full(n, year_idx, np.int16),
                    "week_idx": np.full(n, (np.searchsorted(time_unix, year_unix[year_idx], side="right") - 1), np.int32),
                    "y_idx": y_idx.astype(np.int32),
                    "x_idx": x_idx.astype(np.int32),
                    "coords_y": y_coords[y_idx],
                    "coords_x": x_coords[x_idx],
                    "H_cube": np.full(n, H, np.int32),
                    "W_cube": np.full(n, W, np.int32),
                    "target_d": t_d.astype(np.float32),
                    "target_f": t_f.astype(np.float32),
                    "target_sum": t_sum.astype(np.float32),
                    "class": bins
                }))

            if not all_dfs:
                no_data_marker.touch()
                return {"cube_id": cube_id, "status": "NO_DATA", "num_samples": 0}

            # Save the Parquet file
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_parquet(out_parquet, index=False)
            return {"cube_id": cube_id, "status": "SUCCESS", "num_samples": len(final_df)}

    except Exception as e:
        return {"cube_id": cube_id, "status": "FAILED", "num_samples": 0, "error": str(e)}

# --- Main Logic ---
def main():
    PARQUET_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    log_file = setup_logger(LOG_FILE)
    
    # Identify cubes
    all_cube_ids = [f.stem for f in CUBES_PATH.glob("*.zarr")]

    if not all_cube_ids:
        print(f"No .zarr files found in {CUBES_PATH}")
        return
    
    # Check for already completed cubes (from log file)
    completed_cubes = get_successful_cubes(log_file)
    
    # Also check for parquet/nodata markers (resume logic)
    for cid in all_cube_ids:
        if (PARQUET_TEMP_DIR / f"{cid}.parquet").exists() or \
           (PARQUET_TEMP_DIR / f"{cid}.nodata").exists():
            completed_cubes.add(cid)
    
    cube_ids = [cid for cid in all_cube_ids if cid not in completed_cubes]
    
    print(f"Total cubes: {len(all_cube_ids)}")
    print(f"Already completed: {len(completed_cubes)}")
    print(f"Remaining to process: {len(cube_ids)}")
    print(f"Logging to: {log_file}")
    
    if not cube_ids:
        print("All cubes already processed. Skipping to DuckDB ingestion...")
    else:
        # Run Parallel Extraction
        iters=[
            cube_ids,                      
            itertools.repeat(YEARS), 
            itertools.repeat(THRESHOLD), 
            itertools.repeat(BIN_WIDTH)
        ]

        results = paral(
            function=process_cube_worker,
            iters=iters,
            num_cores=NUM_CORES, 
            backend="loky"
        )

        # Log results
        success_count = 0
        fail_count = 0
        no_data_count = 0
        
        for r in results:
            status = r['status']
            if status.startswith("ALREADY"):
                continue  # Don't re-log already done cubes
            
            log_result(
                log_file, 
                r['cube_id'], 
                status, 
                r['num_samples'], 
                r.get('error', '')
            )
            
            if status == "SUCCESS":
                success_count += 1
            elif status == "NO_DATA":
                no_data_count += 1
            elif status == "FAILED":
                fail_count += 1
                print(f"  ✗ Cube {r['cube_id']} failed: {r.get('error', 'Unknown')}")
        
        print(f"\nProcessing complete: {success_count} success, {no_data_count} no data, {fail_count} failed")

    # Final Step: Bulk Ingest into DuckDB
    # This is extremely fast because DuckDB reads parquet files in parallel.
    print("\nIngesting Parquet files into DuckDB...")
    con = duckdb.connect(str(DUCKDB_PATH))
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta_table (
            pixel_key TEXT PRIMARY KEY, cube_id INTEGER, year INTEGER, year_unix BIGINT, 
            year_idx INTEGER, week_idx INTEGER, y_idx INTEGER, x_idx INTEGER, 
            coords_y FLOAT, coords_x FLOAT, H_cube INTEGER, W_cube INTEGER, 
            target_d FLOAT, target_f FLOAT, target_sum FLOAT, class INTEGER
        )
    """)

    parquet_glob = str(PARQUET_TEMP_DIR / "*.parquet")
    # check if any parquets exist before trying to read
    if any(PARQUET_TEMP_DIR.glob("*.parquet")):
        con.execute(f"INSERT OR IGNORE INTO meta_table SELECT * FROM read_parquet('{parquet_glob}')")
    
    con.close()
    print(f"Processing complete. Results stored in {DUCKDB_PATH}")

    #Quickly check how many samples were processed
    if DUCKDB_PATH.exists():
        con = duckdb.connect(str(DUCKDB_PATH))
        count = con.execute("SELECT COUNT(*) FROM meta_table").fetchone()[0]
        con.close()
        print(f"Total samples in meta_table: {count}")

if __name__ == "__main__":
    main()

