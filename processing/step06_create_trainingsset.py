"""
Create training dataset from metadata and data cubes.
Samples pixels and builds LMDB-backed training/evaluation sets for efficient model training.
"""

import sys
from pathlib import Path
import duckdb
import lmdb
from scipy import stats
import zstandard as zstd
import numpy as np
from tqdm import tqdm
import xarray as xr
import geopandas as gpd
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS
from utils.parall import paral

cfg = {
        "name": "training_set_01",
        "description": "Uniform sample set",
        "num_samples": 5_000_000,
        "map_size": 1024**3 * 1000,  # 1000 GB
        "cube_meta_gpkg": PATHS.meta_data_dir / "training_cube_set.gpkg",
        "duckdb_path": PATHS.meta_data_dir / "pixel_meta_table.duckdb",
        "cubes_path": PATHS.cubes,
        "num_cores": 16,
        #"limit_year": 2023,
        "eval_fraction": 0.2,
    }

def save_config(cfg):
    output_dir = Path(cfg["training_set_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings for JSON
    cfg_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg_serializable, f, indent=4)

def sample_pixels(db_path: Path, excluded_cubes: list, n_samples: int, eval: bool = False):
    con = duckdb.connect(str(db_path))
    excluded_str = ",".join([str(c) for c in excluded_cubes])

    if eval:
        n_samples = int(n_samples * 0.2)  # 20% for eval

        df_samples = con.execute(f"""
            SELECT * FROM (
                SELECT * FROM meta_table 
                WHERE cube_id IN ({excluded_str})
                AND target_f <= 0.25
            ) USING SAMPLE {n_samples} rows
        """).df()
        con.close()
        print(f"Sampled {len(df_samples)} eval pixels from {len(excluded_cubes)} cubes.")
        return df_samples
    else:
        df_samples = con.execute(f"""
            SELECT * FROM (
                SELECT * FROM meta_table 
                WHERE cube_id NOT IN ({excluded_str})
                AND target_f <= 0.25
            ) USING SAMPLE {n_samples} rows
        """).df()
        con.close()
        print(f"Sampled {len(df_samples)} pixels excluding {len(excluded_cubes)} cubes.")
        return df_samples

def sample_pixels_year(
    db_path: Path,
    excluded_cubes: list[int],
    n_samples: int,
    year_limit: int,
    eval_fraction: float = 0.2,
):
    con = duckdb.connect(str(db_path))

    n_eval = int(n_samples * eval_fraction)
    n_train = int(n_samples * (1 - eval_fraction))

    # ---------- TRAIN ----------
    train_conditions = ["target_f <= 0.25", "year <= ?"]
    train_params = [year_limit]

    if excluded_cubes:
        train_conditions.append("cube_id NOT IN ?")
        train_params.append(excluded_cubes)

    # WRAP IN SUBQUERY
    train_query = f"""
        SELECT * FROM (
            SELECT *
            FROM meta_table
            WHERE {' AND '.join(train_conditions)}
        ) USING SAMPLE {n_train} ROWS
    """

    print("sampling training set...")
    df_train = con.execute(train_query, train_params).df()

    # ---------- EVAL ----------
    eval_conditions = ["target_f <= 0.25", "year <= ?"]
    eval_params = [year_limit]

    if excluded_cubes:
        eval_conditions.append("cube_id IN ?")
        eval_params.append(excluded_cubes)

    # WRAP IN SUBQUERY
    eval_query = f"""
        SELECT * FROM (
            SELECT *
            FROM meta_table
            WHERE {' AND '.join(eval_conditions)}
        ) USING SAMPLE {n_eval} ROWS
    """

    print("sampling eval set...")
    df_eval = con.execute(eval_query, eval_params).df()

    con.close()

    print(
        f"Sampled {len(df_train)} train pixels and "
        f"{len(df_eval)} eval pixels "
        f"(cutoff year = {year_limit})."
    )

    return df_train, df_eval

def sample_pixels_updated(db_path: Path, excluded_cubes: list, n_samples: int, eval_fraction: float = 0.2, year_limit: int = 2023):
    con = duckdb.connect(str(db_path))
    excluded_str = ",".join([str(c) for c in excluded_cubes])

    num_train = int(n_samples * (1 - eval_fraction))
    num_eval = int(n_samples * eval_fraction)

    print("Sampling training pixels...")
    df_train = con.execute(f"""
        SELECT * FROM (
            SELECT * FROM meta_table 
            WHERE cube_id NOT IN ({excluded_str})
            AND target_f <= 0.25 AND year <= {year_limit}
        ) USING SAMPLE {num_train} rows
    """).df()
    

    print("Sampling evaluation pixels...")
    df_eval = con.execute(f"""
        SELECT * FROM (
            SELECT * FROM meta_table 
            WHERE cube_id IN ({excluded_str})
            AND target_f <= 0.25 AND year <= {year_limit}
        ) USING SAMPLE {num_eval} rows
    """).df()
    con.close()

    print(f"Sampled {len(df_eval)} eval pixels from {len(excluded_cubes)} cubes.")
    print(f"Sampled {len(df_train)} pixels excluding {len(excluded_cubes)} cubes.")
    return df_train, df_eval

def sample_balanced_pixels(
    db_path: Path,
    excluded_cubes: list[int],
    year_limit: int,
    bin_width: float = 0.05
):
    con = duckdb.connect(str(db_path))
    
    # 1. Prepare conditions
    conditions = ["target_f < 0.25", "year <= ?", "target_d > 0"]
    params = [year_limit]
    if excluded_cubes:
        conditions.append("cube_id IN ?")
        params.append(excluded_cubes)
    
    where_clause = " AND ".join(conditions)

    # 2. Find the bottleneck size using a single query
    # We use a CTE to define the bins consistently
    dist_query = f"""
        WITH binned_data AS (
            SELECT count(*) as n
            FROM meta_table
            WHERE {where_clause}
            GROUP BY floor(target_d / {bin_width})
        )
        SELECT min(n) FROM binned_data
    """
    min_pixels_in_a_bin = con.execute(dist_query, params).fetchone()[0]
    
    if not min_pixels_in_a_bin:
        print("No data found matching criteria.")
        return pd.DataFrame()

    print(f"Bottleneck bin size: {min_pixels_in_a_bin} pixels.")

    # 3. Single-pass Stratified Sampling
    # We assign a random row number to each record within its bin
    final_query = f"""
        WITH ranked_data AS (
            SELECT *,
                row_number() OVER (
                    PARTITION BY floor(target_d / {bin_width}) 
                    ORDER BY random()
                ) as rank
            FROM meta_table
            WHERE {where_clause}
        )
        SELECT * EXCLUDE (rank)
        FROM ranked_data
        WHERE rank <= {min_pixels_in_a_bin}
    """
    
    df_balanced = con.execute(final_query, params).df()
    con.close()
    
    print(f"Total Balanced Set: {len(df_balanced)} pixels.")
    return df_balanced

def save_training_set(df, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / f"{name}.parquet", index=False)
    print(f"Saved {name} subset with {len(df)} samples to {output_dir}")

def extract_and_compress_data(cube_id, meta, cubes_path):
    cube_path = cubes_path / f"{cube_id}.zarr"
    if not cube_path.exists(): return []

    try:
        with xr.open_zarr(cube_path, group="high_res", chunks=None) as ds:
            # Load variable into RAM (approx 2.8GB)
            full_sentle = ds.sentle.values # (band, y, x, time)
            
        y_pts, x_pts = meta["y_idx"].values, meta["x_idx"].values
        plucked = full_sentle[:, y_pts, x_pts, :] # (band, pixels, time)

        # Cross-band validity mask
        # If any band is NaN, set all 12 bands at that timestep to NaN
        any_nan = np.isnan(plucked).any(axis=0) 
        plucked[:, any_nan] = np.nan
        
        # Cast and Transpose to (pixels, band, time)
        plucked = plucked.transpose(1, 0, 2).astype(np.float16)
        plucked = np.ascontiguousarray(plucked)
        
        pixel_keys = meta["pixel_key"].values
        cctx = zstd.ZstdCompressor(level=3)
        
        kv_pairs = []
        for i in range(len(pixel_keys)):
            kv_pairs.append((
                pixel_keys[i].encode("ascii"), 
                cctx.compress(plucked[i].tobytes())
            ))
        return cube_id, kv_pairs
    except Exception as e:
        print(f"Error in {cube_id}: {e}")
        return cube_id, None

def write_sentle_to_lmdb_parallel(df, lmdb_path, map_size, num_cores, cubes_path):
    grouped = df.groupby('cube_id')
    cube_ids, meta_subsets = zip(*[(cid, group) for cid, group in grouped])

    print(f"Extracting {len(df):,} pixels from {len(cube_ids)} cubes...")
    results = paral(
        function=extract_and_compress_data,
        iters=[list(cube_ids), list(meta_subsets), [cubes_path]*len(cube_ids)],
        num_cores=num_cores,
        backend="loky"
    )

    successful = [kv for cube_id, kv in results if kv is not None]
    failed_cubes = [cube_id for cube_id, kv in results if kv is None]

    failed_log = Path(cfg["training_set_path"]) / "failed_cubes.txt"
    if failed_cubes:
        with open(failed_log, "w") as f:
            for cid in failed_cubes:
                f.write(f"{cid}\n")
        print(f"{len(failed_cubes)} cubes failed. Logged to {failed_log}")
    else:
        print("No cubes failed.")

    print("Writing to LMDB...")
    env = lmdb.open(str(lmdb_path), map_size=int(map_size), writemap=True, map_async=True, meminit=False, lock=False)
    with env.begin(write=True) as txn:
        for kv_list in tqdm(successful, desc="Committing"):
            for key, val in kv_list:
                txn.put(key, val)
    env.close()
    print("LMDB writing complete.")

def main():
    # Create output directory
    cfg["training_set_path"] = PATHS.data_dir / "training_sets" / cfg["name"]
    #check if exists and abort if so
    if cfg["training_set_path"].exists():
        print(f"Training set path {cfg['training_set_path']} already exists. Aborting to prevent overwrite.")
        sys.exit(1)
    
    cfg["training_set_path"].mkdir(parents=True, exist_ok=True)
    cfg["lmdb_path"] = cfg["training_set_path"] / "sentle.lmdb"
    save_config(cfg)

    gdf = gpd.read_file(cfg["cube_meta_gpkg"])
    excluded_cubes = gdf.loc[gdf['is_holdout'], 'cube_id'].tolist()
    
    print("Sampling training pixels...")
    df_train = sample_pixels(
        db_path=cfg["duckdb_path"],
        excluded_cubes=excluded_cubes,
        n_samples=cfg["num_samples"],
        eval=False
        )
    print("Sampling eval pixels...")
    df_eval = sample_pixels(
        db_path=cfg["duckdb_path"],
        excluded_cubes=excluded_cubes,
        n_samples=cfg["num_samples"],
        eval=True
    )

    # df_train, df_eval = sample_pixels_year(
    #     db_path=cfg["duckdb_path"],
    #     excluded_cubes=excluded_cubes,
    #     n_samples=cfg["num_samples"],
    #     year_limit=cfg["limit_year"],
    #     eval_fraction = cfg["eval_fraction"],)

    # df = sample_balanced_pixels(
    #     db_path=cfg["duckdb_path"],
    #     excluded_cubes=excluded_cubes,
    #     year_limit=cfg["limit_year"],
    #     bin_width=0.05
    # )
    
    # df1 = pd.read_parquet("/mnt/ssds/mp426/deadwood_forecasting_model/data/training_sets/training_set_01/train_baseline.parquet")
    # overlap_keys = set(df1['pixel_key']).intersection(set(df['pixel_key']))
    # print(f"Overlap with training set 01: {len(overlap_keys)} pixels.")
    # df = df[~df['pixel_key'].isin(overlap_keys)]

    # sample_counts_per_bin = df.groupby(pd.cut(df['target_d'], bins=np.arange(0, 1.05, 0.05))).size()
    # print("Sample counts per target_d bin:")
    # print(sample_counts_per_bin)

    # save_training_set(
    #     df=df,
    #     output_dir=cfg["training_set_path"],
    #     name="uniform_eval_set"
    # )

    assert df_train["pixel_key"].is_unique
    assert df_eval["pixel_key"].is_unique
    assert not df_train["cube_id"].isin(excluded_cubes).any()
    assert df_eval["cube_id"].isin(excluded_cubes).all()
    assert df_train["target_f"].max() <= 0.25
    assert df_eval["target_f"].max() <= 0.25
    # assert df_train["year"].max() <= cfg["limit_year"]
    # assert df_eval["year"].max() <= cfg["limit_year"]

    samples_per_cube = df_train.groupby("cube_id").size()
    print("Training set samples per cube stats:")
    stats = samples_per_cube.agg(["min", "mean", "max", "std"])
    print(stats)

    save_training_set(
        df=df_train,
        output_dir=cfg["training_set_path"],
        name="train_baseline"
    )

    save_training_set(
        df=df_eval,
        output_dir=cfg["training_set_path"],
        name="eval_baseline"
    )

    # df_train = pd.read_parquet(cfg["training_set_path"] / "train_baseline.parquet")
    # df_eval = pd.read_parquet(cfg["training_set_path"] / "eval_baseline.parquet")

    # concatenate train and eval for LMDB writing
    df_all = pd.concat([df_train, df_eval], ignore_index=True)

    write_sentle_to_lmdb_parallel(
        df = df_all,
        lmdb_path = cfg["lmdb_path"],
        map_size=cfg["map_size"],
        num_cores=cfg["num_cores"],
        cubes_path=cfg["cubes_path"]
    )
    print("All processes complete.")

if __name__ == "__main__":
    main()


