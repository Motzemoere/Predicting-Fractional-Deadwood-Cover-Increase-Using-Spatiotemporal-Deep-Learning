"""
Training infrastructure and dataset utilities.
Handles data loading, normalization, sampling, and PyTorch DataLoader creation for model training.
"""

import sys
from pathlib import Path
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import xarray as xr
import torch
import gc
from pyproj import Transformer
from tqdm import tqdm
import pandas as pd
import lmdb
import zstandard as zstd
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
import itertools
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import geopandas as gpd

from utils.parall import paral

def normalize_dataarray(da, stats_group):
    """
    Applies Z-normalization to a DataArray based on a stats dictionary.
    Assumes the first dimension of 'da' contains the variables.
    """
    # Get the name of the dimension (e.g., 'era5_var', 'wc_var', etc.)
    dim_name = da.dims[0]
    var_names = da.coords[dim_name].values
    da = da.astype(np.float32)
    
    for i, var_name in enumerate(var_names):
        var_name_str = str(var_name)
        if var_name_str in stats_group:
            mean = stats_group[var_name_str]["mean"]
            std = stats_group[var_name_str]["std"]
            # Apply normalization: (x - mean) / (std + epsilon)
            da[i] = (da[i] - mean) / (std + 1e-7)     
    return da

def build_era5_mapping_to_subset(cube_crs, cube_x, cube_y, era5_da):
    """Build mapping from cube pixels to indices in ERA5 subset.
    
    Args:
        cube_crs: CRS of the cube
        cube_x, cube_y: Cube coordinate arrays
        era5_da: ERA5 DataArray with dims (era5_var, lat, lon, time) or Dataset
    """
    transformer = Transformer.from_crs(cube_crs, "EPSG:4326", always_xy=True)
    
    # Handle both Dataset and DataArray
    if isinstance(era5_da, xr.Dataset):
        # Get the first variable name if it's a Dataset
        var_names = list(era5_da.data_vars)
        if var_names:
            era5_da = era5_da[var_names[0]]
        else:
            # Try to get the unnamed variable
            era5_da = era5_da["__xarray_dataarray_variable__"]
    
    lon_vals = era5_da.lon.values
    lat_vals = era5_da.lat.values
    
    xx, yy = np.meshgrid(cube_x, cube_y)
    lons, lats = transformer.transform(xx.ravel(), yy.ravel())
    lons = lons.reshape(xx.shape)
    lats = lats.reshape(yy.shape)
    
    def find_nearest_idx_ascending(vals, targets_flat):
        """Find nearest indices assuming vals is sorted in ascending order."""
        idx = np.searchsorted(vals, targets_flat)
        idx = np.clip(idx, 0, len(vals) - 1)
        idx_left = np.maximum(idx - 1, 0)
        dist_left = np.abs(vals[idx_left] - targets_flat)
        dist_right = np.abs(vals[idx] - targets_flat)
        idx = np.where((dist_left < dist_right) & (idx > 0), idx_left, idx)
        return idx
    
    lon_idx = find_nearest_idx_ascending(lon_vals, lons.ravel()).reshape(lons.shape)
    lat_idx = find_nearest_idx_ascending(lat_vals, lats.ravel()).reshape(lats.shape)
    return np.stack([lat_idx, lon_idx], axis=-1)

def build_geotiff_mapping_to_subset(cube_crs, cube_x, cube_y, geotiff_subset):
    """Build mapping from cube pixels to indices in geotiff subset.
    
    Args:
        cube_crs: CRS of the cube
        cube_x, cube_y: Cube coordinate arrays
        geotiff_subset: xarray Dataset or DataArray with x/y coordinates
    """
    transformer = Transformer.from_crs(cube_crs, "EPSG:4326", always_xy=True)
    
    # Handle both Dataset and DataArray
    if isinstance(geotiff_subset, xr.Dataset):
        # Get the first variable name if it's a Dataset
        var_names = list(geotiff_subset.data_vars)
        if var_names:
            geotiff_subset = geotiff_subset[var_names[0]]
        else:
            raise ValueError("No data variables found in geotiff_subset")
    
    x_vals = geotiff_subset.x.values
    y_vals = geotiff_subset.y.values
    
    xx, yy = np.meshgrid(cube_x, cube_y)
    lons, lats = transformer.transform(xx.ravel(), yy.ravel())
    lons = lons.reshape(xx.shape)
    lats = lats.reshape(yy.shape)
    
    def find_nearest_idx_ascending(vals, targets_flat):
        """Find nearest indices assuming vals is sorted in ascending order."""
        idx = np.searchsorted(vals, targets_flat)
        idx = np.clip(idx, 0, len(vals) - 1)
        idx_left = np.maximum(idx - 1, 0)
        dist_left = np.abs(vals[idx_left] - targets_flat)
        dist_right = np.abs(vals[idx] - targets_flat)
        idx = np.where((dist_left < dist_right) & (idx > 0), idx_left, idx)
        return idx
    
    col_idx = find_nearest_idx_ascending(x_vals, lons.ravel()).reshape(lons.shape)
    row_idx = find_nearest_idx_ascending(y_vals, lats.ravel()).reshape(lats.shape)
    return np.stack([row_idx, col_idx], axis=-1)

def build_netcdf_mapping_to_subset(cube_crs, cube_x, cube_y, netcdf_subset):
    """Build mapping from cube pixels to indices in NetCDF subset.
    
    Args:
        cube_crs: CRS of the cube
        cube_x, cube_y: Cube coordinate arrays
        netcdf_subset: xarray Dataset or DataArray with lat/lon or latitude/longitude coordinates
    """
    transformer = Transformer.from_crs(cube_crs, "EPSG:4326", always_xy=True)
    
    # Handle both Dataset and DataArray
    if isinstance(netcdf_subset, xr.Dataset):
        # Get the first variable name if it's a Dataset
        var_names = list(netcdf_subset.data_vars)
        if var_names:
            netcdf_subset = netcdf_subset[var_names[0]]
        else:
            raise ValueError("No data variables found in netcdf_subset")
    
    lon_coord = 'longitude' if 'longitude' in netcdf_subset.coords else 'lon'
    lat_coord = 'latitude' if 'latitude' in netcdf_subset.coords else 'lat'
    lon_vals = netcdf_subset[lon_coord].values
    lat_vals = netcdf_subset[lat_coord].values
    
    xx, yy = np.meshgrid(cube_x, cube_y)
    lons, lats = transformer.transform(xx.ravel(), yy.ravel())
    lons = lons.reshape(xx.shape)
    lats = lats.reshape(yy.shape)
    
    def find_nearest_idx_ascending(vals, targets_flat):
        """Find nearest indices assuming vals is sorted in ascending order."""
        idx = np.searchsorted(vals, targets_flat)
        idx = np.clip(idx, 0, len(vals) - 1)
        idx_left = np.maximum(idx - 1, 0)
        dist_left = np.abs(vals[idx_left] - targets_flat)
        dist_right = np.abs(vals[idx] - targets_flat)
        idx = np.where((dist_left < dist_right) & (idx > 0), idx_left, idx)
        return idx
    
    lon_idx = find_nearest_idx_ascending(lon_vals, lons.ravel()).reshape(lons.shape)
    lat_idx = find_nearest_idx_ascending(lat_vals, lats.ravel()).reshape(lats.shape)
    return np.stack([lat_idx, lon_idx], axis=-1)

def load_and_prepare_data(data_dir, cube_id, stats, include_sentle=False):
    """Load and prepare data for a single cube. Stats dict is passed directly to avoid redundant I/O.
    
    Args:
        data_dir: Directory containing zarr cubes
        cube_id: Cube identifier
        stats: Normalization statistics dictionary
        include_sentle: If True, include Sentinel data (needed for prediction, not training)
    """
    zarr_path = data_dir / f"{cube_id}.zarr"

    # Load high-resolution datasets
    # Use consolidated=False for local files - avoids reading .zmetadata 5 times per cube
    # chunks=None avoids dask overhead
    high_res_ds = xr.open_zarr(zarr_path, group="high_res", consolidated=False, chunks=None)

    # use only specific years for training
    #high = xr.open_zarr(zarr_path, group="high_res", consolidated=False, chunks=None)
    #high_res_ds = high.sel(time=slice(None, "2023-12-31"), time_year=slice(None, "2023-12-31"))


    high_res_ds["ch"] = normalize_dataarray(high_res_ds["ch"], stats["canopy_height"])
    terrain_vars = high_res_ds.terrain.coords["terrain_var"].values
    terrain_da = high_res_ds["terrain"].astype(np.float32)
    for i, v in enumerate(terrain_vars):
        v_str = str(v)
        if v_str in stats["terrain"] and "aspect" not in v_str:
            m, s = stats["terrain"][v_str]["mean"], stats["terrain"][v_str]["std"]
            terrain_da[i] = (terrain_da[i] - m) / (s + 1e-7)
    high_res_ds["terrain"] = terrain_da
    
    # Load low-resolution datasets
    # consolidated=False for local files, chunks=None to avoid dask overhead
    era5_ds = xr.open_zarr(zarr_path, group="era5", consolidated=False, chunks=None)
    
    # use only specific years for training
    #era5 = xr.open_zarr(zarr_path, group="era5", consolidated=False, chunks=None)
    #era5_ds = era5.sel(time=slice(None, "2023-12-31"))

    era5_da = era5_ds["era5"]
    drop_vars = {'lai_hv_q95', 'lai_hv_q05', 'lai_hv_mean'}
    keep_vars = [v for v in era5_da.era5_var.values if v not in drop_vars]
    era5_da = era5_da.sel(era5_var=keep_vars)
    era5_da = normalize_dataarray(era5_da, stats["era5"])
    
    worldclim_ds = xr.open_zarr(zarr_path, group="worldclim", consolidated=False, chunks=None)
    worldclim_da = normalize_dataarray(worldclim_ds["worldclim"], stats["worldclim"])
    
    soilgrids_ds = xr.open_zarr(zarr_path, group="soilgrids", consolidated=False, chunks=None)
    soilgrids_da = normalize_dataarray(soilgrids_ds["soilgrids"], stats["soilgrids"])
    
    stand_age_ds = xr.open_zarr(zarr_path, group="stand_age", consolidated=False, chunks=None)
    stand_age_da = normalize_dataarray(stand_age_ds["stand_age"], stats["stand_age"])
    
    # Create coordinate mappings
    cube_crs = high_res_ds.rio.crs
    cube_x = high_res_ds.x.values
    cube_y = high_res_ds.y.values
    
    era5_mapping = build_era5_mapping_to_subset(cube_crs, cube_x, cube_y, era5_da)
    wc_mapping = build_geotiff_mapping_to_subset(cube_crs, cube_x, cube_y, worldclim_da)
    sg_mapping = build_geotiff_mapping_to_subset(cube_crs, cube_x, cube_y, soilgrids_da)
    sa_mapping = build_netcdf_mapping_to_subset(cube_crs, cube_x, cube_y, stand_age_da)

    # Load into numpy arrays
    res_dict = {
        "ch_data": high_res_ds["ch"].load().values,
        "dw_data": high_res_ds["deadwood_forest"].load().values,
        "terrain_data": high_res_ds["terrain"].load().values,
        "era5_data": era5_da.load().values,
        "worldclim_data": worldclim_da.load().values,
        "soilgrids_data": soilgrids_da.load().values,
        "stand_age_data": stand_age_da.load().values,
        "era5_mapping": era5_mapping,
        "wc_mapping": wc_mapping,
        "sg_mapping": sg_mapping,
        "sa_mapping": sa_mapping,
        "H": high_res_ds.sizes["y"],
        "W": high_res_ds.sizes["x"],
        "time_year_len": high_res_ds.sizes["time_year"],
        "time_len": high_res_ds.sizes["time"]
    }
    
    # Optionally include Sentinel data for prediction
    if include_sentle:
        res_dict["sentle_data"] = high_res_ds["sentle"].load().values

    # Basic shape sanity checks to catch unexpected layouts early
    # def _expect_shape(name, arr, ndim, leading_name):
    #     if arr.ndim != ndim:
    #         raise ValueError(f"{name} expected {ndim} dims ({leading_name}, ...), got {arr.ndim} dims: {arr.shape}")
    #     if arr.shape[0] <= 0:
    #         raise ValueError(f"{name} has empty {leading_name} dimension: {arr.shape}")

    # _expect_shape("ERA5", era5_data, 4, "era5_v")       # (era5_v, lat, lon, time)
    # _expect_shape("WorldClim", wc_data, 3, "wc_v")      # (wc_v, y, x)
    # _expect_shape("SoilGrids", sg_data, 3, "sg_v")      # (sg_v, y, x)
    # _expect_shape("Stand Age", sa_data, 3, "stand_age_v")  # (stand_age_v, lat, lon)
    
    for ds in [high_res_ds, era5_ds, worldclim_ds, soilgrids_ds, stand_age_ds]:
        ds.close()
    
    return res_dict

def materialize_and_share(data_dict, include_sentle=False):
    def to_shared_tensor(data, dtype=torch.float32):
        """Helper to convert data to a shared torch tensor efficiently."""
        if not isinstance(data, torch.Tensor):
            data_np = np.array(data, copy=True) 
            tensor = torch.from_numpy(data_np)
        else:
            tensor = data
        return tensor.to(dtype).clone().share_memory_()

    result = {
        "dw_tensor": to_shared_tensor(data_dict["dw_data"]),
        "terrain_tensor": to_shared_tensor(data_dict["terrain_data"]),
        "canopy_tensor": to_shared_tensor(data_dict["ch_data"]),
        "era5_tensor": to_shared_tensor(data_dict["era5_data"]),
        "wc_tensor": to_shared_tensor(data_dict["worldclim_data"]),
        "sg_tensor": to_shared_tensor(data_dict["soilgrids_data"]),
        "sa_tensor": to_shared_tensor(data_dict["stand_age_data"]),
        "year_pos": torch.arange(data_dict["time_year_len"], dtype=torch.long).share_memory_(),
        "week_pos": torch.arange(data_dict["time_len"], dtype=torch.long).share_memory_(),
        "H": data_dict["H"],
        "W": data_dict["W"],
        "era5_mapping": data_dict["era5_mapping"],
        "wc_mapping": data_dict["wc_mapping"],
        "sg_mapping": data_dict["sg_mapping"],
        "sa_mapping": data_dict["sa_mapping"],
    }
    
    # Optionally include Sentinel tensor for prediction
    if include_sentle and "sentle_data" in data_dict:
        result["sentle_tensor"] = to_shared_tensor(data_dict["sentle_data"], dtype=torch.float16)
    
    return result

def preload_all_cubes(cube_dir, stats_path):
    """
    Loops through a list of cube IDs, loads their data, computes spatial mappings,
    and moves auxiliary data (ERA5, Static, Terrain, Canopy) to shared memory.
    """
    # Load stats once upfront
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    ram_database = {}
    failed_cubes = []
    cube_ids = [p.stem for p in cube_dir.glob("*.zarr")]

    for cid in tqdm(cube_ids, desc="Preloading Cubes to RAM"):
        print(f"\nPreloading cube {cid}...")
        try:
            raw_data = load_and_prepare_data(cube_dir, cid, stats)
            shared_data = materialize_and_share(raw_data)
    
            ram_database[str(cid)] = shared_data
            del raw_data
            
        except Exception as e:
            print(f"\n[ERROR] Failed to preload cube {cid}: {e}")
            failed_cubes.append(cid)
            continue

    gc.collect()
    
    print(f"\nPreload Complete!")
    print(f"Successfully loaded: {len(ram_database)} cubes")
    if failed_cubes:
        print(f"Failed cubes: {len(failed_cubes)}")

    return ram_database

def worker_preload(cid, cube_dir, stats):
    """
    Worker function to process a single cube.
    Returns a tuple of (cube_id, shared_data) or (cube_id, None) on failure.
    Stats dict is passed directly to avoid redundant JSON reads.
    """
    try:
        raw_data = load_and_prepare_data(cube_dir, cid, stats)
        shared_data = materialize_and_share(raw_data)
        return str(cid), shared_data
    except Exception as e:
        print(f"\n[ERROR] Failed to preload cube {cid}: {e}")
        return str(cid), None

def preload_all_cubes_parallel(cube_dir, stats_path, num_cores, cube_ids=None):
    """
    Parallel version of preload_all_cubes.
    Uses threading backend for I/O-bound zarr operations (faster than loky for this workload).
    """
    # Load stats ONCE before spawning workers
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    if cube_ids is None:
        cube_ids = [p.stem for p in cube_dir.glob("*.zarr")]
        print(f"Starting parallel preload of {len(cube_ids)} cubes using {num_cores} cores...")
    else:
        print(f"Starting parallel preload of {len(cube_ids)} specified cubes using {num_cores} cores...")
    # Prepare arguments - pass stats dict directly instead of path
    iters = [
        cube_ids, 
        itertools.repeat(cube_dir), 
        itertools.repeat(stats)  # Pass dict, not path
    ]

    # Execute parallel processing
    # Use threading backend - better for I/O bound zarr operations
    results = paral(
        function=worker_preload,
        iters=iters,
        num_cores=num_cores,
        backend="threading"  # Threading is faster for I/O-bound zarr reads
    )

    # Reconstruct the database
    ram_database = {}
    failed_cubes = []
    
    for cid, shared_data in results:
        if shared_data is not None:
            ram_database[cid] = shared_data
        else:
            failed_cubes.append(cid)

    print(f"\nPreload Complete!")
    print(f"Successfully loaded: {len(ram_database)} cubes")
    if failed_cubes:
        print(f"Failed cubes count: {len(failed_cubes)}")

    return ram_database

def load_training_subsets(output_dir):
    output_dir = Path(output_dir)
    train_baseline = pd.read_parquet(output_dir / "train_baseline.parquet")
    train_rare = pd.read_parquet(output_dir / "train_rare.parquet")
    val_df = pd.read_parquet(output_dir / "eval_natural.parquet")
    
    return train_baseline, val_df, train_rare

def load_training_set(output_dir, name = "train_baseline"):
    output_dir = Path(output_dir)
    df = pd.read_parquet(output_dir / f"{name}.parquet")
    return df

def split_training_subset(df, ratio=0.2, random_state=42):
    cube_ids = df['cube_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(cube_ids)
    split_idx = int(len(cube_ids) * (1 - ratio))
    train_cubes = cube_ids[:split_idx]
    val_cubes = cube_ids[split_idx:]
    
    train_df = df[df['cube_id'].isin(train_cubes)].reset_index(drop=True)
    val_df = df[df['cube_id'].isin(val_cubes)].reset_index(drop=True)
    return train_df, val_df

def split_folds(df, gdf, fold_num):
    cv_folds = gdf[gdf['fold'] != 0]
    val_cubes = cv_folds[cv_folds['fold'] == fold_num]['cube_id'].unique()
    train_cubes = cv_folds[cv_folds['fold'] != fold_num]['cube_id'].unique()

    train_df = df[df['cube_id'].isin(train_cubes)].reset_index(drop=True)
    eval_df = df[df['cube_id'].isin(val_cubes)].reset_index(drop=True)
    return train_df, eval_df

def pixel_lmdb_reader(lmdb_path: Path, pixel_key: str) -> np.ndarray:
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=128,
    )
    with env.begin(write=False) as txn:
        compressed = txn.get(pixel_key.encode("ascii"))
        if compressed is None:
            raise KeyError(f"Pixel key {pixel_key} not found in LMDB.")
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)
        ts = np.frombuffer(raw, dtype=np.float16)
    env.close()
    return ts

def extract_patch_indices(p_size, y_indices, x_indices, array_shape):
    if p_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer.")
    H, W = array_shape
    pad = p_size // 2
    rel = np.arange(-pad, pad + 1)
    y_patch = y_indices[:, None, None] + rel[None, :, None]
    x_patch = x_indices[:, None, None] + rel[None, None, :]
    y_patch = np.broadcast_to(y_patch, (len(y_indices), p_size, p_size))
    x_patch = np.broadcast_to(x_patch, (len(x_indices), p_size, p_size))
    valid_mask = (y_patch >= 0) & (y_patch < H) & (x_patch >= 0) & (x_patch < W)
    y_patch_clipped = np.clip(y_patch, 0, H - 1)
    x_patch_clipped = np.clip(x_patch, 0, W - 1)

    return y_patch_clipped, x_patch_clipped, valid_mask

class TrainingDataset(Dataset):
    def __init__(self, pixel_table, *, shared_tensors, lmdb_path, patch_size, num_years, num_weeks, augment):
        self.shared_tensors = shared_tensors
        self.lmdb_path = lmdb_path
        self.patch_size = patch_size
        self.num_years = num_years
        self.num_weeks = num_weeks

        # Do NOT open LMDB here. 
        # Opening it in __init__ makes the handle "fork-unsafe".
        self.env = None 
        self.dctx = zstd.ZstdDecompressor()
        self.augment = augment

        # Inices as NumPy
        self.cube_ids = pixel_table['cube_id'].values.astype(str)
        self.pixel_keys = pixel_table['pixel_key'].values
        self.y_indices = pixel_table['y_idx'].values.astype(np.int32)
        self.x_indices = pixel_table['x_idx'].values.astype(np.int32)
        self.year_indices = pixel_table['year_idx'].values.astype(np.int16)
        self.week_indices = pixel_table['week_idx'].values.astype(np.int16)

        # Targets as Torch Tensors
        self.target_ds = torch.from_numpy(pixel_table['target_d'].values).float()
        self.target_fs = torch.from_numpy(pixel_table['target_f'].values).float()
        self.target_sums = torch.from_numpy(pixel_table['target_sum'].values).float()

        if 'sample_weight' in pixel_table.columns:
            self.sample_weights = torch.from_numpy(pixel_table['sample_weight'].values).float()
        else:
            self.sample_weights = torch.ones(len(pixel_table), dtype=torch.float32)
            
        self.years = torch.from_numpy(pixel_table['year'].values).int()
        self.length = len(self.target_sums)

    def _init_db(self):
        """Initialize LMDB once per worker process."""
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256 # Increased for many workers
        )
    def __len__(self):
        return len(self.target_sums)

    def __getitem__(self, idx):
        # 1. Lazy init for multi-worker safety
        if self.env is None:
            self._init_db()

        cube_id = self.cube_ids[idx]
        pixel_key = self.pixel_keys[idx]
        cube_data = self.shared_tensors[cube_id]

        y_idx = self.y_indices[idx].item()
        x_idx = self.x_indices[idx].item()
        curr_year_idx = self.year_indices[idx].item()
        curr_week_idx = self.week_indices[idx].item()
        
        target_d = self.target_ds[idx]
        target_f = self.target_fs[idx]
        target_sum = self.target_sums[idx]
        sample_weight = self.sample_weights[idx]
        year = self.years[idx].item()

        H, W = cube_data["H"], cube_data["W"]

        # LOOK-BACK SLICES
        y_slice_start = curr_year_idx - (self.num_years - 1) 
        actual_y_start = max(0, y_slice_start)
        w_slice_start = curr_week_idx - (self.num_weeks - 1)
        actual_w_start = max(0, w_slice_start)

        patch_y_idx, patch_x_idx, patch_valid_mask = extract_patch_indices(
            self.patch_size, np.array([y_idx]), np.array([x_idx]), (H, W)
        )
        patch_y, patch_x, valid_mask = patch_y_idx[0], patch_x_idx[0], patch_valid_mask[0]
        valid_mask_t = torch.from_numpy(valid_mask)
        
        # High-res patches
        dw_patch = cube_data["dw_tensor"][:, patch_y, patch_x, actual_y_start : curr_year_idx + 1]
        dw_patch = torch.where(valid_mask_t[None, :, :, None], dw_patch, torch.nan)
        terrain_patch = cube_data["terrain_tensor"][:, patch_y, patch_x]
        terrain_patch = torch.where(valid_mask_t[None, :, :], terrain_patch, torch.nan)
        canopy_patch = cube_data["canopy_tensor"][:, patch_y, patch_x]
        canopy_patch = torch.where(valid_mask_t[None, :, :], canopy_patch, torch.nan)

        # Low_res pixels
        e5_r, e5_c = cube_data['era5_mapping'][y_idx, x_idx]
        era5_pixel = cube_data["era5_tensor"][:, e5_r, e5_c, actual_w_start : curr_week_idx + 1]

        wc_r, wc_c = cube_data['wc_mapping'][y_idx, x_idx]
        wc_pixel = cube_data["wc_tensor"][:, wc_r, wc_c]
        
        sg_r, sg_c = cube_data['sg_mapping'][y_idx, x_idx]
        sg_pixel = cube_data["sg_tensor"][:, sg_r, sg_c]
        
        sa_r, sa_c = cube_data['sa_mapping'][y_idx, x_idx]
        sa_pixel = cube_data["sa_tensor"][:, sa_r, sa_c]

        with self.env.begin(write=False) as txn:
            compressed = txn.get(pixel_key.encode("ascii"))
            if compressed is None:
                raise KeyError(f"Key {pixel_key} not found")
            raw = self.dctx.decompress(compressed)
            ts_flat = np.frombuffer(raw, dtype=np.float16).copy()
            num_channels = 12 
            pixel_sentle = ts_flat.reshape(num_channels, -1)  # [C, T]
            pixel_sentle = pixel_sentle[:, actual_w_start : curr_week_idx + 1]

        pixel_sentle_t = torch.from_numpy(pixel_sentle).to(torch.float32)

        pad_y = (self.num_years) - dw_patch.shape[-1]
        pad_w = (self.num_weeks) - era5_pixel.shape[-1]

        # F.pad takes pairs of (padding_left, padding_right) for each dimension starting from the last
        if pad_y > 0:
            # Pad the last dimension (time) on the left with NaNs
            dw_patch = F.pad(dw_patch, (pad_y, 0), value=float('nan'))
            
        if pad_w > 0:
            # Pad the last dimension (time) on the left with 0s (or NaNs)
            era5_pixel = F.pad(era5_pixel, (pad_w, 0), value=float('nan'))
            pixel_sentle_t = F.pad(pixel_sentle_t, (pad_w, 0), value=float('nan'))
    
        
        if self.augment:
            # We choose a random transformation
            # 0: original, 1: flip horizontal, 2: flip vertical, 3: rot90, 4: rot180, 5: rot270
            aug_type = torch.randint(0, 6, (1,)).item()

            def apply_aug(t, is_4d=False):
                # For 4D (dw_patch): [C, H, W, T] -> Spatial dims are 1, 2
                # For 3D (terrain/canopy): [C, H, W] -> Spatial dims are 1, 2
                # For 2D (valid_mask): [H, W] -> Spatial dims are 0, 1
                
                # Determine spatial dimension indices based on tensor rank
                if t.ndim == 4:
                    spatial_dims = (1, 2)
                elif t.ndim == 3:
                    spatial_dims = (1, 2)
                else: # 2D mask
                    spatial_dims = (0, 1)

                h_dim, w_dim = spatial_dims

                if aug_type == 1: # Flip Horizontal
                    return torch.flip(t, dims=[h_dim]) 
                elif aug_type == 2: # Flip Vertical
                    return torch.flip(t, dims=[w_dim]) 
                elif aug_type == 3: # Rot 90
                    return torch.rot90(t, k=1, dims=[h_dim, w_dim])
                elif aug_type == 4: # Rot 180
                    return torch.rot90(t, k=2, dims=[h_dim, w_dim])
                elif aug_type == 5: # Rot 270
                    return torch.rot90(t, k=3, dims=[h_dim, w_dim])
                return t

            dw_patch = apply_aug(dw_patch)
            terrain_patch = apply_aug(terrain_patch)
            canopy_patch = apply_aug(canopy_patch)
            valid_mask_t = apply_aug(valid_mask_t)

        return {
            "deadwood_forest": dw_patch,           
            "terrain": terrain_patch,              
            "canopy": canopy_patch,                 
            "era5": era5_pixel,                     
            "pixels_sentle": pixel_sentle_t,
            "wc": wc_pixel,
            "sg": sg_pixel,
            "stand_age": sa_pixel,
            "target_d": target_d,
            "target_f": target_f,
            "target_sum": target_sum,
            "y_idx": y_idx,
            "x_idx": x_idx,
            "year": year,
            "pixel_key": pixel_key,
            "patch_mask": valid_mask_t.to(torch.float32),
            "sample_weight": sample_weight
        }

def get_dataloader(shared_tensors, 
                   pixel_table, 
                   lmdb_path, 
                   batch_size=128, 
                   num_workers=16, 
                   patch_size=33, 
                   num_years=3, 
                   num_weeks=156,
                   sampler_type=None,
                   augment=False):
    
    dataset = TrainingDataset(shared_tensors=shared_tensors, 
                              pixel_table=pixel_table, 
                              lmdb_path=lmdb_path, 
                              patch_size=patch_size, 
                              num_years=num_years, 
                              num_weeks=num_weeks,
                              augment=augment)

    if sampler_type == "weighted":
        # 1. Extract weights from the pixel_table 
        # (Assuming you added 'sample_weight' via the DuckDB step)
        if 'sample_weight' not in pixel_table.columns:
            raise ValueError("pixel_table must contain 'sample_weight' for weighted sampling.")
        
        weights = torch.from_numpy(pixel_table['sample_weight'].values).double()
        
        # 2. Create the WeightedRandomSampler
        # replacement=True is critical: it allows rare high-value pixels 
        # to be picked multiple times in one epoch.
        sampler = WeightedRandomSampler(
            weights=weights, 
            num_samples=len(weights), 
            replacement=True
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, # Use our new sampler object
            num_workers=num_workers, 
            pin_memory=True,
            shuffle=False # Shuffle must be False when using a sampler
        )
    
    elif sampler_type is None:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
    
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

class RareInfusionDataset(torch.utils.data.Dataset):
    def __init__(self, baseline_dataset, rare_dataset, rare_ratio=0.10):
        self.baseline_ds = baseline_dataset
        self.rare_ds = rare_dataset
        self.rare_ratio = rare_ratio
        self.len = len(baseline_dataset)

    def set_ratio(self, ratio):
        """Explicitly update the infusion ratio."""
        self.rare_ratio = ratio

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # random.random() is safe here; PyTorch handles worker seeding
        if random.random() < self.rare_ratio:
            rare_idx = random.randint(0, len(self.rare_ds) - 1)
            return self.rare_ds[rare_idx]
        else:
            return self.baseline_ds[idx]

####################
# Setup Prediction
####################
def create_prediction_table(cube_zarr: Path, cube_id, years: list, threshold: float, bin_width: float):
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

                # year_data shape: (4, H, W, 2)
                
                # Fast mask
                mask = (year_data[0, :, :, 0] > threshold) | (year_data[1, :, :, 0] > threshold)
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

            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df

class PredictionDataset(Dataset):
    def __init__(self, shared_tensors, pixel_table, patch_size, num_years, num_weeks):
        self.shared_tensors = shared_tensors
        self.patch_size = patch_size
        self.num_years = num_years
        self.num_weeks = num_weeks

        # Inices as NumPy
        self.cube_ids = pixel_table['cube_id'].values.astype(str)
        self.pixel_keys = pixel_table['pixel_key'].values
        self.y_indices = pixel_table['y_idx'].values.astype(np.int32)
        self.x_indices = pixel_table['x_idx'].values.astype(np.int32)
        self.year_indices = pixel_table['year_idx'].values.astype(np.int16)
        self.week_indices = pixel_table['week_idx'].values.astype(np.int16)

        # Targets as Torch Tensors
        self.target_ds = torch.from_numpy(pixel_table['target_d'].values).float()
        self.target_fs = torch.from_numpy(pixel_table['target_f'].values).float()
        self.target_sums = torch.from_numpy(pixel_table['target_sum'].values).float()

        if 'sample_weight' in pixel_table.columns:
            self.sample_weights = torch.from_numpy(pixel_table['sample_weight'].values).float()
        else:
            self.sample_weights = torch.ones(len(pixel_table), dtype=torch.float32)
            
        self.years = torch.from_numpy(pixel_table['year'].values).int()
        self.length = len(self.target_sums)

        # Check if all cube_ids are the same
        unique_cubes = np.unique(self.cube_ids)
        if len(unique_cubes) > 1:
            raise ValueError("pixel_table contains multiple cube_ids...")

    def __len__(self):
        return len(self.target_sums)

    def __getitem__(self, idx):
        cube_id = self.cube_ids[idx]
        pixel_key = self.pixel_keys[idx]
        cube_data = self.shared_tensors

        y_idx = self.y_indices[idx].item()
        x_idx = self.x_indices[idx].item()
        curr_year_idx = self.year_indices[idx].item()
        curr_week_idx = self.week_indices[idx].item()
        
        target_d = self.target_ds[idx]
        target_f = self.target_fs[idx]
        target_sum = self.target_sums[idx]
        sample_weight = self.sample_weights[idx]
        year = self.years[idx].item()

        H, W = cube_data["H"], cube_data["W"]

        # LOOK-BACK SLICES
        y_slice_start = curr_year_idx - (self.num_years - 1) 
        actual_y_start = max(0, y_slice_start)
        w_slice_start = curr_week_idx - (self.num_weeks - 1)
        actual_w_start = max(0, w_slice_start)

        patch_y_idx, patch_x_idx, patch_valid_mask = extract_patch_indices(
            self.patch_size, np.array([y_idx]), np.array([x_idx]), (H, W)
        )
        patch_y, patch_x, valid_mask = patch_y_idx[0], patch_x_idx[0], patch_valid_mask[0]
        valid_mask_t = torch.from_numpy(valid_mask)
        
        # High-res patches
        dw_patch = cube_data["dw_tensor"][:, patch_y, patch_x, actual_y_start : curr_year_idx + 1]
        dw_patch = torch.where(valid_mask_t[None, :, :, None], dw_patch, torch.nan)
        terrain_patch = cube_data["terrain_tensor"][:, patch_y, patch_x]
        terrain_patch = torch.where(valid_mask_t[None, :, :], terrain_patch, torch.nan)
        canopy_patch = cube_data["canopy_tensor"][:, patch_y, patch_x]
        canopy_patch = torch.where(valid_mask_t[None, :, :], canopy_patch, torch.nan)

        # Low_res pixels
        e5_r, e5_c = cube_data['era5_mapping'][y_idx, x_idx]
        era5_pixel = cube_data["era5_tensor"][:, e5_r, e5_c, actual_w_start : curr_week_idx + 1]

        wc_r, wc_c = cube_data['wc_mapping'][y_idx, x_idx]
        wc_pixel = cube_data["wc_tensor"][:, wc_r, wc_c]
        
        sg_r, sg_c = cube_data['sg_mapping'][y_idx, x_idx]
        sg_pixel = cube_data["sg_tensor"][:, sg_r, sg_c]
        
        sa_r, sa_c = cube_data['sa_mapping'][y_idx, x_idx]
        sa_pixel = cube_data["sa_tensor"][:, sa_r, sa_c]

        # Sentle
        sentle_pixel = cube_data["sentle_tensor"][:, y_idx, x_idx, actual_w_start : curr_week_idx + 1]
        # Set all timesteps to NaN where not all bands are valid
        sentle_pixel = torch.where(torch.isfinite(sentle_pixel).all(dim=0, keepdim=True), sentle_pixel, torch.nan)
        # to float32
        pixel_sentle_t = sentle_pixel.to(torch.float32)

        pad_y = (self.num_years) - dw_patch.shape[-1]
        pad_w = (self.num_weeks) - era5_pixel.shape[-1]

        # F.pad takes pairs of (padding_left, padding_right) for each dimension starting from the last
        if pad_y > 0:
            # Pad the last dimension (time) on the left with NaNs
            dw_patch = F.pad(dw_patch, (pad_y, 0), value=float('nan'))
            
        if pad_w > 0:
            # Pad the last dimension (time) on the left with 0s (or NaNs)
            era5_pixel = F.pad(era5_pixel, (pad_w, 0), value=float('nan'))
            pixel_sentle_t = F.pad(pixel_sentle_t, (pad_w, 0), value=float('nan'))

        return {
            "deadwood_forest": dw_patch,           
            "terrain": terrain_patch,              
            "canopy": canopy_patch,                 
            "era5": era5_pixel,                     
            "pixels_sentle": pixel_sentle_t,
            "wc": wc_pixel,
            "sg": sg_pixel,
            "stand_age": sa_pixel,
            "target_d": target_d,
            "target_f": target_f,
            "target_sum": target_sum,
            "y_idx": y_idx,
            "x_idx": x_idx,
            "year": year,
            "pixel_key": pixel_key,
            "patch_mask": valid_mask_t.to(torch.float32),
            "sample_weight": sample_weight
        }

def get_prediction_dataloader(shared_tensors: dict, 
                              pixel_table: pd.DataFrame, 
                              batch_size=128, 
                              num_workers=16, 
                              patch_size=33, 
                              num_years=3, 
                              num_weeks=156):
    
    dataset = PredictionDataset(shared_tensors=shared_tensors, 
                                pixel_table=pixel_table, 
                                patch_size=patch_size, 
                                num_years=num_years, 
                                num_weeks=num_weeks)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

def predict_cube(target, model, dataloader, device, feature_keys, criterion, return_predictions=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    outputs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting Cube"):
            inputs = {key: batch[key].to(device) for key in feature_keys}
            targets = batch[target].to(device).unsqueeze(-1)
            targets_log = torch.log1p(targets)

            preds_log = model(**inputs)
            preds = torch.expm1(preds_log) # Back to original scale

            loss = criterion(preds_log, targets_log)
            
            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            if return_predictions:
                outputs_list.append({
                    "preds": preds.cpu().numpy(),
                    "targets": batch[target].numpy(),
                    "year": batch["year"].numpy(),
                    "pixel_keys": np.array(batch["pixel_key"])
                    })

    # Final metrics calculation on full dataset
    y_true = torch.cat(all_targets).squeeze().numpy().flatten()
    y_pred = torch.cat(all_preds).squeeze().numpy().flatten()
    
    # 1. Distribution Masks
    thresh_90 = np.percentile(y_true, 90)
    high_mask = y_true >= thresh_90
    low_mask = y_true < thresh_90

    # 2. Key Baseline Metrics
    avg_loss = total_loss / total_samples
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Natural Forest Performance (The Low End)
    low_mae = mean_absolute_error(y_true[low_mask], y_pred[low_mask]) if low_mask.any() else 0.0
    # Rare Event Performance (The High End)
    high_mae = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.any() else 0.0
    
    # 3. Variance & Trend Metrics
    std_target = y_true.std()
    std_pred = y_pred.std()
    std_ratio = std_pred / (std_target + 1e-8)
    
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    bias = (y_pred - y_true).mean()
    
    metrics = {
        "loss": avg_loss,
        "r2": r2,
        "mae": mae,
        "low_mae": low_mae,
        "high_mae": high_mae,
        "std_ratio": std_ratio,
        "corr": corr,
        "bias": bias,
        "pred_std": std_pred
    }
    if return_predictions:
        # df with true/pred columns
        all_preds = np.concatenate([o["preds"] for o in outputs_list]).flatten()
        all_targets = np.concatenate([o["targets"] for o in outputs_list]).flatten()
        all_keys = np.concatenate([o["pixel_keys"] for o in outputs_list])
        all_years = np.concatenate([o["year"] for o in outputs_list]).flatten()

        results_df = pd.DataFrame({
            "pixel_key": all_keys,
            "prediction": all_preds,
            "target": all_targets,
            "year": all_years
        })

        
        return metrics, results_df
    return metrics

def load_config(run_dir: Path):
    json_path = Path(run_dir) / "config.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    else:
        raise FileNotFoundError("No run.yaml or config.json found in run directory.")

def setup_cube(cube_id: int, cubes_dir: Path, meta_dir: Path):
    cube_path = cubes_dir / f"{cube_id}.zarr"
    stats_dir = meta_dir / "znorm_stats.json"
    with open(stats_dir) as f:
     stats = json.load(f)

    df = create_prediction_table(
        cube_zarr=cube_path,
        cube_id=cube_id,
        years=[2020, 2021, 2022, 2023, 2024],
        threshold=0.1,
        bin_width=0.1
    )
    data_dict = load_and_prepare_data(cubes_dir, cube_id, stats, include_sentle=True)
    shared_tensors = materialize_and_share(data_dict, include_sentle=True)
    del data_dict
    return df, shared_tensors

def setup_prediction(*, cfg: dict, data_dir: Path, df: pd.DataFrame, shared_tensors: dict, device: str):
    run_dir = data_dir / "training_runs" / cfg.name
    
    pred_loader = get_prediction_dataloader(
        shared_tensors,
        df,
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        patch_size=cfg.patch_size, 
        num_years=cfg.max_years, 
        num_weeks=cfg.max_weeks)

    # Setup model
    if cfg.model_variant == "v2":
        from models.model2 import MultimodalDeadwoodTransformer
    if cfg.model_variant == "small":
        from models.model_small import MultimodalDeadwoodTransformer
    else:
        from models.model import MultimodalDeadwoodTransformer
    model = MultimodalDeadwoodTransformer(embed_dim=cfg.embed_dim).to(device)
    model_path = Path(run_dir / "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return pred_loader, model

def run_prediction(cube_id: int, data_dir: Path, cfg: dict, pred_loader, model, device: str, feature_keys, criterion, df: pd.DataFrame):
    results_path = data_dir / "predictions" / f"predictions_{cfg["name"]}" / f"{cube_id}_predictions"
    metrics, results_df = predict_cube(
    cfg.get("target_name"),
    model, 
    pred_loader, 
    device, 
    feature_keys,
    criterion,
    return_predictions=True)

    cube_results = df.merge(results_df, on=['pixel_key', 'year'])
    cube_results_path = results_path / f"{cube_id}_predictions.parquet"
    os.makedirs(cube_results_path.parent, exist_ok=True)
    cube_results.to_parquet(cube_results_path)
    print(f"Saved predictions to {cube_results_path}")
    return metrics, cube_results

def built_cube_array(cube_id: int, cubes_dir: Path, results_df: pd.DataFrame):
    results_df['year'] = results_df['year'] + 1
    # Create array to hold predictions and targets
    H = results_df['H_cube'].iloc[0]
    W = results_df['W_cube'].iloc[0]
    all_years = sorted(results_df['year'].unique())
    num_years = len(all_years)
    pred_array = np.full((H, W, num_years), np.nan)
    target_array = np.full((H, W, num_years), np.nan)

    # Fill arrays
    for i, year in enumerate(all_years):
        yr_data = results_df[results_df['year'] == year]
        pred_array[yr_data['y_idx'].values, yr_data['x_idx'].values, i] = yr_data['prediction'].values
        target_array[yr_data['y_idx'].values, yr_data['x_idx'].values, i] = yr_data['target'].values

    # select only relevant years from deadwood_forest
    start_year = all_years[0]
    end_year = all_years[-1]
    start_year_dt = np.datetime64(f"{start_year}-01-01")
    end_year_dt = np.datetime64(f"{end_year}-12-31")

    with xr.open_zarr(cubes_dir / f"{cube_id}.zarr", group="high_res", chunks=None) as ds:
        deadwood_forest = ds['deadwood_forest']
        ds_selected = deadwood_forest.sel(time_year=slice(start_year_dt, end_year_dt))

        y_coords = ds_selected.coords['y']
        x_coords = ds_selected.coords['x']
        time_coords = ds_selected.coords['time_year']

        ds_output = xr.Dataset(
            data_vars={
                "prediction": (["y", "x", "time_year"], pred_array),
                "target":     (["y", "x", "time_year"], target_array),
            },
            coords={
                "y": y_coords,
                "x": x_coords,
                "time_year": time_coords
            },
            attrs=ds_selected.attrs  # Preserve original metadata (CRS, etc.)
        )
    return ds_output

####################
# Setup Training
####################

def setup_training_datasets(cfg, paths):
    training_set_path = paths.training_sets / cfg.training_set
    lmdb_path = training_set_path / "sentle.lmdb"

    df = load_training_set(training_set_path, "train_baseline")
    cube_ids = df['cube_id'].unique().tolist()
    if cfg.fold is not None:
        gdf = gpd.read_file(paths.meta_data_dir / "training_cube_set.gpkg")
        
        df_folds, _ = split_folds(df, gdf, cfg.fold)
        df_train, df_val = split_training_subset(df_folds, ratio=cfg.val_ratio)
    else:
        df_train, df_val = split_training_subset(
            df, ratio=cfg.val_ratio
        )
        

    all_cubes = preload_all_cubes_parallel(
        paths.cubes,
        stats_path=paths.meta_data_dir / "znorm_stats.json",
        num_cores=16,
        cube_ids=cube_ids
    )

    ds_kwargs = dict(
        shared_tensors=all_cubes,
        lmdb_path=lmdb_path,
        patch_size=cfg.patch_size,
        num_years=cfg.max_years,
        num_weeks=cfg.max_weeks,
    )

    ds_train = TrainingDataset(df_train, augment=True, **ds_kwargs)
    ds_val = TrainingDataset(df_val, augment=False, **ds_kwargs)

    return ds_train, ds_val, df_train, df_val

def build_training_dataloaders(cfg, ds_train, ds_val):
    loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    loader_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return loader_train, loader_val

def setup_evaluation_dataset(cfg, paths):
    # Eval set is in the trainingset
    training_set_path = paths.training_sets / cfg.training_set
    lmdb_path = training_set_path / "sentle.lmdb"

    if cfg.fold is not None:
        gdf = gpd.read_file(paths.meta_data_dir / "training_cube_set.gpkg")
        ds = load_training_set(training_set_path)
        _ , df_eval = split_folds(ds, gdf, cfg.fold)
    else:
        df_eval = load_training_set(training_set_path, "eval_baseline")

    cube_ids = df_eval['cube_id'].unique().tolist()


    all_cubes = preload_all_cubes_parallel(
        paths.cubes,
        stats_path=paths.meta_data_dir / "znorm_stats.json",
        num_cores=16,
        cube_ids=cube_ids
    )

    ds_kwargs = dict(
        shared_tensors=all_cubes,
        lmdb_path=lmdb_path,
        patch_size=cfg.patch_size,
        num_years=cfg.max_years,
        num_weeks=cfg.max_weeks,
    )

    eval_ds = TrainingDataset(df_eval, augment=False, **ds_kwargs)

    return eval_ds, df_eval

def build_evaluation_dataloader(cfg, eval_ds):
    loader_eval = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return loader_eval

def setup_custom_dataset(cfg, name, dir, paths):
    # Eval set is in the trainingset
    
    lmdb_path = dir / "sentle.lmdb"
 
    df = load_training_set(dir, name)

    cube_ids = df['cube_id'].unique().tolist()


    all_cubes = preload_all_cubes_parallel(
        paths.cubes,
        stats_path=paths.meta_data_dir / "znorm_stats.json",
        num_cores=16,
        cube_ids=cube_ids
    )

    ds_kwargs = dict(
        shared_tensors=all_cubes,
        lmdb_path=lmdb_path,
        patch_size=cfg.patch_size,
        num_years=cfg.max_years,
        num_weeks=cfg.max_weeks,
    )

    ds = TrainingDataset(df, augment=False, **ds_kwargs)

    return ds, df

def build_custom_dataloader(cfg, eval_ds):
    loader_eval = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return loader_eval
