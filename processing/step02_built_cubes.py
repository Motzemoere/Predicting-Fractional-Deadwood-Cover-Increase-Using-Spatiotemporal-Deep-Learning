"""
Build multi-modal data cubes from S3 sources.
Downloads and processes Sentinel, predictions, and geospatial data into zarr cubes.
"""

from glob import glob
import geopandas as gpd
import s3fs
import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
from pyproj import Transformer
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
import dask
import logging
import time
import gc
from shapely.geometry import box
import sys
import zarr

# Fix for "Error in sys.excepthook" spam in terminal
def custom_excepthook(exc_type, exc_value, exc_tb):
    """Custom exception hook to avoid broken pipe errors in terminal."""
    try:
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    except (BrokenPipeError, IOError):
        pass

sys.excepthook = custom_excepthook

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS, DATA_SOURCES, S3_CONFIG

# =================
# Path Configuration (from centralized config)
# =================
LOG_FILE = PATHS.logs / "02_built_cubes.csv"
OUT_DIR = PATHS.cubes
CUBE_METADATA = PATHS.meta_data_dir / "balanced_cube_set.gpkg"

# S3 Configuration
RUN_ID = S3_CONFIG["run_id"]
PRED_DIR = f"{S3_CONFIG['pred_dir']}/{RUN_ID}"
SENTINEL_DIR = S3_CONFIG["sentinel_dir"]

# Processing constants
START_YEAR = 2018
REF_BAND = 'B01'
NUM_WORKERS = 16

# Use DATA_SOURCES from centralized config (replaces DATA_SOURCES_RAW)
# Access via: DATA_SOURCES["worldclim"], DATA_SOURCES["era5"], etc.

# Lazy S3 connection - only created when first needed (makes module import-safe)
_FS = None

def get_fs():
    """Get or create S3 filesystem connection (lazy initialization)."""
    global _FS
    if _FS is None:
        _FS = s3fs.S3FileSystem(
            key=os.environ.get("S3_ACCESS_KEY", "TLXEHW2FHZUXADKPZ6YK"),
            secret=os.environ.get("S3_SECRET_KEY", "Q7Kri6BK+xxkkQ7lB2xjtmqZIHHkI3lLebvbwu7Z"),
            client_kwargs={
                "endpoint_url": S3_CONFIG["endpoint_url"],
                "region_name": S3_CONFIG["region_name"]
            },
        )
    return _FS

def retrieve_S3_blocks(block_id):
    """Load Sentinel and prediction cubes from S3."""
    fs = get_fs()
    prediction_path = f"{PRED_DIR}/{block_id}_{RUN_ID}_inference.zarr"
    path_sentinel = f"{SENTINEL_DIR}/{block_id}.zarr"

    missing = [p for p in (path_sentinel, prediction_path) if not fs.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing S3 objects: {', '.join(missing)}")

    s3_store = s3fs.S3Map(root=prediction_path, s3=fs, check=False)
    ds_pred = xr.open_zarr(s3_store, consolidated=True)
    wkt_str_pred = ds_pred.attrs.get("crs_wkt")
    input_crs_pred = rxr.rioxarray.crs.CRS.from_wkt(wkt_str_pred)
    if ds_pred.rio.crs is None:
        ds_pred = ds_pred.rio.write_crs(input_crs_pred)

    ds_sentle = xr.open_zarr(s3fs.S3Map(root=path_sentinel, s3=fs, check=False), consolidated=True)
    wkt_str_sentle = ds_sentle.attrs.get("crs_wkt")
    input_crs_sentle = rxr.rioxarray.crs.CRS.from_wkt(wkt_str_sentle)
    if ds_sentle.rio.crs is None:
        ds_sentle = ds_sentle.rio.write_crs(input_crs_sentle)

    if ds_sentle.rio.crs != ds_pred.rio.crs:
        raise ValueError("CRS mismatch between Sentinel and Prediction cubes")

    return ds_sentle, ds_pred

def subset_cube(ds, minx, miny, maxx, maxy):
    """
    Subsets an xarray Dataset or DataArray based on UTM coordinates.
    Handles both ascending and descending coordinate order.
    """
    # Determine if y is ascending or descending to ensure slice works
    y_is_descending = ds.y.values[0] > ds.y.values[-1]
    if y_is_descending:
        y_slice = slice(maxy, miny)
    else:
        y_slice = slice(miny, maxy)
    x_slice = slice(minx, maxx)
    return ds.sel(x=x_slice, y=y_slice)

def preprocess_prediction_cube(ds_pred):
    """Preprocess prediction cube and compute increments."""
    
    # Rename time to time_year and sort
    ds_processed = ds_pred.rename({"time": "time_year"}).sortby("time_year")
    ds_processed["time_year"] = pd.to_datetime(ds_processed["time_year"].values)
    time_year_vals = ds_processed.time_year.values
    time_year_unix = (time_year_vals.astype("int64") // 1_000_000_000).astype(np.int64)
    ds_processed = ds_processed.assign_coords(time_year_unix=("time_year", time_year_unix))
    
    # Fill NaNs with 0
    deadwood_data = ds_processed["deadwood"].fillna(0)
    forest_data = ds_processed["forest"].fillna(0)
    
    # First normalize from uint8 [0-255] to [0-1] and convert to float32
    # Data was stored as uint8 for efficiency, originally normalized to [0-1]
    deadwood_data = (deadwood_data / 255.0).astype(np.float32)
    forest_data = (forest_data / 255.0).astype(np.float32)
    
    # Calculate increments on normalized data (positive for deadwood, negative for forest)
    def calculate_increments(data, mode="positive"):
        diff = np.diff(data.values, axis=0, prepend=0)
        if mode == "positive":
            increments = np.clip(diff, 0, None)
        elif mode == "negative":
            increments = np.abs(np.clip(diff, None, 0))
        else:
            raise ValueError("mode must be 'positive' or 'negative'")
        return increments
    
    increment_deadwood = xr.DataArray(
        calculate_increments(deadwood_data, mode="positive"),
        dims=deadwood_data.dims,
        coords=deadwood_data.coords
    )
    
    increment_forest = xr.DataArray(
        calculate_increments(forest_data, mode="negative"),
        dims=forest_data.dims,
        coords=forest_data.coords
    )
    
    # Stack: (d_f, y, x, time_year)
    stacked = xr.concat(
        [deadwood_data, forest_data, increment_deadwood, increment_forest], 
        dim="d_f"
    )
    stacked = stacked.transpose("d_f", "y", "x", "time_year")
    
    # Create processed dataset
    processed_ds = xr.Dataset(
        {"deadwood_forest": stacked},
        coords={
            "d_f": ["deadwood", "forest", "deadwood_inc", "forest_dec"],
            "y": ds_processed["y"],
            "x": ds_processed["x"],
            "time_year": ds_processed["time_year"],
        }
    )
    processed_ds.attrs = ds_pred.attrs
    return {"prediction": processed_ds}

def preprocess_sentinel_cube(ds_sentle):
    """Preprocess Sentinel-2 cube and normalize bands."""
    
    # Sort by time
    ds_processed = ds_sentle.sortby("time")
    ds_processed["time"] = pd.to_datetime(ds_processed["time"].values)
    time_vals = ds_processed.time.values
    time_unix = (time_vals.astype("int64") // 1_000_000_000).astype(np.int64)
    ds_processed = ds_processed.assign_coords(time_unix=("time", time_unix))

    
    # Ensure correct dimension order: (band, y, x, time)
    ds_processed["sentle"] = ds_processed["sentle"].transpose("band", "y", "x", "time")
    
    # Normalize Sentinel bands: divide by 10000 and clip to [0, 1]
    # Save as float16 to reduce footprint; output_dtypes controls dtype lazily
    ds_processed["sentle"] = xr.apply_ufunc(
        lambda x: np.clip(x / 10000.0, 0, 1),
        ds_processed["sentle"],
        dask="parallelized",
        output_dtypes=[np.float16]
    )
    return {"sentinel": ds_processed}

def get_cube_wgs84_bounds(cube_crs, cube_x, cube_y, buffer_degrees=0.2):
    """Return WGS84 bounding box with buffer."""
    transformer = Transformer.from_crs(cube_crs, "EPSG:4326", always_xy=True)
    corners_x = [cube_x.min(), cube_x.max(), cube_x.min(), cube_x.max()]
    corners_y = [cube_y.min(), cube_y.max(), cube_y.max(), cube_y.min()]
    lons, lats = transformer.transform(corners_x, corners_y)
    return min(lons) - buffer_degrees, max(lons) + buffer_degrees, min(lats) - buffer_degrees, max(lats) + buffer_degrees

def get_crs_info(cube_ds):
    """Extract CRS, bounding box, and reference info from cube."""
    wkt_str = cube_ds.attrs.get("crs_wkt")
    input_crs = rxr.rioxarray.crs.CRS.from_wkt(wkt_str)
    if cube_ds.rio.crs is None:
        cube_ds = cube_ds.rio.write_crs(input_crs)
    
    ref_layer = cube_ds.sel(band=REF_BAND)
    if ref_layer.rio.crs is None:
        ref_layer = ref_layer.rio.write_crs(input_crs)
    
    ref_times_raw = ref_layer["time"].values if "time" in ref_layer.coords else None
    ref_times = np.sort(np.unique(ref_times_raw)) if ref_times_raw is not None else None
    
    cube_crs = cube_ds.rio.crs
    cube_x = cube_ds.x.values
    cube_y = cube_ds.y.values
    lon_min, lon_max, lat_min, lat_max = get_cube_wgs84_bounds(cube_crs, cube_x, cube_y, buffer_degrees=0.2)
    bbox = (lon_min, lat_min, lon_max, lat_max)

    return cube_crs, bbox, ref_layer, ref_times

def load_era5_subset(era5_path, lon_min, lon_max, lat_min, lat_max):
    """Load ERA5 data within WGS84 bounds."""
    ds = xr.open_zarr(era5_path)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds_subset = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        ds_subset = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return ds_subset

def load_geotiff_subset(geotiff_path, lon_min, lon_max, lat_min, lat_max):
    """Load GeoTIFF data within WGS84 bounds."""
    raster = rxr.open_rasterio(geotiff_path, masked=True).rio.write_crs("EPSG:4326")
    if raster.y.values[0] > raster.y.values[-1]:
        raster_subset = raster.sel(y=slice(lat_max, lat_min), x=slice(lon_min, lon_max))
    else:
        raster_subset = raster.sel(y=slice(lat_min, lat_max), x=slice(lon_min, lon_max))
    return raster_subset

def load_netcdf_subset(netcdf_path, lon_min, lon_max, lat_min, lat_max):
    """Load NetCDF data within WGS84 bounds."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf', chunks='auto')
    lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'
    lat_coord = 'latitude' if 'latitude' in ds.coords else 'lat'
    lat_vals = ds[lat_coord].values
    if lat_vals[0] > lat_vals[-1]:
        ds_subset = ds.sel({lat_coord: slice(lat_max, lat_min), lon_coord: slice(lon_min, lon_max)})
    else:
        ds_subset = ds.sel({lat_coord: slice(lat_min, lat_max), lon_coord: slice(lon_min, lon_max)})
    return ds_subset

def align_time_to_ref(ds, ref_times):
    """Align time axis to reference times using nearest neighbor."""
    if "time" not in ds.coords or ref_times is None:
        return ds
    
    # Remove duplicates and sort
    ds = ds.sel(time=~ds.indexes["time"].duplicated()).sortby("time")
    ref_times_sorted = np.sort(np.unique(ref_times))
    
    # Get ERA5's actual time range
    ds_time_min = ds.time.values.min()
    ds_time_max = ds.time.values.max()
    
    # Reindex to all ref_times
    ds_aligned = ds.reindex(time=ref_times_sorted, method="nearest")
    
    # Mask values outside ERA5's original range with NaN
    ref_times_pd = pd.to_datetime(ref_times_sorted)
    outside_range = (ref_times_pd < ds_time_min) | (ref_times_pd > ds_time_max)
    
    # Create DataArray mask with proper dimensions for broadcasting
    mask_da = xr.DataArray(outside_range, coords={"time": ref_times_sorted}, dims=["time"])
    
    # Handle both Dataset and DataArray
    if isinstance(ds_aligned, xr.Dataset):
        # Set all data variables to NaN for times outside ERA5 range
        for var in ds_aligned.data_vars:
            ds_aligned[var] = ds_aligned[var].where(~mask_da, np.nan)
    else:
        # DataArray - apply mask directly
        ds_aligned = ds_aligned.where(~mask_da, np.nan)
    
    return ds_aligned.sortby("time")

def add_bbox_buffer_degrees(bbox, buffer_deg=0.01):
    """Add buffer in degrees to lat/lon bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lon - buffer_deg, min_lat - buffer_deg, max_lon + buffer_deg, max_lat + buffer_deg)

def reproject_match_with_buffer(f, r, bbox, buffer_deg=0.01, var_name=None):
    """Clip raster to bbox and reproject to match target."""
    minx, miny, maxx, maxy = add_bbox_buffer_degrees(bbox, buffer_deg)
    with rxr.open_rasterio(f, masked=True) as da_r:
        da_r = da_r.squeeze(dim='band', drop=True)
        if da_r.rio.crs is None:
            da_r = da_r.rio.write_crs("EPSG:4326")
        da_clipped = da_r.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, auto_expand=True)
        da_aligned = da_clipped.rio.reproject_match(r).chunk({'x': 'auto', 'y': 'auto'}).astype(np.float32)
        da_aligned = da_aligned.drop_vars("spatial_ref", errors="ignore")
        if var_name is None:
            var_name = f.stem
        da_aligned.name = var_name
    return da_aligned

def create_era5_dataset(ref_times, bbox):
    """Create and normalize ERA5 dataset."""
    lon_min, lat_min, lon_max, lat_max = bbox
    # Load all ERA5 subsets (without aligning yet)
    era5_subsets = {}
    for era5_file in DATA_SOURCES["era5"]:
        era5_subset = load_era5_subset(era5_file, lon_min, lon_max, lat_min, lat_max)
        file_basename = Path(era5_file).stem
        era5_subsets[file_basename] = era5_subset
    
    # Stack all variables together
    era5_var_names = []
    era5_arrays = []
    for ds in era5_subsets.values():
        for var in ds.data_vars:
            era5_var_names.append(var)
            era5_arrays.append(ds[var])
    era5_stack = xr.concat(era5_arrays, dim="era5_var").assign_coords(era5_var=era5_var_names)
    era5_stack = align_time_to_ref(era5_stack, ref_times)
    
    drop_vars = ["ssrd_q05"]
    to_keep = [v for v in era5_stack.era5_var.values if v not in drop_vars]
    era5_stack = era5_stack.sel(era5_var=to_keep)

    # Transpose to (era5_var, lat, lon, time) for consistency with other datasets
    era5_stack = era5_stack.transpose("era5_var", "lat", "lon", "time")
    
    # Ensure ascending coordinate order for consistency
    if era5_stack.lat.values[0] > era5_stack.lat.values[-1]:
        era5_stack = era5_stack.sortby("lat")
    if era5_stack.lon.values[0] > era5_stack.lon.values[-1]:
        era5_stack = era5_stack.sortby("lon")
    
    era5_ds = era5_stack.to_dataset(name="era5")
    return {"era5": era5_ds}

def create_worldclim_dataset(bbox):
    """Create WorldClim dataset."""
    lon_min, lat_min, lon_max, lat_max = bbox
    wc_subsets = {}
    for wc_file in DATA_SOURCES["worldclim"]:
        wc_subset = load_geotiff_subset(wc_file, lon_min, lon_max, lat_min, lat_max)
        
        # Squeeze band dimension if present
        if 'band' in wc_subset.dims:
            wc_subset = wc_subset.squeeze('band', drop=True)
            
        # Keep the naming structure from the filename
        file_basename = wc_file.stem
        wc_subsets[file_basename] = wc_subset
    
    # Stack the individual GeoTIFFs into one DataArray
    wc_stack = xr.concat(
        list(wc_subsets.values()), 
        dim="worldclim_var"
    ).assign_coords(worldclim_var=list(wc_subsets.keys()))
    
    # Ensure ascending coordinate order for consistency (geographic alignment)
    if wc_stack.y.values[0] > wc_stack.y.values[-1]:
        wc_stack = wc_stack.sortby("y")
    if wc_stack.x.values[0] > wc_stack.x.values[-1]:
        wc_stack = wc_stack.sortby("x")
        
    return {"worldclim": wc_stack}

def create_soilgrids_dataset(bbox):
    """Create SoilGrids dataset."""
    lon_min, lat_min, lon_max, lat_max = bbox
    
    sg_subsets = {}
    for sg_file in DATA_SOURCES["soilgrids"]:
        sg_subset = load_geotiff_subset(sg_file, lon_min, lon_max, lat_min, lat_max)
        
        # Squeeze band dimension if present
        if 'band' in sg_subset.dims:
            sg_subset = sg_subset.squeeze('band', drop=True)
            
        # Keep the naming structure (e.g., 'clay_0-5cm_mean')
        file_basename = sg_file.stem
        sg_subsets[file_basename] = sg_subset
    
    # Stack individual variables into one DataArray
    sg_stack = xr.concat(
        list(sg_subsets.values()), 
        dim="soilgrids_var"
    ).assign_coords(soilgrids_var=list(sg_subsets.keys()))
    
    # Ensure ascending coordinate order for geographic consistency
    if sg_stack.y.values[0] > sg_stack.y.values[-1]:
        sg_stack = sg_stack.sortby("y")
    if sg_stack.x.values[0] > sg_stack.x.values[-1]:
        sg_stack = sg_stack.sortby("x")
    
    return {"soilgrids": sg_stack}

def create_stand_age_dataset(bbox):
    """Create Stand Age dataset."""
    lon_min, lat_min, lon_max, lat_max = bbox
    sa_subsets = {}
    for sa_file in DATA_SOURCES["stand_age"]:
        # Using the specific netcdf loader as in your original snippet
        sa_subset = load_netcdf_subset(sa_file, lon_min, lon_max, lat_min, lat_max)
        file_basename = sa_file.stem
        sa_subsets[file_basename] = sa_subset
    
    # --- Stacking Logic ---
    if isinstance(list(sa_subsets.values())[0], xr.Dataset):
        sa_var_names = []
        sa_arrays = []
        for ds in sa_subsets.values():
            for var in ds.data_vars:
                sa_var_names.append(var)
                sa_arrays.append(ds[var])
        sa_stack = xr.concat(sa_arrays, dim="stand_age_var").assign_coords(stand_age_var=sa_var_names)
    else:
        # If it's already a DataArray, take the first one
        sa_stack = list(sa_subsets.values())[0]

    # --- Ensure ascending coordinate order for consistency ---
    lon_coord = 'longitude' if 'longitude' in sa_stack.coords else 'lon'
    lat_coord = 'latitude' if 'latitude' in sa_stack.coords else 'lat'
    if lat_coord in sa_stack.coords and sa_stack[lat_coord].values[0] > sa_stack[lat_coord].values[-1]:
        sa_stack = sa_stack.sortby(lat_coord)
    if lon_coord in sa_stack.coords and sa_stack[lon_coord].values[0] > sa_stack[lon_coord].values[-1]:
        sa_stack = sa_stack.sortby(lon_coord)
    
    return {"stand_age": sa_stack}

def create_terrain_dataset(ds_sentle, ref_layer, bbox):
    """Create Terrain dataset sin/cos encoding."""
    data_list = []
    var_names = []
    keywords = ['dem', 'aspect', 'slope']
    
    for tf in DATA_SOURCES["dem_aspect_slope"]:
        filename = tf.stem
        keyword = next((k for k in keywords if k in filename), None)
        if keyword is None or keyword in ds_sentle.data_vars:
            continue
        
        aligned = reproject_match_with_buffer(tf, ref_layer, bbox, buffer_deg=0.01, var_name=keyword)
        
        if keyword == "aspect":
            radians = np.deg2rad(aligned.astype(np.float32))
            sin_aspect = xr.DataArray(
                np.sin(radians).astype(np.float32), coords=aligned.coords, dims=aligned.dims,
                attrs=aligned.attrs).chunk({'x': 'auto', 'y': 'auto'})
            cos_aspect = xr.DataArray(
                np.cos(radians).astype(np.float32), coords=aligned.coords, dims=aligned.dims,
                attrs=aligned.attrs).chunk({'x': 'auto', 'y': 'auto'})
            data_list.extend([sin_aspect, cos_aspect])
            var_names.extend(["sin_aspect", "cos_aspect"])
        else:
            # DEM and Slope are kept as raw values (meters and degrees)
            aligned = aligned.rename(keyword).chunk({'x': 'auto', 'y': 'auto'})
            data_list.append(aligned)
            var_names.append(keyword)
    
    terrain_dict = dict(zip(var_names, data_list))
    if "slope" in terrain_dict and "dem" in terrain_dict:
        slope = terrain_dict["slope"]
        dem = terrain_dict["dem"]
        
        # Convert slope to radians
        slope_rad = np.deg2rad(slope)
        
        # Avoid division by zero on flat ground
        tan_slope = np.tan(slope_rad) + 0.001
        
        # Note: using dem.min() works, but ensure it's the min of the current patch
        local_relief = dem - dem.min()
        
        # Calculated Index
        downslope_index = (local_relief / tan_slope)
        
        # Ensure it has the same metadata and chunking
        downslope_index = downslope_index.rename("downslope_index").chunk({'x': 'auto', 'y': 'auto'})
        
        data_list.append(downslope_index)
        var_names.append("downslope_index")
    else:
        print("Warning: 'slope' or 'dem' missing, skipping 'downslope_index' calculation.")
    
    # Combine the arrays
    combined = xr.concat(data_list, dim=xr.DataArray(var_names, dims="terrain_var"))
    combined.name = "terrain"
    combined = combined.chunk({"terrain_var": len(var_names), "y": 100, "x": 100})
    
    # Create Dataset using the raw 'combined' array
    terrain_ds = xr.Dataset({"terrain": combined})
    
    return {"terrain": terrain_ds}

def create_canopy_height_dataset(ds_sentle, ref_layer, bbox):
    """Create Canopy Height dataset."""
    output = {}
    for f in DATA_SOURCES["canopy_height"]:
        filename = f.stem
        # Distinguish between height (ch) and standard deviation (ch_sd)
        var_name = "ch_sd" if filename.endswith('_SD') else "ch"
        
        if var_name in ds_sentle.data_vars:
            continue
        
        # Reproject and align with the reference layer
        aligned = reproject_match_with_buffer(f, ref_layer, bbox, buffer_deg=0.01, var_name=var_name)
        output[var_name] = aligned
    
    # Stack the variables into a single DataArray
    canopy_stack = xr.concat(
        list(output.values()), 
        dim="canopy_height_var"
    ).assign_coords(canopy_height_var=list(output.keys()))
    
    canopy_ds = xr.Dataset({"ch": canopy_stack})
    
    return {"canopy_height": canopy_ds}

# ================
# OUTPUT FUNCTIONS
# ================
def create_static_datasets(merged_ds, num_workers):
    """Create all static datasets in parallel."""
    cube_crs, bbox, ref_layer, ref_times = get_crs_info(merged_ds)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                "worldclim": executor.submit(create_worldclim_dataset, bbox),
                "soilgrids": executor.submit(create_soilgrids_dataset, bbox),
                "stand_age": executor.submit(create_stand_age_dataset, bbox),
                "terrain": executor.submit(create_terrain_dataset, merged_ds, ref_layer, bbox),
                "canopy_height": executor.submit(create_canopy_height_dataset, merged_ds, ref_layer, bbox),
            }
            results = {}
            for name, future in futures.items():
                try:
                    results.update(future.result())
                except Exception as e:
                    print(f"    ✗ {name} failed: {e}")
        # Load ERA5 outside the pool
    results.update(create_era5_dataset(ref_times, bbox))
    #print(f"✓ Static datasets complete")
    return results

# ================
# Main functions
# ================
def setup_logger(log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("timestamp,block_id,cube_id,status,error_msg\n")
    return log_file

def log_result(log_file, block_id, cube_id, status, error_msg=""):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # Clean error message to avoid CSV breakages
    clean_msg = str(error_msg).replace(",", ";").replace("\n", " ")
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{block_id},{cube_id},{status},{clean_msg}\n")

def process_single_cube(cube_meta, ds_sentinel_full, ds_pred_full, block_crs, out_dir_base, cube_crs):
    cube_id = cube_meta['cube_id']
    
    # Setup Output Directory
    zarr_path = out_dir_base / f"{cube_id}.zarr"

    # Subset
    cube_geom_in_block_crs = gpd.GeoSeries([cube_meta.geometry], crs=cube_crs).to_crs(block_crs).iloc[0]
    minx, miny, maxx, maxy = cube_geom_in_block_crs.bounds
    
    ds_sentle_sub = subset_cube(ds_sentinel_full, minx, miny, maxx, maxy)
    ds_pred_sub = subset_cube(ds_pred_full, minx, miny, maxx, maxy)

    # Preprocess & Merge
    ds_prediction_proc = preprocess_prediction_cube(ds_pred_sub)["prediction"]
    ds_sentinel_proc = preprocess_sentinel_cube(ds_sentle_sub)["sentinel"]

    # Process additional predictors
    results = create_static_datasets(ds_sentle_sub, num_workers=5)

    # Merge datasets wit similar spatial resolution
    high_res_ds = xr.merge([
        ds_prediction_proc, 
        ds_sentinel_proc, 
        results["terrain"], 
        results["canopy_height"]
    ], compat='override')

    # Filter by START_YEAR
    start_dt = np.datetime64(f"{START_YEAR}-01-01T00:00:00")
    high_res_ds = high_res_ds.sel(
        time_year=high_res_ds.time_year >= start_dt,
        time=high_res_ds.time >= start_dt,
    )
    high_res_ds.attrs["crs_wkt"] = ds_sentle_sub.attrs["crs_wkt"]

    # Chunking
    high_res_ds['sentle'] = high_res_ds['sentle'].chunk({"band": -1, "y": 100, "x": 100, "time": 10})
    high_res_ds['terrain'] = high_res_ds['terrain'].chunk({"terrain_var": -1, "y": 100, "x": 100})
    high_res_ds['deadwood_forest'] = high_res_ds['deadwood_forest'].chunk({"d_f": -1, "y": 100, "x": 100, "time_year": -1})
    
    results["era5"] = results["era5"].chunk({"era5_var": -1, "lat": -1, "lon": -1, "time": 100})
    results["worldclim"] = results["worldclim"].chunk({"worldclim_var": -1, "y": -1, "x": -1})
    results["soilgrids"] = results["soilgrids"].chunk({"soilgrids_var": -1, "y": -1, "x": -1})
    results["stand_age"] = results["stand_age"].chunk({"stand_age_var": -1, "latitude": -1, "longitude": -1})
    
    def clear_all_encodings(ds):
        ds.encoding = {}
        for var in ds.variables:
            ds[var].encoding = {}
        return ds

    # Write as zarr with groups
    with dask.config.set(scheduler='threads', num_workers=NUM_WORKERS):
        # Create store with first group
        high_res_ds = clear_all_encodings(high_res_ds)
        high_res_ds.to_zarr(zarr_path, group="high_res", mode="w", consolidated=False)
        
        # Append other groups
        era5_ds = clear_all_encodings(results["era5"])
        era5_ds.to_zarr(zarr_path, group="era5", mode="a", consolidated=False)
        
        for key in ["worldclim", "soilgrids", "stand_age"]:
            if key in results:
                ds_to_save = results[key]
                if isinstance(ds_to_save, xr.DataArray):
                    ds_to_save = ds_to_save.to_dataset(name=key)
                # Clear encoding to prevent old metadata conflicts
                ds_to_save = clear_all_encodings(ds_to_save) 
                ds_to_save.to_zarr(zarr_path, group=key, mode="a", consolidated=False)

        zarr.consolidate_metadata(str(zarr_path))
        
    del high_res_ds, results, ds_sentle_sub, ds_pred_sub
    gc.collect()

def process_single_block(block_id, gdf_cubes, out_dir_base, log_file):
    log_file = setup_logger(log_file)
    
    # Get all cubes in this block
    cubes_in_block = gdf_cubes[gdf_cubes['block_id'] == block_id]
    cube_crs = gdf_cubes.crs 
    
    try:
        # Load the large S3 blocks once per group
        ds_sentinel_full, ds_pred_full = retrieve_S3_blocks(block_id)
        
        # Determine CRS once per block
        wkt_str_sentle = ds_sentinel_full.attrs.get("crs_wkt")
        block_crs = rxr.rioxarray.crs.CRS.from_wkt(wkt_str_sentle)
        
        # Process each cube within this block
        for _, cube_meta in cubes_in_block.iterrows():
            cube_id = cube_meta['cube_id']
            try:
                process_single_cube(cube_meta, ds_sentinel_full, ds_pred_full, block_crs, out_dir_base, cube_crs)
                log_result(log_file, block_id, cube_id, "SUCCESS")
            except Exception as e:
                print(f"    ✗ Failed Cube {cube_id}: {e}")
                log_result(log_file, block_id, cube_id, "FAILED", e)
                
        ds_sentinel_full.close()
        ds_pred_full.close()
        del ds_sentinel_full
        del ds_pred_full
        gc.collect()

    except Exception as e:
        print(f"  ✗ Failed to load Block {block_id}: {e}")
        # Log all cubes in this block as failed
        for _, cube_meta in cubes_in_block.iterrows():
            log_result(log_file, block_id, cube_meta['cube_id'], "BLOCK_LOAD_FAILURE", e)
    
    if 'ds_sentinel_full' in locals():
        ds_sentinel_full.close()
        del ds_sentinel_full
    if 'ds_pred_full' in locals():
        ds_pred_full.close()
        del ds_pred_full

def main():
    # Cube meta file
    if not CUBE_METADATA.exists():
        raise FileNotFoundError(f"Could not find cube set at {CUBE_METADATA}")
    
    gdf_cubes = gpd.read_file(CUBE_METADATA)
    
    # 2. Setup/Load log to check for existing progress
    log_file = setup_logger(LOG_FILE)
    try:
        processed_cubes = pd.read_csv(log_file)
        if not processed_cubes.empty:
            # Check only for SUCCESS to allow retrying FAILED ones
            # Keep as int to match gdf_cubes['cube_id'] dtype
            successful_ids = set(processed_cubes[processed_cubes['status'] == 'SUCCESS']['cube_id'].astype(int).tolist())
        else:
            successful_ids = set()
    except Exception:
        successful_ids = set()

    # 3. Identify unique blocks to process
    unique_blocks = gdf_cubes['block_id'].unique()
    print(f"Total blocks to process: {len(unique_blocks)}")
    print(f"Total cubes in set: {len(gdf_cubes)}")
    print(f"Already completed: {len(successful_ids)}")

    # 4. Loop over blocks
    for i, block_id in enumerate(unique_blocks):
        # Filter cubes for this block that aren't finished yet
        cubes_in_block = gdf_cubes[gdf_cubes['block_id'] == block_id]
        remaining_cubes = cubes_in_block[~cubes_in_block['cube_id'].isin(successful_ids)]
        
        if remaining_cubes.empty:
            print(f"[{i+1}/{len(unique_blocks)}] Block {block_id}: All cubes already processed. Skipping.")
            continue
            
        print(f"\n[{i+1}/{len(unique_blocks)}] Processing Block: {block_id}")
        print(f"  -> {len(remaining_cubes)} cubes remaining in this block.")
        
        # Call the block processor
        process_single_block(block_id, remaining_cubes, OUT_DIR, log_file)

def process_specific_cube(cube_meta, data_dir):
    cube_id = cube_meta['cube_id']
    block_id = cube_meta['block_id']
    
    print(f"Opening S3 datasets for block: {block_id}...")
    ds_sentinel_full, ds_pred_full = retrieve_S3_blocks(block_id)
    wkt_str_sentle = ds_sentinel_full.attrs.get("crs_wkt")
    block_crs = rxr.rioxarray.crs.CRS.from_wkt(wkt_str_sentle)

    try:
        print(f"Starting processing for cube: {cube_id}...")
        # Note: We pass the version in block_crs to the function 
        # so it doesn't have to re-transform it during subsetting.
        process_single_cube(
            cube_meta=cube_meta,
            ds_sentinel_full=ds_sentinel_full,
            ds_pred_full=ds_pred_full,
            block_crs=block_crs,
            out_dir_base=data_dir, 
            cube_crs=block_crs 
        )
        print(f"✓ {cube_id} processed successfully.")
    except Exception as e:
        print(f"✗ Failed to process {cube_id}: {e}")
    finally:
        # Cleanup
        ds_sentinel_full.close()
        ds_pred_full.close()
        del ds_sentinel_full, ds_pred_full
        gc.collect()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up S3 connections
        if _FS is not None:
            _FS.clear_instance_cache()
            print("S3 connection closed.")

    