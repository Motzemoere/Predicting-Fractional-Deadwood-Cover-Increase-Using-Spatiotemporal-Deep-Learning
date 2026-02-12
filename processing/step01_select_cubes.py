"""
Select and grid deadwood spatial cubes for Germany.
Creates 5x5 km analysis cubes within Germany's boundaries from S3 metadata.
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio.features import rasterize
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import re
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS, DATA_SOURCES

# ================
# Path Configuration (from centralized config)
# ================
META_DATA_DIR = PATHS.meta_data_dir
OUTPUT_DIR = PATHS.data_dir  # For new outputs

# External data sources
MORTALITY_TIFS = glob('/mnt/gsdata/projects/deadtrees/standing_deadwood_germany_schiefer/*.tif')

if __name__ == "__main__":

    # Load Boundary of Germany
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    germany_boundary = world[world['NAME'] == 'Germany'][['geometry']]

    # Load S3 metadata
    gdf = gpd.read_file(META_DATA_DIR / "europe_inference_blocks.gpkg")
    germany_boundary = germany_boundary.to_crs(gdf.crs)
    germany_gdf = gpd.sjoin(gdf, germany_boundary, how="inner", predicate="intersects")
    if 'index_right' in germany_gdf.columns:
        germany_gdf = germany_gdf.drop(columns=['index_right'])

    print(f"Filtered {len(germany_gdf)} blocks within Germany territory.")

    ####################
    # Create 5x5 km grid
    ####################
    crs_metric = "EPSG:25832" # Standard for Germany
    germany_boundary_utm = germany_boundary.to_crs(crs_metric)

    cell_size_m = 5000  # 5 km
    minx, miny, maxx, maxy = germany_boundary_utm.total_bounds
    x_coords = np.arange(minx, maxx + cell_size_m, cell_size_m)
    y_coords = np.arange(miny, maxy + cell_size_m, cell_size_m)
    grid_cells = [
        box(x, y, x + cell_size_m, y + cell_size_m)
        for x in x_coords[:-1] 
        for y in y_coords[:-1]
    ]
    grid_utm = gpd.GeoDataFrame(geometry=grid_cells, crs=crs_metric)
    germany_union_utm = germany_boundary_utm.union_all()
    grid_utm = grid_utm[
        grid_utm.intersects(germany_union_utm)
    ].reset_index(drop=True)
    grid_utm['cell_idx'] = np.arange(0, len(grid_utm))
    print(f"{len(grid_utm)} 5x5km grid cells over Germany (EPSG:25832)")

    # Select only cells within a Block
    meta_utm = germany_gdf.to_crs("EPSG:25832")
    grid_utm_filtered = gpd.sjoin(
        grid_utm, 
        meta_utm[['block_id', 'geometry']], 
        how='inner', 
        predicate='within'
    )
    # Remove duplicates caused by overlapping blocks
    grid_utm_filtered = grid_utm_filtered.drop_duplicates(subset=['cell_idx'])

    # Built grid_id
    minx_grid = grid_utm_filtered.geometry.bounds.minx.astype(int)
    miny_grid = grid_utm_filtered.geometry.bounds.miny.astype(int)
    grid_utm_filtered['grid_id'] = (
        "cube_" + crs_metric.split(":")[1] + "_" +
        minx_grid.astype(str) + "_" +
        miny_grid.astype(str)
    )
    grid_utm_filtered = grid_utm_filtered.drop(columns=['index_right'])
    print(f"{len(grid_utm_filtered)} cells remaining: ")
    print(grid_utm_filtered[['cell_idx', 'grid_id']].head())

    # Check were Orthophotos are available
    gdf_ortho = gpd.read_file(META_DATA_DIR / "metadata_filtered_v10.gpkg")
    germany_orthos = gdf_ortho[gdf_ortho['admin_level_1'] == 'Germany']
    orthos_utm = germany_orthos.to_crs(crs_metric)

    # Set grid cells with orthophotos
    ortho_hits = gpd.sjoin(
        grid_utm_filtered, 
        orthos_utm[['geometry']], 
        how='left', 
        predicate='intersects'
    )
    has_ortho_indices = ortho_hits[ortho_hits['index_right'].notnull()].index.unique()
    grid_utm_filtered['has_ortho'] = grid_utm_filtered.index.isin(has_ortho_indices)
    num_with_ortho = grid_utm_filtered['has_ortho'].sum()
    print(f"Orthophotos found for {num_with_ortho} out of {len(grid_utm_filtered)} cells.")

    ####################
    # Get Mortality Data From Schiefer et.al 
    ####################
    def compute_mortality_binary(grid_gdf, tif_path, threshold=5000):
        with rasterio.open(tif_path) as src:
            # 1. Project grid to match raster CRS (EPSG:3035)
            # This is vital for the rasterize logic to align with pixels
            grid_projected = grid_gdf.to_crs(src.crs)
            
            # 2. Use a Windowed Read (Optional but saves RAM if TIF is huge)
            bbox = grid_projected.total_bounds
            window = src.window(*bbox).round_offsets().round_lengths()
            data = src.read(1, window=window)
            nodata = src.nodata
            transform = src.window_transform(window)

            # 3. Binarize
            # -1 = NA, 0 = Healthy, 1 = Mortality
            data_bin = np.full(data.shape, -1, dtype=np.int8)
            valid_mask = (data != nodata) & (~np.isnan(data))
            data_bin[valid_mask & (data <= threshold)] = 0
            data_bin[valid_mask & (data > threshold)] = 1

            # 4. Rasterize using LOCAL indices (0 to N-1)
            # This makes np.bincount memory efficient regardless of how big cell_idx is
            shapes = ((geom, i) for i, geom in enumerate(grid_projected.geometry))
            grid_raster = rasterize(
                shapes,
                out_shape=data.shape,
                transform=transform,
                fill=-1,
                dtype='int32'
            )

        # 5. Fast Aggregation
        gids = grid_raster.ravel()
        vals = data_bin.ravel()

        # Filter to pixels inside our 5km grid
        mask = gids >= 0
        gids = gids[mask]
        vals = vals[mask]

        n_cells = len(grid_projected)
        
        # Count pixels using the local index 'i'
        c_total = np.bincount(gids, minlength=n_cells)
        c_valid = np.bincount(gids[vals != -1], minlength=n_cells)
        c_high = np.bincount(gids[vals == 1], minlength=n_cells)

        # 6. Build Result
        # We create a simple dataframe and then join it back to avoid geometry copying
        stats_df = pd.DataFrame({
            'c_valid': c_valid,
            'c_total': c_total,
            'c_high': c_high
        })
        
        # Calculate percentages
        stats_df['pct_high'] = (stats_df['c_high'] / stats_df['c_valid'].replace(0, np.nan)) * 100
        stats_df['pct_valid'] = (stats_df['c_valid'] / stats_df['c_total'].replace(0, np.nan)) * 100

        # Attach the stats back to the grid using the same order
        result = grid_gdf.copy()
        for col in stats_df.columns:
            result[col] = stats_df[col].values
            
        return result.fillna(0)

    calculate = False
    if calculate == True:
        files = MORTALITY_TIFS
        all_counts = []
        for f in tqdm(files, desc="Processing mortality"):
            filename = os.path.basename(f)
            year_match = re.search(r'(\d{4})', filename)
            if year_match:
                year = year_match.group(1)
            else:
                # Fallback if regex fails
                year = filename.split('-')[1].replace('.tif', '')
            mort_stats = compute_mortality_binary(grid_utm_filtered, f, threshold=5000)
            stats_only = mort_stats.drop(columns='geometry')
            stats_only['year'] = year
            all_counts.append(stats_only)

        stacked = pd.concat(all_counts, ignore_index=True)
        idx = stacked.groupby('cell_idx')['pct_high'].idxmax()
        stacked_max = stacked.loc[idx].reset_index(drop=True)

        final_grid = grid_utm_filtered.merge(
            stacked_max[['cell_idx', 'year', 'pct_high', 'c_high', 'pct_valid']], 
            on='cell_idx', 
            how='left'
        )
        # Save to GeoPackage
        output_path = META_DATA_DIR / "available_cubes.gpkg"
        if output_path.exists():
            print(f"File {output_path} already exists. Aborting to prevent overwrite.")
        else:
            final_grid.to_file(output_path, layer='mortality_stats', driver="GPKG")
            print(f"File saved successfully to {output_path}")

####################
# Built final cube set
####################

    gdf = gpd.read_file(META_DATA_DIR / "available_cubes.gpkg")
    gdf.head()
    gdf.crs

    valid_mort = gdf[gdf['pct_valid'] >= 50]
    print(f"Cells with >=50% valid data: {len(valid_mort)}")
    low_mort = valid_mort[valid_mort['pct_high'] < 20]
    high_mort = valid_mort[valid_mort['pct_high'] >= 20]

    # Balance dataset by sampling low mortality cells
    n = len(high_mort)
    low_mort_sampled = low_mort.sample(n=n, random_state=42).reset_index(drop=True)

    balanced_gdf = pd.concat([low_mort_sampled, high_mort], ignore_index=True)
    balanced_gdf['mortality_class'] = np.where(
        balanced_gdf['pct_high'] >= 20, 'high', 'low'
    )
    # Add unique cube_id
    balanced_gdf['cube_id'] = np.arange(len(balanced_gdf))

    output_combined = META_DATA_DIR / "balanced_cube_set.gpkg"
    # check if file exists, if so abort
    if output_combined.exists():
        print(f"File {output_combined} already exists. Aborting to prevent overwrite.")
    else:
        balanced_gdf.to_file(output_combined, layer='balanced_dataset', driver="GPKG")
        print(f"Combined balanced dataset saved: {len(balanced_gdf)} total cells.")

    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    germany_boundary = world[world['NAME'] == 'Germany'][['geometry']]
    germany_boundary = germany_boundary.to_crs(gdf.crs)

    fig, ax = plt.subplots(figsize=(10, 10))
    low_mort_sampled.plot(ax=ax, color='green', alpha=0.7)
    high_mort.plot(ax=ax, color='red', alpha=0.7)
    balanced_gdf[balanced_gdf['has_ortho'] == True].plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.5)
    balanced_gdf[balanced_gdf['has_ortho'] == True]
    germany_boundary.boundary.plot(ax=ax, color='black', linewidth=1)
    # Create custom legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label=f'Low mortality sampled ({len(low_mort_sampled)})'),
        Patch(facecolor='red', alpha=0.7, label=f'High mortality ({len(high_mort)})'),
        Patch(facecolor='none', edgecolor='blue', label='Cells with Orthophotos', linewidth=2),
        Patch(facecolor='none', edgecolor='black', label='Germany Boundary')
    ]
    ax.legend(handles=legend_elements)