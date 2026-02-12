"""
Create spatially-constrained holdout and cross-validation folds.
Ensures training/validation/test sets are spatially separated to avoid data leakage.
"""

import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS
from utils.parall import paral

# Paths from centralized config
CUBE_META_GPKG = PATHS.meta_data_dir / "balanced_cube_set.gpkg"
FRACTION = 0.1
MIN_DIST = 50_000  # in CRS units (EPSG:25832 -> meters)
RANDOM_SEED = 42
N_FOLDS = 4

def select_spatial_holdouts(gdf, fraction, min_dist, random_seed):
    """
    Spatially constrained holdout selection.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame.
    fraction : float
        Fraction of total rows to mark as holdout.
    min_dist : float
        Minimum separation distance (CRS units).
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    GeoDataFrame
        Copy of gdf with boolean column 'is_holdout'.
    """

    rng = np.random.default_rng(random_seed)

    gdf = gdf.copy()
    gdf['centroid'] = gdf.geometry.centroid
    gdf = gdf.set_geometry('centroid')

    target_n = int(np.ceil(len(gdf) * fraction))
    holdout_ids = set()

    def _select(candidates):
        nonlocal holdout_ids

        sindex = candidates.sindex
        order = rng.permutation(len(candidates))

        for i in order:
            row = candidates.iloc[i]
            cid = row['cube_id']
            pt = row.geometry

            nearby = sindex.query(pt.buffer(min_dist))
            if any(
                candidates.iloc[j]['cube_id'] in holdout_ids
                for j in nearby
            ):
                continue

            holdout_ids.add(cid)

            if len(holdout_ids) >= target_n:
                break

    # 1. sample from has_ortho == True
    ortho = gdf.loc[gdf['has_ortho'] == True]
    _select(ortho)

    # 2. sample remaining from the rest
    if len(holdout_ids) < target_n:
        remaining = gdf.loc[~gdf['cube_id'].isin(holdout_ids)]
        _select(remaining)

    # Mark holdouts
    gdf['is_holdout'] = gdf['cube_id'].apply(lambda cid: cid in holdout_ids)
    return gdf.drop(columns='centroid').set_geometry('geometry')

def assign_spatial_folds_kmeans(
    gdf,
    n_folds=4,
    random_seed=42,
    holdout_fold=0
):
    """
    Assign spatial CV folds.
    Fold 0 is reserved for holdouts.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain 'is_holdout'.
    n_folds : int
        Number of CV folds (excluding holdout).
    random_seed : int
        Random seed.
    holdout_fold : int
        Label for holdout fold.

    Returns
    -------
    GeoDataFrame
        Copy of gdf with integer column 'fold'.
    """

    gdf = gdf.copy()
    gdf['fold'] = holdout_fold

    # Non-holdout samples
    train = gdf.loc[~gdf['is_holdout']].copy()
    centroids = train.geometry.centroid

    X = np.column_stack([centroids.x, centroids.y])

    kmeans = KMeans(
        n_clusters=n_folds,
        random_state=random_seed,
        n_init="auto"
    )
    labels = kmeans.fit_predict(X)

    # Assign folds 1..n_folds
    train['fold'] = labels + 1

    gdf.loc[train.index, 'fold'] = train['fold']

    return gdf

def assign_spatial_folds(
    gdf, 
    n_folds=4, 
    random_seed=42, 
    holdout_fold=0, 
    block_size=50000
):
    """
    Assigns spatial CV folds using a block_size (m) grid and returns both 
    the updated points and the grid polygons for plotting.
    """
    # 1. CRS Safety Check
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError(
            f"Input GeoDataFrame must have a projected CRS (e.g., EPSG:25832). "
            f"Current CRS is: {gdf.crs}"
        )

    gdf = gdf.copy()
    
    # Initialize all rows with the holdout fold label
    gdf['fold'] = holdout_fold
    
    # Filter to non-holdout training data for fold assignment
    train_mask = ~gdf['is_holdout']
    train = gdf.loc[train_mask].copy()

    # 2. Calculate Grid Indices
    # Using floor division // to bin coordinates into blocks
    centroids = train.geometry.centroid
    grid_x = (centroids.x // block_size).astype(int)
    grid_y = (centroids.y // block_size).astype(int)
    train['block_id'] = grid_x.astype(str) + "_" + grid_y.astype(str)

    # 3. Assign Blocks to Folds
    unique_blocks = train['block_id'].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_blocks)
    
    # Map unique blocks to fold IDs (1 to n_folds)
    block_fold_map = {
        block: (i % n_folds) + 1 
        for i, block in enumerate(unique_blocks)
    }
    
    train['fold'] = train['block_id'].map(block_fold_map)
    
    # Update the original GDF with the new fold assignments
    gdf.loc[train_mask, 'fold'] = train['fold']

    # 4. Create the Grid GeoDataFrame for plotting
    grid_polygons = []
    for block_id in unique_blocks:
        # Reconstruct the corner coordinates from the block_id
        x_idx, y_idx = map(int, block_id.split('_'))
        x_min, y_min = x_idx * block_size, y_idx * block_size
        
        grid_polygons.append({
            'geometry': box(x_min, y_min, x_min + block_size, y_min + block_size),
            'block_id': block_id,
            'fold': block_fold_map[block_id]
        })
    
    grid_gdf = gpd.GeoDataFrame(grid_polygons, crs=gdf.crs)
    
    return gdf, grid_gdf

def main():
    gdf = gpd.read_file(CUBE_META_GPKG)
    if gdf.crs.to_epsg() !=  25832:
        raise ValueError(
            f"Invalid CRS: expected EPSG:{25832}, "
            f"got {gdf.crs.to_string()}"
        )
    gdf = select_spatial_holdouts(gdf, fraction=FRACTION, min_dist=MIN_DIST, random_seed=RANDOM_SEED)

    gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
    gdf, grid_gdf = assign_spatial_folds(
        gdf,
        n_folds=N_FOLDS,
        holdout_fold=0,
        random_seed=RANDOM_SEED,
        block_size=50000
    )

    from utils.plots import plot_holdouts_folds, FIGURE_CLASSES
    plot_holdouts_folds(gdf, grid_gdf=grid_gdf, save=False)

    # Save updated cube set with holdout and fold info
    # check if exists if so abort
    if (PATHS.meta_data_dir / "training_cube_set.gpkg").exists():
        raise FileExistsError("training_cube_set.gpkg already exists. Aborting to prevent overwrite.")
    gdf.to_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
    grid_gdf.to_file(PATHS.meta_data_dir / "spatial_folds_grid.gpkg")
