"""
Comprehensive analysis and visualization pipeline.
Generates plots for cube selection, inspection sites, predictions, model evaluation, and label inspection.
"""

import os
import sys
from pathlib import Path
import geopandas as gpd
import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

from utils.plots import (plot_inspection_sites, plot_cube_selection, 
                         plot_inspection_sites_folds, plot_prediction_vs_target, 
                         plot_prediction_vs_target_small, 
                         plot_labels_xarray, plot_layers, 
                         plot_training_timeline, 
                         plot_binned_precision_recall_percent,
                         plot_training_curves, plot_cube_polygon_leaflet)
from utils.random import add_osm_links_and_bund
from analysis.regression import get_run_evaluation_table
from analysis.classification import get_stats
from analysis.inspect_labels import (compute_label_inspection, 
                                     calculate_object_metrics, 
                                     objects_for_cubes, 
                                     aggregate_stats)
from analysis.model_selection import load_fold_data
from analysis.ig_df import collapse_modalities

# Run the fix
#fix_ig_channel_labels(run_name="run_10", threshold=0.2)

####################
# Locations
####################
gdf_full = gpd.read_file(PATHS.meta_data_dir / "available_cubes.gpkg")
gdf_balanced = gpd.read_file(PATHS.meta_data_dir / "balanced_cube_set.gpkg")
plot_cube_selection(gdf_full, gdf_balanced, outdir="results",save=True)


gdf_set = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
gdf_grid = gpd.read_file(PATHS.meta_data_dir / "spatial_folds_grid.gpkg")
plot_inspection_sites_folds(gdf_set, gdf_grid, outdir="results", save=True)

SELECTED_LOCATIONS=[51, 99, 104, 171, 185, 212, 221, 243, 251, 315]
plot_inspection_sites(gdf_set, selected_locations=SELECTED_LOCATIONS, outdir="results", save=True)

df = add_osm_links_and_bund(gdf_set[gdf_set["is_holdout"]==True], selected_locations=SELECTED_LOCATIONS)

df_final = df.rename(columns={"cube_id": "Cube ID","federal_state": "Federal State", "lat": "Latitude", "lon": "Longitude", "OSM_Link": "OSM Link"})
test = df_final.to_latex(
    index=False,
    float_format='%.6f',
    column_format='c l r r r',
    caption='List of all inspection site locations coordinates (EPSG:4326) with OpenStreetMap links for further spatial context.',
    label='tab:inspection_cubes_location',
    position='!h')

print(test)

####################
# Training Timeline
####################

df = pd.read_csv(PATHS.training_runs / "run_10" / "training_logs.csv")
plot_training_timeline(df, outdir="results", save=False)

# Training curves
summary_df, fold_data = load_fold_data(run_name="run_10")
plot_training_curves(summary_df, fold_data, outdir="results", save=False)

# format latex
smry = summary_df.copy()
smry = smry.drop(columns=['eval_samples'])
smry = smry.rename(columns={
    "fold": "Fold",
    "best_epoch": "Best epoch",
    "composite_score": "CPI",
    "val_loss": "Val loss",
    "train_samples": "Training samples",
})
smry = smry.to_latex(
    index=False,
    float_format='%.5f',
    column_format='c| r r r',
    caption=('Summary of training curves across folds. Best epoch, Composition Performance Index (CPI), and validation loss at best epoch are shown for each fold.','Training curves'),
    label='tab:training_curves',
    position='htbp'
)
print(smry)

####################
# Regression
####################
RUN = "run_10"
THRESHOLD = 0.02
regression_df = get_run_evaluation_table(RUN)
#regression_df = regression_df.to_csv(PATHS.training_runs / "run_10" / "regression_results.csv", index=False)

# Latex
regression_df = regression_df[regression_df['Set'] != 'Holdout']
processed_df = regression_df.copy()

processed_df['Samples'] = processed_df['Samples'].fillna(0).astype(int)

REGRESSION_CAPTION = ('Regression performance metrics for annual fractional deadwood cover increase. '
                      'Results are reported for individual spatial cross-validation folds, '
                      'the concatenated global mean with standard deviation, and the independent holdout set. ')

latex_table = processed_df.to_latex(
    index=False,
    float_format='%.3f',
    column_format='l| r r r',
    caption=(REGRESSION_CAPTION,'Regression performance metrics'),
    label='tab:regression_results',
    position='htbp'
)
print(latex_table)

####################
# Classification
####################
RUN = "run_10"
THRESHOLD = 0.02
holdout_stats, cv_concat_stats, fold_stats_list = get_stats(RUN, threshold=THRESHOLD, n_folds=4, min_obs=200)
plot_binned_precision_recall_percent(holdout_stats, cv_concat_stats, fold_stats_list, outdir="results", save=True)

####################
# Get prediction
####################
CUBE_ID = 51
RUN = "run_10"
pred_path = PATHS.predictions / RUN / f"{CUBE_ID}_{RUN}_prediction.zarr"
pred_ds = xr.open_zarr(pred_path)
plot_prediction_vs_target(pred_ds, CUBE_ID, vmax=0.6, outdir="results", save=False, show=False)
plot_prediction_vs_target_small(pred_ds, CUBE_ID, vmax=0.6, outdir="results/full_scale_predictions", save=False, show=False)

####################
# Plot all holdout predictions
####################
RUN = "run_10"
gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
holdout_ids = gdf[gdf["is_holdout"]==True]["cube_id"].tolist()

import matplotlib
matplotlib.use("Agg")
outdir= Path("results/full_scale_predictions")
outdir_selected = outdir / "selected_locations"
os.makedirs(outdir_selected, exist_ok=True)
for cube_id in tqdm(holdout_ids, desc="Plotting holdout predictions", total=len(holdout_ids)):
    pred_path = PATHS.predictions / RUN / f"{cube_id}_{RUN}_prediction.zarr"
    pred_ds = xr.open_zarr(pred_path)
    if cube_id == 51:
        
        plot_prediction_vs_target(pred_ds, cube_id, vmax=0.8, outdir=outdir_selected, save=True, show=False)
        continue
    elif cube_id in SELECTED_LOCATIONS:
        plot_prediction_vs_target_small(pred_ds, cube_id, vmax=0.8, outdir=outdir_selected, save=True, show=False)
    else:
        plot_prediction_vs_target_small(pred_ds, cube_id, vmax=0.8, outdir=outdir, save=True, show=False)

####################
# Check the prediction against the actual data_cube
####################

gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
plot_cube_polygon_leaflet(gdf[gdf['cube_id'] == 251], cube_id=251)

CUBE_ID = 251
RUN = "run_10"
pred_path = PATHS.predictions / RUN / f"{CUBE_ID}_{RUN}_prediction.zarr"
pred_ds = xr.open_zarr(pred_path)
ds_pred_years = pred_ds.time_year.values

#CHECK_ID = 258
CHECK_ID = 251
zarr_path = PATHS.cubes / f"{CHECK_ID}.zarr"
cube_ds = xr.open_zarr(zarr_path, group="high_res")
deadwood_forest = cube_ds['deadwood_forest']
ds_selected = deadwood_forest.sel(time_year=ds_pred_years)

# Mask if it is the same cube ass the predciton
# valid_mask = (
#     pred_ds["prediction"].notnull() &
#     pred_ds["target"].notnull()
# )
# ds_selected_masked = ds_selected.where(valid_mask)

plot_layers(ds_selected, pred_ds, CUBE_ID, save=False)

####################
# Integrated gradients
####################
# Run IG analysis in tmux session (takes a while):
#   python analysis/ig_df.py --run run_10 --threshold 0.2 --n_steps 50 --device 0
#   python analysis/ig_cube.py --run run_10 --cube_id 51 --threshold 0.2 --n_steps 50 --device 0
RUN = "run_10"
ig_df_path = PATHS.training_runs / RUN / "integrated_gradients" / f"ig_attributions_0.2_summary.csv"
ig_51_path = PATHS.training_runs / RUN / "integrated_gradients" / f"ig_cube_51_t0.2_y2023_summary.csv"
ig_251_path = PATHS.training_runs / RUN / "integrated_gradients" / f"ig_cube_251_t0.2_y2022_summary.csv"

ig_df = pd.read_csv(ig_df_path)
ig_51 = pd.read_csv(ig_51_path)
ig_251 = pd.read_csv(ig_251_path)

ig_df_collapsed = collapse_modalities(ig_df, modalities_to_collapse=['worldclim', 'soilgrids'])
ig_51_collapsed = collapse_modalities(ig_51, modalities_to_collapse=['worldclim', 'soilgrids'])
ig_251_collapsed = collapse_modalities(ig_251, modalities_to_collapse=['worldclim', 'soilgrids'])

MODALITY_LABELS = {
    'deadwood_forest': 'Cover Predictions',
    'sentinel': 'Sentinel',
    'era5': 'ERA5-Land',
    'worldclim': 'WorldClim',
    'canopy': 'CanopyHeight',
    'terrain': 'Terrain',
    'soilgrids': 'SoilGrids',
    'stand_age': 'StandAge',}

def clean_df(df):
    df_clean = df.copy()
    df_clean['modality'] = df_clean['modality'].str.replace('_', ' ').str.title()
    df_clean['channel'] = df_clean['channel'].str.replace('_', ' ').str.title()
    df_clean['modality'] = df_clean['modality'].replace(MODALITY_LABELS)
    n_samples = df['n_samples'].iloc[0]
    delta = df['mean_delta'].abs().mean()
    # Convert to scientific notation for LaTeX
    delta = f"{delta:.2e}"
    df_clean = df_clean.drop(columns=['n_samples', 'mean_delta'])
    df_clean = df_clean.rename(columns={"modality": "Modality","channel": "Channel","mean":"Mean","std":"Std","cv":"Cv"})
    return df_clean , n_samples, delta

df_clean , n_samples, delta = clean_df(ig_df_collapsed)
CAPTION_IG_DF = ('Integrated Gradients feature attributions for deadwood cover increase prediction. '
                f'Mean absolute attribution magnitudes across  \\num{{{n_samples}}} pixels with fractional deadwood cover increase \\qty{{> 20}}{{pp}}. '
                'Spatial and temporal dimensions are summed per channel; auxiliary modalities (WorldClim, SoilGrids) show mean-per-channel values. '
                f'Cv = coefficient of variation. Mean delta (convergence): \\num{{{delta}}}.')
latex_table = df_clean.to_latex(
    index=False,
    float_format='%.4f',
    column_format='l l| r r r',
    caption=(CAPTION_IG_DF,'Integrated gradients feature attributions'),
    label='tab:attribution_summary_long',
    position='!h'
)
print(latex_table)


df_clean , n_samples, delta = clean_df(ig_51_collapsed)
CAPTION_IG_51 = ('Integrated Gradients feature attributions for deadwood cover increase prediction of cube \\num{51} in \\num{2024}. '
                f'Mean absolute attribution magnitudes across \\num{{{n_samples}}} pixels  with fractional deadwood cover increase \\qty{{> 20}}{{pp}}. '
                'Spatial and temporal dimensions are summed per channel; auxiliary modalities (WorldClim, SoilGrids) show mean-per-channel values. '
                f'Cv = coefficient of variation. Mean delta (convergence): \\num{{{delta}}}.')
latex_table = df_clean.to_latex(
    index=False,
    float_format='%.4f',
    column_format='l l| r r r',
    caption=(CAPTION_IG_51,'Integrated gradients feature attributions cube \\num{51}'),
    label='tab:attribution_summary_51_long',
    position='!h'
)
print(latex_table)


df_clean , n_samples, delta = clean_df(ig_251_collapsed)
CAPTION_IG_251 = ('Integrated Gradients feature attributions for deadwood cover increase prediction of cube \\num{251} in \\num{2023}. '
                f'Mean absolute attribution magnitudes across \\num{{{n_samples}}} pixels  with fractional deadwood cover increase \\qty{{> 20}}{{pp}}. '
                'Spatial and temporal dimensions are summed per channel; auxiliary modalities (WorldClim, SoilGrids) show mean-per-channel values. '
                f'Cv = coefficient of variation. Mean delta (convergence): \\num{{{delta}}}.')
latex_table = df_clean.to_latex(
    index=False,
    float_format='%.4f',
    column_format='l l| r r r',
    caption=(CAPTION_IG_251,'Integrated gradients feature attributions cube \\num{251}'),
    label='tab:attribution_summary_251_long',
    position='!h'
)
print(latex_table)


####################
# Inspect labeling
####################
CUBE_ID = 51
RUN = "run_10"
pred_path = PATHS.predictions / RUN / f"{CUBE_ID}_{RUN}_prediction.zarr"
pred_ds = xr.open_zarr(pred_path)

BINARY_THRESHOLD = 0.1
MERGE_RADIUS = 1
MIN_PIXELS = 12
COVERAGE_THRESHOLD = 0.5
PRECISION_OVERLAP = 0.1

result = compute_label_inspection(
    pred_da=pred_ds,
    year_idx=3,
    binary_threshold=BINARY_THRESHOLD,
    merge_radius=MERGE_RADIUS,
    min_pixels=MIN_PIXELS,
)

plot_labels_xarray(result=result, cube_id=CUBE_ID, save=False)

metrics = calculate_object_metrics(
    p_labels=result["p_labels"],
    t_labels=result["t_labels"],
    coverage_threshold=COVERAGE_THRESHOLD,
    precision_overlap=PRECISION_OVERLAP,
)

metrics

# Calculate for all holdout cubes
RUN = "run_10"
gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
holdout_ids = gdf[gdf["is_holdout"]==True]["cube_id"].tolist()

all_objects = objects_for_cubes(
    holdout_ids, 
    run=RUN, 
    binary_threshold=BINARY_THRESHOLD,
    merge_radius=MERGE_RADIUS,
    min_pixels=MIN_PIXELS, 
    coverage_threshold=COVERAGE_THRESHOLD, 
    precision_overlap=PRECISION_OVERLAP,
)
summary_stats = aggregate_stats(all_objects)
summary_stats


# Global performance was computed by micro-averaging, i.e., 
# summing true positives, false positives, and false negatives across all cubes 
# and then recomputing precision, recall, and F1 from these totals.





