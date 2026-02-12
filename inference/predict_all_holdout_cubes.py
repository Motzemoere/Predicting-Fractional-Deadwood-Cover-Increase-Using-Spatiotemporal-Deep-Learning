"""
Generate predictions for all holdout cubes from a trained model run.
Saves predictions as zarr files and optionally generates comparison plots.
"""

import sys
from pathlib import Path
import geopandas as gpd
import argparse

import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

from training.train_utils import load_config_from_json, predict, FEATURE_KEYS
from training.setup_training import setup_cube, setup_prediction, built_cube_array
from utils.plots import plot_prediction_vs_target

# Save as zarr
def clear_all_encodings(ds):
    ds.encoding = {}
    for var in ds.variables:
        ds[var].encoding = {}
    return ds

def predict_all_holdout_cubes(run, device, plot=False):
    run_dir = PATHS.training_runs / run
    out_dir = PATHS.predictions / run
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir_figs = out_dir / "figs"
    cfg = load_config_from_json(run_dir / "config.json")
    cfg.device = device
    gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")
    holdout_gdf = gdf[gdf["is_holdout"] == True]

    # check which cubes have already been predicted
    existing_predictions = list(out_dir.glob("*_prediction.zarr"))
    existing_ids = [int(f.name.split('_')[0]) for f in existing_predictions]
    to_process = holdout_gdf[~holdout_gdf["cube_id"].isin(existing_ids)]

    print(f"Predicting {len(to_process)} holdout cubes for {run}...")
    for cube_id in to_process["cube_id"].values:

        df, shared_tensors = setup_cube(
            cube_id = cube_id, 
            cubes_dir = PATHS.cubes, 
            meta_dir =  PATHS.meta_data_dir)

        pred_loader, model = setup_prediction(
            cfg = cfg, 
            data_dir = PATHS.data_dir, 
            df = df, 
            shared_tensors=shared_tensors,
            device=cfg.device)

        metrics, result_df = predict(
                cfg=cfg,
                target=cfg.target_name,
                model=model,
                dataloader=pred_loader,
                feature_keys=FEATURE_KEYS,
                log_space=cfg.train_log_space,
                return_predictions=True
            )

        cube_results = df.merge(result_df, on=['pixel_key', 'year'])


        pred_ds = built_cube_array(cube_id = cube_id,
                                 cubes_dir = PATHS.cubes, 
                                 results_df = cube_results)

        # Save as zarr
        pred_ds = clear_all_encodings(pred_ds)
        pred_ds.to_zarr(out_dir /  f"{cube_id}_{run}_prediction.zarr", mode='w', zarr_format=2, consolidated=True)

        # save the plot
        if plot:
            out_dir_figs.mkdir(parents=True, exist_ok=True)
            plot_prediction_vs_target(pred_ds, cube_id, vmax=1, save=True, out_dir=out_dir_figs)
            plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict all holdout cubes for a given training run.")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--plot", required=True, action='store_true')
    args = parser.parse_args()

    device = f"cuda:{args.device}"

    predict_all_holdout_cubes(run=args.run, device=device, plot=args.plot)