"""
Model selection and cross-validation analysis.
Loads training logs and evaluates per-fold performance metrics for model comparison.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import geopandas as gpd
from training.setup_training import split_folds
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

# Load training logs and best model metadata for all folds
def load_fold_data(run_name="run_10"):
    """Load training logs and best model metadata for all folds."""
    run_dir = PATHS.training_runs / run_name
    
    results = []
    fold_data = {}
    
    # Load training set metadata once for efficiency
    gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")


    training_set_path = PATHS.data_dir / "training_sets" / "training_set_01" / "train_baseline.parquet"
    training_meta = pd.read_parquet(training_set_path)
    
    # Find all fold directories
    fold_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name.split("_")[1])

        df_train, df_eval = split_folds(training_meta, gdf, fold_num=fold_num)
        n_t = len(df_train)
        n_e = len(df_eval)
        
        # Load training logs
        log_path = fold_dir / "training_logs.csv"
        logs_df = pd.read_csv(log_path)
        
        # Load best model metadata
        best_meta_path = fold_dir / "best_model_meta.json"
        
        with open(best_meta_path, "r") as f:
            best_meta = json.load(f)
        
        best_epoch = best_meta.get("epoch", None)
        composite_score = best_meta.get("score", None)

        # Lookup val_loss at best epoch from logs
        if best_epoch is not None and "epoch" in logs_df.columns and "val_loss" in logs_df.columns:
            epoch_match = logs_df.loc[logs_df["epoch"] == best_epoch, "val_loss"]
            best_val_loss = float(epoch_match.iloc[0]) if len(epoch_match) > 0 else np.nan
        else:
            best_val_loss = np.nan
        
        # Store fold data
        results.append({
            "fold": fold_num,
            "best_epoch": best_epoch,
            "composite_score": composite_score,
            "val_loss": best_val_loss,
            "train_samples": n_t,
            "eval_samples": n_e,
        })
        
        fold_data[fold_num] = {
            "logs_df": logs_df,
            "best_epoch": best_epoch,
            "composite_score": composite_score,
            "best_val_loss": best_val_loss,
        }
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results).sort_values("fold").reset_index(drop=True)
    
    return summary_df, fold_data
