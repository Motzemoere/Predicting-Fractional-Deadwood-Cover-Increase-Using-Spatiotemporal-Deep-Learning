"""
Regression metrics and model evaluation.
Computes Pearson correlation, RMSE, and aggregates cross-validation fold results.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

def calculate_metrics(df, label, thresh):
    """Calculates regression metrics for a given subset."""
    mask = (df['prediction'] > thresh) | (df['target'] > thresh)
    df_filt = df[mask].copy()
    
    if len(df_filt) < 2:
        return {
            "Set": label, 
            "Pearson r": np.nan, 
            "RMSE": np.nan, 
            "Samples": len(df_filt)
        }

    r_val, _ = pearsonr(df_filt['prediction'], df_filt['target'])
    rmse_val = root_mean_squared_error(df_filt['target'], df_filt['prediction'])
    
    return {
        "Set": label,
        "Pearson r": r_val,
        "RMSE": rmse_val,
        "Samples": int(len(df_filt))
    }

def get_run_evaluation_table(run_name, threshold=0.02, n_folds=4):
    """
    Aggregates CV fold results and holdout results into a single summary table.
    Returns a formatted pandas DataFrame.
    """
    # Holdout Data
    main_path = PATHS.training_runs / run_name / "evaluation_results.parquet"
    if not main_path.exists():
        raise FileNotFoundError(f"Main results not found at {main_path}")
    df_holdout = pd.read_parquet(main_path)

    # Folds
    fold_results = []
    for i in range(1, n_folds + 1):
        fold_dir = f"fold_{i}"
        f_path = PATHS.training_runs / run_name / fold_dir / "evaluation_results.parquet"
        
        if f_path.exists():
            df_f = pd.read_parquet(f_path)
            fold_results.append(calculate_metrics(df_f, f"Fold {i}", threshold))
        else:
            print(f"Warning: {f_path} not found. Skipping.")

    cv_results = pd.DataFrame(fold_results)
    
    cv_summary = pd.DataFrame([
        {
            "Set": "Global",
            "Pearson r": cv_results["Pearson r"].mean(),
            "RMSE": cv_results["RMSE"].mean(),
            "Samples": cv_results["Samples"].sum()
        },
        {
            "Set": "Std Dev",
            "Pearson r": cv_results["Pearson r"].std(),
            "RMSE": cv_results["RMSE"].std(),
            "Samples": np.nan
        }
    ])

    holdout_metrics = calculate_metrics(df_holdout, "Holdout", threshold)
    holdout_df = pd.DataFrame([holdout_metrics])

    final_table = pd.concat([cv_results, cv_summary, holdout_df], ignore_index=True)
    final_table['Samples'] = pd.to_numeric(final_table['Samples'])
    
    # Convert RMSE from fractional to percentage points for display
    final_table['RMSE'] = final_table['RMSE'] * 100
    
    return final_table
