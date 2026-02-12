"""
Classification metrics and binned evaluation statistics.
Computes precision, recall, and performance metrics binned by prediction/target magnitude ranges.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

from utils.plots import FIGURE_CLASSES

def calculate_binned_stats(df, threshold=0.02, min_obs=20):
    """
    Calculate precision and recall binned by magnitude.
    Bins with fewer than min_obs are set to NaN.
    """
    bins = np.arange(0, 1.05, 0.05)
    labels = [f"{round(b, 2)}-{round(b+0.05, 2)}" for b in bins[:-1]]
    df = df.copy()

    df['target_is_present'] = df['target'] > threshold
    df['pred_is_present'] = df['prediction'] > threshold

    # --- PRECISION (Grouped by Predicted Range) ---
    df['pred_bin'] = pd.cut(df['prediction'], bins=bins, labels=labels, include_lowest=True)
    precision_stats = df.groupby('pred_bin', observed=False)['target_is_present'].agg(['count', 'mean'])
    precision_stats.columns = ['n_obs_pred', 'precision']

    # --- RECALL (Grouped by Target Range) ---
    df_recall_source = df[df['target'] > threshold].copy()
    df_recall_source['target_bin'] = pd.cut(df_recall_source['target'], bins=bins, labels=labels, include_lowest=True)
    recall_stats = df_recall_source.groupby('target_bin', observed=False)['pred_is_present'].agg(['count', 'mean'])
    recall_stats.columns = ['n_obs_target', 'recall']

    # --- UNIFY ---
    stats_df = precision_stats.join(recall_stats).reset_index()
    stats_df.rename(columns={'pred_bin': 'bin_range'}, inplace=True)

    # Apply the min_obs filter: convert metrics to NaN if count is too low
    stats_df.loc[stats_df['n_obs_pred'] < min_obs, 'precision'] = np.nan
    stats_df.loc[stats_df['n_obs_target'] < min_obs, 'recall'] = np.nan

    return stats_df

def _wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / denom
    return center - half, center + half

def calculate_binned_stats_global(df, threshold=0.02, min_obs=20, alpha=0.05):
    """
    Calculate binned precision/recall with global (all-pixel) confidence intervals.
    Bins with fewer than min_obs are set to NaN.
    """
    bins = np.arange(0, 1.05, 0.05)
    labels = [f"{round(b, 2)}-{round(b+0.05, 2)}" for b in bins[:-1]]
    df = df.copy()

    df['target_is_present'] = df['target'] > threshold
    df['pred_is_present'] = df['prediction'] > threshold

    # --- PRECISION (Grouped by Predicted Range) ---
    df['pred_bin'] = pd.cut(df['prediction'], bins=bins, labels=labels, include_lowest=True)
    precision_stats = df.groupby('pred_bin', observed=False)['target_is_present'].agg(['count', 'sum', 'mean'])
    precision_stats.columns = ['n_obs_pred', 'tp_pred', 'precision']

    # --- RECALL (Grouped by Target Range) ---
    df_recall_source = df[df['target'] > threshold].copy()
    df_recall_source['target_bin'] = pd.cut(df_recall_source['target'], bins=bins, labels=labels, include_lowest=True)
    recall_stats = df_recall_source.groupby('target_bin', observed=False)['pred_is_present'].agg(['count', 'sum', 'mean'])
    recall_stats.columns = ['n_obs_target', 'tp_target', 'recall']

    # --- UNIFY ---
    stats_df = precision_stats.join(recall_stats).reset_index()
    stats_df.rename(columns={'pred_bin': 'bin_range'}, inplace=True)

    # Apply the min_obs filter: convert metrics to NaN if count is too low
    stats_df.loc[stats_df['n_obs_pred'] < min_obs, ['precision']] = np.nan
    stats_df.loc[stats_df['n_obs_target'] < min_obs, ['recall']] = np.nan

    # Compute Wilson CIs (only for bins that pass min_obs)
    precision_ci = stats_df.apply(
        lambda r: _wilson_interval(int(r['tp_pred']), int(r['n_obs_pred']), alpha)
        if r['n_obs_pred'] >= min_obs else (np.nan, np.nan),
        axis=1
    )
    recall_ci = stats_df.apply(
        lambda r: _wilson_interval(int(r['tp_target']), int(r['n_obs_target']), alpha)
        if r['n_obs_target'] >= min_obs else (np.nan, np.nan),
        axis=1
    )

    stats_df['precision_low'] = [x[0] for x in precision_ci]
    stats_df['precision_high'] = [x[1] for x in precision_ci]
    stats_df['recall_low'] = [x[0] for x in recall_ci]
    stats_df['recall_high'] = [x[1] for x in recall_ci]

    return stats_df

def calculate_binned_stats_global_standard_ci(df, threshold=0.02, min_obs=20, alpha=0.05):
    """
    Calculate binned precision/recall with standard (Wald) confidence intervals.
    Bins with fewer than min_obs are set to NaN.
    """
    bins = np.arange(0, 1.05, 0.05)
    labels = [f"{round(b, 2)}-{round(b+0.05, 2)}" for b in bins[:-1]]
    df = df.copy()

    df['target_is_present'] = df['target'] > threshold
    df['pred_is_present'] = df['prediction'] > threshold

    # --- PRECISION (Grouped by Predicted Range) ---
    df['pred_bin'] = pd.cut(df['prediction'], bins=bins, labels=labels, include_lowest=True)
    precision_stats = df.groupby('pred_bin', observed=False)['target_is_present'].agg(['count', 'sum', 'mean'])
    precision_stats.columns = ['n_obs_pred', 'tp_pred', 'precision']

    # --- RECALL (Grouped by Target Range) ---
    df_recall_source = df[df['target'] > threshold].copy()
    df_recall_source['target_bin'] = pd.cut(df_recall_source['target'], bins=bins, labels=labels, include_lowest=True)
    recall_stats = df_recall_source.groupby('target_bin', observed=False)['pred_is_present'].agg(['count', 'sum', 'mean'])
    recall_stats.columns = ['n_obs_target', 'tp_target', 'recall']

    # --- UNIFY ---
    stats_df = precision_stats.join(recall_stats).reset_index()
    stats_df.rename(columns={'pred_bin': 'bin_range'}, inplace=True)

    # Apply the min_obs filter: convert metrics to NaN if count is too low
    stats_df.loc[stats_df['n_obs_pred'] < min_obs, ['precision']] = np.nan
    stats_df.loc[stats_df['n_obs_target'] < min_obs, ['recall']] = np.nan

    z = norm.ppf(1 - alpha / 2)

    def _wald_ci(p, n):
        if n <= 0 or np.isnan(p):
            return np.nan, np.nan
        se = np.sqrt((p * (1 - p)) / n)
        low = max(0.0, p - z * se)
        high = min(1.0, p + z * se)
        return low, high

    precision_ci = stats_df.apply(
        lambda r: _wald_ci(r['precision'], int(r['n_obs_pred']))
        if r['n_obs_pred'] >= min_obs else (np.nan, np.nan),
        axis=1
    )
    recall_ci = stats_df.apply(
        lambda r: _wald_ci(r['recall'], int(r['n_obs_target']))
        if r['n_obs_target'] >= min_obs else (np.nan, np.nan),
        axis=1
    )

    stats_df['precision_low'] = [x[0] for x in precision_ci]
    stats_df['precision_high'] = [x[1] for x in precision_ci]
    stats_df['recall_low'] = [x[0] for x in recall_ci]
    stats_df['recall_high'] = [x[1] for x in recall_ci]

    return stats_df

def get_stats(run_name, threshold=0.02, n_folds=4, min_obs=20):
    all_cv_dfs = []
    fold_individual_stats = []
    
    for i in range(1, n_folds + 1):
        # Adjusted path to include kmeans_folds as per your snippet
        f_path = PATHS.training_runs / run_name / f"fold_{i}" / "evaluation_results.parquet"
        if f_path.exists():
            df_f = pd.read_parquet(f_path)
            all_cv_dfs.append(df_f)
            fold_individual_stats.append(calculate_binned_stats(df_f, threshold, min_obs=min_obs))
    
    df_all_cv = pd.concat(all_cv_dfs)
    cv_concatenated_stats = calculate_binned_stats(df_all_cv, threshold, min_obs=min_obs)
    
    main_path = PATHS.training_runs / run_name / "evaluation_results.parquet"
    df_holdout = pd.read_parquet(main_path)
    holdout_stats = calculate_binned_stats(df_holdout, threshold, min_obs=min_obs)
    
    return holdout_stats, cv_concatenated_stats, fold_individual_stats

def get_global_stats_from_folds(run_name, threshold=0.02, n_folds=4, min_obs=20, alpha=0.05):
    """
    Load evaluation data from all folds, concatenate, and compute global binned stats
    with confidence intervals across all pixels.
    """
    all_cv_dfs = []

    for i in range(1, n_folds + 1):
        f_path = PATHS.training_runs / run_name / f"fold_{i}" / "evaluation_results.parquet"
        if f_path.exists():
            df_f = pd.read_parquet(f_path)
            all_cv_dfs.append(df_f)
        else:
            print(f"Warning: {f_path} not found. Skipping.")

    if len(all_cv_dfs) == 0:
        raise FileNotFoundError("No fold evaluation results found.")

    df_all_cv = pd.concat(all_cv_dfs)
    return calculate_binned_stats_global(df_all_cv, threshold=threshold, min_obs=min_obs, alpha=alpha)

def get_global_stats_from_folds_standard_ci(run_name, threshold=0.02, n_folds=4, min_obs=20, alpha=0.05):
    """
    Load evaluation data from all folds, concatenate, and compute global binned stats
    with standard (Wald) confidence intervals across all pixels.
    """
    all_cv_dfs = []

    for i in range(1, n_folds + 1):
        f_path = PATHS.training_runs / run_name / f"fold_{i}" / "evaluation_results.parquet"
        if f_path.exists():
            df_f = pd.read_parquet(f_path)
            all_cv_dfs.append(df_f)
        else:
            print(f"Warning: {f_path} not found. Skipping.")

    if len(all_cv_dfs) == 0:
        raise FileNotFoundError("No fold evaluation results found.")

    df_all_cv = pd.concat(all_cv_dfs)
    return calculate_binned_stats_global_standard_ci(
        df_all_cv,
        threshold=threshold,
        min_obs=min_obs,
        alpha=alpha,
    )

def plot_single_location_performance(df, threshold=0.02, title="Local Site Performance"):
    """
    Plots Precision and Recall for a single prediction dataframe.
    X-axis is scaled to 0-100%.
    """
    # 1. Calculate the binned stats for this specific DF
    stats = calculate_binned_stats(df, threshold=threshold)
    stats = stats.dropna(subset=['precision', 'recall'], how='all')
    # 2. Convert bin ranges to percentage midpoints
    def get_percent_midpoint(bin_str):
        low, high = bin_str.split('-')
        return ((float(low) + float(high)) / 2) * 100
    
    stats['perc_mid'] = stats['bin_range'].apply(get_percent_midpoint)
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Precision Line + Blobs
    ax.plot(stats['perc_mid'], stats['precision'], 
            color='royalblue', linewidth=2, marker='o', 
            markersize=8, label='Precision', markeredgecolor='white')
    
    # Recall Line + Blobs
    ax.plot(stats['perc_mid'], stats['recall'], 
            color='forestgreen', linewidth=2, marker='s', 
            markersize=8, label='Recall', markeredgecolor='white')
    
    # Formatting
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    
    ax.set_xlabel("Annual Fractional Deadwood Cover Increase [pp]", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    return fig, ax

