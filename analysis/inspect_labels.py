"""
Label inspection and validation utilities.
Computes binary masks, object detection metrics, and TP/FP/FN overlays for prediction validation.
"""

import sys
from pathlib import Path
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from skimage.morphology import closing, footprint_rectangle
from skimage.measure import label, regionprops
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

def compute_label_inspection(pred_da, year_idx, binary_threshold, merge_radius, min_pixels):
    """
    Pure computation:
    From xarray -> masks -> labels -> TP/FP/FN overlay.
    Returns all arrays and metadata needed for plotting.
    """
    t = pred_da.time_year.values[year_idx]
    year = np.datetime_as_string(t, unit='Y')

    pred_slice = pred_da.prediction.sel(time_year=t).squeeze().compute().values
    target_slice = pred_da.target.sel(time_year=t).squeeze().compute().values

    pred_arr = np.nan_to_num(pred_slice, nan=0.0)
    target_arr = np.nan_to_num(target_slice, nan=0.0)

    x = pred_da.x.values
    y = pred_da.y.values
    extent = [x.min(), x.max(), y.min(), y.max()]

    # --- Binary masks
    p_mask = (pred_arr > binary_threshold).astype(int)
    t_mask = (target_arr > binary_threshold).astype(int)

    if merge_radius > 0:
        size = (merge_radius * 2) + 1
        footprint = footprint_rectangle((size, size))
        p_mask = closing(p_mask, footprint)
        t_mask = closing(t_mask, footprint)

    # --- Labels
    p_labels = label(p_mask, connectivity=2)
    t_labels = label(t_mask, connectivity=2)

    clean_p_mask = np.zeros_like(p_mask)
    for region in regionprops(p_labels):
        if region.area >= min_pixels:
            coords = region.coords
            clean_p_mask[coords[:, 0], coords[:, 1]] = 1

    clean_t_mask = np.zeros_like(t_mask)
    for region in regionprops(t_labels):
        if region.area >= min_pixels:
            coords = region.coords
            clean_t_mask[coords[:, 0], coords[:, 1]] = 1

    # --- Relabel after cleaning
    p_labels = label(clean_p_mask, connectivity=2)
    t_labels = label(clean_t_mask, connectivity=2)

    p_mask = clean_p_mask
    t_mask = clean_t_mask

    return {
        "year": year,
        "extent": extent,
        "pred_arr": pred_arr,
        "target_arr": target_arr,
        "p_mask": p_mask,
        "t_mask": t_mask,
        "p_labels": p_labels,
        "t_labels": t_labels,
    }

def calculate_object_metrics(
    p_labels: np.ndarray,
    t_labels: np.ndarray,
    coverage_threshold: float,
    precision_overlap: float,
):
    """
    Object-level precision/recall based on spatial coverage between labeled blobs.

    Parameters
    ----------
    p_labels : np.ndarray
        Labeled prediction objects (0 = background).
    t_labels : np.ndarray
        Labeled target objects (0 = background).
    coverage_threshold : float
        Minimum fraction of a target object that must be covered by a prediction
        to count as a True Positive (recall criterion).
    precision_overlap : float
        Minimum fraction of a prediction object that must overlap a target
        to count as a True Positive (precision criterion).

    Returns
    -------
    dict with precision, recall, f1 and raw counts.
    """

    # Boolean masks for fast overlap checks
    p_mask = p_labels > 0
    t_mask = t_labels > 0

    # Extract object properties
    p_props = regionprops(p_labels)
    t_props = regionprops(t_labels)

    # -------------------------
    # Recall (target-centric)
    # -------------------------
    tp_targets = 0
    total_targets = len(t_props)

    for t_prop in t_props:
        rows, cols = t_prop.coords[:, 0], t_prop.coords[:, 1]
        overlap_pixels = np.sum(p_mask[rows, cols])
        coverage_ratio = overlap_pixels / t_prop.area

        if coverage_ratio >= coverage_threshold:
            tp_targets += 1

    recall = tp_targets / total_targets if total_targets > 0 else 0.0

    # -------------------------
    # Precision (prediction-centric)
    # -------------------------
    tp_preds = 0
    total_preds = len(p_props)

    for p_prop in p_props:
        rows, cols = p_prop.coords[:, 0], p_prop.coords[:, 1]
        overlap_pixels = np.sum(t_mask[rows, cols])

        pred_coverage_ratio = overlap_pixels / p_prop.area
        if pred_coverage_ratio >= precision_overlap:
            tp_preds += 1

    precision = tp_preds / total_preds if total_preds > 0 else 0.0

    # -------------------------
    # F1 score
    # -------------------------
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "targets_found": tp_targets,
        "total_targets": total_targets,
        "preds_valid": tp_preds,
        "total_preds": total_preds,
    }

def objects_for_cubes(holdout_ids, run, binary_threshold, merge_radius, min_pixels, coverage_threshold, precision_overlap):
    holdout_stats = {}
    for cube_id in tqdm(holdout_ids, desc="Inspecting holdout cubes", total=len(holdout_ids)):
        pred_path = PATHS.predictions / run / f"{cube_id}_{run}_prediction.zarr"
        pred_ds = xr.open_zarr(pred_path)

        for i in range(len(pred_ds.time_year)):

            result = compute_label_inspection(
                pred_da=pred_ds,
                year_idx=i,
                binary_threshold=binary_threshold,
                merge_radius=merge_radius,
                min_pixels=min_pixels,
            )

            metrics = calculate_object_metrics(
                p_labels=result["p_labels"],
                t_labels=result["t_labels"],
                coverage_threshold=coverage_threshold,
                precision_overlap=precision_overlap,
            )
            holdout_stats[str(cube_id) + f"_{i}"] = metrics
    return holdout_stats

def aggregate_stats(holdout_stats):
    tp = 0
    fp = 0
    fn = 0

    for m in holdout_stats.values():
        tp += m["targets_found"]
        fp += m["total_targets"] - m["targets_found"]
        fn += m["total_preds"] - m["preds_valid"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    global_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
    return global_metrics
