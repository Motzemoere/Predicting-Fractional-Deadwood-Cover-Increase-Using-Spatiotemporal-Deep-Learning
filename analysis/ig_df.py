"""
Integrated Gradients attribution analysis across multiple cubes.
Computes and aggregates feature importance scores for model interpretability.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS
from training.train_utils import FEATURE_KEYS, load_config_from_json
from training.setup_training import preload_all_cubes_parallel, TrainingDataset
from models.model_small import MultimodalDeadwoodTransformer

####################
# CONSTANTS
####################
CHANNEL_LABELS = {
    "deadwood_forest": ['Deadwood Cover', 'Forest Cover', 'Deadwood Inc', 'Forest Dec'],
    "terrain": ['Slope', 'DEM', 'Downslope Index', 'Aspect'],
    "canopy": ['Canopy Height', 'Canopy Height Std'],
    "sentinel": ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
    "era5": ['t2m_q05', 't2m_mean', 't2m_q95', 'tp_q95', 'tp_mean', 'tp_q05', 
             'd2m_q95', 'd2m_q05', 'd2m_mean', 'ssrd_mean', 'ssrd_q95'],
    "worldclim": [f'BIO_{i}' for i in range(1, 20)],
    "soilgrids": [
        'bdod_0-5cm', 'bdod_100-200cm', 'bdod_15-30cm', 'bdod_30-60cm', 'bdod_5-15cm', 'bdod_60-100cm',
        'cec_0-5cm', 'cec_100-200cm', 'cec_15-30cm', 'cec_30-60cm', 'cec_5-15cm', 'cec_60-100cm',
        'cfvo_0-5cm', 'cfvo_100-200cm', 'cfvo_15-30cm', 'cfvo_30-60cm', 'cfvo_5-15cm', 'cfvo_60-100cm',
        'clay_0-5cm', 'clay_100-200cm', 'clay_15-30cm', 'clay_30-60cm', 'clay_5-15cm', 'clay_60-100cm',
        'nitrogen_0-5cm', 'nitrogen_100-200cm', 'nitrogen_15-30cm', 'nitrogen_30-60cm', 'nitrogen_5-15cm', 'nitrogen_60-100cm',
        'ocd_0-5cm', 'ocd_100-200cm', 'ocd_15-30cm', 'ocd_30-60cm', 'ocd_5-15cm', 'ocd_60-100cm',
        'ocs_0-30cm',
        'phh2o_0-5cm', 'phh2o_100-200cm', 'phh2o_15-30cm', 'phh2o_30-60cm', 'phh2o_5-15cm', 'phh2o_60-100cm',
        'sand_0-5cm', 'sand_100-200cm', 'sand_15-30cm', 'sand_30-60cm', 'sand_5-15cm', 'sand_60-100cm',
        'silt_0-5cm', 'silt_100-200cm', 'silt_15-30cm', 'silt_30-60cm', 'silt_5-15cm', 'silt_60-100cm',
        'soc_0-5cm', 'soc_100-200cm', 'soc_15-30cm', 'soc_30-60cm', 'soc_5-15cm', 'soc_60-100cm',
    ],
    "stand_age": ['ForestAge_TC000', 'ForestAge_TC010', 'ForestAge_TC020', 'ForestAge_TC030', 
                  'TCloss_intensity', 'LastTimeTCloss_std'],
}

ORDERED_KEYS = [
    "deadwood_forest", "terrain", "canopy", "pixels_sentle", 
    "era5", "wc", "sg", "stand_age"
]

KEY_TO_LABEL = {
    "deadwood_forest": "deadwood_forest",
    "terrain": "terrain", 
    "canopy": "canopy",
    "pixels_sentle": "sentinel",
    "era5": "era5",
    "wc": "worldclim",
    "sg": "soilgrids",
    "stand_age": "stand_age",
}

REDUCE_DIMS = {
    "deadwood_forest": (2, 3, 4),
    "terrain": (2, 3),
    "canopy": (2, 3),
    "pixels_sentle": (2,),
    "era5": (2,),
    "wc": None,
    "sg": None,
    "stand_age": None,
}

####################
# DATA LOADING
####################
def load_baseline_stats() -> Dict:
    """Load mean statistics for baseline computation."""
    stats_path = PATHS.meta_data_dir / "mean_sentinel_deadwood_stats.json"
    with open(stats_path, "r") as f:
        return json.load(f)

def load_model(run_dir: Path, device: str) -> Tuple[torch.nn.Module, object]:
    """Load trained model from checkpoint."""
    cfg = load_config_from_json(run_dir / "config.json")
    model = MultimodalDeadwoodTransformer(cfg.embed_dim).to(device)
    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg

def prepare_dataset(cfg, df: pd.DataFrame) -> TrainingDataset:
    """Prepare dataset for IG analysis."""
    lmdb_path = PATHS.data_dir / "training_sets" / cfg.training_set / "sentle.lmdb"
    cube_ids = df['cube_id'].unique().tolist()
    
    all_cubes = preload_all_cubes_parallel(
        PATHS.cubes,
        stats_path=PATHS.meta_data_dir / "znorm_stats.json",
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
    
    return TrainingDataset(df, augment=False, **ds_kwargs)

####################
# DATABASE OPERATIONS
####################
def get_channel_columns() -> List[Tuple[str, str, str]]:
    """Generate list of (modality, channel_name, column_name) tuples."""
    all_channels = []
    for key in ORDERED_KEYS:
        label_key = KEY_TO_LABEL[key]
        for ch_name in CHANNEL_LABELS[label_key]:
            col_name = f"{label_key}__{ch_name}".replace(' ', '_').replace('-', '_')
            all_channels.append((label_key, ch_name, col_name))
    return all_channels

def compute_aggregated_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated attribution statistics from DataFrame."""
    all_channels = get_channel_columns()
    
    # Compute mean delta convergence
    mean_delta = df['delta'].mean() if 'delta' in df.columns else None
    
    agg_rows = []
    for label_key, ch_name, col_name in all_channels:
        if col_name in df.columns:
            values = df[col_name]
            agg_rows.append({
                'modality': label_key,
                'channel': ch_name,
                'mean': values.mean(),
                'std': values.std(),
                'n_samples': len(values),
                'mean_delta': mean_delta
            })
    
    result = pd.DataFrame(agg_rows)
    result['cv'] = result['std'] / result['mean']
    result = result.sort_values('mean', ascending=False).reset_index(drop=True)
    return result

####################
# INTEGRATED GRADIENTS
####################
def create_baselines(input_tuple: Tuple[torch.Tensor, ...], 
                     stats: Dict, device: str) -> Tuple[torch.Tensor, ...]:
    """Create baseline tensors for IG computation."""
    dw_means = torch.tensor(stats['deadwood_forest']).float().to(device)
    sentinel_means = torch.tensor(stats['sentinel']).float().to(device)
    
    baselines_list = []
    for i, t in enumerate(input_tuple):
        if i == 0:  # deadwood_forest [B, 4, 33, 33, 3]
            b = dw_means.view(1, 4, 1, 1, 1).expand_as(t)
        elif i == 3:  # sentinel [B, 12, 156]
            b = sentinel_means.view(1, 12, 1).expand_as(t)
        else:
            b = torch.zeros_like(t)
        baselines_list.append(b)
    
    return tuple(baselines_list)

def reduce_attributions(attributions: Tuple[torch.Tensor, ...]) -> Dict[int, Dict[str, float]]:
    """Reduce spatial/temporal dims and extract per-sample, per-channel attributions."""
    batch_size = attributions[0].shape[0]
    batch_data = {s: {} for s in range(batch_size)}
    
    for i, key in enumerate(ORDERED_KEYS):
        label_key = KEY_TO_LABEL[key]
        reduce_dims = REDUCE_DIMS[key]
        attr = attributions[i].abs()
        
        if reduce_dims is not None:
            attr = attr.sum(dim=reduce_dims)
        
        attr_np = attr.cpu().detach().numpy()
        
        for ch_idx, ch_name in enumerate(CHANNEL_LABELS[label_key]):
            col_name = f"{label_key}__{ch_name}".replace(' ', '_').replace('-', '_')
            for s in range(batch_size):
                batch_data[s][col_name] = float(attr_np[s, ch_idx])
    
    return batch_data

def run_ig_analysis(
    model: torch.nn.Module,
    dataset: TrainingDataset,
    stats: Dict,
    device: str = "cuda:0",
    batch_size: int = 10,
    n_steps: int = 50,
) -> pd.DataFrame:
    """
    Run Integrated Gradients analysis on dataset.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    dataset : TrainingDataset
        Dataset of samples to analyze.
    stats : Dict
        Baseline statistics for IG.
    device : str
        CUDA device string.
    batch_size : int
        Samples per IG batch.
    n_steps : int
        Integration steps for IG.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with pixel_key, delta, and all attribution columns.
    """
    all_channels = get_channel_columns()
    ig = IntegratedGradients(model)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    all_rows = []
    
    pbar = tqdm(loader, desc="IG Analysis", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        pixel_keys = list(batch['pixel_key'])  # pixel_key is a tuple of strings
        
        # Prepare inputs
        inputs = {key: batch[key].to(device) for key in FEATURE_KEYS}
        input_list = [
            torch.nan_to_num(inputs[key], nan=0.0).clone().requires_grad_(True)
            for key in ORDERED_KEYS
        ]
        input_tuple = tuple(input_list)
        baselines = create_baselines(input_tuple, stats, device)
        
        # Run IG
        attributions, delta = ig.attribute(
            inputs=input_tuple,
            baselines=baselines,
            target=0,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        # Extract delta values per sample
        deltas = delta.cpu().detach().numpy().flatten().tolist()
        
        # Process attributions
        batch_data = reduce_attributions(attributions)
        
        # Build rows for DataFrame
        for s, pk in enumerate(pixel_keys):
            row = {'pixel_key': pk, 'delta': deltas[s]}
            row.update(batch_data[s])
            all_rows.append(row)
        
        pbar.set_postfix(samples=f"{len(all_rows):,}")
        
        # Cleanup
        del attributions, input_tuple, baselines
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_rows)

def collapse_modalities(df, modalities_to_collapse=['worldclim', 'soilgrids']):
    """Collapse specified modalities into single rows showing mean across channels."""
    # Keep rows for non-collapsed modalities
    df_keep = df[~df['modality'].isin(modalities_to_collapse)].copy()
    
    # Aggregate collapsed modalities
    collapsed_rows = []
    for mod in modalities_to_collapse:
        mod_df = df[df['modality'] == mod]
        if len(mod_df) == 0:
            continue
        n_channels = len(mod_df)
        collapsed_rows.append({
            'modality': mod,
            'channel': f'MEAN ({n_channels} vars)',
            'mean': mod_df['mean'].mean(),
            'std': mod_df['mean'].std(),  # std of the channel means
            'n_samples': mod_df['n_samples'].iloc[0],
            'cv': mod_df['mean'].std() / mod_df['mean'].mean()
        })
    
    df_collapsed = pd.concat([df_keep, pd.DataFrame(collapsed_rows)], ignore_index=True)
    df_collapsed = df_collapsed.sort_values('mean', ascending=False).reset_index(drop=True)
    return df_collapsed

####################
# MAIN
####################
def main(args):
    """Main entry point for IG analysis."""
    print("=" * 60)
    print("Integrated Gradients Analysis")
    print(f"Run: {args.run} | Threshold: {args.threshold}")
    print(f"Device: cuda:{args.device} | Steps: {args.n_steps}")
    print("=" * 60)
    
    # Setup
    run_dir = PATHS.training_runs / args.run
    device = f"cuda:{args.device}"
    
    # Load model and config
    print("\n[1/4] Loading model...")
    model, cfg = load_model(run_dir, device)
    stats = load_baseline_stats()
    
    # Load and filter data
    print("\n[2/4] Preparing dataset...")
    df = pd.read_parquet(PATHS.data_dir / "training_sets" / cfg.training_set / "eval_baseline.parquet")
    df = df[df['target_d'] > args.threshold]
    print(f"Selected {len(df):,} samples with target_d > {args.threshold}")
    
    dataset = prepare_dataset(cfg, df)
    
    # Run IG
    print("\n[3/4] Running Integrated Gradients...")
    attribution_df = run_ig_analysis(
        model=model,
        dataset=dataset,
        stats=stats,
        device=device,
        batch_size=10,
        n_steps=args.n_steps,
    )
    
    # Save per-pixel parquet (includes delta)
    output_path = run_dir / "integrated_gradients" / f"ig_attributions_{args.threshold}.parquet"
    attribution_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(attribution_df):,} samples to {output_path}")
    
    # Aggregate
    print("\n[4/4] Computing aggregated statistics...")
    summary_df = compute_aggregated_stats(attribution_df)
    
    summary_path = run_dir / "integrated_gradients" / f"ig_attributions_{args.threshold}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES BY MEAN ATTRIBUTION")
    print("=" * 60)
    print(summary_df.head(20).to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Integrated Gradients analysis")
    parser.add_argument("--run", type=str, default="run_10", help="Training run name")
    parser.add_argument("--threshold", type=float, default=0.4, help="Min target_d threshold")
    parser.add_argument("--n_steps", type=int, default=50, help="IG integration steps")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number (e.g., 0 -> cuda:0)")
    
    args = parser.parse_args()
    main(args)



