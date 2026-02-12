import sys
from pathlib import Path
from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import json
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

from training.train_utils import FEATURE_KEYS, load_config_from_json
from training.setup_training import setup_cube, setup_prediction
from utils.plots import plot_prediction_vs_target

with open(PATHS.meta_data_dir / "mean_sentinel_deadwood_stats.json", "r") as f:
    stats = json.load(f)

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

# Ordered keys matching model's forward() input order
ORDERED_KEYS = [
    "deadwood_forest", "terrain", "canopy", "pixels_sentle", 
    "era5", "wc", "sg", "stand_age"
]

# Mapping from ordered_keys to channel_labels keys
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

# Reduction dims for each modality (to collapse spatial/temporal dims)
REDUCE_DIMS = {
    "deadwood_forest": (2, 3, 4),  # [B, C, H, W, T] -> sum over H, W, T
    "terrain": (2, 3),              # [B, C, H, W] -> sum over H, W
    "canopy": (2, 3),               # [B, C, H, W] -> sum over H, W
    "pixels_sentle": (2,),          # [B, C, T] -> sum over T
    "era5": (2,),                   # [B, C, T] -> sum over T
    "wc": None,                     # [B, C] -> no reduction needed
    "sg": None,                     # [B, C] -> no reduction needed
    "stand_age": None,              # [B, C] -> no reduction needed
}

def prepare_ig_inputs(
    cfg,
    df: pd.DataFrame,
    shared_tensors: dict,
    stats: dict,
    device: str = "cuda:0",
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], torch.nn.Module]:
    """
    Prepare inputs and baselines for Integrated Gradients.
    
    Parameters
    ----------
    cfg : ExperimentConfig
        Configuration object for the model.
    df : pd.DataFrame
        Metadata DataFrame for pixels to analyze.
    shared_tensors : dict
        Shared tensors from setup_cube.
    stats : dict
        Dictionary with mean values for deadwood_forest and sentinel.
    device : str
        CUDA device string.
        
    Returns
    -------
    input_tuple : Tuple[torch.Tensor, ...]
        Input tensors ready for IG (with requires_grad=True).
    baselines : Tuple[torch.Tensor, ...]
        Baseline tensors for IG.
    model : torch.nn.Module
        Loaded model in eval mode.
    """
    # Setup prediction loader and model
    pred_loader, model = setup_prediction(
        cfg=cfg, 
        data_dir=PATHS.data_dir, 
        df=df, 
        shared_tensors=shared_tensors,
        device=device
    )
    
    # Get batch
    batch = next(iter(pred_loader))
    inputs = {key: batch[key].to(device) for key in FEATURE_KEYS}
    
    # Create input tuple in correct order
    input_list = [
        torch.nan_to_num(inputs[key], nan=0.0).clone().to(device) 
        for key in ORDERED_KEYS
    ]
    input_tuple = tuple(t.requires_grad_() for t in input_list)
    
    # Create baselines
    dw_means = torch.tensor(stats['deadwood_forest']).float().to(device)
    sentinel_means = torch.tensor(stats['sentinel']).float().to(device)
    
    baselines_list = []
    for i, t in enumerate(input_tuple):
        if i == 0:  # deadwood_forest [B, 4, 33, 33, 3]
            b = dw_means.view(1, 4, 1, 1, 1).expand_as(t)
        elif i == 3:  # sentinel [B, 12, 156]
            b = sentinel_means.view(1, 12, 1).expand_as(t)
        else:
            # Z-score normalized features: 0.0 is the global mean
            b = torch.zeros_like(t)
        baselines_list.append(b)
    
    baselines = tuple(baselines_list)
    
    return input_tuple, baselines, model

def run_integrated_gradients(
    model: torch.nn.Module,
    input_tuple: Tuple[torch.Tensor, ...],
    baselines: Tuple[torch.Tensor, ...],
    n_steps: int = 50,
    target: int = 0,
    batch_size: int = 10,
) -> Tuple[Tuple[torch.Tensor, ...], pd.DataFrame]:
    """
    Run Integrated Gradients and produce per-channel attribution DataFrame.
    
    Processes samples in batches to avoid GPU OOM errors.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to explain.
    input_tuple : Tuple[torch.Tensor, ...]
        Input tensors (from prepare_ig_inputs).
    baselines : Tuple[torch.Tensor, ...]
        Baseline tensors (from prepare_ig_inputs).
    n_steps : int
        Number of integration steps for IG.
    target : int
        Target output index (0 for single-output models).
    batch_size : int
        Number of samples to process at once. Lower if OOM. Default: 10.
        
    Returns
    -------
    attributions : Tuple[torch.Tensor, ...]
        Raw attribution tensors for each modality (on CPU to save GPU memory).
    attribution_df : pd.DataFrame
        Per-channel attribution table with columns: modality, channel, mean, std, cv.
    """
    n_total = input_tuple[0].shape[0]
    n_batches = (n_total + batch_size - 1) // batch_size
    
    print(f"Running IG on {n_total} samples in {n_batches} batches (batch_size={batch_size})")
    
    ig = IntegratedGradients(model)
    
    # Accumulate attributions across batches (on CPU)
    all_attributions = [[] for _ in range(len(input_tuple))]
    all_deltas = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_total)
        
        # Slice inputs and baselines for this batch
        batch_inputs = tuple(t[start_idx:end_idx] for t in input_tuple)
        batch_baselines = tuple(b[start_idx:end_idx] for b in baselines)
        
        # Run IG on batch
        batch_attr, batch_delta = ig.attribute(
            inputs=batch_inputs,
            baselines=batch_baselines,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        # Move to CPU immediately to free GPU memory
        for i, attr in enumerate(batch_attr):
            all_attributions[i].append(attr.cpu().detach())
        all_deltas.append(batch_delta.cpu().detach())
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"  Batch {batch_idx + 1}/{n_batches} done (samples {start_idx}-{end_idx-1})")
    
    # Concatenate all batches
    attributions = tuple(torch.cat(attr_list, dim=0) for attr_list in all_attributions)
    delta = torch.cat(all_deltas, dim=0)
    
    print(f"Convergence Delta: {delta.abs().mean().item():.6f}")
    
    n_samples = attributions[0].shape[0]
    
    # Helper function
    def get_channel_stats(attr_tensor, reduce_dims):
        if reduce_dims is not None:
            per_sample_per_channel = attr_tensor.abs().sum(dim=reduce_dims).numpy()
        else:
            per_sample_per_channel = attr_tensor.abs().numpy()
        mean_attr = per_sample_per_channel.mean(axis=0)
        std_attr = per_sample_per_channel.std(axis=0)
        return mean_attr, std_attr
    
    # Collect all attributions
    attribution_rows = []
    
    for i, key in enumerate(ORDERED_KEYS):
        label_key = KEY_TO_LABEL[key]
        reduce_dims = REDUCE_DIMS[key]
        mean_attr, std_attr = get_channel_stats(attributions[i], reduce_dims)
        
        for ch, m, s in zip(CHANNEL_LABELS[label_key], mean_attr, std_attr):
            attribution_rows.append({
                'modality': label_key, 
                'channel': ch, 
                'mean': m, 
                'std': s
            })
    
    # Create DataFrame
    attribution_df = pd.DataFrame(attribution_rows)
    attribution_df['cv'] = attribution_df['std'] / attribution_df['mean']
    attribution_df = attribution_df.sort_values('mean', ascending=False).reset_index(drop=True)
    
    print(f"Per-Channel Attribution DataFrame: {len(attribution_df)} channels from {n_samples} samples")
    
    return attributions, attribution_df

def aggregate_attribution_table(
    attribution_df: pd.DataFrame,
    attributions: Tuple[torch.Tensor, ...],
    aggregate_modalities: List[str] = ['worldclim', 'soilgrids', 'stand_age'],
) -> pd.DataFrame:
    """
    Create a collapsed attribution table by aggregating specified modalities.
    
    Uses mean-per-channel (not sum) for fair comparison across modalities
    with different numbers of channels.
    
    Parameters
    ----------
    attribution_df : pd.DataFrame
        Full per-channel attribution DataFrame (from run_integrated_gradients).
    attributions : Tuple[torch.Tensor, ...]
        Raw attribution tensors (from run_integrated_gradients).
    aggregate_modalities : List[str]
        List of modality names to collapse. Default: ['worldclim', 'soilgrids', 'stand_age'].
        
    Returns
    -------
    attribution_df_collapsed : pd.DataFrame
        Collapsed attribution table with aggregated rows for specified modalities.
    """
    # Keep detailed rows for non-aggregated modalities
    collapsed_rows = attribution_df[~attribution_df['modality'].isin(aggregate_modalities)].to_dict('records')
    
    # Map modality names to attribution indices
    modality_to_idx = {KEY_TO_LABEL[key]: i for i, key in enumerate(ORDERED_KEYS)}
    
    # Add aggregated rows for specified modalities
    for modality in aggregate_modalities:
        if modality not in modality_to_idx:
            print(f"Warning: modality '{modality}' not found, skipping.")
            continue
            
        idx = modality_to_idx[modality]
        n_channels = len(CHANNEL_LABELS[modality])
        
        # Mean across channels per sample, then mean/std across samples
        # Attributions are already on CPU from run_integrated_gradients
        per_sample_mean = attributions[idx].abs().mean(dim=1).numpy()
        
        collapsed_rows.append({
            'modality': modality, 
            'channel': f'MEAN ({n_channels} vars)', 
            'mean': per_sample_mean.mean(), 
            'std': per_sample_mean.std()
        })
    
    # Create collapsed DataFrame
    attribution_df_collapsed = pd.DataFrame(collapsed_rows)
    attribution_df_collapsed['cv'] = attribution_df_collapsed['std'] / attribution_df_collapsed['mean']
    attribution_df_collapsed = attribution_df_collapsed.sort_values('mean', ascending=False).reset_index(drop=True)
    
    print(f"Collapsed Attribution DataFrame: {len(attribution_df_collapsed)} rows")
    
    return attribution_df_collapsed

RUN = "run_10"
DEVICE = "cuda:0"
CUBE_ID = 51


run_dir = PATHS.training_runs / RUN
cfg = load_config_from_json(run_dir / "config.json")
cfg.device = DEVICE
gdf = gpd.read_file(PATHS.meta_data_dir / "training_cube_set.gpkg")


# Load a cube and select a year
pred_path = PATHS.predictions / RUN / f"{CUBE_ID}_{RUN}_prediction.zarr"
ds = xr.open_zarr(pred_path)
plot_prediction_vs_target(ds, CUBE_ID, vmax=0.6, save=False)
ds = ds.compute()
ds_year = ds.sel(time_year=ds.time_year[3])
flat_pixels = ds_year.prediction.stack(pixel=('y', 'x'))
top_pixels = flat_pixels.dropna('pixel').sortby(flat_pixels, ascending=False).isel(pixel=slice(0, 100))
num_points = len(top_pixels)
import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, num_points))
plt.figure(figsize=(15, 15))
ax = plt.gca()
ds_year.prediction.plot(
    ax=ax,
    cmap='inferno_r', 
    vmin=0, 
    vmax=0.6,
    cbar_kwargs={'orientation': 'horizontal', 'label': 'Prediction Value', 'pad': 0.05, 'shrink': 0.5}
)
i = 4
point = top_pixels.isel(pixel=i)
val = float(point.values)
plt.scatter(
    point.x, point.y, 
    color="green", marker='x', s=150, linewidths=3,
    label=f"Idx {i}: {val:.4f}"
)
ax.set_aspect('equal', adjustable='box') 
plt.title(f"Top {num_points} Pixels (0-indexed) | Cube {CUBE_ID} | Year {int(ds_year.time_year.dt.year)}")
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Pixel Index & Value")

plt.tight_layout()
plt.show()

top_pixels = top_pixels.to_dataframe(name='prediction').reset_index()
top_pixels['idx'] = top_pixels.index
top_pixels = top_pixels[['idx','x', 'y', 'time_year', 'prediction']]
df_top = top_pixels.rename(columns={'x': 'coords_x', 'y': 'coords_y', 'idx': 'pixel_key', 'time_year': 'year'})
df_top['year'] = df_top['year'].dt.year
df_top['year'] = df_top['year'].astype(int)
df_top['year'] = df_top['year']-1  # Align years to match current year


df = pd.read_parquet(PATHS.data_dir / "training_sets" / cfg.training_set / "eval_baseline.parquet")
df = df[df['target_d']>0.4]

lmdb_path = PATHS.data_dir / "training_sets" / cfg.training_set / "sentle.lmdb"
cube_ids = df['cube_id'].unique().tolist()

from training.setup_training import preload_all_cubes_parallel, TrainingDataset
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

ds = TrainingDataset(df, augment=False, **ds_kwargs)

len(ds)

# Get full metadata for the cube
df, shared_tensors = setup_cube(
            cube_id = CUBE_ID, 
            cubes_dir = PATHS.cubes, 
            meta_dir =  PATHS.meta_data_dir)


# Perform the Merge
top_meta = pd.merge(
    df_top, 
    df.drop(columns=['pixel_key']),
    on=['coords_x', 'coords_y', 'year'], 
    how='inner'
)

####################
# Run Integrated Gradients for Pixel i = 4
####################

# Extract single pixel (i = 4) from top_meta
pixel_idx = 20
df_single_pixel = top_meta.iloc[[pixel_idx]].copy()

print(f"\n{'='*70}")
print(f"Analyzing Pixel {pixel_idx}:")
print(f"  Coordinates: ({df_single_pixel['coords_x'].values[0]}, {df_single_pixel['coords_y'].values[0]})")
print(f"  Year: {df_single_pixel['year'].values[0]}")
print(f"  Prediction: {df_single_pixel['prediction'].values[0]:.4f}")
print(f"{'='*70}\n")

# STEP 1: Prepare inputs, baselines, and model for single pixel
input_tuple_single, baselines_single, model_single = prepare_ig_inputs(
    cfg=cfg,
    df=df_single_pixel,
    shared_tensors=shared_tensors,
    stats=stats,
    device=DEVICE
)

# STEP 2: Run Integrated Gradients on the single pixel
attributions_single, attribution_df_single = run_integrated_gradients(
    model=model_single,
    input_tuple=input_tuple_single,
    baselines=baselines_single,
    n_steps=100,
    target=0,
    batch_size=1,
)


####################
# Comprehensive Pixel Analysis: 2x3 Subplot
####################

fig, axes = plt.subplots(2, 3, figsize=(22, 14), gridspec_kw={'width_ratios': [1, 1, 1.08]})

# Get pixel coordinates from df_single_pixel
px_x = float(df_single_pixel['coords_x'].values[0])
px_y = float(df_single_pixel['coords_y'].values[0])

# -------- COLUMN 1: Cube-wide view and Feature Importance --------

# [0, 0] Cube-wide Deadwood Increase with pixel marked
ax = axes[0, 0]
extent = [ds_year.x.values[0], ds_year.x.values[-1], ds_year.y.values[-1], ds_year.y.values[0]]
im = ax.imshow(ds_year.prediction.values, cmap='inferno_r', vmin=0, vmax=0.6, extent=extent)
ax.scatter(px_x, px_y, color='lime', marker='x', s=200, linewidths=3, label=f'Pixel of interest', zorder=5)
ax.set_title(f'Cube {CUBE_ID} - Deadwood Inc Prediction 2024', fontsize=20)
ax.set_aspect('equal', adjustable='box')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.legend(fontsize=18, loc='upper right')

# [1, 0] Top 5 Features Barplot
ax = axes[1, 0]
top5 = attribution_df_single.head(5).copy()
colors_bar = plt.cm.inferno(np.linspace(0.2, 0.8, len(top5)))
bars = ax.barh(range(len(top5)), top5['mean'].values, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top5)))
ax.set_yticklabels([f"{row['channel']}" for _, row in top5.iterrows()], fontsize=18)
ax.set_xlabel('Mean Attribution', fontsize=18)
ax.set_title('Top 5 Most Important Features', fontsize=20)
ax.invert_yaxis()
ax.tick_params(axis='x', labelsize=16)
ax.set_xlim(0.1, 0.3)
from matplotlib.ticker import FormatStrFormatter
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
for i, (idx, row) in enumerate(top5.iterrows()):
    ax.text(row['mean'], i, f" {row['mean']:.3f}", va='center', fontsize=18)

# -------- COLUMN 2: Deadwood Increase (channel 2) --------

# [0, 1] Deadwood Inc input patch
ax = axes[0, 1]
dw_inc_patch = input_tuple_single[0][0, 2, :, :, -1].cpu().detach().numpy()
im = ax.imshow(dw_inc_patch, cmap='inferno_r', vmin=0, vmax=1)
ax.scatter(16, 16, color='lime', marker='x', s=200, linewidths=3, label='Center')
ax.set_title('Deadwood Increase\nInput Patch 2023', fontsize=20)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.legend(fontsize=18)

# [1, 1] Deadwood Inc attribution map
ax = axes[1, 1]
dw_inc_attr = attributions_single[0][0, 2, :, :, -1].cpu().detach().numpy()
v_limit = np.abs(dw_inc_attr).max() * 0.9
im = ax.imshow(dw_inc_attr, cmap='seismic', vmin=-v_limit, vmax=v_limit)
ax.scatter(16, 16, color='lime', marker='x', s=200, linewidths=3, label='Center')
ax.set_title('Deadwood Increase\nAttribution Map', fontsize=20)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.legend(fontsize=18)

# -------- COLUMN 3: Deadwood Cover (channel 0) --------

# [0, 2] Deadwood Cover input patch
ax = axes[0, 2]
dw_cover_patch = input_tuple_single[0][0, 0, :, :, -1].cpu().detach().numpy()
im = ax.imshow(dw_cover_patch, cmap='inferno_r', vmin=0, vmax=0.6)
ax.scatter(16, 16, color='lime', marker='x', s=200, linewidths=3, label='Center')
ax.set_title('Deadwood Cover\nInput Patch 2023', fontsize=20)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.legend(fontsize=18)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Value', fontsize=18)

# [1, 2] Deadwood Cover attribution map
ax = axes[1, 2]
dw_cover_attr = attributions_single[0][0, 0, :, :, -1].cpu().detach().numpy()
im = ax.imshow(dw_cover_attr, cmap='seismic', vmin=-0.03, vmax=0.03)
ax.scatter(16, 16, color='lime', marker='x', s=200, linewidths=3, label='Center')
ax.set_title('Deadwood Cover\nAttribution Map', fontsize=20)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.legend(fontsize=18)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Influence', fontsize=18)

from matplotlib.patches import Rectangle

# Coordinates in figure fraction: (0,0)=bottom-left, (1,1)=top-right
rect_width = 0.08
rect_height = 0.13

r1_x = 0.36   # left
r1_y = 0.76   # bottom
rect1 = Rectangle(
    (r1_x, r1_y),
    rect_width,
    rect_height,
    linewidth=3,
    edgecolor='lime',
    facecolor='none',
    transform=fig.transFigure,  # << important: figure coords
    zorder=10
)

r2_x = 0.36   # left
r2_y = 0.27   # bottom
rect2 = Rectangle(
    (r2_x, r2_y),
    rect_width,
    rect_height,
    linewidth=3,
    edgecolor='lime',
    facecolor='none',
    transform=fig.transFigure,  # << important: figure coords
    zorder=10
)

r3_x = 0.63   # left
r3_y = 0.76   # bottom
rect3 = Rectangle(
    (r3_x, r3_y),
    rect_width,
    rect_height,
    linewidth=3,
    edgecolor='red',
    facecolor='none',
    transform=fig.transFigure,  # << important: figure coords
    zorder=10
)

r4_x = 0.63   # left
r4_y = 0.27   # bottom
rect4 = Rectangle(
    (r4_x, r4_y),
    rect_width,
    rect_height,
    linewidth=3,
    edgecolor='red',
    facecolor='none',
    transform=fig.transFigure,  # << important: figure coords
    zorder=10
)
fig.add_artist(rect1)
fig.add_artist(rect2)
fig.add_artist(rect3)
fig.add_artist(rect4)

plt.tight_layout()
plt.savefig("results/pixel_ig_analysis.pdf", dpi=300)
plt.show()
