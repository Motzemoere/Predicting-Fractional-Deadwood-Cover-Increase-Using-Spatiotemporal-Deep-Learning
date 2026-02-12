"""
Plotting utilities for visualization and analysis.
Creates spatial maps, predictions, holdouts, and model evaluation figures.
"""

from click import style
from matplotlib import colors
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from pathlib import Path
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import folium
import folium.plugins

FIGURE_CLASSES = {
    "map_4_3": dict(
        figsize=(20, 15),
        f1=32,
        f2=30,
        f3=28,
    ),
    "grid_2_1": dict(
        figsize=(20, 10),
        f1=28,
        f2=26,
        f3=20,
    ),
    "square_1_1": dict(
        figsize=(20, 20),
        f1=35,
        f2=33,
        f3=29,
    ),
    "4_1": dict(
        figsize=(20, 5),
        f1=28,
        f2=26,
        f3=20,
    ),
}

def plot_inspection_sites(gdf, selected_locations: list, outdir=None, save=False):
    
    # --- Correct style for 0.6 textwidth ---
    figsize = (13.33, 10)   # 4:3 scaled from 20x15 by 2/3
    f1, f2, f3 = 21, 20, 19

    hold_outs = gdf[gdf["is_holdout"] == True]
    # remove selected locations from holdouts for special annotation
    hold_outs = hold_outs[~hold_outs["cube_id"].isin(selected_locations)]
    
    selected_locations_gdf = gdf[gdf["cube_id"].isin(selected_locations)]

    # --- Load Germany outline ---
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    world = gpd.read_file(url)
    germany = world[world["NAME"] == "Germany"][["geometry"]].copy()
    germany = germany.to_crs(gdf.crs)

    fig, ax = plt.subplots(figsize=figsize)

    # --- Background map ---
    germany.plot(ax=ax, color="white", edgecolor="#6c757d", zorder=1)

    # --- All training points ---
    gdf.plot(
        ax=ax,
        color="lightgrey",
        edgecolor="lightgrey",
        linewidth=1,
        markersize=5,
        zorder=2
    )

    # --- Holdouts ---

    hold_outs.plot(
        ax=ax,
        color="black",
        edgecolor="black",
        linewidth=1,
        markersize=15,
        zorder=3
    )

    selected_locations_gdf.plot(
        ax=ax,
        color="red",
        edgecolor="red",
        linewidth=1,
        markersize=15,
        zorder=3
    )

    # --- Annotate cube IDs ---
    special_ids = {212, 162, 221, 262}

    for _, row in hold_outs.iterrows():
        c = row.geometry.centroid
        xt, yt = 5, 5
        if row["cube_id"] in special_ids:
            yt = -10
        ax.annotate(
            text=str(row["cube_id"]),
            xy=(c.x, c.y),
            xytext=(xt, yt),
            textcoords="offset points",
            fontsize=f3 - 2,
            color="black",
            zorder=4
        )

    for _, row in selected_locations_gdf.iterrows():
        c = row.geometry.centroid
        xt, yt = 5, 5
        if row["cube_id"] in special_ids:
            yt = -10
        ax.annotate(
            text=str(row["cube_id"]),
            xy=(c.x, c.y),
            xytext=(xt, yt),
            textcoords="offset points",
            fontsize=f3 - 2,
            color="red",
            zorder=5
        )

    # --- Axis formatting ---
    def div_1000(x, pos):
        return f"{x/1000:g}"

    formatter = FuncFormatter(div_1000)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=f3)
    ax.set_ylabel("Northing [m] 1e3", fontsize=f2, labelpad=15)
    ax.set_xlabel("Easting [m] 1e3", fontsize=f2, labelpad=15)

    # --- Legend ---
    legend_elements = [
        Patch(facecolor="red", label="Presented Inspection Sites"),
        Patch(facecolor="black", label="Inspection Sites"),
        Patch(facecolor="lightgrey", label="Training Sites"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=f2, frameon=True)

    plt.tight_layout()

    if save:
        if outdir is not None:
            outdir_path = Path(outdir)
            out_path = outdir_path / f"inspection_sites.pdf"
        else:
            out_path = "inspection_sites.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_inspection_sites_folds(gdf, grid_gdf=None, outdir=None, save=False):
    """
    Plots inspection sites and spatial folds.
    If 'raster_file' is provided, it is plotted as the basemap on the right axis.
    """
    style = FIGURE_CLASSES["map_4_3"]
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]
    cmap = plt.get_cmap("viridis").copy()
    
    folds = gdf[gdf['fold'] > 0]
    inspection_sites = gdf[gdf["is_holdout"] == True]
    
    # Load World/Germany Data
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    world = gpd.read_file(url)
    germany = world[world["NAME"] == "Germany"][["geometry"]].copy()
    germany = germany.to_crs(gdf.crs)

    fig, axes = plt.subplots(1, 2, figsize=style["figsize"], sharex=True, sharey=True)

    # --- LEFT PLOT: Holdouts ---
    # Standard white background
    germany.plot(ax=axes[0], color="white", edgecolor="#6c757d", zorder=1)

    inspection_sites.plot(ax=axes[0],
                   color="red",
                   edgecolor="red",
                   linewidth=1,
                   markersize=15,  # Adjusted for point visibility
                   zorder=3)

    gdf.plot(ax=axes[0],
             color="lightgrey",
             edgecolor="lightgrey",
             linewidth=1,
             markersize=5,
             zorder=2)

    # --- RIGHT PLOT: Spatial Folds ---
    
    # 1. Plot the Background (Raster or White)
    germany.plot(ax=axes[1], color="white", edgecolor="#6c757d", zorder=1)

    if grid_gdf is not None:
        # Plot the 25km blocks themselves as semi-transparent squares
        grid_gdf.plot(ax=axes[1],
                      column='fold',
                      cmap=None,
                      alpha=0.08,       # Light transparency
                      edgecolor=None,
                      linewidth=0.5,
                      zorder=2)

    # Plot the actual data points on top of the grid
    folds = gdf[gdf['fold'] > 0]
    folds.plot(ax=axes[1],
               column='fold',
               categorical=True,
               cmap=cmap,
               markersize=5,
               zorder=3)

    # --- Formatting ---
    def div_1000(x, pos):
        return f'{x/1000:g}'
    formatter = FuncFormatter(div_1000)

    # Axis 0 (Left)
    axes[0].tick_params(axis="y", which="both", left=True, labelleft=False, labelsize=f3)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].tick_params(axis="both", which="both", labelsize=f3)

    # Axis 1 (Right)
    axes[1].yaxis.set_ticks_position("both")
    axes[1].tick_params(axis="y", which="both", left=True, right=True, 
                        labelleft=False, labelright=True, labelsize=f3)
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("Northing [m] 1e3", fontsize=f2, labelpad=15)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].tick_params(axis="both", which="both", labelsize=f3)

    fig.text(0.47, 0.05, "Easting [m] 1e3", ha='center', va='top', fontsize=f2)

    # Legends
    legend_elements = [Patch(facecolor="red", label="Inspection Sites"),
                       Patch(facecolor="lightgrey", label="Training")]

    unique_folds = sorted(folds['fold'].unique())
    colors = [cmap(i / (len(unique_folds) - 1)) if len(unique_folds) > 1 else cmap(0) 
              for i in range(len(unique_folds))]
    
    legend_elements_2 = [Patch(facecolor=colors[i], edgecolor='k', label=f"Fold {int(fold)}")
                         for i, fold in enumerate(unique_folds)]

    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=f2, frameon=True)
    axes[1].legend(handles=legend_elements_2, loc='upper left', title='Spatial Folds',
                   fontsize=f2, title_fontsize=f2, frameon=True)

    plt.tight_layout()
    if save:
        if outdir is not None:
            outdir_path = Path(outdir)
            out_path = outdir_path / f"inspection_sites_and_folds.pdf"
        else:
            out_path = "inspection_sites_and_folds.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_cube_selection(gdf1, gdf2, outdir=None, save=False):
    """
    Plots inspection sites and spatial folds.
    If 'raster_file' is provided, it is plotted as the basemap on the right axis.
    """
    style = FIGURE_CLASSES["map_4_3"]
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]

    cmap = plt.colormaps["BrBG"]
    c2 = cmap(0.3)
    c3 = cmap(0.0)
    c1 = cmap(0.5)
    
    low = gdf2[gdf2["mortality_class"] == "low"]
    high = gdf2[gdf2["mortality_class"] == "high"]
    
    
    # Load World/Germany Data
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    world = gpd.read_file(url)
    germany = world[world["NAME"] == "Germany"][["geometry"]].copy()
    germany = germany.to_crs(gdf1.crs)

    fig, axes = plt.subplots(1, 2, figsize=style["figsize"], sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, right=0.92, wspace=0.02, bottom=0.22, top=0.95)
    
    # Standard white background
    germany.plot(ax=axes[0], color="white", edgecolor="#6c757d", zorder=1)

    gdf1.plot(ax=axes[0],
             column='pct_high',
             cmap='Reds',
             linewidth=1,
             markersize=5,
             zorder=2)
    vmin, vmax = gdf1['pct_high'].min(), gdf1['pct_high'].max()
    # sc = gdf1.plot(ax=axes[0],
    #                column='pct_high',
    #                cmap='Reds',
    #                linewidth=1,
    #                markersize=5,
    #                zorder=2)

    # --- RIGHT PLOT: Spatial Folds ---
    
    # 1. Plot the Background (Raster or White)
    germany.plot(ax=axes[1], color="white", edgecolor="#6c757d", zorder=1)

    low.plot(ax=axes[1],
                   color=c2,
                   edgecolor=c2,
                   linewidth=1,
                   markersize=15,  # Adjusted for point visibility
                   zorder=2)
    

    high.plot(ax=axes[1],
                   color=c3,
                   edgecolor=c3,
                   linewidth=1,
                   markersize=15,  # Adjusted for point visibility
                   zorder=3)

    
    # --- Formatting ---
    def div_1000(x, pos):
        return f'{x/1000:g}'
    formatter = FuncFormatter(div_1000)

    # Axis 0 (Left)
    axes[0].tick_params(axis="y", which="both", left=True, labelleft=False, labelsize=f3)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].tick_params(axis="both", which="both", labelsize=f3)

    # Axis 1 (Right)
    axes[1].yaxis.set_ticks_position("both")
    axes[1].tick_params(axis="y", which="both", left=True, right=True, 
                        labelleft=False, labelright=True, labelsize=f3)
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("Northing [m] 1e3", fontsize=f2, labelpad=15)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].tick_params(axis="both", which="both", labelsize=f3)

    # --- Shared Label ---
    # Move the Easting label up slightly to make room for legends
    fig.text(0.46, 0.17, "Easting [m] 1e3", ha='center', va='top', fontsize=f2)

    import matplotlib.cm as cm
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap='Reds')
    sm.set_array([]) # Dummy array for the colorbar


    # --- 1. Horizontal Colorbar (Beneath Left Plot) ---
    # [left, bottom, width, height]
    # Positioned centered under the left axis (axes[0])
    cax = fig.add_axes([0.14, 0.11, 0.25, 0.015]) 
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=f3)
    cb.set_label('%-High Mortality Pixels', fontsize=f3)

    # --- 2. Categorical Legend (Beneath Right Plot) ---
    legend_elements = [Patch(facecolor=c2, label=r"Low ($\leq 20\%$)"),
                       Patch(facecolor=c3, label=r"High ($> 20\%$)")]

    # We use bbox_to_anchor to pin it relative to axes[1]
    axes[1].legend(handles=legend_elements, 
                   loc='upper center', 
                   bbox_to_anchor=(0.5, -0.11), # Moves it below the axis
                   ncol=2,                       # Horizontal layout
                   fontsize=f3, 
                   frameon=True,
                   title='Mortality Class', 
                   title_fontsize=f3)
    
    if save:
        if outdir is not None:
            outdir_path = Path(outdir)
            out_path = outdir_path / f"cube_selection.pdf"
        else:
            out_path = "cube_selection.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_training_timeline(df, save=False, outdir=None):
    def calculate_score(row):
        # Weights from your configuration
        w_r2 = 0.5
        w_corr = 0.2
        w_mae = 0.3
        
        # Calculate Rare/High MAE Score component
        # "If MAE is 0.1, score is 0.8. If MAE is 0.5, score is 0.0."
        mae_score = max(0, 1.0 - (row['val_high_mae'] / 0.5))
        
        # Final Score
        score = (w_r2 * row['val_r2']) + (w_corr * row['val_corr']) + (w_mae * mae_score)
        return score
    df['composite_score'] = df.apply(calculate_score, axis=1)
    style = FIGURE_CLASSES["grid_2_1"]
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]
    f1 += 2
    f2 += 2
    f3 += 2
    fig, axes = plt.subplots(3, 1, figsize=style["figsize"], sharex=True)

    # --- Panel 1: Loss (Overfitting Detection) ---
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='royalblue', linestyle='--', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='darkorange', linewidth=2)
    axes[0].axvline(x=18, color='crimson', linestyle=':', label='Selected Model', linewidth=2)
    axes[0].set_ylabel('Loss (WMSE)', fontsize=f2)
    axes[0].tick_params(axis='both', which='major', labelsize=f3)
    #axes[0].set_title('Training vs Validation Loss', loc='left', fontsize=f1)
    axes[0].legend(fontsize=f3, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Performance Metrics ---
    axes[1].plot(df['epoch'], df['val_r2'], label='Validation R²', color='forestgreen', linewidth=2)
    axes[1].plot(df['epoch'], df['val_corr'], label='Validation Pearson r', color='purple', linewidth=2)
    axes[1].axvline(x=18, color='crimson', linestyle=':', linewidth=2)
    axes[1].set_ylabel('Score', fontsize=f2)
    axes[1].tick_params(axis='both', which='major', labelsize=f3)
    #axes[1].set_title('Validation Metrics', loc='left', fontsize=f1)
    axes[1].legend(fontsize=f3, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Composite Score (Selection Criteria) ---
    axes[2].plot(df['epoch'], df['composite_score'], label='CPI', color='black', linewidth=2)

    # Annotate the Max Point
    max_epoch = df.loc[df['composite_score'].idxmax(), 'epoch']
    max_score = df['composite_score'].max()

    axes[2].scatter(max_epoch, max_score, color='crimson', s=100, zorder=5, label='Max Score')
    axes[2].axvline(x=18, color='crimson', linestyle=':', linewidth=2)

    axes[2].annotate(f'Optimal Epoch: {int(max_epoch)}\nScore: {max_score:.3f}', 
                    xy=(max_epoch, max_score), 
                    xytext=(max_epoch+3, max_score-0.03),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=f3,)

    axes[2].set_ylabel('CPI', fontsize=f2)
    axes[2].set_xlabel('Epoch', fontsize=f2)
    axes[2].tick_params(axis='both', which='major', labelsize=f3)
    #axes[2].set_title('Model Selection Criterion (CPI)', loc='left', fontsize=f1)
    axes[2].legend(fontsize=f3, loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        if outdir is not None:
            outdir_path = Path(outdir)
            out_path = outdir_path / f"training_timeline.pdf"
        else:
            out_path = f"training_timeline.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if not save:
        plt.show()

def plot_training_curves(summary_df, fold_data, outdir="results", save=True):
    style = FIGURE_CLASSES['grid_2_1']
    f1, f2, f3 = style['f1'], style['f2'], style['f3']
    """Plot training and validation loss for all folds in one graph."""
    cmap = plt.get_cmap("viridis")
    c1 = cmap(0.0)
    c2 = cmap(0.33)
    c3 = cmap(0.66)
    c4 = cmap(1.0)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Predefined colors for folds
    colors = [c4, c1, c2, c3]
    
    # Plot each fold
    for idx, (fold_num, fold_info) in enumerate(sorted(fold_data.items())):
        logs_df = fold_info["logs_df"]
        best_epoch = fold_info["best_epoch"]
        color = colors[idx % len(colors)]
        
        # Extract epochs and losses
        epochs = logs_df["epoch"].values
        train_loss = logs_df["train_loss"].values
        val_loss = logs_df["val_loss"].values
        
        # Plot train loss (solid line)
        ax.plot(epochs, train_loss, color=color, linestyle="-", linewidth=2,
                label=f"Fold {fold_num} (train)", alpha=0.8)
        
        # Plot val loss (dashed line)
        ax.plot(epochs, val_loss, color=color, linestyle="--", linewidth=2,
                label=f"Fold {fold_num} (val)", alpha=0.8)
        
        # Mark best epoch
        if best_epoch is not None:
            best_idx = np.where(epochs == best_epoch)[0]
            if len(best_idx) > 0:
                best_val_loss = val_loss[best_idx[0]]
                ax.scatter(best_epoch, best_val_loss, color=color, s=150, 
                          marker="x", zorder=5, linewidth=1.5)
    
    # Formatting
    ax.set_xlabel("Epoch", fontsize=f2)
    ax.set_ylabel("Loss", fontsize=f2)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=f3, ncol=4, framealpha=0.95)
    ax.tick_params(labelsize=f3)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.subplots_adjust(bottom=0.25)
    
    if save:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        out_path = outdir_path / "model_selection_curves.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    
    plt.show()

def plot_prediction_vs_target(pred_ds, cube_id, vmax=1, save=False, outdir=None, show=True):
    style = FIGURE_CLASSES["grid_2_1"]
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]
    n_years = pred_ds.sizes["time_year"]
    years = [np.datetime_as_string(t.values, unit='Y') for t in pred_ds.time_year]

    cmap = plt.get_cmap("inferno_r").copy()
    cmap.set_bad("lightgrey")

    fig = plt.figure(figsize=style["figsize"])
    # GridSpec: 3 rows, 5 columns. Last row is only for colorbar
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.3], hspace=0.01, wspace=0.05)

    axes = []
    for i in range(2):  # two rows for plots
        row_axes = []
        for j in range(5):  # 5 columns
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)

    
    def div_1000(x, pos):
        return f'{x/1000:g}' # :g handles decimals nicely
    formatter = FuncFormatter(div_1000)
    # --- Plotting the data ---
    for i, var in enumerate(["target", "prediction"]):
        for j, t in enumerate(pred_ds.time_year):
            ax = axes[i][j]
            # Multiply by 100 to convert to percentage points
            data_pp = pred_ds[var].sel(time_year=t) * 100
            im = data_pp.plot.imshow(
                cmap=cmap, vmin=0, vmax=vmax * 100, ax=ax,
                add_colorbar=False, add_labels=False
            )
            ax.set_aspect("equal")

            # 1. HANDLE Y-AXIS (Right-most column only)
            ax.yaxis.tick_left()
            if j == 4:
                
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('both')
                ax.yaxis.set_major_formatter(formatter)
                
                # Only put the label on the top plot of the right column
                if i == 0:
                    # Incorporate the multiplier into the label for clarity
                    ax.text(1.35, -0.05, "Northing [m] 1e3", 
                    transform=ax.transAxes, 
                    rotation=270,        # Rotate downward
                    va='center',         # Center the text on the anchor
                    ha='center', 
                    fontsize=f2)
            else:
                # Remove ticks for all other columns
                ax.yaxis.set_ticklabels([])

            # 2. HANDLE X-AXIS (Bottom row only)
            if i == 1:

                ax.xaxis.set_major_formatter(formatter)
                
                # Only put the label on the right-most plot of the bottom row
                if j == 2:
                    ax.set_xlabel("Easting [m] 1e3", fontsize=f2)
                else:
                    ax.set_xlabel("") # Keep empty for others
            else:
                # Remove ticks for the top row
                ax.xaxis.set_ticklabels([])

            # 3. ROW LABELS (Target/Prediction)
            if j == 0:
                ax.text(-0.1, 0.5, var.capitalize(), transform=ax.transAxes, 
                        rotation=90, va='center', ha='center', 
                        fontsize=f2)

            # 4. TITLES (Years - Top row only)
            if i == 0:
                ax.set_title(str(years[j]), fontsize=f2, pad=1)
            else:
                ax.set_title("")

            ax.tick_params(axis="both", which="major", labelsize=f3, pad=1)
    # --- Colorbar ---
    cax = fig.add_axes([0.35, 0.1, 0.3, 0.02])  
    # [left, bottom, width, height] in figure coordinates
    # Adjust bottom and height to create space and thin the bar

    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label("Annual Fractional Deadwood Cover Increase [pp]", fontsize=f2)
    cbar.ax.tick_params(labelsize=f3)

    if save:

        if outdir is not None:
            outdir_path = Path(outdir)
            out_path = outdir_path / f"{cube_id}.pdf"
        else:
            out_path = f"{cube_id}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

def plot_prediction_vs_target_small(pred_ds, cube_id, vmax=1, save=False, outdir=None, show=True):
    style = FIGURE_CLASSES["grid_2_1"]
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]
    n_years = pred_ds.sizes["time_year"]
    years = [np.datetime_as_string(t.values, unit='Y') for t in pred_ds.time_year]

    cmap = plt.get_cmap("inferno_r").copy()
    cmap.set_bad("lightgrey")

    fig = plt.figure(figsize=(20, 8))
    # GridSpec: 3 rows, 5 columns. Last row is only for colorbar
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.01, wspace=0.05)

    axes = []
    for i in range(2):  # two rows for plots
        row_axes = []
        for j in range(5):  # 5 columns
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)

    
    def div_1000(x, pos):
        return f'{x/1000:g}' # :g handles decimals nicely
    formatter = FuncFormatter(div_1000)
    # --- Plotting the data ---
    for i, var in enumerate(["target", "prediction"]):
        for j, t in enumerate(pred_ds.time_year):
            ax = axes[i][j]
            # Multiply by 100 to convert to percentage points
            data_pp = pred_ds[var].sel(time_year=t) * 100
            im = data_pp.plot.imshow(
                cmap=cmap, vmin=0, vmax=vmax * 100, ax=ax,
                add_colorbar=False, add_labels=False
            )
            ax.set_aspect("equal")

            # 1. HANDLE Y-AXIS (Right-most column only)
            ax.yaxis.tick_left()
            if j == 4:
                
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('both')
                ax.yaxis.set_major_formatter(formatter)
                
                # Only put the label on the top plot of the right column
                if i == 0:
                    # Incorporate the multiplier into the label for clarity
                    ax.text(1.35, -0.05, "Northing [m] 1e3", 
                    transform=ax.transAxes, 
                    rotation=270,        # Rotate downward
                    va='center',         # Center the text on the anchor
                    ha='center', 
                    fontsize=f2)
            else:
                # Remove ticks for all other columns
                ax.yaxis.set_ticklabels([])

            # 2. HANDLE X-AXIS (Bottom row only)
            if i == 1:

                ax.xaxis.set_major_formatter(formatter)
                
                # Only put the label on the right-most plot of the bottom row
                if j == 2:
                    ax.set_xlabel("Easting [m] 1e3", fontsize=f2)
                else:
                    ax.set_xlabel("") # Keep empty for others
            else:
                # Remove ticks for the top row
                ax.xaxis.set_ticklabels([])

            # 3. ROW LABELS (Target/Prediction)
            if j == 0:
                ax.text(-0.1, 0.5, var.capitalize(), transform=ax.transAxes, 
                        rotation=90, va='center', ha='center', 
                        fontsize=f2)

            # 4. TITLES (Years - Top row only)
            if i == 0:
                ax.set_title(str(years[j]), fontsize=f2, pad=1)
            else:
                ax.set_title("")

            ax.tick_params(axis="both", which="major", labelsize=f3, pad=1)
    # --- Colorbar ---
    if save:

        if outdir is not None:
            outdir_path = Path(outdir)
            out_path = outdir_path / f"{cube_id}.pdf"
        else:
            out_path = f"{cube_id}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

def plot_binned_precision_recall_percent(holdout_stats, cv_concat_stats, fold_stats_list, outdir=None, save = False):
    """
    Plots Recall (Left) and Precision (Right) in grayscale with a single centered x-label.
    """
    style = FIGURE_CLASSES['grid_2_1']
    f1, f2, f3 = style['f1'], style['f2'], style['f3']

    f1 += 2
    f2 += 2
    f3 += 2

    # 1. Prepare data
    df_folds = pd.concat(fold_stats_list, ignore_index=True)

    def get_bin_midpoint(bin_str):
        low, high = bin_str.split('-')
        return (float(low) + float(high)) / 2

    holdout_stats = holdout_stats.copy()
    cv_concat_stats = cv_concat_stats.copy()
    df_folds = df_folds.copy()

    for df in [holdout_stats, cv_concat_stats, df_folds]:
        df['bin_mid'] = df['bin_range'].astype(str).apply(get_bin_midpoint) * 100  # Convert to pp

    # Determine dynamic x-axis ticks based on 5 pp bins
    all_mids = pd.concat([
        holdout_stats[['bin_mid']],
        cv_concat_stats[['bin_mid']],
        df_folds[['bin_mid']]
    ], ignore_index=True)['bin_mid']
    x_min = float(all_mids.min())
    x_max = float(all_mids.max())
    # Snap to nearest 5 pp boundaries
    start = np.floor(x_min / 5) * 5
    end = np.ceil(x_max / 5) * 5
    xticks = np.arange(start, end + 0.01, 5)

    # 2. Initialize Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7), sharey=True)
    fig.subplots_adjust(bottom=0.17)
    # --- LEFT PLOT: RECALL ---
    sns.lineplot(data=df_folds, x='bin_mid', y='recall', 
                 errorbar='sd', color='grey', ax=ax1, label='_nolegend_')
    ax1.plot(cv_concat_stats['bin_mid'], cv_concat_stats['recall'], 
             color='black', linewidth=2.5, label='_nolegend_')
    
    ax1.set_title("Recall (Sensitivity)", fontsize=f1)
    ax1.set_ylabel("Score (0.0 - 1.0)", fontsize=f2)

    # --- RIGHT PLOT: PRECISION ---
    sns.lineplot(data=df_folds, x='bin_mid', y='precision', 
                 errorbar='sd', color='grey', ax=ax2, label='_nolegend_')
    ax2.plot(cv_concat_stats['bin_mid'], cv_concat_stats['precision'], 
             color='black', linewidth=2.5, label='_nolegend_')
    
    ax2.set_title("Precision (Reliability)", fontsize=f1)
    ax2.set_ylabel("")
    
    # Create custom legend with patches
    from matplotlib.patches import Patch
    legend_handles = [
        plt.Line2D([0], [0], color='black', linewidth=2.5, label='Global'),
        Patch(facecolor='grey', alpha=0.4, label='Fold SD'),
    ]
    ax2.legend(handles=legend_handles, loc='lower right', frameon=True, fontsize=f2)

    # Formatting for both axes
    for ax in [ax1, ax2]:
        ax.set_ylim(0, 1.05)
        ax.set_xlim(start, end)
        ax.set_xticks(xticks)
        ax.set_xlabel("") # Explicitly clear individual labels
        ax.tick_params(axis='both', labelsize=f3)
        # Angle x tick labels for readability
        ax.tick_params(axis='x', labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha='right')
        ax.grid(True, linestyle=':', alpha=0.4, color='grey')

    # 3. Single shared X-label centered below subplots
    fig.supxlabel("Annual Fractional Deadwood Cover Increase [pp]", fontsize=f2,y=0.01)

    if save:
        if outdir is not None:
            outdir_path = Path(outdir)
            outdir_path.mkdir(parents=True, exist_ok=True)
            out_path = outdir_path / "classification_results.pdf"
        else:
            out_path = "classification_results.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_cube_polygon_leaflet(cube_gdf, cube_id=251):
    """Plot cube polygon on interactive leaflet map."""
    # Convert to WGS84 (EPSG:4326) for leaflet
    cube_wgs84 = cube_gdf.to_crs('EPSG:4326')
    
    # Get bounds for map centering
    bounds = cube_wgs84.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add polygon
    for idx, row in cube_wgs84.iterrows():
        geom = row.geometry
        folium.GeoJson(
            data=geom.__geo_interface__,
            style_function=lambda x: {
                'fillColor': '#ff7f0e',
                'color': '#d62728',
                'weight': 3,
                'fillOpacity': 0.3
            }
        ).add_to(m)
        
        # Add popup with cube info
        popup_text = f"<b>Cube ID:</b> {row['cube_id']}<br><b>Is Holdout:</b> {row['is_holdout']}"
        folium.Popup(popup_text).add_to(
            folium.Marker(
                location=[geom.centroid.y, geom.centroid.x],
                popup=popup_text
            )
        )
    
    # Save map
    return m

def plot_labels_xarray(result, cube_id, save=False):
    style = FIGURE_CLASSES["square_1_1"]
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]

    def div_1000(x, pos):
        return f'{x/1000:g}' # :g handles decimals nicely
    formatter = FuncFormatter(div_1000)

    # Data from result dict
    target_arr = result["target_arr"]
    pred_arr = result["pred_arr"]
    p_mask = result["p_mask"]
    t_mask = result["t_mask"]
    p_labels = result["p_labels"]
    t_labels = result["t_labels"]
    extent = result["extent"]
    year = result["year"]

    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(3, 3, figsize=style["figsize"])
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.95,
        wspace=0.03,
        hspace=0.05
    )
    
    # Overlay colors
    VMIN, VMAX = 0, 0.6 
    cmap = plt.get_cmap("viridis")
    TP = cmap(0.5)[:3]
    FP = cmap(0.0)[:3]
    FN = cmap(1.0)[:3]

    overlay = np.ones((*p_labels.shape, 3), dtype=float)
    p_blob = p_labels > 0
    t_blob = t_labels > 0

    overlay[p_blob & ~t_blob] = FP
    overlay[~p_blob & t_blob] = FN
    overlay[p_blob & t_blob] = TP

    # Helper to simplify imshow calls with the extent
    def show(ax, data, cmap, vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin='upper')
        return im

    show(axes[0, 0], target_arr, 'inferno_r', VMIN, VMAX)
    axes[0, 0].set_title(f"Raw", fontsize=f1)
    axes[0, 0].text(-0.07, 0.5, "Target", transform=axes[0, 0].transAxes, 
                        rotation=90, va='center', ha='center', 
                        fontsize=f2)
    

    show(axes[0, 1], t_mask, 'gray_r')
    axes[0, 1].set_title(f"Binarized", fontsize=f1)
    
    t_labels_masked = np.ma.masked_where(t_labels == 0, t_labels)
    show(axes[0, 2], t_labels_masked, 'tab20')
    axes[0, 2].set_title(f"Labels", fontsize=f1)
    
    im2 = show(axes[1, 0], pred_arr, 'inferno_r', VMIN, VMAX)
    axes[1, 0].text(-0.07, 0.5, "Prediction", transform=axes[1, 0].transAxes, 
                        rotation=90, va='center', ha='center', 
                        fontsize=f2)

    show(axes[1, 1], p_mask, 'gray_r')
    
    p_labels_masked = np.ma.masked_where(p_labels == 0, p_labels)
    show(axes[1, 2], p_labels_masked, 'tab20')
    
    show(axes[2, 2], overlay, None)

    # --- EXTRAS (Legend, Colorbar) ---
    legend_elements = [
        Patch(facecolor=TP, edgecolor="lightgray", label='True Positive'),
        Patch(facecolor=FP, edgecolor="lightgray", label='False Positive'),
        Patch(facecolor=FN, edgecolor="lightgray", label='False Negative')
    ]
    axes[2, 1].legend(handles=legend_elements, loc='center', frameon=True, fontsize=f2,
                      title="Labels Overlay", title_fontsize=f1)
        
    axes[2, 0].cla()
    pos = axes[2, 0].get_position()
    new_pos = [pos.x0, pos.y0 + 0.1, pos.width, 0.03] 
    axes[2, 0].set_position(new_pos)
    cbar = fig.colorbar(im2, cax=axes[2, 0], orientation="horizontal")
    
    cbar.set_label("Fractional Deadwood\nCover Increase [pp]", fontsize=f2)
    cbar.ax.tick_params(labelsize=f3)
    pos = axes[2, 0].get_position()
    # [left, bottom, width, height] -> making it thinner (height=0.02)
    axes[2, 0].set_position([pos.x0, pos.y0 + 0.09, pos.width, 0.02])




    # axins = inset_axes(axes[2, 0], width="90%", height="5%", loc='center', borderpad=1)
    # cbar = fig.colorbar(im2, cax=axins, orientation="horizontal")
    # cbar.set_label(f"Fractional Deadwood \n Cover Increase [pp]", fontsize=f2)
    # cbar.ax.tick_params(labelsize=f3)
    
    # Formatting
    for ax in axes.flat:
        ax.title.set_fontsize(f1)

    axes[0,0].yaxis.set_ticklabels([])
    axes[0,0].xaxis.set_ticklabels([])

    axes[0,1].yaxis.set_ticklabels([])
    axes[0,1].xaxis.set_ticklabels([])

    axes[0,2].xaxis.set_ticklabels([])
    axes[0,2].yaxis.tick_right()
    axes[0,2].yaxis.get_offset_text().set_visible(False)
    axes[0,2].yaxis.set_major_formatter(formatter)
    axes[0,2].tick_params(axis="y", which="major", labelsize=f3, pad=1)
       

    axes[1,0].yaxis.set_ticklabels([])
    axes[1,0].xaxis.get_offset_text().set_visible(False)
    axes[1,0].xaxis.set_major_formatter(formatter)
    axes[1,0].tick_params(axis="x", which="major", labelsize=f3, pad=1)
    
    axes[1,1].yaxis.set_ticklabels([])
    axes[1,1].xaxis.get_offset_text().set_visible(False)
    axes[1,1].xaxis.set_major_formatter(formatter)
    axes[1,1].tick_params(axis="x", which="major", labelsize=f3, pad=1)
    fig.text(0.38, 0.33, "Easting [m] 1e3", 
             ha='center', va='top', fontsize=f2)
    
    axes[1,2].xaxis.set_ticklabels([])
    axes[1,2].yaxis.tick_right()
    axes[1,2].yaxis.get_offset_text().set_visible(False)
    axes[1,2].yaxis.set_major_formatter(formatter)
    axes[1,2].tick_params(axis="y", which="major", labelsize=f3, pad=1)
    axes[1,2].yaxis.set_label_position("right")
    axes[1,2].set_ylabel("Northing [m] 1e3", fontsize=f2, rotation=270,labelpad=45)

    axes[2,2].yaxis.tick_right()
    axes[2,2].yaxis.get_offset_text().set_visible(False)
    axes[2,2].yaxis.set_major_formatter(formatter)
    axes[2,2].xaxis.get_offset_text().set_visible(False)
    axes[2,2].xaxis.set_major_formatter(formatter)
    axes[2,2].tick_params(axis="both", which="major", labelsize=f3, pad=1)

    #axes[2,0].axis("off")
    axes[2,1].axis("off")

    if save:
        plt.savefig(f"{cube_id}_label_inspection_{year}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_layers(ds_selected, pred_ds, cube_id, save=False):

    style = FIGURE_CLASSES["square_1_1"]
    fig = plt.figure(figsize=style["figsize"])
    f1, f2, f3 = style["f1"], style["f2"], style["f3"]
    gs = gridspec.GridSpec(6, 5, height_ratios=[1, 1, 1, 1, 0.3 ,0.1], hspace=0.01, wspace=0.05)

    ext = [
        ds_selected.x.min().item(), 
        ds_selected.x.max().item(), 
        ds_selected.y.min().item(), 
        ds_selected.y.max().item()
    ]

    for i in range(5):
        # --- ROW 1: Overlay d_f=0 and d_f=1 ---
        ax1 = fig.add_subplot(gs[0, i])
        
        # Select data for the specific year
        data_df0 = ds_selected.isel(d_f=0, time_year=i)
        data_df1 = ds_selected.isel(d_f=1, time_year=i)
        
            
        
        # Create an alpha mask: 1.0 where >= 0.3, 0.0 where < 0.3
        # This effectively "masks out" the low values
        alpha_mask = np.where(data_df0 >= 0.2, 1.0, 0.0)
        
        # Plot d_f=1 as the base layer
        im1 = ax1.imshow(data_df1, cmap='Greens', vmin=0, vmax=1, alpha=0.8, extent=ext)
        if i == 0:
            ax1.text(-0.1, 0.5, "D/F Cover", transform=ax1.transAxes, 
                            rotation=90, va='center', ha='center', 
                            fontsize=f2)

        
        # Plot d_f=0 on top with the mask
        # Note: .values is used to ensure the numpy mask applies correctly
        im2 = ax1.imshow(data_df0, cmap='Reds', alpha = alpha_mask, vmin=0, vmax=1, extent=ext)
        ax1.set_title(f"{ds_selected.time_year.dt.year.values[i]}", fontsize=f1)
        ax1.xaxis.set_ticklabels([])
        if i !=4:
            ax1.yaxis.set_ticklabels([])

        # --- ROW 2: d_f=3 for all years ---
        ax2 = fig.add_subplot(gs[1, i])
        data_df3 = ds_selected.isel(d_f=3, time_year=i)
        im3 = ax2.imshow(data_df3, cmap='inferno_r', vmin=0, vmax=0.8, extent=ext)
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        if i == 0:
            ax2.text(-0.1, 0.5, "Forest dec", transform=ax2.transAxes, 
                            rotation=90, va='center', ha='center', 
                            fontsize=f2)
        if i == 4:   
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            ax1.yaxis.set_ticks_position('both')
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))
            ax1.yaxis.get_offset_text().set_visible(False) # Hide stubborn 1e6
            ax1.tick_params(axis="y", which="major", labelsize=f3, pad=1)
            ax1.text(1.45, -1.3, "Northing [m] 1e3", 
                    transform=ax1.transAxes, 
                    rotation=270,        # Rotate downward
                    va='center',         # Center the text on the anchor
                    ha='center',
                    fontsize=f2)


        if i == 4:
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.set_ticks_position('both')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))
            ax2.yaxis.get_offset_text().set_visible(False) # Hide stubborn 1e6
            ax2.tick_params(axis="y", which="major", labelsize=f3, pad=1)


        # --- ROW 3: d_f=2 for all years ---
        ax3 = fig.add_subplot(gs[2, i])
        data_df2 = ds_selected.isel(d_f=2, time_year=i)
        im4 = ax3.imshow(data_df2, cmap='inferno_r', vmin=0, vmax=0.8, extent=ext)
        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
        if i == 0:
            ax3.text(-0.1, 0.5, "Dead inc", transform=ax3.transAxes, 
                            rotation=90, va='center', ha='center', 
                            fontsize=f2)
        if i == 4:
            #turn ticks and label on and on the left side
            ax3.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.set_ticks_position('both')
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))
            ax3.yaxis.get_offset_text().set_visible(False) # Hide stubborn 1e6
            ax3.tick_params(axis="y", which="major", labelsize=f3, pad=1)

        ax4 = fig.add_subplot(gs[3, i])
        data_pred = pred_ds['prediction'].isel(time_year=i)
        im5 = ax4.imshow(data_pred, cmap='inferno_r', vmin=0, vmax=0.8, extent=ext)
        ax4.yaxis.set_ticklabels([])
        ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))
        ax4.xaxis.get_offset_text().set_visible(False) # Hide stubborn 1e6
        ax4.tick_params(axis="x", which="major", labelsize=f3, pad=1)

        if i == 0:
            ax4.text(-0.1, 0.5, "Prediction", transform=ax4.transAxes, 
                            rotation=90, va='center', ha='center', 
                            fontsize=f2)
        if i == 2:
            ax4.set_xlabel("Easting [m] 1e3", fontsize=f2)

        if i == 4:
            #turn ticks and label on and on the left side
            ax4.yaxis.tick_right()
            ax4.yaxis.set_label_position("right")
            ax4.yaxis.set_ticks_position('both')
            ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))
            ax4.yaxis.get_offset_text().set_visible(False) # Hide stubborn 1e6
            ax4.tick_params(axis="y", which="major", labelsize=f3, pad=1)

    cbar_configs = [
        (im1, 0, "Forest Cover [%]"),
        (im2, 2, "Deadwood Cover [%]"),
        (im4, 4, "Cover Change [pp]"),
    ]

    for mappable, col, label in cbar_configs:
        # Create an axis in the 5th row (index 4)
        cax = fig.add_subplot(gs[5, col])
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - 0.05, pos.width, 0.015])
        
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", format='%g')
        cbar.set_label(label, fontsize=f2)
        cbar.ax.tick_params(labelsize=f3)

    #plt.tight_layout()
    plt.subplots_adjust(
        left=0.08,
        right=0.92,
        top=0.95,
        bottom=0.1  # Increase this if the colorbar labels are cut off
    )
    if save:
        plt.savefig(f"{cube_id}_layers.pdf", dpi=300, bbox_inches="tight")
    plt.show()

