# Deadwood Forecasting Model

A multimodal transformer-based deep learning system for predicting deadwood occurrence and forest characteristics from satellite and climate data. The model integrates satellite imagery, elevation, canopy density, ERA5 climate data, and other geospatial features to forecast deadwood and forest dynamics.

This repository is currently in alpha stage. If you have any questions feel free to reach out.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)

## Overview

This project implements a **MultimodalDeadwoodTransformer** that predicts:
- **Annual Fractional Deadwood Cover Increase**

The model leverages:
- **Satellite imagery**: Sentinel-2 and other remote sensing data
- **Climate data**: ERA5 reanalysis (temperature, precipitation, etc.)
- **Geospatial features**: Terrain, canopy density, forest age
- **Transformer architecture**: Multi-head attention for capturing spatial-temporal dependencies

## Features

- 🌍 **Multimodal input**: Combines satellite, climate, and geospatial data
- 🤖 **Transformer-based**: Multi-head attention architecture for improved feature interaction
- 📊 **Curriculum learning**: Rare sample infusion for better minority class performance
- 💾 **Checkpoint management**: Automatic best model selection based on composite metrics
- 📈 **Comprehensive logging**: Timestamps, training curves, validation metrics
- 🔧 **Flexible configuration**: YAML-based experiment management
- ⚡ **Distributed training**: Support for multiple GPUs and fold-based cross-validation
- 🎯 **Multiple loss functions**: MSE, Huber, weighted variants

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.4+ (for GPU training)
- 100+ GB disk space (for data cubes)

### Setup Environment

```bash
# Create conda environment
conda create -n deadwood_forecasting_model python=3.12 -c conda-forge -y
conda activate deadwood_forecasting_model

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda install xarray zarr zstandard rasterio rioxarray geopandas -y
conda install h5netcdf ipykernel s3fs pyarrow scikit-learn dask -y
conda install tqdm captum scikit-image -y
```

Or use the environment file:
```bash
conda env create -f docs/environment.yml
conda activate deadwood_forecasting_model
```

## Project Structure

```
deadwood_forecasting_model/
├── analysis/                    # Data analysis and exploration
│   ├── analysis.py
│   ├── classification.py
│   ├── ig_cube.py              # Integrated gradients analysis
│   ├── model_selection.py
│   └── regression.py
│
├── config/                      # Configuration management
│   ├── base.yaml               # Default settings
│   ├── paths.py                # Centralized path definitions
│   └── experiments/            # Experiment-specific configs
│       ├── run_00_dw_inc_f_tresh.yaml
│       ├── run_01_dw_inc_f_tresh.yaml
│       ├── run_02.yaml
│       └── ...
│
├── data/                        # Data storage
│   ├── cubes/                  # Zarr format data cubes (0.zarr - 299.zarr)
│   ├── figs/                   # Generated visualizations
│   ├── logs/                   # Training logs
│   ├── meta_data/              # Metadata tables
│   ├── predictions/            # Model predictions
│   ├── training_sets/          # Training/validation splits
│   └── training_runs/          # Experiment checkpoints & logs
│
├── docs/                        # Documentation
│   ├── EXPERIMENT_GUIDE.md     # Detailed experiment workflow
│   ├── env_setup.txt
│   └── environment.yml
│
├── inference/                   # Prediction pipeline
│   ├── evaluate_model.py       # Model evaluation
│   ├── predict_all_holdout_cubes.py
│   ├── step07_create_specifc_cubes.py
│   ├── step08_create_prediction.py
│   └── step09_analyze_predictions.py
│
├── models/                      # Neural network architectures
│   ├── model.py                # v1: Single-output transformer
│   ├── model2.py               # v2: Dual-output transformer
│   └── model_small.py          # Lightweight variant
│
├── processing/                  # Data preparation pipeline
│   ├── step01_select_cubes.py
│   ├── step02_built_cubes.py
│   ├── step03_calculate_znorm_stats.py
│   ├── step04_create_training_meta_table.py
│   ├── step05_create_holdouts_folds.py
│   └── step06_create_trainingsset.py
│
├── training/                    # Training pipeline
│   ├── trainer.py              # Main training entry point
│   ├── train_utils.py          # Training utilities & config management
│   ├── setup_training.py       # Dataset/dataloader setup
│   ├── losses.py               # Custom loss functions
│   └── trainer.py
│
├── utils/                       # Utility functions
│   ├── built_era5_cube.py      # ERA5 data processing
│   ├── era5_downloader.py      # ERA5 download manager
│   ├── means_dw_sentle.py      # Feature computation
│   ├── parall.py               # Parallelization utilities
│   ├── plots.py                # Visualization helpers
│   └── random.py
│
├── scripts/                     # Standalone analysis scripts
│   ├── analyze_meta_table.py
│   ├── analyze_training_set.py
│   └── select_locations.py
│
├── results/                     # Final results and outputs
│   └── full_scale_predictions/
│
└── README.md                    # This file
```

## Data Pipeline

### Data Flow

```
Raw Data (Sentinel-2, ERA5, DEM, etc.)
    ↓
step01_select_cubes.py      → Select geographic regions
    ↓
step02_built_cubes.py       → Create spatiotemporal Zarr cubes
    ↓
step03_calculate_znorm_stats.py → Compute normalization statistics
    ↓
step04_create_training_meta_table.py → Build metadata index
    ↓
step05_create_holdouts_folds.py → Split into train/val folds
    ↓
step06_create_trainingsset.py → Create training datasets
    ↓
Training / Inference
```

### Data Format

- **Cubes**: Zarr-formatted geospatial data cubes (3D: time × height × width)
  - Located in `data/cubes/{cube_id}.zarr/`
  - Each cube contains multiple spatiotemporal variables
  
- **Features**:
  - `deadwood_forest`: Deadwood observations
  - `terrain`: Elevation/slope
  - `canopy`: Canopy density
  - `pixels_sentle`: Sentinel-2 imagery
  - `era5`: Climate variables
  - `wc`: Worldclim data
  - `sg`: Soil/geology
  - `stand_age`: Forest age
