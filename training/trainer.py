"""
Main training orchestration script.
Loads config, initializes models and dataloaders, and executes the training loop.
"""

import argparse
import sys
from pathlib import Path
import torch
import geopandas as gpd
import torch.optim as optim

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import PATHS
from training.train_utils import load_config_from_json, apply_cli_overrides
from training.setup_training import  (setup_training_datasets, 
                                      build_training_dataloaders)

from training.train_utils import (
    ExperimentTracker, setup_training_objects, 
    maybe_resume, compute_composite_score, train_epochs, FEATURE_KEYS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config_from_json(PATHS.training_runs / args.run / "config.json")
    cfg = apply_cli_overrides(cfg, args)

    tracker = ExperimentTracker(cfg)

    ds_train, ds_val, df_train, df_val = setup_training_datasets(cfg, PATHS)

    loader_train, loader_val = build_training_dataloaders(cfg, ds_train, ds_val)

    model, optimizer, scheduler, criterion = setup_training_objects(
        cfg, len(loader_train)
    )

    model, optimizer, scheduler, start_epoch = maybe_resume(cfg, model, optimizer, scheduler)

    train_epochs(
        cfg = cfg,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        loader_train = loader_train,
        loader_val = loader_val,
        tracker = tracker,
        feature_keys = FEATURE_KEYS,
        compute_score_fn = compute_composite_score,
        start_epoch = start_epoch
    )

if __name__ == "__main__":
    main()