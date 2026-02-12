"""
Evaluate a trained model on an evaluation dataset and save prediction results.
Generates evaluation metrics and result predictions for specified fold or full run.
"""

import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import PATHS

from training.train_utils import (FEATURE_KEYS, load_config_from_json, 
                                  build_model, load_best_model, predict, 
                                  apply_cli_overrides)
from training.setup_training import setup_evaluation_dataset, build_evaluation_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    run_dir = PATHS.training_runs / args.run
    
    if args.fold is not None:
        run_dir = run_dir / f"fold_{args.fold}"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    cfg = load_config_from_json(run_dir / "config.json")
    cfg = apply_cli_overrides(cfg, args)

    cfg.run_dir = run_dir

    ds_eval, df_eval = setup_evaluation_dataset(cfg, PATHS)
    loader_eval = build_evaluation_dataloader(cfg, ds_eval)
    model = build_model(cfg)
    model = load_best_model(cfg, model)

    metrics, result_df = predict(
        cfg=cfg,
        target=cfg.target_name,
        model=model,
        dataloader=loader_eval,
        feature_keys=FEATURE_KEYS,
        log_space=cfg.train_log_space,
        return_predictions=True
    )

    # save results
    result_df.to_parquet(run_dir / "evaluation_results.parquet")
    pd.DataFrame([metrics]).to_csv(run_dir / "evaluation_metrics.csv", index=False)


if __name__ == "__main__":
    main()