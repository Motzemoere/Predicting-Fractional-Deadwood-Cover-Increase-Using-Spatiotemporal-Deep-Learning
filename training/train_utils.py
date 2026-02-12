"""
Training utilities and experiment configuration management.
Handles config serialization, model checkpointing, evaluation metrics, and training loops.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
from dataclasses import fields
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path
from datetime import datetime
import torch.optim as optim

from config.paths import PATHS

FEATURE_KEYS = [
    "deadwood_forest",
    "terrain",
    "canopy",
    "pixels_sentle",
    "era5",
    "wc",
    "sg",
    "stand_age",
]

@dataclass
class ExperimentConfig:
    """Structured configuration for a training run."""
    # Experiment metadata
    name: str
    description: str = ""
    tags: list = None

    fold: int = None  # If None, use full training set
    
    # Paths (defaults from centralized config)
    data_dir: str = None  # Will default to PATHS.data_dir
    training_set: str = "training_set_003"
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = str(PATHS.data_dir)
        if self.tags is None:
            self.tags = []
    
    # Model
    model_variant: str = "v1"  # v1 or v2
    embed_dim: int = 256      
    
    # Data params
    max_years: int = 3
    max_weeks: int = 156
    patch_size: int = 33
    val_ratio: float = 0.2
    
    # DataLoader
    batch_size: int = 256
    num_workers: int = 32
    
    # Training
    num_epochs: int = 50
    device: str = "cuda:0"
    
    # Optimizer
    peak_lr: float = 1e-4
    weight_decay: float = 0.1
    
    # Scheduler
    warmup_epochs: int = 5
    plateau_epochs: int = 3
    min_lr_ratio: float = 0.01

    # Log-space training
    train_log_space: bool = False
    log_space_switch_epoch: int = 999 
    
    # Loss
    loss_name: str = "MSELoss"
    huber_delta: float = 0.15
    weight_exponent: float = 1.0   # For WeightedMSELoss
    max_gain: float = 5.0          # For WeightedMSELoss
    
    # Rare infusion (warmup → gradual ramp → stable)
    rare_infusion_enabled: bool = True
    rare_warmup_epochs: int = 10      # No rare samples during warmup
    rare_ramp_end_epoch: int = 30     # Ramp completes at this epoch
    rare_ratio: float = 0.05          # Target ratio (reached at ramp_end_epoch)
    
    # Target
    target_name: str = "target_d"
    multi_output: bool = False
    
    # Checkpoint / Model Selection
    min_std_ratio: float = 0.1
    r2_weight: float = 0.4         # Weight for R2 (global accuracy)
    corr_weight: float = 0.3       # Weight for correlation (signal detection)
    high_mae_weight: float = 0.3   # Weight for high-value MAE (rare accuracy)

def load_config_from_json(
    json_path: str | Path,
    overrides: Dict[str, Any] | None = None,
    strict: bool = True,
) -> ExperimentConfig:
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        raw_cfg = json.load(f)

    # 1. Get all fields defined in the dataclass
    config_fields = fields(ExperimentConfig)
    valid_names = {f.name for f in config_fields}

    # 2. Strict Check: Are there keys in JSON that don't belong?
    if strict:
        unknown = set(raw_cfg) - valid_names
        if unknown:
            raise ValueError(f"Unknown keys in {json_path}: {sorted(unknown)}")

    # 4. Filter and Apply Overrides
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in valid_names}
    if overrides:
        for k, v in overrides.items():
            if strict and k not in valid_names:
                raise ValueError(f"Invalid override key: {k}")
            cfg_kwargs[k] = v

    # 5. Reconstruct
    return ExperimentConfig(**cfg_kwargs)

class ExperimentTracker:
    """Simple experiment tracking - saves everything needed to reproduce."""
    
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        base_run_dir = PATHS.training_runs / cfg.name

        if cfg.fold is None:
            self.out_dir = base_run_dir
            print(f"--- Tracker: Setting up for full training set ---")
        else:
            self.fold = int(cfg.fold)
            print(f"--- Tracker: Setting up for fold {self.fold} ---")
            self.out_dir = base_run_dir / f"fold_{self.fold}"

        self.cfg.run_dir = str(self.out_dir)

        # Setup directory: Don't crash if it exists
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._save_config()
            self.logs = []
            self.best_score = -np.inf
        else:
            print(f"--- Tracker: Connecting to existing directory {self.out_dir} ---")
            self._load_existing_state()

    def _load_existing_state(self):
        meta_path = self.out_dir / "best_model_meta.json"
        log_path = self.out_dir / "training_logs.csv"

        # 1. Load best score from tiny JSON
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.best_score = meta.get("score", -np.inf)
                print(f"--- Tracker: Recovered best score {self.best_score:.4f} from Meta JSON ---")
        else:
            self.best_score = -np.inf

        if log_path.exists():
                try:
                    df = pd.read_csv(log_path)
                    self.logs = df.to_dict('records')
                except Exception as e:
                    print(f"Warning: Could not load existing logs: {e}")
                    self.logs = []
        else:
            self.logs = []
        
    def _save_config(self):
        """Save full configuration for reproducibility."""
        config_path = self.out_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.cfg), f, indent=2)
    
    def log_epoch(self, metrics: Dict[str, Any]):
        """Log metrics for one epoch."""
        self.logs.append(metrics)
        
        # Save logs after each epoch
        df = pd.DataFrame(self.logs)
        df.to_csv(self.out_dir / "training_logs.csv", index=False)
        
    def save_checkpoint(self, *, model: nn.Module, score: float, 
                        is_best: bool , epoch: int , 
                        optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "score": float(score),
            "timestamp": str(datetime.now().isoformat())
        }

        meta = {
        "epoch": epoch,
        "score": score,
        "timestamp": datetime.now().isoformat(),
        "is_best": is_best
        }

        # Always save 'last' so we can resume if the power cuts out
        torch.save(checkpoint, self.out_dir / "last_model.pth")
        
        if is_best:
            self.best_score = score 
            # Best resume checkpoint
            torch.save(checkpoint, self.out_dir / "best_model.pth")
            # Weights-only best model for easy loading
            torch.save(model.state_dict(), self.out_dir / "best_weights.pth")
            # Save tiny JSON with best model metadata
            with open(self.out_dir / "best_model_meta.json", 'w') as f:
                    json.dump(meta, f, indent=2)
                
            
    def finish(self):
        """Finalize tracking - save final summary."""
        pass

def apply_cli_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply command line overrides to config."""
    if args.device is not None:
            device_index = int(args.device)
            cfg.device = f"cuda:{device_index}"
    if args.fold is not None:
        cfg.fold = args.fold
    return cfg

def build_model(cfg: ExperimentConfig):
    """Build model based on configuration."""
    if cfg.model_variant == "v2":
        from models.model2 import MultimodalDeadwoodTransformer
    elif cfg.model_variant == "small":
        from models.model_small import MultimodalDeadwoodTransformer
    else:
        from models.model import MultimodalDeadwoodTransformer
    return MultimodalDeadwoodTransformer(embed_dim=cfg.embed_dim).to(cfg.device)

def build_loss(cfg: ExperimentConfig):
    """Build loss function based on configuration."""
    if cfg.loss_name == "HuberLoss":
        return nn.HuberLoss(delta=cfg.huber_delta).to(cfg.device)
    elif cfg.loss_name == "MSELoss":
        return nn.MSELoss().to(cfg.device)
    elif cfg.loss_name == "L1Loss":
        return nn.L1Loss().to(cfg.device)
    elif cfg.loss_name == "WeightedMSELoss":
        from training.losses import WeightedMSELoss
        return WeightedMSELoss(exponent=cfg.weight_exponent, max_gain=cfg.max_gain).to(cfg.device)
    else:
        raise ValueError(f"Unknown loss: {cfg.loss_name}")

def build_scheduler(optimizer, cfg: ExperimentConfig, steps_per_epoch: int):
    """Build learning rate scheduler with warmup → plateau → cosine decay."""
    total_steps = cfg.num_epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    plateau_steps = cfg.plateau_epochs * steps_per_epoch
    decay_steps = total_steps - (warmup_steps + plateau_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < (warmup_steps + plateau_steps):
            return 1.0
        else:
            step_in_decay = current_step - (warmup_steps + plateau_steps)
            progress = float(step_in_decay) / float(max(1, decay_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(cfg.min_lr_ratio, cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def setup_training_objects(cfg, train_loader_len):
    model = build_model(cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, train_loader_len)
    criterion = build_loss(cfg)

    return model, optimizer, scheduler, criterion

def maybe_resume(cfg, model, optimizer, scheduler):
    resume_path = Path(cfg.run_dir) / "last_model.pth"
    start_epoch = 1

    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=cfg.device, weights_only=False) # Trusted source
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, start_epoch

def compute_composite_score(eval_metrics: Dict, cfg: ExperimentConfig) -> float:
    r2_key = "dw_r2" if cfg.multi_output else "r2"
    mae_key = "dw_high_mae" if cfg.multi_output else "high_mae"
    corr_key = "dw_corr" if cfg.multi_output else "corr"
    
    r2 = eval_metrics.get(r2_key, 0) #(Global accuracy)
    corr = eval_metrics.get(corr_key, 0) #Correlation (Signal detection - insensitive to bias)
    # 3. High MAE Score (Error on important rare samples)
    # If MAE is 0.1, score is 0.8. If MAE is 0.5, score is 0.0.
    high_mae = eval_metrics.get(mae_key, 1.0)
    high_mae_score = max(0, 1.0 - (high_mae / 0.5))
    
    # Balanced weighting
    # 40% R2, 30% Correlation, 30% Rare Accuracy
    score = (cfg.r2_weight * r2) + (cfg.corr_weight * corr) + (cfg.high_mae_weight * high_mae_score)
    return score

def evaluate_model(target, model, dataloader, device, feature_keys, criterion, log_space=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: batch[key].to(device) for key in feature_keys}
            targets = batch[target].to(device).unsqueeze(-1)
            if log_space:
                targets = torch.log1p(targets)
            preds = model(**inputs)
            loss = criterion(preds, targets)

            if log_space:
                preds = torch.expm1(preds)
                targets = torch.expm1(targets)
            
            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Final metrics calculation on full dataset
    y_true = torch.cat(all_targets).squeeze().numpy().flatten()
    y_pred = torch.cat(all_preds).squeeze().numpy().flatten()
    
    # 1. Distribution Masks
    thresh_90 = np.percentile(y_true, 90)
    high_mask = y_true >= thresh_90
    low_mask = y_true < thresh_90

    # 2. Key Baseline Metrics
    avg_loss = total_loss / total_samples
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Natural Forest Performance (The Low End)
    low_mae = mean_absolute_error(y_true[low_mask], y_pred[low_mask]) if low_mask.any() else 0.0
    # Rare Event Performance (The High End)
    high_mae = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.any() else 0.0
    
    # 3. Variance & Trend Metrics
    std_target = y_true.std()
    std_pred = y_pred.std()
    std_ratio = std_pred / (std_target + 1e-8)
    
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    bias = (y_pred - y_true).mean()
    
    metrics = {
        "loss": avg_loss,
        "r2": r2,
        "mae": mae,
        "low_mae": low_mae,
        "high_mae": high_mae,
        "std_ratio": std_ratio,
        "corr": corr,
        "bias": bias,
        "pred_std": std_pred
    }
    return metrics

# def evaluate_model2(model, dataloader, device, feature_keys, criterion):
#     model.eval()
#     total_loss = 0.0
#     total_samples = 0
    
#     # We will store predictions and targets as lists of tensors
#     # Shape will eventually be converted to [Total_Samples, 2]
#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = {key: batch[key].to(device) for key in feature_keys}
            
#             # --- 1. Load Both Targets ---
#             t_d = batch["target_d"].to(device).unsqueeze(-1) # [B, 1] Deadwood
#             t_f = batch["target_f"].to(device).unsqueeze(-1) # [B, 1] Forest Decrease
#             targets = torch.cat([t_d, t_f], dim=1)           # [B, 2]
            
#             preds = model(**inputs)       # [B, 2]

#             loss = criterion(preds, targets)
            
#             bs = targets.size(0)
#             total_loss += loss.item() * bs
#             total_samples += bs

#             all_preds.append(preds.cpu())
#             all_targets.append(targets.cpu())

#     # --- 3. Prepare Data for Metrics ---
#     # Concatenate all batches -> [Total_Samples, 2]
#     y_true_all = torch.cat(all_targets).numpy()
#     y_pred_all = torch.cat(all_preds).numpy()

#     avg_loss = total_loss / total_samples

#     # Helper function to calculate metrics for a single column (task)
#     def calc_single_task_metrics(y_true, y_pred, prefix):
#         # 1. Distribution Masks (Top 10% vs Bottom 90%)
#         # Note: If data is very sparse (mostly 0s), the 90th percentile might be 0.
#         # We ensure thresh is at least small epsilon (1e-4) to find actual "events"
#         thresh_90 = max(np.percentile(y_true, 90), 1e-4)
        
#         high_mask = y_true >= thresh_90
#         low_mask = y_true < thresh_90

#         # 2. Key Baseline Metrics
#         # Handle edge case where R2 is undefined (e.g., constant target)
#         if len(y_true) > 1 and np.var(y_true) > 0:
#             r2 = r2_score(y_true, y_pred)
#             corr, _ = pearsonr(y_true, y_pred)
#         else:
#             r2 = 0.0
#             corr = 0.0

#         mae = mean_absolute_error(y_true, y_pred)
        
#         # Natural/Background Performance (The Low End)
#         low_mae = mean_absolute_error(y_true[low_mask], y_pred[low_mask]) if low_mask.any() else 0.0
        
#         # Rare Event Performance (The High End)
#         high_mae = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.any() else 0.0
        
#         # 3. Variance & Trend Metrics
#         std_target = y_true.std()
#         std_pred = y_pred.std()
#         std_ratio = std_pred / (std_target + 1e-8)
        
#         bias = (y_pred - y_true).mean()

#         return {
#             f"{prefix}_r2": r2,
#             f"{prefix}_mae": mae,
#             f"{prefix}_low_mae": low_mae,
#             f"{prefix}_high_mae": high_mae,
#             f"{prefix}_std_ratio": std_ratio,
#             f"{prefix}_corr": corr,
#             f"{prefix}_bias": bias,
#             f"{prefix}_pred_std": std_pred
#         }

#     # --- 4. Calculate Metrics for Each Task ---
#     # Column 0: Deadwood (dw)
#     metrics_dw = calc_single_task_metrics(y_true_all[:, 0], y_pred_all[:, 0], prefix="dw")
    
#     # Column 1: Forest Decrease (fd)
#     metrics_fd = calc_single_task_metrics(y_true_all[:, 1], y_pred_all[:, 1], prefix="fd")

#     # --- 5. Merge ---
#     metrics = {
#         "loss": avg_loss,
#         **metrics_dw,
#         **metrics_fd
#     }
    
#     return metrics

def train_epochs(
    *,
    cfg,
    model,
    loader_train,
    loader_val,
    optimizer,
    scheduler,
    criterion,
    feature_keys,
    tracker,
    compute_score_fn,
    start_epoch
):
    """
    Run the training + validation loop for all epochs.

    Parameters
    ----------
    cfg : ExperimentConfig
    model : torch.nn.Module
    loader_train : DataLoader
    loader_val : DataLoader
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    criterion : torch.nn.Module
    device : torch.device
    feature_keys : list[str]
    tracker : ExperimentTracker | FoldTracker
    compute_score_fn : callable
        Typically compute_composite_score
    """
    train_log_space = cfg.train_log_space
    if start_epoch >= cfg.log_space_switch_epoch:
        train_log_space = False
    for epoch in tqdm(range(start_epoch, cfg.num_epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        if epoch == cfg.log_space_switch_epoch:
            train_log_space = False
            
        for batch in loader_train:
            optimizer.zero_grad()

            inputs = {
                key: batch[key].to(cfg.device)
                for key in feature_keys
            }

            # Targets
            if cfg.multi_output:
                raise NotImplementedError("evaluate_model2 is currently disabled")
                t_d = batch["target_d"].to(cfg.device).unsqueeze(-1)
                t_f = batch["target_f"].to(cfg.device).unsqueeze(-1)
                targets = torch.cat([t_d, t_f], dim=1)
            else:
                targets = batch[cfg.target_name].to(cfg.device).unsqueeze(-1)

            if train_log_space:
                targets = torch.log1p(targets)

            preds = model(**inputs)
            loss = criterion(preds, targets)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if train_log_space:
                preds = torch.expm1(preds)

            bs = targets.size(0)
            epoch_loss += loss.item() * bs
            epoch_samples += bs

        epoch_loss /= epoch_samples
        current_lr = optimizer.param_groups[0]["lr"]

        if cfg.multi_output:
            raise NotImplementedError("evaluate_model2 is currently disabled")
            val_metrics = evaluate_model2(
                model,
                loader_val,
                cfg.device,
                feature_keys,
                criterion,
                log_space=train_log_space,
            )
        else:
            val_metrics = evaluate_model(
                target=cfg.target_name,
                model=model,
                dataloader=loader_val,
                device=cfg.device,
                feature_keys=feature_keys,
                criterion=criterion,
                log_space=train_log_space,
            )

        # -------------------------------
        # Logging
        # -------------------------------
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "lr": f"{current_lr:.2e}",
            "train_loss": epoch_loss,
            "rare_ratio": (
                loader_train.dataset.rare_ratio
                if cfg.rare_infusion_enabled
                else 0
            ),
        }
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        tracker.log_epoch(metrics)

        # -------------------------------
        # Checkpointing
        # -------------------------------
        std_key = "dw_std_ratio" if cfg.multi_output else "std_ratio"
        score = compute_score_fn(val_metrics, cfg)

        if score > tracker.best_score and val_metrics.get(std_key, 1) > cfg.min_std_ratio:
            tracker.save_checkpoint(
                model=model, score=score, is_best=True, epoch=epoch, 
                optimizer=optimizer, scheduler=scheduler)
            print(f"  → New best model! Score: {score:.4f}")
        else:
            tracker.save_checkpoint(
                model=model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                epoch=epoch, 
                score=score, 
                is_best=False
            )

    tracker.finish()

    print(f"\n{'='*60}")
    print(f"Training complete! Best score: {tracker.best_score:.4f}")
    print(f"Results saved to: {tracker.out_dir}")
    print(f"{'='*60}\n")

def load_best_model(cfg, model, weights_only: bool = True):
    weights_path = Path(cfg.run_dir) / "best_weights.pth"
    best_ckpt_path = Path(cfg.run_dir) / "best_model.pth"

    if weights_only:
        print("Loading best model (weights_only=True)...")

        if weights_path.exists():
            state_dict = torch.load(
                weights_path,
                map_location=cfg.device,
                weights_only=True
            )

        elif best_ckpt_path.exists():
            # Backward compatibility for old runs
            print("No weights file found. Extracting weights from full checkpoint (legacy).")
            checkpoint = torch.load(
                best_ckpt_path,
                map_location=cfg.device,
                weights_only=False
            )
            state_dict = checkpoint["model_state_dict"]

        else:
            raise FileNotFoundError("No best model found.")

        model.load_state_dict(state_dict)
        model.to(cfg.device)
        model.eval()
        return model

    # ---- Full training checkpoint path ----
    print("Loading full best model checkpoint...")
    checkpoint = torch.load(
        best_ckpt_path,
        map_location=cfg.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)
    model.eval()
    return model

def predict(*, cfg, target, model, dataloader, feature_keys, log_space, return_predictions=False):
    model.eval()
    total_samples = 0
    all_preds = []
    all_targets = []
    
    outputs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {key: batch[key].to(cfg.device) for key in feature_keys}
            targets = batch[target].to(cfg.device).unsqueeze(-1)
            if log_space:
                targets = torch.log1p(targets)

            preds = model(**inputs)

            if log_space:
                preds = torch.expm1(preds)
                targets = torch.expm1(targets)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            bs = targets.size(0)
            total_samples += bs

            if return_predictions:
                outputs_list.append({
                    "preds": preds.cpu().numpy(),
                    "targets": batch[target].cpu().numpy(),
                    "year": batch["year"].cpu().numpy(),
                    "pixel_keys": np.array(batch["pixel_key"]),
                    })

    # Final metrics calculation on full dataset
    y_true = torch.cat(all_targets).squeeze().numpy().flatten()
    y_pred = torch.cat(all_preds).squeeze().numpy().flatten()
    
    # 1. Distribution Masks
    thresh_90 = np.percentile(y_true, 90)
    high_mask = y_true >= thresh_90
    low_mask = y_true < thresh_90

    # 2. Key Baseline Metrics

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Natural Forest Performance (The Low End)
    low_mae = mean_absolute_error(y_true[low_mask], y_pred[low_mask]) if low_mask.any() else 0.0
    # Rare Event Performance (The High End)
    high_mae = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.any() else 0.0
    
    # 3. Variance & Trend Metrics
    std_target = y_true.std()
    std_pred = y_pred.std()
    std_ratio = std_pred / (std_target + 1e-8)
    
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    bias = (y_pred - y_true).mean()
    
    metrics = {
        "r2": r2,
        "mae": mae,
        "low_mae": low_mae,
        "high_mae": high_mae,
        "std_ratio": std_ratio,
        "corr": corr,
        "bias": bias,
        "pred_std": std_pred,
        "samples": total_samples
    }
    if return_predictions:
        # df with true/pred columns
        all_preds = np.concatenate([o["preds"] for o in outputs_list]).flatten()
        all_targets = np.concatenate([o["targets"] for o in outputs_list]).flatten()
        all_keys = np.concatenate([o["pixel_keys"] for o in outputs_list])
        all_years = np.concatenate([o["year"] for o in outputs_list]).flatten()

        results_df = pd.DataFrame({
            "pixel_key": all_keys,
            "prediction": all_preds,
            "target": all_targets,
            "year": all_years
        })

        
        return metrics, results_df
    return metrics
