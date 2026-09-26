"""
Training loop for U-Net and ConvLSTM wildfire spread models.

Usage (via Hydra):
  python src/training/train.py --config-name unet
  python src/training/train.py --config-name convlstm model.seq_len=5
  python src/training/train.py --config-name convlstm model.seq_len=1  # ablation T=1

Features:
  - Hydra config composition
  - MLflow experiment tracking (all metrics, artefacts, model checkpoints)
  - Early stopping on val AUC-PR
  - Apple Silicon MPS support (falls back to CPU)
  - Cosine LR schedule
  - Weighted Dice+BCE loss (pos_weight from training split)
"""

from __future__ import annotations

import datetime
import logging
import subprocess
from pathlib import Path
from typing import Optional

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import make_dataloaders
from src.data.transforms import build_train_transforms, build_eval_transforms
from src.models.losses import CombinedLoss
from src.models.unet import build_unet
from src.models.convlstm import build_convlstm
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_training_curves, spatial_confusion_map

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> nn.Module:
    model_name = cfg.model.name
    if model_name == "unet":
        return build_unet(
            in_channels=cfg.model.in_channels,
            base_filters=cfg.model.base_filters,
            depth=cfg.model.depth,
            dropout=cfg.model.dropout,
        )
    elif model_name == "convlstm":
        return build_convlstm(
            in_channels=cfg.model.in_channels,
            hidden_channels=list(cfg.model.hidden_channels),
            kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    seq_mode: bool = False,
) -> float:
    """Returns mean training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        X, y = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)                          # (B, 1, H, W)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Returns (mean_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        X, y = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())

    all_probs_np = np.concatenate([p.ravel() for p in all_probs])
    all_targets_np = np.concatenate([t.ravel() for t in all_targets])
    metrics = compute_metrics(all_targets_np, all_probs_np)

    return total_loss / max(n_batches, 1), metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="unet", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    device = get_device()
    log.info(f"Device: {device}")

    # ---- Seed ----
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Determine seq_len ----
    seq_len = getattr(cfg.data, "seq_len", 1)
    seq_mode = seq_len > 1

    # ---- Data loaders ----
    train_transform = build_train_transforms(
        hflip_prob=cfg.augmentation.hflip_prob,
        vflip_prob=cfg.augmentation.vflip_prob,
        rot90_prob=cfg.augmentation.rot90_prob,
        noise_std=cfg.augmentation.noise_std,
    ) if getattr(cfg, "augmentation", {}).get("enabled", True) else None

    loaders = make_dataloaders(
        splits_dir=Path(cfg.data.splits_dir),
        patches_dir=Path(cfg.data.patches_dir),
        stats_path=Path(cfg.data.stats_dir) / "channel_stats.json",
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        seq_len=seq_len,
        train_transform=train_transform,
        eval_transform=build_eval_transforms(),
    )

    if "train" not in loaders:
        log.error("Training split not found. Run scripts/assemble_patches.py first.")
        return

    # ---- Model + loss ----
    model = build_model(cfg).to(device)

    # Estimate pos_weight from training split metadata
    _, pos_weight = loaders["train"].dataset.pixel_class_weights() if hasattr(
        loaders["train"].dataset, "pixel_class_weights"
    ) else (1.0, 10.0)

    criterion = CombinedLoss(
        pos_weight=pos_weight,
        dice_weight=cfg.training.dice_weight,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    # ---- MLflow ----
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    run_name = f"{cfg.model.name}-{datetime.date.today()}-{git_sha}"
    if seq_mode:
        run_name += f"-T{seq_len}"

    # ---- Training loop ----
    best_val_auc_pr = -1.0
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_auc_prs: list[float] = []

    ckpt_dir = Path("experiments") / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{run_name}_best.pt"

    with mlflow.start_run(run_name=run_name):
        # Log config
        mlflow.log_params({
            "model": cfg.model.name,
            "seq_len": seq_len,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.learning_rate,
            "dice_weight": cfg.training.dice_weight,
            "pos_weight": round(pos_weight, 2),
            "device": str(device),
            "git_sha": git_sha,
        })

        for epoch in range(1, cfg.training.epochs + 1):
            train_loss = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, seq_mode
            )
            scheduler.step()

            if "val" in loaders:
                val_loss, val_metrics = evaluate_epoch(
                    model, loaders["val"], criterion, device
                )
                val_auc_pr = val_metrics["auc_pr"]
            else:
                val_loss, val_metrics, val_auc_pr = train_loss, {}, 0.0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_auc_prs.append(val_auc_pr)

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            log.info(
                f"Epoch {epoch:03d}/{cfg.training.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_auc_pr={val_auc_pr:.4f}"
            )

            # Early stopping + best model checkpoint
            if val_auc_pr > best_val_auc_pr:
                best_val_auc_pr = val_auc_pr
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt)
                log.info(f"  ✓ New best val AUC-PR={val_auc_pr:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= cfg.training.early_stopping_patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        # ---- Final test evaluation ----
        log.info("Loading best checkpoint for test evaluation…")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        if "test" in loaders:
            _, test_metrics = evaluate_epoch(
                model, loaders["test"], criterion, device
            )
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            log.info(
                f"Test results:\n"
                f"  AUC-PR  = {test_metrics.get('auc_pr', float('nan')):.4f}\n"
                f"  AUC-ROC = {test_metrics.get('auc_roc', float('nan')):.4f}\n"
                f"  F1      = {test_metrics.get('f1', float('nan')):.4f}\n"
                f"  Dice    = {test_metrics.get('dice', float('nan')):.4f}\n"
                f"  IoU     = {test_metrics.get('iou', float('nan')):.4f}\n"
                f"  Brier   = {test_metrics.get('brier', float('nan')):.4f}"
            )

        # ---- Artefacts ----
        mlflow.log_artifact(str(best_ckpt))

        fig = plot_training_curves(train_losses, val_losses, val_auc_prs, title=run_name)
        curve_path = ckpt_dir / f"{run_name}_training_curves.png"
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(curve_path))
        plt.close(fig)

    log.info(f"Run complete. View in MLflow: mlflow ui --backend-store-uri {cfg.mlflow.tracking_uri}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()
