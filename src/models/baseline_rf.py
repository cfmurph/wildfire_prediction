"""
Random Forest Baseline for Canadian Wildfire Spread Prediction
==============================================================
Pixel-level baseline: flatten each 64×64 patch into individual pixel rows
(each with N_CHANNELS features), train a scikit-learn RandomForestClassifier,
and evaluate on the test split.

This is Baseline 1 in the model progression. It:
  - Reproduces the approach from Huot et al. 2022 (NDWS) on our Canadian data
  - Establishes the AUC-PR anchor for the Canadian benchmark (no prior exists)
  - Ignores spatial structure (each pixel classified independently)

The spatial U-Net baseline (Baseline 2) is expected to beat this by exploiting
local context (fire front geometry, terrain continuity, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

from src.utils.config import CHANNELS, N_CHANNELS, PATCH_SIZE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel feature extractor
# ---------------------------------------------------------------------------

def patches_to_pixels(
    patches_dir: Path,
    manifest_path: Path,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load .npz patches and flatten to pixel-level (X, y) arrays.

    Each pixel becomes one row with N_CHANNELS features.
    For a 64×64 patch: 4096 pixel rows per patch.

    Parameters
    ----------
    patches_dir : Path
    manifest_path : Path
        JSON manifest with list of records having "patch_id".
    max_samples : int, optional
        If given, subsample to this many pixels (for RAM efficiency in training).
    seed : int

    Returns
    -------
    X : np.ndarray, shape (N_pixels, N_CHANNELS), float32
    y : np.ndarray, shape (N_pixels,), float32 — binary {0, 1}
    """
    patches_dir = Path(patches_dir)
    with open(manifest_path) as f:
        records = json.load(f)

    Xs, ys = [], []
    for rec in records:
        p = patches_dir / f"{rec['patch_id']}.npz"
        if not p.exists():
            continue
        data = np.load(p)
        X_patch = data["X"].astype(np.float32)   # (C, H, W)
        y_patch = data["y"].astype(np.float32)   # (H, W)

        C = X_patch.shape[0]
        if C < N_CHANNELS:
            pad = np.zeros((N_CHANNELS - C, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            X_patch = np.concatenate([X_patch, pad], axis=0)
        elif C > N_CHANNELS:
            X_patch = X_patch[:N_CHANNELS]

        # (C, H, W) → (H*W, C)
        X_flat = X_patch.reshape(N_CHANNELS, -1).T
        y_flat = y_patch.ravel()

        Xs.append(X_flat)
        ys.append(y_flat)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    if max_samples is not None and len(X_all) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_all), size=max_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    log.info(
        f"Loaded {len(X_all):,} pixels from {len(records)} patches "
        f"(pos={y_all.mean():.3%})"
    )
    return X_all, y_all


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_rf(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 4,
    class_weight: str = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------

def feature_importance_report(model: RandomForestClassifier) -> dict[str, float]:
    """Return channel name → mean decrease impurity importance."""
    importances = model.feature_importances_
    return {ch: float(imp) for ch, imp in zip(CHANNELS, importances)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rf(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a trained RF on pixel-level data.

    Returns dict with AUC-PR, AUC-ROC, F1 (at 0.5 threshold).
    """
    from sklearn.metrics import f1_score

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc_pr": float(average_precision_score(y, probs)),
        "auc_roc": float(roc_auc_score(y, probs)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
    return metrics


# ---------------------------------------------------------------------------
# Train + evaluate entry point
# ---------------------------------------------------------------------------

def train_and_evaluate(
    patches_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 4,
    max_samples: int = 200_000,
    seed: int = 42,
    mlflow_experiment: str = "wildfire-phase1",
    mlflow_tracking_uri: str = "experiments/mlruns",
) -> dict[str, float]:
    """
    Full train → evaluate pipeline for the RF baseline.
    Logs all metrics and the trained model to MLflow.

    Parameters
    ----------
    patches_dir : Path
    splits_dir : Path
        Directory containing train.json, val.json, test.json.
    output_dir : Path
        Where to save the trained model (joblib).
    """
    patches_dir = Path(patches_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    import subprocess, datetime
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    run_name = f"baseline_rf-{datetime.date.today()}-{git_sha}"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model": "RandomForest",
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "min_samples_leaf": min_samples_leaf,
            "max_train_pixels": max_samples,
            "seed": seed,
        })

        # Train
        log.info("Loading training pixels…")
        X_train, y_train = patches_to_pixels(
            patches_dir, splits_dir / "train.json", max_samples=max_samples, seed=seed
        )

        log.info(f"Training RF ({n_estimators} trees)…")
        model = build_rf(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        log.info("Training complete.")

        # Val
        log.info("Evaluating on validation split…")
        X_val, y_val = patches_to_pixels(patches_dir, splits_dir / "val.json")
        val_metrics = evaluate_rf(model, X_val, y_val)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        log.info(f"  Val  AUC-PR={val_metrics['auc_pr']:.4f}  AUC-ROC={val_metrics['auc_roc']:.4f}")

        # Test
        log.info("Evaluating on test split (2023 season)…")
        X_test, y_test = patches_to_pixels(patches_dir, splits_dir / "test.json")
        test_metrics = evaluate_rf(model, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log.info(f"  Test AUC-PR={test_metrics['auc_pr']:.4f}  AUC-ROC={test_metrics['auc_roc']:.4f}")

        # Feature importance
        fi = feature_importance_report(model)
        top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
        log.info(f"  Top-5 features: {top5}")
        mlflow.log_dict(fi, "feature_importance.json")

        # Save model
        model_path = output_dir / "rf_baseline.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        log.info(f"  Model saved → {model_path}")

    return test_metrics
