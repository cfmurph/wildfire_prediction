"""
Evaluation metrics for Canadian wildfire spread prediction.

Primary metric:  AUC-PR (Area Under Precision-Recall curve)
  - Preferred over AUC-ROC for imbalanced data (burning pixels ~10–20%)
  - AUC-ROC can be misleadingly high when negatives dominate

Full metric suite:
  - AUC-PR       (sklearn.metrics.average_precision_score)
  - AUC-ROC      (sklearn.metrics.roc_auc_score)
  - F1 @ 0.5     (sklearn.metrics.f1_score)
  - Dice         (2 * |P ∩ G| / (|P| + |G|), hard threshold 0.5)
  - IoU/Jaccard  (|P ∩ G| / |P ∪ G|, hard threshold 0.5)
  - Brier score  (mean squared error of probabilities — calibration)

All functions accept numpy arrays of arbitrary shape (flattened internally).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the full evaluation metric suite.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1), any shape.
    y_prob : np.ndarray
        Predicted probabilities in [0, 1], same shape as y_true.
    threshold : float
        Decision threshold for hard metrics (F1, Dice, IoU).

    Returns
    -------
    dict mapping metric name → float value
    """
    y_true = y_true.ravel().astype(np.float32)
    y_prob = y_prob.ravel().astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    # Guard against degenerate cases (all-zero or all-one targets)
    n_pos = y_true.sum()
    n_neg = (1 - y_true).sum()

    metrics: dict[str, float] = {}

    # --- AUC-PR (primary) ---
    if n_pos > 0 and n_neg > 0:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc_pr"] = float("nan")
        metrics["auc_roc"] = float("nan")

    # --- F1 ---
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # --- Dice ---
    intersection = (y_pred * y_true).sum()
    dice_denom = y_pred.sum() + y_true.sum()
    metrics["dice"] = float(2.0 * intersection / max(dice_denom, 1))

    # --- IoU (Jaccard) ---
    union = (y_pred + y_true).clip(0, 1).sum()
    metrics["iou"] = float(intersection / max(union, 1))

    # --- Brier score (calibration) ---
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    # --- Burn fraction in target ---
    metrics["burn_fraction"] = float(n_pos / max(len(y_true), 1))

    return metrics


def compute_batch_metrics(
    y_true_batch: np.ndarray,  # (B, H, W)
    y_prob_batch: np.ndarray,  # (B, H, W)
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute metrics over a batch of patches.
    Metrics are computed on the full concatenated arrays, not averaged per-patch.
    """
    return compute_metrics(y_true_batch, y_prob_batch, threshold=threshold)


def precision_recall_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true.ravel(), y_prob.ravel())


def roc_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true.ravel(), y_prob.ravel())


def calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data.

    Returns (mean_predicted_prob, fraction_positive) per bin,
    for use in calibration curve plots.
    """
    y_true = y_true.ravel()
    y_prob = y_prob.ravel()

    bins = np.linspace(0, 1, n_bins + 1)
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            mean_pred[i] = y_prob[mask].mean()
            frac_pos[i] = y_true[mask].mean()
        else:
            mean_pred[i] = (bins[i] + bins[i + 1]) / 2
            frac_pos[i] = float("nan")

    return mean_pred, frac_pos


def compare_models(results: dict[str, dict[str, float]]) -> str:
    """
    Format a metric comparison table across multiple models.

    Parameters
    ----------
    results : dict mapping model_name → metrics dict

    Returns
    -------
    str — formatted ASCII table
    """
    metrics_to_show = ["auc_pr", "auc_roc", "f1", "dice", "iou", "brier"]
    header = f"{'Model':<20}" + "".join(f"{m:>10}" for m in metrics_to_show)
    sep = "-" * len(header)
    rows = [header, sep]

    for model_name, metrics in results.items():
        row = f"{model_name:<20}"
        for m in metrics_to_show:
            val = metrics.get(m, float("nan"))
            row += f"{val:>10.4f}"
        rows.append(row)

    return "\n".join(rows)
