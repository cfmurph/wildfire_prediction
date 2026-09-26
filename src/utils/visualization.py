"""
Visualization utilities for Canadian wildfire spread prediction.

Produces:
  - Spatial confusion maps (TP/FP/FN/TN overlaid on elevation/terrain)
  - Probability heatmaps
  - Precision-Recall curves
  - ROC curves
  - Calibration (reliability diagram) plots
  - Feature importance bar charts
  - Fire season coverage maps (folium, for EDA)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Colour scheme for spatial confusion maps
# ---------------------------------------------------------------------------

# RGB colour for each outcome
_CONFUSION_COLOURS = {
    "tp": (1.00, 0.20, 0.10),   # red — hit: model predicted burn, did burn
    "fp": (1.00, 0.65, 0.00),   # orange — false alarm: predicted burn, didn't
    "fn": (0.00, 0.30, 0.80),   # blue — miss: didn't predict, did burn
    "tn": (0.90, 0.90, 0.90),   # light grey — correct no-burn
}


def spatial_confusion_map(
    y_true: np.ndarray,       # (H, W) float32 binary
    y_prob: np.ndarray,       # (H, W) float32 in [0,1]
    elevation: Optional[np.ndarray] = None,  # (H, W) for hillshade background
    threshold: float = 0.5,
    title: str = "Spatial Confusion Map",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a spatial confusion map overlaid on terrain.

    Colours:
      Red    — True Positive  (correct burn prediction)
      Orange — False Positive (spurious burn prediction)
      Blue   — False Negative (missed burn)
      Grey   — True Negative  (correct non-burn)

    Parameters
    ----------
    y_true : np.ndarray (H, W)
    y_prob : np.ndarray (H, W)
    elevation : np.ndarray (H, W), optional
        Used for hillshade background. If None, plain white background.
    """
    y_pred = (y_prob >= threshold).astype(np.float32)

    tp = (y_pred == 1) & (y_true == 1)
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    tn = (y_pred == 0) & (y_true == 0)

    H, W = y_true.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    for mask, key in [(tn, "tn"), (fp, "fp"), (fn, "fn"), (tp, "tp")]:
        colour = np.array(_CONFUSION_COLOURS[key])
        rgb[mask] = colour

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: confusion map (optionally with hillshade)
    ax = axes[0]
    if elevation is not None:
        elev_norm = (elevation - elevation.min()) / (elevation.ptp() + 1e-6)
        ax.imshow(elev_norm, cmap="gray", alpha=0.35, origin="upper")
    ax.imshow(rgb, alpha=0.8, origin="upper")
    ax.set_title(title)
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=_CONFUSION_COLOURS["tp"], label=f"True Positive  (n={tp.sum():,})"),
        mpatches.Patch(color=_CONFUSION_COLOURS["fp"], label=f"False Positive (n={fp.sum():,})"),
        mpatches.Patch(color=_CONFUSION_COLOURS["fn"], label=f"False Negative (n={fn.sum():,})"),
        mpatches.Patch(color=_CONFUSION_COLOURS["tn"], label=f"True Negative  (n={tn.sum():,})"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8)

    # Right: probability heatmap
    ax2 = axes[1]
    im = ax2.imshow(y_prob, cmap="hot", vmin=0, vmax=1, origin="upper")
    ax2.contour(y_true, levels=[0.5], colors="cyan", linewidths=1.5)
    ax2.set_title("Burn Probability (cyan = ground truth perimeter)")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="P(burn)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    auc_scores: dict[str, float],
    title: str = "Precision-Recall Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot precision-recall curves for multiple models.

    Parameters
    ----------
    curves : dict mapping model_name → (precision_array, recall_array)
    auc_scores : dict mapping model_name → AUC-PR scalar
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, (precision, recall) in curves.items():
        auc = auc_scores.get(model_name, float("nan"))
        ax.plot(recall, precision, label=f"{model_name} (AUC-PR={auc:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    auc_scores: dict[str, float],
    title: str = "ROC Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    curves : dict mapping model_name → (fpr_array, tpr_array)
    auc_scores : dict mapping model_name → AUC-ROC scalar
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (fpr, tpr) in curves.items():
        auc = auc_scores.get(model_name, float("nan"))
        ax.plot(fpr, tpr, label=f"{model_name} (AUC-ROC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration_curve(
    mean_pred: np.ndarray,
    frac_pos: np.ndarray,
    model_name: str = "Model",
    title: str = "Calibration (Reliability Diagram)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a reliability diagram for probability calibration assessment.

    A perfectly calibrated model lies on the diagonal y=x.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", label=model_name, linewidth=2, markersize=6)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance: dict[str, float],
    top_n: int = 15,
    title: str = "Random Forest Feature Importance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances, sorted descending."""
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 0.5 * top_n + 1))
    bars = ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_auc_prs: list[float],
    title: str = "Training Progress",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot loss curves and validation AUC-PR across epochs."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_auc_prs, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val AUC-PR")
    ax2.set_title(f"{title} — Validation AUC-PR")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def bc_fire_coverage_map(
    manifest_records: list[dict],
    save_path: Optional[Path] = None,
) -> "folium.Map":
    """
    Interactive folium map showing patch locations coloured by year.

    Parameters
    ----------
    manifest_records : list of PatchMeta dicts with centre_lon, centre_lat, year
    save_path : Path, optional — saves HTML file
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        raise ImportError("folium is required for map output: pip install folium")

    # Centre on BC
    m = folium.Map(location=[54.0, -125.0], zoom_start=6, tiles="CartoDB positron")

    year_colours = {
        2012: "#1f77b4", 2013: "#ff7f0e", 2014: "#2ca02c", 2015: "#d62728",
        2016: "#9467bd", 2017: "#8c564b", 2018: "#e377c2", 2019: "#7f7f7f",
        2020: "#bcbd22", 2021: "#17becf", 2022: "#aec7e8", 2023: "#ff0000",
    }

    cluster = MarkerCluster().add_to(m)
    for rec in manifest_records:
        lat = rec.get("centre_lat")
        lon = rec.get("centre_lon")
        year = rec.get("year", 0)
        if lat is None or lon is None:
            continue
        colour = year_colours.get(year, "#888888")
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=colour,
            fill=True,
            fill_opacity=0.7,
            popup=f"{rec.get('fire_id')} | {rec.get('date')} | "
                  f"burn={rec.get('burn_fraction', 0):.1%}",
        ).add_to(cluster)

    if save_path:
        m.save(str(save_path))
    return m
