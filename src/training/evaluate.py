"""
Evaluation and temporal hold-out analysis.

Handles Phase 1.4: measure year-over-year generalization gap.
Tests the 2023 fire season hold-out (Donnie Creek, etc.) and
reports per-year performance breakdown to surface domain shift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.metrics import compute_metrics, compare_models
from src.utils.visualization import spatial_confusion_map, plot_pr_curves, plot_calibration_curve
from src.utils.metrics import precision_recall_data, calibration_data

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-patch evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_manifest(
    model: nn.Module,
    manifest_path: Path,
    patches_dir: Path,
    stats_path: Path,
    device: torch.device,
    batch_size: int = 32,
    seq_len: int = 1,
) -> tuple[dict[str, float], list[dict]]:
    """
    Evaluate a model on all patches in a manifest.

    Returns
    -------
    metrics : dict of aggregate metrics
    per_patch : list of dicts with per-patch metrics (for domain-shift analysis)
    """
    from src.data.dataset import WildfireDataset, WildfireSequenceDataset, ChannelStats
    from torch.utils.data import DataLoader

    DatasetClass = WildfireDataset if seq_len == 1 else WildfireSequenceDataset
    kwargs = dict(
        manifest_path=manifest_path,
        patches_dir=patches_dir,
        stats_path=stats_path,
        transform=None,
    )
    if seq_len > 1:
        kwargs["seq_len"] = seq_len

    ds = DatasetClass(**kwargs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    per_patch: list[dict] = []

    for i, (X, y) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(X)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        targets = y.numpy()

        all_probs.append(probs)
        all_targets.append(targets)

        # Per-patch metrics
        for j in range(probs.shape[0]):
            patch_idx = i * batch_size + j
            rec = ds.records[patch_idx] if patch_idx < len(ds.records) else {}
            m = compute_metrics(targets[j], probs[j])
            m["year"] = rec.get("year", 0)
            m["fire_id"] = rec.get("fire_id", "")
            m["date"] = rec.get("date", "")
            per_patch.append(m)

    all_probs_np = np.concatenate([p.ravel() for p in all_probs])
    all_targets_np = np.concatenate([t.ravel() for t in all_targets])
    aggregate = compute_metrics(all_targets_np, all_probs_np)

    return aggregate, per_patch


# ---------------------------------------------------------------------------
# Temporal hold-out analysis (Phase 1.4)
# ---------------------------------------------------------------------------

def temporal_generalization_report(
    per_patch_records: list[dict],
    output_path: Optional[Path] = None,
) -> dict[int, dict[str, float]]:
    """
    Break down model performance by year to surface temporal domain shift.

    Specifically surfaces:
      - Year-over-year AUC-PR to identify difficult seasons
      - 2021/2023 mega-fire years vs. normal years
      - Trend in performance (does the model degrade on recent years?)

    Parameters
    ----------
    per_patch_records : list of per-patch metric dicts (from evaluate_on_manifest)
    output_path : Path, optional — saves JSON report

    Returns
    -------
    dict mapping year → aggregate metrics for that year
    """
    from collections import defaultdict

    year_records: dict[int, list[dict]] = defaultdict(list)
    for rec in per_patch_records:
        year = rec.get("year", 0)
        year_records[year].append(rec)

    year_metrics: dict[int, dict[str, float]] = {}
    for year in sorted(year_records.keys()):
        recs = year_records[year]
        # Re-aggregate: for proper AUC-PR we'd need raw probs
        # Here we report mean of per-patch values as a proxy
        metric_names = ["auc_pr", "auc_roc", "f1", "dice", "iou", "brier", "burn_fraction"]
        agg = {}
        for m in metric_names:
            values = [r[m] for r in recs if m in r and not np.isnan(r[m])]
            agg[m] = float(np.mean(values)) if values else float("nan")
        agg["n_patches"] = len(recs)
        year_metrics[year] = agg

        log.info(
            f"  {year}: AUC-PR={agg.get('auc_pr', float('nan')):.4f}  "
            f"F1={agg.get('f1', float('nan')):.4f}  "
            f"n_patches={len(recs)}"
        )

    if output_path:
        with open(output_path, "w") as f:
            json.dump({str(y): v for y, v in year_metrics.items()}, f, indent=2)
        log.info(f"Temporal report saved → {output_path}")

    return year_metrics


# ---------------------------------------------------------------------------
# Full evaluation report generator
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    model: nn.Module,
    model_name: str,
    manifest_path: Path,
    patches_dir: Path,
    stats_path: Path,
    device: torch.device,
    output_dir: Path,
    seq_len: int = 1,
) -> dict[str, float]:
    """
    Generate the full evaluation suite for one model on one split:
      - Aggregate metrics
      - Per-year breakdown (temporal generalization)
      - PR curve data
      - Calibration curve

    Saves artefacts to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Evaluating {model_name} on {manifest_path.stem} split…")
    metrics, per_patch = evaluate_on_manifest(
        model, manifest_path, patches_dir, stats_path, device, seq_len=seq_len
    )

    log.info(
        f"  AUC-PR={metrics.get('auc_pr', float('nan')):.4f}  "
        f"AUC-ROC={metrics.get('auc_roc', float('nan')):.4f}  "
        f"F1={metrics.get('f1', float('nan')):.4f}"
    )

    # Temporal breakdown
    year_report = temporal_generalization_report(
        per_patch,
        output_path=output_dir / f"{model_name}_{manifest_path.stem}_by_year.json",
    )

    # Save aggregate metrics
    with open(output_dir / f"{model_name}_{manifest_path.stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
