#!/usr/bin/env python3
"""
Checkpoint Evaluation
======================
Scores every trained spread model and two baselines on one split, on the
same samples, so a model's number can be read against what it must beat:

  - all-pixel AUC-PR      the metric training reports (rewards copying day T)
  - persistence AUC-PR    predict the day-T fire mask (all pixels)
  - new-burn AUC-PR       only pixels unburned at T
  - distance AUC-PR       new-burn baseline: closer to the fire burns first

Each model gets one row for the whole split and one per season, so a result can be
checked in every test year. Models whose checkpoint is missing are skipped.

Usage:
  python scripts/evaluate_checkpoints.py [--split test] [--unet a.pt b.pt ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.dataset import WildfireDataset, WildfireSequenceDataset
from src.models.convlstm import build_convlstm
from src.models.unet import build_unet
from src.utils.config import CHANNELS, N_CHANNELS
from src.utils.metrics import compute_metrics, compute_spread_metrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

PREV = CHANNELS.index("PrevFireMask")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score spread checkpoints against baselines")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    p.add_argument("--unet", nargs="*", type=Path, help="U-Net checkpoints, e.g. one per seed "
                   "(default: <checkpoints-dir>/unet_best.pt)")
    return p.parse_args()


def raw_patch(patches_dir: Path, rec: dict) -> np.lib.npyio.NpzFile:
    return np.load(patches_dir / f"{rec['patch_id']}.npz")


@torch.no_grad()
def frame_models(args, manifest, patches_dir, stats) -> list[tuple]:
    ds = WildfireDataset(manifest, patches_dir, stats)
    ckpt = Path(args.checkpoints_dir)
    raws = [raw_patch(patches_dir, rec) for rec in ds.records]
    y = np.stack([r["y"] for r in raws])
    prev = np.stack([r["X"][PREV] for r in raws])
    years = np.array([rec["year"] for rec in ds.records])
    runs = []

    if (ckpt / "rf_baseline.joblib").exists():
        rf = joblib.load(ckpt / "rf_baseline.joblib")
        rf_prob = np.stack([
            rf.predict_proba(r["X"][:N_CHANNELS].reshape(N_CHANNELS, -1).T)[:, 1].reshape(y.shape[1:])
            for r in raws
        ])
        runs.append(("rf_baseline", y, rf_prob, prev, years))

    cfg = OmegaConf.load("configs/unet.yaml").model
    for path in args.unet if args.unet is not None else [ckpt / "unet_best.pt"]:
        if not path.exists():
            log.warning(f"Skipping missing checkpoint {path}")
            continue
        unet = build_unet(N_CHANNELS, cfg.base_filters, cfg.depth, cfg.dropout)
        unet.load_state_dict(torch.load(path, map_location="cpu"))
        unet.eval()
        prob = np.stack([torch.sigmoid(unet(ds[i][0][None]))[0, 0].numpy() for i in range(len(ds))])
        runs.append((f"unet:{path.stem}", y, prob, prev, years))
    return runs


@torch.no_grad()
def convlstm(args, manifest, patches_dir, stats, seq_len: int):
    ds = WildfireSequenceDataset(manifest, patches_dir, stats, seq_len=seq_len)
    cfg = OmegaConf.load("configs/convlstm.yaml").model
    path = Path(args.checkpoints_dir) / f"convlstm_T{seq_len}_best.pt"
    if not path.exists():
        return None
    model = build_convlstm(N_CHANNELS, list(cfg.hidden_channels), cfg.kernel_size, cfg.dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    y, prob, prev = [], [], []
    for i, (inputs, _target) in enumerate(ds.sequences):
        X_seq, y_i = ds[i]
        y.append(y_i.numpy())
        prob.append(torch.sigmoid(model(X_seq[None]))[0, 0].numpy())
        # "Already burned" = the latest fire mask the model saw
        prev.append(raw_patch(patches_dir, inputs[-1])["X"][PREV])
    years = np.array([target["year"] for _, target in ds.sequences])
    return (f"convlstm_T{seq_len}", np.stack(y), np.stack(prob), np.stack(prev), years)


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    patches_dir = processed / "patches"
    manifest = processed / "splits" / f"{args.split}.json"
    stats = processed / "stats" / "channel_stats.json"

    runs = frame_models(args, manifest, patches_dir, stats)
    runs += [r for t in (1, 5) if (r := convlstm(args, manifest, patches_dir, stats, t))]

    print(f"\nSplit: {args.split}\n")
    print(f"{'model':32s} {'season':>6s} {'n':>5s} {'all-px AUC-PR':>14s} {'persistence':>12s} "
          f"{'new-burn':>9s} {'distance':>9s} {'new rate':>9s}")
    for name, y, prob, prev, years in runs:
        # Already-burned pixels stay burned (a new-burn-only model never learns them)
        prob = np.maximum(prob, prev)
        for season in ["all", *sorted(set(years.tolist()))]:
            sel = slice(None) if season == "all" else years == season
            m = compute_metrics(y[sel], prob[sel])
            s = compute_spread_metrics(y[sel], prob[sel], prev[sel])
            print(f"{name:32s} {season!s:>6s} {len(y[sel]):5d} {m['auc_pr']:14.3f} {s['persistence_auc_pr']:12.3f} "
                  f"{s['new_burn_auc_pr']:9.3f} {s['new_burn_distance_auc_pr']:9.3f} {s['new_burn_rate']:9.4f}")


if __name__ == "__main__":
    main()
