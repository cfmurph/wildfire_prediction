#!/usr/bin/env python3
"""
Which inputs a trained U-Net relies on for new burn.

Shuffles one group of input channels between patches at a time and measures how much
new-burn AUC-PR drops. A big drop means the model uses that input; none means it
ignores it. No retraining: one checkpoint, one split.

Usage:
  python scripts/channel_importance.py --checkpoint experiments/checkpoints/unet_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.dataset import WildfireDataset
from src.models.unet import build_unet
from src.utils.config import CHANNELS, N_CHANNELS
from src.utils.metrics import compute_spread_metrics

GROUPS = {
    "terrain": ["elevation", "slope_east", "slope_north"],
    "wind": ["wind_u", "wind_v"],
    "weather": ["temp_min", "temp_max", "humidity", "precip"],
    "fire weather (FWI system)": ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"],
    "vegetation": ["NDVI", "landcover"],
    "years since fire": ["years_since_fire"],
    "current fire mask": ["PrevFireMask"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-patches", type=int, default=2000, help="random sample, to bound memory and time")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def new_burn_auc_pr(model, X: torch.Tensor, y: np.ndarray, prev: np.ndarray) -> float:
    prob = np.concatenate([torch.sigmoid(model(X[i : i + 64]))[:, 0].numpy() for i in range(0, len(X), 64)])
    return compute_spread_metrics(y, prob, prev)["new_burn_auc_pr"]


def main() -> None:
    args = parse_args()
    p = args.processed_dir
    ds = WildfireDataset(p / "splits" / f"{args.split}.json", p / "patches", p / "stats" / "channel_stats.json")
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(ds), size=min(args.max_patches, len(ds)), replace=False)
    X = torch.stack([ds[i][0] for i in pick])
    y = np.stack([ds[i][1].numpy() for i in pick])
    raw = [np.load(p / "patches" / f"{ds.records[i]['patch_id']}.npz") for i in pick]
    # "Already burned" comes from the real mask, so shuffling the mask input doesn't move the scored pixels
    prev = np.stack([r["X"][CHANNELS.index("PrevFireMask")] for r in raw])

    cfg = OmegaConf.load("configs/unet.yaml").model
    model = build_unet(N_CHANNELS, cfg.base_filters, cfg.depth, cfg.dropout)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    base = new_burn_auc_pr(model, X, y, prev)
    print(f"\n{args.checkpoint.name}, {args.split}, {len(pick)} patches: new-burn AUC-PR {base:.4f}\n")
    print(f"{'shuffled input':28s} {'new-burn':>9s} {'drop':>8s}")
    for name, channels in GROUPS.items():
        idx = [CHANNELS.index(c) for c in channels]
        shuffled = X.clone()
        shuffled[:, idx] = X[torch.from_numpy(rng.permutation(len(X)))][:, idx]
        score = new_burn_auc_pr(model, shuffled, y, prev)
        print(f"{name:28s} {score:9.4f} {base - score:+8.4f}")


if __name__ == "__main__":
    main()
