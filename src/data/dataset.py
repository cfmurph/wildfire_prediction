"""
Canadian Wildfire Spread Dataset
==================================
PyTorch Dataset classes for loading assembled 64×64 patch tensors.

Two modes:
  WildfireDataset          — single-frame (X, y) for RF and U-Net
  WildfireSequenceDataset  — temporal sequence (X_seq, y) for ConvLSTM
                             where X_seq has shape (T, C, H, W)

Both classes:
  - Load patches from data/processed/patches/*.npz
  - Apply per-channel z-score normalisation using pre-computed train stats
  - Support pluggable transform pipelines (augmentation during training)
  - Return float32 tensors on CPU; DataLoader handles device placement

Channel order follows CHANNELS in src/utils/config.py (19 channels total):
  elevation, slope, aspect, wind_dir, wind_speed, temp_min, temp_max,
  humidity, precip, FFMC, DMC, DC, ISI, BUI, FWI, NDVI, landcover,
  PrevFireMask, burnp3_burn_prob
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import CHANNELS, N_CHANNELS, PATCH_SIZE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel statistics loader
# ---------------------------------------------------------------------------

class ChannelStats:
    """
    Loads per-channel mean and std from data/processed/stats/channel_stats.json
    and provides z-score normalisation helpers.

    If the stats file does not exist (data not yet assembled), falls back to
    identity transform (mean=0, std=1) so the Dataset can still be instantiated
    for testing with synthetic data.
    """

    def __init__(self, stats_path: Path):
        stats_path = Path(stats_path)
        if stats_path.exists():
            with open(stats_path) as f:
                raw = json.load(f)
            self.mean = np.array(
                [raw.get(ch, {}).get("mean", 0.0) for ch in CHANNELS], dtype=np.float32
            )
            self.std = np.array(
                [max(raw.get(ch, {}).get("std", 1.0), 1e-6) for ch in CHANNELS], dtype=np.float32
            )
        else:
            log.warning(
                f"Channel stats not found at {stats_path}. "
                "Using identity normalisation (mean=0, std=1). "
                "Run scripts/assemble_patches.py to generate stats."
            )
            self.mean = np.zeros(N_CHANNELS, dtype=np.float32)
            self.std = np.ones(N_CHANNELS, dtype=np.float32)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Z-score normalise a (C, H, W) array using per-channel stats.

        Parameters
        ----------
        X : np.ndarray, shape (C, H, W), dtype float32

        Returns
        -------
        np.ndarray, shape (C, H, W), dtype float32
        """
        mean = self.mean[:, None, None]   # broadcast over H, W
        std = self.std[:, None, None]
        return (X - mean) / std


# ---------------------------------------------------------------------------
# Single-frame Dataset (RF, U-Net)
# ---------------------------------------------------------------------------

class WildfireDataset(Dataset):
    """
    Single-frame wildfire patch dataset.

    Each item is (X, y) where:
      X : torch.Tensor, shape (C, H, W) = (19, 64, 64), float32, normalised
      y : torch.Tensor, shape (H, W) = (64, 64), float32, binary {0, 1}

    Parameters
    ----------
    manifest_path : Path
        JSON manifest file (train.json / val.json / test.json).
        Each entry must have a "patch_id" key.
    patches_dir : Path
        Directory containing .npz patch files.
    stats_path : Path
        Path to channel_stats.json produced by assemble_patches.py.
    transform : callable, optional
        A (X_tensor, y_tensor) → (X_tensor, y_tensor) callable applied
        after normalisation (used for augmentation during training).
    min_burn_pixels : int
        Skip patches with fewer than this many burning pixels in the target.
        Reduces the number of all-zero targets in the dataset.
    """

    def __init__(
        self,
        manifest_path: Path,
        patches_dir: Path,
        stats_path: Path,
        transform: Optional[Callable] = None,
        min_burn_pixels: int = 1,
    ):
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.min_burn_pixels = min_burn_pixels
        self.stats = ChannelStats(stats_path)

        with open(manifest_path) as f:
            records = json.load(f)

        # Filter to patches that exist on disk and have sufficient burn signal
        self.records = []
        for rec in records:
            p = self.patches_dir / f"{rec['patch_id']}.npz"
            if p.exists() and rec.get("n_burn_pixels", 1) >= min_burn_pixels:
                self.records.append(rec)

        log.info(
            f"WildfireDataset loaded: {len(self.records)} patches "
            f"(from {len(records)} in manifest, min_burn={min_burn_pixels})"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        npz = np.load(self.patches_dir / f"{rec['patch_id']}.npz")

        X = npz["X"].astype(np.float32)   # (C, H, W)
        y = npz["y"].astype(np.float32)   # (H, W)

        # Pad or truncate channels to N_CHANNELS in case of version mismatch
        C_actual = X.shape[0]
        if C_actual < N_CHANNELS:
            pad = np.zeros((N_CHANNELS - C_actual, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            X = np.concatenate([X, pad], axis=0)
        elif C_actual > N_CHANNELS:
            X = X[:N_CHANNELS]

        # Normalise
        X = self.stats.normalize(X)

        # Convert to tensors
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        # Augmentation (training only)
        if self.transform is not None:
            X_t, y_t = self.transform(X_t, y_t)

        return X_t, y_t

    def pixel_class_weights(self) -> tuple[float, float]:
        """
        Compute positive / negative pixel counts over the whole dataset
        for weighted loss initialisation.

        Returns (neg_weight, pos_weight) suitable for BCE pos_weight.
        Samples up to 500 patches to estimate quickly.
        """
        n_pos = 0
        n_neg = 0
        sample_size = min(len(self.records), 500)
        indices = np.random.choice(len(self.records), sample_size, replace=False)
        for i in indices:
            rec = self.records[i]
            n_pos += rec.get("n_burn_pixels", 0)
            n_neg += rec.get("n_total_pixels", PATCH_SIZE * PATCH_SIZE) - rec.get("n_burn_pixels", 0)
        pos_weight = n_neg / max(n_pos, 1)
        return 1.0, pos_weight


# ---------------------------------------------------------------------------
# Temporal Sequence Dataset (ConvLSTM)
# ---------------------------------------------------------------------------

class WildfireSequenceDataset(Dataset):
    """
    Temporal sequence dataset for ConvLSTM training.

    Each item is (X_seq, y) where:
      X_seq : torch.Tensor, shape (T, C, H, W), float32, normalised
              T consecutive days of the same fire event
      y     : torch.Tensor, shape (H, W), float32
              Next-day burn mask (day T+1)

    Sequences are constructed by grouping the manifest by fire_id, sorting by
    date, and sliding a window of length seq_len over each fire's timeline.

    Parameters
    ----------
    manifest_path : Path
    patches_dir : Path
    stats_path : Path
    seq_len : int
        Number of input days T (default 5). Ablation sweeps over [1, 3, 5].
    transform : callable, optional
        Applied to each frame in the sequence independently, then jointly
        to all frames + target (spatial transforms only).
    min_burn_pixels : int
    """

    def __init__(
        self,
        manifest_path: Path,
        patches_dir: Path,
        stats_path: Path,
        seq_len: int = 5,
        transform: Optional[Callable] = None,
        min_burn_pixels: int = 1,
    ):
        self.patches_dir = Path(patches_dir)
        self.seq_len = seq_len
        self.transform = transform
        self.stats = ChannelStats(stats_path)

        with open(manifest_path) as f:
            records = json.load(f)

        # Build fire_id → sorted list of records
        from collections import defaultdict
        fire_days: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            p = self.patches_dir / f"{rec['patch_id']}.npz"
            if p.exists() and rec.get("n_burn_pixels", 1) >= min_burn_pixels:
                fire_days[rec["fire_id"]].append(rec)

        for fid in fire_days:
            fire_days[fid].sort(key=lambda r: r["date"])

        # Build sequence windows: (list_of_T_records, target_record)
        self.sequences: list[tuple[list[dict], dict]] = []
        for fid, days in fire_days.items():
            for i in range(len(days) - seq_len):
                input_records = days[i : i + seq_len]
                target_record = days[i + seq_len]
                self.sequences.append((input_records, target_record))

        log.info(
            f"WildfireSequenceDataset loaded: {len(self.sequences)} sequences "
            f"(seq_len={seq_len}, {len(fire_days)} fires)"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_records, target_record = self.sequences[idx]

        # Stack T frames
        frames: list[torch.Tensor] = []
        for rec in input_records:
            npz = np.load(self.patches_dir / f"{rec['patch_id']}.npz")
            X = npz["X"].astype(np.float32)
            C_actual = X.shape[0]
            if C_actual < N_CHANNELS:
                pad = np.zeros((N_CHANNELS - C_actual, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                X = np.concatenate([X, pad], axis=0)
            elif C_actual > N_CHANNELS:
                X = X[:N_CHANNELS]
            X = self.stats.normalize(X)
            frames.append(torch.from_numpy(X))

        # Load target
        npz_t = np.load(self.patches_dir / f"{target_record['patch_id']}.npz")
        y = torch.from_numpy(npz_t["y"].astype(np.float32))

        X_seq = torch.stack(frames, dim=0)  # (T, C, H, W)

        # Apply spatial augmentation consistently across all frames
        if self.transform is not None:
            augmented_frames = []
            # Use first frame + target to sample transform parameters
            # (we apply the same spatial transform to all frames)
            X_seq, y = _apply_sequence_transform(self.transform, X_seq, y)

        return X_seq, y


def _apply_sequence_transform(
    transform: Callable,
    X_seq: torch.Tensor,  # (T, C, H, W)
    y: torch.Tensor,       # (H, W)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a spatial transform consistently across all T frames.

    We pass the first frame through the transform, capturing the random
    state, then replay the same spatial operation on all other frames.
    This ensures all frames are flipped/rotated identically.
    """
    import random as _random

    # Fix random seed for this sample so all frames get the same transform
    seed = _random.randint(0, 2**31)

    frames_out = []
    for t in range(X_seq.shape[0]):
        _random.seed(seed)
        torch.manual_seed(seed)
        frame_t, _ = transform(X_seq[t], y)
        frames_out.append(frame_t)

    # Apply transform to target using the same seed
    _random.seed(seed)
    torch.manual_seed(seed)
    _, y_out = transform(X_seq[0], y)

    return torch.stack(frames_out, dim=0), y_out


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    splits_dir: Path,
    patches_dir: Path,
    stats_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    seq_len: int = 1,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    min_burn_pixels: int = 1,
) -> dict[str, torch.utils.data.DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    seq_len : int
        1 = single-frame (WildfireDataset), >1 = sequence (WildfireSequenceDataset)
    """
    from torch.utils.data import DataLoader

    splits_dir = Path(splits_dir)
    DatasetClass = WildfireDataset if seq_len == 1 else WildfireSequenceDataset

    def _kwargs(split: str) -> dict:
        base = dict(
            manifest_path=splits_dir / f"{split}.json",
            patches_dir=patches_dir,
            stats_path=stats_path,
            transform=train_transform if split == "train" else eval_transform,
            min_burn_pixels=min_burn_pixels,
        )
        if seq_len > 1:
            base["seq_len"] = seq_len
        return base

    loaders = {}
    for split in ["train", "val", "test"]:
        manifest = splits_dir / f"{split}.json"
        if not manifest.exists():
            log.warning(f"Manifest not found: {manifest} — skipping {split} split")
            continue
        ds = DatasetClass(**_kwargs(split))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders
