"""
Dataclass configuration schema for the wildfire prediction platform.

Used as a typed wrapper around Hydra/OmegaConf configs to enable
IDE completion, runtime validation, and self-documenting code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Channel constants
# ---------------------------------------------------------------------------

CHANNELS: list[str] = [
    "elevation",
    "slope_east",    # terrain gradient (rise per metre) toward east
    "slope_north",   # ... and toward north
    "wind_u",        # 16:00 PDT wind toward east (m/s)
    "wind_v",        # ... and toward north
    "temp_min",
    "temp_max",
    "humidity",
    "precip",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
    "NDVI",
    "landcover",
    "PrevFireMask",
    # Years since the pixel last burned, from earlier seasons only (capped at 40)
    "years_since_fire",
]

N_CHANNELS: int = len(CHANNELS)          # 19

# (east, north) components; augmentation flips and rotates these with the patch
VECTOR_CHANNEL_PAIRS: list[tuple[str, str]] = [("slope_east", "slope_north"), ("wind_u", "wind_v")]
PATCH_SIZE: int = 64
TARGET_RESOLUTION_M: int = 1000          # 1 km / pixel

# Fire seasons used for each split
TRAIN_YEARS: list[int] = list(range(2012, 2021))   # 2012–2020
VAL_YEARS: list[int] = [2021, 2022]
TEST_YEARS: list[int] = [2023]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    province: str = "BC"
    patch_size: int = PATCH_SIZE
    channels: list[str] = field(default_factory=lambda: list(CHANNELS))
    train_years: list[int] = field(default_factory=lambda: list(TRAIN_YEARS))
    val_years: list[int] = field(default_factory=lambda: list(VAL_YEARS))
    test_years: list[int] = field(default_factory=lambda: list(TEST_YEARS))
    patches_dir: str = "data/processed/patches"
    splits_dir: str = "data/processed/splits"
    stats_dir: str = "data/processed/stats"
    seq_len: int = 1   # >1 enables temporal sequence mode (ConvLSTM)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rot90_prob: float = 0.5
    noise_std: float = 0.01   # applied to all non-mask channels


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    dice_weight: float = 0.5
    early_stopping_patience: int = 8
    num_workers: int = 4
    seed: int = 42
    max_samples_per_split: Optional[int] = None   # for RF subsampling


@dataclass
class UNetConfig:
    name: str = "unet"
    in_channels: int = N_CHANNELS   # 19
    base_filters: int = 32
    depth: int = 4
    dropout: float = 0.2


@dataclass
class ConvLSTMConfig:
    name: str = "convlstm"
    in_channels: int = N_CHANNELS   # 19
    hidden_channels: list[int] = field(default_factory=lambda: [64, 64])
    kernel_size: int = 3
    dropout: float = 0.2


@dataclass
class RFConfig:
    name: str = "baseline_rf"
    n_estimators: int = 500
    max_depth: Optional[int] = None
    min_samples_leaf: int = 4
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class MLflowConfig:
    experiment_name: str = "wildfire-phase1"
    tracking_uri: str = "experiments/mlruns"
