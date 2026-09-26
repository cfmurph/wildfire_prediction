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
    "slope",
    "aspect",
    "wind_dir",
    "wind_speed",
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
    # Physics prior from BurnP3+ (Canadian Forest Service)
    # Zero-filled when no BurnP3+ run available; ablation measures its value
    "burnp3_burn_prob",
]

N_CHANNELS: int = len(CHANNELS)          # 19
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
    in_channels: int = N_CHANNELS   # 19 (18 obs + burnp3_burn_prob)
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
class BurnP3Config:
    """
    BurnP3+ physics simulation integration (Phase 2).

    BurnP3+ is the CFS operational fire spread model. We integrate it as a
    physics prior — the ML model learns the residual correction on top of
    BurnP3+ burn probability outputs.

    Phase 1: burnp3_burn_prob channel is zero-filled (no BurnP3+ install needed)
    Phase 2: pysyncrosim drives live BurnP3+ scenarios; results fill the channel

    Reference: https://burnp3.github.io/BurnP3Plus/
    Discord:   https://discord.gg/76QzY8eAYr
    Contact:   info@burnp3plus.ca
    """
    enabled: bool = False                   # True in Phase 2
    syncrosim_version: str = "3.0.9"
    fire_growth_model: str = "FireSTARR"    # or "Prometheus"
    n_iterations: int = 100                 # Monte Carlo draws per scenario
    weather_uncertainty: float = 0.1       # fractional perturbation for stochastic weather
    output_channel: str = "burnp3_burn_prob"


@dataclass
class MLflowConfig:
    experiment_name: str = "wildfire-phase1"
    tracking_uri: str = "experiments/mlruns"


@dataclass
class GrokConfig:
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-3"
    max_tokens: int = 2048
    temperature: float = 0.2
    timeout_seconds: int = 30
    situation_report_prompt: str = "prompts/situation_report.txt"
