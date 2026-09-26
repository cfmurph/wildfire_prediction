"""
Augmentation transforms for wildfire patch tensors.

All transforms operate on (C, H, W) input tensors and (H, W) target masks.
They are designed to be composable via TransformPipeline and are only
applied during training (not validation or test).

Spatial augmentations (flip, rotate) are applied identically to both
input X and target y to preserve the spatial correspondence.

Channel noise is applied only to non-mask input channels to avoid
corrupting the fire boundary signal.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


# Index of PrevFireMask in the channel list (channel 17, 0-indexed)
# This channel and burnp3_burn_prob (channel 18) are excluded from noise
_MASK_CHANNEL_INDICES = {17, 18}


class RandomHorizontalFlip:
    """Flip both X and y horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            X = torch.flip(X, dims=[-1])
            y = torch.flip(y, dims=[-1])
        return X, y


class RandomVerticalFlip:
    """Flip both X and y vertically with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            X = torch.flip(X, dims=[-2])
            y = torch.flip(y, dims=[-2])
        return X, y


class RandomRot90:
    """Rotate both X and y by a random multiple of 90° with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            X = torch.rot90(X, k=k, dims=[-2, -1])
            y = torch.rot90(y, k=k, dims=[-2, -1])
        return X, y


class GaussianNoise:
    """
    Add Gaussian noise to all non-mask input channels.

    Mask channels (PrevFireMask, burnp3_burn_prob) are excluded to
    avoid corrupting fire boundary and physics prior signals.
    """

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.std <= 0:
            return X, y
        noise = torch.randn_like(X) * self.std
        # Zero out noise on mask channels
        for idx in _MASK_CHANNEL_INDICES:
            if idx < noise.shape[0]:
                noise[idx] = 0.0
        return X + noise, y


class TransformPipeline:
    """
    Compose a sequence of (X, y) → (X, y) transforms.

    Parameters
    ----------
    transforms : list of callables
        Each callable takes (X: Tensor, y: Tensor) and returns (X, y).
    """

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            X, y = t(X, y)
        return X, y


def build_train_transforms(
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rot90_prob: float = 0.5,
    noise_std: float = 0.01,
) -> TransformPipeline:
    """Build the standard training augmentation pipeline."""
    return TransformPipeline([
        RandomHorizontalFlip(p=hflip_prob),
        RandomVerticalFlip(p=vflip_prob),
        RandomRot90(p=rot90_prob),
        GaussianNoise(std=noise_std),
    ])


def build_eval_transforms() -> TransformPipeline:
    """No-op pipeline for validation / test (keeps the same API)."""
    return TransformPipeline([])
