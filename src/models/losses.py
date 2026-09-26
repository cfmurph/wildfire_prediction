"""
Loss functions for wildfire spread prediction.

The primary loss is a weighted combination of:
  - Binary Cross-Entropy with class-weighted pos_weight (handles imbalance)
  - Dice Loss (directly optimises spatial overlap between prediction and target)

Combined: L = dice_weight * Dice + (1 - dice_weight) * BCE

Why both?
  BCE alone: well-calibrated probabilities, but insensitive to spatial overlap
  Dice alone: maximises overlap but can produce poorly calibrated probabilities
  Combined: retains calibration benefit of BCE while encouraging spatial precision
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - Dice

    The "soft" version uses sigmoid probabilities (not hard thresholds),
    making it fully differentiable.

    Parameters
    ----------
    smooth : float
        Laplace smoothing to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape (B, 1, H, W) or (B, H, W) — raw model outputs
        targets : Tensor, shape (B, H, W) — binary {0, 1}
        """
        probs = torch.sigmoid(logits)
        if probs.dim() == 4:
            probs = probs.squeeze(1)    # (B, H, W)

        # Flatten spatial dims
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of BCE + Dice losses.

    L = dice_weight * Dice + (1 - dice_weight) * WeightedBCE

    Parameters
    ----------
    pos_weight : float
        Weight for positive (burning) pixels in BCE.
        Set to (n_neg / n_pos) to handle class imbalance.
    dice_weight : float
        Weight of Dice loss in the combination (0–1).
    smooth : float
        Dice smoothing factor.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.register_buffer(
            "pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape (B, 1, H, W) or (B, H, W)
        targets : Tensor, shape (B, H, W), float32, values in {0, 1}
        """
        if logits.dim() == 4:
            logits_bce = logits.squeeze(1)
        else:
            logits_bce = logits

        bce = F.binary_cross_entropy_with_logits(
            logits_bce,
            targets,
            pos_weight=self.pos_weight_tensor.to(logits.device),
        )
        dice = self.dice_loss(logits, targets)

        return self.dice_weight * dice + (1.0 - self.dice_weight) * bce
