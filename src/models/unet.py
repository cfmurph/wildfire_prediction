"""
U-Net CNN for Canadian Wildfire Spread Prediction
===================================================
4-level encoder-decoder with skip connections.

Input:  (B, C, 64, 64)  — C=19 channels
Output: (B, 1, 64, 64)  — logit map (apply sigmoid for probabilities)

Architecture:
  Encoder: 4× [Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2×2]
  Bottleneck: [Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU]
  Decoder: 4× [ConvTranspose2×2 → cat(skip) → Conv3×3 → BN → ReLU × 2]
  Head:    Conv1×1 → single logit channel

Filter widths: base_filters × [1, 2, 4, 8] = [32, 64, 128, 256] (default)

Designed to run on Apple Silicon MPS or CPU without modifications.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock followed by MaxPool2d. Returns (pooled, skip)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """ConvTranspose2d upsampling + skip connection + ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch (e.g. odd spatial dims) by centre-cropping skip
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            skip = skip[
                :,
                :,
                diff_h // 2 : skip.shape[2] - (diff_h - diff_h // 2),
                diff_w // 2 : skip.shape[3] - (diff_w - diff_w // 2),
            ]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    4-level U-Net for binary wildfire spread segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (default 19).
    base_filters : int
        Filter count at the first encoder level. Doubles at each level.
    depth : int
        Number of encoder/decoder levels (default 4).
    dropout : float
        Dropout probability inside ConvBlocks (applied to feature maps).
    """

    def __init__(
        self,
        in_channels: int = 19,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        c_in = in_channels
        for i in range(depth):
            c_out = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(c_in, c_out, dropout=dropout))
            c_in = c_out

        # Bottleneck
        c_bottleneck = base_filters * (2 ** depth)
        self.bottleneck = ConvBlock(c_in, c_bottleneck, dropout=dropout)

        # Decoder
        self.decoders = nn.ModuleList()
        c_in = c_bottleneck
        for i in range(depth - 1, -1, -1):
            c_skip = base_filters * (2 ** i)
            c_out = c_skip
            self.decoders.append(DecoderBlock(c_in, c_out, dropout=dropout))
            c_in = c_out

        # Output head
        self.head = nn.Conv2d(c_in, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)

        Returns
        -------
        logits : Tensor, shape (B, 1, H, W)
        """
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip)

        return self.head(x)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_unet(
    in_channels: int = 19,
    base_filters: int = 32,
    depth: int = 4,
    dropout: float = 0.2,
) -> UNet:
    model = UNet(
        in_channels=in_channels,
        base_filters=base_filters,
        depth=depth,
        dropout=dropout,
    )
    n_params = model.count_parameters()
    import logging
    logging.getLogger(__name__).info(
        f"UNet built: depth={depth}, base_filters={base_filters}, "
        f"params={n_params:,}"
    )
    return model
