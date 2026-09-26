"""
ConvLSTM for Spatiotemporal Wildfire Spread Prediction
=======================================================
Processes a sequence of T daily fire-state patches to predict next-day spread.

Architecture:
  Input:     (B, T, C, H, W)  — T frames of C=19 channels at 64×64
  ConvLSTM:  2 stacked ConvLSTM layers (hidden_channels=[64, 64])
  Decoder:   U-Net-style decoder on the final hidden state
  Output:    (B, 1, H, W)  — next-day burn probability logit map

Research hypothesis: explicitly modelling the trajectory of fire spread over
T=5 days improves next-day prediction vs a single-frame U-Net.

Ablation: train with seq_len ∈ {1, 3, 5} to measure temporal benefit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import ConvBlock, DecoderBlock


# ---------------------------------------------------------------------------
# ConvLSTM Cell
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Replaces the fully-connected gates in standard LSTM with convolutional
    operations, preserving spatial structure across time steps.

    Parameters
    ----------
    in_channels : int
        Channels of the input at this step.
    hidden_channels : int
        Number of hidden/cell state channels.
    kernel_size : int
        Convolution kernel size for gate operations (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Single conv computes all 4 gates at once: [i, f, g, o]
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, in_channels, H, W) — input at time t
        h : Tensor (B, hidden_channels, H, W) — previous hidden state
        c : Tensor (B, hidden_channels, H, W) — previous cell state

        Returns
        -------
        h_new : Tensor (B, hidden_channels, H, W)
        c_new : Tensor (B, hidden_channels, H, W)
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


# ---------------------------------------------------------------------------
# Stacked ConvLSTM encoder
# ---------------------------------------------------------------------------

class ConvLSTMEncoder(nn.Module):
    """
    Stack of ConvLSTM cells that processes a T-step sequence.

    Returns the final hidden state of the last layer, which encodes
    the spatiotemporal evolution of the fire over T days.

    Parameters
    ----------
    in_channels : int
    hidden_channels : list[int]
        Channel widths for each ConvLSTM layer.
    kernel_size : int
    dropout : float
        Applied between ConvLSTM layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cells = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        c_in = in_channels
        for c_hid in hidden_channels:
            self.cells.append(ConvLSTMCell(c_in, c_hid, kernel_size=kernel_size))
            self.dropouts.append(nn.Dropout2d(p=dropout))
            c_in = c_hid
        self.out_channels = hidden_channels[-1]

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq : Tensor (B, T, C, H, W)

        Returns
        -------
        h_final : Tensor (B, hidden_channels[-1], H, W)
            Final hidden state of the last ConvLSTM layer after processing all T steps.
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        h_states = []
        c_states = []
        for cell in self.cells:
            h, c = cell.init_hidden(B, H, W, device)
            h_states.append(h)
            c_states.append(c)

        current_input = x_seq  # (B, T, C, H, W)

        for layer_idx, (cell, drop) in enumerate(zip(self.cells, self.dropouts)):
            layer_outputs = []
            h, c = h_states[layer_idx], c_states[layer_idx]
            for t in range(T):
                h, c = cell(current_input[:, t], h, c)
                layer_outputs.append(h)
            # Stack time steps; apply dropout on the output sequence
            layer_out = torch.stack(layer_outputs, dim=1)  # (B, T, hid, H, W)
            # Apply dropout per-frame
            B_, T_, hid, H_, W_ = layer_out.shape
            layer_out = drop(layer_out.view(B_ * T_, hid, H_, W_)).view(B_, T_, hid, H_, W_)
            current_input = layer_out

        # Return final time step of last layer
        return current_input[:, -1]  # (B, hidden_channels[-1], H, W)


# ---------------------------------------------------------------------------
# ConvLSTM wildfire model
# ---------------------------------------------------------------------------

class ConvLSTMWildfire(nn.Module):
    """
    Spatiotemporal wildfire spread model.

    Encodes T days of history with stacked ConvLSTM, then decodes
    the final hidden state to a next-day burn probability map using
    a lightweight U-Net decoder.

    Parameters
    ----------
    in_channels : int
        Input feature channels (default 19).
    hidden_channels : list[int]
        ConvLSTM hidden channel widths (default [64, 64]).
    kernel_size : int
        ConvLSTM gate kernel size (default 3).
    dropout : float
    decoder_filters : int
        Base filter count for the decoder ConvBlocks (default 64).
    """

    def __init__(
        self,
        in_channels: int = 19,
        hidden_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        decoder_filters: int = 64,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 64]

        # Spatiotemporal encoder
        self.encoder = ConvLSTMEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        enc_out_ch = hidden_channels[-1]

        # Lightweight decoder: two ConvBlocks + prediction head
        # (No skip connections from encoder — the LSTM state carries spatial info)
        self.decoder = nn.Sequential(
            ConvBlock(enc_out_ch, decoder_filters, dropout=dropout),
            ConvBlock(decoder_filters, decoder_filters // 2, dropout=dropout),
        )
        self.head = nn.Conv2d(decoder_filters // 2, 1, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq : Tensor (B, T, C, H, W)
            T-step history of fire-state patches, normalised.

        Returns
        -------
        logits : Tensor (B, 1, H, W)
        """
        h = self.encoder(x_seq)        # (B, enc_out_ch, H, W)
        decoded = self.decoder(h)      # (B, decoder_filters//2, H, W)
        return self.head(decoded)      # (B, 1, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_convlstm(
    in_channels: int = 19,
    hidden_channels: list[int] = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
    decoder_filters: int = 64,
) -> ConvLSTMWildfire:
    if hidden_channels is None:
        hidden_channels = [64, 64]

    model = ConvLSTMWildfire(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        decoder_filters=decoder_filters,
    )
    import logging
    logging.getLogger(__name__).info(
        f"ConvLSTMWildfire built: hidden={hidden_channels}, "
        f"params={model.count_parameters():,}"
    )
    return model
