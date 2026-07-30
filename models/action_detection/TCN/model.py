"""Small causal temporal convolutional network for pose classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.left_padding,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convolution(inputs)
        if self.left_padding:
            outputs = outputs[:, :, : -self.left_padding]
        return outputs


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(inputs) + self.residual(inputs))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        class_count: int,
        channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if class_count < 2:
            raise ValueError("class_count must be at least 2")
        if not channels or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain positive values")

        blocks = []
        current_channels = input_size
        for block_index, output_channels in enumerate(channels):
            blocks.append(
                TemporalResidualBlock(
                    current_channels,
                    int(output_channels),
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
            current_channels = int(output_channels)

        self.temporal_network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(current_channels, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, features]")
        temporal_features = self.temporal_network(inputs.transpose(1, 2))
        return self.classifier(temporal_features[:, :, -1])
