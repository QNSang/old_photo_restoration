from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


VALID_COLOR_MODEL_MODES = {"rgb_residual", "lab_ab"}


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


class ColorConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ColorUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ColorConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class ColorRestorationUNet(nn.Module):
    def __init__(
        self,
        mode: str = "lab_ab",
        base_channels: int = 64,
        residual_scale_init: float = 0.5,
        residual_scale_min: float = 0.1,
        residual_scale_max: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in VALID_COLOR_MODEL_MODES:
            raise ValueError(f"Unsupported color model mode: {mode}")
        if not residual_scale_min < residual_scale_init < residual_scale_max:
            raise ValueError("residual_scale_init must be strictly between min and max")
        self.mode = mode
        self.base_channels = int(base_channels)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_min = float(residual_scale_min)
        self.residual_scale_max = float(residual_scale_max)

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoder1 = ColorConvBlock(3, channels[0])
        self.encoder2 = ColorConvBlock(channels[0], channels[1])
        self.encoder3 = ColorConvBlock(channels[1], channels[2])
        self.encoder4 = ColorConvBlock(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ColorConvBlock(channels[3], channels[3] * 2)
        self.decoder4 = ColorUpBlock(channels[3] * 2, channels[3], channels[3])
        self.decoder3 = ColorUpBlock(channels[3], channels[2], channels[2])
        self.decoder2 = ColorUpBlock(channels[2], channels[1], channels[1])
        self.decoder1 = ColorUpBlock(channels[1], channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], 3 if mode == "rgb_residual" else 2, 1)

        normalized_init = (residual_scale_init - residual_scale_min) / (residual_scale_max - residual_scale_min)
        self.residual_scale_raw = nn.Parameter(torch.tensor(_logit(normalized_init), dtype=torch.float32))

    @property
    def residual_scale(self) -> torch.Tensor:
        span = self.residual_scale_max - self.residual_scale_min
        return self.residual_scale_min + span * torch.sigmoid(self.residual_scale_raw)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        x = self.bottleneck(self.pool(skip4))
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return torch.tanh(self.head(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(x)
        if self.mode == "rgb_residual":
            return torch.clamp(x + self.residual_scale * raw, -1.0, 1.0)
        return raw

    def get_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "base_channels": self.base_channels,
            "residual_scale_init": self.residual_scale_init,
            "residual_scale_min": self.residual_scale_min,
            "residual_scale_max": self.residual_scale_max,
        }
