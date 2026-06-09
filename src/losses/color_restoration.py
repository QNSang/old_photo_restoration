from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_kornia() -> Any:
    try:
        import kornia
    except ImportError as exc:
        raise ImportError("Color restoration requires kornia. Install it with `pip install kornia`.") from exc
    return kornia


def normalized_to_rgb01(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp((image + 1.0) * 0.5, 0.0, 1.0)


def reconstruct_lab_ab(input_rgb_normalized: torch.Tensor, predicted_ab_normalized: torch.Tensor) -> torch.Tensor:
    kornia = require_kornia()
    input_lab = kornia.color.rgb_to_lab(normalized_to_rgb01(input_rgb_normalized))
    predicted_ab = torch.clamp(predicted_ab_normalized, -1.0, 1.0) * 128.0
    output_lab = torch.cat([input_lab[:, :1], predicted_ab], dim=1)
    output_rgb = torch.clamp(kornia.color.lab_to_rgb(output_lab), 0.0, 1.0)
    return output_rgb * 2.0 - 1.0


class ColorRestorationLoss(nn.Module):
    def __init__(self, rgb_weight: float = 1.0, ab_weight: float = 1.0, ssim_weight: float = 0.2) -> None:
        super().__init__()
        self.rgb_weight = float(rgb_weight)
        self.ab_weight = float(ab_weight)
        self.ssim_weight = float(ssim_weight)

    def forward(self, pred_rgb_normalized: torch.Tensor, target_rgb_normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        kornia = require_kornia()
        pred_rgb = normalized_to_rgb01(pred_rgb_normalized)
        target_rgb = normalized_to_rgb01(target_rgb_normalized)
        rgb_l1 = F.l1_loss(pred_rgb, target_rgb)
        pred_ab = kornia.color.rgb_to_lab(pred_rgb)[:, 1:] / 128.0
        target_ab = kornia.color.rgb_to_lab(target_rgb)[:, 1:] / 128.0
        ab_l1 = F.l1_loss(pred_ab, target_ab)
        ssim = kornia.losses.ssim_loss(pred_rgb, target_rgb, window_size=11, reduction="mean")
        total = self.rgb_weight * rgb_l1 + self.ab_weight * ab_l1 + self.ssim_weight * ssim
        return {"total": total, "rgb_l1": rgb_l1, "ab_l1": ab_l1, "ssim": ssim}
