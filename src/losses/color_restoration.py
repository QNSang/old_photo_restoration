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


def _rgb01_to_lab_float32(image: torch.Tensor) -> torch.Tensor:
    kornia = require_kornia()
    with torch.autocast(device_type=image.device.type, enabled=False):
        return kornia.color.rgb_to_lab(image.float())


def reconstruct_lab_ab(input_rgb_normalized: torch.Tensor, predicted_ab_normalized: torch.Tensor) -> torch.Tensor:
    kornia = require_kornia()
    with torch.autocast(device_type=input_rgb_normalized.device.type, enabled=False):
        input_lab = _rgb01_to_lab_float32(normalized_to_rgb01(input_rgb_normalized))
        predicted_ab = torch.clamp(predicted_ab_normalized.float(), -1.0, 1.0) * 128.0
        output_lab = torch.cat([input_lab[:, :1], predicted_ab], dim=1)
        output_rgb = torch.clamp(kornia.color.lab_to_rgb(output_lab), 0.0, 1.0)
        return output_rgb * 2.0 - 1.0


def reconstruct_lab_residual(
    input_rgb_normalized: torch.Tensor,
    predicted_lab_residual: torch.Tensor,
    *,
    max_l_shift: float = 15.0,
    max_ab_shift: float = 40.0,
) -> torch.Tensor:
    if predicted_lab_residual.shape[1] != 3:
        raise ValueError(f"Lab residual prediction must have 3 channels, got {predicted_lab_residual.shape}")
    kornia = require_kornia()
    with torch.autocast(device_type=input_rgb_normalized.device.type, enabled=False):
        input_lab = _rgb01_to_lab_float32(normalized_to_rgb01(input_rgb_normalized))
        residual = torch.clamp(predicted_lab_residual.float(), -1.0, 1.0)
        output_l = torch.clamp(input_lab[:, :1] + residual[:, :1] * float(max_l_shift), 0.0, 100.0)
        output_ab = torch.clamp(input_lab[:, 1:] + residual[:, 1:] * float(max_ab_shift), -128.0, 127.0)
        output_rgb = torch.clamp(kornia.color.lab_to_rgb(torch.cat([output_l, output_ab], dim=1)), 0.0, 1.0)
        return output_rgb * 2.0 - 1.0


def _soft_histogram(values: torch.Tensor, bins: int) -> torch.Tensor:
    if bins < 2:
        raise ValueError("hist_bins must be >= 2")
    centers = torch.linspace(-1.0, 1.0, bins, device=values.device, dtype=values.dtype)
    bin_width = 2.0 / (bins - 1)
    weights = torch.relu(1.0 - torch.abs(values.unsqueeze(-1) - centers) / bin_width)
    histogram = weights.sum(dim=-2)
    return histogram / histogram.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _sample_ab_pixels(ab: torch.Tensor, max_samples: int) -> torch.Tensor:
    values = ab.flatten(start_dim=2)
    pixel_count = values.shape[-1]
    if max_samples <= 0:
        raise ValueError("hist_max_samples must be > 0")
    if pixel_count <= max_samples:
        return values
    indices = torch.linspace(0, pixel_count - 1, max_samples, device=values.device).long()
    return values.index_select(-1, indices)


def lab_ab_histogram_emd(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    bins: int = 64,
    max_samples: int = 4096,
) -> torch.Tensor:
    with torch.autocast(device_type=pred_rgb.device.type, enabled=False):
        pred_ab = _rgb01_to_lab_float32(pred_rgb)[:, 1:] / 128.0
        pred_values = _sample_ab_pixels(pred_ab.clamp(-1.0, 1.0), max_samples)
        pred_histogram = _soft_histogram(pred_values, bins)
        with torch.no_grad():
            target_ab = _rgb01_to_lab_float32(target_rgb)[:, 1:] / 128.0
            target_values = _sample_ab_pixels(target_ab.clamp(-1.0, 1.0), max_samples)
            target_histogram = _soft_histogram(target_values, bins)
        pred_cdf = pred_histogram.cumsum(dim=-1)
        target_cdf = target_histogram.cumsum(dim=-1)
        return torch.abs(pred_cdf - target_cdf).mean()


class ColorRestorationLoss(nn.Module):
    def __init__(
        self,
        rgb_weight: float = 0.5,
        l_weight: float = 0.5,
        ab_weight: float = 1.0,
        ssim_weight: float = 0.2,
        hist_emd_weight: float = 0.1,
        identity_weight: float = 0.2,
        hist_bins: int = 64,
        hist_max_samples: int = 4096,
    ) -> None:
        super().__init__()
        self.rgb_weight = float(rgb_weight)
        self.l_weight = float(l_weight)
        self.ab_weight = float(ab_weight)
        self.ssim_weight = float(ssim_weight)
        self.hist_emd_weight = float(hist_emd_weight)
        self.identity_weight = float(identity_weight)
        self.hist_bins = int(hist_bins)
        self.hist_max_samples = int(hist_max_samples)
        if self.hist_bins < 2:
            raise ValueError("hist_bins must be >= 2")
        if self.hist_max_samples <= 0:
            raise ValueError("hist_max_samples must be > 0")

    def forward(
        self,
        pred_rgb_normalized: torch.Tensor,
        target_rgb_normalized: torch.Tensor,
        *,
        input_rgb_normalized: torch.Tensor | None = None,
        identity_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        kornia = require_kornia()
        with torch.autocast(device_type=pred_rgb_normalized.device.type, enabled=False):
            pred_rgb = normalized_to_rgb01(pred_rgb_normalized.float())
            target_rgb = normalized_to_rgb01(target_rgb_normalized.float())
            rgb_l1 = F.l1_loss(pred_rgb, target_rgb)
            pred_lab = _rgb01_to_lab_float32(pred_rgb)
            with torch.no_grad():
                target_lab = _rgb01_to_lab_float32(target_rgb)
            l_l1 = F.l1_loss(pred_lab[:, :1] / 100.0, target_lab[:, :1] / 100.0)
            ab_l1 = F.l1_loss(pred_lab[:, 1:] / 128.0, target_lab[:, 1:] / 128.0)
            ssim = kornia.losses.ssim_loss(pred_rgb, target_rgb, window_size=11, reduction="mean")
            identity = pred_rgb.new_zeros(())
            if input_rgb_normalized is not None and identity_mask is not None:
                input_rgb = normalized_to_rgb01(input_rgb_normalized.float())
                per_sample_identity = torch.abs(pred_rgb - input_rgb).mean(dim=(1, 2, 3))
                mask = identity_mask.float().flatten()
                identity = (per_sample_identity * mask).sum() / mask.sum().clamp_min(1.0)
        hist_emd = lab_ab_histogram_emd(
            pred_rgb,
            target_rgb,
            bins=self.hist_bins,
            max_samples=self.hist_max_samples,
        )
        total = (
            self.rgb_weight * rgb_l1
            + self.l_weight * l_l1
            + self.ab_weight * ab_l1
            + self.ssim_weight * ssim
            + self.hist_emd_weight * hist_emd
            + self.identity_weight * identity
        )
        return {
            "total": total,
            "rgb_l1": rgb_l1,
            "l_l1": l_l1,
            "ab_l1": ab_l1,
            "ssim": ssim,
            "hist_emd": hist_emd,
            "identity": identity,
        }

    def get_config(self) -> dict[str, float | int]:
        return {
            "rgb_weight": self.rgb_weight,
            "l_weight": self.l_weight,
            "ab_weight": self.ab_weight,
            "ssim_weight": self.ssim_weight,
            "hist_emd_weight": self.hist_emd_weight,
            "identity_weight": self.identity_weight,
            "hist_bins": self.hist_bins,
            "hist_max_samples": self.hist_max_samples,
        }
