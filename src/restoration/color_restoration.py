from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.losses.color_restoration import reconstruct_lab_ab
from src.models.color_unet import ColorRestorationUNet


CHECKPOINT_FORMAT_VERSION = 1


def resolve_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def load_color_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[ColorRestorationUNet, dict[str, Any], torch.device]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Color checkpoint not found: {path}")
    resolved_device = resolve_torch_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"Invalid color checkpoint contract: {path}")
    model = ColorRestorationUNet(**checkpoint["model_config"]).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, resolved_device


def _model_rgb_output(model: ColorRestorationUNet, input_tensor: torch.Tensor) -> torch.Tensor:
    prediction = model(input_tensor)
    if model.mode == "lab_ab":
        return reconstruct_lab_ab(input_tensor, prediction)
    return prediction


def _padding_mode(height: int, width: int, pad_bottom: int, pad_right: int) -> str:
    if min(height, width) <= 1 or pad_bottom >= height or pad_right >= width:
        return "replicate"
    return "reflect"


def _coverage_size(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return tile_size
    steps = (length - tile_size + stride - 1) // stride
    return tile_size + steps * stride


def tiled_color_inference(
    image_rgb: np.ndarray,
    model: ColorRestorationUNet,
    device: torch.device,
    *,
    tile_size: int = 256,
    overlap: int = 64,
    tile_batch_size: int = 4,
) -> np.ndarray:
    image = np.clip(np.asarray(image_rgb), 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape HxWx3, got {image.shape}")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if tile_size % 16 != 0:
        raise ValueError("tile_size must be divisible by 16 for ColorRestorationUNet")

    height, width = image.shape[:2]
    stride = tile_size - overlap
    padded_height = _coverage_size(height, tile_size, stride)
    padded_width = _coverage_size(width, tile_size, stride)
    pad_bottom = padded_height - height
    pad_right = padded_width - width

    input_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    input_tensor = (input_tensor / 127.5 - 1.0).unsqueeze(0)
    input_tensor = F.pad(
        input_tensor,
        (0, pad_right, 0, pad_bottom),
        mode=_padding_mode(height, width, pad_bottom, pad_right),
    )

    window_1d = torch.hann_window(tile_size, periodic=False).clamp_min(0.05)
    window = (window_1d[:, None] * window_1d[None, :]).unsqueeze(0).unsqueeze(0)
    accumulator = torch.zeros((1, 3, padded_height, padded_width), dtype=torch.float32)
    weights = torch.zeros((1, 1, padded_height, padded_width), dtype=torch.float32)
    positions = [
        (top, left)
        for top in range(0, padded_height - tile_size + 1, stride)
        for left in range(0, padded_width - tile_size + 1, stride)
    ]

    with torch.inference_mode():
        for start in range(0, len(positions), max(1, tile_batch_size)):
            batch_positions = positions[start : start + max(1, tile_batch_size)]
            patches = torch.cat(
                [input_tensor[:, :, top : top + tile_size, left : left + tile_size] for top, left in batch_positions],
                dim=0,
            ).to(device)
            predictions = _model_rgb_output(model, patches).detach().cpu()
            for prediction, (top, left) in zip(predictions, batch_positions):
                accumulator[:, :, top : top + tile_size, left : left + tile_size] += prediction.unsqueeze(0) * window
                weights[:, :, top : top + tile_size, left : left + tile_size] += window

    output = accumulator / weights.clamp_min(1e-6)
    output = output[:, :, :height, :width].squeeze(0)
    output = ((output.clamp(-1.0, 1.0) + 1.0) * 127.5).permute(1, 2, 0).numpy()
    return np.clip(output, 0, 255).astype(np.uint8)


def run_color_restoration(
    image_rgb: np.ndarray,
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    tile_size: int = 256,
    overlap: int = 64,
    tile_batch_size: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, checkpoint, resolved_device = load_color_checkpoint(checkpoint_path, device=device)
    restored = tiled_color_inference(
        image_rgb,
        model,
        resolved_device,
        tile_size=tile_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
    )
    return restored, {
        "status": "applied",
        "backend": "color_restoration_unet",
        "reason": "applied",
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_format_version": checkpoint.get("checkpoint_format_version"),
        "model_mode": model.mode,
        "model_config": model.get_config(),
        "device": str(resolved_device),
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_batch_size": tile_batch_size,
        "residual_scale": float(model.residual_scale.detach().cpu()),
    }
