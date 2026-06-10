from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _paired_crop(input_image: np.ndarray, clean_image: np.ndarray, size: int, train: bool) -> tuple[np.ndarray, np.ndarray]:
    height, width = input_image.shape[:2]
    if clean_image.shape[:2] != (height, width):
        raise ValueError(f"Input/clean size mismatch: {input_image.shape} vs {clean_image.shape}")
    if height < size or width < size:
        raise ValueError(f"Pair is smaller than crop size {size}: {width}x{height}")
    if train:
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
    else:
        top = (height - size) // 2
        left = (width - size) // 2
    return (
        input_image[top : top + size, left : left + size],
        clean_image[top : top + size, left : left + size],
    )


def _augment_pair(input_image: np.ndarray, clean_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        input_image = cv2.flip(input_image, 1)
        clean_image = cv2.flip(clean_image, 1)
    angle = random.uniform(-3.0, 3.0)
    if abs(angle) > 0.1:
        height, width = input_image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        input_image = cv2.warpAffine(input_image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        clean_image = cv2.warpAffine(clean_image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return input_image, clean_image


def rgb_to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 127.5 - 1.0
    return tensor


def validate_clean_target_contract(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing color dataset metadata: {metadata_path}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_profile = metadata.get("target_profile")
    if target_profile != "clean_rgb":
        raise ValueError(
            f"Color dataset targets must be original clean RGB crops; "
            f"found target_profile={target_profile!r} in {metadata_path}"
        )
    target_transform = metadata.get("target_transform")
    if target_transform != "clean_crop_only":
        raise ValueError(
            f"Color dataset targets must use target_transform='clean_crop_only'; "
            f"found {target_transform!r} in {metadata_path}"
        )

    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing color dataset manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Color dataset manifest is empty: {manifest_path}")
    for row in rows:
        target_profile = row.get("target_profile")
        if target_profile != "clean_rgb":
            raise ValueError(
                f"Color dataset sample {row.get('sample_id')!r} has unsupported "
                f"target_profile={target_profile!r}; regenerate with clean RGB targets"
            )
        target_transform = row.get("target_transform")
        if target_transform != "clean_crop_only":
            raise ValueError(
                f"Color dataset sample {row.get('sample_id')!r} has unsupported "
                f"target_transform={target_transform!r}"
            )
        if not row.get("target_path") or not row.get("clean_path"):
            raise ValueError(
                f"Color dataset sample {row.get('sample_id')!r} must define both "
                "target_path and clean_path"
            )
        if row["target_path"] != row["clean_path"]:
            raise ValueError(
                f"Color dataset sample {row.get('sample_id')!r} must use the same "
                "file for target_path and clean_path"
            )
    return metadata


class ColorRestorationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool | None = None,
        return_paths: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = split == "train" if augment is None else bool(augment)
        self.return_paths = return_paths
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        validate_clean_target_contract(self.dataset_root)
        manifest_path = self.dataset_root / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.samples = [row for row in csv.DictReader(handle) if row.get("split") == split]
        if not self.samples:
            raise ValueError(f"No samples for split {split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        input_path = self.dataset_root / row["input_path"]
        clean_path = self.dataset_root / row["target_path"]
        input_image, clean_image = _paired_crop(
            _load_rgb(input_path),
            _load_rgb(clean_path),
            self.image_size,
            train=self.augment,
        )
        if self.augment:
            input_image, clean_image = _augment_pair(input_image, clean_image)
        result: dict[str, Any] = {
            "degraded": rgb_to_normalized_tensor(input_image),
            "clean": rgb_to_normalized_tensor(clean_image),
            "sample_id": row["sample_id"],
        }
        if self.return_paths:
            result["input_path"] = str(input_path)
            result["clean_path"] = str(clean_path)
        return result
