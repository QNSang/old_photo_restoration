from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.generate_color_dataset import audit_sources, main as generate_dataset_main, split_sources
from scripts.train_restoration import resolve_num_workers
from src.data.color_degradation import ColorDegradationConfig, DegradationSimulator
from src.models.color_unet import ColorRestorationUNet
from src.restoration.color_restoration import CHECKPOINT_FORMAT_VERSION, run_color_restoration, tiled_color_inference
from src.restoration.post_restoration import run_pikfix_post_pipeline


def test_color_degradation_is_deterministic_and_configurable() -> None:
    image = np.full((48, 64, 3), [120, 150, 190], dtype=np.uint8)
    config = ColorDegradationConfig(
        yellowing_prob=1,
        fading_prob=1,
        sepia_prob=1,
        blur_prob=1,
        noise_prob=1,
        jpeg_prob=1,
        gamma_prob=1,
        color_cast_prob=1,
    )
    simulator = DegradationSimulator(config)
    first, first_metadata = simulator.apply(image, seed=123, return_metadata=True)
    second, second_metadata = simulator.apply(image, seed=123, return_metadata=True)

    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert set(first_metadata["applied"]) == {"yellowing", "fading", "sepia", "gamma", "color_cast", "blur", "noise", "jpeg"}


def test_source_audit_rejects_small_images_and_split_has_no_leakage(tmp_path: Path) -> None:
    import cv2

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    for index, size in enumerate([400, 410, 420, 430, 440, 450, 460, 470, 480, 490]):
        cv2.imwrite(str(clean_root / f"large_{index}.png"), np.zeros((size, size, 3), dtype=np.uint8))
    cv2.imwrite(str(clean_root / "small.png"), np.zeros((100, 100, 3), dtype=np.uint8))
    rows = audit_sources(sorted(clean_root.iterdir()), clean_root, crop_size=320, allow_upscale=False)
    splits = split_sources(rows, seed=42)

    assert sum(int(row["accepted"]) for row in rows) == 10
    split_ids = [{row["source_id"] for row in split_rows} for split_rows in splits.values()]
    assert not split_ids[0].intersection(split_ids[1])
    assert not split_ids[0].intersection(split_ids[2])
    assert not split_ids[1].intersection(split_ids[2])


def test_dataset_generator_writes_manifest_audit_and_comparison_grid(tmp_path: Path, monkeypatch) -> None:
    import cv2

    clean_root = tmp_path / "clean"
    output_root = tmp_path / "dataset"
    clean_root.mkdir()
    for index in range(3):
        cv2.imwrite(str(clean_root / f"source_{index}.png"), np.full((48, 48, 3), 70 + index * 30, dtype=np.uint8))
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_color_dataset.py",
            "--clean-dir", str(clean_root),
            "--output-dir", str(output_root),
            "--crop-size", "32",
            "--train-variants", "1",
            "--eval-variants", "1",
            "--quality-mode", "off",
            "--num-previews", "2",
        ],
    )

    assert generate_dataset_main() == 0
    with (output_root / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert (output_root / "source_audit.csv").exists()
    assert (output_root / "previews" / "comparison_grid.png").exists()


def test_rgb_residual_model_has_bounded_learnable_scale_and_backward() -> None:
    model = ColorRestorationUNet(mode="rgb_residual", base_channels=4, residual_scale_init=0.5)
    inputs = torch.rand(2, 3, 32, 32) * 2 - 1
    outputs = model(inputs)
    outputs.mean().backward()

    assert outputs.shape == inputs.shape
    assert float(outputs.min()) >= -1.0
    assert float(outputs.max()) <= 1.0
    assert 0.1 <= float(model.residual_scale.detach()) <= 1.0
    assert model.residual_scale_raw.grad is not None


def test_lab_ab_model_and_unified_loss() -> None:
    pytest.importorskip("kornia")
    from src.losses.color_restoration import ColorRestorationLoss, reconstruct_lab_ab

    model = ColorRestorationUNet(mode="lab_ab", base_channels=4)
    inputs = torch.rand(2, 3, 32, 32) * 2 - 1
    targets = torch.rand(2, 3, 32, 32) * 2 - 1
    predicted_ab = model(inputs)
    predicted_rgb = reconstruct_lab_ab(inputs, predicted_ab)
    losses = ColorRestorationLoss()(predicted_rgb, targets)
    losses["total"].backward()

    assert predicted_ab.shape == (2, 2, 32, 32)
    assert predicted_rgb.shape == inputs.shape
    assert set(losses) == {"total", "rgb_l1", "ab_l1", "ssim"}


@pytest.mark.parametrize("height,width", [(256, 256), (1200, 800), (1201, 803), (173, 221)])
def test_tiled_inference_preserves_exact_size(height: int, width: int) -> None:
    model = ColorRestorationUNet(mode="rgb_residual", base_channels=1).eval()
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    output = tiled_color_inference(image, model, torch.device("cpu"), tile_size=256, overlap=64, tile_batch_size=8)

    assert output.shape == image.shape
    assert output.dtype == np.uint8


def test_checkpoint_inference_and_experimental_pipeline_apply_color_stage(tmp_path: Path) -> None:
    model = ColorRestorationUNet(mode="rgb_residual", base_channels=2)
    checkpoint = tmp_path / "color.pth"
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "training_config": {},
        },
        checkpoint,
    )
    image = np.full((48, 64, 3), 128, dtype=np.uint8)
    inferred, metadata = run_color_restoration(image, checkpoint, device="cpu", tile_batch_size=1)
    restored, pipeline_metadata = run_pikfix_post_pipeline(
        image,
        tmp_path / "pipeline",
        quality_mode="off",
        color_checkpoint=checkpoint,
        realesrgan_repo=None,
        face_mode="off",
        final_color_match=False,
    )

    assert inferred.shape == image.shape
    assert metadata["model_mode"] == "rgb_residual"
    assert restored.shape == image.shape
    assert pipeline_metadata["color_restoration_applied"] is True


def test_kaggle_safe_worker_contract(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert resolve_num_workers("auto") == 0
    assert resolve_num_workers(2) == 2
