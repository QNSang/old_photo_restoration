from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.restoration import face_restoration, realesrgan_adapter
from src.restoration.post_restoration import (
    apply_final_chroma_matching,
    apply_quality_restoration,
    run_pikfix_post_pipeline,
)


def make_test_image(height: int = 48, width: int = 64) -> np.ndarray:
    y = np.linspace(40, 210, height, dtype=np.float32)[:, None]
    x = np.linspace(20, 180, width, dtype=np.float32)[None, :]
    image = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.broadcast_to((x + y) / 2, (height, width)),
        ],
        axis=-1,
    )
    return np.clip(image, 0, 255).astype(np.uint8)


def test_quality_restoration_is_conservative_and_preserves_contract() -> None:
    image = make_test_image()

    restored, metadata = apply_quality_restoration(image, mode="opencv_conservative")

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
    assert metadata["status"] == "applied"
    mean_change = float(np.abs(restored.astype(np.float32) - image.astype(np.float32)).mean())
    assert 0.0 < mean_change < 35.0


def test_final_chroma_matching_moves_chroma_without_large_luminance_change() -> None:
    image = np.full((48, 48, 3), [155, 130, 105], dtype=np.uint8)
    reference = np.full((48, 48, 3), [110, 140, 165], dtype=np.uint8)
    before_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

    matched, metadata = apply_final_chroma_matching(image, reference, strength=0.35)
    after_lab = cv2.cvtColor(matched, cv2.COLOR_RGB2LAB).astype(np.float32)

    before_distance = np.linalg.norm(before_lab[:, :, 1:].mean(axis=(0, 1)) - reference_lab[:, :, 1:].mean(axis=(0, 1)))
    after_distance = np.linalg.norm(after_lab[:, :, 1:].mean(axis=(0, 1)) - reference_lab[:, :, 1:].mean(axis=(0, 1)))
    assert matched.shape == image.shape
    assert matched.dtype == np.uint8
    assert metadata["status"] == "applied"
    assert after_distance < before_distance
    assert abs(float(after_lab[:, :, 0].mean() - before_lab[:, :, 0].mean())) <= 2.0


def test_post_pipeline_saves_every_stage_and_skips_unconfigured_models(tmp_path: Path) -> None:
    image = make_test_image()

    restored, metadata = run_pikfix_post_pipeline(
        image,
        tmp_path,
        quality_mode="off",
        realesrgan_repo=None,
        face_mode="off",
        final_color_match=False,
    )

    expected_files = [
        "restored_before_post.png",
        "quality_restored.png",
        "color_restored.png",
        "realesrgan_restored.png",
        "restored_before_face.png",
        "codeformer_restored.png",
        "final_color_matched.png",
        "restored_final.png",
        "post_pipeline_metadata.json",
    ]
    assert restored.shape == image.shape
    assert all((tmp_path / filename).exists() for filename in expected_files)
    assert [stage["status"] for stage in metadata["post_pipeline_stages"]] == [
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert metadata["color_restoration_applied"] is False
    assert metadata["realesrgan_applied"] is False
    assert metadata["face_restoration_applied"] is False


def test_realesrgan_adapter_builds_expected_subprocess_command(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "Real-ESRGAN"
    repo.mkdir()
    (repo / "inference_realesrgan.py").write_text("# test stub", encoding="utf-8")
    input_path = tmp_path / "input.png"
    cv2.imwrite(str(input_path), np.zeros((8, 8, 3), dtype=np.uint8))
    captured: dict[str, object] = {}

    def fake_run(command: list[str], cwd: Path, timeout_sec: int):
        captured["command"] = command
        captured["cwd"] = cwd
        output_dir = Path(command[command.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "input_out.png"), np.zeros((16, 16, 3), dtype=np.uint8))

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr(realesrgan_adapter, "_run", fake_run)
    result = realesrgan_adapter.run_realesrgan_subprocess(
        input_path,
        tmp_path / "output",
        repo_path=repo,
        env_name="realesrgan_test",
        model_name="RealESRGAN_x4plus",
        outscale=2,
    )

    command = captured["command"]
    assert result["ok"] is True
    assert captured["cwd"] == repo
    assert command[:5] == ["conda", "run", "-n", "realesrgan_test", "python"]
    assert command[command.index("--outscale") + 1] == "2"


def test_face_restoration_can_disable_face_upsample(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "CodeFormer"
    repo.mkdir()
    captured: dict[str, object] = {}

    def fake_codeformer(*args, **kwargs):
        captured["face_upsample"] = kwargs["face_upsample"]
        return {"ok": False, "reason": "subprocess_failed"}

    monkeypatch.setattr(face_restoration, "CODEFORMER_REPO", repo)
    monkeypatch.setattr(face_restoration, "run_codeformer_subprocess", fake_codeformer)

    image = make_test_image()
    restored, metadata = face_restoration.apply_face_restoration(
        image,
        mode="codeformer_if_available",
        output_dir=tmp_path / "face",
        face_upsample=False,
    )

    assert restored.shape == image.shape
    assert metadata["face_restoration_applied"] is False
    assert captured["face_upsample"] is False
