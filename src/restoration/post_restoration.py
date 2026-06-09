from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.restoration.face_restoration import apply_face_restoration
from src.restoration.color_restoration import run_color_restoration
from src.restoration.realesrgan_adapter import run_realesrgan_subprocess


VALID_POST_PIPELINES = {"legacy", "pikfix_experimental"}
VALID_QUALITY_MODES = {"off", "opencv_conservative"}
SKIPPED_REALESRGAN_REASONS = {
    "repo_not_configured",
    "repo_missing",
    "inference_script_missing",
    "env_not_configured",
    "conda_env_unavailable",
}


def _ensure_rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image_rgb), 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape HxWx3, got {image.shape}")
    return image


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    image = _ensure_rgb_uint8(image_rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Cannot write image: {path}")


def _read_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _stage(
    name: str,
    status: str,
    backend: str,
    reason: str,
    output_path: Path,
    **details: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "backend": backend,
        "reason": reason,
        "output": str(output_path),
        **details,
    }


def _stage_details(info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key not in {"status", "backend", "reason", "output", "name"}}


def apply_quality_restoration(image_rgb: np.ndarray, mode: str = "opencv_conservative") -> tuple[np.ndarray, dict[str, Any]]:
    """Apply conservative non-generative cleanup before color restoration."""
    image = _ensure_rgb_uint8(image_rgb)
    if mode not in VALID_QUALITY_MODES:
        raise ValueError(f"Unsupported quality mode: {mode}. Valid: {sorted(VALID_QUALITY_MODES)}")
    if mode == "off":
        return image.copy(), {"status": "skipped", "backend": "none", "reason": "disabled"}

    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    contrast_restored = cv2.cvtColor(cv2.merge([enhanced_l, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(contrast_restored, (0, 0), sigmaX=1.0)
    restored = cv2.addWeighted(contrast_restored, 1.12, blurred, -0.12, 0)
    return _ensure_rgb_uint8(restored), {
        "status": "applied",
        "backend": "opencv",
        "reason": "applied",
        "mode": mode,
        "denoise_h": 3,
        "clahe_clip_limit": 1.35,
        "unsharp_amount": 0.12,
    }


def apply_pikfix_color_restoration(
    image_rgb: np.ndarray,
    checkpoint_path: Path | None = None,
    reference_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the trained color model while preserving a safe pass-through contract."""
    image = _ensure_rgb_uint8(image_rgb)
    checkpoint_exists = checkpoint_path is not None and checkpoint_path.exists()
    reference_exists = reference_path is not None and reference_path.exists()
    base_metadata = {
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_exists": checkpoint_exists,
        "reference": None if reference_path is None else str(reference_path),
        "reference_exists": reference_exists,
        "reference_used": False,
    }
    if checkpoint_path is None:
        return image.copy(), {
            "status": "skipped",
            "backend": "color_restoration_unet",
            "reason": "checkpoint_not_configured",
            **base_metadata,
        }
    if not checkpoint_exists:
        return image.copy(), {
            "status": "skipped",
            "backend": "color_restoration_unet",
            "reason": "checkpoint_missing",
            **base_metadata,
        }
    try:
        restored, metadata = run_color_restoration(image, checkpoint_path)
        return restored, {**base_metadata, **metadata}
    except Exception as exc:
        return image.copy(), {
            "status": "failed",
            "backend": "color_restoration_unet",
            "reason": str(exc),
            **base_metadata,
        }


def apply_final_chroma_matching(
    image_rgb: np.ndarray,
    chroma_reference_rgb: np.ndarray,
    strength: float = 0.35,
    max_shift: float = 8.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match only mean Lab chroma while preserving the output image luminance."""
    image = _ensure_rgb_uint8(image_rgb)
    reference = _ensure_rgb_uint8(chroma_reference_rgb)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")
    if reference.shape[:2] != image.shape[:2]:
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    raw_shift_a = float(reference_lab[:, :, 1].mean() - image_lab[:, :, 1].mean())
    raw_shift_b = float(reference_lab[:, :, 2].mean() - image_lab[:, :, 2].mean())
    shift_a = float(np.clip(raw_shift_a * strength, -max_shift, max_shift))
    shift_b = float(np.clip(raw_shift_b * strength, -max_shift, max_shift))
    image_lab[:, :, 1] = np.clip(image_lab[:, :, 1] + shift_a, 0, 255)
    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] + shift_b, 0, 255)
    matched = cv2.cvtColor(image_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return _ensure_rgb_uint8(matched), {
        "status": "applied",
        "backend": "opencv_lab_chroma",
        "reason": "applied",
        "strength": float(strength),
        "max_shift": float(max_shift),
        "shift_a": shift_a,
        "shift_b": shift_b,
    }


def run_pikfix_post_pipeline(
    image_rgb: np.ndarray,
    output_dir: Path,
    *,
    quality_mode: str = "opencv_conservative",
    color_checkpoint: Path | None = None,
    color_reference: Path | None = None,
    realesrgan_repo: Path | None = None,
    realesrgan_env: str = "realesrgan",
    realesrgan_model_name: str = "RealESRGAN_x4plus",
    realesrgan_outscale: float = 2.0,
    face_mode: str = "off",
    codeformer_fidelity: float = 0.7,
    final_color_match: bool = True,
    final_color_match_strength: float = 0.35,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the optional post-inpainting research pipeline with auditable pass-through stages."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    current = _ensure_rgb_uint8(image_rgb).copy()
    stages: list[dict[str, Any]] = []
    stage_outputs: dict[str, str] = {}
    warnings: list[str] = []

    restored_before_post = output_path / "restored_before_post.png"
    _save_rgb(restored_before_post, current)
    stage_outputs["restored_before_post"] = str(restored_before_post)

    quality_output = output_path / "quality_restored.png"
    try:
        current, info = apply_quality_restoration(current, mode=quality_mode)
        stage = _stage(
            "quality_restoration",
            info["status"],
            info["backend"],
            info["reason"],
            quality_output,
            **_stage_details(info),
        )
    except Exception as exc:
        stage = _stage("quality_restoration", "failed", "opencv", str(exc), quality_output)
        warnings.append(f"quality_restoration: {exc}")
    _save_rgb(quality_output, current)
    stages.append(stage)
    stage_outputs["quality_restored"] = str(quality_output)

    color_output = output_path / "color_restored.png"
    current, info = apply_pikfix_color_restoration(current, checkpoint_path=color_checkpoint, reference_path=color_reference)
    _save_rgb(color_output, current)
    stages.append(
        _stage(
            "color_restoration",
            info["status"],
            info["backend"],
            info["reason"],
            color_output,
            **_stage_details(info),
        )
    )
    stage_outputs["color_restored"] = str(color_output)
    if info["status"] == "failed" or info["reason"] not in {"checkpoint_not_configured"}:
        warnings.append(f"color_restoration: {info['reason']}")

    realesrgan_output = output_path / "realesrgan_restored.png"
    realesrgan_result = run_realesrgan_subprocess(
        color_output,
        output_path / "realesrgan_module",
        repo_path=realesrgan_repo,
        env_name=realesrgan_env,
        model_name=realesrgan_model_name,
        outscale=realesrgan_outscale,
    )
    if realesrgan_result.get("ok"):
        restored = _read_rgb(Path(str(realesrgan_result["output"])))
        if restored is not None:
            current = restored
            realesrgan_status = "applied"
            realesrgan_reason = "applied"
        else:
            realesrgan_status = "failed"
            realesrgan_reason = "output_unreadable"
    else:
        realesrgan_reason = str(realesrgan_result.get("reason", "unknown"))
        realesrgan_status = "skipped" if realesrgan_reason in SKIPPED_REALESRGAN_REASONS else "failed"
    if realesrgan_status == "failed":
        warnings.append(f"realesrgan: {realesrgan_reason}")
    _save_rgb(realesrgan_output, current)
    stages.append(
        _stage(
            "realesrgan",
            realesrgan_status,
            "realesrgan",
            realesrgan_reason,
            realesrgan_output,
            result=realesrgan_result,
        )
    )
    stage_outputs["realesrgan_restored"] = str(realesrgan_output)

    restored_before_face = output_path / "restored_before_face.png"
    _save_rgb(restored_before_face, current)
    stage_outputs["restored_before_face"] = str(restored_before_face)
    before_face = current.copy()
    face_output = output_path / "codeformer_restored.png"
    try:
        current, face_metadata = apply_face_restoration(
            current,
            mode=face_mode,
            strength=codeformer_fidelity,
            output_dir=output_path / "face_module",
            face_upsample=False,
        )
        if face_metadata.get("face_restoration_applied"):
            face_status = "applied"
        elif face_metadata.get("reason") in {"disabled", "dependency_not_available", "output_dir_required_for_codeformer"}:
            face_status = "skipped"
        else:
            face_status = "failed"
            warnings.append(f"codeformer: {face_metadata.get('reason', 'unknown')}")
    except Exception as exc:
        face_metadata = {"face_restoration_applied": False, "reason": str(exc), "warning": str(exc)}
        face_status = "failed"
        warnings.append(f"codeformer: {exc}")
    _save_rgb(face_output, current)
    stages.append(
        _stage(
            "codeformer",
            face_status,
            "codeformer",
            str(face_metadata.get("reason", "unknown")),
            face_output,
            face_upsample=False,
            metadata=face_metadata,
        )
    )
    stage_outputs["codeformer_restored"] = str(face_output)

    final_color_output = output_path / "final_color_matched.png"
    if final_color_match:
        try:
            current, info = apply_final_chroma_matching(
                current,
                before_face,
                strength=final_color_match_strength,
            )
            final_stage = _stage(
                "final_color_match",
                info["status"],
                info["backend"],
                info["reason"],
                final_color_output,
                **_stage_details(info),
            )
        except Exception as exc:
            final_stage = _stage("final_color_match", "failed", "opencv_lab_chroma", str(exc), final_color_output)
            warnings.append(f"final_color_match: {exc}")
    else:
        final_stage = _stage("final_color_match", "skipped", "none", "disabled", final_color_output)
    _save_rgb(final_color_output, current)
    stages.append(final_stage)
    stage_outputs["final_color_matched"] = str(final_color_output)

    final_output = output_path / "restored_final.png"
    _save_rgb(final_output, current)
    stage_outputs["restored_final"] = str(final_output)

    metadata = {
        "post_pipeline": "pikfix_experimental",
        "post_pipeline_stages": stages,
        "quality_restoration_applied": stages[0]["status"] == "applied",
        "color_restoration_applied": stages[1]["status"] == "applied",
        "color_reference": None if color_reference is None else str(color_reference),
        "realesrgan_applied": stages[2]["status"] == "applied",
        "realesrgan_outscale": float(realesrgan_outscale),
        "face_restoration_applied": bool(face_metadata.get("face_restoration_applied", False)),
        "final_color_match_applied": final_stage["status"] == "applied",
        "stage_outputs": stage_outputs,
        "stage_warnings": warnings,
        "face_metadata": face_metadata,
    }
    (output_path / "post_pipeline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return current, metadata
