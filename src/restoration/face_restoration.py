from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.restoration.codeformer_adapter import CODEFORMER_REPO, run_codeformer_subprocess


VALID_FACE_MODES = {"off", "auto", "codeformer_if_available"}


def _base_metadata(enabled: bool, reason: str) -> dict[str, Any]:
    return {
        "face_module_enabled": enabled,
        "face_detection_backend": "none",
        "face_restoration_backend": "none",
        "face_backend": "none",
        "faces_detected": 0,
        "face_restoration_applied": False,
        "reason": reason,
        "warning": None,
    }


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def _read_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _detect_dependency() -> tuple[str | None, str | None]:
    if CODEFORMER_REPO.exists():
        return "codeformer", None
    if Path("external/CodeFormer").exists() or Path("CodeFormer").exists():
        return "codeformer", "Đã thấy thư mục CodeFormer nhưng chưa khớp đường dẫn adapter subprocess."
    if importlib.util.find_spec("gfpgan") is not None:
        return "gfpgan", "Đã thấy package GFPGAN nhưng wrapper chưa được bật cho inference xác định trong project."
    if importlib.util.find_spec("facexlib") is not None:
        return "facexlib", "Đã thấy facexlib nhưng thiếu backend restoration."
    return None, None


def _write_metadata(output_path: Path | None, metadata: dict[str, Any]) -> None:
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "face_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_face_restoration(
    image_rgb: np.ndarray,
    mode: str = "auto",
    strength: float = 0.5,
    output_dir: Path | None = None,
    face_upsample: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Áp dụng Module 3 nếu adapter CodeFormer đã sẵn sàng; nếu chưa thì trả ảnh gốc kèm metadata."""
    if mode not in VALID_FACE_MODES:
        raise ValueError(f"mode không hợp lệ: {mode}. Hợp lệ: {sorted(VALID_FACE_MODES)}")
    image = np.clip(np.asarray(image_rgb), 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb phải có shape HxWx3, nhận được {image.shape}")

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        _save_rgb(output_path / "face_input.png", image)

    if mode == "off":
        metadata = _base_metadata(enabled=False, reason="disabled")
        metadata["face_strength"] = float(strength)
        _write_metadata(output_path, metadata)
        return image, metadata

    dependency, warning = _detect_dependency()
    if dependency != "codeformer":
        metadata = _base_metadata(enabled=True, reason="dependency_not_available")
        metadata["face_strength"] = float(strength)
        metadata["warning"] = warning
        _write_metadata(output_path, metadata)
        return image, metadata

    if output_path is None:
        metadata = _base_metadata(enabled=True, reason="output_dir_required_for_codeformer")
        metadata["face_detection_backend"] = "codeformer"
        metadata["face_restoration_backend"] = "codeformer"
        metadata["face_backend"] = "codeformer"
        metadata["face_strength"] = float(strength)
        metadata["warning"] = warning
        return image, metadata

    codeformer_result = run_codeformer_subprocess(
        output_path / "face_input.png",
        output_path,
        fidelity=float(strength),
        face_upsample=face_upsample,
    )
    if codeformer_result.get("ok"):
        codeformer_output = Path(str(codeformer_result["output"]))
        restored = _read_rgb(codeformer_output)
        if restored is not None:
            metadata = _base_metadata(enabled=True, reason="applied")
            metadata["face_detection_backend"] = "codeformer"
            metadata["face_restoration_backend"] = "codeformer"
            metadata["face_backend"] = "codeformer"
            metadata["faces_detected"] = None
            metadata["face_restoration_applied"] = True
            metadata["face_strength"] = float(strength)
            metadata["codeformer_fidelity"] = float(strength)
            metadata["checkpoint_used"] = str(CODEFORMER_REPO / "weights" / "CodeFormer" / "codeformer.pth")
            metadata["codeformer_output"] = str(codeformer_output)
            metadata["codeformer_result"] = codeformer_result
            _save_rgb(output_path / "face_output.png", restored)
            _write_metadata(output_path, metadata)
            return restored, metadata
        codeformer_result["reason"] = "output_unreadable"

    metadata = _base_metadata(enabled=True, reason=str(codeformer_result.get("reason", "subprocess_failed")))
    metadata["face_detection_backend"] = "codeformer"
    metadata["face_restoration_backend"] = "none"
    metadata["face_backend"] = "none"
    metadata["face_strength"] = float(strength)
    metadata["checkpoint_used"] = str(CODEFORMER_REPO / "weights" / "CodeFormer" / "codeformer.pth")
    metadata["warning"] = codeformer_result.get("stderr_tail") or warning
    metadata["codeformer_result"] = codeformer_result
    _save_rgb(output_path / "face_output.png", image)
    _write_metadata(output_path, metadata)
    return image, metadata
