from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if text else ""


def _list_output_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _run(command: list[str], cwd: Path, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_sec,
    )


def run_realesrgan_subprocess(
    input_image_path: Path,
    output_dir: Path,
    repo_path: Path | None,
    env_name: str = "realesrgan",
    model_name: str = "RealESRGAN_x4plus",
    outscale: float = 2.0,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    """Run Real-ESRGAN in its own conda environment without importing it here."""
    input_path = Path(input_image_path)
    output_path = Path(output_dir)
    repo = Path(repo_path) if repo_path is not None else None

    base = {
        "backend": "realesrgan",
        "input": str(input_path),
        "repo": None if repo is None else str(repo),
        "env": env_name,
        "model_name": model_name,
        "outscale": float(outscale),
    }
    if not input_path.exists():
        return {**base, "ok": False, "reason": "input_missing"}
    if repo is None:
        return {**base, "ok": False, "reason": "repo_not_configured"}
    if not repo.exists():
        return {**base, "ok": False, "reason": "repo_missing"}

    inference_script = repo / "inference_realesrgan.py"
    if not inference_script.exists():
        return {**base, "ok": False, "reason": "inference_script_missing"}
    if not env_name.strip():
        return {**base, "ok": False, "reason": "env_not_configured"}
    if outscale <= 0:
        return {**base, "ok": False, "reason": "invalid_outscale"}

    run_id = f"{input_path.stem}_{int(time.time() * 1000)}"
    prepared_dir = output_path / "_realesrgan_input" / run_id
    raw_output_dir = output_path / "realesrgan_raw" / run_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    copied_input = prepared_dir / f"{input_path.stem}{input_path.suffix.lower()}"
    shutil.copy2(input_path, copied_input)

    command = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(inference_script),
        "-n",
        model_name,
        "-i",
        str(copied_input),
        "-o",
        str(raw_output_dir),
        "--outscale",
        f"{float(outscale):g}",
    ]
    try:
        result = _run(command, cwd=repo, timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "ok": False,
            "reason": "timeout",
            "command": command,
            "raw_output_dir": str(raw_output_dir),
            "stderr_tail": str(exc),
        }
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "reason": "subprocess_exception",
            "command": command,
            "raw_output_dir": str(raw_output_dir),
            "stderr_tail": str(exc),
        }

    outputs = _list_output_images(raw_output_dir)
    if result.returncode == 0 and outputs:
        return {
            **base,
            "ok": True,
            "reason": "applied",
            "command": command,
            "copied_input": str(copied_input),
            "output": str(outputs[0]),
            "raw_output_dir": str(raw_output_dir),
            "returncode": result.returncode,
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
        }

    failure_text = f"{result.stdout}\n{result.stderr}".lower()
    failure_reason = "subprocess_failed"
    if "could not find conda environment" in failure_text or "environmentlocationnotfound" in failure_text:
        failure_reason = "conda_env_unavailable"

    return {
        **base,
        "ok": False,
        "reason": "output_missing" if result.returncode == 0 else failure_reason,
        "command": command,
        "raw_output_dir": str(raw_output_dir),
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }
