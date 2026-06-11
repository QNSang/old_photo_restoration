from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_LAMA_WORKSPACE = (
    Path(r"F:\deeplearning\external_models\lama")
    if os.name == "nt"
    else PROJECT_ROOT / "external_models" / "lama"
)
OFFICIAL_LAMA_REPO = Path(
    os.environ.get("OFFICIAL_LAMA_REPO", str(DEFAULT_OFFICIAL_LAMA_WORKSPACE / "lama"))
).expanduser()
OFFICIAL_LAMA_CPU_ENV = os.environ.get("OFFICIAL_LAMA_CPU_ENV", "lama")
OFFICIAL_LAMA_GPU_ENV = os.environ.get("OFFICIAL_LAMA_GPU_ENV", "lama_gpu")
OFFICIAL_LAMA_ENV = OFFICIAL_LAMA_CPU_ENV
OFFICIAL_LAMA_CHECKPOINT = Path(
    os.environ.get(
        "OFFICIAL_LAMA_CHECKPOINT",
        str(DEFAULT_OFFICIAL_LAMA_WORKSPACE / "weights" / "big-lama" / "models" / "best.ckpt"),
    )
).expanduser()
OFFICIAL_LAMA_MODEL_DIR = Path(
    os.environ.get("OFFICIAL_LAMA_MODEL_DIR", str(OFFICIAL_LAMA_CHECKPOINT.parent.parent))
).expanduser()
OFFICIAL_LAMA_BACKEND = "official_lama_pretrained"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _tail(text: str, limit: int = 2500) -> str:
    return text[-limit:] if text else ""


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def probe_lama_env(env_name: str, timeout_sec: int = 30) -> dict[str, Any]:
    script = (
        "import json, torch; "
        "print(json.dumps({"
        "'available': True, "
        "'reason': 'available', "
        "'torch_version': torch.__version__, "
        "'cuda_build': torch.version.cuda, "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': int(torch.cuda.device_count()), "
        "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    command = ["conda", "run", "-n", env_name, "python", "-c", script]
    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    try:
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"env": env_name, "available": False, "reason": "probe_timeout"}
    except Exception as exc:
        return {"env": env_name, "available": False, "reason": f"probe_exception: {exc}"}

    payload = _extract_json_from_stdout(completed.stdout) or {}
    payload["env"] = env_name
    payload["returncode"] = completed.returncode
    payload["stdout_tail"] = _tail(completed.stdout, 800)
    payload["stderr_tail"] = _tail(completed.stderr, 800)
    if completed.returncode != 0:
        payload["available"] = False
        payload["reason"] = payload.get("reason") or "conda_env_unavailable"
    return payload


def select_official_lama_runtime(prefer_gpu: bool = True) -> dict[str, Any]:
    gpu_probe = probe_lama_env(OFFICIAL_LAMA_GPU_ENV) if prefer_gpu else {
        "env": OFFICIAL_LAMA_GPU_ENV,
        "available": False,
        "reason": "gpu_not_requested",
    }
    cpu_probe = probe_lama_env(OFFICIAL_LAMA_CPU_ENV)
    runtime: dict[str, Any] = {
        "official_lama_env_requested": OFFICIAL_LAMA_GPU_ENV if prefer_gpu else OFFICIAL_LAMA_CPU_ENV,
        "official_lama_device_requested": "cuda" if prefer_gpu else "cpu",
        "gpu_probe": gpu_probe,
        "cpu_probe": cpu_probe,
    }
    if prefer_gpu and gpu_probe.get("available") and gpu_probe.get("cuda_available"):
        runtime.update(
            {
                "available": True,
                "reason": "gpu_available",
                "official_lama_env_actual": OFFICIAL_LAMA_GPU_ENV,
                "official_lama_device_actual": "cuda",
                "official_lama_cuda_available": True,
                "official_lama_torch_version": gpu_probe.get("torch_version"),
                "official_lama_cuda_build": gpu_probe.get("cuda_build"),
            }
        )
        return runtime
    if cpu_probe.get("available"):
        runtime.update(
            {
                "available": True,
                "reason": "cpu_available",
                "official_lama_env_actual": OFFICIAL_LAMA_CPU_ENV,
                "official_lama_device_actual": "cpu",
                "official_lama_cuda_available": bool(cpu_probe.get("cuda_available", False)),
                "official_lama_torch_version": cpu_probe.get("torch_version"),
                "official_lama_cuda_build": cpu_probe.get("cuda_build"),
            }
        )
        return runtime
    runtime.update(
        {
            "available": False,
            "reason": "no_official_lama_env_available",
            "official_lama_env_actual": None,
            "official_lama_device_actual": None,
            "official_lama_cuda_available": False,
            "official_lama_torch_version": None,
            "official_lama_cuda_build": None,
        }
    )
    return runtime


def _base_result(input_path: Path, mask_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "ok": False,
        "input": str(input_path),
        "mask": str(mask_path),
        "output": None,
        "backend": OFFICIAL_LAMA_BACKEND,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "reason": "",
        "device": None,
        "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        "model_path": str(OFFICIAL_LAMA_MODEL_DIR),
        "repo": str(OFFICIAL_LAMA_REPO),
        "output_dir": str(output_dir),
        "official_lama_env_requested": OFFICIAL_LAMA_GPU_ENV,
        "official_lama_env_actual": None,
        "official_lama_device_requested": "cuda",
        "official_lama_device_actual": None,
        "official_lama_cuda_available": False,
        "official_lama_torch_version": None,
        "official_lama_cuda_build": None,
        "attempts": [],
    }


def _read_color(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Không ghi được file: {path}")


def _list_output_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _select_official_output(raw_output_dir: Path, input_stem: str) -> Path | None:
    preferred_names = [
        raw_output_dir / f"{input_stem}_mask.png",
        raw_output_dir / f"{input_stem}_mask.jpg",
        raw_output_dir / f"{input_stem}.png",
        raw_output_dir / f"{input_stem}.jpg",
    ]
    for candidate in preferred_names:
        if candidate.exists():
            return candidate
    images = _list_output_images(raw_output_dir)
    return images[0] if images else None


def _prepare_input(input_path: Path, mask_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    image = _read_color(input_path)
    mask = _read_mask(mask_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được input image: {input_path}")
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Size mismatch: image H/W={image.shape[:2]}, mask H/W={mask.shape[:2]}")

    run_id = f"{input_path.stem}_{int(time.time() * 1000)}"
    prepared_dir = output_dir / "_official_lama_input" / run_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_image = prepared_dir / f"{input_path.stem}.png"
    prepared_mask = prepared_dir / f"{input_path.stem}_mask.png"
    _write_png(prepared_image, image)
    _write_png(prepared_mask, mask)
    return prepared_dir, prepared_image, prepared_mask


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(OFFICIAL_LAMA_REPO) if not existing_pythonpath else f"{OFFICIAL_LAMA_REPO}{os.pathsep}{existing_pythonpath}"
    return env


def _run_predict(
    *,
    prepared_dir: Path,
    output_root: Path,
    input_stem: str,
    env_name: str,
    device: str,
    timeout_sec: int,
) -> dict[str, Any]:
    predict_script = OFFICIAL_LAMA_REPO / "bin" / "predict.py"
    raw_output_dir = output_root / "official_lama_raw" / f"{prepared_dir.name}_{env_name}_{device}"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(predict_script),
        f"model.path={OFFICIAL_LAMA_MODEL_DIR}",
        f"indir={prepared_dir}",
        f"outdir={raw_output_dir}",
        f"device={device}",
    ]
    attempt: dict[str, Any] = {
        "env": env_name,
        "device": device,
        "command": command,
        "raw_output_dir": str(raw_output_dir),
        "ok": False,
        "returncode": None,
        "reason": "",
        "stdout_tail": "",
        "stderr_tail": "",
        "selected_raw_output": None,
        "elapsed_sec": None,
    }
    start_time = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=OFFICIAL_LAMA_REPO,
            env=_build_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        attempt["elapsed_sec"] = time.perf_counter() - start_time
        attempt["reason"] = "timeout"
        attempt["stdout_tail"] = _tail(exc.stdout if isinstance(exc.stdout, str) else "")
        attempt["stderr_tail"] = _tail(exc.stderr if isinstance(exc.stderr, str) else "")
        return attempt
    except Exception as exc:
        attempt["elapsed_sec"] = time.perf_counter() - start_time
        attempt["reason"] = f"subprocess_exception: {exc}"
        return attempt

    attempt["elapsed_sec"] = time.perf_counter() - start_time
    attempt["returncode"] = completed.returncode
    attempt["stdout_tail"] = _tail(completed.stdout)
    attempt["stderr_tail"] = _tail(completed.stderr)
    if completed.returncode != 0:
        attempt["reason"] = "subprocess_failed"
        return attempt

    selected_output = _select_official_output(raw_output_dir, input_stem)
    if selected_output is None:
        attempt["reason"] = "output_missing"
        return attempt
    attempt["ok"] = True
    attempt["reason"] = "applied"
    attempt["selected_raw_output"] = str(selected_output)
    return attempt


def _copy_success_output(result: dict[str, Any], attempt: dict[str, Any], output_path: Path) -> None:
    final_output = output_path / "official_lama_restored.png"
    shutil.copy2(Path(str(attempt["selected_raw_output"])), final_output)
    result["ok"] = True
    result["output"] = str(final_output)
    result["returncode"] = attempt.get("returncode")
    result["stdout_tail"] = attempt.get("stdout_tail", "")
    result["stderr_tail"] = attempt.get("stderr_tail", "")
    result["selected_raw_output"] = attempt.get("selected_raw_output")


def _apply_runtime_metadata(result: dict[str, Any], runtime: dict[str, Any]) -> None:
    for key in [
        "official_lama_env_requested",
        "official_lama_env_actual",
        "official_lama_device_requested",
        "official_lama_device_actual",
        "official_lama_cuda_available",
        "official_lama_torch_version",
        "official_lama_cuda_build",
    ]:
        result[key] = runtime.get(key)
    result["device"] = runtime.get("official_lama_device_actual")
    result["runtime_probe"] = runtime


def run_official_lama_subprocess(
    input_image_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    timeout_sec: int = 900,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    """Chạy official/pretrained LaMa qua conda subprocess, không import LaMa vào env chính."""
    input_path = Path(input_image_path)
    mask = Path(mask_path)
    output_path = Path(output_dir)
    result = _base_result(input_path, mask, output_path)

    if not input_path.exists():
        result["reason"] = "input_missing"
        return result
    if not mask.exists():
        result["reason"] = "mask_missing"
        return result
    if not OFFICIAL_LAMA_REPO.exists():
        result["reason"] = "repo_missing"
        return result
    predict_script = OFFICIAL_LAMA_REPO / "bin" / "predict.py"
    if not predict_script.exists():
        result["reason"] = "predict_script_missing"
        return result
    if not OFFICIAL_LAMA_CHECKPOINT.exists():
        result["reason"] = "checkpoint_missing"
        return result

    runtime = select_official_lama_runtime(prefer_gpu=prefer_gpu)
    _apply_runtime_metadata(result, runtime)
    if not runtime.get("available"):
        result["reason"] = runtime.get("reason", "runtime_unavailable")
        return result

    output_path.mkdir(parents=True, exist_ok=True)
    try:
        prepared_dir, prepared_image, prepared_mask = _prepare_input(input_path, mask, output_path)
    except Exception as exc:
        result["reason"] = f"prepare_failed: {exc}"
        return result

    result.update(
        {
            "prepared_input_dir": str(prepared_dir),
            "prepared_image": str(prepared_image),
            "prepared_mask": str(prepared_mask),
        }
    )

    primary_env = str(runtime["official_lama_env_actual"])
    primary_device = str(runtime["official_lama_device_actual"])
    primary_attempt = _run_predict(
        prepared_dir=prepared_dir,
        output_root=output_path,
        input_stem=input_path.stem,
        env_name=primary_env,
        device=primary_device,
        timeout_sec=timeout_sec,
    )
    result["attempts"].append(primary_attempt)
    if primary_attempt.get("ok"):
        _copy_success_output(result, primary_attempt, output_path)
        result["reason"] = "applied"
        return result

    result["gpu_failed_reason"] = primary_attempt.get("reason") if primary_device == "cuda" else None
    if primary_device == "cuda":
        cpu_runtime = select_official_lama_runtime(prefer_gpu=False)
        cpu_runtime["official_lama_env_requested"] = runtime.get("official_lama_env_requested")
        cpu_runtime["official_lama_device_requested"] = runtime.get("official_lama_device_requested")
        if cpu_runtime.get("available"):
            cpu_attempt = _run_predict(
                prepared_dir=prepared_dir,
                output_root=output_path,
                input_stem=input_path.stem,
                env_name=OFFICIAL_LAMA_CPU_ENV,
                device="cpu",
                timeout_sec=timeout_sec,
            )
            result["attempts"].append(cpu_attempt)
            if cpu_attempt.get("ok"):
                _copy_success_output(result, cpu_attempt, output_path)
                _apply_runtime_metadata(result, cpu_runtime)
                result["official_lama_env_actual"] = OFFICIAL_LAMA_CPU_ENV
                result["official_lama_device_actual"] = "cpu"
                result["device"] = "cpu"
                result["reason"] = "gpu_failed_then_cpu_applied"
                return result

    result["returncode"] = primary_attempt.get("returncode")
    result["stdout_tail"] = primary_attempt.get("stdout_tail", "")
    result["stderr_tail"] = primary_attempt.get("stderr_tail", "")
    result["reason"] = primary_attempt.get("reason", "subprocess_failed")
    return result
