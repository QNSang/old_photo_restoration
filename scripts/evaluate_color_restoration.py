from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.color_dataset import ColorRestorationDataset
from src.restoration.color_restoration import load_color_checkpoint, model_rgb_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a color restoration checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-lpips", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalized_to_numpy(batch: torch.Tensor) -> np.ndarray:
    return ((batch.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).permute(0, 2, 3, 1).numpy()


def make_lpips(device: torch.device, skip: bool):
    if skip:
        return None, "disabled"
    try:
        import lpips

        return lpips.LPIPS(net="vgg").to(device).eval(), None
    except Exception as exc:
        return None, str(exc)


def empty_metric_values() -> dict[str, list[float]]:
    return {"psnr": [], "ssim": [], "lpips_vgg": [], "delta_e_ciede2000": []}


def append_reference_metrics(values: dict[str, list[float]], prediction: np.ndarray, target: np.ndarray) -> None:
    values["psnr"].append(float(peak_signal_noise_ratio(target, prediction, data_range=1.0)))
    values["ssim"].append(float(structural_similarity(target, prediction, channel_axis=-1, data_range=1.0)))
    values["delta_e_ciede2000"].append(float(deltaE_ciede2000(rgb2lab(target), rgb2lab(prediction)).mean()))


def summarize(values: dict[str, list[float]]) -> dict[str, float | None]:
    return {name: (float(np.mean(items)) if items else None) for name, items in values.items()}


def improvement(input_metrics: dict[str, float | None], output_metrics: dict[str, float | None]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    higher_is_better = {"psnr", "ssim"}
    for name, output_value in output_metrics.items():
        input_value = input_metrics.get(name)
        if input_value is None or output_value is None:
            result[name] = {"absolute": None, "percent": None}
            continue
        absolute = output_value - input_value if name in higher_is_better else input_value - output_value
        percent = absolute / max(abs(input_value), 1e-8) * 100.0
        result[name] = {"absolute": float(absolute), "percent": float(percent)}
    return result


def upsert_registry(path: Path, row: dict[str, Any]) -> None:
    header = ["run_id", "checkpoint", "dataset_id", "split", "psnr", "ssim", "lpips_vgg", "delta_e_ciede2000", "created_at"]
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [existing for existing in rows if not (existing.get("run_id") == row["run_id"] and existing.get("split") == row["split"])]
    rows.append({key: row.get(key, "") for key in header})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, device = load_color_checkpoint(checkpoint_path, args.device)
    image_size = int(checkpoint.get("dataset_config", {}).get("image_size", 256))
    dataset = ColorRestorationDataset(dataset_root, args.split, image_size=image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=device.type == "cuda")
    lpips_model, lpips_warning = make_lpips(device, args.skip_lpips)
    input_values = empty_metric_values()
    output_values = empty_metric_values()
    by_subprofile: dict[str, dict[str, dict[str, list[float]]]] = {}

    with torch.inference_mode():
        for batch in loader:
            inputs = batch["degraded"].to(device)
            targets = batch["clean"].to(device)
            predictions = model_rgb_output(model, inputs)
            input_np = normalized_to_numpy(inputs)
            prediction_np = normalized_to_numpy(predictions)
            target_np = normalized_to_numpy(targets)
            subprofiles = list(batch.get("degradation_subprofile", ["unknown"] * len(prediction_np)))
            for degraded, pred, target, subprofile in zip(input_np, prediction_np, target_np, subprofiles):
                append_reference_metrics(input_values, degraded, target)
                append_reference_metrics(output_values, pred, target)
                group = by_subprofile.setdefault(
                    str(subprofile),
                    {"input": empty_metric_values(), "output": empty_metric_values()},
                )
                append_reference_metrics(group["input"], degraded, target)
                append_reference_metrics(group["output"], pred, target)
            if lpips_model is not None:
                input_scores = lpips_model(inputs, targets).flatten().detach().cpu().tolist()
                output_scores = lpips_model(predictions, targets).flatten().detach().cpu().tolist()
                input_values["lpips_vgg"].extend(float(score) for score in input_scores)
                output_values["lpips_vgg"].extend(float(score) for score in output_scores)
                for subprofile, input_score, output_score in zip(subprofiles, input_scores, output_scores):
                    group = by_subprofile[str(subprofile)]
                    group["input"]["lpips_vgg"].append(float(input_score))
                    group["output"]["lpips_vgg"].append(float(output_score))

    input_metrics = summarize(input_values)
    metrics = summarize(output_values)
    metrics_by_subprofile = {
        name: {
            "num_samples": len(group["output"]["psnr"]),
            "input_metrics": summarize(group["input"]),
            "output_metrics": summarize(group["output"]),
            "improvement": improvement(summarize(group["input"]), summarize(group["output"])),
        }
        for name, group in sorted(by_subprofile.items())
    }
    run_id = checkpoint_path.parent.name
    payload = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "dataset_id": dataset_root.name,
        "split": args.split,
        "num_samples": len(dataset),
        "model_config": checkpoint["model_config"],
        "input_metrics": input_metrics,
        "metrics": metrics,
        "improvement": improvement(input_metrics, metrics),
        "metrics_by_subprofile": metrics_by_subprofile,
        "lpips_warning": lpips_warning,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    upsert_registry(
        PROJECT_ROOT / "results" / "registry" / "color_metric_registry.csv",
        {
            "run_id": run_id,
            "checkpoint": str(checkpoint_path),
            "dataset_id": dataset_root.name,
            "split": args.split,
            **metrics,
            "created_at": payload["created_at"],
        },
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
