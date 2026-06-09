from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.color_degradation import DegradationSimulator, VALID_DEGRADATION_PROFILES
from src.restoration.post_restoration import apply_quality_restoration


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VALID_TARGET_PROFILES = {"clean_rgb", "conservative_real_old_photo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired synthetic data for color restoration.")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--train-variants", type=int, default=3)
    parser.add_argument("--eval-variants", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=None, help="Deterministically select at most this many clean source images.")
    parser.add_argument("--degradation-profile", choices=sorted(VALID_DEGRADATION_PROFILES), default="balanced")
    parser.add_argument(
        "--target-profile",
        choices=sorted(VALID_TARGET_PROFILES),
        default="clean_rgb",
        help="Use conservative targets for near-grayscale old photos to avoid forced creative colorization.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-upscale-small", action="store_true")
    parser.add_argument("--save-raw-degraded", action="store_true")
    parser.add_argument(
        "--quality-mode",
        default="off",
        choices=["off", "opencv_conservative"],
        help="Optional pre-processing ablation. Default off trains the color model directly on synthetic degradation.",
    )
    parser.add_argument("--num-previews", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Cannot write image: {path}")


def source_id(path: Path, clean_root: Path) -> str:
    relative = path.relative_to(clean_root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem[:48]}-{digest}"


def list_sources(clean_root: Path) -> list[Path]:
    return sorted(path for path in clean_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def select_source_subset(sources: list[Path], max_sources: int | None, seed: int) -> list[Path]:
    if max_sources is None:
        return sources
    if max_sources < 3:
        raise ValueError("max-sources must be >= 3")
    if len(sources) <= max_sources:
        return sources
    return sorted(random.Random(seed).sample(sources, max_sources))


def audit_sources(sources: list[Path], clean_root: Path, crop_size: int, allow_upscale: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sources:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"source_id": source_id(path, clean_root), "source_path": str(path), "width": "", "height": "", "accepted": 0, "reason": "unreadable"})
            continue
        height, width = image.shape[:2]
        accepted = min(height, width) >= crop_size or allow_upscale
        rows.append(
            {
                "source_id": source_id(path, clean_root),
                "source_path": str(path),
                "width": width,
                "height": height,
                "accepted": int(accepted),
                "reason": "accepted" if accepted else f"min_side_below_{crop_size}",
            }
        )
    return rows


def split_sources(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    accepted = [row for row in rows if int(row["accepted"]) == 1]
    if len(accepted) < 3:
        raise ValueError("At least 3 accepted clean images are required for train/val/test splitting")
    random.Random(seed).shuffle(accepted)
    count = len(accepted)
    train_count = max(1, int(count * 0.8))
    val_count = max(1, int(count * 0.1))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1
    return {
        "train": accepted[:train_count],
        "val": accepted[train_count : train_count + val_count],
        "test": accepted[train_count + val_count :],
    }


def prepare_clean_crop(image: np.ndarray, crop_size: int, rng: random.Random, allow_upscale: bool) -> np.ndarray:
    height, width = image.shape[:2]
    if min(height, width) < crop_size:
        if not allow_upscale:
            raise ValueError(f"Image smaller than crop size: {width}x{height}")
        scale = crop_size / min(height, width)
        image = cv2.resize(
            image,
            (max(crop_size, round(width * scale)), max(crop_size, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        height, width = image.shape[:2]
    top = rng.randint(0, height - crop_size)
    left = rng.randint(0, width - crop_size)
    return np.ascontiguousarray(image[top : top + crop_size, left : left + crop_size])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_model_input(degraded: np.ndarray, quality_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if quality_mode == "off":
        return degraded.copy(), {
            "status": "skipped",
            "backend": "none",
            "reason": "color_only_training",
            "mode": "off",
        }
    return apply_quality_restoration(degraded, mode=quality_mode)


def scale_saturation(image: np.ndarray, saturation_scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def prepare_training_target(
    clean: np.ndarray,
    degradation_metadata: dict[str, Any],
    target_profile: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if target_profile == "clean_rgb":
        return clean.copy(), {
            "profile": target_profile,
            "reason": "full_clean_rgb_target",
            "saturation_scale": 1.0,
        }
    if target_profile != "conservative_real_old_photo":
        raise ValueError(f"Unsupported target profile: {target_profile}")

    subprofile = str(degradation_metadata.get("subprofile") or "faded_old_color")
    saturation_ranges = {
        "strong_sepia": (0.45, 0.70),
        "warm_near_grayscale": (0.08, 0.25),
        "faded_old_color": (0.72, 0.95),
        "mild_degradation": (0.92, 1.0),
        "identity": (1.0, 1.0),
    }
    low, high = saturation_ranges.get(subprofile, (0.65, 0.90))
    saturation_scale = random.Random(seed ^ 0x51A7).uniform(low, high)
    return scale_saturation(clean, saturation_scale), {
        "profile": target_profile,
        "reason": "saturation_matched_to_information_remaining",
        "source_subprofile": subprofile,
        "saturation_scale": saturation_scale,
    }


def make_preview(
    clean: np.ndarray,
    degraded: np.ndarray,
    model_input: np.ndarray,
    target: np.ndarray,
    quality_mode: str,
    target_profile: str,
) -> np.ndarray:
    panels = [clean, degraded]
    if quality_mode != "off":
        panels.append(model_input)
    if target_profile != "clean_rgb":
        panels.append(target)
    return np.concatenate(panels, axis=1)


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def main() -> int:
    args = parse_args()
    clean_root = resolve_path(args.clean_dir)
    output_root = resolve_path(args.output_dir)
    if not clean_root.is_dir():
        raise NotADirectoryError(f"Clean image directory not found: {clean_root}")
    if args.crop_size <= 0 or args.train_variants <= 0 or args.eval_variants <= 0:
        raise ValueError("crop-size and variant counts must be > 0")

    discovered_sources = list_sources(clean_root)
    if not discovered_sources:
        raise FileNotFoundError(f"No clean images found under: {clean_root}")
    sources = select_source_subset(discovered_sources, args.max_sources, args.seed)
    prepare_output(output_root, args.overwrite)
    audit_rows = audit_sources(sources, clean_root, args.crop_size, args.allow_upscale_small)
    splits = split_sources(audit_rows, args.seed)
    source_split = {row["source_id"]: split for split, split_rows in splits.items() for row in split_rows}
    for row in audit_rows:
        row["split"] = source_split.get(row["source_id"], "")
    write_csv(
        output_root / "source_audit.csv",
        audit_rows,
        ["source_id", "source_path", "width", "height", "accepted", "reason", "split"],
    )

    simulator = DegradationSimulator(profile=args.degradation_profile)
    master_rng = random.Random(args.seed)
    manifest: list[dict[str, Any]] = []
    previews: list[np.ndarray] = []
    for split, split_rows in splits.items():
        variants = args.train_variants if split == "train" else args.eval_variants
        for source_row in split_rows:
            path = Path(source_row["source_path"])
            source_image = read_rgb(path)
            for variant in range(variants):
                sample_seed = master_rng.randrange(2**31)
                sample_rng = random.Random(sample_seed)
                clean = prepare_clean_crop(source_image, args.crop_size, sample_rng, args.allow_upscale_small)
                degraded, degradation_metadata = simulator.apply(clean, seed=sample_seed, return_metadata=True)
                model_input, quality_metadata = prepare_model_input(degraded, args.quality_mode)
                target, target_metadata = prepare_training_target(
                    clean,
                    degradation_metadata,
                    args.target_profile,
                    sample_seed,
                )
                sample_id = f"{source_row['source_id']}-v{variant + 1:02d}"
                input_path = output_root / split / "input" / f"{sample_id}.png"
                target_path = output_root / split / "target" / f"{sample_id}.png"
                raw_path = output_root / split / "raw_degraded" / f"{sample_id}.png"
                write_rgb(input_path, model_input)
                write_rgb(target_path, target)
                if args.save_raw_degraded:
                    write_rgb(raw_path, degraded)
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_row["source_id"],
                        "source_path": str(path),
                        "split": split,
                        "variant": variant + 1,
                        "seed": sample_seed,
                        "input_path": input_path.relative_to(output_root).as_posix(),
                        "target_path": target_path.relative_to(output_root).as_posix(),
                        "clean_path": target_path.relative_to(output_root).as_posix(),
                        "raw_degraded_path": raw_path.relative_to(output_root).as_posix() if args.save_raw_degraded else "",
                        "crop_size": args.crop_size,
                        "quality_mode": args.quality_mode,
                        "degradation_profile": args.degradation_profile,
                        "target_profile": args.target_profile,
                        "quality_metadata": json.dumps(quality_metadata, sort_keys=True),
                        "degradation_metadata": json.dumps(degradation_metadata, sort_keys=True),
                        "target_metadata": json.dumps(target_metadata, sort_keys=True),
                    }
                )
                if len(previews) < args.num_previews:
                    previews.append(
                        make_preview(
                            clean,
                            degraded,
                            model_input,
                            target,
                            args.quality_mode,
                            args.target_profile,
                        )
                    )

    manifest_fields = [
        "sample_id", "source_id", "source_path", "split", "variant", "seed", "input_path", "target_path", "clean_path",
        "raw_degraded_path", "crop_size", "quality_mode", "degradation_profile", "target_profile",
        "quality_metadata", "degradation_metadata", "target_metadata",
    ]
    write_csv(output_root / "manifest.csv", manifest, manifest_fields)
    previews_dir = output_root / "previews"
    for index, preview in enumerate(previews, start=1):
        write_rgb(previews_dir / f"comparison_{index:03d}.png", preview)
    if previews:
        columns = 2
        blank = np.full_like(previews[0], 245)
        padded_previews = previews + [blank] * ((-len(previews)) % columns)
        comparison_grid = np.concatenate(
            [np.concatenate(padded_previews[index : index + columns], axis=1) for index in range(0, len(padded_previews), columns)],
            axis=0,
        )
        write_rgb(previews_dir / "comparison_grid.png", comparison_grid)

    created_at = datetime.now(timezone.utc).isoformat()
    split_counts = {split: sum(1 for row in manifest if row["split"] == split) for split in ("train", "val", "test")}
    metadata = {
        "dataset_id": output_root.name,
        "created_at": created_at,
        "clean_root": str(clean_root),
        "crop_size": args.crop_size,
        "seed": args.seed,
        "allow_upscale_small": args.allow_upscale_small,
        "save_raw_degraded": args.save_raw_degraded,
        "quality_mode": args.quality_mode,
        "degradation_profile": args.degradation_profile,
        "target_profile": args.target_profile,
        "input_profile": (
            f"synthetic_{args.degradation_profile}_direct"
            if args.quality_mode == "off"
            else f"synthetic_{args.degradation_profile}_after_{args.quality_mode}"
        ),
        "training_objective": (
            "conservative_color_restoration"
            if args.target_profile == "conservative_real_old_photo"
            else "color_restoration_only"
        ),
        "preview_panels": (
            ["clean_source", "synthetic_degraded_input", "conservative_target"]
            if args.target_profile == "conservative_real_old_photo" and args.quality_mode == "off"
            else ["clean_source", "synthetic_degraded", "model_input_after_quality_stage", "conservative_target"]
            if args.target_profile == "conservative_real_old_photo"
            else ["clean_target", "synthetic_degraded_input"]
            if args.quality_mode == "off"
            else ["clean_target", "synthetic_degraded", "model_input_after_quality_stage"]
        ),
        "train_variants": args.train_variants,
        "eval_variants": args.eval_variants,
        "discovered_sources": len(discovered_sources),
        "selected_sources": len(sources),
        "max_sources": args.max_sources,
        "accepted_sources": sum(int(row["accepted"]) for row in audit_rows),
        "rejected_sources": sum(1 - int(row["accepted"]) for row in audit_rows),
        "split_counts": split_counts,
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
