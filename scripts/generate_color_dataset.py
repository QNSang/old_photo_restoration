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

from src.data.color_degradation import DegradationSimulator
from src.restoration.post_restoration import apply_quality_restoration


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired synthetic data for color restoration.")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--train-variants", type=int, default=3)
    parser.add_argument("--eval-variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-upscale-small", action="store_true")
    parser.add_argument("--save-raw-degraded", action="store_true")
    parser.add_argument("--quality-mode", default="opencv_conservative", choices=["off", "opencv_conservative"])
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


def make_preview(clean: np.ndarray, degraded: np.ndarray, quality_input: np.ndarray) -> np.ndarray:
    return np.concatenate([clean, degraded, quality_input], axis=1)


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

    sources = list_sources(clean_root)
    if not sources:
        raise FileNotFoundError(f"No clean images found under: {clean_root}")
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

    simulator = DegradationSimulator()
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
                quality_input, quality_metadata = apply_quality_restoration(degraded, mode=args.quality_mode)
                sample_id = f"{source_row['source_id']}-v{variant + 1:02d}"
                input_path = output_root / split / "input" / f"{sample_id}.png"
                clean_path = output_root / split / "clean" / f"{sample_id}.png"
                raw_path = output_root / split / "raw_degraded" / f"{sample_id}.png"
                write_rgb(input_path, quality_input)
                write_rgb(clean_path, clean)
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
                        "clean_path": clean_path.relative_to(output_root).as_posix(),
                        "raw_degraded_path": raw_path.relative_to(output_root).as_posix() if args.save_raw_degraded else "",
                        "crop_size": args.crop_size,
                        "quality_mode": args.quality_mode,
                        "quality_metadata": json.dumps(quality_metadata, sort_keys=True),
                        "degradation_metadata": json.dumps(degradation_metadata, sort_keys=True),
                    }
                )
                if len(previews) < args.num_previews:
                    previews.append(make_preview(clean, degraded, quality_input))

    manifest_fields = [
        "sample_id", "source_id", "source_path", "split", "variant", "seed", "input_path", "clean_path",
        "raw_degraded_path", "crop_size", "quality_mode", "quality_metadata", "degradation_metadata",
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
        "train_variants": args.train_variants,
        "eval_variants": args.eval_variants,
        "accepted_sources": sum(int(row["accepted"]) for row in audit_rows),
        "rejected_sources": sum(1 - int(row["accepted"]) for row in audit_rows),
        "split_counts": split_counts,
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
