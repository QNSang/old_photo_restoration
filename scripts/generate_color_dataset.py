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

from src.data.color_degradation import DegradationSimulator, VALID_DEGRADATION_PROFILES, mean_lab_chroma
from src.restoration.post_restoration import apply_quality_restoration


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_PROFILE = "clean_rgb"
TARGET_TRANSFORM = "clean_crop_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired synthetic data for color restoration.")
    parser.add_argument(
        "--clean-dir",
        required=True,
        action="append",
        help="Clean source directory. Repeat the flag to mix multiple datasets.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--train-variants", type=int, default=3)
    parser.add_argument("--eval-variants", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=None, help="Deterministically select at most this many clean source images.")
    parser.add_argument("--degradation-profile", choices=sorted(VALID_DEGRADATION_PROFILES), default="faded_color_v2")
    parser.add_argument(
        "--min-source-chroma",
        type=float,
        default=None,
        help="Reject clean sources below this mean Lab chroma. V2 defaults to 4.0; legacy profiles default to 0.",
    )
    parser.add_argument(
        "--target-profile",
        choices=[TARGET_PROFILE],
        default=TARGET_PROFILE,
        help="Training targets are always the original clean RGB crop.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-upscale-small", action="store_true")
    parser.add_argument("--save-raw-degraded", action="store_true")
    parser.add_argument(
        "--quality-mode",
        default="opencv_conservative",
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


def _as_roots(clean_roots: Path | list[Path]) -> list[Path]:
    return [clean_roots] if isinstance(clean_roots, Path) else clean_roots


def source_id(path: Path, clean_roots: Path | list[Path]) -> str:
    roots = _as_roots(clean_roots)
    matching_root = next((root for root in roots if path.is_relative_to(root)), None)
    identity = path.as_posix() if matching_root is None else f"{matching_root.name}/{path.relative_to(matching_root).as_posix()}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem[:48]}-{digest}"


def list_sources(clean_roots: Path | list[Path]) -> list[Path]:
    return sorted(
        path
        for clean_root in _as_roots(clean_roots)
        for path in clean_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_source_subset(sources: list[Path], max_sources: int | None, seed: int) -> list[Path]:
    if max_sources is None:
        return sources
    if max_sources < 3:
        raise ValueError("max-sources must be >= 3")
    if len(sources) <= max_sources:
        return sources
    return sorted(random.Random(seed).sample(sources, max_sources))


def audit_sources(
    sources: list[Path],
    clean_root: Path | list[Path],
    crop_size: int,
    allow_upscale: bool,
    min_source_chroma: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sources:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"source_id": source_id(path, clean_root), "source_path": str(path), "width": "", "height": "", "mean_chroma": "", "accepted": 0, "reason": "unreadable"})
            continue
        height, width = image.shape[:2]
        chroma = mean_lab_chroma(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        size_accepted = min(height, width) >= crop_size or allow_upscale
        chroma_accepted = chroma >= min_source_chroma
        accepted = size_accepted and chroma_accepted
        if not size_accepted:
            reason = f"min_side_below_{crop_size}"
        elif not chroma_accepted:
            reason = f"mean_chroma_below_{min_source_chroma:g}"
        else:
            reason = "accepted"
        rows.append(
            {
                "source_id": source_id(path, clean_root),
                "source_path": str(path),
                "width": width,
                "height": height,
                "mean_chroma": chroma,
                "accepted": int(accepted),
                "reason": reason,
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


def make_preview(
    clean: np.ndarray,
    degraded: np.ndarray,
    model_input: np.ndarray,
    quality_mode: str,
) -> np.ndarray:
    panels = [clean, degraded]
    if quality_mode != "off":
        panels.append(model_input)
    return np.concatenate(panels, axis=1)


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def main() -> int:
    args = parse_args()
    clean_roots = [resolve_path(value) for value in args.clean_dir]
    output_root = resolve_path(args.output_dir)
    missing_roots = [path for path in clean_roots if not path.is_dir()]
    if missing_roots:
        raise NotADirectoryError(f"Clean image directories not found: {missing_roots}")
    if args.crop_size <= 0 or args.train_variants <= 0 or args.eval_variants <= 0:
        raise ValueError("crop-size and variant counts must be > 0")
    if args.min_source_chroma is None:
        min_source_chroma = 4.0 if args.degradation_profile == "faded_color_v2" else 0.0
    else:
        min_source_chroma = float(args.min_source_chroma)
    if min_source_chroma < 0:
        raise ValueError("min-source-chroma must be >= 0")

    discovered_sources = list_sources(clean_roots)
    if not discovered_sources:
        raise FileNotFoundError(f"No clean images found under: {clean_roots}")
    sources = select_source_subset(discovered_sources, args.max_sources, args.seed)
    prepare_output(output_root, args.overwrite)
    audit_rows = audit_sources(sources, clean_roots, args.crop_size, args.allow_upscale_small, min_source_chroma)
    splits = split_sources(audit_rows, args.seed)
    source_split = {row["source_id"]: split for split, split_rows in splits.items() for row in split_rows}
    for row in audit_rows:
        row["split"] = source_split.get(row["source_id"], "")
    write_csv(
        output_root / "source_audit.csv",
        audit_rows,
        ["source_id", "source_path", "width", "height", "mean_chroma", "accepted", "reason", "split"],
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
                degradation_subprofile = str(degradation_metadata.get("subprofile") or "mixed")
                identity_sample = int(degradation_subprofile == "identity")
                clean_mean_chroma = float(degradation_metadata.get("clean_mean_chroma", mean_lab_chroma(clean)))
                input_mean_chroma = mean_lab_chroma(model_input)
                target = clean.copy()
                target_metadata = {
                    "profile": TARGET_PROFILE,
                    "transform": TARGET_TRANSFORM,
                    "reason": "original_clean_rgb_crop",
                }
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
                        "degradation_subprofile": degradation_subprofile,
                        "identity_sample": identity_sample,
                        "clean_mean_chroma": clean_mean_chroma,
                        "input_mean_chroma": input_mean_chroma,
                        "chroma_retention": input_mean_chroma / max(clean_mean_chroma, 1e-6),
                        "target_profile": TARGET_PROFILE,
                        "target_transform": TARGET_TRANSFORM,
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
                            args.quality_mode,
                        )
                    )

    manifest_fields = [
        "sample_id", "source_id", "source_path", "split", "variant", "seed", "input_path", "target_path", "clean_path",
        "raw_degraded_path", "crop_size", "quality_mode", "degradation_profile", "degradation_subprofile",
        "identity_sample", "clean_mean_chroma", "input_mean_chroma", "chroma_retention", "target_profile",
        "target_transform", "quality_metadata", "degradation_metadata", "target_metadata",
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
    subprofile_counts = {
        name: sum(1 for row in manifest if row["degradation_subprofile"] == name)
        for name in sorted({str(row["degradation_subprofile"]) for row in manifest})
    }
    metadata = {
        "dataset_id": output_root.name,
        "created_at": created_at,
        "clean_roots": [str(path) for path in clean_roots],
        "crop_size": args.crop_size,
        "seed": args.seed,
        "allow_upscale_small": args.allow_upscale_small,
        "save_raw_degraded": args.save_raw_degraded,
        "quality_mode": args.quality_mode,
        "degradation_profile": args.degradation_profile,
        "min_source_chroma": min_source_chroma,
        "target_profile": TARGET_PROFILE,
        "target_transform": TARGET_TRANSFORM,
        "input_profile": (
            f"synthetic_{args.degradation_profile}_direct"
            if args.quality_mode == "off"
            else f"synthetic_{args.degradation_profile}_after_{args.quality_mode}"
        ),
        "training_objective": "clean_rgb_color_restoration",
        "preview_panels": (
            ["clean_target", "synthetic_degraded_input"]
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
        "subprofile_counts": subprofile_counts,
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
