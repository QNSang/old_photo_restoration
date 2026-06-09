from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.restoration.color_restoration import run_color_restoration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled color-restoration inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output)
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")
    restored, metadata = run_color_restoration(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        checkpoint_path,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        tile_batch_size=args.tile_batch_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"output: {output_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
