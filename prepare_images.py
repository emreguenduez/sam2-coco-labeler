#!/usr/bin/env python3
"""Normalise raw camera photos into a clean dataset folder before labeling.

- converts HEIC/HEIF (iPhone) and other formats to JPEG
- bakes the EXIF rotation into the pixels, so every tool sees the same orientation
- optionally downsizes (long side) to keep SAM2 and the COCO file fast
- names files with a running index plus an optional tag (e.g. camera / location)

Usage:
    .venv/bin/python prepare_images.py RAW_DIR DATASET_DIR [--tag kitchen_iphone] [--max-side 2048]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    register_heif_opener = None

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--tag", default="", help="added to file names, e.g. 'lab_daylight_iphone'")
    parser.add_argument("--max-side", type=int, default=2048, help="0 keeps the original resolution")
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    next_index = 1 + max(
        (int(p.stem.split("_")[0]) for p in args.dataset_dir.glob("*.jpg") if p.stem.split("_")[0].isdigit()),
        default=0,
    )
    sources = sorted(p for p in args.raw_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    for source in sources:
        if source.suffix.lower() in {".heic", ".heif"} and register_heif_opener is None:
            raise SystemExit("HEIC photos found: run  .venv/bin/pip install pillow-heif  first.")
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if args.max_side and max(image.size) > args.max_side:
                image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            name = f"{next_index:04d}" + (f"_{args.tag}" if args.tag else "") + ".jpg"
            image.save(args.dataset_dir / name, quality=95)
            print(f"{source.name} -> {name} {image.size[0]}x{image.size[1]}")
            next_index += 1
    print(f"{len(sources)} image(s) written to {args.dataset_dir}")


if __name__ == "__main__":
    main()
