#!/usr/bin/env python3
"""Draw the masks, boxes and labels from a COCO file onto its images.

Usage:
    .venv/bin/python visualize_coco.py path/to/annotations_coco_objects.json [--images DIR] [--out DIR]

Images are looked up relative to the JSON file (or --images). Overlays are written
to <json folder>/visualized/ (or --out) and a per-class summary is printed.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def decode_rle(segmentation: dict, height: int, width: int) -> np.ndarray:
    counts = segmentation["counts"]
    if isinstance(counts, str):  # compressed RLE -> use pycocotools
        from pycocotools import mask as mask_utils

        return mask_utils.decode(segmentation).astype(bool)
    flat = np.zeros(height * width, dtype=bool)
    position, value = 0, False
    for run in counts:
        if value:
            flat[position : position + run] = True
        position += run
        value = not value
    return flat.reshape((height, width), order="F")


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        return decode_rle(segmentation, height, width)
    canvas = Image.new("L", (width, height), 0)  # polygon list
    draw = ImageDraw.Draw(canvas)
    for polygon in segmentation:
        draw.polygon(list(map(float, polygon)), fill=1)
    return np.array(canvas, dtype=bool)


def color_for(index: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((index * 0.618033988749895) % 1.0, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coco_json", type=Path)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.coco_json.read_text())
    image_root = args.images or args.coco_json.parent
    out_dir = args.out or args.coco_json.parent / "visualized"
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    per_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        if "segmentation" in ann:
            per_image.setdefault(ann["image_id"], []).append(ann)

    class_counts: Counter[str] = Counter()
    for image in data.get("images", []):
        path = image_root / image["file_name"]
        if not path.exists():
            print(f"missing image: {path}")
            continue
        base = Image.open(path).convert("RGB")
        if base.size != (image["width"], image["height"]):
            # Phone photos carry an EXIF rotation; the app works on the raw pixels.
            rotated = ImageOps.exif_transpose(base)
            if rotated.size == (image["width"], image["height"]):
                base = rotated
        canvas = np.array(base, dtype=np.float32)
        h, w = canvas.shape[:2]
        anns = per_image.get(image["id"], [])
        for ann in anns:
            mask = decode_segmentation(ann["segmentation"], h, w)
            color = np.array(color_for(ann["category_id"]), dtype=np.float32)
            canvas[mask] = canvas[mask] * 0.45 + color * 0.55
            class_counts[categories.get(ann["category_id"], str(ann["category_id"]))] += 1
        result = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(result)
        font_size = max(14, w // 60)
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        line = max(2, w // 500)
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            color = color_for(ann["category_id"])
            name = categories.get(ann["category_id"], str(ann["category_id"]))
            draw.rectangle([x, y, x + bw, y + bh], outline=color, width=line)
            text_box = draw.textbbox((x, y), name, font=font)
            label_top = max(0, y - (text_box[3] - text_box[1]) - 2 * line)
            text_box = draw.textbbox((x + line, label_top), name, font=font)
            draw.rectangle([text_box[0] - line, text_box[1] - line, text_box[2] + line, text_box[3] + line], fill=color)
            draw.text((x + line, label_top), name, fill=(0, 0, 0), font=font)
        target = out_dir / (Path(image["file_name"]).stem + "_annotated.jpg")
        result.save(target, quality=92)
        print(f"{image['file_name']}: {len(anns)} objects -> {target}")

    print("\nObjects per class:")
    for name, count in class_counts.most_common():
        print(f"  {name:20s} {count}")
    print(f"Images: {len(data.get('images', []))} | annotations: {sum(class_counts.values())}")


if __name__ == "__main__":
    main()
