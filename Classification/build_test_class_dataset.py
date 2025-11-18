#!/usr/bin/env python3
"""
Build a classifier-ready test dataset (adipose vs not-adipose).

Adapts the tiling/mask pipeline from build_class_dataset.py but targets the
/_test folder (e.g., clean_test_class) and keeps all tiles (no white/blurry filtering).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so src imports work when script is run from Classification/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.seed_utils import get_project_seed

try:
    from src.utils.stain_normalization import (
        ReinhardStainNormalizer,
        load_best_reference,
    )

    STAIN_NORMALIZATION_AVAILABLE = True
except ImportError:
    STAIN_NORMALIZATION_AVAILABLE = False

GLOBAL_SEED = get_project_seed()


@dataclass
class SlideRecord:
    base: str
    image_path: Path
    json_path: Path
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Build classifier test dataset (adipose vs not-adipose)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing pseudocolored .jpg tiles (e.g., Pseudocolored/test).")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Directory containing fat JSON annotations for the same slides.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Target directory (e.g., data/.../_test/original/clean_test_class).")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--adipose-threshold", type=float, default=0.10,
                        help="Minimum fat mask coverage for a tile to be labeled adipose.")
    parser.add_argument("--white-threshold", type=int, default=235,
                        help="Pixel intensity threshold for white detection.")
    parser.add_argument("--white-ratio-limit", type=float, default=0.70,
                        help="Max ratio of white pixels before tile is classified as white.")
    parser.add_argument("--blurry-threshold", type=float, default=7.5,
                        help="Minimum Laplacian variance for non-blurry tiles.")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--min-confidence", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--stain-normalize", action="store_true", default=True)
    parser.add_argument("--no-stain-normalize", action="store_false", dest="stain_normalize")
    parser.add_argument("--reference-metadata", type=str, default="src/utils/stain_reference_metadata.json")
    parser.add_argument("--reference-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args()


def parse_image_stem_from_json(json_path: Path, cls: str = "fat") -> str:
    stem = json_path.stem
    token = f"_{cls}_annotations"
    if token in stem:
        return stem.split(token)[0]
    token2 = f"_{cls}_"
    if token2 in stem:
        return stem.split(token2)[0]
    return stem


@lru_cache(maxsize=256)
def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (width, height)


@lru_cache(maxsize=1024)
def re_search_cached(pattern: str, text: str):
    return re.search(pattern, text)


def extract_filename_timestamp(json_path: Path) -> Optional[datetime]:
    filename = json_path.stem
    pattern = r"_(\d{8})_(\d{6})$"
    match = re_search_cached(pattern, filename)
    if not match:
        return None
    date_str, time_str = match.groups()
    try:
        month = int(date_str[:2])
        day = int(date_str[2:4])
        year = int(date_str[4:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def load_json_annotations(json_path: Path, min_confidence: int) -> List[np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload if isinstance(payload, list) else [payload]
    polys: List[np.ndarray] = []
    for ann in records:
        if not isinstance(ann, dict):
            continue
        confidence = ann.get("confidenceScore")
        if confidence is not None and confidence < min_confidence:
            continue
        elements = ann.get("annotation", {}).get("elements", [])
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            if elem.get("type") != "polyline":
                continue
            pts = elem.get("points", [])
            if pts and len(pts) >= 3:
                coords = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)
                if len(coords) >= 3:
                    polys.append(coords)
    return polys


def create_binary_mask(polygons: List[np.ndarray], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    cv_polys = [np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2) for poly in polygons]
    if cv_polys:
        cv2.fillPoly(mask, cv_polys, 1)
    return mask


def tile_coords(height: int, width: int, tile: int, stride: int) -> List[Tuple[int, int, int, int]]:
    if height < tile or width < tile:
        return []
    if stride >= tile:
        x_steps = width // tile
        y_steps = height // tile
    else:
        x_steps = max(1, math.ceil((width - tile) / stride) + 1)
        y_steps = max(1, math.ceil((height - tile) / stride) + 1)

    coords: List[Tuple[int, int, int, int]] = []
    for ri in range(y_steps):
        for ci in range(x_steps):
            xs = min(ci * stride, width - tile)
            ys = min(ri * stride, height - tile)
            if xs >= 0 and ys >= 0 and xs + tile <= width and ys + tile <= height:
                coords.append((ri, ci, ys, xs))
    return coords


def jpeg_params(quality: int) -> List[int]:
    return [
        cv2.IMWRITE_JPEG_QUALITY,
        int(quality),
        cv2.IMWRITE_JPEG_PROGRESSIVE,
        0,
    ]


def extract_tile(img_path: Path, x: int, y: int, size: int,
                 stain_normalizer: Optional[ReinhardStainNormalizer]) -> np.ndarray:
    with Image.open(img_path) as img:
        tile_rgb = np.array(img.crop((x, y, x + size, y + size)))
    if stain_normalizer is not None:
        try:
            tile_rgb = stain_normalizer.normalize_image(tile_rgb)
            if tile_rgb.max() <= 1.0:
                tile_rgb = (tile_rgb * 255).astype(np.uint8)
            else:
                tile_rgb = tile_rgb.astype(np.uint8)
        except Exception as exc:
            print(f"[Stain] WARN: normalization failed at ({x},{y}) in {img_path.name}: {exc}")
    return cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)


def evaluate_quality(tile_bgr: np.ndarray, white_threshold: int,
                     white_ratio_limit: float, blurry_threshold: float) -> Tuple[str, float, float]:
    """Evaluate tile quality (white/blurry/tissue)."""
    white_mask = np.all(tile_bgr >= white_threshold, axis=2)
    white_ratio = float(white_mask.mean())
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if white_ratio > white_ratio_limit:
        return "white", white_ratio, lap_var
    if lap_var < blurry_threshold:
        return "blurry", white_ratio, lap_var
    return "tissue", white_ratio, lap_var


def discover_slides(images_dir: Path, masks_dir: Path, min_confidence: int) -> Dict[str, SlideRecord]:
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        raise SystemExit(f"No .jpg files found in {images_dir}")
    image_stems = {p.stem for p in image_files}

    json_files = list(masks_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json files found in {masks_dir}")

    latest_by_base: Dict[str, Path] = {}
    for jpath in json_files:
        base = parse_image_stem_from_json(jpath, "fat")
        if base not in image_stems:
            continue
        latest_by_base.setdefault(base, []).append(jpath)

    selections: Dict[str, SlideRecord] = {}
    for base, files in latest_by_base.items():
        if len(files) == 1:
            chosen = files[0]
        else:
            with_ts: List[Tuple[datetime, Path]] = []
            without_ts: List[Path] = []
            for jpath in files:
                ts = extract_filename_timestamp(jpath)
                if ts:
                    with_ts.append((ts, jpath))
                else:
                    without_ts.append(jpath)
            if with_ts:
                with_ts.sort(key=lambda x: x[0], reverse=True)
                chosen = with_ts[0][1]
            else:
                without_ts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                chosen = without_ts[0]

        img_path = images_dir / f"{base}.jpg"
        if not img_path.exists():
            continue
        width, height = get_image_dimensions(str(img_path))
        selections[base] = SlideRecord(base, img_path, chosen, width, height)

    if not selections:
        raise SystemExit("No matching image/JSON pairs discovered.")

    print(f"[Discovery] Found {len(selections)} slide(s) for classifier test build.")
    return selections


def init_stain_normalizer(args: argparse.Namespace) -> Optional[ReinhardStainNormalizer]:
    if not getattr(args, "stain_normalize", False):
        return None
    if not STAIN_NORMALIZATION_AVAILABLE:
        print("[Stain] WARN: stain normalization requested but modules unavailable.")
        return None
    try:
        if args.reference_path:
            print(f"[Stain] Using explicit reference: {args.reference_path}")
            return ReinhardStainNormalizer(args.reference_path)
        print(f"[Stain] Loading best reference from {args.reference_metadata}")
        return load_best_reference(args.reference_metadata)
    except Exception as exc:
        print(f"[Stain] WARN: failed to initialize stain normalizer: {exc}")
        return None


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    adipose_dir = output_dir / "adipose"
    not_dir = output_dir / "not_adipose"
    adipose_dir.mkdir(parents=True, exist_ok=True)
    not_dir.mkdir(parents=True, exist_ok=True)
    return {"adipose": adipose_dir, "not_adipose": not_dir, "output_root": output_dir}


def process_slide(slide: SlideRecord, args: argparse.Namespace,
                  stain_normalizer, out_dirs, manifest_rows: List[List[str]]):
    polys = load_json_annotations(slide.json_path, args.min_confidence)
    mask = create_binary_mask(polys, slide.width, slide.height)
    coords = tile_coords(slide.height, slide.width, args.tile_size, args.stride)
    if not coords:
        return
    jpeg_settings = jpeg_params(args.jpeg_quality)

    for row, col, ys, xs in coords:
        tile = extract_tile(slide.image_path, xs, ys, args.tile_size, stain_normalizer)
        
        # Evaluate tile quality (white/blurry filtering)
        quality, white_ratio, lap_var = evaluate_quality(
            tile, args.white_threshold, args.white_ratio_limit, args.blurry_threshold
        )
        
        # Filtered tiles (white/blurry) go to not_adipose with 0% adipose ratio
        if quality in ("white", "blurry"):
            label = "not_adipose"
            adipose_ratio = 0.0
        else:
            # Normal classification based on mask
            mask_patch = mask[ys:ys + args.tile_size, xs:xs + args.tile_size]
            adipose_ratio = float(mask_patch.mean()) if mask_patch.size else 0.0
            label = "adipose" if adipose_ratio >= args.adipose_threshold else "not_adipose"
        
        out_name = f"{slide.base}_r{row}_c{col}.jpg"
        out_path = out_dirs[label] / out_name
        cv2.imwrite(str(out_path), tile, jpeg_settings)
        rel = out_path.relative_to(out_dirs["output_root"])
        manifest_rows.append([
            str(rel),
            label,
            f"{adipose_ratio:.6f}",
        ])


def write_manifest(manifest_rows: List[List[str]], output_root: Path):
    if not manifest_rows:
        return
    csv_path = output_root / "test_manifest.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,label,adipose_ratio\n")
        for row in manifest_rows:
            f.write(",".join(row) + "\n")
    print(f"[Output] Wrote manifest: {csv_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Images: {images_dir}")
    print(f"[Setup] Masks:  {masks_dir}")
    print(f"[Setup] Output: {output_dir}")

    slides = discover_slides(images_dir, masks_dir, args.min_confidence)
    stain_normalizer = init_stain_normalizer(args)
    out_dirs = ensure_output_dirs(output_dir)

    manifest_rows: List[List[str]] = []
    for slide in tqdm(sorted(slides.values(), key=lambda s: s.base), desc="Processing slides"):
        process_slide(slide, args, stain_normalizer, out_dirs, manifest_rows)

    write_manifest(manifest_rows, out_dirs["output_root"])

    print("\n[Summary]")
    for label in ("adipose", "not_adipose"):
        count = len(list(out_dirs[label].glob("*.jpg")))
        print(f"  {label:12s}: {count}")
    print(f"  Output root: {out_dirs['output_root']}")
    print("Done.")


if __name__ == "__main__":
    main()
