#!/usr/bin/env python3
"""
Build a classifier-ready test dataset (adipose vs not-adipose).

Adapts the tiling/mask pipeline from build_class_dataset.py but targets the
/_test folder (e.g., clean_test_class) and keeps all tiles (no white/blurry filtering).

Differences from build_class_dataset.py:
  - No train/val/test split (all tiles in root adipose/not_adipose)
  - Default: keep all tiles (trusts annotator judgment)
  - Can include ambiguous tiles (0 < adipose_ratio < threshold)
  - Designed for held-out test sets with manual annotations

USAGE EXAMPLES:

1. Build standard test set (excludes ambiguous, keeps all quality):
   python Classification/build_test_class_dataset.py \
     --images-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/ECM_channel \
     --masks-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/Masks/fat \
     --output-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/_test/ecm_2_test \
     --stain-normalize false

2. Include ambiguous tiles for edge case testing:
   python Classification/build_test_class_dataset.py \
     --images-dir data/Meat_MS_Tulane/ECM_channel \
     --masks-dir data/Meat_MS_Tulane/Masks/fat \
     --output-dir data/Meat_MS_Tulane/_test/ecm_2_all_annotations \
     --stain-normalize false \
     --include-ambiguous true \
     --min-confidence 1

3. Strict quality filtering (exclude white and blurry):
   python Classification/build_test_class_dataset.py \
     --images-dir data/Meat_MS_Tulane/ECM_channel \
     --masks-dir data/Meat_MS_Tulane/Masks/fat \
     --output-dir data/Meat_MS_Tulane/_test/ecm_2_strict \
     --stain-normalize false \
     --keep-white false \
     --keep-blurry false \
     --white-ratio-limit 0.70 \
     --blurry-threshold 10.0

4. High confidence annotations only:
   python Classification/build_test_class_dataset.py \
     --images-dir data/Meat_Luci_Tulane/Pseudocolored \
     --masks-dir data/Meat_Luci_Tulane/Masks/fat \
     --output-dir data/Meat_Luci_Tulane/_test/pseudocolored_high_conf \
     --stain-normalize true \
     --min-confidence 3

OUTPUT STRUCTURE:
  output_dir/
    ├── adipose/              # Tiles with adipose >= threshold
    │   └── *.jpg
    ├── not_adipose/          # Tiles with adipose < threshold
    │   └── *.jpg
    ├── test_manifest.csv     # All tiles with metadata
    ├── build_log.json        # Build parameters and statistics
    └── build_summary.txt     # Human-readable summary
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
import tifffile
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
    parser.add_argument("--adipose-threshold", type=float, default=0.025,
                        help="Minimum fat mask coverage for a tile to be labeled adipose.")
    parser.add_argument("--white-threshold", type=int, default=245,
                        help="Pixel intensity threshold for white detection.")
    parser.add_argument("--white-ratio-limit", type=float, default=0.70,
                        help="Max ratio of white pixels before tile is classified as white.")
    parser.add_argument("--blurry-threshold", type=float, default=7.5,
                        help="Minimum Laplacian variance for non-blurry tiles.")
    parser.add_argument("--keep-white", type=lambda x: x.lower() == 'true', default=True,
                        metavar='BOOL',
                        help="Keep white tiles (true/false, default: true - trusts annotator judgment).")
    parser.add_argument("--keep-blurry", type=lambda x: x.lower() == 'true', default=True,
                        metavar='BOOL',
                        help="Keep blurry tiles (true/false, default: true - trusts annotator judgment).")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--min-confidence", type=int, default=2, choices=[1, 2, 3],
                        help="Minimum annotation confidence level (default: 2 - medium/high, excludes conf=1 uncertain annotations).")
    parser.add_argument("--include-ambiguous", type=lambda x: x.lower() == 'true', default=False,
                        metavar='BOOL',
                        help="Include ambiguous tiles (0 < adipose_ratio < threshold) in test set (true/false, default: false - excludes ambiguous).")
    parser.add_argument("--stain-normalize", type=lambda x: x.lower() == 'true', required=True,
                        metavar='BOOL',
                        help="Apply SYBR Gold + Eosin stain normalization (true/false, required).")
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


def slide_has_valid_annotations(json_path: Path, min_confidence: int) -> bool:
    """Check if slide has at least one annotation meeting confidence threshold.
    
    This ensures we only process slides that were actually annotated.
    Slides with no valid annotations are skipped entirely.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload if isinstance(payload, list) else [payload]
    
    for ann in records:
        if not isinstance(ann, dict):
            continue
        confidence = ann.get("confidenceScore")
        # Check if this annotation meets confidence threshold
        if confidence is None or confidence >= min_confidence:
            # Check if it has valid polygon data
            elements = ann.get("annotation", {}).get("elements", [])
            for elem in elements:
                if not isinstance(elem, dict):
                    continue
                if elem.get("type") == "polyline":
                    pts = elem.get("points", [])
                    if pts and len(pts) >= 3:
                        return True  # Found at least one valid annotation
    return False  # No valid annotations found


def get_tile_annotations(json_path: Path, tile_bbox: Tuple[int, int, int, int], min_confidence: int) -> Tuple[List[np.ndarray], bool]:
    """Get annotations within tile bounds and flag tiles that only contain low-confidence marks.
    
    Returns:
        (polygons, low_confidence_only): polygons meeting the confidence threshold (shifted to tile coords) and
        a flag indicating that the tile intersected *only* low-confidence annotations.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload if isinstance(payload, list) else [payload]
    polys: List[np.ndarray] = []
    has_low_conf = False
    has_high_conf = False
    x1, y1, x2, y2 = tile_bbox
    
    for ann in records:
        if not isinstance(ann, dict):
            continue
        confidence = ann.get("confidenceScore")
        
        elements = ann.get("annotation", {}).get("elements", [])
        for elem in elements:
            if not isinstance(elem, dict) or elem.get("type") != "polyline":
                continue
            pts = elem.get("points", [])
            if not pts or len(pts) < 3:
                continue
                
            # Check if polygon intersects with tile
            coords = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)
            if len(coords) < 3:
                continue
                
            # Simple bounding box intersection check
            poly_xs = coords[:, 0]
            poly_ys = coords[:, 1]
            if (poly_xs.max() < x1 or poly_xs.min() > x2 or 
                poly_ys.max() < y1 or poly_ys.min() > y2):
                continue  # Polygon doesn't intersect tile
            
            # Polygon intersects tile - check confidence
            if confidence is not None and confidence < min_confidence:
                has_low_conf = True
                continue
            # Shift polygon coordinates to tile-local space
            local_coords = coords - np.array([x1, y1])
            polys.append(local_coords)
            has_high_conf = True
    
    # Only treat as low-confidence skip if we saw low-confidence marks and no usable high-confidence polygons
    return polys, (has_low_conf and not has_high_conf)


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
    # Use tifffile for TIFF files, PIL for others
    if img_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            full_img = tifffile.imread(str(img_path))
            # Normalize if 16-bit
            if full_img.dtype == np.uint16:
                full_img = (full_img / 256).astype(np.uint8)
            # Extract tile
            tile = full_img[y:y + size, x:x + size]
        except Exception as e:
            print(f"[TIFF] WARN: tifffile failed for {img_path.name}, falling back to PIL: {e}")
            with Image.open(img_path) as img:
                tile = np.array(img.crop((x, y, x + size, y + size)))
    else:
        # Use PIL for PNG, JPG, etc.
        with Image.open(img_path) as img:
            tile = np.array(img.crop((x, y, x + size, y + size)))
    
    # Handle both grayscale and RGB images
    if tile.ndim == 2:
        # Grayscale image - convert to RGB for stain normalization if needed
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
    elif tile.ndim == 3:
        tile_rgb = tile
    else:
        raise ValueError(f"Unexpected image dimensions: {tile.shape}")
    
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
    # Collect images with multiple extensions
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        image_files.extend(images_dir.glob(ext))
    
    if not image_files:
        raise SystemExit(f"No image files (.jpg, .jpeg, .png, .tif, .tiff) found in {images_dir}")
    
    # Create mapping from stem to full path
    image_by_stem = {p.stem: p for p in image_files}

    json_files = list(masks_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json files found in {masks_dir}")

    latest_by_base: Dict[str, Path] = {}
    for jpath in json_files:
        base = parse_image_stem_from_json(jpath, "fat")
        if base not in image_by_stem:
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

        # Get the actual image path from the stem mapping
        img_path = image_by_stem[base]
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
                  stain_normalizer, out_dirs, manifest_rows: List[List[str]], stats: Dict[str, int]):
    # Skip slides with no valid annotations (ensures negatives only from annotated slides)
    if not slide_has_valid_annotations(slide.json_path, args.min_confidence):
        print(f"[Skip] {slide.base}: no annotations meeting confidence threshold {args.min_confidence}")
        stats["skipped_slides_no_annotations"] += 1
        return
    
    skipped_low_conf = 0
    coords = tile_coords(slide.height, slide.width, args.tile_size, args.stride)
    if not coords:
        stats["skipped_slides_small"] += 1
        return
    jpeg_settings = jpeg_params(args.jpeg_quality)

    for row, col, ys, xs in coords:
        tile = extract_tile(slide.image_path, xs, ys, args.tile_size, stain_normalizer)
        
        # Check for low-confidence annotations in this tile
        tile_bbox = (xs, ys, xs + args.tile_size, ys + args.tile_size)
        tile_polys, has_low_conf = get_tile_annotations(slide.json_path, tile_bbox, args.min_confidence)
        
        # Skip tiles that only have low-confidence annotations
        if has_low_conf:
            skipped_low_conf += 1
            stats["skipped_low_confidence"] += 1
            continue
        
        # Evaluate tile quality
        quality, white_ratio, lap_var = evaluate_quality(
            tile, args.white_threshold, args.white_ratio_limit, args.blurry_threshold
        )
        
        # Create mask from tile-level annotations and classify
        tile_mask = create_binary_mask(tile_polys, args.tile_size, args.tile_size)
        adipose_ratio = float(tile_mask.mean()) if tile_mask.size else 0.0
        
        # Exclude ambiguous tiles by default: has some fat but below threshold
        # These are neither clear positives nor clear negatives
        # With --include-ambiguous true: keep as not_adipose to test edge cases
        # With --include-ambiguous false: exclude entirely
        if 0 < adipose_ratio < args.adipose_threshold:
            if not args.include_ambiguous:
                stats["skipped_ambiguous"] += 1
                continue  # Skip ambiguous tiles
            # Otherwise keep as not_adipose (below threshold, worth testing)
        
        label = "adipose" if adipose_ratio >= args.adipose_threshold else "not_adipose"
        
        # Filtering logic:
        # - ALWAYS keep positive (adipose) tiles, regardless of quality
        # - For negative (not_adipose) tiles, apply quality filters if enabled
        # - Default: keep all tiles (trusts annotator judgment)
        if label == "not_adipose":
            if quality == "white" and not args.keep_white:
                stats["skipped_white"] += 1
                continue  # Skip white negatives
            if quality == "blurry" and not args.keep_blurry:
                stats["skipped_blurry"] += 1
                continue  # Skip blurry negatives
        
        out_name = f"{slide.base}_r{row}_c{col}.jpg"
        out_path = out_dirs[label] / out_name
        cv2.imwrite(str(out_path), tile, jpeg_settings)
        rel = out_path.relative_to(out_dirs["output_root"])
        manifest_rows.append([
            str(rel),
            label,
            f"{adipose_ratio:.6f}",
            f"{white_ratio:.6f}",
            f"{lap_var:.3f}",
            quality,
        ])
        stats["tiles_written"] += 1
    
    if skipped_low_conf > 0:
        print(f"[Skip] {slide.base}: {skipped_low_conf} tiles with low-confidence annotations excluded")


def write_manifest(manifest_rows: List[List[str]], output_root: Path):
    if not manifest_rows:
        return
    csv_path = output_root / "test_manifest.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,label,adipose_ratio,white_ratio,laplacian_var,quality\n")
        for row in manifest_rows:
            f.write(",".join(row) + "\n")
    print(f"[Output] Wrote manifest: {csv_path}")


def write_build_log(args: argparse.Namespace, output_root: Path, slides: Dict[str, SlideRecord], 
                    manifest_rows: List[List[str]], stats: Dict[str, int]):
    """Write comprehensive build log documenting test set parameters."""
    timestamp = datetime.now().isoformat()
    
    # Count tiles by label
    label_counts = {"adipose": 0, "not_adipose": 0}
    for row in manifest_rows:
        label = row[1]
        if label in label_counts:
            label_counts[label] += 1
    
    # JSON log
    log_data = {
        "build_info": {
            "script": "build_test_class_dataset.py",
            "purpose": "Classifier test dataset (adipose vs not_adipose)",
            "timestamp": timestamp,
            "output_directory": str(output_root)
        },
        "input_paths": {
            "images_directory": args.images_dir,
            "masks_directory": args.masks_dir
        },
        "processing_parameters": {
            "tile_size": args.tile_size,
            "stride": args.stride,
            "adipose_threshold": args.adipose_threshold,
            "white_threshold": args.white_threshold,
            "white_ratio_limit": args.white_ratio_limit,
            "blurry_threshold": args.blurry_threshold,
            "jpeg_quality": args.jpeg_quality,
            "min_confidence": args.min_confidence,
            "stain_normalize": args.stain_normalize,
            "reference_metadata": args.reference_metadata,
            "reference_path": args.reference_path,
            "seed": args.seed
        },
        "dataset_statistics": {
            "num_slides": len(slides),
            "total_tiles": len(manifest_rows),
            "adipose_tiles": label_counts["adipose"],
            "not_adipose_tiles": label_counts["not_adipose"],
            "adipose_percentage": 100 * label_counts["adipose"] / len(manifest_rows) if manifest_rows else 0
        },
        "skips": {
            "slides_no_annotations": stats["skipped_slides_no_annotations"],
            "slides_too_small": stats["skipped_slides_small"],
            "tiles_low_confidence_only": stats["skipped_low_confidence"],
            "tiles_ambiguous": stats["skipped_ambiguous"],
            "tiles_white": stats["skipped_white"],
            "tiles_blurry": stats["skipped_blurry"],
        },
        "slides_processed": sorted(slides.keys())
    }
    
    json_path = output_root / "build_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    
    # Human-readable summary
    summary_path = output_root / "build_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFIER TEST DATASET BUILD LOG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Build Timestamp: {timestamp}\n")
        f.write(f"Output Directory: {output_root}\n\n")
        
        f.write("INPUT SOURCES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Images: {args.images_dir}\n")
        f.write(f"Masks:  {args.masks_dir}\n\n")
        
        f.write("PROCESSING PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tile Size:          {args.tile_size}x{args.tile_size} pixels\n")
        f.write(f"Stride:             {args.stride} pixels\n")
        f.write(f"Adipose Threshold:  {args.adipose_threshold:.2f} (minimum mask coverage)\n")
        f.write(f"White Threshold:    {args.white_threshold}\n")
        f.write(f"White Ratio Limit:  {args.white_ratio_limit:.2f}\n")
        f.write(f"Blur Threshold:     {args.blurry_threshold:.1f}\n")
        f.write(f"JPEG Quality:       {args.jpeg_quality}\n")
        f.write(f"Min Confidence:     {args.min_confidence}\n")
        f.write(f"Stain Normalize:    {args.stain_normalize}\n")
        f.write(f"Random Seed:        {args.seed}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Slides Processed:   {len(slides)}\n")
        f.write(f"Total Tiles:        {len(manifest_rows)}\n")
        f.write(f"Adipose Tiles:      {label_counts['adipose']} ({100 * label_counts['adipose'] / len(manifest_rows):.1f}%)\n")
        f.write(f"Not-Adipose Tiles:  {label_counts['not_adipose']} ({100 * label_counts['not_adipose'] / len(manifest_rows):.1f}%)\n\n")
        
        f.write("DROPS / SKIPS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Slides (no annotations >= conf): {stats['skipped_slides_no_annotations']}\n")
        f.write(f"Slides (too small for tiling):   {stats['skipped_slides_small']}\n")
        f.write(f"Tiles (low confidence only):     {stats['skipped_low_confidence']}\n")
        f.write(f"Tiles (ambiguous fat coverage):  {stats['skipped_ambiguous']}\n")
        f.write(f"Tiles (white negatives):         {stats['skipped_white']}\n")
        f.write(f"Tiles (blurry negatives):        {stats['skipped_blurry']}\n\n")
        
        f.write("SLIDES PROCESSED:\n")
        f.write("-" * 40 + "\n")
        for slide_name in sorted(slides.keys()):
            f.write(f"  {slide_name}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("For detailed technical parameters, see build_log.json\n")
        f.write("=" * 80 + "\n")
    
    print(f"[Log] Build log saved: {json_path}")
    print(f"[Log] Build summary saved: {summary_path}")


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

    stats = {
        "tiles_written": 0,
        "skipped_slides_no_annotations": 0,
        "skipped_slides_small": 0,
        "skipped_low_confidence": 0,
        "skipped_ambiguous": 0,
        "skipped_white": 0,
        "skipped_blurry": 0,
    }
    manifest_rows: List[List[str]] = []
    for slide in tqdm(sorted(slides.values(), key=lambda s: s.base), desc="Processing slides"):
        process_slide(slide, args, stain_normalizer, out_dirs, manifest_rows, stats)

    write_manifest(manifest_rows, out_dirs["output_root"])
    write_build_log(args, out_dirs["output_root"], slides, manifest_rows, stats)

    print("\n[Summary]")
    for label in ("adipose", "not_adipose"):
        count = len(list(out_dirs[label].glob("*.jpg")))
        print(f"  {label:12s}: {count}")
    print(f"  Tiles written: {stats['tiles_written']}")
    print(f"  Skipped - low confidence only: {stats['skipped_low_confidence']}")
    print(f"  Skipped - ambiguous:           {stats['skipped_ambiguous']}")
    print(f"  Skipped - white:               {stats['skipped_white']}")
    print(f"  Skipped - blurry:              {stats['skipped_blurry']}")
    print(f"  Slides skipped (no annotations >= conf): {stats['skipped_slides_no_annotations']}")
    print(f"  Slides skipped (too small):               {stats['skipped_slides_small']}")
    print(f"  Output root: {out_dirs['output_root']}")
    print("Done.")


if __name__ == "__main__":
    main()
