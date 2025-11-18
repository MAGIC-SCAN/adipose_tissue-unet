#!/usr/bin/env python3
"""
Build a tile classification dataset for adipose vs. not-adipose tissue.

The script mirrors key conveniences from build_dataset.py (timestamped builds,
confidence filtering, white/blurry quality checks, optional stain normalization)
but focuses on producing classification-ready JPEG tiles that can feed the
cell_classifier training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path to enable src imports
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

# Default data layout mirrors the segmentation builder, but rooted inside repo data/.
DEFAULT_DATA_ROOT = Path("/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane")
DEFAULT_REFERENCE_METADATA = "src/utils/stain_reference_metadata.json"

# Globals rebound at runtime after CLI parsing
DATA_ROOT = DEFAULT_DATA_ROOT
PSEUDO_DIR = DATA_ROOT / "Pseudocolored"
FAT_JSON_DIR = DATA_ROOT / "Masks" / "fat"
BUILD_ROOT: Optional[Path] = None


@dataclass
class SlideRecord:
    base: str
    image_path: Path
    json_path: Path
    width: int  # pixels
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Build adipose vs. not-adipose classifier dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--tile-size", type=int, default=1024,
                        help="Tile edge length in pixels (use 1024 so tiles can double as segmentation inputs).")
    parser.add_argument("--stride", type=int, default=1024,
                        help="Stride in pixels; defaults to non-overlapping 1024x1024 tiling.")
    parser.add_argument("--adipose-threshold", type=float, default=0.10,
                        help="Minimum adipose mask coverage (0-1) for a tile to be labeled adipose.")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="Ratio of slides reserved for test split (default: 0.0, use external test set).")
    parser.add_argument("--white-threshold", type=int, default=235)
    parser.add_argument("--white-ratio-limit", type=float, default=0.70)
    parser.add_argument("--blurry-threshold", type=float, default=7.5)
    parser.add_argument("--min-confidence", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--keep-white", action="store_true", default=True,
                        help="Keep white tiles as not_adipose (default: True for binary classification).")
    parser.add_argument("--drop-white", action="store_false", dest="keep_white",
                        help="Drop white tiles instead of keeping them.")
    parser.add_argument("--keep-blurry", action="store_true", default=True,
                        help="Keep blurry tiles as not_adipose (default: True for binary classification).")
    parser.add_argument("--drop-blurry", action="store_false", dest="keep_blurry",
                        help="Drop blurry tiles instead of keeping them.")
    parser.add_argument("--balance-classes", action="store_true", default=True,
                        help="Balance classes by undersampling majority class (default: True for binary classification).")
    parser.add_argument("--no-balance", dest="balance_classes", action="store_false",
                        help="Disable class balancing (keep natural imbalanced distribution).")
    parser.add_argument("--target-adipose-ratio", type=float, default=0.40,
                        help="Target ratio for adipose class when balancing (0.4 = 40%% adipose, 60%% not_adipose).")
    parser.add_argument("--stain-normalize", dest="stain_normalize", action="store_true", default=True,
                        help="Apply SYBR Gold + Eosin stain normalization before tiling.")
    parser.add_argument("--no-stain-normalize", action="store_false", dest="stain_normalize",
                        help="Disable stain normalization (overrides --stain-normalize).")
    parser.add_argument("--reference-path", type=str, default=None,
                        help="Explicit reference image for stain normalization.")
    parser.add_argument("--reference-metadata", type=str, default=DEFAULT_REFERENCE_METADATA,
                        help="Fallback metadata file used by load_best_reference().")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0:
        raise SystemExit("tile-size must be positive.")
    if args.stride <= 0:
        raise SystemExit("stride must be positive.")
    if not (0.0 < args.adipose_threshold <= 1.0):
        raise SystemExit("adipose-threshold must be in (0, 1].")
    if not (0.0 <= args.val_ratio < 1.0) or not (0.0 <= args.test_ratio < 1.0):
        raise SystemExit("val-ratio and test-ratio must be within [0, 1).")
    if args.val_ratio + args.test_ratio >= 0.95:
        raise SystemExit("val-ratio + test-ratio must leave room for training data.")


def setup_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    global DATA_ROOT, PSEUDO_DIR, FAT_JSON_DIR, BUILD_ROOT
    DATA_ROOT = Path(os.path.expanduser(args.data_root))
    PSEUDO_DIR = DATA_ROOT / "Pseudocolored"
    FAT_JSON_DIR = DATA_ROOT / "Masks" / "fat"

    missing = [p for p in (DATA_ROOT, PSEUDO_DIR, FAT_JSON_DIR) if not p.exists()]
    if missing:
        raise SystemExit(f"Required directory missing: {missing[0]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BUILD_ROOT = DATA_ROOT / f"_build_class_{timestamp}"
    BUILD_ROOT.mkdir(parents=True, exist_ok=False)

    dataset_dir = BUILD_ROOT / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_path = BUILD_ROOT / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[Setup] Build directory: {BUILD_ROOT}")
    return BUILD_ROOT, dataset_dir, config_path


def parse_image_stem_from_json(json_path: Path, cls: str = "fat") -> str:
    stem = json_path.stem
    token = f"_{cls}_annotations"
    if token in stem:
        return stem.split(token)[0]
    token2 = f"_{cls}_"
    if token2 in stem:
        return stem.split(token2)[0]
    return stem


def extract_filename_timestamp(json_path: Path) -> Optional[datetime]:
    filename = json_path.stem
    pattern = r"_(\d{8})_(\d{6})$"
    match = re_search_cached(pattern, filename)
    if not match:
        return None
    date_str, time_str = match.groups()
    try:
        month = int(date_str[0:2])
        day = int(date_str[2:4])
        year = int(date_str[4:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


@lru_cache(maxsize=1024)
def re_search_cached(pattern: str, text: str):
    return re.search(pattern, text)


@lru_cache(maxsize=256)
def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (width, height)


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


def discover_slides(min_confidence: int) -> Dict[str, SlideRecord]:
    if not FAT_JSON_DIR.exists():
        raise SystemExit(f"No fat JSON directory found at {FAT_JSON_DIR}")

    grouped: Dict[str, List[Path]] = {}
    for json_path in FAT_JSON_DIR.glob("*.json"):
        base = parse_image_stem_from_json(json_path, "fat")
        grouped.setdefault(base, []).append(json_path)

    selections: Dict[str, SlideRecord] = {}
    for base, files in grouped.items():
        chosen = select_latest_json(files)
        if not chosen:
            continue
        image_path = PSEUDO_DIR / f"{base}.jpg"
        if not image_path.exists():
            continue
        width, height = get_image_dimensions(str(image_path))
        selections[base] = SlideRecord(base, image_path, chosen, width, height)

    if not selections:
        raise SystemExit("No slide/image pairs discovered under the provided data root.")

    print(f"[Discovery] Using {len(selections)} slide(s) with >= {min_confidence} confidence annotations.")
    return selections


def select_latest_json(files: Sequence[Path]) -> Optional[Path]:
    if not files:
        return None
    files_with_ts = []
    files_without_ts = []
    for fpath in files:
        ts = extract_filename_timestamp(fpath)
        if ts:
            files_with_ts.append((ts, fpath))
        else:
            files_without_ts.append(fpath)

    if files_with_ts:
        files_with_ts.sort(key=lambda item: item[0], reverse=True)
        return files_with_ts[0][1]
    files_without_ts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files_without_ts[0]


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
    white_mask = np.all(tile_bgr >= white_threshold, axis=2)
    white_ratio = float(white_mask.mean())
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if white_ratio > white_ratio_limit:
        return "white", white_ratio, lap_var
    if lap_var < blurry_threshold:
        return "blurry", white_ratio, lap_var
    return "tissue", white_ratio, lap_var


def assign_splits(bases: Sequence[str], val_ratio: float, test_ratio: float,
                  seed: int) -> Dict[str, str]:
    bases = list(bases)
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)

    total = len(bases)
    n_test = int(round(total * test_ratio))
    n_val = int(round(total * val_ratio))
    max_assignable = max(total - 1, 1)
    if n_test + n_val > max_assignable:
        overflow = n_test + n_val - max_assignable
        if n_test >= overflow:
            n_test -= overflow
        else:
            overflow -= n_test
            n_test = 0
            n_val = max(0, n_val - overflow)

    test_set = set(bases[:n_test])
    val_set = set(bases[n_test:n_test + n_val])
    split_map: Dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for base in bases:
        if base in test_set:
            split = "test"
        elif base in val_set:
            split = "val"
        else:
            split = "train"
        split_map[base] = split
        counts[split] += 1
    print(f"[Split] Slides -> train:{counts['train']} val:{counts['val']} test:{counts['test']}")
    return split_map


def init_stain_normalizer(args: argparse.Namespace) -> Optional[ReinhardStainNormalizer]:
    if not getattr(args, "stain_normalize", False):
        return None
    if not STAIN_NORMALIZATION_AVAILABLE:
        print("[Stain] WARN: stain normalization requested but utilities unavailable.")
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


def ensure_split_dirs(dataset_dir: Path, splits: Sequence[str]) -> Dict[str, Dict[str, Path]]:
    split_dirs: Dict[str, Dict[str, Path]] = {}
    for split in splits:
        root = dataset_dir / split
        for label in ("adipose", "not_adipose"):
            (root / label).mkdir(parents=True, exist_ok=True)
        split_dirs[split] = {
            "adipose": root / "adipose",
            "not_adipose": root / "not_adipose",
        }
    return split_dirs


def process_slide(slide: SlideRecord, split: str, args: argparse.Namespace,
                  stain_normalizer, dest_dirs, manifests, stats) -> None:
    polygons = load_json_annotations(slide.json_path, args.min_confidence)
    mask = create_binary_mask(polygons, slide.width, slide.height)

    coords = tile_coords(slide.height, slide.width, args.tile_size, args.stride)
    if not coords:
        stats["skipped_slides_small"] += 1
        return

    tile_quality_params = (
        args.white_threshold,
        args.white_ratio_limit,
        args.blurry_threshold,
    )
    jpeg_settings = jpeg_params(args.jpeg_quality)
    dataset_root = dest_dirs[split]

    for row, col, ys, xs in coords:
        tile = extract_tile(slide.image_path, xs, ys, args.tile_size, stain_normalizer)
        quality, white_ratio, lap_var = evaluate_quality(tile, *tile_quality_params)
        if quality == "white" and not args.keep_white:
            stats["skipped_white"] += 1
            continue
        if quality == "blurry" and not args.keep_blurry:
            stats["skipped_blurry"] += 1
            continue

        mask_patch = mask[ys:ys + args.tile_size, xs:xs + args.tile_size]
        adipose_ratio = float(mask_patch.mean()) if mask_patch.size else 0.0
        label = "adipose" if adipose_ratio >= args.adipose_threshold else "not_adipose"

        out_name = f"{slide.base}_r{row}_c{col}.jpg"
        out_path = dataset_root[label] / out_name
        cv2.imwrite(str(out_path), tile, jpeg_settings)

        rel_path = out_path.relative_to(BUILD_ROOT)
        manifests[split].append([
            str(rel_path),
            label,
            f"{adipose_ratio:.6f}",
            f"{white_ratio:.6f}",
            f"{lap_var:.3f}",
            quality,
        ])
        stats["tiles_written"] += 1
        stats["per_split"][split][label] += 1


def balance_tiles(tiles_by_class: Dict[str, List], target_adipose_ratio: float = 0.40,
                  seed: int = 865) -> Dict[str, List]:
    """
    Undersample majority class to achieve target balance.
    
    Args:
        tiles_by_class: {"adipose": [...], "not_adipose": [...]}
        target_adipose_ratio: Target proportion of adipose tiles (0.4 = 40%)
        seed: Random seed for reproducible sampling
        
    Returns:
        Balanced tiles_by_class dictionary
    """
    adipose_count = len(tiles_by_class.get("adipose", []))
    not_adipose_count = len(tiles_by_class.get("not_adipose", []))
    
    if adipose_count == 0 or not_adipose_count == 0:
        return tiles_by_class
    
    # Calculate target counts
    # If we want 40% adipose, 60% not_adipose:
    # adipose / (adipose + not_adipose) = 0.4
    # Solving: not_adipose = adipose * (1 - target) / target
    target_not_adipose = int(adipose_count * (1 - target_adipose_ratio) / target_adipose_ratio)
    
    print(f"\n[Balance] Original counts:")
    print(f"  Adipose:     {adipose_count:6d}")
    print(f"  Not adipose: {not_adipose_count:6d}")
    print(f"  Ratio:       {not_adipose_count/max(adipose_count,1):.2f}:1")
    
    if not_adipose_count > target_not_adipose:
        # Undersample not_adipose
        rng = np.random.default_rng(seed)
        indices = rng.choice(not_adipose_count, target_not_adipose, replace=False)
        tiles_by_class["not_adipose"] = [tiles_by_class["not_adipose"][i] for i in sorted(indices)]
        
        total = adipose_count + target_not_adipose
        adipose_pct = 100 * adipose_count / total
        
        print(f"\n[Balance] After balancing:")
        print(f"  Adipose:     {adipose_count:6d} (kept all)")
        print(f"  Not adipose: {target_not_adipose:6d} (sampled {100*target_not_adipose/not_adipose_count:.1f}%)")
        print(f"  Ratio:       {target_not_adipose/adipose_count:.2f}:1")
        print(f"  Adipose:     {adipose_pct:.1f}%")
        
    elif adipose_count > int(not_adipose_count * target_adipose_ratio / (1 - target_adipose_ratio)):
        # Rare case: undersample adipose
        target_adipose = int(not_adipose_count * target_adipose_ratio / (1 - target_adipose_ratio))
        rng = np.random.default_rng(seed)
        indices = rng.choice(adipose_count, target_adipose, replace=False)
        tiles_by_class["adipose"] = [tiles_by_class["adipose"][i] for i in sorted(indices)]
        
        total = target_adipose + not_adipose_count
        adipose_pct = 100 * target_adipose / total
        
        print(f"\n[Balance] After balancing:")
        print(f"  Adipose:     {target_adipose:6d} (sampled {100*target_adipose/adipose_count:.1f}%)")
        print(f"  Not adipose: {not_adipose_count:6d} (kept all)")
        print(f"  Ratio:       {not_adipose_count/target_adipose:.2f}:1")
        print(f"  Adipose:     {adipose_pct:.1f}%")
    else:
        print(f"  Classes already balanced, no sampling needed.")
    
    return tiles_by_class


def report_class_balance(stats: Dict, log_path: Optional[Path] = None) -> str:
    """
    Report detailed class balance statistics.
    
    Args:
        stats: Statistics dictionary with per_split counts
        log_path: Optional path to save report
        
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CLASS BALANCE REPORT")
    report_lines.append("=" * 70)
    
    for split in ["train", "val", "test"]:
        if split not in stats["per_split"]:
            continue
            
        counts = stats["per_split"][split]
        total = sum(counts.values())
        
        if total == 0:
            continue
        
        report_lines.append(f"\n{split.upper()} Split:")
        report_lines.append("-" * 40)
        
        for cls in sorted(counts.keys()):
            count = counts[cls]
            pct = 100 * count / total
            report_lines.append(f"  {cls:15s}: {count:6d} ({pct:5.1f}%)")
        
        # Imbalance metrics
        if len(counts) == 2:
            adipose = counts.get("adipose", 0)
            not_adipose = counts.get("not_adipose", 0)
            
            if adipose > 0 and not_adipose > 0:
                ratio = not_adipose / adipose
                adipose_pct = 100 * adipose / total
                
                report_lines.append(f"\n  Metrics:")
                report_lines.append(f"    Ratio (not_adipose:adipose): {ratio:.2f}:1")
                report_lines.append(f"    Adipose percentage:          {adipose_pct:.1f}%")
                
                # Quality assessment
                if 35 <= adipose_pct <= 45:
                    status = "✓ EXCELLENT (40±5%)"
                elif 30 <= adipose_pct <= 50:
                    status = "✓ GOOD (40±10%)"
                elif 25 <= adipose_pct <= 55:
                    status = "⚠ ACCEPTABLE (40±15%)"
                else:
                    status = f"✗ IMBALANCED (target: 40%, got: {adipose_pct:.1f}%)"
                
                report_lines.append(f"    Balance quality:             {status}")
    
    report_lines.append("\n" + "=" * 70)
    
    # Print to console
    report_str = '\n'.join(report_lines)
    print(report_str)
    
    # Save to file
    if log_path:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
            f.write('\n')
        print(f"\n✓ Class balance report saved to: {log_path}")
    
    return report_str


def write_manifests(manifests: Dict[str, List[List[str]]], dataset_dir: Path) -> None:
    header = "path,label,adipose_ratio,white_ratio,laplacian_var,quality\n"
    for split, rows in manifests.items():
        if not rows:
            continue
        out_csv = dataset_dir / f"{split}_manifest.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(header)
            for row in rows:
                f.write(",".join(row) + "\n")
        print(f"[Manifest] Wrote {out_csv.name} with {len(rows)} entries.")


def main():
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)

    _, dataset_dir, _ = setup_paths(args)
    slides = discover_slides(args.min_confidence)
    split_map = assign_splits(slides.keys(), args.val_ratio, args.test_ratio, args.seed)
    used_splits = sorted(set(split_map.values()))
    dest_dirs = ensure_split_dirs(dataset_dir, used_splits)
    stain_normalizer = init_stain_normalizer(args)

    manifests: Dict[str, List[List[str]]] = {split: [] for split in used_splits}
    stats = {
        "tiles_written": 0,
        "skipped_white": 0,
        "skipped_blurry": 0,
        "skipped_slides_small": 0,
        "per_split": {split: {"adipose": 0, "not_adipose": 0} for split in used_splits},
    }

    for base in tqdm(sorted(slides.keys()), desc="Tiling slides"):
        split = split_map[base]
        process_slide(
            slides[base],
            split,
            args,
            stain_normalizer,
            dest_dirs,
            manifests,
            stats,
        )

    # Apply class balancing if requested (only to training set)
    if args.balance_classes:
        print(f"\n{'='*70}")
        print(f"APPLYING CLASS BALANCING (target: {args.target_adipose_ratio:.0%} adipose)")
        print(f"ONLY to TRAINING set - validation/test keep natural distribution")
        print(f"{'='*70}")
        
        for split in manifests.keys():
            # Only balance training set - validation/test should reflect real distribution
            if split != "train":
                continue
                
            if not manifests[split]:
                continue
                
            # Collect tiles by class
            tiles_by_class = {"adipose": [], "not_adipose": []}
            for row in manifests[split]:
                label = row[1]  # Label is at index 1
                tiles_by_class[label].append(row)
            
            # Balance
            print(f"\n{split.upper()} Split:")
            balanced = balance_tiles(tiles_by_class, args.target_adipose_ratio, args.seed)
            
            # Update manifest
            manifests[split] = balanced["adipose"] + balanced["not_adipose"]
            
            # Update stats
            stats["per_split"][split] = {
                "adipose": len(balanced["adipose"]),
                "not_adipose": len(balanced["not_adipose"])
            }

    # Write manifests
    write_manifests(manifests, dataset_dir)
    
    # Generate and save class balance report
    balance_report_path = BUILD_ROOT / "class_balance_report.txt"
    report_class_balance(stats, balance_report_path)

    print("\n[Summary]")
    print(f"  Tiles written: {stats['tiles_written']}")
    print(f"  Skipped (white): {stats['skipped_white']}")
    print(f"  Skipped (blurry): {stats['skipped_blurry']}")
    print(f"  Skipped slides (too small): {stats['skipped_slides_small']}")
    for split, counts in stats["per_split"].items():
        print(f"  {split}: adipose={counts['adipose']} not_adipose={counts['not_adipose']}")
    print(f"\n[Output] Dataset root: {dataset_dir}")


if __name__ == "__main__":
    main()
