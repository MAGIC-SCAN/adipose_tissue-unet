#!/usr/bin/env python3
"""
Test Dataset Builder for Adipose Tissue U-Net

A focused script for creating test datasets with clean output structure.
Extracts core preprocessing logic from build_dataset.py but simplified for test-only use.

Output Structure:
target_folder/
├── images/              # Final tiled images (ready for testing)
├── masks/               # Final tiled masks (ready for testing)  
└── build/               # Intermediate processing artifacts
    ├── masks/           # Full-image masks (JSON→TIFF conversion)
    ├── tiles/           # All tiles by category (tissue/blurry/empty)
    └── tiles_masks/     # All mask tiles (before final filtering)

Usage:
python build_test_dataset.py \
    --images-dir /path/to/pseudocolored/images \
    --masks-dir /path/to/json/masks \
    --output-dir /path/to/target/folder \
    --target-mask fat
"""

import os
import sys
import json
import math
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import time
import re
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import tifffile as tiff

# Load seed from seed.csv for reproducibility
from src.utils.seed_utils import get_project_seed
GLOBAL_SEED = get_project_seed()
np.random.seed(GLOBAL_SEED)

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("tqdm not found. Install with: pip install tqdm")

# Stain normalization imports
try:
    from src.utils.stain_normalization import load_best_reference
    STAIN_NORMALIZATION_AVAILABLE = True
except ImportError:
    STAIN_NORMALIZATION_AVAILABLE = False
    print("[WARN] Stain normalization not available - check src/utils/stain_normalization.py")

# ------------------------------
# Configuration and Constants
# ------------------------------
CLASS_NAMES = ["bubbles", "fat", "muscle"]
OVERLAY_COLORS = {
    "bubbles": (0, 0, 255),   # Blue (BGR)
    "fat":     (0, 255, 255), # Yellow
    "muscle":  (0, 255, 0),   # Green
}

# Test-optimized defaults for comprehensive evaluation
TEST_DEFAULTS = {
    "target_mask": "fat",
    "subtract": False,
    "subtract_class": "bubbles", 
    "morph_close_k": 0,
    "min_cc_px": 0,
    "tile_size": 1024,
    "stride": 1024,           # No overlap for test data
    "white_threshold": 235,
    "white_ratio_limit": 0.70,
    "blurry_threshold": 7.5,
    "min_mask_ratio": 0.0,    # Include all mask densities  
    "jpeg_quality": 100,
    "compression": "auto",
    "workers": None,          # None = cpu_count() - 1
    "neg_pct": 1.0,          # Include all negative tiles
    "min_confidence": 2,      # Annotation confidence threshold
    "stain_normalize": True,  # Enable SYBR Gold + Eosin stain normalization (color correction only)
    "reference_metadata": "src/utils/stain_reference_metadata.json",
    "include_white": False,   # Exclude white tiles
    "include_blurry": False,  # Exclude blurry tiles
    "seed": 865,
}

# ------------------------------
# Utility Functions (extracted from build_dataset.py)
# ------------------------------
def _save_tiff_mask(path: Path, arr: np.ndarray, compression: str = "auto"):
    """Write a uint8 0/1 TIFF mask with compression and tiling for performance."""
    arr = arr.astype(np.uint8)
    comp = (compression or "auto").lower()
    
    # Use tiled TIFF for better I/O performance
    tile_size = (256, 256) if arr.shape[0] >= 256 and arr.shape[1] >= 256 else None
    
    if comp == "none":
        tiff.imwrite(str(path), arr, photometric="minisblack", tile=tile_size)
        return
    if comp == "packbits":
        tiff.imwrite(str(path), arr, compression="packbits", photometric="minisblack", tile=tile_size)
        return
    if comp == "lzw":
        tiff.imwrite(str(path), arr, compression="lzw", photometric="minisblack", tile=tile_size)
        return
    # auto
    try:
        tiff.imwrite(str(path), arr, compression="lzw", photometric="minisblack", tile=tile_size)
    except (KeyError, RuntimeError) as e:
        if "imagecodecs" in str(e) or "COMPRESSION.LZW" in str(e):
            tiff.imwrite(str(path), arr, compression="packbits", photometric="minisblack", tile=tile_size)
        else:
            raise

@lru_cache(maxsize=256)
def get_image_dimensions(img_path: str) -> Tuple[int, int]:
    """Cache image dimensions to avoid repeated file opens."""
    with Image.open(img_path) as im:
        return im.size

def parse_image_stem_from_json(json_path: Path, cls: str) -> str:
    """Return the image stem for a given JSON, robust to many underscores."""
    stem = json_path.stem
    token = f"_{cls}_annotations"
    if token in stem:
        return stem.split(token)[0]
    token2 = f"_{cls}_"
    if token2 in stem:
        return stem.split(token2)[0]
    return stem

def extract_filename_timestamp(json_path: Path) -> datetime | None:
    """Extract timestamp from filename in format: *_MMDDYYYY_HHMMSS.json"""
    filename = json_path.stem
    timestamp_pattern = r'_(\d{8})_(\d{6})$'
    match = re.search(timestamp_pattern, filename)
    
    if not match:
        return None
    
    date_str = match.group(1)  # MMDDYYYY
    time_str = match.group(2)  # HHMMSS
    
    try:
        month = int(date_str[:2])
        day = int(date_str[2:4])
        year = int(date_str[4:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError):
        return None

def load_json_annotations(json_path: Path, min_confidence: int = 1) -> Tuple[List[np.ndarray], bool]:
    """Load polygon annotations from JSON file, filtering by confidence score."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_elements = []
    missing_confidence = False
    
    if isinstance(data, list):
        for ann in data:
            if isinstance(ann, dict):
                confidence_score = ann.get("confidenceScore")
                
                if confidence_score is None:
                    missing_confidence = True
                    elems = ann.get("annotation", {}).get("elements", [])
                    if isinstance(elems, list):
                        all_elements.extend(elems)
                elif confidence_score >= min_confidence:
                    elems = ann.get("annotation", {}).get("elements", [])
                    if isinstance(elems, list):
                        all_elements.extend(elems)
    elif isinstance(data, dict):
        confidence_score = data.get("confidenceScore")
        
        if confidence_score is None:
            missing_confidence = True
            elems = data.get("annotation", {}).get("elements", [])
            if isinstance(elems, list):
                all_elements.extend(elems)
        elif confidence_score >= min_confidence:
            elems = data.get("annotation", {}).get("elements", [])
            if isinstance(elems, list):
                all_elements.extend(elems)

    polys = []
    for elem in all_elements:
        if not isinstance(elem, dict):
            continue
        if elem.get("type") != "polyline":
            continue
        pts = elem.get("points", [])
        if pts and len(pts) >= 3:
            coords = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)
            if len(coords) >= 3:
                polys.append(coords)
    
    return polys, missing_confidence

def create_binary_mask(polygons: List[np.ndarray], width: int, height: int) -> np.ndarray:
    """Rasterize many polygons into a single 0/1 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    cv_polys = [np.asarray(p, dtype=np.int32).reshape(-1, 1, 2) for p in polygons if len(p) >= 3]
    if cv_polys:
        cv2.fillPoly(mask, cv_polys, 1)
    return mask

def morph_close(mask: np.ndarray, k: int) -> np.ndarray:
    """Apply morphological closing operation."""
    if k <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than min_px."""
    if min_px <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_px:
            out[labels == lbl] = 1
    return out

# ------------------------------
# Core Processing Functions
# ------------------------------
def _process_single_json_worker(args_tuple):
    """Worker function for parallel mask generation."""
    (jpath_str, cls, base, img_path_str, width, height, 
     compression, masks_out_root_str, min_confidence) = args_tuple
    
    jpath = Path(jpath_str)
    img_path = Path(img_path_str)
    masks_out_root = Path(masks_out_root_str)
    
    try:
        polys, missing_confidence = load_json_annotations(jpath, min_confidence=min_confidence)
        
        if not polys:
            return (base, cls, "no_polygons", missing_confidence)
        
        mask = create_binary_mask(polys, width, height)
        
        # Save class mask
        out_dir = masks_out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base}_{cls}.tif"
        _save_tiff_mask(out_path, mask, compression=compression)
        
        return (base, cls, "success", missing_confidence)
    except Exception as e:
        return (base, cls, f"error: {str(e)}", False)

def build_masks_from_json(images_dir: Path, masks_dir: Path, output_build_dir: Path, 
                         target_mask: str, args):
    """Build binary masks from JSON annotations."""
    print(f"[Masks] Building masks for target class '{target_mask}'...")
    
    # Find image files
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No .jpg files found in {images_dir}")
    
    image_stems = {f.stem for f in image_files}
    print(f"[Masks] Found {len(image_stems)} images")
    
    # Find corresponding JSON files for target class
    json_files = list(masks_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No .json files found in {masks_dir}")
    
    # Group files by base image and select newest
    files_by_base = {}
    for jpath in json_files:
        base = parse_image_stem_from_json(jpath, target_mask)
        if base in image_stems:
            if base not in files_by_base:
                files_by_base[base] = []
            files_by_base[base].append(jpath)
    
    # Select newest file for each base
    latest_by_base = {}
    for base, jpaths in files_by_base.items():
        if len(jpaths) == 1:
            latest_by_base[base] = jpaths[0]
        else:
            files_with_timestamps = []
            files_without_timestamps = []
            
            for jpath in jpaths:
                timestamp = extract_filename_timestamp(jpath)
                if timestamp:
                    files_with_timestamps.append((timestamp, jpath))
                else:
                    files_without_timestamps.append(jpath)
            
            if files_with_timestamps:
                files_with_timestamps.sort(key=lambda x: x[0], reverse=True)
                latest_by_base[base] = files_with_timestamps[0][1]
                print(f"[Masks] Selected newest {base}_{target_mask}: {files_with_timestamps[0][1].name}")
            elif files_without_timestamps:
                files_without_timestamps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest_by_base[base] = files_without_timestamps[0]
    
    print(f"[Masks] Processing {len(latest_by_base)} image/mask pairs")
    
    # Prepare work items
    work_items = []
    size_cache = {}
    
    for base, jpath in latest_by_base.items():
        img_path = images_dir / f"{base}.jpg"
        if not img_path.exists():
            continue
            
        if base in size_cache:
            width, height = size_cache[base]
        else:
            width, height = get_image_dimensions(str(img_path))
            size_cache[base] = (width, height)
        
        work_items.append((
            str(jpath), target_mask, base, str(img_path), width, height,
            args.compression, str(output_build_dir / "masks"), args.min_confidence
        ))
    
    if not work_items:
        raise ValueError("No valid image/mask pairs found")
    
    # Process with multiprocessing
    n_workers = args.workers if args.workers else max(1, cpu_count() - 1)
    print(f"[Masks] Processing {len(work_items)} annotations using {n_workers} workers")
    
    results = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_single_json_worker, work_items),
            total=len(work_items),
            desc="Building masks",
            unit="file",
            smoothing=0.1
        ):
            results.append(result)
    
    # Report results
    success = sum(1 for _, _, status, _ in results if status == "success")
    failed = sum(1 for _, _, status, _ in results if status.startswith("error"))
    
    missing_confidence_files = []
    for base, cls, status, missing_conf in results:
        if status == "success" and missing_conf:
            json_name = f"{base}_{cls}_annotations.json"
            missing_confidence_files.append(json_name)
    
    print(f"[Masks] Complete: {success} successful, {failed} failed")
    
    if missing_confidence_files:
        print(f"\n[Masks] ⚠️  {len(missing_confidence_files)} file(s) missing confidenceScore field:")
        for fname in sorted(missing_confidence_files)[:5]:
            print(f"  - {fname}")
        if len(missing_confidence_files) > 5:
            print(f"  ... and {len(missing_confidence_files) - 5} more")

def prepare_target_masks(output_build_dir: Path, target_mask: str, args):
    """Prepare target masks with subtraction and morphology operations."""
    print(f"[Target] Preparing target masks for class '{target_mask}'...")
    
    base_target_dir = output_build_dir / "masks" / target_mask
    if not base_target_dir.exists():
        print(f"[WARN] Base masks for target '{target_mask}' not found (expected {base_target_dir}).")
        return
    
    subtract_dir = None
    if args.subtract:
        subtract_dir = output_build_dir / "masks" / args.subtract_class
        if not subtract_dir.exists():
            print(f"[WARN] Subtraction enabled, but masks folder missing: {subtract_dir}")
            subtract_dir = None
    
    mask_files = list(base_target_dir.glob(f"*_{target_mask}.tif"))
    
    for mask_path in tqdm(mask_files, desc="Target masks", unit="img"):
        base = mask_path.stem.replace(f"_{target_mask}", "")
        
        target_mask_data = (tiff.imread(str(mask_path)) > 0).astype(np.uint8)
        
        # Apply subtraction if requested
        if subtract_dir is not None and args.subtract_class != target_mask:
            sub_path = subtract_dir / f"{base}_{args.subtract_class}.tif"
            if sub_path.exists():
                sub_mask = (tiff.imread(str(sub_path)) > 0).astype(np.uint8)
                target_mask_data = np.clip(target_mask_data.astype(np.int16) - sub_mask.astype(np.int16), 0, 1).astype(np.uint8)
        
        # Apply morphological operations
        target_mask_data = morph_close(target_mask_data, args.morph_close_k)
        target_mask_data = remove_small_components(target_mask_data, args.min_cc_px)
        
        # Save processed mask
        _save_tiff_mask(mask_path, target_mask_data, compression=args.compression)

# ------------------------------
# Tiling Functions
# ------------------------------
JPEG_PARAMS_CACHE = {}

def jpeg_params(quality: int):
    if quality not in JPEG_PARAMS_CACHE:
        JPEG_PARAMS_CACHE[quality] = [
            cv2.IMWRITE_JPEG_QUALITY, int(quality),
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
        ]
    return JPEG_PARAMS_CACHE[quality]

def extract_tile_pil(img_path: Path, x: int, y: int, size: int, stain_normalizer=None) -> np.ndarray:
    """Extract tile using PIL's efficient cropping with optional stain normalization."""
    with Image.open(img_path) as img:
        tile = img.crop((x, y, x + size, y + size))
        tile_rgb = np.array(tile)
        
        # Apply stain normalization (color correction only) if provided
        if stain_normalizer is not None:
            try:
                tile_rgb = stain_normalizer.normalize_image(tile_rgb)
                if tile_rgb.max() <= 1.0:
                    tile_rgb = (tile_rgb * 255).astype(np.uint8)
                else:
                    tile_rgb = tile_rgb.astype(np.uint8)
            except Exception as e:
                print(f"[WARN] Stain normalization failed for tile at ({x},{y}): {e}")
        
        return cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

def classify_tiles_batch(tiles_bgr: List[np.ndarray], white_threshold: int, 
                         white_ratio_limit: float, blurry_threshold: float) -> List[str]:
    """Classify multiple tiles at once."""
    if not tiles_bgr:
        return []
    
    classifications = []
    
    # Per-tile computation to reduce memory usage
    for tile in tiles_bgr:
        white_mask = np.all(tile >= white_threshold, axis=2)
        white_ratio = white_mask.mean()
        
        if white_ratio > white_ratio_limit:
            classifications.append("empty")
            continue
            
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if lap_var < blurry_threshold:
            classifications.append("blurry")
        else:
            classifications.append("tissue")
    
    return classifications

def tile_coords(h: int, w: int, tile: int, stride: int):
    """Generate tile coordinates with proper boundary checking."""
    if h < tile or w < tile:
        return []
    
    if stride >= tile:
        x_steps = w // tile
        y_steps = h // tile
    else:
        x_steps = max(1, math.ceil((w - tile) / stride) + 1)
        y_steps = max(1, math.ceil((h - tile) / stride) + 1)
    
    coords = []
    for ri in range(y_steps):
        for ci in range(x_steps):
            xs = min(ci * stride, w - tile)
            ys = min(ri * stride, h - tile)
            
            if xs >= 0 and ys >= 0 and xs + tile <= w and ys + tile <= h:
                coords.append((ri, ci, ys, xs))
    
    return coords

def tile_and_filter(images_dir: Path, output_build_dir: Path, target_mask: str, args):
    """Extract tiles from images and filter them."""
    print(f"[Tiling] Processing images with tile={args.tile_size}, stride={args.stride}")
    
    # Setup directories
    tiles_img_tissue = output_build_dir / "tiles" / "tissue"
    tiles_img_blurry = output_build_dir / "tiles" / "blurry"
    tiles_img_empty  = output_build_dir / "tiles" / "empty"
    tiles_msk_dir    = output_build_dir / "tiles_masks" / target_mask
    
    for d in [tiles_img_tissue, tiles_img_blurry, tiles_img_empty, tiles_msk_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize stain normalizer if requested
    stain_normalizer = None
    if args.stain_normalize:
        if not STAIN_NORMALIZATION_AVAILABLE:
            print("[WARN] Stain normalization requested but modules not available; continuing without it.")
        else:
            try:
                print(f"[Stain] Loading best reference from: {args.reference_metadata}")
                stain_normalizer = load_best_reference(args.reference_metadata)
                print(f"[Stain] SYBR Gold + Eosin stain normalization enabled (color correction only)")
            except Exception as e:
                print(f"[WARN] Failed to initialize stain normalizer: {e} — continuing without it.")
    
    # Find processed target masks
    mask_dir = output_build_dir / "masks" / target_mask
    mask_files = {p.stem.replace(f"_{target_mask}", ""): p for p in mask_dir.glob(f"*_{target_mask}.tif")}
    
    # Process images
    image_files = [f for f in images_dir.glob("*.jpg") if f.stem in mask_files]
    print(f"[Tiling] Found {len(image_files)} images with matching masks")
    
    pos_kept = 0
    neg_candidates = []
    batch_size = 16
    jpeg_params_cached = jpeg_params(args.jpeg_quality)
    
    for img_path in tqdm(image_files, desc="Tiling images", unit="img"):
        base = img_path.stem
        msk_path = mask_files[base]
        
        # Get image dimensions and generate tile coordinates
        w, h = get_image_dimensions(str(img_path))
        if h < args.tile_size or w < args.tile_size:
            continue
        
        coords = tile_coords(h, w, args.tile_size, args.stride)
        if not coords:
            continue
        
        # Load mask
        msk = tiff.imread(str(msk_path)).astype(np.uint8)
        if msk.ndim == 3:
            msk = msk.squeeze()
        
        # Process in batches
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            
            # Extract tiles
            tiles = []
            for r, c, ys, xs in batch_coords:
                tile = extract_tile_pil(img_path, xs, ys, args.tile_size, stain_normalizer)
                tiles.append((tile, r, c, ys, xs))
            
            # Classify batch
            tile_arrays = [t[0] for t in tiles]
            classifications = classify_tiles_batch(
                tile_arrays, 
                args.white_threshold, 
                args.white_ratio_limit, 
                args.blurry_threshold
            )
            
            # Apply override logic for test dataset
            if args.include_white or args.include_blurry:
                for i, cls in enumerate(classifications):
                    if cls == 'empty' and args.include_white:
                        classifications[i] = 'tissue'
                    elif cls == 'blurry' and args.include_blurry:
                        classifications[i] = 'tissue'
            
            # Save tiles
            for (tile, r, c, ys, xs), cls in zip(tiles, classifications):
                out_img_dir = {
                    "empty": tiles_img_empty, 
                    "blurry": tiles_img_blurry, 
                    "tissue": tiles_img_tissue
                }[cls]
                out_name = f"{base}_r{r}_c{c}.jpg"
                cv2.imwrite(str(out_img_dir / out_name), tile, jpeg_params_cached)
                
                if cls != "tissue":
                    continue
                
                # Extract and check mask tile
                m_tile = msk[ys:ys+args.tile_size, xs:xs+args.tile_size]
                if m_tile.shape[:2] != (args.tile_size, args.tile_size):
                    continue
                
                pos_ratio = float(m_tile.sum()) / (args.tile_size * args.tile_size)
                msk_out_path = tiles_msk_dir / out_name.replace('.jpg', '.tif')
                
                if pos_ratio >= args.min_mask_ratio:
                    # Positive tile → keep mask immediately
                    _save_tiff_mask(msk_out_path, m_tile, compression=args.compression)
                    pos_kept += 1
                else:
                    # Negative candidate
                    neg_candidates.append(msk_out_path)
    
    # Handle negative tiles
    rng = np.random.default_rng(args.seed)
    f_neg = float(max(0.0, min(args.neg_pct, 1.0)))
    neg_target = int(round(f_neg * len(neg_candidates)))
    
    total_neg_written = 0
    if neg_target > 0:
        if f_neg >= 1.0:
            chosen_paths = neg_candidates
        else:
            chosen_idx = rng.choice(len(neg_candidates), size=neg_target, replace=False)
            chosen_paths = [neg_candidates[i] for i in chosen_idx]
        
        for zp in tqdm(chosen_paths, desc="Writing negative masks", unit="tile"):
            if not zp.exists():
                _save_tiff_mask(zp, np.zeros((args.tile_size, args.tile_size), dtype=np.uint8), compression=args.compression)
        total_neg_written = len(chosen_paths)
    
    print(f"[Tiling] Positives kept: {pos_kept} | Negatives written: {total_neg_written}")
    return pos_kept + total_neg_written

def copy_final_dataset(output_build_dir: Path, output_dir: Path, target_mask: str):
    """Copy final tiles to the clean output structure."""
    print("[Final] Copying tiles to final output structure...")
    
    # Create final output directories
    final_images_dir = output_dir / "images"
    final_masks_dir = output_dir / "masks"
    final_images_dir.mkdir(parents=True, exist_ok=True)
    final_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tissue tiles and their corresponding masks
    tiles_img_dir = output_build_dir / "tiles" / "tissue"
    tiles_msk_dir = output_build_dir / "tiles_masks" / target_mask
    
    img_files = sorted(tiles_img_dir.glob("*.jpg"))
    mask_files = {p.stem: p for p in tiles_msk_dir.glob("*.tif")}
    
    pairs = []
    for img_path in img_files:
        stem = img_path.stem
        if stem in mask_files:
            pairs.append((img_path, mask_files[stem]))
    
    print(f"[Final] Copying {len(pairs)} final tiles...")
    for img_path, mask_path in tqdm(pairs, desc="Copying tiles", unit="tile"):
        shutil.copy2(img_path, final_images_dir / img_path.name)
        shutil.copy2(mask_path, final_masks_dir / mask_path.name)
    
    return len(pairs)

# ------------------------------
# Main Script
# ------------------------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Dataset Builder - Create clean test datasets for evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--images-dir", type=str, required=True,
                       help="Directory containing pseudocolored .jpg images")
    parser.add_argument("--masks-dir", type=str, required=True,
                       help="Directory containing .json annotation files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for test dataset")
    
    # Core parameters
    parser.add_argument("--target-mask", type=str, default=TEST_DEFAULTS["target_mask"],
                       choices=CLASS_NAMES, help="Target class for training/testing")
    
    # Subtraction options
    parser.add_argument("--subtract", dest="subtract", action="store_true",
                       default=TEST_DEFAULTS["subtract"], help="Subtract another class from target")
    parser.add_argument("--no-subtract", dest="subtract", action="store_false")
    parser.add_argument("--subtract-class", type=str, default=TEST_DEFAULTS["subtract_class"],
                       choices=CLASS_NAMES, help="Class to subtract if --subtract enabled")
    
    # Morphology parameters
    parser.add_argument("--morph-close-k", type=int, default=TEST_DEFAULTS["morph_close_k"],
                       help="Kernel size for morphological closing (0 disables)")
    parser.add_argument("--min-cc-px", type=int, default=TEST_DEFAULTS["min_cc_px"],
                       help="Remove connected components smaller than this (0 disables)")
    
    # Tiling parameters
    parser.add_argument("--tile-size", type=int, default=TEST_DEFAULTS["tile_size"],
                       help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=TEST_DEFAULTS["stride"],
                       help="Stride for tiling (< tile-size enables overlap)")
    
    # Quality filtering
    parser.add_argument("--white-threshold", type=int, default=TEST_DEFAULTS["white_threshold"],
                       help="White threshold per channel")
    parser.add_argument("--white-ratio-limit", type=float, default=TEST_DEFAULTS["white_ratio_limit"],
                       help="Fraction of white pixels to classify tile as empty")
    parser.add_argument("--blurry-threshold", type=float, default=TEST_DEFAULTS["blurry_threshold"],
                       help="Variance of Laplacian threshold for blur detection")
    parser.add_argument("--min-mask-ratio", type=float, default=TEST_DEFAULTS["min_mask_ratio"],
                       help="Minimum positive mask fraction to keep a tile")
    
    # Test-specific overrides
    parser.add_argument("--include-white", dest="include_white", action="store_true",
                       default=TEST_DEFAULTS["include_white"],
                       help="Include white tiles in test dataset (override quality filter)")
    parser.add_argument("--include-blurry", dest="include_blurry", action="store_true",
                       default=TEST_DEFAULTS["include_blurry"], 
                       help="Include blurry tiles in test dataset (override quality filter)")
    
    # Output parameters
    parser.add_argument("--jpeg-quality", type=int, default=TEST_DEFAULTS["jpeg_quality"],
                       help="JPEG quality for tile images")
    parser.add_argument("--compression", type=str, default=TEST_DEFAULTS["compression"],
                       choices=["auto", "lzw", "packbits", "none"],
                       help="TIFF compression for mask files")
    
    # Processing parameters
    parser.add_argument("--workers", type=int, default=TEST_DEFAULTS["workers"],
                       help="Number of parallel workers (None = cpu_count - 1)")
    parser.add_argument("--neg-pct", type=float, default=TEST_DEFAULTS["neg_pct"],
                       help="Target fraction of negative tiles in final dataset")
    parser.add_argument("--min-confidence", type=int, default=TEST_DEFAULTS["min_confidence"],
                       choices=[1, 2, 3], help="Minimum annotation confidence score")
    parser.add_argument("--seed", type=int, default=TEST_DEFAULTS["seed"],
                       help="Random seed for reproducibility")
    
    # Stain normalization
    parser.add_argument("--stain-normalize", dest="stain_normalize", action="store_true",
                       default=TEST_DEFAULTS["stain_normalize"],
                       help="Enable SYBR Gold + Eosin stain normalization")
    parser.add_argument("--no-stain-normalize", dest="stain_normalize", action="store_false")
    parser.add_argument("--reference-metadata", type=str, default=TEST_DEFAULTS["reference_metadata"],
                       help="Path to reference metadata JSON file for stain normalization")
    
    return parser.parse_args()

def create_build_summary(output_dir: Path, args, processing_time: float, final_tiles: int):
    """Create a simple build summary for the test dataset."""
    summary_file = output_dir / "build_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEST DATASET BUILD SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Processing Time: {processing_time:.1f}s ({processing_time/60:.1f} min)\n")
        f.write(f"Final Tiles Generated: {final_tiles}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Images Source: {args.images_dir}\n")
        f.write(f"Masks Source: {args.masks_dir}\n")
        f.write(f"Output Directory: {args.output_dir}\n")
        f.write(f"Target Class: {args.target_mask}\n")
        f.write(f"Subtraction: {'YES' if args.subtract else 'NO'}")
        if args.subtract:
            f.write(f" (subtract {args.subtract_class})")
        f.write("\n")
        
        f.write(f"Tile Size: {args.tile_size}x{args.tile_size}\n")
        f.write(f"Stride: {args.stride}\n")
        f.write(f"Min Mask Ratio: {args.min_mask_ratio}\n")
        f.write(f"Negative Percentage: {args.neg_pct:.1%}\n")
        f.write(f"JPEG Quality: {args.jpeg_quality}\n")
        f.write(f"Confidence Threshold: {args.min_confidence}\n")
        f.write(f"Stain Normalization: {'YES' if args.stain_normalize else 'NO'}\n")
        
        f.write("\nOUTPUT STRUCTURE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{args.output_dir}/images/     # Final image tiles\n")
        f.write(f"{args.output_dir}/masks/      # Final mask tiles\n")
        f.write(f"{args.output_dir}/build/      # Processing artifacts\n")
        
        f.write("\n" + "=" * 60 + "\n")

def main():
    """Main function."""
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
    
    args = parse_args()
    start_time = time.time()
    
    # Convert string paths to Path objects
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directories
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Create output structure
    output_build_dir = output_dir / "build"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_build_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Setup] Test Dataset Builder")
    print(f"[Setup] Images: {images_dir}")
    print(f"[Setup] Masks: {masks_dir}")
    print(f"[Setup] Output: {output_dir}")
    print(f"[Setup] Target class: {args.target_mask}")
    
    # Validate subtraction configuration
    if args.subtract and args.subtract_class == args.target_mask:
        raise ValueError(f"Cannot subtract '{args.subtract_class}' from itself (target='{args.target_mask}')")
    
    try:
        # Step 1: Build masks from JSON
        build_masks_from_json(images_dir, masks_dir, output_build_dir, args.target_mask, args)
        
        # Step 2: Prepare target masks (subtraction, morphology)
        prepare_target_masks(output_build_dir, args.target_mask, args)
        
        # Step 3: Tile and filter
        total_tiles = tile_and_filter(images_dir, output_build_dir, args.target_mask, args)
        
        # Step 4: Copy final dataset
        final_tiles = copy_final_dataset(output_build_dir, output_dir, args.target_mask)
        
        # Report results
        elapsed = time.time() - start_time
        print(f"\n✅ Test dataset build complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"   Output: {output_dir}")
        print(f"   Final tiles: {final_tiles}")
        print(f"   Images: {output_dir / 'images'}")
        print(f"   Masks: {output_dir / 'masks'}")
        print(f"   Build artifacts: {output_dir / 'build'}")
        
        # Create summary
        create_build_summary(output_dir, args, elapsed, final_tiles)
        print(f"   Build summary: {output_dir / 'build_summary.txt'}")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
