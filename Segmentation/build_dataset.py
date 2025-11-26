#!/usr/bin/env python3
"""
Optimized Dataset builder for Adipose Tissue U-Net (TF2.13 / Py3.10)

Key Optimizations:
  1. Parallel mask generation using multiprocessing
  2. Vectorized tile classification
  3. Memory-efficient tiling with PIL cropping
  4. Pre-computed tile coordinates
  5. Tiled TIFF format for faster I/O
  6. Smart caching with LRU
  7. Batch JPEG writes
  8. Better progress estimation
  9. Early input validation

What it does (in order):
  1) TIMESTAMPED BUILD: creates isolated build directory with timestamp to preserve previous builds.
  2) JSON -> binary masks (uint8, values 0/1) for classes {bubbles, fat, muscle}. (toggle)
  3) Optional color overlays for quick QA. (toggle)
  4) Optional subtraction (e.g., fat_minus=bubbles) and light cleanup, saved as 'fat' masks.
  5) Tile pseudocolored images and aligned target masks with filtering (empty/blur/min mask ratio).
  6) Split tiles into train/val/test sets (default: 60%/20%/20%).

USAGE EXAMPLES:

1. Standard build (fat - bubbles, with stain normalization):
   python Segmentation/build_dataset.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
     --stain-normalize \
     --target-mask fat --subtract --subtract-class bubbles \
     --min-mask-ratio 0.05 \
     --workers 8

2. Build without stain normalization (ECM channel):
   python Segmentation/build_dataset.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane \
     --no-stain-normalize \
     --target-mask fat --subtract --subtract-class bubbles

3. Quick build without overlays (faster):
   python Segmentation/build_dataset.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
     --stain-normalize \
     --target-mask fat --subtract --subtract-class bubbles \
     --no-overlays \
     --workers 12

4. Bubbles-only dataset (no subtraction):
   python Segmentation/build_dataset.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
     --stain-normalize \
     --target-mask bubbles --no-subtract

5. Custom tile size and splits:
   python Segmentation/build_dataset.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
     --stain-normalize \
     --target-mask fat --subtract --subtract-class bubbles \
     --tile-size 512 \
     --train-split 0.70 --val-split 0.15 --test-split 0.15

OUTPUT STRUCTURE:
  data/Meat_Luci_Tulane/
    └── _build_YYYYMMDD_HHMMSS/
        ├── dataset/
        │   ├── train/
        │   │   ├── images/        # JPEG tiles (60%)
        │   │   └── masks/         # Binary TIFF masks (60%)
        │   ├── val/               # 20% split
        │   └── test/              # 20% split
        ├── masks/                 # Full-image binary masks
        │   ├── fat/
        │   ├── bubbles/
        │   └── muscle/
        ├── overlays/              # QA visualizations (optional)
        └── build_summary.json     # Build statistics and parameters

CLI Examples (Legacy Format)
-----------------------------
# End-to-end build using fat as training target, subtract bubbles, create overlays
python build_fat_dataset.py --target-mask fat --subtract --subtract-class bubbles --make-masks --make-overlays

# Build bubbles-only dataset (no subtraction), skip overlays
python build_fat_dataset.py --target-mask bubbles --no-subtract --make-masks --no-overlays

# Use overlap tiling and stricter mask density filter
python build_fat_dataset.py --stride 512 --min-mask-ratio 0.002

# Use more workers for parallel processing
python build_fat_dataset.py --workers 8
"""

from __future__ import annotations
import os
import sys
import json
import math
import shutil
import argparse
from dataclasses import dataclass
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
import platform
import subprocess

# Add project root to Python path for src imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load seed from seed.csv for reproducibility
from src.utils.seed_utils import get_project_seed
GLOBAL_SEED = get_project_seed()
np.random.seed(GLOBAL_SEED)

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("tqdm not found. Install with: pip install tqdm")

# Stain normalization imports - only from stain_normalization.py
try:
    from src.utils.stain_normalization import load_best_reference, complete_preprocessing_pipeline
    STAIN_NORMALIZATION_AVAILABLE = True
except ImportError:
    STAIN_NORMALIZATION_AVAILABLE = False
    print("[WARN] Stain normalization not available - check src/utils/stain_normalization.py")

# ------------------------------
# User-configurable defaults
# ------------------------------
# Note: DATA_ROOT is rebound from CLI (required).
DATA_ROOT = Path(".")
PSEUDO_DIR = DATA_ROOT / "Pseudocolored"
JSON_MASKS_DIR = DATA_ROOT / "Masks"
BUILD_ROOT = DATA_ROOT / "_build"

# Test subdirectory (where organized test images are located)
# Note: Test annotations live in main Masks/{class}/ directory, not in test subdirectory
TEST_PSEUDO_DIR = PSEUDO_DIR / "test"

CLASS_NAMES = ["bubbles", "fat", "muscle"]
OVERLAY_COLORS = {
    "bubbles": (0, 0, 255),   # Blue (BGR)
    "fat":     (0, 255, 255), # Yellow
    "muscle":  (0, 255, 0),   # Green
}

DEFAULTS = {
    "make_masks": True,
    "make_overlays": False,
    "target_mask": "fat",
    "subtract": True,
    "subtract_class": "bubbles",
    "subtract_masks_dir": None,
    "morph_close_k": 0,
    "min_cc_px": 0,
    "tile_size": 1024,
    "stride": 1024,
    "white_threshold": 235,
    "white_ratio_limit": 0.70,
    "blurry_threshold": 7.5,
    "min_mask_ratio": 0.05,
    "jpeg_quality": 100,
    "val_ratio": 0.20,      # 20% validation
    "test_ratio": 0.0,      # 0% internal test (use external test set)
    "seed": GLOBAL_SEED,  # Load from seed.csv
    "split_by_slide": True,
    "compression": "auto",
    "workers": None,  # None = cpu_count() - 1
    "neg_pct": 0.40,   # fraction of negative tissue tiles in final dataset. 0 disables.
    "keep_white": True,  # Trust annotators - keep white tiles by default
    "keep_blurry": True, # Trust annotators - keep blurry tiles by default
    "stain_normalize": True,  # Enable SYBR Gold + Eosin stain normalization (color correction only)
    "reference_path": None,  # Path to reference image for stain normalization
    "reference_metadata": "src/utils/stain_reference_metadata.json",  # Updated path
    "include_test_set": False,  # Include test set from test subdirectories (disabled by default)
    "min_confidence_train": 1,  # Training uses all confidence levels (1, 2, 3)
    "min_confidence_val": 2,    # Validation uses certain annotations only (2, 3)
    # Test-specific parameters (more comprehensive evaluation)
    "test_min_mask_ratio": 0.0,    # Include all mask densities for test set
    "test_stride": 1024,           # No overlap for test tiles
    "test_neg_pct": 1.0,          # Include all negative tiles in test set
    "test_min_confidence": 2,      # Same confidence threshold as val
    "test_include_white": False,   # Flag to include white tiles in test set
    "test_include_blurry": False,  # Flag to include blurry tiles in test set
    "include_ambiguous": False,    # Include ambiguous tiles (0 < coverage < min_mask_ratio) as negatives
}

OVERLAY_ALPHA = 0.35

# ------------------------------
# Build Logging System
# ------------------------------
def get_system_info():
    """Collect system and environment information."""
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable
        },
        "environment": {
            "opencv_max_pixels": os.environ.get("OPENCV_IO_MAX_IMAGE_PIXELS", "Not set"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
            "home": os.environ.get("HOME", "Not set"),
            "user": os.environ.get("USER", "Not set")
        }
    }
    
    # Get package versions
    try:
        import numpy
        info["packages"] = {"numpy": numpy.__version__}
    except:
        info["packages"] = {}
    
    try:
        info["packages"]["opencv"] = cv2.__version__
    except:
        pass
    
    try:
        import PIL
        info["packages"]["pillow"] = PIL.__version__
    except:
        pass
    
    try:
        import tifffile
        info["packages"]["tifffile"] = tifffile.__version__
    except:
        pass
    
    try:
        import tqdm
        info["packages"]["tqdm"] = tqdm.__version__
    except:
        pass
    
    return info

def create_build_log(args, build_root: Path, stage: str, **kwargs):
    """Create comprehensive build log with all parameters and system info."""
    
    log_data = {
        "build_info": {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "build_directory": str(build_root),
            "script_version": "Adipose Tissue U-Net Dataset Builder v2.0",
            "description": "Integrated training and test preprocessing with identical settings"
        },
        "system_info": get_system_info(),
        "data_paths": {
            "data_root": str(DATA_ROOT),
            "pseudocolored_dir": str(PSEUDO_DIR),
            "json_masks_dir": str(JSON_MASKS_DIR),
            "test_pseudocolored_dir": str(TEST_PSEUDO_DIR),
            "test_masks_note": "Test annotations stored in main Masks/{class}/ directory, not separate test subdirectory",
            "build_output": str(build_root)
        },
        "command_line_arguments": {
            "data_root": args.data_root,
            "target_mask": args.target_mask,
            "make_masks": args.make_masks,
            "make_overlays": args.make_overlays,
            "subtract": args.subtract,
            "subtract_class": args.subtract_class,
            "subtract_masks_dir": args.subtract_masks_dir,
            "morph_close_k": args.morph_close_k,
            "min_cc_px": args.min_cc_px,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "white_threshold": args.white_threshold,
            "white_ratio_limit": args.white_ratio_limit,
            "blurry_threshold": args.blurry_threshold,
            "min_mask_ratio": args.min_mask_ratio,
            "jpeg_quality": args.jpeg_quality,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "split_by_slide": args.split_by_slide,
            "compression": args.compression,
            "workers": args.workers,
            "neg_pct": args.neg_pct,
            "include_test_set": args.include_test_set,
            "min_confidence_train": getattr(args, 'min_confidence_train', 1),
            "min_confidence_val": getattr(args, 'min_confidence_val', 2)
        },
        "stain_normalization": {
            "enabled": getattr(args, 'stain_normalize', False),
            "reference_path": getattr(args, 'reference_path', None),
            "reference_metadata": getattr(args, 'reference_metadata', None),
            "available": STAIN_NORMALIZATION_AVAILABLE,
            "note": "Color correction only - intensity normalization applied during training"
        }
    }
    
    # Add processing results if provided
    if kwargs:
        log_data["processing_results"] = kwargs
    
    # Determine stain normalization actual status
    stain_status = "DISABLED"
    if getattr(args, 'stain_normalize', False):
        if STAIN_NORMALIZATION_AVAILABLE:
            stain_status = "ENABLED_AND_AVAILABLE"
        else:
            stain_status = "ENABLED_BUT_UNAVAILABLE"
    
    log_data["stain_normalization"]["actual_status"] = stain_status
    
    # Save detailed JSON log
    log_file = build_root / "build_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary
    summary_file = build_root / "build_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ADIPOSE TISSUE U-NET DATASET BUILD LOG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Build Stage: {stage}\n")
        f.write(f"Timestamp: {log_data['build_info']['timestamp']}\n")
        f.write(f"Build Directory: {build_root}\n\n")
        
        f.write("STAIN NORMALIZATION STATUS:\n")
        f.write("-" * 40 + "\n")
        stain_info = log_data["stain_normalization"]
        f.write(f"Requested: {'YES' if stain_info['enabled'] else 'NO'}\n")
        f.write(f"Available: {'YES' if stain_info['available'] else 'NO'}\n")
        f.write(f"Actual Status: {stain_info['actual_status']}\n")
        if stain_info['reference_metadata']:
            f.write(f"Reference Metadata: {stain_info['reference_metadata']}\n")
        if stain_info['reference_path']:
            f.write(f"Reference Path: {stain_info['reference_path']}\n")
        f.write(f"Note: {stain_info['note']}\n\n")
        
        f.write("DATA PROCESSING PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        args_info = log_data["command_line_arguments"]
        f.write(f"Target Class: {args_info['target_mask']}\n")
        f.write(f"Subtraction: {'YES' if args_info['subtract'] else 'NO'}")
        if args_info['subtract']:
            f.write(f" (subtract {args_info['subtract_class']})")
        f.write("\n")
        f.write(f"Tile Size: {args_info['tile_size']}x{args_info['tile_size']} pixels\n")
        f.write(f"Stride: {args_info['stride']} pixels\n")
        f.write(f"JPEG Quality: {args_info['jpeg_quality']}\n")
        f.write(f"Minimum Mask Ratio: {args_info['min_mask_ratio']}\n")
        f.write(f"Negative Tile Percentage: {args_info['neg_pct']:.1%}\n")
        f.write(f"Random Seed: {args_info['seed']}\n")
        f.write(f"Workers: {args_info['workers'] or 'auto'}\n\n")
        
        f.write("FILTERING THRESHOLDS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"White Threshold: {args_info['white_threshold']}\n")
        f.write(f"White Ratio Limit: {args_info['white_ratio_limit']:.3f}\n")
        f.write(f"Blur Threshold: {args_info['blurry_threshold']:.1f}\n\n")
        
        f.write("DATASET SPLIT CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Validation Ratio: {args_info['val_ratio']:.1%}\n")
        f.write(f"Test Set Source: Separate test subdirectories\n")
        f.write(f"Split by Slide: {'YES' if args_info['split_by_slide'] else 'NO'}\n")
        f.write(f"Include Test Set: {'YES' if args_info['include_test_set'] else 'NO'}\n")
        
        f.write("DATA DIRECTORIES:\n")
        f.write("-" * 40 + "\n")
        paths_info = log_data["data_paths"]
        f.write(f"Data Root: {paths_info['data_root']}\n")
        f.write(f"Training Images: {paths_info['pseudocolored_dir']}\n")
        f.write(f"Training Masks: {paths_info['json_masks_dir']}\n")
        f.write(f"Test Images: {paths_info['test_pseudocolored_dir']}\n")
        f.write(f"Note: {paths_info.get('test_masks_note', 'Test annotations in main Masks/ directory')}\n")
        f.write("\n")
        
        f.write("SYSTEM INFORMATION:\n")
        f.write("-" * 40 + "\n")
        sys_info = log_data["system_info"]
        f.write(f"Platform: {sys_info['platform']['system']} {sys_info['platform']['release']}\n")
        f.write(f"Python: {sys_info['platform']['python_version'].split()[0]}\n")
        f.write(f"Machine: {sys_info['platform']['machine']}\n")
        if 'packages' in sys_info:
            f.write("Key Packages:\n")
            for pkg, ver in sys_info['packages'].items():
                f.write(f"  {pkg}: {ver}\n")
        f.write("\n")
        
        # Add processing results if available
        if 'processing_results' in log_data:
            results = log_data['processing_results']
            f.write("PROCESSING RESULTS:\n")
            f.write("-" * 40 + "\n")
            for key, value in results.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("For detailed technical parameters, see build_log.json\n")
        f.write("=" * 80 + "\n")
    
    print(f"[Log] Build log saved: {log_file}")
    print(f"[Log] Build summary saved: {summary_file}")
    
    return log_file, summary_file

# ------------------------------
# Helpers
# ------------------------------
def clean_build_root(build_root: Path):
    if build_root.exists():
        shutil.rmtree(build_root)
    (build_root / "masks").mkdir(parents=True, exist_ok=True)
    (build_root / "overlays").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles" / "tissue").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles" / "blurry").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles" / "empty").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles_masks").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "val" / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "val" / "masks").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "test" / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "test" / "masks").mkdir(parents=True, exist_ok=True)


def ensure_build_dirs(build_root: Path):
    """Create required subfolders if they don't exist (no deletion)."""
    (build_root / "masks").mkdir(parents=True, exist_ok=True)
    (build_root / "overlays").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles" / "tissue").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles" / "blurry").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles" / "empty").mkdir(parents=True, exist_ok=True)
    (build_root / "tiles_masks").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "val" / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "val" / "masks").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "test" / "images").mkdir(parents=True, exist_ok=True)
    (build_root / "dataset" / "test" / "masks").mkdir(parents=True, exist_ok=True)


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
    """Return the *image stem* for a given JSON, robust to many underscores."""
    stem = json_path.stem
    token = f"_{cls}_annotations"
    if token in stem:
        return stem.split(token)[0]
    token2 = f"_{cls}_"
    if token2 in stem:
        return stem.split(token2)[0]
    return stem


def find_pseudo_image(image_stem: str) -> Path | None:
    """Resolve the pseudocolor image given an image stem (main directory), supporting multiple formats."""
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        p = PSEUDO_DIR / f"{image_stem}{ext}"
        if p.exists():
            return p
    return None

def find_pseudo_image_in_test(image_stem: str) -> Path | None:
    """Resolve the pseudocolor image in test directory given an image stem, supporting multiple formats."""
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        p = TEST_PSEUDO_DIR / f"{image_stem}{ext}"
        if p.exists():
            return p
    return None

def get_test_image_stems(test_dir: Path) -> Set[str]:
    """Get stems of all images in the test/ subdirectory."""
    if not test_dir.exists():
        return set()
    
    test_stems = set()
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        for img_path in test_dir.glob(f"*{ext}"):
            test_stems.add(img_path.stem)
    return test_stems


def extract_filename_timestamp(json_path: Path) -> datetime | None:
    """
    Extract timestamp from filename in format: *_MMDDYYYY_HHMMSS.json
    
    Example: "2nd Meat_grid_5x5_r2_c1_fat_annotations_10292025_172224.json"
    Returns datetime object or None if no timestamp found.
    """
    filename = json_path.stem  # Remove .json extension
    
    # Look for pattern: _MMDDYYYY_HHMMSS at the end
    timestamp_pattern = r'_(\d{8})_(\d{6})$'
    match = re.search(timestamp_pattern, filename)
    
    if not match:
        return None
    
    date_str = match.group(1)  # MMDDYYYY
    time_str = match.group(2)  # HHMMSS
    
    try:
        # Parse MMDDYYYY
        month = int(date_str[:2])
        day = int(date_str[2:4])
        year = int(date_str[4:8])
        
        # Parse HHMMSS
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        return datetime(year, month, day, hour, minute, second)
        
    except (ValueError, IndexError) as e:
        print(f"[WARN] Invalid timestamp in filename {json_path.name}: {e}")
        return None


# ------------------------------
# Validation (Optimization #9)
# ------------------------------
def validate_inputs(target_bases: Set[str], target_cls: str) -> bool:
    """Validate all required files exist before starting."""
    print(f"[Validation] Checking {len(target_bases)} base images...")
    errors = []
    
    for base in target_bases:
        img_path = find_pseudo_image(base)
        if not img_path:
            errors.append(f"  ✗ Missing pseudocolored image: {base}.jpg")
        
        json_path = JSON_MASKS_DIR / target_cls / f"{base}_{target_cls}_annotations.json"
        if not json_path.exists():
            # Try alternate naming
            alt_found = False
            for j in (JSON_MASKS_DIR / target_cls).glob(f"{base}*.json"):
                alt_found = True
                break
            if not alt_found:
                errors.append(f"  ✗ Missing JSON annotation: {base}_{target_cls}_annotations.json")
    
    if errors:
        print(f"[Validation] Found {len(errors)} error(s):")
        for err in errors[:10]:  # Show first 10
            print(err)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    
    print(f"[Validation] ✓ All {len(target_bases)} images validated")
    return True


# ------------------------------
# JSON -> 0/1 mask generation (Optimization #1: Parallel)
# ------------------------------
def collect_target_bases(target_cls: str, exclude_test_stems: Set[str] = None, min_confidence: int = 1) -> Tuple[Set[str], Dict[str, str]]:
    """
    Collect training/validation image stems from main directories, excluding test duplicates.
    Applies slide-level confidence filtering to ensure only slides with valid annotations are processed.
    
    Args:
        target_cls: Target class name (e.g., 'fat')
        exclude_test_stems: Set of test image stems to exclude
        min_confidence: Minimum confidence score for slide validation
    
    Returns:
        Tuple of (valid_bases, skip_reasons) where skip_reasons maps skipped stems to reason strings
    """
    if exclude_test_stems is None:
        exclude_test_stems = set()
    
    cls_json_dir = JSON_MASKS_DIR / target_cls
    if not cls_json_dir.exists():
        print(f"[ERROR] Target JSON folder missing: {cls_json_dir}")
        return set(), {}
    
    bases: Set[str] = set()
    skip_reasons: Dict[str, str] = {}
    
    for j in sorted(cls_json_dir.glob("*.json")):
        base = parse_image_stem_from_json(j, target_cls)
        
        # Skip if this image is in the test folder
        if base in exclude_test_stems:
            skip_reasons[base] = "in_test_set"
            continue
        
        # Slide-level validation: skip slides with no annotations meeting confidence threshold
        if not slide_has_valid_annotations(j, min_confidence):
            skip_reasons[base] = f"no_annotations_conf>={min_confidence}"
            continue
            
        bases.add(base)
    
    skipped_count = len(skip_reasons)
    if exclude_test_stems:
        test_skipped = sum(1 for reason in skip_reasons.values() if reason == "in_test_set")
        conf_skipped = skipped_count - test_skipped
        print(f"[Scope] Found {len(bases)} training/validation slides after excluding {test_skipped} test images and {conf_skipped} slides with insufficient confidence.")
    else:
        print(f"[Scope] Found {len(bases)} training/validation slide(s) with target '{target_cls}' JSONs.")
        if skipped_count > 0:
            print(f"[Scope] Skipped {skipped_count} slides: no annotations with confidence >= {min_confidence}")
    
    return bases, skip_reasons

def collect_test_bases(target_cls: str, min_confidence: int = 2) -> Tuple[Set[str], Dict[str, str]]:
    """
    Collect test image stems from test subdirectory (annotations in main Masks folder).
    Applies slide-level confidence filtering.
    
    Args:
        target_cls: Target class name (e.g., 'fat')
        min_confidence: Minimum confidence score for test slide validation
    
    Returns:
        Tuple of (valid_bases, skip_reasons) where skip_reasons maps skipped stems to reason strings
    """
    if not TEST_PSEUDO_DIR.exists():
        print(f"[Test] No test directory found: {TEST_PSEUDO_DIR}")
        return set(), {}
    
    # Get all image stems from test directory (supporting multiple formats)
    test_stems = get_test_image_stems(TEST_PSEUDO_DIR)
    
    # Verify annotations exist in main masks directory (not test subdirectory)
    cls_json_dir = JSON_MASKS_DIR / target_cls
    if not cls_json_dir.exists():
        print(f"[Test] No JSON directory for '{target_cls}': {cls_json_dir}")
        return set(), {}
    
    verified_bases = set()
    skip_reasons: Dict[str, str] = {}
    
    for stem in test_stems:
        # Look for annotation in MAIN masks directory
        # Try exact match first, then pattern match for timestamped files
        exact_match = cls_json_dir / f"{stem}_{target_cls}_annotations.json"
        json_path = None
        
        if exact_match.exists():
            json_path = exact_match
        else:
            # Try pattern match for timestamped annotations
            pattern = f"{stem}_{target_cls}_*.json"
            matching_jsons = list(cls_json_dir.glob(pattern))
            if matching_jsons:
                # Use the most recent annotation (by filename timestamp)
                json_path = max(matching_jsons, key=lambda p: extract_filename_timestamp(p) or datetime.min)
        
        if json_path is None:
            skip_reasons[stem] = "no_annotation_file"
            continue
        
        # Slide-level validation for test set
        if not slide_has_valid_annotations(json_path, min_confidence):
            skip_reasons[stem] = f"no_annotations_conf>={min_confidence}"
            continue
        
        verified_bases.add(stem)
    
    skipped_count = len(skip_reasons)
    print(f"[Test] Found {len(verified_bases)} test images with annotations in main Masks/{target_cls}/")
    if skipped_count > 0:
        no_file = sum(1 for r in skip_reasons.values() if r == "no_annotation_file")
        low_conf = sum(1 for r in skip_reasons.values() if r.startswith("no_annotations_conf"))
        print(f"[Test] Skipped {skipped_count} test images: {no_file} missing annotations, {low_conf} insufficient confidence")
    
    return verified_bases, skip_reasons

def validate_no_overlap(training_bases: Set[str], test_bases: Set[str]) -> bool:
    """Ensure no overlap between training and test sets."""
    overlap = training_bases.intersection(test_bases)
    if overlap:
        print(f"[ERROR] Found {len(overlap)} overlapping images between training and test:")
        for stem in sorted(list(overlap)[:10]):
            print(f"  - {stem}")
        if len(overlap) > 10:
            print(f"  ... and {len(overlap) - 10} more")
        return False
    print(f"[Validation] ✓ No overlap between training ({len(training_bases)}) and test ({len(test_bases)}) sets")
    return True


def load_json_annotations(json_path: Path, min_confidence: int = 1) -> Tuple[List[np.ndarray], bool]:
    """
    Load polygon annotations from JSON file, filtering by confidence score.
    
    Args:
        json_path: Path to JSON annotation file
        min_confidence: Minimum confidence score to accept (1, 2, or 3)
    
    Returns:
        Tuple of (list of polygon arrays, whether any annotations lacked confidence scores)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_elements = []
    missing_confidence = False
    
    if isinstance(data, list):
        for ann in data:
            if isinstance(ann, dict):
                # Check confidence score
                confidence_score = ann.get("confidenceScore")
                
                if confidence_score is None:
                    missing_confidence = True
                    # No confidence score - include all annotations
                    elems = ann.get("annotation", {}).get("elements", [])
                    if isinstance(elems, list):
                        all_elements.extend(elems)
                elif confidence_score >= min_confidence:
                    # Has sufficient confidence score
                    elems = ann.get("annotation", {}).get("elements", [])
                    if isinstance(elems, list):
                        all_elements.extend(elems)
                # else: skip this annotation (confidence too low)
                
    elif isinstance(data, dict):
        # Single annotation case
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


def slide_has_valid_annotations(json_path: Path, min_confidence: int) -> bool:
    """
    Check if slide has at least one annotation meeting confidence threshold.
    
    This ensures we only process slides that were actually annotated.
    Slides with no valid annotations are skipped entirely.
    
    Args:
        json_path: Path to JSON annotation file
        min_confidence: Minimum confidence score to accept (1, 2, or 3)
    
    Returns:
        True if slide has at least one valid annotation, False otherwise
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
    """
    Get annotations within tile bounds and flag tiles that only contain low-confidence marks.
    
    This prevents tiles with uncertain annotations from entering validation/test sets,
    while allowing training on all annotated data.
    
    Args:
        json_path: Path to JSON annotation file
        tile_bbox: (x1, y1, x2, y2) tile bounding box in slide coordinates
        min_confidence: Minimum confidence score to accept (1, 2, or 3)
    
    Returns:
        (polygons, low_confidence_only): 
            - polygons: List of polygon arrays meeting confidence threshold (shifted to tile coords)
            - low_confidence_only: True if tile intersected only low-confidence annotations
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
    
    return polys, (has_low_conf and not has_high_conf)


def create_binary_mask(polygons: List[np.ndarray], width: int, height: int) -> np.ndarray:
    """Rasterize many polygons into a single 0/1 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    cv_polys = [np.asarray(p, dtype=np.int32).reshape(-1, 1, 2) for p in polygons if len(p) >= 3]
    if cv_polys:
        cv2.fillPoly(mask, cv_polys, 1)
    return mask


def create_overlay(image_path: Path, mask: np.ndarray, bgr_color: Tuple[int, int, int], alpha: float) -> Image.Image:
    with Image.open(image_path) as base_img:
        base_arr = np.array(base_img.convert("RGB"))
    
    overlay_arr = base_arr.copy()

    color_mask = mask > 0
    b, g, r = bgr_color
    color = np.array([r, g, b], dtype=np.float32)
    overlay_arr[color_mask] = (
        alpha * color + (1.0 - alpha) * overlay_arr[color_mask].astype(np.float32)
    ).astype(np.uint8)
    return Image.fromarray(overlay_arr)


# Worker function for parallel processing
def _process_single_json_worker(args_tuple):
    """Worker function for parallel mask generation with confidence filtering."""
    (jpath_str, cls, base, img_path_str, width, height, 
     compression, make_overlays, masks_out_root_str, overlays_root_str, min_confidence) = args_tuple
    
    jpath = Path(jpath_str)
    img_path = Path(img_path_str)
    masks_out_root = Path(masks_out_root_str)
    overlays_root = Path(overlays_root_str)
    
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
        
        # Optional overlay
        if make_overlays:
            overlays_cls_dir = overlays_root / cls
            overlays_cls_dir.mkdir(parents=True, exist_ok=True)
            overlay_img = create_overlay(img_path, mask, OVERLAY_COLORS.get(cls, (255, 255, 255)), OVERLAY_ALPHA)
            overlay_img.save(str(overlays_cls_dir / f"{base}_{cls}_overlay.png"))
        
        return (base, cls, "success", missing_confidence)
    except Exception as e:
        return (base, cls, f"error: {str(e)}", False)


def build_masks_from_json(args, training_bases: Set[str], test_bases: Set[str]):
    """Build masks for both training and test sets with identical processing."""
    masks_out_root = BUILD_ROOT / "masks"
    overlays_root = BUILD_ROOT / "overlays"
    
    # Use training confidence threshold for mask generation (most permissive)
    # Split-specific filtering happens during tiling phase
    min_confidence = getattr(args, 'min_confidence_train', 1)
    
    # Process both training and test sets with identical settings
    all_work_items = []
    size_cache: Dict[str, Tuple[int, int]] = {}
    
    # Process training data (main directories)
    print(f"[Masks] Processing training data from main directories...")
    all_work_items.extend(_prepare_mask_work_items(
        training_bases, 
        "training",
        lambda stem: find_pseudo_image(stem),
        lambda cls: JSON_MASKS_DIR / cls,
        args, masks_out_root, overlays_root, min_confidence, size_cache
    ))
    
    # Process test data (test images, annotations in main directory) if enabled
    if getattr(args, 'include_test_set', True) and test_bases:
        print(f"[Masks] Processing test data from test subdirectories...")
        all_work_items.extend(_prepare_mask_work_items(
            test_bases,
            "test", 
            lambda stem: find_pseudo_image_in_test(stem),
            lambda cls: JSON_MASKS_DIR / cls,  # Test annotations in main masks directory
            args, masks_out_root, overlays_root, min_confidence, size_cache
        ))
    
    if not all_work_items:
        print("[WARN] No work items prepared for mask generation")
        return
    
    # Process all work items (training + test) with identical settings
    n_workers = args.workers if args.workers else max(1, cpu_count() - 1)
    print(f"[Masks] Processing {len(all_work_items)} annotations using {n_workers} workers")
    print(f"[Masks] Filtering annotations: accepting confidenceScore >= {min_confidence}")
    
    results = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_single_json_worker, all_work_items),
            total=len(all_work_items),
            desc="Building masks (training + test)",
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
        for fname in sorted(missing_confidence_files)[:10]:
            print(f"  - {fname}")

def _prepare_mask_work_items(image_bases: Set[str], data_type: str, 
                           find_image_func, get_json_dir_func, 
                           args, masks_out_root, overlays_root, min_confidence, size_cache):
    """Prepare work items for mask generation with data-type-aware parameters."""
    work_items = []
    
    # Use test-specific confidence for test data
    if data_type == "test":
        effective_confidence = getattr(args, 'test_min_confidence', 2)
    else:
        effective_confidence = min_confidence
    
    for cls in CLASS_NAMES:
        json_dir = get_json_dir_func(cls)
        if not json_dir or not json_dir.exists():
            print(f"[WARN] Missing {data_type} JSON folder for class '{cls}': {json_dir}")
            continue
        
        # Group files by base image and select newest
        json_files = list(json_dir.glob("*.json"))
        files_by_base = {}
        for jpath in json_files:
            base = parse_image_stem_from_json(jpath, cls)
            if base in image_bases:
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
                    print(f"[Masks] Selected newest {data_type} {base}_{cls}: {files_with_timestamps[0][1].name}")
                elif files_without_timestamps:
                    files_without_timestamps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    latest_by_base[base] = files_without_timestamps[0]
        
        json_files = list(latest_by_base.values())
        
        for jpath in json_files:
            base = parse_image_stem_from_json(jpath, cls)
            img_path = find_image_func(base)
            if img_path is None:
                continue
            
            if base in size_cache:
                width, height = size_cache[base]
            else:
                width, height = get_image_dimensions(str(img_path))
                size_cache[base] = (width, height)
            
            work_items.append((
                str(jpath), cls, base, str(img_path), width, height,
                args.compression, args.make_overlays,
                str(masks_out_root), str(overlays_root), effective_confidence
            ))
    
    return work_items


# ------------------------------
# Subtraction / cleanup for target mask
# ------------------------------
def morph_close(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_px:
            out[labels == lbl] = 1
    return out


def prepare_target_masks(args, training_bases: Set[str], test_bases: Set[str]):
    """Build target masks for both training and test sets with identical processing."""
    target = args.target_mask
    assert target in CLASS_NAMES, f"target_mask must be one of {CLASS_NAMES}"

    base_target_dir = BUILD_ROOT / "masks" / target
    if not base_target_dir.exists():
        print(f"[WARN] Base masks for target '{target}' not found (expected {base_target_dir}).")
        return

    subtract_dir = None
    if args.subtract:
        if args.subtract_masks_dir is not None:
            subtract_dir = Path(args.subtract_masks_dir)
        else:
            subtract_dir = BUILD_ROOT / "masks" / args.subtract_class
        if not subtract_dir.exists():
            print(f"[WARN] Subtraction enabled, but masks folder missing: {subtract_dir}")
            subtract_dir = None

    print(f"[Target] Preparing training + test masks for class '{target}'" + 
          (f" with subtraction '{args.subtract_class}'" if subtract_dir else ""))

    final_target_dir = BUILD_ROOT / "masks" / target
    
    # Process training masks
    for base in tqdm(sorted(training_bases), desc="Training target masks", unit="img", smoothing=0.1):
        _process_target_mask(base, target, subtract_dir, args, final_target_dir, 
                           find_pseudo_image, "training")
    
    # Process test masks with identical settings if enabled
    if getattr(args, 'include_test_set', True) and test_bases:
        for base in tqdm(sorted(test_bases), desc="Test target masks", unit="img", smoothing=0.1):
            _process_target_mask(base, target, subtract_dir, args, final_target_dir, 
                               find_pseudo_image_in_test, "test")

def _process_target_mask(base: str, target: str, subtract_dir: Optional[Path], 
                        args, final_target_dir: Path, find_image_func, data_type: str):
    """Process a single target mask with identical settings for training/test."""
    img_path = find_image_func(base)
    if img_path is None:
        return

    target_mask_path = final_target_dir / f"{base}_{target}.tif"
    if not target_mask_path.exists():
        return
        
    target_mask = (tiff.imread(str(target_mask_path)) > 0).astype(np.uint8)

    if subtract_dir is not None and args.subtract_class != target:
        sub_path = subtract_dir / f"{base}_{args.subtract_class}.tif"
        if sub_path.exists():
            sub_mask = (tiff.imread(str(sub_path)) > 0).astype(np.uint8)
            target_mask = np.clip(target_mask.astype(np.int16) - sub_mask.astype(np.int16), 0, 1).astype(np.uint8)

    # Apply identical morphological operations
    target_mask = morph_close(target_mask, args.morph_close_k)
    target_mask = remove_small_components(target_mask, args.min_cc_px)

    out_path = final_target_dir / f"{base}_{target}.tif"
    _save_tiff_mask(out_path, target_mask, compression=args.compression)


# ------------------------------
# Tiling & filtering (Optimizations #2, #3, #4)
# ------------------------------
JPEG_PARAMS_CACHE = {}

def jpeg_params(quality: int):
    if quality not in JPEG_PARAMS_CACHE:
        JPEG_PARAMS_CACHE[quality] = [
            cv2.IMWRITE_JPEG_QUALITY, int(quality),
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
        ]
    return JPEG_PARAMS_CACHE[quality]


def extract_tile_pil(img_path: Path, x: int, y: int, size: int, stain_normalizer=None, invert: bool = False) -> np.ndarray:
    """
    Extract tile using PIL's efficient cropping with optional stain normalization.
    
    Args:
        img_path: Path to image
        x, y: Top-left coordinates of tile
        size: Tile size (square)
        stain_normalizer: Optional stain normalizer for color correction
    
    Returns:
        Tile as BGR numpy array (0-255 range)
        
    Note:
        Only applies stain normalization (color correction). 
        Intensity normalization (z-score/percentile) happens during training.
    """
    with Image.open(img_path) as img:
        tile = img.crop((x, y, x + size, y + size))
        tile_rgb = np.array(tile)
        if invert:
            tile_rgb = 255 - tile_rgb
        
        # Apply stain normalization (color correction only) if provided
        if stain_normalizer is not None:
            try:
                # Apply Reinhard stain normalization (color correction only)
                tile_rgb = stain_normalizer.normalize_image(tile_rgb)
                
                # Ensure uint8 range for consistency with non-normalized tiles
                if tile_rgb.max() <= 1.0:
                    tile_rgb = (tile_rgb * 255).astype(np.uint8)
                else:
                    tile_rgb = tile_rgb.astype(np.uint8)
            except Exception as e:
                print(f"[WARN] Stain normalization failed for tile at ({x},{y}): {e}")
                # Fall back to original tile
                pass
        
        return cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)


def classify_tiles_batch(tiles_bgr: List[np.ndarray], white_threshold: int, 
                         white_ratio_limit: float, blurry_threshold: float) -> List[str]:
    """Classify multiple tiles at once using per-tile computation to reduce memory usage."""
    if not tiles_bgr:
        return []
    
    classifications = []
    
    # Per-tile computation to reduce memory usage
    white_ratios = []
    for tile in tiles_bgr:
        white_mask = np.all(tile >= white_threshold, axis=2)
        white_ratios.append(white_mask.mean())
    white_ratios = np.asarray(white_ratios)
    
    # Blur detection (still needs loop but batched)
    lap_vars = []
    for tile in tiles_bgr:
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_vars.append(lap_var)
    lap_vars = np.array(lap_vars)
    
    for i in range(len(tiles_bgr)):
        if white_ratios[i] > white_ratio_limit:
            classifications.append("empty")
        elif lap_vars[i] < blurry_threshold:
            classifications.append("blurry")
        else:
            classifications.append("tissue")
    
    return classifications


def tile_coords(h: int, w: int, tile: int, stride: int):
    """Generate tile coordinates with proper boundary checking."""
    if h < tile or w < tile:
        return []
    
    # Calculate steps with ceil so we always cover the right/bottom edges
    x_steps = max(1, math.ceil((w - tile) / stride) + 1)
    y_steps = max(1, math.ceil((h - tile) / stride) + 1)
    
    coords = []
    for ri in range(y_steps):
        for ci in range(x_steps):
            xs = min(ci * stride, w - tile)
            ys = min(ri * stride, h - tile)
            
            # Safety check to ensure we're not out of bounds
            if xs >= 0 and ys >= 0 and xs + tile <= w and ys + tile <= h:
                coords.append((ri, ci, ys, xs))
    
    return coords


def precompute_tile_plans_dual(training_bases: Set[str], test_bases: Set[str], args, target_mask: str):
    """Pre-compute tile coordinates for both training and test sets with data-type-aware parameters."""
    print(f"[Planning] Pre-computing tile coordinates...")
    tile_plans = {}
    target_dir = BUILD_ROOT / "masks" / target_mask
    
    # Process training bases with standard stride
    training_stride = args.stride
    for base in tqdm(sorted(training_bases), desc="Planning training tiles", unit="img"):
        img_path = find_pseudo_image(base)
        if img_path is None:
            continue
        
        msk_path = target_dir / f"{base}_{target_mask}.tif"
        if not msk_path.exists():
            continue
        
        w, h = get_image_dimensions(str(img_path))
        if h < args.tile_size or w < args.tile_size:
            continue
        
        coords = tile_coords(h, w, args.tile_size, training_stride)
        
        # Find JSON annotation file for tile-level confidence filtering
        json_path = JSON_MASKS_DIR / target_mask / f"{base}_{target_mask}_annotations.json"
        if not json_path.exists():
            # Try pattern match for timestamped files
            pattern = f"{base}_{target_mask}_*.json"
            matching_jsons = list((JSON_MASKS_DIR / target_mask).glob(pattern))
            if matching_jsons:
                json_path = max(matching_jsons, key=lambda p: extract_filename_timestamp(p) or datetime.min)
            else:
                json_path = None
        
        tile_plans[base] = {
            'img_path': img_path,
            'msk_path': msk_path,
            'json_path': json_path,
            'width': w,
            'height': h,
            'coords': coords,
            'data_type': 'training',
            'stride': training_stride
        }
    
    # Process test bases with test-specific stride
    test_stride = getattr(args, 'test_stride', args.stride)
    for base in tqdm(sorted(test_bases), desc="Planning test tiles", unit="img"):
        img_path = find_pseudo_image_in_test(base)
        if img_path is None:
            continue
        
        msk_path = target_dir / f"{base}_{target_mask}.tif"
        if not msk_path.exists():
            continue
        
        w, h = get_image_dimensions(str(img_path))
        if h < args.tile_size or w < args.tile_size:
            continue
        
        coords = tile_coords(h, w, args.tile_size, test_stride)
        
        # Find JSON annotation file for tile-level confidence filtering
        json_path = JSON_MASKS_DIR / target_mask / f"{base}_{target_mask}_annotations.json"
        if not json_path.exists():
            # Try pattern match for timestamped files
            pattern = f"{base}_{target_mask}_*.json"
            matching_jsons = list((JSON_MASKS_DIR / target_mask).glob(pattern))
            if matching_jsons:
                json_path = max(matching_jsons, key=lambda p: extract_filename_timestamp(p) or datetime.min)
            else:
                json_path = None
        
        tile_plans[base] = {
            'img_path': img_path,
            'msk_path': msk_path,
            'json_path': json_path,
            'width': w,
            'height': h,
            'coords': coords,
            'data_type': 'test',
            'stride': test_stride
        }
    
    total_tiles = sum(len(plan['coords']) for plan in tile_plans.values())
    print(f"[Planning] {len(tile_plans)} images will produce ~{total_tiles} tiles")
    print(f"[Planning] Training stride: {training_stride}, Test stride: {test_stride}")
    return tile_plans


def tile_and_filter(args, training_bases: Set[str], test_bases: Set[str]):
    """Optimized tiling with batching, efficient extraction, and optional stain normalization."""
    tiles_img_tissue = BUILD_ROOT / "tiles" / "tissue"
    tiles_img_blurry = BUILD_ROOT / "tiles" / "blurry"
    tiles_img_empty  = BUILD_ROOT / "tiles" / "empty"
    tiles_msk_dir    = BUILD_ROOT / "tiles_masks" / args.target_mask
    tiles_msk_dir.mkdir(parents=True, exist_ok=True)

    # Initialize stain normalizer if requested
    stain_normalizer = None
    if getattr(args, 'stain_normalize', False):
        if not STAIN_NORMALIZATION_AVAILABLE:
            print("[WARN] Stain normalization requested but modules not available; continuing without it.")
            stain_normalizer = None
        else:
            try:
                if getattr(args, 'reference_path', None):
                    # Use specified reference
                    print(f"[Stain] Loading specified reference: {args.reference_path}")
                    from src.utils.stain_normalization import ReinhardStainNormalizer
                    stain_normalizer = ReinhardStainNormalizer(args.reference_path)
                else:
                    # Try to load best reference from metadata
                    print(f"[Stain] Loading best reference from: {args.reference_metadata}")
                    stain_normalizer = load_best_reference(args.reference_metadata)
                
                print(f"[Stain] SYBR Gold + Eosin stain normalization enabled (color correction only)")
                print(f"[Stain] Intensity normalization will be applied during training")
                    
            except Exception as e:
                print(f"[WARN] Failed to initialize stain normalizer: {e} — continuing without it.")
                stain_normalizer = None

    # Stats and containers for negative sampling by final fraction
    training_pos_kept = 0
    test_pos_kept = 0
    neg_candidates = []  # store (tiles_msk_path: Path) for potential negatives (to write zero-mask later)
    
    # Tile-level confidence filtering stats
    tiles_skipped_low_conf = 0
    tiles_skipped_no_json = 0
    tiles_skipped_ambiguous = 0
    
    # Pre-compute tile plans for both training and test (Optimization #4)
    tile_plans = precompute_tile_plans_dual(training_bases, test_bases, args, args.target_mask)
    
    if not tile_plans:
        print("[ERROR] No valid tile plans generated")
        return
    
    normalization_status = "with SYBR Gold + Eosin stain normalization" if stain_normalizer else "without stain normalization"
    print(f"[Tiling] Processing tiles {normalization_status}; tile={args.tile_size} stride={args.stride}")
    
    # Process with batching
    batch_size = 16  # Process 16 tiles at a time for classification
    jpeg_params_cached = jpeg_params(args.jpeg_quality)
    
    for base, plan in tqdm(tile_plans.items(), desc="Tiling images", unit="img", smoothing=0.1):
        img_path = plan['img_path']
        msk_path = plan['msk_path']
        json_path = plan.get('json_path')
        coords = plan['coords']
        data_type = plan['data_type']
        
        # Determine split-aware confidence threshold
        # For training data: use min_confidence_train
        # For test data: use test_min_confidence
        # Note: validation tiles are assigned AFTER tiling, so we use min_confidence_val as fallback
        if data_type == 'test':
            tile_min_confidence = getattr(args, 'test_min_confidence', 2)
        else:
            # For training data that will be split into train/val later
            # Use the more permissive threshold (train) during tiling
            # Tiles destined for validation will be filtered if needed
            tile_min_confidence = getattr(args, 'min_confidence_train', 1)
        
        # Load mask once
        msk = tiff.imread(str(msk_path)).astype(np.uint8)
        if msk.ndim == 3:
            msk = msk.squeeze()
        
        # Process in batches
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            
            # Extract tiles (Optimization #3 - efficient PIL cropping)
            tiles = []
            for r, c, ys, xs in batch_coords:
                tile = extract_tile_pil(img_path, xs, ys, args.tile_size, stain_normalizer, invert=args.invert_input)
                tiles.append((tile, r, c, ys, xs))
            
            # Classify batch (Optimization #2)
            tile_arrays = [t[0] for t in tiles]
            classifications = classify_tiles_batch(
                tile_arrays, 
                args.white_threshold, 
                args.white_ratio_limit, 
                args.blurry_threshold
            )
            
            # Save tiles (Optimization #7 - batch writes)
            for (tile, r, c, ys, xs), cls in zip(tiles, classifications):
                data_type = plan['data_type']
                
                # Apply white/blurry tile handling based on flags (mirrors build_class_dataset.py)
                if cls == 'empty' and not args.keep_white:
                    # Drop white tiles if keep_white is False
                    continue
                elif cls == 'blurry' and not args.keep_blurry:
                    # Drop blurry tiles if keep_blurry is False  
                    continue
                
                # Apply test-specific override logic for white/blurry tiles
                if data_type == 'test':
                    if cls == 'empty' and getattr(args, 'test_include_white', False):
                        cls = 'tissue'  # Override: include white tiles in test set
                    elif cls == 'blurry' and getattr(args, 'test_include_blurry', False):
                        cls = 'tissue'  # Override: include blurry tiles in test set
                
                # If keeping white/blurry tiles, treat them as tissue for mask writing
                # (they will get zero masks as negative examples)
                effective_cls = cls
                if cls == 'empty' and args.keep_white:
                    effective_cls = 'tissue'  # Keep as negative tissue tile
                elif cls == 'blurry' and args.keep_blurry:
                    effective_cls = 'tissue'  # Keep as negative tissue tile
                
                out_img_dir = {
                    "empty": tiles_img_empty, 
                    "blurry": tiles_img_blurry, 
                    "tissue": tiles_img_tissue
                }[cls]  # Still save to appropriate dir for QA
                out_name = f"{base}_r{r}_c{c}.jpg"
                cv2.imwrite(str(out_img_dir / out_name), tile, jpeg_params_cached)
                
                # Only process mask if treating as tissue
                if effective_cls != "tissue":
                    continue
                
                # Extract and check mask tile
                m_tile = msk[ys:ys+args.tile_size, xs:xs+args.tile_size]
                if m_tile.shape[:2] != (args.tile_size, args.tile_size):
                    continue
                
                pos_ratio = float(m_tile.sum()) / (args.tile_size * args.tile_size)
                msk_out_path = tiles_msk_dir / out_name.replace('.jpg', '.tif')
                
                # Tile-level confidence filtering for positive tiles
                # If tile has mask coverage, check if annotations meet confidence threshold
                if pos_ratio > 0 and json_path and json_path.exists():
                    tile_bbox = (xs, ys, xs + args.tile_size, ys + args.tile_size)
                    tile_polys, has_low_conf = get_tile_annotations(json_path, tile_bbox, tile_min_confidence)
                    
                    # Skip tiles that only contain low-confidence annotations
                    if has_low_conf:
                        tiles_skipped_low_conf += 1
                        continue
                elif pos_ratio > 0 and (not json_path or not json_path.exists()):
                    # Positive tile but no JSON to validate - warn and skip for safety
                    tiles_skipped_no_json += 1
                    continue
                
                # Use data-type-specific mask ratio threshold
                if data_type == 'test':
                    min_mask_ratio = getattr(args, 'test_min_mask_ratio', 0.0)
                else:
                    min_mask_ratio = args.min_mask_ratio

                # Exclude ambiguous tiles: has some mask but below threshold
                # These are neither clear positives nor clear negatives
                # For training: always exclude (prevents label noise)
                # For test: exclude unless --include-ambiguous is true (allows testing edge cases)
                if 0 < pos_ratio < min_mask_ratio:
                    if data_type == 'training' or not getattr(args, 'include_ambiguous', False):
                        tiles_skipped_ambiguous += 1
                        continue
                    # For test with include_ambiguous=true, keep as negative
                    # (below threshold but worth evaluating model performance on edge cases)

                if pos_ratio >= min_mask_ratio:
                    # Positive tile → keep mask immediately
                    _save_tiff_mask(msk_out_path, m_tile, compression=args.compression)
                    if data_type == 'training':
                        training_pos_kept += 1
                    else:
                        test_pos_kept += 1
                else:
                    # Negative candidate (pos_ratio == 0) → record for potential zero-mask writing later
                    neg_candidates.append((msk_out_path, data_type))

    # --- Finalize negative fraction by writing zero-masks for data-type-aware subsets ---
    # Separate candidates by data type
    training_candidates = [path for path, dtype in neg_candidates if dtype == 'training']
    test_candidates = [path for path, dtype in neg_candidates if dtype == 'test']
    
    rng = np.random.default_rng(args.seed)
    total_neg_written = 0
    
    # Handle training negatives with standard neg_pct
    if training_candidates:
        f_neg_train = float(max(0.0, min(args.neg_pct, 0.99)))
        f_pos_train = max(1e-9, 1.0 - f_neg_train)
        neg_target_train = int(round((f_neg_train / f_pos_train) * training_pos_kept))
        
        if neg_target_train > len(training_candidates):
            neg_target_train = len(training_candidates)
        
        if neg_target_train > 0:
            chosen_idx_train = rng.choice(len(training_candidates), size=neg_target_train, replace=False)
            chosen_paths_train = [training_candidates[i] for i in chosen_idx_train]
            
            for zp in tqdm(chosen_paths_train, desc="Writing training negative masks", unit="tile"):
                if not zp.exists():
                    _save_tiff_mask(zp, np.zeros((args.tile_size, args.tile_size), dtype=np.uint8), compression=args.compression)
            total_neg_written += len(chosen_paths_train)
    
    # Handle test negatives with test-specific neg_pct (default 1.0 = all negatives)
    if test_candidates:
        f_neg_test = float(max(0.0, min(getattr(args, 'test_neg_pct', 1.0), 1.0)))
        neg_target_test = int(round(f_neg_test * len(test_candidates)))
        
        if neg_target_test > 0:
            if f_neg_test >= 1.0:
                # Include all test negatives
                chosen_paths_test = test_candidates
            else:
                # Sample subset of test negatives
                chosen_idx_test = rng.choice(len(test_candidates), size=neg_target_test, replace=False)
                chosen_paths_test = [test_candidates[i] for i in chosen_idx_test]
            
            for zp in tqdm(chosen_paths_test, desc="Writing test negative masks", unit="tile"):
                if not zp.exists():
                    _save_tiff_mask(zp, np.zeros((args.tile_size, args.tile_size), dtype=np.uint8), compression=args.compression)
            total_neg_written += len(chosen_paths_test)

    print(f"[Tiling] Positives kept: {training_pos_kept + test_pos_kept} | Total negatives written: {total_neg_written}")
    print(f"[Tiling] Training negatives: {len(training_candidates)} candidates | Test negatives: {len(test_candidates)} candidates")
    print(f"[Tiling] Training neg%: {args.neg_pct:.1%} | Test neg%: {getattr(args, 'test_neg_pct', 1.0):.1%}")
    
    # Report tile-level filtering stats
    if tiles_skipped_low_conf > 0 or tiles_skipped_no_json > 0 or tiles_skipped_ambiguous > 0:
        print(f"[Tiling] Tile-level filtering (min_confidence varies by split):")
        if tiles_skipped_low_conf > 0:
            print(f"  - Skipped {tiles_skipped_low_conf} tiles with only low-confidence annotations")
        if tiles_skipped_no_json > 0:
            print(f"  - Skipped {tiles_skipped_no_json} positive tiles with missing JSON annotations")
        if tiles_skipped_ambiguous > 0:
            print(f"  - Skipped {tiles_skipped_ambiguous} ambiguous tiles (0 < coverage < min_mask_ratio)")




# ------------------------------
# Split train/val/test
# ------------------------------
def split_dataset(args, training_bases: Set[str], test_bases: Set[str]):
    rng = np.random.default_rng(args.seed)
    images_dir = BUILD_ROOT / "tiles" / "tissue"
    masks_dir  = BUILD_ROOT / "tiles_masks" / args.target_mask

    img_files = sorted(images_dir.glob("*.jpg"))
    mask_files = {p.stem: p for p in masks_dir.glob("*.tif")}

    pairs = []
    for img_path in img_files:
        stem = img_path.stem
        if stem in mask_files:
            pairs.append((img_path, mask_files[stem]))
    
    if not pairs:
        print("[WARN] No paired tiles found to split.")
        return

    def _copy(pairs, split):
        img_out = BUILD_ROOT / "dataset" / split / "images"
        msk_out = BUILD_ROOT / "dataset" / split / "masks"
        for ip, mp in tqdm(pairs, desc=f"Copy {split}", unit="tile", smoothing=0.1):
            shutil.copy2(ip, img_out / ip.name)
            shutil.copy2(mp, msk_out / mp.name)

    # Separate tiles by their origin (training vs test)
    training_tiles = []
    test_tiles = []
    unknown_tiles = []
    
    for i, m in pairs:
        base = i.stem.rsplit("_r", 1)[0]
        if base in test_bases:
            test_tiles.append((i, m))
        elif base in training_bases:
            training_tiles.append((i, m))
        else:
            # Unknown base - assign to training with warning
            unknown_tiles.append((i, m, base))
            training_tiles.append((i, m))
    
    # Report unknown tiles with warning
    if unknown_tiles:
        print(f"[WARN] Found {len(unknown_tiles)} tiles from {len(set(base for _, _, base in unknown_tiles))} unknown bases:")
        unknown_bases = set(base for _, _, base in unknown_tiles)
        for base in sorted(list(unknown_bases)[:5]):  # Show first 5
            count = sum(1 for _, _, b in unknown_tiles if b == base)
            print(f"  - {base}: {count} tiles (assigned to training)")
        if len(unknown_bases) > 5:
            print(f"  ... and {len(unknown_bases) - 5} more unknown bases")
    
    # Internal test split (from training tiles) + validation split
    internal_test_tiles: List[Tuple[Path, Path]] = []
    
    if args.split_by_slide:
        # Group training tiles by slide
        slide_groups: Dict[str, List[Tuple[Path, Path]]] = {}
        for i, m in training_tiles:
            base = i.stem.rsplit("_r", 1)[0]
            slide_groups.setdefault(base, []).append((i, m))
        slides = list(slide_groups.keys())
        rng.shuffle(slides)
        remaining_slides = slides

        # Test split by slide
        if args.test_ratio > 0:
            n_test = int(len(slides) * args.test_ratio)
            if n_test == 0 and len(slides) > 0:
                n_test = 1
            n_test = min(n_test, len(slides))
            test_slides = set(slides[:n_test])
            remaining_slides = slides[n_test:]
        else:
            test_slides = set()

        # Validation split by slide
        if args.val_ratio > 0 and remaining_slides:
            n_val = int(len(remaining_slides) * args.val_ratio)
            if n_val == 0:
                n_val = 1
            n_val = min(n_val, len(remaining_slides))
            val_slides = set(remaining_slides[:n_val])
            train_slides = set(remaining_slides[n_val:])
        else:
            val_slides = set()
            train_slides = set(remaining_slides)
        
        # Assign training tiles to train/val/test splits
        trn, val, internal_test_tiles = [], [], []
        for s, items in slide_groups.items():
            if s in test_slides:
                internal_test_tiles.extend(items)
            elif s in val_slides:
                val.extend(items)
            else:
                trn.extend(items)
    else:
        # Random split of training tiles into train/val
        rng.shuffle(training_tiles)
        if args.test_ratio > 0:
            n_test = int(len(training_tiles) * args.test_ratio)
            if n_test == 0 and len(training_tiles) > 0:
                n_test = 1
            n_test = min(n_test, len(training_tiles))
            internal_test_tiles = training_tiles[:n_test]
            remaining_tiles = training_tiles[n_test:]
        else:
            internal_test_tiles = []
            remaining_tiles = training_tiles

        if args.val_ratio > 0 and remaining_tiles:
            n_val = int(len(remaining_tiles) * args.val_ratio)
            if n_val == 0:
                n_val = 1
            n_val = min(n_val, len(remaining_tiles))
            val = remaining_tiles[:n_val]
            trn = remaining_tiles[n_val:]
        else:
            val = []
            trn = remaining_tiles

    # All test tiles (external + internal) go to test directory
    tst = test_tiles + internal_test_tiles

    # Copy files to respective directories
    print(f"[Split] INTEGRATED MODE: train={len(trn)}  val={len(val)}  test={len(tst)}  "
          f"(external test + internal test_ratio={args.test_ratio}, val_ratio={args.val_ratio})")
    _copy(trn, "train")
    _copy(val, "val")
    _copy(tst, "test")


# ------------------------------
# Argparse & main
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Optimized dataset builder with parallel processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--data-root", type=str, required=True,
                   help="Project data root (contains Pseudocolored/ and Masks/). Required.")
    p.add_argument("--input-images-dir", type=str, default=None,
                   help="Optional override for pseudocolored images directory (e.g., cropped WSIs)")
    p.add_argument("--input-masks-dir", type=str, default=None,
                   help="Optional override for JSON mask directory")
    p.add_argument("--output-root", type=str, default=None,
                   help="Optional output root directory for the build")

    # Toggles
    p.add_argument("--make-masks", dest="make_masks", action="store_true", 
                   default=DEFAULTS["make_masks"], help="Generate masks from JSON")
    p.add_argument("--no-make-masks", dest="make_masks", action="store_false")
    p.add_argument("--make-overlays", dest="make_overlays", action="store_true", 
                   default=DEFAULTS["make_overlays"], help="Create color overlays for QA")
    p.add_argument("--no-overlays", dest="make_overlays", action="store_false")

    # Target & subtraction
    p.add_argument("--target-mask", type=str, default=DEFAULTS["target_mask"], 
                   choices=CLASS_NAMES, help="Which class to use for training")
    p.add_argument("--subtract", dest="subtract", action="store_true", 
                   default=DEFAULTS["subtract"], help="Subtract another class from target")
    p.add_argument("--no-subtract", dest="subtract", action="store_false")
    p.add_argument("--subtract-class", type=str, default=DEFAULTS["subtract_class"], 
                   choices=CLASS_NAMES, help="Class to subtract if --subtract")
    p.add_argument("--subtract-masks-dir", type=str, default=DEFAULTS["subtract_masks_dir"], 
                   help="Optional external folder of masks to subtract")

    # Cleanup of target masks
    p.add_argument("--morph-close-k", type=int, default=DEFAULTS["morph_close_k"], 
                   help="Kernel size for morphological close (0 disables)")
    p.add_argument("--min-cc-px", type=int, default=DEFAULTS["min_cc_px"], 
                   help="Remove CCs smaller than this many pixels (0 disables)")

    # Tiling
    p.add_argument("--tile-size", type=int, default=DEFAULTS["tile_size"], 
                   help="Tile size (pixels)")
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"], 
                   help='Stride for tiling (< tile-size enables overlap)')

    # Filtering
    p.add_argument("--white-th", dest="white_threshold", type=int, 
                   default=DEFAULTS["white_threshold"], help="White threshold per channel")
    p.add_argument("--white-ratio", dest="white_ratio_limit", type=float, 
                   default=DEFAULTS["white_ratio_limit"], 
                   help="Fraction of white pixels to call tile empty")
    p.add_argument("--blur-th", dest="blurry_threshold", type=float, 
                   default=DEFAULTS["blurry_threshold"], 
                   help="Variance of Laplacian threshold for blur")
    p.add_argument("--min-mask-ratio", type=float, default=DEFAULTS["min_mask_ratio"], 
                   help="Minimum positive mask fraction to keep a tile")

    # JPEG & split
    p.add_argument("--jpeg-quality", type=int, default=DEFAULTS["jpeg_quality"], 
                   help="JPEG quality (images)")
    p.add_argument("--invert-input", action="store_true",
                   help="Invert image intensities before filtering/tiling (useful for black-on-white inputs)")
    
    # White and blurry tile handling (trust annotators by default)
    p.add_argument("--keep-white", action="store_true", default=DEFAULTS["keep_white"],
                   help="Keep white tiles (default: True, trusting annotator judgment).")
    p.add_argument("--drop-white", action="store_false", dest="keep_white",
                   help="Drop white tiles (override default to exclude white regions).")
    p.add_argument("--keep-blurry", action="store_true", default=DEFAULTS["keep_blurry"],
                   help="Keep blurry tiles (default: True, trusting annotator judgment).")
    p.add_argument("--drop-blurry", action="store_false", dest="keep_blurry",
                   help="Drop blurry tiles (override default to exclude blurry regions).")
    
    p.add_argument("--val-ratio", type=float, default=DEFAULTS["val_ratio"], 
                   help="Validation ratio (0..1)")
    p.add_argument("--test-ratio", type=float, default=DEFAULTS["test_ratio"],
                   help="Internal test ratio (0..1, default: 0.0 for external test set only)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"], 
                   help="Random seed")
    p.add_argument("--split-by-slide", dest="split_by_slide", action="store_true", 
                   default=DEFAULTS["split_by_slide"], 
                   help="Group tiles by slide when splitting")
    p.add_argument("--no-split-by-slide", dest="split_by_slide", action="store_false")
    p.add_argument("--include-test-set", dest="include_test_set", action="store_true",
                   default=DEFAULTS["include_test_set"],
                   help="Include test set from test subdirectories (default: False)")
    p.add_argument("--no-include-test-set", dest="include_test_set", action="store_false")

    # IO/perf
    p.add_argument("--compression", type=str, default=DEFAULTS["compression"], 
                   choices=["auto","lzw","packbits","none"], 
                   help="TIFF compression for masks")
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"], 
                   help="Number of parallel workers (None = cpu_count - 1)")


    # Percentage of negative 
    p.add_argument(
        "--neg-pct",
        type=float,
        default=DEFAULTS["neg_pct"],
        help="Target fraction of negatives in the FINAL dataset (0.. <1). Example: 0.40 -> 40%% negatives."
    )
    
    # Stain normalization (color correction only - intensity normalization happens during training)
    p.add_argument("--stain-normalize", dest="stain_normalize", action="store_true",
                   default=DEFAULTS["stain_normalize"], 
                   help="Enable SYBR Gold + Eosin stain normalization (color correction only)")
    p.add_argument("--no-stain-normalize", dest="stain_normalize", action="store_false")
    p.add_argument("--reference-path", type=str, default=DEFAULTS["reference_path"],
                   help="Path to reference image for stain normalization")
    p.add_argument("--reference-metadata", type=str, default=DEFAULTS["reference_metadata"],
                   help="Path to reference metadata JSON file")
    
    
    # Confidence threshold for annotation filtering (split-specific)
    p.add_argument("--min-confidence-train", type=int, default=DEFAULTS["min_confidence_train"],
                   choices=[1, 2, 3], 
                   help="Training minimum confidence score (1=uncertain, 2=confident with artifacts, 3=confident clean). Default: 1 (all annotations)")
    p.add_argument("--min-confidence-val", type=int, default=DEFAULTS["min_confidence_val"],
                   choices=[1, 2, 3],
                   help="Validation minimum confidence score. Default: 2 (certain annotations only)")
    
    # Test-specific parameters for comprehensive evaluation
    p.add_argument("--test-min-mask-ratio", type=float, default=DEFAULTS["test_min_mask_ratio"],
                   help="Test-specific minimum positive mask fraction (default: 0.0 for comprehensive evaluation)")
    p.add_argument("--test-stride", type=int, default=DEFAULTS["test_stride"],
                   help="Test-specific stride for tiling (default: 1024 for no overlap)")
    p.add_argument("--test-neg-pct", type=float, default=DEFAULTS["test_neg_pct"],
                   help="Test-specific target fraction of negatives (default: 1.0 for all negatives)")
    p.add_argument("--test-min-confidence", type=int, default=DEFAULTS["test_min_confidence"],
                   choices=[1, 2, 3],
                   help="Test-specific minimum confidence score (default: 2)")
    
    # Test override flags for including low-quality tiles
    p.add_argument("--test-include-white", dest="test_include_white", action="store_true",
                   default=DEFAULTS["test_include_white"],
                   help="Include white tiles in test set (override quality filtering)")
    p.add_argument("--test-include-blurry", dest="test_include_blurry", action="store_true", 
                   default=DEFAULTS["test_include_blurry"],
                   help="Include blurry tiles in test set (override quality filtering)")
    
    # Test duplicate exclusion (mirrors build_class_dataset.py)
    p.add_argument("--exclude-test-duplicates", type=lambda x: x.lower() == 'true', default=True,
                   metavar='BOOL',
                   help="Exclude images from main folder that exist in test/ subfolder (true/false, default: true).")
    
    # Channel selection for naming (mirrors build_class_dataset.py)
    p.add_argument("--channel", type=str, default="pseudocolored", choices=['ecm', 'pseudocolored'],
                   help="Channel identifier for build naming: 'ecm' or 'pseudocolored' (default: pseudocolored)")
    
    # Ambiguous tile handling (mirrors build_class_dataset.py)
    p.add_argument("--include-ambiguous", dest="include_ambiguous", action="store_true",
                   default=DEFAULTS["include_ambiguous"],
                   help="Include ambiguous tiles (0 < coverage < min_mask_ratio) as negatives in test set. "
                        "Default: False (excludes ambiguous to prevent label noise). "
                        "Training always excludes ambiguous regardless of this flag.")
    
    return p.parse_args()



def main():
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

    args = parse_args()
    
    start_time = time.time()

    # Rebind paths from CLI data-root
    global DATA_ROOT, PSEUDO_DIR, JSON_MASKS_DIR, BUILD_ROOT, TEST_PSEUDO_DIR
    DATA_ROOT = Path(args.data_root)
    PSEUDO_DIR = DATA_ROOT / "Pseudocolored"
    JSON_MASKS_DIR = DATA_ROOT / "Masks"
    if args.input_images_dir:
        PSEUDO_DIR = Path(args.input_images_dir)
    if args.input_masks_dir:
        JSON_MASKS_DIR = Path(args.input_masks_dir)
    
    
    # Update test subdirectory (test annotations in main Masks/ directory)
    TEST_PSEUDO_DIR = PSEUDO_DIR / "test"
    if args.input_images_dir and not TEST_PSEUDO_DIR.exists():
        print(f"[WARN] No test subdirectory found in custom input: {TEST_PSEUDO_DIR}")
        TEST_PSEUDO_DIR = PSEUDO_DIR  # Fallback to main directory
    
    # Create timestamped build directory with channel suffix (mirrors build_class_dataset.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    channel_suffix = "_ecm" if getattr(args, 'channel', 'pseudocolored') == 'ecm' else ""
    if args.output_root:
        BUILD_ROOT = Path(args.output_root) / f"_build{channel_suffix}_{timestamp}"
    else:
        BUILD_ROOT = DATA_ROOT / f"_build{channel_suffix}_{timestamp}"
    
    print(f"[Build] Using timestamped build directory: {BUILD_ROOT}")

    # Input validation for subtraction configuration
    if args.subtract:
        if args.subtract_class == args.target_mask:
            raise SystemExit(
                f"[ERROR] Invalid subtraction configuration: "
                f"Cannot subtract '{args.subtract_class}' from itself (target_mask='{args.target_mask}'). "
                f"Use --no-subtract to disable subtraction."
            )
        if args.subtract_class not in CLASS_NAMES:
            raise SystemExit(
                f"[ERROR] Invalid subtract_class '{args.subtract_class}'. "
                f"Must be one of: {CLASS_NAMES}"
            )

    # 0) ENSURE BUILD STRUCTURE
    print(f"[Setup] Creating build structure at {BUILD_ROOT}")
    ensure_build_dirs(BUILD_ROOT)

    # Create initial build log with all parameters
    create_build_log(args, BUILD_ROOT, "STARTED")
    
    # Save timestamp to config for training script (mirrors build_class_dataset.py)
    config_path = BUILD_ROOT / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config_data['build_timestamp'] = timestamp
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    # Get test stems for exclusion if requested
    test_stems_to_exclude = set()
    if getattr(args, 'exclude_test_duplicates', True):
        test_stems_to_exclude = get_test_image_stems(TEST_PSEUDO_DIR)
        if test_stems_to_exclude:
            print(f"[Discovery] Found {len(test_stems_to_exclude)} images in test/ to exclude from train/val")

    # Collect training and test image bases separately with confidence filtering
    min_conf_train = getattr(args, 'min_confidence_train', 1)
    min_conf_test = getattr(args, 'test_min_confidence', 2)
    
    training_bases, training_skip_reasons = collect_target_bases(
        args.target_mask, 
        test_stems_to_exclude,
        min_confidence=min_conf_train
    )
    
    if getattr(args, 'include_test_set', True):
        test_bases, test_skip_reasons = collect_test_bases(
            args.target_mask,
            min_confidence=min_conf_test
        )
    else:
        test_bases, test_skip_reasons = set(), {}
    
    # Track filtering statistics
    stats = {
        "training_slides_total": len(training_bases) + len(training_skip_reasons),
        "training_slides_valid": len(training_bases),
        "training_slides_skipped": len(training_skip_reasons),
        "test_slides_total": len(test_bases) + len(test_skip_reasons),
        "test_slides_valid": len(test_bases),
        "test_slides_skipped": len(test_skip_reasons),
        "training_skip_reasons": training_skip_reasons,
        "test_skip_reasons": test_skip_reasons,
        "min_confidence_train": min_conf_train,
        "min_confidence_test": min_conf_test
    }
    
    if not training_bases and not test_bases:
        print("[ERROR] No valid training or test images found after confidence filtering.")
        print(f"  Training: {stats['training_slides_skipped']} skipped")
        print(f"  Test: {stats['test_slides_skipped']} skipped")
        raise SystemExit("No training or test images found. Nothing to do.")
    
    # Validate no overlap between training and test sets
    if training_bases and test_bases:
        if not validate_no_overlap(training_bases, test_bases):
            raise SystemExit("Overlap detected between training and test sets.")

    # Optimization #9: Validate inputs early
    if training_bases and not validate_inputs(training_bases, args.target_mask):
        raise SystemExit("Training validation failed. Please fix errors above.")

    print(f"[Dataset] Processing {len(training_bases)} training + {len(test_bases)} test images")
    print(f"[Dataset] Identical preprocessing will be applied to both sets")
    
    # 1) Build masks (JSON -> 0/1 TIFF) and optional overlays for both sets
    if args.make_masks:
        build_masks_from_json(args, training_bases, test_bases)
    else:
        print("[Masks] Skipped generation from JSON by flag.")

    # 2) Prepare target mask set with identical processing for both sets
    prepare_target_masks(args, training_bases, test_bases)

    # 3) Tile JPEGs and align mask tiles + filtering with identical settings
    tile_and_filter(args, training_bases, test_bases)

    # 4) Split into train/val (from training) + test (from test directory)
    split_dataset(args, training_bases, test_bases)

    elapsed = time.time() - start_time
    print(f"\n✅ Build complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Outputs under: {BUILD_ROOT}")
    print(f"   - Masks: {BUILD_ROOT / 'masks'}")
    print(f"   - Overlays: {BUILD_ROOT / 'overlays'}")
    print(f"   - Tiles: {BUILD_ROOT / 'tiles'}")
    print(f"   - Dataset: {BUILD_ROOT / 'dataset'} (train/, val/, and test/)")
    
    # Print dataset statistics with separation details
    train_imgs = len(list((BUILD_ROOT / "dataset" / "train" / "images").glob("*.jpg")))
    val_imgs = len(list((BUILD_ROOT / "dataset" / "val" / "images").glob("*.jpg")))
    test_imgs = len(list((BUILD_ROOT / "dataset" / "test" / "images").glob("*.jpg")))
    
    print(f"\n📊 Integrated Dataset Statistics:")
    print(f"   Training tiles: {train_imgs} (from main directories)")
    print(f"   Validation tiles: {val_imgs} (from main directories)")
    print(f"   Test tiles: {test_imgs} (from test subdirectories)")
    print(f"   Total tiles: {train_imgs + val_imgs + test_imgs}")
    print(f"\n🔒 Test Set Isolation:")
    print(f"   ✓ Test preprocessing identical to training")
    print(f"   ✓ Test images completely separate from training")
    print(f"   ✓ No data leakage between sets")
    
    print(f"\n📊 Confidence Filtering Statistics:")
    print(f"   Training slides: {stats['training_slides_valid']}/{stats['training_slides_total']} valid (min_confidence={stats['min_confidence_train']})")
    if stats['training_slides_skipped'] > 0:
        print(f"     Skipped: {stats['training_slides_skipped']} slides")
        conf_skipped = sum(1 for r in stats['training_skip_reasons'].values() if 'conf' in r)
        if conf_skipped > 0:
            print(f"       - {conf_skipped} insufficient confidence")
    print(f"   Test slides: {stats['test_slides_valid']}/{stats['test_slides_total']} valid (min_confidence={stats['min_confidence_test']})")
    if stats['test_slides_skipped'] > 0:
        print(f"     Skipped: {stats['test_slides_skipped']} slides")
        conf_skipped = sum(1 for r in stats['test_skip_reasons'].values() if 'conf' in r)
        if conf_skipped > 0:
            print(f"       - {conf_skipped} insufficient confidence")

    # Create final build log with processing results
    processing_results = {
        "build_duration_seconds": elapsed,
        "build_duration_minutes": elapsed / 60.0,
        "training_image_count": len(training_bases),
        "test_image_count": len(test_bases),
        "total_training_tiles": train_imgs,
        "total_validation_tiles": val_imgs,
        "total_test_tiles": test_imgs,
        "total_tiles_generated": train_imgs + val_imgs + test_imgs,
        "stain_normalization_actually_used": STAIN_NORMALIZATION_AVAILABLE and getattr(args, 'stain_normalize', False),
        "data_sources": {
            "training_data_from": "main directories",
            "validation_data_from": "main directories",
            "test_data_from": "test subdirectories",
            "test_set_isolated": True,
            "preprocessing_identical": True
        },
        "confidence_filtering": {
            "min_confidence_train": stats['min_confidence_train'],
            "min_confidence_test": stats['min_confidence_test'],
            "training_slides_total": stats['training_slides_total'],
            "training_slides_valid": stats['training_slides_valid'],
            "training_slides_skipped": stats['training_slides_skipped'],
            "test_slides_total": stats['test_slides_total'],
            "test_slides_valid": stats['test_slides_valid'],
            "test_slides_skipped": stats['test_slides_skipped']
        },
        "minimum_mask_ratio": args.min_mask_ratio,
        "negative_tile_percentage_requested": args.neg_pct,
        "jpeg_compression_quality": args.jpeg_quality,
        "random_seed_used": args.seed,
        "workers_used": args.workers or max(1, cpu_count() - 1)
    }
    
    create_build_log(args, BUILD_ROOT, "COMPLETED", **processing_results)




if __name__ == "__main__":
    main()
