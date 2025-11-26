#!/usr/bin/env python3
"""
Cut huge WSIs into manageable chunks for 1024×1024 patch extraction.

Strategy:
  • Scan from the top-left creating as many 6144×6144 tiles as possible.
  • Edge regions fall back to tiles sized in multiples of 1024 (e.g., 4096×6144 or 3072×3072)
    to minimize overlap while covering the entire image.
  • Tiles keep the input format by default (metadata preserved) but can be overridden.
  • Default limits: ≤50 MB per tile, process images larger than 13 112 px on either side
    or heavier than the file-size cap. A CLI flag lets you adjust the minimum pixel threshold.

USAGE EXAMPLES:

1. Basic tiling (6144×6144 primary, 3072×3072 edge fallback):
   python pre-post-processing_tools/large_wsi_to_small_wsi_MS.py \
     --input-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/raw_WSI \
     --output-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/WSI/ECM_channel

2. Save enhanced tiles for annotation (CLAHE, percentile, z-score):
   python pre-post-processing_tools/large_wsi_to_small_wsi_MS.py \
     --input-dir data/raw_WSI \
     --output-dir data/tiles \
     --save-enhanced \
     --enhancement-method clahe

3. Convert 16-bit TIFF to 8-bit JPG:
   python pre-post-processing_tools/large_wsi_to_small_wsi_MS.py \
     --input-dir data/raw_WSI \
     --output-dir data/tiles \
     --output-bit-depth 8 \
     --output-format jpg

4. Invert intensity for dark-background stains:
   python pre-post-processing_tools/large_wsi_to_small_wsi_MS.py \
     --input-dir data/raw_WSI \
     --output-dir data/tiles \
     --invert

5. Adjust dimension threshold (process smaller images):
   python pre-post-processing_tools/large_wsi_to_small_wsi_MS.py \
     --input-dir data/raw_WSI \
     --output-dir data/tiles \
     --min-dimension-px 10000

OUTPUT STRUCTURE:
  output_dir/
    ├── Meat_11_13_S5_2_001_x0_y0_w6144_h6144.jpg         # Primary 6144×6144 tile
    ├── Meat_11_13_S5_2_002_x6144_y0_w6144_h6144.jpg
    ├── Meat_11_13_S5_2_028_x18240_y36309_w5120_h5120.jpg # Edge fallback (3072 multiple)
    └── enhanced/                                          # If --save-enhanced
        ├── Meat_11_13_S5_2_001_x0_y0_w6144_h6144_clahe.jpg
        └── ...

ENHANCEMENT METHODS:
  - zscore: Z-score normalization (±3 std → [0,255])
  - percentile: 1-99% intensity stretching
  - clahe: Adaptive histogram equalization (best for annotation)

NOTE: Use this for MS data (ECM channel). For Lucy data, use large_wsi_to_small_wsi_Lucy.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import argparse
import io
import csv

import numpy as np
import cv2
from PIL import Image, PngImagePlugin, TiffImagePlugin
import tifffile

# Disable decompression bomb check for legitimately large stitched images
Image.MAX_IMAGE_PIXELS = None

DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_MAX_DIMENSION_PX = 13112
DEFAULT_MIN_DIMENSION_PX = 13112
PRIMARY_TILE_SIZE = 6144
SECONDARY_TILE_SIZE = 3072
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Runtime-configurable values (populated from CLI args)
MAX_FILE_SIZE_MB = DEFAULT_MAX_FILE_SIZE_MB
MAX_DIMENSION_PX = DEFAULT_MAX_DIMENSION_PX
MIN_DIMENSION_PX = DEFAULT_MIN_DIMENSION_PX
IMAGE_EXTENSIONS = DEFAULT_IMAGE_EXTENSIONS
OUTPUT_FORMAT_OVERRIDE = None
OUTPUT_BIT_DEPTH = "auto"  # auto, 8, 16, 32f
OUTPUT_FORCE_FORMAT = None
INVERT_OUTPUT = False

# Enhancement options for dual output
SAVE_ENHANCED = False          # If True, save enhanced copy for annotation
ENHANCEMENT_METHOD = "clahe"   # zscore, percentile, or clahe

# Skip existing tiles
SKIP_EXISTING = False

# Output options
SAVE_TILES = True
DRY_RUN = False         # If True, only analyze without saving
OUTPUT_FILENAMES = []   # Track all saved tile filenames for CSV export


@dataclass
class SaveConfig:
    """Holds output format information and a factory for save kwargs."""
    format: str
    extension: str
    params_factory: Callable[[], dict]

# ======================================================================== #
# BIT DEPTH AND INVERSION FUNCTIONS
# ======================================================================== #

def convert_bit_depth(arr: np.ndarray, target_depth: str) -> np.ndarray:
    """
    Convert image array to target bit depth.
    
    Args:
        arr: Input image array
        target_depth: One of 'auto', '8', '16', '32f'
        
    Returns:
        Converted array
    """
    if target_depth == "auto":
        # Keep original bit depth
        return arr
    
    # Determine source range info
    if np.issubdtype(arr.dtype, np.floating):
        # Assume float is in [0, 1] range
        arr_normalized = np.clip(arr, 0.0, 1.0)
    elif arr.dtype == np.uint8:
        arr_normalized = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr_normalized = arr.astype(np.float32) / 65535.0
    else:
        # Generic integer type
        max_val = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1
        if max_val == 0:
            max_val = 1
        arr_normalized = arr.astype(np.float32) / max_val
    
    # Convert to target depth
    if target_depth == "8":
        return (arr_normalized * 255.0).round().astype(np.uint8)
    elif target_depth == "16":
        return (arr_normalized * 65535.0).round().astype(np.uint16)
    elif target_depth == "32f":
        return arr_normalized.astype(np.float32)
    else:
        raise ValueError(f"Unknown target bit depth: {target_depth}")


def invert_image(arr: np.ndarray) -> np.ndarray:
    """
    Invert image intensities, handling different bit depths correctly.
    
    Args:
        arr: Input image array
        
    Returns:
        Inverted array with same dtype
    """
    if arr.dtype == np.uint8:
        return (255 - arr).astype(np.uint8)
    elif arr.dtype == np.uint16:
        return (65535 - arr).astype(np.uint16)
    elif np.issubdtype(arr.dtype, np.floating):
        return (1.0 - np.clip(arr, 0.0, 1.0)).astype(arr.dtype)
    else:
        # Generic integer type
        max_val = np.iinfo(arr.dtype).max
        return (max_val - arr).astype(arr.dtype)


def enhance_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel to mean 0, std 1 scaled back to 0-255."""
    arr_float = arr.astype(np.float32)
    mean = arr_float.mean(axis=(0, 1), keepdims=True)
    std = arr_float.std(axis=(0, 1), keepdims=True) + 1e-8
    normalized = (arr_float - mean) / std
    normalized = np.clip((normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8), 0, 1)
    return (normalized * 255).astype(np.uint8)


def enhance_percentile(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based contrast stretching."""
    arr_float = arr.astype(np.float32)
    low = np.percentile(arr_float, p_low, axis=(0, 1), keepdims=True)
    high = np.percentile(arr_float, p_high, axis=(0, 1), keepdims=True)
    stretched = (arr_float - low) / (high - low + 1e-8)
    stretched = np.clip(stretched, 0, 1)
    return (stretched * 255).astype(np.uint8)


def enhance_clahe(arr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Channel-wise CLAHE enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if arr.ndim == 2:
        return clahe.apply(arr)
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        out[:, :, c] = clahe.apply(arr[:, :, c])
    return out


# ======================================================================== #
# ENHANCEMENT FUNCTIONS FOR ANNOTATION
# ======================================================================== #

def enhance_zscore(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalization for contrast enhancement.
    Matches training pipeline normalization.
    
    Args:
        arr: Input image array (any bit depth)
        
    Returns:
        Enhanced 8-bit image
    """
    # Convert to float for processing
    arr_float = arr.astype(np.float32)
    
    # Z-score normalization
    mean = arr_float.mean()
    std = arr_float.std() + 1e-10
    normalized = (arr_float - mean) / std
    
    # Stretch to [0, 255] for visualization (±3 std devs covers ~99.7%)
    stretched = (normalized + 3) / 6 * 255
    
    return np.clip(stretched, 0, 255).astype(np.uint8)


def enhance_percentile(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Percentile-based normalization for robust contrast enhancement.
    Matches training pipeline percentile normalization.
    
    Args:
        arr: Input image array (any bit depth)
        p_low: Lower percentile (default: 1.0)
        p_high: Upper percentile (default: 99.0)
        
    Returns:
        Enhanced 8-bit image
    """
    # Convert to float for processing
    arr_float = arr.astype(np.float32)
    
    # Percentile normalization
    plow, phigh = np.percentile(arr_float, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    normalized = np.clip((arr_float - plow) / scale, 0, 1)
    
    return (normalized * 255).astype(np.uint8)


def enhance_clahe(arr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement.
    Best for visualization of histology images with varying local contrast.
    
    Args:
        arr: Input image array (any bit depth)
        clip_limit: Contrast limiting threshold (default: 2.0)
        tile_size: Size of grid for histogram equalization (default: 8)
        
    Returns:
        Enhanced 8-bit image
    """
    # Convert to 8-bit if needed
    if arr.dtype != np.uint8:
        # Normalize to 8-bit range
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr_8bit = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr_8bit = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_8bit = arr
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(arr_8bit)


# ======================================================================== #
# FORMAT-SPECIFIC SAVE PARAMETER FUNCTIONS
# ======================================================================== #

def get_jpeg_save_params(img: Image.Image) -> dict:
    """
    Return JPEG save params copied from the opened image if available.
    For non-JPEG sources, provide reasonable defaults.
    Works with derived images (crops) — no 'quality=\"keep\"'.
    """
    params = {
        "format": "JPEG",
        "quality": 100,        # default fallback
        "subsampling": 0,     # 4:4:4 preserves detail better for masks/overlays
        "optimize": True,
        "progressive": False,
    }
    try:
        if getattr(img, "format", None) == "JPEG" or img.format == "JPG":
            info = img.info or {}
            if "qtables" in info:
                params["qtables"] = info["qtables"]
            if "subsampling" in info:
                params["subsampling"] = info["subsampling"]
            if "progressive" in info:
                params["progressive"] = info["progressive"]
            if "quality" in info and isinstance(info["quality"], int):
                params["quality"] = info["quality"]
    except Exception:
        pass
    return params


def get_png_save_params(img: Image.Image) -> Callable[[], dict]:
    """Return PNG save params preserving metadata and compression settings."""
    info = dict(img.info or {})
    text_items = [(k, v) for k, v in info.items() if isinstance(v, str)]

    def factory():
        params = {
            "format": "PNG",
            "optimize": info.get("optimize", True),
        }
        if "compress_level" in info:
            params["compress_level"] = info["compress_level"]
        if "dpi" in info:
            params["dpi"] = info["dpi"]
        if "transparency" in info:
            params["transparency"] = info["transparency"]
        if "transparent" in info:
            params["transparent"] = info["transparent"]
        if "gamma" in info:
            params["gamma"] = info["gamma"]
        if "icc_profile" in info:
            params["icc_profile"] = info["icc_profile"]
        if "bits" in info:
            params["bits"] = info["bits"]
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in text_items:
            pnginfo.add_text(key, value)
        if pnginfo.chunks:
            params["pnginfo"] = pnginfo
        return params

    return factory


def get_tiff_save_params(img: Image.Image) -> Callable[[], dict]:
    """Return TIFF save params preserving compression, DPI, and tags."""
    info = dict(img.info or {})
    tag_data = getattr(img, "tag_v2", None)

    def factory():
        params = {"format": "TIFF"}
        if "compression" in info:
            params["compression"] = info["compression"]
        if "dpi" in info:
            params["dpi"] = info["dpi"]
        if "resolution" in info:
            params["resolution"] = info["resolution"]
        if "icc_profile" in info:
            params["icc_profile"] = info["icc_profile"]
        if tag_data is not None:
            try:
                params["tiffinfo"] = tag_data.copy()
            except Exception:
                params["tiffinfo"] = tag_data
        return params

    return factory


def get_fallback_params(fmt: str) -> dict:
    """Fallback save parameters per format in case metadata preservation fails."""
    fmt = fmt.upper()
    if fmt == "JPEG":
        return {"format": "JPEG", "quality": 90, "subsampling": 0, "optimize": True}
    if fmt == "PNG":
        return {"format": "PNG", "compress_level": 6, "optimize": True}
    if fmt in ("TIFF", "TIF"):
        return {"format": "TIFF", "compression": "tiff_deflate"}
    return {"format": fmt}


def build_save_config(image_path: Path, img: Image.Image) -> SaveConfig:
    """Determine output format and parameter factory based on the source image or override."""
    ext = image_path.suffix.lower()
    override = OUTPUT_FORMAT_OVERRIDE.lower() if OUTPUT_FORMAT_OVERRIDE else "auto"
    
    # Determine output format
    if override != "auto":
        img_format = override.upper()
    else:
        img_format = (img.format or ext.lstrip(".") or "png").upper()
    
    # Normalize format name
    if img_format == "JPG":
        img_format = "JPEG"
    
    # Determine file extension
    format_to_ext = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "TIFF": ".tif",
    }
    
    if override != "auto":
        ext = format_to_ext.get(img_format, f".{img_format.lower()}")
    elif not ext:
        ext = f".{img_format.lower()}"

    # Select appropriate parameter factory
    if img_format == "JPEG":
        params_factory = lambda: get_jpeg_save_params(img)
    elif img_format == "PNG":
        params_factory = get_png_save_params(img)
    elif img_format in ("TIFF", "TIF"):
        params_factory = get_tiff_save_params(img)
    else:
        # Fallback to PNG
        params_factory = get_png_save_params(img)
        img_format = "PNG"
        ext = ".png"

    return SaveConfig(format=img_format, extension=ext, params_factory=params_factory)


def save_image_with_config(image: Image.Image, path: Path, config: SaveConfig):
    """Save an image using the provided configuration, with fallback on failure."""
    params = config.params_factory()
    fmt = params.pop("format", config.format)
    
    # Override format if specified
    if OUTPUT_FORCE_FORMAT:
        fmt = OUTPUT_FORCE_FORMAT
    
    try:
        image.save(path, format=fmt, **params)
    except Exception:
        fallback = get_fallback_params(fmt)
        fmt = fallback.pop("format", fmt)
        image.save(path, format=fmt, **fallback)


def estimate_encoded_size(img: Image.Image, save_config: SaveConfig, sample_crop_size: int = 2048) -> float:
    """
    Estimate encoded size by saving a sample crop using the configured output format.
    """
    width, height = img.size
    crop_size = min(sample_crop_size, width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    sample = img.crop((left, top, left + crop_size, top + crop_size))

    buffer = io.BytesIO()
    params = save_config.params_factory()
    fmt = params.pop("format", save_config.format)
    try:
        sample.save(buffer, format=fmt, **params)
    except Exception:
        fallback = get_fallback_params(fmt)
        fmt = fallback.pop("format", fmt)
        sample.save(buffer, format=fmt, **fallback)

    sample_bytes = buffer.tell()
    sample_pixels = crop_size * crop_size
    total_pixels = width * height
    estimated_bytes = (sample_bytes / sample_pixels) * total_pixels
    return estimated_bytes / (1024 * 1024)


def generate_axis_segments(length: int) -> List[Tuple[int, int]]:
    """Generate tile start positions and sizes for one axis."""
    segments: List[Tuple[int, int]] = []
    if length <= 0:
        return segments

    pos = 0
    while pos + PRIMARY_TILE_SIZE <= length:
        segments.append((pos, PRIMARY_TILE_SIZE))
        pos += PRIMARY_TILE_SIZE

    remainder = length - pos
    if remainder > 0:
        n = max(1, int(np.ceil(remainder / 1024.0)))
        fallback_size = min(PRIMARY_TILE_SIZE, n * 1024)
        fallback_size = min(fallback_size, length)
        start = max(0, length - fallback_size)
        if segments and start <= segments[-1][0]:
            start = segments[-1][0] + PRIMARY_TILE_SIZE - fallback_size
            start = max(0, start)
        if start + fallback_size > length:
            start = max(0, length - fallback_size)
        segments.append((start, fallback_size))

    segments = sorted(set(segments), key=lambda s: s[0])
    return segments


def process_image(image_path: Path, output_dir: Path):
    """Process a single large image."""
    print(f"\n{'='*70}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*70}")
    
    # Load image with progress indication
    print(f"  Loading image...", end='', flush=True)
    img = None
    try:
        img = Image.open(image_path)
        
        # For multi-layer TIFFs, ensure we're using the largest (first) layer
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            print(f" (multi-layer TIFF: {img.n_frames} layers, using layer 0)...", end='', flush=True)
            img.seek(0)  # Explicitly seek to first frame (largest resolution)
        
        # Force load to catch decompression issues early
        img.load()
        print(" ✓")
    except Exception as pil_error:
        # Fallback to tifffile for TIFFs that PIL can't handle
        if image_path.suffix.lower() in ('.tif', '.tiff'):
            try:
                print(f" (PIL failed, trying tifffile)...", end='', flush=True)
                with tifffile.TiffFile(image_path) as tif:
                    arr = tif.pages[0].asarray()
                img = Image.fromarray(arr)
                print(" ✓")
            except Exception as tiff_error:
                print(f" ✗\n  ❌ Failed to load TIFF: {tiff_error}")
                return
        else:
            print(f" ✗\n  ❌ Failed to load image: {pil_error}")
            return
    
    width, height = img.size
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    save_config = build_save_config(image_path, img)
    
    print(f"  Dimensions: {width}×{height} px")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Mode: {img.mode}, Format: {img.format}")
    
    x_segments = generate_axis_segments(width)
    y_segments = generate_axis_segments(height)
    
    print(f"  Horizontal segments: {len(x_segments)} (primary size {PRIMARY_TILE_SIZE}, fallback {SECONDARY_TILE_SIZE})")
    print(f"  Vertical segments: {len(y_segments)} (primary size {PRIMARY_TILE_SIZE}, fallback {SECONDARY_TILE_SIZE})")
    
    if DRY_RUN:
        print(f"  🏃 DRY RUN - not saving")
        return
    
    saved_count = 0
    tile_number = 1  # Sequential numbering starting from 1
    enhanced_dir = output_dir / "enhanced"
    if SAVE_ENHANCED:
        enhanced_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem
    total_tiles = len(x_segments) * len(y_segments)
    
    print(f"  Processing {total_tiles} tiles...")
    
    for y_idx, (y_start, y_tile) in enumerate(y_segments):
        for x_idx, (x_start, x_tile) in enumerate(x_segments):
            print(f"    Tile {tile_number}/{total_tiles} [row {y_idx+1}/{len(y_segments)}, col {x_idx+1}/{len(x_segments)}]...", end='', flush=True)
            
            left = x_start
            upper = y_start
            right = min(left + x_tile, width)
            lower = min(upper + y_tile, height)
            
            if right <= left or lower <= upper:
                continue
            
            # Crop tile from original image
            piece_img = img.crop((left, upper, right, lower))
            
            # Convert to numpy array for processing
            arr = np.array(piece_img)
            
            # Step 1: Convert bit depth if requested
            arr = convert_bit_depth(arr, OUTPUT_BIT_DEPTH)
            
            # Step 2: Invert if requested (after bit depth conversion)
            if INVERT_OUTPUT:
                arr = invert_image(arr)
            
            # Convert back to PIL Image
            piece_img = Image.fromarray(arr)
            tile_w, tile_h = piece_img.size
            
            piece_name = f"{base_name}_{tile_number:03d}_x{left}_y{upper}_w{tile_w}_h{tile_h}{save_config.extension}"
            piece_path = output_dir / piece_name
            
            if SAVE_TILES:
                estimated_mb = estimate_encoded_size(piece_img, save_config, sample_crop_size=min(2048, tile_w, tile_h))
                if estimated_mb > MAX_FILE_SIZE_MB:
                    print(f" ⚠️  Estimated {estimated_mb:.2f} MB exceeds limit ({MAX_FILE_SIZE_MB:.2f} MB)")
                save_image_with_config(piece_img, piece_path, save_config)
                actual_mb = piece_path.stat().st_size / (1024 * 1024)
                print(f" ✓ {tile_w}×{tile_h} px -> {actual_mb:.2f} MB")
                saved_count += 1
                OUTPUT_FILENAMES.append(piece_name)
                
                if SAVE_ENHANCED:
                    enhanced_arr = np.array(piece_img)
                    if ENHANCEMENT_METHOD == "zscore":
                        enhanced_arr = enhance_zscore(enhanced_arr)
                    elif ENHANCEMENT_METHOD == "percentile":
                        enhanced_arr = enhance_percentile(enhanced_arr)
                    else:
                        enhanced_arr = enhance_clahe(enhanced_arr)
                    enhanced_img = Image.fromarray(enhanced_arr)
                    enhanced_name = f"{base_name}_{tile_number:03d}_x{left}_y{upper}_w{tile_w}_h{tile_h}_{ENHANCEMENT_METHOD}{save_config.extension}"
                    enhanced_path = enhanced_dir / enhanced_name
                    save_image_with_config(enhanced_img, enhanced_path, save_config)
                
                tile_number += 1  # Increment sequential counter
    
    print(f"\n  ✅ Saved {saved_count} tile(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut large pseudocolored images into stitch-aligned tiles."
    )
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Directory containing source images (non-recursive)")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory where cropped tiles will be saved")
    parser.add_argument("--max-file-size-mb", type=float, default=DEFAULT_MAX_FILE_SIZE_MB,
                        help="Maximum allowed tile size in MB before retiling (default: %(default)s)")
    parser.add_argument("--max-dimension-px", type=int, default=DEFAULT_MAX_DIMENSION_PX,
                        help="Maximum allowed dimension in pixels before retiling (default: %(default)s)")
    parser.add_argument("--min-dimension-px", type=int, default=DEFAULT_MIN_DIMENSION_PX,
                        help="Process only images with at least one side larger than this (default: %(default)s)")
    parser.add_argument("--extensions", type=str,
                        default=",".join(ext.lstrip('.') for ext in DEFAULT_IMAGE_EXTENSIONS),
                        help="Comma-separated list of file extensions to process (default: %(default)s)")
    parser.add_argument("--output-format", type=str, default="auto",
                        choices=["auto", "jpeg", "png", "tiff"],
                        help="Override output format (auto keeps original format)")
    parser.add_argument("--bit-depth", type=str, default="auto",
                        choices=["auto", "8", "16", "32f"],
                        help="Output bit depth: auto (preserve original), 8 (8-bit), 16 (16-bit), 32f (32-bit float)")
    parser.add_argument("--invert", required=True, type=lambda x: x.lower() in ['true', 'yes', '1'],
                        metavar='TRUE|FALSE',
                        help="Invert image intensities before saving tiles (required: true or false)")
    parser.add_argument("--save-enhanced", action="store_true",
                        help="Save an additional enhanced copy for each tile")
    parser.add_argument("--enhancement-method", type=str, default="clahe",
                        choices=["zscore", "percentile", "clahe"],
                        help="Enhancement method for the additional copy (default: clahe)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip processing WSIs that already have tiles in the output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze images without writing crops")
    return parser.parse_args()


def gather_images(input_dir: Path) -> List[Path]:
    images = []
    for item in input_dir.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            if "_grid_" in item.stem:
                continue
            images.append(item)
    return images


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"❌ Folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    global MAX_FILE_SIZE_MB, MAX_DIMENSION_PX, MIN_DIMENSION_PX, DRY_RUN, IMAGE_EXTENSIONS, OUTPUT_FORMAT_OVERRIDE, OUTPUT_BIT_DEPTH, OUTPUT_FORCE_FORMAT, INVERT_OUTPUT, SAVE_ENHANCED, ENHANCEMENT_METHOD, OUTPUT_FILENAMES, SKIP_EXISTING
    OUTPUT_FILENAMES = []  # Reset for each run
    MAX_FILE_SIZE_MB = args.max_file_size_mb
    MAX_DIMENSION_PX = args.max_dimension_px
    MIN_DIMENSION_PX = args.min_dimension_px
    DRY_RUN = args.dry_run
    SKIP_EXISTING = args.skip_existing
    OUTPUT_FORMAT_OVERRIDE = args.output_format
    OUTPUT_BIT_DEPTH = args.bit_depth
    override_lower = OUTPUT_FORMAT_OVERRIDE.lower()
    OUTPUT_FORCE_FORMAT = (
        "PNG" if override_lower == "png" else
        "TIFF" if override_lower == "tiff" else
        "JPEG" if override_lower == "jpeg" else
        None
    )
    INVERT_OUTPUT = args.invert
    SAVE_ENHANCED = args.save_enhanced
    ENHANCEMENT_METHOD = args.enhancement_method

    extensions = []
    for ext in args.extensions.split(','):
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = f'.{ext}'
        extensions.append(ext.lower())
    IMAGE_EXTENSIONS = tuple(extensions) if extensions else DEFAULT_IMAGE_EXTENSIONS

    print("=" * 70)
    print("Large Image Tile Cutter")
    print("=" * 70)
    print(f"Folder: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Limits: ≤{MAX_FILE_SIZE_MB}MB, ≤{MAX_DIMENSION_PX}px")
    print(f"Tile config: primary {PRIMARY_TILE_SIZE}px, fallback {SECONDARY_TILE_SIZE}px")
    print(f"Extensions: {', '.join(IMAGE_EXTENSIONS)}")
    print(f"Output format: {OUTPUT_FORMAT_OVERRIDE}")
    print(f"Bit depth: {OUTPUT_BIT_DEPTH}")
    print(f"Invert: {INVERT_OUTPUT}")
    print(f"Skip existing: {SKIP_EXISTING}")
    print(f"Dry run: {DRY_RUN}")

    images = gather_images(input_dir)
    if not images:
        print(f"\n⚠️  No matching image files found")
        return

    print(f"\n🔍 Found {len(images)} image(s)")

    large_images = []
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            size_mb = img_path.stat().st_size / (1024 * 1024)

            if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX or size_mb > MAX_FILE_SIZE_MB:
                large_images.append(img_path)
        except Exception as pil_error:
            # Try tifffile for TIFFs that PIL can't open
            if img_path.suffix.lower() in ('.tif', '.tiff'):
                try:
                    with tifffile.TiffFile(img_path) as tif:
                        shape = tif.pages[0].shape
                        h, w = shape[0], shape[1]
                    size_mb = img_path.stat().st_size / (1024 * 1024)
                    
                    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX or size_mb > MAX_FILE_SIZE_MB:
                        large_images.append(img_path)
                except Exception as tiff_error:
                    print(f"⚠️  Could not check {img_path.name}: {tiff_error}")
            else:
                print(f"⚠️  Could not check {img_path.name}: {pil_error}")

    if not large_images:
        print(f"\n✅ No images exceed limits - nothing to process!")
        return

    # Filter out images with existing tiles if skip-existing is enabled
    if SKIP_EXISTING:
        filtered_images = []
        skipped_count = 0
        for img_path in large_images:
            base_name = img_path.stem
            # MS script uses pattern: {base}_NNN_x*_y*_w*_h*.{ext}
            # Check if any tiles exist for this base name
            existing_tiles = list(output_dir.glob(f"{base_name}_*_x*_y*_w*_h*.*"))
            if existing_tiles:
                skipped_count += 1
                if not DRY_RUN:
                    print(f"⏭️  Skipping {img_path.name} ({len(existing_tiles)} tiles already exist)")
            else:
                filtered_images.append(img_path)
        
        if skipped_count > 0:
            print(f"\n⏭️  Skipped {skipped_count} image(s) with existing tiles")
        
        large_images = filtered_images
        
        if not large_images:
            print(f"\n✅ All images already processed!")
            return

    print(f"📏 {len(large_images)} image(s) exceed limits and will be processed")

    for img_path in large_images:
        try:
            process_image(img_path, output_dir)
        except Exception as e:
            print(f"\n❌ FAILED {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Export CSV with output filenames
    if OUTPUT_FILENAMES and not DRY_RUN:
        csv_path = output_dir / "output_tiles.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'column2', 'include'])  # Header
            for filename in OUTPUT_FILENAMES:
                writer.writerow([filename, '', 'yes'])
        print(f"\n📄 Saved tile list to: {csv_path}")
        print(f"   Total tiles: {len(OUTPUT_FILENAMES)}")
    
    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
