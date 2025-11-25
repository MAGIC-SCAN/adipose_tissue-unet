#!/usr/bin/env python3
"""
Legacy tiling script (grid-based) for splitting WSIs along stitch boundaries.

This version preserves the original behavior:
  • Uses 2048 px tiles with 204 px overlap (stride = 1844 px).
  • Finds the largest grid (5×5 down to 2×2) that meets dimension/file-size limits.
  • Saves tiles next to the source image using the same format by default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import argparse
import io

import numpy as np
import tifffile
from PIL import Image, PngImagePlugin, TiffImagePlugin

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MAX_FILE_SIZE_MB = 15
DEFAULT_MAX_DIMENSION_PX = 13112
DEFAULT_TILE_SIZE = 2048
DEFAULT_OVERLAP = 204
DEFAULT_STRIDE = DEFAULT_TILE_SIZE - DEFAULT_OVERLAP
DEFAULT_PREFERRED_GRIDS = [5, 4, 3, 2]
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

MAX_FILE_SIZE_MB = DEFAULT_MAX_FILE_SIZE_MB
MAX_DIMENSION_PX = DEFAULT_MAX_DIMENSION_PX
MIN_DIMENSION_PX = DEFAULT_MAX_DIMENSION_PX
TILE_SIZE = DEFAULT_TILE_SIZE
STRIDE = DEFAULT_STRIDE
OVERLAP = DEFAULT_OVERLAP
PREFERRED_GRIDS = DEFAULT_PREFERRED_GRIDS
IMAGE_EXTENSIONS = DEFAULT_IMAGE_EXTENSIONS
OUTPUT_FORMAT_OVERRIDE = None
DRY_RUN = False
INVERT_IMAGE = False
SAVE_TILES = True


@dataclass
class SaveConfig:
    format: str
    extension: str
    params_factory: Callable[[], dict]


def get_jpeg_save_params(img: Image.Image) -> dict:
    params = {
        "format": "JPEG",
        "quality": 90,
        "subsampling": 0,
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
    fmt = fmt.upper()
    if fmt == "JPEG":
        return {"format": "JPEG", "quality": 90, "subsampling": 0, "optimize": True}
    if fmt == "PNG":
        return {"format": "PNG", "compress_level": 6, "optimize": True}
    if fmt in ("TIFF", "TIF"):
        return {"format": "TIFF", "compression": "tiff_deflate"}
    return {"format": fmt}


def build_save_config(image_path: Path, img: Image.Image) -> SaveConfig:
    ext = image_path.suffix.lower()
    if OUTPUT_FORMAT_OVERRIDE and OUTPUT_FORMAT_OVERRIDE.lower() != "auto":
        img_format = OUTPUT_FORMAT_OVERRIDE.upper()
        format_to_ext = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif"}
        ext = format_to_ext.get(img_format, f".{img_format.lower()}")
    else:
        img_format = (img.format or ext.lstrip(".") or "png").upper()
        if img_format == "JPG":
            img_format = "JPEG"
        if not ext:
            ext = f".{img_format.lower()}"

    if img_format == "JPEG":
        params_factory = lambda: get_jpeg_save_params(img)
    elif img_format == "PNG":
        params_factory = get_png_save_params(img)
    elif img_format in ("TIFF", "TIF"):
        params_factory = get_tiff_save_params(img)
    else:
        params_factory = get_png_save_params(img)
        img_format = "PNG"
        ext = ".png"

    return SaveConfig(format=img_format, extension=ext, params_factory=params_factory)


def convert_to_jpeg_compatible(image: Image.Image) -> Image.Image:
    """Convert image to 8-bit RGB for JPEG compatibility."""
    # Handle 16-bit images
    if image.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        # Normalize to 0-255 range
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)
        # Convert to RGB
        return Image.fromarray(img_array).convert('RGB')
    # Handle grayscale
    elif image.mode in ('L', 'LA'):
        return image.convert('RGB')
    # Handle palette mode
    elif image.mode == 'P':
        return image.convert('RGB')
    # Already RGB or RGBA
    elif image.mode in ('RGB', 'RGBA'):
        return image.convert('RGB')
    else:
        return image.convert('RGB')


def save_image_with_config(image: Image.Image, path: Path, config: SaveConfig):
    params = config.params_factory()
    fmt = params.pop("format", config.format)
    
    # Convert to JPEG-compatible format if needed
    if fmt.upper() == 'JPEG' and image.mode not in ('RGB', 'L'):
        image = convert_to_jpeg_compatible(image)
    
    try:
        image.save(path, format=fmt, **params)
    except Exception:
        fallback = get_fallback_params(fmt)
        fmt = fallback.pop("format", fmt)
        image.save(path, format=fmt, **fallback)


def estimate_encoded_size(img: Image.Image, save_config: SaveConfig, sample_crop_size: int = 2048) -> float:
    width, height = img.size
    crop_size = min(sample_crop_size, width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    sample = img.crop((left, top, left + crop_size, top + crop_size))

    buffer = io.BytesIO()
    params = save_config.params_factory()
    fmt = params.pop("format", save_config.format)
    
    # Convert to JPEG-compatible format if needed
    if fmt.upper() == 'JPEG' and sample.mode not in ('RGB', 'L'):
        sample = convert_to_jpeg_compatible(sample)
    
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


def calculate_grid_dimensions(image_width: int, image_height: int) -> Tuple[int, int]:
    cols = 1 + max(0, int(np.ceil((image_width - TILE_SIZE) / STRIDE)))
    rows = 1 + max(0, int(np.ceil((image_height - TILE_SIZE) / STRIDE)))
    return cols, rows


def calculate_piece_size(grid_size: int) -> Tuple[int, int]:
    dimension = TILE_SIZE + (grid_size - 1) * STRIDE
    return dimension, dimension


def find_optimal_grid(total_cols: int, total_rows: int, img: Image.Image, save_config: SaveConfig) -> Optional[int]:
    # If forced grid size is specified, validate and use it
    if FORCE_GRID_SIZE is not None:
        if FORCE_GRID_SIZE > total_cols or FORCE_GRID_SIZE > total_rows:
            print(f"  ⚠️  WARNING: Forced grid {FORCE_GRID_SIZE}x{FORCE_GRID_SIZE} exceeds image dimensions")
            print(f"             (total_cols={total_cols}, total_rows={total_rows})")
            print(f"             Falling back to automatic grid selection")
        else:
            piece_w, piece_h = calculate_piece_size(FORCE_GRID_SIZE)
            # When force-grid-size is set, override all limits (file size, dimensions)
            if piece_w > MAX_DIMENSION_PX or piece_h > MAX_DIMENSION_PX:
                print(f"  ⚠️  WARNING: Forced grid produces oversized pieces ({piece_w}x{piece_h} px)")
                print(f"             Overriding dimension limit ({MAX_DIMENSION_PX}px) - forced grid enabled")
            
            # Test file size but don't block - just warn
            test_crop = img.crop((0, 0, min(piece_w, img.width), min(piece_h, img.height)))
            test_size_mb = estimate_encoded_size(test_crop, save_config, sample_crop_size=1024)
            if test_size_mb > MAX_FILE_SIZE_MB:
                print(f"  ⚠️  WARNING: Forced grid produces oversized files (~{test_size_mb:.1f} MB)")
                print(f"             Overriding file size limit ({MAX_FILE_SIZE_MB}MB) - forced grid enabled")
            
            print(f"  ℹ️  Using forced grid size: {FORCE_GRID_SIZE}x{FORCE_GRID_SIZE}")
            return FORCE_GRID_SIZE
    
    # Automatic grid selection
    for grid_size in PREFERRED_GRIDS:
        if grid_size > total_cols or grid_size > total_rows:
            continue

        piece_w, piece_h = calculate_piece_size(grid_size)
        if piece_w > MAX_DIMENSION_PX or piece_h > MAX_DIMENSION_PX:
            continue

        test_crop = img.crop((0, 0, min(piece_w, img.width), min(piece_h, img.height)))
        test_size_mb = estimate_encoded_size(test_crop, save_config, sample_crop_size=1024)
        if test_size_mb <= MAX_FILE_SIZE_MB:
            return grid_size

    return None


def extract_tile_piece(img: Image.Image, start_col: int, start_row: int,
                       grid_size: int, total_cols: int, total_rows: int
                       ) -> Tuple[Image.Image, bool, int, int]:
    x_start = start_col * STRIDE
    y_start = start_row * STRIDE

    actual_cols = min(grid_size, total_cols - start_col)
    actual_rows = min(grid_size, total_rows - start_row)

    piece_w = TILE_SIZE + (actual_cols - 1) * STRIDE
    piece_h = TILE_SIZE + (actual_rows - 1) * STRIDE

    x_end = min(x_start + piece_w, img.width)
    y_end = min(y_start + piece_h, img.height)

    cropped = img.crop((x_start, y_start, x_end, y_end))

    is_partial = (actual_cols < grid_size) or (actual_rows < grid_size) or \
                 (cropped.width < piece_w) or (cropped.height < piece_h)

    return cropped, is_partial, actual_cols, actual_rows


def load_image(image_path: Path) -> Image.Image:
    """Load image using tifffile for TIFF files, PIL for others."""
    ext = image_path.suffix.lower()
    if ext in ('.tif', '.tiff'):
        # Use tifffile to load TIFF
        img_array = tifffile.imread(str(image_path))
        
        # Handle 16-bit images - normalize to 8-bit
        if img_array.dtype in (np.uint16, np.int16):
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = ((img_array.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)
        
        # Invert if requested
        if INVERT_IMAGE:
            img_array = 255 - img_array
        
        # Convert to PIL Image
        if img_array.ndim == 2:
            return Image.fromarray(img_array, mode='L')
        elif img_array.ndim == 3:
            if img_array.shape[2] == 3:
                return Image.fromarray(img_array, mode='RGB')
            elif img_array.shape[2] == 4:
                return Image.fromarray(img_array, mode='RGBA')
            else:
                # Take first channel if multi-channel
                return Image.fromarray(img_array[:, :, 0], mode='L')
        else:
            raise ValueError(f"Unexpected array shape: {img_array.shape}")
    else:
        # Use PIL for non-TIFF files
        img = Image.open(image_path)
        if INVERT_IMAGE:
            img = Image.eval(img, lambda x: 255 - x)
        return img


def process_image(image_path: Path, output_dir: Path):
    print(f"\n{'='*70}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*70}")

    img = load_image(image_path)
    width, height = img.size
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    save_config = build_save_config(image_path, img)

    print(f"  Dimensions: {width}×{height} px")
    print(f"  File size: {file_size_mb:.2f} MB")

    if width <= MIN_DIMENSION_PX and height <= MIN_DIMENSION_PX and file_size_mb <= MAX_FILE_SIZE_MB:
        print(f"  ✓ Image already within limits, skipping")
        return

    total_cols, total_rows = calculate_grid_dimensions(width, height)
    print(f"  Original grid: {total_cols}×{total_rows} tiles (stride={STRIDE}px, overlap={OVERLAP}px)")

    print(f"  Finding optimal grid size...")
    grid_size = find_optimal_grid(total_cols, total_rows, img, save_config)

    if grid_size is None:
        print(f"  ❌ ERROR: Cannot fit image within limits even with 2×2 grid!")
        return

    piece_w, piece_h = calculate_piece_size(grid_size)
    print(f"  ✓ Using {grid_size}×{grid_size} tile grid per piece ({piece_w}×{piece_h} px)")

    pieces_cols = int(np.ceil(total_cols / grid_size))
    pieces_rows = int(np.ceil(total_rows / grid_size))
    total_pieces = pieces_cols * pieces_rows
    print(f"  Will create: {pieces_cols}×{pieces_rows} = {total_pieces} piece(s)")

    if DRY_RUN:
        print(f"  🏃 DRY RUN - not saving")
        return

    saved_count = 0
    partial_count = 0
    base_name = image_path.stem

    for row_idx in range(pieces_rows):
        for col_idx in range(pieces_cols):
            start_col = col_idx * grid_size
            start_row = row_idx * grid_size

            piece_img, is_partial, actual_cols, actual_rows = extract_tile_piece(
                img, start_col, start_row, grid_size, total_cols, total_rows
            )

            piece_name = f"{base_name}_grid_{grid_size}x{grid_size}_r{row_idx}_c{col_idx}{save_config.extension}"
            piece_path = output_dir / piece_name

            if SAVE_TILES:
                save_image_with_config(piece_img, piece_path, save_config)

                piece_size_mb = piece_path.stat().st_size / (1024 * 1024)

                status = "PARTIAL" if is_partial else "full"
                print(f"    [{row_idx},{col_idx}] {status}: {piece_img.width}×{piece_img.height} px, "
                      f"{piece_size_mb:.2f} MB", end="")

                if is_partial:
                    expected_w, expected_h = calculate_piece_size(grid_size)
                    partial_w = expected_w - piece_img.width
                    partial_h = expected_h - piece_img.height
                    print(f" (short by {partial_w}×{partial_h} px, "
                          f"covers {actual_cols}×{actual_rows} tiles)")
                    partial_count += 1
                else:
                    print()

                saved_count += 1

    print(f"\n  ✅ Saved {saved_count} piece(s)")
    if partial_count > 0:
        print(f"  ⚠️  {partial_count} partial piece(s) at edges")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy grid-based cutter for large WSIs."
    )
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Directory containing source images (non-recursive)")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="Directory where cropped tiles will be saved")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help="Tile size in pixels (default: %(default)s)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help="Overlap between adjacent tiles (default: %(default)s)")
    parser.add_argument("--max-file-size-mb", type=float, default=DEFAULT_MAX_FILE_SIZE_MB,
                        help="Maximum allowed tile size in MB (default: %(default)s)")
    parser.add_argument("--max-dimension-px", type=int, default=DEFAULT_MAX_DIMENSION_PX,
                        help="Maximum allowed dimension in pixels (default: %(default)s)")
    parser.add_argument("--min-dimension-px", type=int, default=DEFAULT_MAX_DIMENSION_PX,
                        help="Skip images if both dimensions <= this value (default: %(default)s)")
    parser.add_argument("--extensions", type=str,
                        default=",".join(ext.lstrip('.') for ext in DEFAULT_IMAGE_EXTENSIONS),
                        help="Comma-separated list of file extensions to process (default: %(default)s)")
    parser.add_argument("--output-format", type=str, default="auto",
                        choices=["auto", "jpeg", "png", "tiff"],
                        help="Override output format (auto keeps original)")
    parser.add_argument("--invert", required=True, 
                        type=lambda x: x.lower() in ['true', 'yes', '1'],
                        help="Invert image intensities (required: true/yes/1 or false/no/0)")
    parser.add_argument("--force-grid-size", type=int, default=None,
                        choices=[2, 3, 4, 5],
                        help="Force specific grid size (2-5) instead of automatic optimization. "
                             "Useful for matching grid sizes across different file formats.")
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

    if args.tile_size <= 0:
        raise SystemExit("Tile size must be positive")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise SystemExit("Overlap must be between 0 and tile_size-1")

    global TILE_SIZE, STRIDE, OVERLAP, PREFERRED_GRIDS
    global MAX_FILE_SIZE_MB, MAX_DIMENSION_PX, MIN_DIMENSION_PX, DRY_RUN, IMAGE_EXTENSIONS, OUTPUT_FORMAT_OVERRIDE, INVERT_IMAGE, SAVE_TILES, FORCE_GRID_SIZE

    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap
    STRIDE = TILE_SIZE - OVERLAP
    if STRIDE <= 0:
        raise SystemExit("Resulting stride must be positive")

    MAX_FILE_SIZE_MB = args.max_file_size_mb
    MAX_DIMENSION_PX = args.max_dimension_px
    MIN_DIMENSION_PX = args.min_dimension_px
    DRY_RUN = args.dry_run
    OUTPUT_FORMAT_OVERRIDE = args.output_format
    INVERT_IMAGE = args.invert
    SAVE_TILES = not args.dry_run
    FORCE_GRID_SIZE = args.force_grid_size

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
    print("Legacy Large Image Tile Cutter")
    print("=" * 70)
    print(f"Folder: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {TILE_SIZE}px, overlap: {OVERLAP}px, stride: {STRIDE}px")
    print(f"Limits: ≤{MAX_FILE_SIZE_MB}MB, ≤{MAX_DIMENSION_PX}px")
    print(f"Extensions: {', '.join(IMAGE_EXTENSIONS)}")
    print(f"Output format: {OUTPUT_FORMAT_OVERRIDE}")
    print(f"Invert: {INVERT_IMAGE}")
    print(f"Dry run: {DRY_RUN}")

    images = gather_images(input_dir)
    if not images:
        print(f"\n⚠️  No matching image files found")
        return

    print(f"\n🔍 Found {len(images)} image(s)")

    large_images = []
    for img_path in images:
        try:
            img = load_image(img_path)
            w, h = img.size
            img.close()
            size_mb = img_path.stat().st_size / (1024 * 1024)

            if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX or size_mb > MAX_FILE_SIZE_MB:
                large_images.append(img_path)
        except Exception as e:
            print(f"⚠️  Could not check {img_path.name}: {e}")

    if not large_images:
        print(f"\n✅ No images exceed limits - nothing to process!")
        return

    print(f"📏 {len(large_images)} image(s) exceed limits and will be processed")

    for img_path in large_images:
        try:
            process_image(img_path, output_dir)
        except Exception as e:
            print(f"\n❌ FAILED {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
