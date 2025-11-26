#!/usr/bin/env python3
"""
Convert TIF files to 8-bit JPG format using the same methods as large_wsi_to_small_wsi_Lucy.py

Handles:
  - 16-bit → 8-bit conversion with normalization
  - Grayscale → RGB conversion
  - Metadata preservation (where possible)

USAGE EXAMPLES:

1. Convert directory of TIF files to JPG:
   python pre-post-processing_tools/convert_tif_to_jpg.py \
     --input-dir /home/luci/adipose_tissue-unet/data/raw_tif \
     --output-dir /home/luci/adipose_tissue-unet/data/converted_jpg

2. Convert with custom JPEG quality:
   python pre-post-processing_tools/convert_tif_to_jpg.py \
     --input-dir data/raw_tif \
     --output-dir data/converted_jpg \
     --quality 95

3. Preserve directory structure:
   python pre-post-processing_tools/convert_tif_to_jpg.py \
     --input-dir data/raw_tif \
     --output-dir data/converted_jpg \
     --recursive

4. Convert single file:
   python pre-post-processing_tools/convert_tif_to_jpg.py \
     --input data/raw_tif/image.tif \
     --output data/converted_jpg/image.jpg \
     --quality 100

CONVERSION STEPS:
  1. Load TIF (supports 8-bit, 16-bit, RGB, grayscale)
  2. Normalize 16-bit to 8-bit (min-max scaling)
  3. Convert grayscale/palette to RGB
  4. Save as JPEG with specified quality

OUTPUT:
  - 8-bit RGB JPEG files
  - Same base names as input files
  - Quality: 95 (default) or custom

NOTE: For batch conversion with enhancement, use large_wsi_to_small_wsi_MS.py
      with --output-format jpg --output-bit-depth 8.
"""

from pathlib import Path
import argparse
import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


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


def load_image(image_path: Path, invert: bool = False) -> Image.Image:
    """Load image using tifffile for TIFF files."""
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
    if invert:
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


def convert_tif_to_jpg(input_path: Path, output_path: Path, invert: bool = False, quality: int = 95):
    """Convert a single TIF file to 8-bit JPG."""
    print(f"Converting: {input_path.name}")
    
    # Load the image
    img = load_image(input_path, invert=invert)
    
    print(f"  Original size: {img.size[0]}×{img.size[1]} px")
    print(f"  Mode: {img.mode}")
    
    # Convert to JPEG-compatible format
    img_rgb = convert_to_jpeg_compatible(img)
    
    print(f"  Converted mode: {img_rgb.mode}")
    
    # Save as JPEG
    img_rgb.save(output_path, format='JPEG', quality=quality, optimize=True, subsampling=0)
    
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path.name} ({output_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF files to 8-bit JPG format"
    )
    parser.add_argument("--input-dir", required=True, type=str,
                        help="Directory containing TIF files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to same as input)")
    parser.add_argument("--invert", action="store_true",
                        help="Invert image intensities")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (1-100, default: 95)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be converted without actually converting")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        raise SystemExit(f"❌ Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIF files
    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    if not tif_files:
        print(f"⚠️  No TIF files found in {input_dir}")
        return
    
    print("=" * 70)
    print("TIF to JPG Converter")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"JPEG quality: {args.quality}")
    print(f"Invert: {args.invert}")
    print(f"Dry run: {args.dry_run}")
    print(f"\nFound {len(tif_files)} TIF file(s)\n")
    
    if args.dry_run:
        for tif_path in tif_files:
            jpg_path = output_dir / f"{tif_path.stem}.jpg"
            print(f"Would convert: {tif_path.name} -> {jpg_path.name}")
        return
    
    # Convert each file
    converted = 0
    failed = 0
    
    for tif_path in tif_files:
        jpg_path = output_dir / f"{tif_path.stem}.jpg"
        
        try:
            convert_tif_to_jpg(tif_path, jpg_path, invert=args.invert, quality=args.quality)
            converted += 1
        except Exception as e:
            print(f"❌ Failed to convert {tif_path.name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 70)
    print(f"✅ Conversion complete!")
    print(f"   Converted: {converted}")
    if failed > 0:
        print(f"   Failed: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
