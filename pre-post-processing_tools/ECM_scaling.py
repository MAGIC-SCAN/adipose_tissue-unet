#!/usr/bin/env python3
"""
ECM Image Scaling Script

Resamples ECM channel images to match the exact dimensions of corresponding
Pseudocolored reference images. This ensures pixel-perfect alignment for
mask transfer between modalities.

Usage:
    python ECM_scaling.py \
        --reference-dir /path/to/Pseudocolored \
        --target-dir /path/to/ECM_channel \
        --output-dir /path/to/ECM_channel/resampled
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    ext = image_path.suffix.lower()
    
    if ext in ('.tif', '.tiff'):
        img_array = tifffile.imread(str(image_path))
        # tifffile returns (height, width) or (height, width, channels)
        if img_array.ndim == 2:
            h, w = img_array.shape
        else:
            h, w = img_array.shape[:2]
        return (w, h)
    else:
        # Use PIL for JPEG/PNG
        with Image.open(image_path) as img:
            return img.size  # (width, height)


def build_reference_dict(reference_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    Build dictionary mapping base filenames to their dimensions.
    
    Returns:
        Dict[basename, (width, height)]
    """
    ref_sizes = {}
    
    for img_path in reference_dir.iterdir():
        if not img_path.is_file():
            continue
        
        ext = img_path.suffix.lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            continue
        
        # Get base name without extension
        base_name = img_path.stem
        
        try:
            width, height = get_image_size(img_path)
            ref_sizes[base_name] = (width, height)
            print(f"Reference: {base_name} → {width}×{height}")
        except Exception as e:
            print(f"⚠️  Failed to read {img_path.name}: {e}")
    
    return ref_sizes


def resample_image(input_path: Path, target_size: Tuple[int, int], output_path: Path):
    """
    Resample image to target size using LANCZOS interpolation.
    
    Args:
        input_path: Path to input image
        target_size: (width, height) tuple
        output_path: Path to save resampled image
    """
    ext = input_path.suffix.lower()
    
    if ext in ('.tif', '.tiff'):
        # Load TIFF with tifffile
        img_array = tifffile.imread(str(input_path))
        
        # Convert to PIL Image for resampling
        if img_array.ndim == 2:
            # Grayscale
            if img_array.dtype in (np.uint16, np.int16):
                # Normalize 16-bit to 8-bit
                img_min, img_max = img_array.min(), img_array.max()
                if img_max > img_min:
                    img_array = ((img_array.astype(np.float32) - img_min) / 
                                (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)
            
            img = Image.fromarray(img_array, mode='L')
        elif img_array.ndim == 3:
            # Multi-channel - take first channel or handle as RGB
            if img_array.shape[2] == 1:
                img_array = img_array[:, :, 0]
                if img_array.dtype in (np.uint16, np.int16):
                    img_min, img_max = img_array.min(), img_array.max()
                    if img_max > img_min:
                        img_array = ((img_array.astype(np.float32) - img_min) / 
                                    (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_array = np.zeros_like(img_array, dtype=np.uint8)
                img = Image.fromarray(img_array, mode='L')
            else:
                # Assume RGB/RGBA
                if img_array.dtype in (np.uint16, np.int16):
                    img_min, img_max = img_array.min(), img_array.max()
                    if img_max > img_min:
                        img_array = ((img_array.astype(np.float32) - img_min) / 
                                    (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_array = np.zeros_like(img_array, dtype=np.uint8)
                
                if img_array.shape[2] == 3:
                    img = Image.fromarray(img_array, mode='RGB')
                elif img_array.shape[2] == 4:
                    img = Image.fromarray(img_array, mode='RGBA')
                else:
                    # Fallback to first channel
                    img = Image.fromarray(img_array[:, :, 0], mode='L')
        else:
            raise ValueError(f"Unexpected array shape: {img_array.shape}")
        
        # Resample
        img_resampled = img.resize(target_size, Image.LANCZOS)
        
        # Save as TIFF
        img_resampled.save(output_path, format='TIFF', compression='tiff_deflate')
        
    else:
        # Load with PIL
        with Image.open(input_path) as img:
            # Resample
            img_resampled = img.resize(target_size, Image.LANCZOS)
            
            # Save with same format
            if ext in ('.jpg', '.jpeg'):
                img_resampled.save(output_path, format='JPEG', quality=95, optimize=True)
            elif ext == '.png':
                img_resampled.save(output_path, format='PNG', optimize=True)
            else:
                img_resampled.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Resample ECM images to match Pseudocolored reference dimensions"
    )
    parser.add_argument("--reference-dir", required=True, type=str,
                        help="Directory containing Pseudocolored reference images")
    parser.add_argument("--target-dir", required=True, type=str,
                        help="Directory containing ECM images to resample")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: target-dir/resampled)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without writing files")
    
    args = parser.parse_args()
    
    reference_dir = Path(args.reference_dir)
    target_dir = Path(args.target_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = target_dir / "resampled"
    
    # Validate directories
    if not reference_dir.exists() or not reference_dir.is_dir():
        print(f"❌ Reference directory not found: {reference_dir}")
        sys.exit(1)
    
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"❌ Target directory not found: {target_dir}")
        sys.exit(1)
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ECM Image Scaling")
    print("=" * 70)
    print(f"Reference dir: {reference_dir}")
    print(f"Target dir:    {target_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Dry run:       {args.dry_run}")
    print()
    
    # Build reference size dictionary
    print("📏 Reading reference image sizes...")
    ref_sizes = build_reference_dict(reference_dir)
    
    if not ref_sizes:
        print("❌ No valid reference images found")
        sys.exit(1)
    
    print(f"\n✓ Found {len(ref_sizes)} reference images")
    
    # Process target images
    print("\n🔄 Processing target images...")
    processed = 0
    skipped = 0
    matched = 0
    
    for img_path in sorted(target_dir.iterdir()):
        if not img_path.is_file():
            continue
        
        ext = img_path.suffix.lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
            continue
        
        base_name = img_path.stem
        
        # Strip numeric suffixes like -002, -003, -005, etc.
        # Pattern: ends with dash followed by 3 digits
        import re
        base_name_stripped = re.sub(r'-\d{3}$', '', base_name)
        
        # Try exact match first, then try stripped version
        if base_name in ref_sizes:
            matched_name = base_name
        elif base_name_stripped in ref_sizes:
            matched_name = base_name_stripped
        else:
            matched_name = None
        
        # Check for match in reference
        if matched_name is None:
            print(f"⚠️  No reference match for: {base_name}")
            skipped += 1
            continue
        
        matched += 1
        target_size = ref_sizes[matched_name]
        
        try:
            current_size = get_image_size(img_path)
            output_path = output_dir / img_path.name
            
            if current_size == target_size:
                print(f"✓ {base_name}: Already correct size {current_size[0]}×{current_size[1]}")
                processed += 1
                continue
            
            print(f"🔄 {base_name}:")
            print(f"   Current: {current_size[0]}×{current_size[1]}")
            print(f"   Target:  {target_size[0]}×{target_size[1]}")
            print(f"   Delta:   {current_size[0]-target_size[0]:+d}×{current_size[1]-target_size[1]:+d}")
            
            if not args.dry_run:
                resample_image(img_path, target_size, output_path)
                print(f"   ✅ Saved to: {output_path.name}")
            else:
                print(f"   [DRY RUN] Would save to: {output_path.name}")
            
            processed += 1
            
        except Exception as e:
            print(f"❌ Failed to process {base_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Reference images:     {len(ref_sizes)}")
    print(f"Matched targets:      {matched}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (no match):   {skipped}")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No files were written")
    else:
        print(f"\n✅ Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
