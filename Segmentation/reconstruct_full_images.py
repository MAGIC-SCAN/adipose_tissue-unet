#!/usr/bin/env python3
"""
Full Image Reconstruction from Overlapping Tiles

Reconstructs full slide images from overlapping tile predictions (e.g., clean_test_50_overlap).
Supports all post-processing enhancements from full_evaluation_enhanced.py.

Key Features:
- Gaussian/linear blending for seamless reconstruction
- Missing tile detection and handling
- Optional TTA and boundary refinement
- Per-slide metrics computation
- Overlay and comparison visualizations

USAGE EXAMPLES:

1. Basic reconstruction (Gaussian blending):
   python Segmentation/reconstruct_full_images.py \
     --weights checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5 \
     --data-root /path/to/clean_test_50_overlap \
     --output-dir reconstructed_images

2. Reconstruction with TTA:
   python Segmentation/reconstruct_full_images.py \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --data-root /path/to/clean_test_50_overlap \
     --output-dir reconstructed_tta \
     --use-tta --tta-mode full

3. Reconstruction with boundary refinement:
   python Segmentation/reconstruct_full_images.py \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --data-root /path/to/clean_test_50_overlap \
     --output-dir reconstructed_refined \
     --boundary-refine

4. Full reconstruction pipeline:
   python Segmentation/reconstruct_full_images.py \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --data-root /path/to/clean_test_50_overlap \
     --output-dir reconstructed_full \
     --use-tta --tta-mode full \
     --boundary-refine \
     --blend-mode gaussian

5. Linear blending (faster, slightly lower quality):
   python Segmentation/reconstruct_full_images.py \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --data-root /path/to/clean_test_50_overlap \
     --output-dir reconstructed_linear \
     --blend-mode linear

Output Structure:
    output_dir/
    ├── masks/           # Full reconstructed prediction masks (TIFF)
    ├── overlays/        # Predictions overlaid on original images
    ├── comparisons/     # 3-panel GT vs Pred comparisons
    ├── metrics/         # Per-slide metrics (JSON)
    │   └── summary.csv  # Aggregated metrics
    └── reconstruction_log.json

Usage (Legacy Format):
    # Basic reconstruction
    python reconstruct_full_images.py \
        --weights checkpoints/best.h5 \
        --data-root /path/to/clean_test_50_overlap \
        --output-dir reconstructed_images

    # With all enhancements
    python reconstruct_full_images.py \
        --weights checkpoints/best.h5 \
        --data-root /path/to/clean_test_50_overlap \
        --output-dir reconstructed_images \
        --use-tta --tta-mode full \
        --blend-mode gaussian \
        --boundary-refine \
        --threshold 0.5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from datetime import datetime
import time
import warnings

import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
import pandas as pd

# Import post-processing classes from full_evaluation_enhanced
from full_evaluation_enhanced import (
    GaussianBlender, LinearBlender, BoundaryRefiner,
    TestTimeAugmentation, AdiposeUNet,
    load_training_stats, set_deterministic_seeds,
    calculate_pixel_metrics, safe_bootstrap_ci
)

# TensorFlow setup
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ====================================================================
# TILE PARSING AND GROUPING
# ====================================================================

def parse_tile_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse tile filename to extract slide ID and position.
    
    Examples:
        "6 BEEF Shoulder -1_grid_5x5_r1_c2_r0_c1.jpg"
        → slide_id = "6 BEEF Shoulder -1_grid_5x5_r1_c2"
        → row = 0, col = 1
        
        "slide_name_r5_c3.jpg"
        → slide_id = "slide_name"
        → row = 5, col = 3
    
    Args:
        filename: Tile filename (with or without extension)
    
    Returns:
        (slide_id, row, col)
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    # Last two parts should be r{row}_c{col}
    if len(parts) >= 2 and parts[-2].startswith('r') and parts[-1].startswith('c'):
        try:
            tile_row = int(parts[-2][1:])
            tile_col = int(parts[-1][1:])
            slide_id = "_".join(parts[:-2])
            return slide_id, tile_row, tile_col
        except (ValueError, IndexError):
            pass
    
    raise ValueError(f"Cannot parse tile position from filename: {filename}")


def group_tiles_by_slide(images_dir: Path, masks_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Group tiles by slide ID and collect their positions.
    
    Args:
        images_dir: Directory containing tile images
        masks_dir: Optional directory containing ground truth masks
    
    Returns:
        Dictionary mapping slide_id to tile information:
        {
            'slide_id': {
                'tiles': [(row, col, image_path, mask_path), ...],
                'row_range': (min_row, max_row),
                'col_range': (min_col, max_col)
            }
        }
    """
    slides = defaultdict(lambda: {'tiles': [], 'positions': set()})
    
    # Find all image tiles
    image_files = sorted(images_dir.glob("*.jpg"))
    
    # Index mask files if provided
    mask_files = {}
    if masks_dir and masks_dir.exists():
        for mask_path in masks_dir.glob("*.tif"):
            mask_files[mask_path.stem] = mask_path
    
    # Group by slide
    for img_path in image_files:
        try:
            slide_id, row, col = parse_tile_filename(img_path.name)
            
            # Find corresponding mask
            mask_path = mask_files.get(img_path.stem)
            
            slides[slide_id]['tiles'].append((row, col, img_path, mask_path))
            slides[slide_id]['positions'].add((row, col))
            
        except ValueError as e:
            warnings.warn(f"Skipping file {img_path.name}: {e}")
            continue
    
    # Calculate ranges for each slide
    for slide_id, info in slides.items():
        if info['positions']:
            rows = [r for r, c in info['positions']]
            cols = [c for r, c in info['positions']]
            info['row_range'] = (min(rows), max(rows))
            info['col_range'] = (min(cols), max(cols))
    
    return dict(slides)


def get_source_image_dimensions(slide_id: str, data_root: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Get actual dimensions from the original source image.
    
    Recursively searches ~/Data_for_ML/Meat_Luci_Tulane/Pseudocolored/ 
    and all subdirectories for the source image.
    
    Args:
        slide_id: Slide identifier
        data_root: Optional data root path (unused, kept for compatibility)
    
    Returns:
        (height, width) tuple if source found, None otherwise
    """
    # Base source directory
    base_source_dir = Path.home() / "Data_for_ML" / "Meat_Luci_Tulane" / "Pseudocolored"
    
    # Recursively search for the image file
    matches = list(base_source_dir.rglob(f"{slide_id}.jpg"))
    
    if matches:
        source_path = matches[0]  # Use first match
        print(f"  ✓ Found source image: {source_path}")
        try:
            # Use PIL to get image dimensions without loading entire image
            from PIL import Image
            with Image.open(source_path) as img:
                width, height = img.size
                return (height, width)
        except Exception as e:
            warnings.warn(f"Failed to load source image {source_path}: {e}")
            return None
    
    return None


def infer_full_image_dimensions(tile_positions: Set[Tuple[int, int]], 
                               tile_size: int, stride: int) -> Tuple[int, int]:
    """
    Infer the full image dimensions from tile positions (fallback method).
    
    This is a fallback when the source image cannot be found.
    Note: This may not perfectly match the original dimensions if edge tiles
    were clamped during extraction.
    
    Args:
        tile_positions: Set of (row, col) positions
        tile_size: Size of each tile (assumes square)
        stride: Stride between tiles
    
    Returns:
        (height, width) of full image
    """
    if not tile_positions:
        return (0, 0)
    
    rows = [r for r, c in tile_positions]
    cols = [c for r, c in tile_positions]
    
    max_row = max(rows)
    max_col = max(cols)
    
    # Calculate dimensions - this assumes ideal positioning
    # May be incorrect if edge tiles were clamped
    height = max_row * stride + tile_size
    width = max_col * stride + tile_size
    
    return (height, width)


def get_full_image_dimensions(slide_id: str, tile_positions: Set[Tuple[int, int]],
                              tile_size: int, stride: int, data_root: Optional[str] = None) -> Tuple[int, int]:
    """
    Get full image dimensions, trying source image first, then fallback to inference.
    
    Args:
        slide_id: Slide identifier for source image lookup
        tile_positions: Set of (row, col) positions for fallback
        tile_size: Size of each tile
        stride: Stride between tiles
        data_root: Optional data root path to help infer source location
    
    Returns:
        (height, width) of full image
    """
    # Try to get actual dimensions from source image
    dims = get_source_image_dimensions(slide_id, data_root)
    
    if dims is not None:
        print(f"  ✓ Using actual source image dimensions: {dims[1]}x{dims[0]}")
        return dims
    
    # Fallback to inference
    print(f"  ⚠️  Source image not found, inferring dimensions from tiles")
    return infer_full_image_dimensions(tile_positions, tile_size, stride)


def find_missing_tiles(expected_positions: Set[Tuple[int, int]], 
                      found_positions: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Identify missing tiles in a grid.
    
    Args:
        expected_positions: Complete grid of expected positions
        found_positions: Actual positions found
    
    Returns:
        Set of missing positions
    """
    return expected_positions - found_positions


def create_expected_grid(row_range: Tuple[int, int], 
                        col_range: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """Create a complete grid of expected tile positions."""
    min_row, max_row = row_range
    min_col, max_col = col_range
    
    positions = set()
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            positions.add((r, c))
    
    return positions


# ====================================================================
# RECONSTRUCTION
# ====================================================================

def reconstruct_slide(model, tiles_info: List[Tuple], full_shape: Tuple[int, int],
                     tile_size: int, stride: int, mean: float, std: float,
                     blender, boundary_refiner: Optional[BoundaryRefiner] = None,
                     use_tta: bool = False, tta_mode: str = 'basic') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct full slide image, prediction, and GT from tiles.
    
    Args:
        model: Model for prediction
        tiles_info: List of (row, col, image_path, mask_path)
        full_shape: (height, width) of full image
        tile_size: Size of tiles
        stride: Stride between tiles
        mean, std: Normalization parameters
        blender: Blending object (GaussianBlender or LinearBlender)
        boundary_refiner: Optional boundary refinement
        use_tta: Whether to use TTA
        tta_mode: TTA mode if enabled
    
    Returns:
        Tuple of (reconstructed_image_rgb, prediction_mask, gt_mask) all [0-1]
    """
    # Prepare tiles and positions
    predictions = []
    gt_tiles = []
    image_tiles_rgb = []
    positions = []
    
    for row, col, img_path, mask_path in tqdm(tiles_info, desc="  Processing tiles", leave=False):
        # Load tile for prediction (grayscale)
        tile_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Load tile in RGB for reconstruction
        tile_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        tile_rgb = cv2.cvtColor(tile_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tiles_rgb.append(tile_rgb)
        
        # Predict with optional TTA
        if use_tta:
            tta = TestTimeAugmentation(mode=tta_mode)
            pred, _ = tta.predict_with_tta(model, tile_gray, mean, std)
        else:
            pred = model.predict_single(tile_gray, mean, std)
        
        # Apply boundary refinement if enabled
        if boundary_refiner is not None:
            pred = boundary_refiner.refine(pred, tile_gray)
        
        predictions.append(pred)
        
        # Load GT if available
        if mask_path is not None:
            gt_tile = tiff.imread(str(mask_path)).astype(np.float32)
            if gt_tile.ndim == 3:
                gt_tile = gt_tile.squeeze()
            
            # Smart normalization - only divide by 255 if needed
            if gt_tile.max() > 1.0:
                gt_tile = gt_tile / 255.0  # Masks in [0-255] range
            # else: already in [0-1] range
            
            gt_tiles.append(gt_tile)
        
        # Calculate pixel position (clamp to image bounds for edge tiles)
        # This matches how build_test_dataset.py extracts edge tiles
        y = min(row * stride, full_shape[0] - tile_size)
        x = min(col * stride, full_shape[1] - tile_size)
        positions.append((y, x))
    
    # Reconstruct prediction with blending
    full_pred = blender.reconstruct(predictions, positions, full_shape)
    
    # Reconstruct GT if available
    full_gt = None
    if gt_tiles:
        full_gt = blender.reconstruct(gt_tiles, positions, full_shape)
    
    # Reconstruct RGB image (separate blending for each channel)
    full_image_rgb = np.zeros((full_shape[0], full_shape[1], 3), dtype=np.float32)
    for ch in range(3):
        channel_tiles = [tile[:, :, ch] for tile in image_tiles_rgb]
        full_image_rgb[:, :, ch] = blender.reconstruct(channel_tiles, positions, full_shape)
    
    return full_image_rgb, full_pred, full_gt


# ====================================================================
# VISUALIZATION
# ====================================================================

def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, 
                  color: Tuple[int, int, int] = (255, 0, 255),
                  alpha: float = 0.4) -> np.ndarray:
    """
    Create overlay of mask on RGB image (matching full_evaluation_enhanced).
    
    Args:
        image_rgb: RGB image [0-1] float or [0-255] uint8
        mask: Binary mask [0-1]
        color: RGB color for mask overlay (0-255)
        alpha: Transparency for colored mask
    
    Returns:
        RGB overlay image uint8
    """
    # Ensure image is uint8 RGB
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)
    
    # Create colored mask
    color_mask = np.zeros_like(image_rgb)
    mask_binary = (mask > 0.5).astype(bool)
    color_mask[mask_binary] = color
    
    # Blend with cv2.addWeighted (60% image, 40% color mask)
    overlay = cv2.addWeighted(image_rgb, 0.6, color_mask.astype(np.uint8), 0.4, 0)
    
    return overlay


def create_4panel_comparison(original_rgb: np.ndarray, gt_mask: np.ndarray,
                             pred_mask: np.ndarray, slide_id: str,
                             dice_score: float) -> np.ndarray:
    """
    Create 4-panel comparison matching full_evaluation_enhanced style.
    
    Panels:
    1. Original RGB image
    2. GT overlay (yellow on RGB)
    3. Prediction overlay (magenta on RGB)
    4. Discrepancy map (TP=green, FP=red, FN=blue, TN=black)
    
    Args:
        original_rgb: RGB image [0-1]
        gt_mask: Ground truth mask [0-1]
        pred_mask: Prediction mask [0-1]
        slide_id: Slide identifier
        dice_score: Dice score for title
    
    Returns:
        4-panel comparison image
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    # Panel 1: Original
    original_uint8 = (original_rgb * 255).astype(np.uint8) if original_rgb.max() <= 1.0 else original_rgb.astype(np.uint8)
    
    # Panel 2: GT overlay (yellow)
    gt_overlay = create_overlay(original_rgb, gt_mask, color=(255, 255, 0), alpha=0.4)
    
    # Panel 3: Prediction overlay (magenta)
    pred_overlay = create_overlay(original_rgb, pred_mask, color=(255, 0, 255), alpha=0.4)
    
    # Panel 4: Discrepancy map
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    
    discrepancy = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    
    # TP = green, FP = red, FN = blue, TN = black
    tp_mask = (gt_binary == 1) & (pred_binary == 1)
    fp_mask = (gt_binary == 0) & (pred_binary == 1)
    fn_mask = (gt_binary == 1) & (pred_binary == 0)
    
    discrepancy[tp_mask] = [0, 255, 0]    # Green
    discrepancy[fp_mask] = [255, 0, 0]    # Red
    discrepancy[fn_mask] = [0, 0, 255]    # Blue
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    axes[0, 0].imshow(original_uint8)
    axes[0, 0].set_title('Original Image (RGB Pseudocolored)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title('Ground Truth (Yellow)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(pred_overlay)
    axes[1, 0].set_title(f'Prediction (Magenta) - Dice: {dice_score:.4f}', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(discrepancy)
    axes[1, 1].set_title('Discrepancy Map (TP=Green, FP=Red, FN=Blue)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    fig.suptitle(f'Full Image Reconstruction: {slide_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    # Use tobytes() instead of deprecated tostring_rgb()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return img


# ====================================================================
# RECONSTRUCTION LOG
# ====================================================================

def create_reconstruction_log(args, output_dir: Path, slide_results: List[Dict]) -> Path:
    """Create comprehensive reconstruction log."""
    
    log_data = {
        'reconstruction_info': {
            'timestamp': datetime.now().isoformat(),
            'script_version': 'Full Image Reconstruction v1.0',
            'output_directory': str(output_dir)
        },
        'configuration': {
            'weights_path': args.weights,
            'data_root': args.data_root,
            'tile_size': args.tile_size,
            'stride': args.stride,
            'threshold': args.threshold,
            'blend_mode': args.blend_mode,
            'use_tta': args.use_tta,
            'tta_mode': args.tta_mode if args.use_tta else None,
            'boundary_refine': args.boundary_refine,
            'refine_kernel': args.refine_kernel if args.boundary_refine else None
        },
        'slides_processed': len(slide_results),
        'slide_results': slide_results,
        'summary_statistics': {
            'mean_dice': np.mean([r['metrics']['dice_score'] for r in slide_results]),
            'mean_coverage': np.mean([r['reconstruction']['coverage_ratio'] for r in slide_results]),
            'total_tiles_used': sum(r['reconstruction']['tiles_used'] for r in slide_results),
            'total_tiles_missing': sum(r['reconstruction']['tiles_missing'] for r in slide_results)
        }
    }
    
    log_file = output_dir / "reconstruction_log.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_file


# ====================================================================
# MAIN PROCESSING
# ====================================================================

def reconstruct_all_slides(args):
    """Main reconstruction pipeline."""
    
    print(f"\n{'='*80}")
    print(f"FULL IMAGE RECONSTRUCTION FROM OVERLAPPING TILES")
    print(f"{'='*80}")
    
    # Set deterministic environment
    set_deterministic_seeds(1337)
    
    # Setup paths
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    output_dir = Path(args.output_dir)
    
    # Modify output directory name if max-tiles is specified
    if args.max_tiles:
        output_dir = output_dir.parent / f"{output_dir.name}_{args.max_tiles}x{args.max_tiles}"
        print(f"✓ Output directory modified for {args.max_tiles}x{args.max_tiles} reconstruction:")
        print(f"  {output_dir}")
    
    # Validate inputs
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Create output directories
    for subdir in ['masks', 'overlays', 'comparisons', 'metrics']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    checkpoint_dir = Path(args.weights).parent
    train_mean, train_std = load_training_stats(str(checkpoint_dir))
    
    model = AdiposeUNet()
    model.build_model()
    model.load_weights(args.weights)
    print("✓ Model loaded successfully")
    
    # Initialize post-processing
    if args.blend_mode == 'gaussian':
        blender = GaussianBlender(tile_size=args.tile_size, sigma_factor=0.25)
    elif args.blend_mode == 'linear':
        blender = LinearBlender()
    else:
        blender = LinearBlender()  # Default
    
    boundary_refiner = None
    if args.boundary_refine:
        boundary_refiner = BoundaryRefiner(
            kernel_size=args.refine_kernel,
            bilateral_d=5,
            bilateral_sigma_color=50,
            bilateral_sigma_space=50
        )
        print(f"✓ Boundary refinement enabled (kernel={args.refine_kernel})")
    
    # Group tiles by slide
    print(f"\nGrouping tiles by slide...")
    slides_data = group_tiles_by_slide(images_dir, masks_dir)
    print(f"✓ Found {len(slides_data)} slide(s)")
    
    # Process each slide
    slide_results = []
    
    for slide_id, slide_info in slides_data.items():
        print(f"\n{'='*80}")
        print(f"Processing: {slide_id}")
        print(f"{'='*80}")
        
        tiles_info = slide_info['tiles']
        positions = slide_info['positions']
        
        print(f"  Tiles found: {len(tiles_info)}")
        
        # Apply max-tiles filtering if specified
        if args.max_tiles:
            original_count = len(tiles_info)
            tiles_info = [(r, c, img, mask) for r, c, img, mask in tiles_info 
                         if r < args.max_tiles and c < args.max_tiles]
            positions = {(r, c) for r, c in positions if r < args.max_tiles and c < args.max_tiles}
            discarded_count = original_count - len(tiles_info)
            
            print(f"  Limiting to {args.max_tiles}x{args.max_tiles} grid:")
            print(f"    - Tiles used: {len(tiles_info)}")
            print(f"    - Tiles discarded: {discarded_count}")
        
        # Check for missing tiles
        # When max_tiles is specified, use the filtered range (0 to max_tiles-1)
        # Otherwise use the original slide range
        if args.max_tiles:
            # For limited reconstruction, expected grid is 0 to max_tiles-1
            row_range = (0, args.max_tiles - 1)
            col_range = (0, args.max_tiles - 1)
        else:
            row_range = slide_info['row_range']
            col_range = slide_info['col_range']
        
        expected_grid = create_expected_grid(row_range, col_range)
        missing_positions = find_missing_tiles(expected_grid, positions)
        
        if missing_positions:
            print(f"  ⚠️  Missing {len(missing_positions)} tile(s):")
            for pos in sorted(list(missing_positions)[:5]):
                print(f"    - r{pos[0]}_c{pos[1]}")
            if len(missing_positions) > 5:
                print(f"    ... and {len(missing_positions) - 5} more")
        
        coverage_ratio = len(positions) / len(expected_grid)
        print(f"  Coverage: {coverage_ratio:.1%}")
        
        if coverage_ratio < args.min_coverage:
            print(f"  ⚠️  Skipping (coverage {coverage_ratio:.1%} < {args.min_coverage:.1%})")
            continue
        
        # Get full image dimensions
        if args.max_tiles:
            # Calculate dimensions based on limited grid
            full_shape = ((args.max_tiles - 1) * args.stride + args.tile_size,
                         (args.max_tiles - 1) * args.stride + args.tile_size)
            print(f"  ✓ Using limited dimensions: {full_shape[1]}x{full_shape[0]} ({args.max_tiles}x{args.max_tiles} tiles)")
        else:
            # Get full image dimensions (tries source image first, then inference)
            full_shape = get_full_image_dimensions(slide_id, positions, args.tile_size, args.stride, args.data_root)
        
        # Reconstruct image, prediction, and GT
        print(f"  Reconstructing with {args.blend_mode} blending...")
        full_image_rgb, full_pred, full_gt = reconstruct_slide(
            model, tiles_info, full_shape, args.tile_size, args.stride,
            train_mean, train_std, blender, boundary_refiner,
            args.use_tta, args.tta_mode
        )
        
        # Create slide-specific output directory
        slide_dir = output_dir / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original RGB image
        original_path = slide_dir / "original_image.tif"
        original_uint8 = (full_image_rgb * 255).astype(np.uint8)
        original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
        tiff.imwrite(str(original_path), original_bgr, compression='lzw')
        print(f"  ✓ Saved original RGB: {original_path.name}")
        
        # Save prediction mask (binary)
        pred_path = slide_dir / "prediction_mask.tif"
        tiff.imwrite(str(pred_path), (full_pred * 255).astype(np.uint8), compression='lzw')
        print(f"  ✓ Saved prediction mask: {pred_path.name}")
        
        # Calculate metrics if GT available
        has_gt = (full_gt is not None)
        metrics = {}
        
        if has_gt:
            print(f"  Computing metrics...")
            
            # Save GT mask
            gt_path = slide_dir / "ground_truth_mask.tif"
            tiff.imwrite(str(gt_path), (full_gt * 255).astype(np.uint8), compression='lzw')
            print(f"  ✓ Saved GT mask: {gt_path.name}")
            
            # Compute metrics
            metrics = calculate_pixel_metrics(full_pred, full_gt, args.threshold)
            
            print(f"  Dice: {metrics['dice_score']:.4f}")
            print(f"  IoU: {metrics['jaccard_index']:.4f}")
            
            # Create GT overlay (yellow on RGB)
            gt_overlay = create_overlay(full_image_rgb, full_gt, color=(255, 255, 0), alpha=0.4)
            gt_overlay_path = slide_dir / "gt_overlay.png"
            gt_overlay_bgr = cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(gt_overlay_path), gt_overlay_bgr)
            print(f"  ✓ Saved GT overlay: {gt_overlay_path.name}")
            
            # Create prediction overlay (magenta on RGB)
            pred_overlay = create_overlay(full_image_rgb, full_pred, color=(255, 0, 255), alpha=0.4)
            pred_overlay_path = slide_dir / "pred_overlay.png"
            pred_overlay_bgr = cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(pred_overlay_path), pred_overlay_bgr)
            print(f"  ✓ Saved prediction overlay: {pred_overlay_path.name}")
            
            # Create 4-panel comparison
            comparison_4panel = create_4panel_comparison(
                full_image_rgb, full_gt, full_pred, slide_id, metrics['dice_score']
            )
            comparison_path = slide_dir / "comparison_4panel.png"
            comparison_bgr = cv2.cvtColor(comparison_4panel, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(comparison_path), comparison_bgr)
            print(f"  ✓ Saved 4-panel comparison: {comparison_path.name}")
            
            # Save human-readable metrics.txt
            metrics_txt_path = slide_dir / "metrics.txt"
            with open(metrics_txt_path, 'w') as f:
                f.write(f"Full Image Reconstruction Metrics\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Slide: {slide_id}\n")
                f.write(f"Image Size: {full_shape[1]} x {full_shape[0]} pixels\n")
                f.write(f"Tiles Used: {len(tiles_info)}\n")
                f.write(f"Coverage: {coverage_ratio:.1%}\n\n")
                f.write(f"Reconstruction Settings:\n")
                f.write(f"  Blend Mode: {args.blend_mode}\n")
                f.write(f"  TTA: {'Yes (' + args.tta_mode + ')' if args.use_tta else 'No'}\n")
                f.write(f"  Boundary Refinement: {'Yes' if args.boundary_refine else 'No'}\n")
                f.write(f"  Threshold: {args.threshold}\n\n")
                f.write(f"Performance Metrics:\n")
                f.write(f"  Dice Score:     {metrics['dice_score']:.4f}\n")
                f.write(f"  IoU (Jaccard):  {metrics['jaccard_index']:.4f}\n")
                f.write(f"  Sensitivity:    {metrics['sensitivity']:.4f}\n")
                f.write(f"  Specificity:    {metrics['specificity']:.4f}\n")
                f.write(f"  Precision:      {metrics['precision']:.4f}\n")
                f.write(f"  F1-Score:       {metrics['f1_score']:.4f}\n")
            print(f"  ✓ Saved metrics.txt: {metrics_txt_path.name}")
        
        # Save per-slide metrics
        if args.save_metrics:
            slide_result = {
                'slide_id': slide_id,
                'reconstruction': {
                    'tiles_used': len(tiles_info),
                    'tiles_missing': len(missing_positions),
                    'coverage_ratio': coverage_ratio,
                    'blend_mode': args.blend_mode,
                    'tta_enabled': args.use_tta,
                    'tta_mode': args.tta_mode if args.use_tta else None,
                    'boundary_refined': args.boundary_refine
                },
                'dimensions': {
                    'width': full_shape[1],
                    'height': full_shape[0],
                    'tiles_rows': slide_info['row_range'][1] - slide_info['row_range'][0] + 1,
                    'tiles_cols': slide_info['col_range'][1] - slide_info['col_range'][0] + 1
                },
                'metrics': metrics if has_gt else None
            }
            
            slide_results.append(slide_result)
            
            # Save individual JSON
            metrics_path = output_dir / "metrics" / f"{slide_id}_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            with open(metrics_path, 'w') as f:
                json.dump(slide_result, f, indent=2)
    
    # Create summary
    if slide_results:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        
        # Save aggregated CSV
        summary_data = []
        for result in slide_results:
            if result['metrics']:
                summary_data.append({
                    'slide_id': result['slide_id'],
                    'dice_score': result['metrics']['dice_score'],
                    'jaccard_iou': result['metrics']['jaccard_index'],
                    'sensitivity': result['metrics']['sensitivity'],
                    'specificity': result['metrics']['specificity'],
                    'precision': result['metrics']['precision'],
                    'tiles_used': result['reconstruction']['tiles_used'],
                    'tiles_missing': result['reconstruction']['tiles_missing'],
                    'coverage': result['reconstruction']['coverage_ratio']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_dir / "metrics" / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Summary saved: {summary_path}")
            
            print(f"\nMean Dice: {summary_df['dice_score'].mean():.4f}")
            print(f"Mean IoU: {summary_df['jaccard_iou'].mean():.4f}")
        
        # Create reconstruction log
        log_path = create_reconstruction_log(args, output_dir, slide_results)
        print(f"✓ Reconstruction log: {log_path}")
    
    print(f"\n✅ Reconstruction complete!")
    print(f"   Output directory: {output_dir}")


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Reconstruct full images from overlapping tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset directory (contains images/ and masks/)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for reconstructed images')
    
    # Reconstruction parameters
    parser.add_argument('--tile-size', type=int, default=1024,
                       help='Tile size (default: 1024)')
    parser.add_argument('--stride', type=int, default=512,
                       help='Stride between tiles (default: 512 for 50%% overlap)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary masks (default: 0.5)')
    
    # Blending
    parser.add_argument('--blend-mode', type=str, default='gaussian',
                       choices=['gaussian', 'linear'],
                       help='Blending mode (default: gaussian)')
    
    # Post-processing
    parser.add_argument('--use-tta', action='store_true', default=False,
                       help='Enable Test Time Augmentation')
    parser.add_argument('--tta-mode', type=str, default='basic',
                       choices=['minimal', 'basic', 'full'],
                       help='TTA mode (default: basic)')
    parser.add_argument('--boundary-refine', action='store_true', default=False,
                       help='Enable boundary refinement')
    parser.add_argument('--refine-kernel', type=int, default=5,
                       help='Kernel size for boundary refinement (default: 5)')
    
    # Output options
    parser.add_argument('--save-masks', action='store_true', default=True,
                       help='Save reconstructed masks (default: True)')
    parser.add_argument('--save-overlays', action='store_true', default=False,
                       help='Save overlay visualizations')
    parser.add_argument('--save-comparisons', action='store_true', default=False,
                       help='Save GT vs Pred comparisons')
    parser.add_argument('--save-metrics', action='store_true', default=False,
                       help='Compute and save per-slide metrics')
    
    # Missing tile handling
    parser.add_argument('--min-coverage', type=float, default=0.90,
                       help='Minimum tile coverage to process slide (default: 0.90)')
    
    # Tile limiting
    parser.add_argument('--max-tiles', type=int, default=None,
                       help='Limit reconstruction to NxN tiles (e.g., 6 for 6x6 grid)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    args = parse_args()
    
    # GPU setup
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    
    try:
        reconstruct_all_slides(args)
        return 0
    except Exception as e:
        print(f"\n❌ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
