#!/usr/bin/env python3
"""
Reconstruct WSI with classification overlay from evaluation results.

Reads predictions.csv from eval_adipose_classifier.py and creates visualization
overlays showing True Positives (green), False Positives (red), False Negatives (orange),
and True Negatives (light blue) on the original WSI.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Reconstruct WSI with classification overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-csv", type=str, required=True,
                        help="Path to predictions.csv from eval_adipose_classifier.py")
    parser.add_argument("--metrics-json", type=str, required=True,
                        help="Path to metrics.json containing best_threshold")
    parser.add_argument("--wsi-dir", type=str, required=True,
                        help="Directory containing original WSI files (.tif, .tiff, .png)")
    parser.add_argument("--tiles-dir", type=str, required=True,
                        help="Directory containing intermediate tiles (e.g., ECM_channel/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save reconstructed WSI with overlays")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification threshold (default: use best_threshold from metrics.json)")
    parser.add_argument("--overlay-alpha", type=float, default=0.4,
                        help="Opacity of overlay rectangles (0=transparent, 1=opaque)")
    parser.add_argument("--tile-size", type=int, default=1024,
                        help="Size of classification tiles (patches)")
    parser.add_argument("--combine-patches", type=int, default=1,
                        help="Combine N×N patches into single overlay block (1=no combining, 2=2×2, etc.)")
    parser.add_argument("--save-original", action="store_true", default=True,
                        help="Save original WSI as PNG (enabled by default)")
    parser.add_argument("--no-save-original", dest="save_original", action="store_false",
                        help="Don't save original WSI")
    parser.add_argument("--downsample", type=int, default=8,
                        help="Downsample factor for output images (1=full resolution, 2=half, 4=quarter, 8=eighth)")
    return parser.parse_args()


# Color scheme (RGB format)
COLORS = {
    "TP": (0, 255, 0),      # Green - True Positive (correctly detected adipose)
    "FP": (255, 0, 0),      # Red - False Positive (incorrectly detected as adipose)
    "FN": (255, 140, 0),    # Orange - False Negative (missed adipose)
    "TN": (0, 255, 255),    # Cyan - True Negative (correctly rejected)
}


def parse_subtile_filename(filename: str) -> Optional[Dict[str, int]]:
    """
    Parse subtile filename to extract parent tile and position info.
    
    Example: Meat_11_13_S1_1_004_x18240_y0_w5120_h6144_r4_c1.jpg
    Returns: {
        'slide_base': 'Meat_11_13_S1_1',
        'tile_id': '004',
        'tile_x': 18240,
        'tile_y': 0,
        'tile_w': 5120,
        'tile_h': 6144,
        'row': 4,
        'col': 1
    }
    """
    stem = Path(filename).stem
    
    # Pattern: {slide_base}_{tile_id}_x{x}_y{y}_w{w}_h{h}_r{row}_c{col}
    pattern = r'^(.+?)_(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)_r(\d+)_c(\d+)$'
    match = re.match(pattern, stem)
    
    if not match:
        return None
    
    slide_base, tile_id, x, y, w, h, row, col = match.groups()
    
    return {
        'slide_base': slide_base,
        'tile_id': tile_id,
        'tile_x': int(x),
        'tile_y': int(y),
        'tile_w': int(w),
        'tile_h': int(h),
        'row': int(row),
        'col': int(col),
    }


def parse_tile_filename(filename: str) -> Optional[Dict[str, int]]:
    """
    Parse intermediate tile filename.
    
    Example: Meat_11_13_S1_1_001_x0_y0_w6144_h6144.png
    Returns: {
        'slide_base': 'Meat_11_13_S1_1',
        'tile_id': '001',
        'x': 0,
        'y': 0,
        'w': 6144,
        'h': 6144
    }
    """
    stem = Path(filename).stem
    
    # Pattern: {slide_base}_{tile_id}_x{x}_y{y}_w{w}_h{h}
    pattern = r'^(.+?)_(\d+)_x(\d+)_y(\d+)_w(\d+)_h(\d+)$'
    match = re.match(pattern, stem)
    
    if not match:
        return None
    
    slide_base, tile_id, x, y, w, h = match.groups()
    
    return {
        'slide_base': slide_base,
        'tile_id': tile_id,
        'x': int(x),
        'y': int(y),
        'w': int(w),
        'h': int(h),
    }


def load_predictions(csv_path: Path, threshold: float) -> Dict[str, Dict]:
    """
    Load predictions from CSV and classify each tile.
    
    Returns:
        Dict mapping filename to prediction info:
        {
            'filename': {
                'label': 0 or 1,
                'prob': float,
                'pred': 0 or 1,
                'category': 'TP', 'FP', 'FN', or 'TN'
            }
        }
    """
    predictions = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = Path(row['path']).name
            label = int(row['label'])
            prob = float(row['prob'])
            pred = 1 if prob >= threshold else 0
            
            # Determine category
            if pred == 1 and label == 1:
                category = 'TP'
            elif pred == 1 and label == 0:
                category = 'FP'
            elif pred == 0 and label == 1:
                category = 'FN'
            else:  # pred == 0 and label == 0
                category = 'TN'
            
            predictions[filename] = {
                'label': label,
                'prob': prob,
                'pred': pred,
                'category': category,
            }
    
    return predictions


def load_wsi(wsi_path: Path) -> np.ndarray:
    """Load WSI from TIFF or PNG."""
    if wsi_path.suffix.lower() in ['.tif', '.tiff']:
        img = tifffile.imread(str(wsi_path))
    else:
        with Image.open(wsi_path) as pil_img:
            img = np.array(pil_img)
    
    # Convert to 8-bit RGB
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    
    if img.ndim == 2:
        # Grayscale to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    return img


def group_predictions_by_wsi(predictions: Dict, tiles_dir: Path) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Group predictions by WSI slide.
    
    Returns:
        Dict mapping slide_base to list of (filename, prediction_info) tuples
    """
    grouped = defaultdict(list)
    
    for filename, pred_info in predictions.items():
        parsed = parse_subtile_filename(filename)
        if not parsed:
            print(f"[WARN] Could not parse filename: {filename}")
            continue
        
        slide_base = parsed['slide_base']
        grouped[slide_base].append((filename, pred_info, parsed))
    
    return grouped


def combine_patches(predictions_with_coords: List[Tuple[str, Dict, Dict]], 
                   combine_size: int, tile_size: int) -> List[Tuple[int, int, int, int, str, float]]:
    """
    Combine N×N patches into single overlay blocks.
    
    Args:
        predictions_with_coords: List of (filename, pred_info, parsed_coords)
        combine_size: N for N×N combining (1 = no combining)
        tile_size: Size of each patch
        
    Returns:
        List of (x, y, width, height, category, avg_prob) for combined blocks
    """
    if combine_size == 1:
        # No combining - return individual patches
        blocks = []
        for filename, pred_info, parsed in predictions_with_coords:
            # Calculate absolute position in WSI
            abs_x = parsed['tile_x'] + parsed['col'] * tile_size
            abs_y = parsed['tile_y'] + parsed['row'] * tile_size
            
            blocks.append((
                abs_x,
                abs_y,
                tile_size,
                tile_size,
                pred_info['category'],
                pred_info['prob']
            ))
        return blocks
    
    # Group patches into combine_size × combine_size blocks
    # Key: (block_row, block_col, tile_x, tile_y)
    block_groups = defaultdict(list)
    
    for filename, pred_info, parsed in predictions_with_coords:
        # Determine which block this patch belongs to
        block_row = parsed['row'] // combine_size
        block_col = parsed['col'] // combine_size
        
        key = (block_row, block_col, parsed['tile_x'], parsed['tile_y'])
        block_groups[key].append((parsed, pred_info))
    
    # Create combined blocks
    combined_blocks = []
    for (block_row, block_col, tile_x, tile_y), patches in block_groups.items():
        # Calculate block position
        abs_x = tile_x + block_col * combine_size * tile_size
        abs_y = tile_y + block_row * combine_size * tile_size
        block_w = combine_size * tile_size
        block_h = combine_size * tile_size
        
        # Determine dominant category using hierarchical priority logic
        categories = [p[1]['category'] for p in patches]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        # Priority hierarchy:
        # 1. If any TP exists, mark as TP (true positive trumps all)
        # 2. Else if any TN exists, mark as TN (true negative is second priority)
        # 3. Else if FN and FP both exist, or only FN/FP:
        #    - Prefer FP over FN
        
        if category_counts.get('TP', 0) > 0:
            dominant_category = 'TP'
        elif category_counts.get('TN', 0) > 0:
            dominant_category = 'TN'
        else:
            # Only FP and/or FN remain - prefer FP
            fp_count = category_counts.get('FP', 0)
            fn_count = category_counts.get('FN', 0)
            
            if fp_count > 0:
                dominant_category = 'FP'
            else:
                dominant_category = 'FN'
        
        # Average probability
        avg_prob = np.mean([p[1]['prob'] for p in patches])
        
        combined_blocks.append((abs_x, abs_y, block_w, block_h, dominant_category, avg_prob))
    
    return combined_blocks


def create_overlay(wsi: np.ndarray, blocks: List[Tuple[int, int, int, int, str, float]], 
                   alpha: float) -> np.ndarray:
    """
    Create overlay on WSI with colored rectangles.
    
    Args:
        wsi: Original WSI image (RGB)
        blocks: List of (x, y, w, h, category, prob)
        alpha: Opacity (0-1)
        
    Returns:
        WSI with overlay applied
    """
    overlay = wsi.copy()
    
    for x, y, w, h, category, prob in blocks:
        color = COLORS[category]
        
        # Draw semi-transparent rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    
    # Blend overlay with original
    result = cv2.addWeighted(overlay, alpha, wsi, 1 - alpha, 0)
    
    return result


def add_legend(img: np.ndarray, category_counts: Dict[str, int]) -> np.ndarray:
    """Add legend showing category counts and colors."""
    # Add white border at bottom for legend
    legend_height = 120
    h, w = img.shape[:2]
    
    # Create canvas with extra space
    canvas = np.ones((h + legend_height, w, 3), dtype=np.uint8) * 255
    canvas[:h, :] = img
    
    # Draw legend
    x_offset = 20
    y_base = h + 30
    
    legend_items = [
        ("TP", "True Positive (Correct Adipose)", category_counts.get('TP', 0)),
        ("FP", "False Positive (Incorrect Adipose)", category_counts.get('FP', 0)),
        ("FN", "False Negative (Missed Adipose)", category_counts.get('FN', 0)),
        ("TN", "True Negative (Correct Non-Adipose)", category_counts.get('TN', 0)),
    ]
    
    for i, (cat, label, count) in enumerate(legend_items):
        # Color box
        box_x = x_offset + i * (w // 4)
        cv2.rectangle(canvas, (box_x, y_base), (box_x + 30, y_base + 30), COLORS[cat], -1)
        cv2.rectangle(canvas, (box_x, y_base), (box_x + 30, y_base + 30), (0, 0, 0), 2)
        
        # Text
        text = f"{label}: {count}"
        cv2.putText(canvas, text, (box_x + 40, y_base + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return canvas


def downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample image by factor."""
    if factor == 1:
        return img
    
    h, w = img.shape[:2]
    new_h, new_w = h // factor, w // factor
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    
    # Load best threshold from metrics
    if args.threshold is None:
        with open(args.metrics_json, 'r') as f:
            metrics = json.load(f)
            threshold = metrics.get('best_threshold', 0.5)
            print(f"[Threshold] Using best F1 threshold from metrics: {threshold:.3f}")
    else:
        threshold = args.threshold
        print(f"[Threshold] Using user-specified threshold: {threshold:.3f}")
    
    # Load predictions
    print(f"[Load] Reading predictions from {args.predictions_csv}")
    predictions = load_predictions(Path(args.predictions_csv), threshold)
    print(f"[Load] Loaded {len(predictions)} tile predictions")
    
    # Group by WSI
    tiles_dir = Path(args.tiles_dir)
    grouped = group_predictions_by_wsi(predictions, tiles_dir)
    print(f"[Group] Found predictions for {len(grouped)} WSI slides")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each WSI
    wsi_dir = Path(args.wsi_dir)
    
    for slide_base, pred_list in tqdm(grouped.items(), desc="Reconstructing WSIs"):
        # Find corresponding WSI file
        wsi_candidates = list(wsi_dir.glob(f"{slide_base}.*"))
        if not wsi_candidates:
            print(f"[WARN] Could not find WSI for {slide_base}")
            continue
        
        wsi_path = wsi_candidates[0]
        print(f"\n[Process] {slide_base} ({len(pred_list)} tiles)")
        print(f"[WSI] Loading from {wsi_path}")
        
        # Load WSI
        wsi = load_wsi(wsi_path)
        print(f"[WSI] Dimensions: {wsi.shape[1]}×{wsi.shape[0]}")
        
        # Combine patches if requested
        blocks = combine_patches(pred_list, args.combine_patches, args.tile_size)
        print(f"[Overlay] Creating {len(blocks)} overlay blocks (combine={args.combine_patches}×{args.combine_patches})")
        
        # Count categories
        category_counts = {}
        for _, _, _, _, cat, _ in blocks:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print(f"[Stats] TP={category_counts.get('TP', 0)}, "
              f"FP={category_counts.get('FP', 0)}, "
              f"FN={category_counts.get('FN', 0)}, "
              f"TN={category_counts.get('TN', 0)}")
        
        # Create overlay
        wsi_with_overlay = create_overlay(wsi, blocks, args.overlay_alpha)
        
        # Add legend
        wsi_with_overlay = add_legend(wsi_with_overlay, category_counts)
        
        # Downsample if requested
        if args.downsample > 1:
            wsi = downsample_image(wsi, args.downsample)
            wsi_with_overlay = downsample_image(wsi_with_overlay, args.downsample)
            print(f"[Downsample] Reduced to {wsi.shape[1]}×{wsi.shape[0]} (factor={args.downsample})")
        
        # Save original
        if args.save_original:
            orig_path = output_dir / f"{slide_base}_original.png"
            wsi_bgr = cv2.cvtColor(wsi, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(orig_path), wsi_bgr)
            print(f"[Save] Original: {orig_path}")
        
        # Save overlay
        overlay_path = output_dir / f"{slide_base}_overlay.png"
        overlay_bgr = cv2.cvtColor(wsi_with_overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        print(f"[Save] Overlay: {overlay_path}")
        
        # Save stats
        stats_path = output_dir / f"{slide_base}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'slide': slide_base,
                'wsi_path': str(wsi_path),
                'threshold': threshold,
                'total_blocks': len(blocks),
                'combine_patches': args.combine_patches,
                'category_counts': category_counts,
            }, f, indent=2)
    
    print(f"\n[Done] Reconstructed WSIs saved to {output_dir}")
    print(f"[Info] Use --combine-patches N to group N×N tiles into larger overlay blocks")
    print(f"[Info] Use --downsample N to reduce output size by factor N")


if __name__ == "__main__":
    main()
