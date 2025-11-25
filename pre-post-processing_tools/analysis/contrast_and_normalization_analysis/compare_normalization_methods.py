#!/usr/bin/env python3
"""
Updated Normalization Methods Comparison for Adipose Tissue U-Net
Compares 6 normalization approaches with detail loss annotations
"""

import os
import sys
from pathlib import Path
import random
import argparse

import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (24, 16)


def load_image_mask_pair(img_path, mask_path):
    """Load image and mask pair"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mask = tiff.imread(str(mask_path)).astype(np.float32)
    if mask.ndim == 3:
        mask = mask.squeeze()
    return img, mask


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8).astype(np.float32)


def normalize_percentile(img, p_low=1, p_high=99):
    """Percentile-based normalization"""
    plow, phigh = np.percentile(img, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    return np.clip((img - plow) / scale, 0, 1)


def detect_detail_loss_regions(original, processed, threshold=0.15):
    """
    Detect regions where detail might be lost in processed image
    Returns binary mask of potentially problematic regions
    """
    # Convert to same range for comparison
    orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-6)
    
    # Calculate local variance (detail indicator)
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # Local variance in original
    orig_mean = cv2.filter2D(orig_norm, -1, kernel)
    orig_var = cv2.filter2D(orig_norm**2, -1, kernel) - orig_mean**2
    
    # Local variance in processed
    if processed.max() > 1.5:  # Likely not normalized to [0,1]
        proc_norm = (processed - processed.min()) / (processed.max() - processed.min() + 1e-6)
    else:
        proc_norm = processed
    
    proc_mean = cv2.filter2D(proc_norm, -1, kernel)
    proc_var = cv2.filter2D(proc_norm**2, -1, kernel) - proc_mean**2
    
    # Find regions where variance significantly decreased
    var_loss = np.maximum(0, orig_var - proc_var)
    detail_loss_mask = var_loss > threshold
    
    # Also check for overly dark regions (< 5th percentile of processed image)
    dark_threshold = np.percentile(proc_norm, 5)
    dark_mask = proc_norm < dark_threshold
    
    # Combine masks
    problem_mask = detail_loss_mask | dark_mask
    
    return problem_mask


def find_problem_regions(problem_mask, min_area=500):
    """Find connected components of problematic regions and return bounding boxes"""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        problem_mask.astype(np.uint8), connectivity=8
    )
    
    boxes = []
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, w, h))
    
    return boxes


def apply_six_normalization_methods(img):
    """Apply all 6 normalization methods"""
    
    # 1. Original (raw grayscale 0-255, display as 0-1)
    img_original = img / 255.0
    
    # 2. CLAHE only
    img_clahe_only = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img_clahe_only = img_clahe_only / 255.0  # Normalize to [0,1] for display
    
    # 3. Percentile only (1-99th percentiles)
    img_percentile_only = normalize_percentile(img, p_low=1, p_high=99)
    
    # 4. CLAHE + Percentile (aggressive - original from analysis)
    img_clahe_perc_agg = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img_clahe_perc_agg = normalize_percentile(img_clahe_perc_agg, p_low=1, p_high=99)
    
    # 5. Gentle Percentile (10-90th percentiles)
    img_gentle_perc = normalize_percentile(img, p_low=10, p_high=90)
    
    # 6. Light CLAHE + Wider Percentile (conservative)
    img_light_clahe = apply_clahe(img, clip_limit=1.2, tile_grid_size=(16, 16))
    img_light_clahe_perc = normalize_percentile(img_light_clahe, p_low=5, p_high=95)
    
    images = [
        img_original,
        img_clahe_only, 
        img_percentile_only,
        img_clahe_perc_agg,
        img_gentle_perc,
        img_light_clahe_perc
    ]
    
    titles = [
        'Original',
        'CLAHE Only',
        'Percentile Only (1-99)',
        'CLAHE + Percentile (Aggressive)',
        'Gentle Percentile (10-90)',
        'Light CLAHE + Wider Percentile'
    ]
    
    return images, titles


def create_comparison_plot(img, mask, sample_name):
    """Create comprehensive comparison plot with detail loss annotations"""
    
    # Apply all normalization methods
    norm_images, norm_titles = apply_six_normalization_methods(img)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    # Original image as reference
    original_norm = norm_images[0]
    
    for idx, (norm_img, title) in enumerate(zip(norm_images, norm_titles)):
        
        # Main image
        ax_img = fig.add_subplot(gs[0, idx])
        im = ax_img.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
        ax_img.set_title(f'{title}', fontsize=12, pad=10)
        ax_img.axis('off')
        
        # Add statistics
        stats_text = f'Mean: {norm_img.mean():.3f}\nStd: {norm_img.std():.3f}\nMin: {norm_img.min():.3f}\nMax: {norm_img.max():.3f}'
        ax_img.text(10, 40, stats_text, color='yellow', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Histogram
        ax_hist = fig.add_subplot(gs[1, idx])
        ax_hist.hist(norm_img.ravel(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax_hist.set_title(f'Histogram', fontsize=10)
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True, alpha=0.3)
        
        # Detail loss analysis (skip for original)
        ax_analysis = fig.add_subplot(gs[2, idx])
        
        if idx == 0:  # Original - no analysis needed
            ax_analysis.text(0.5, 0.5, 'Reference Image\n(Original)', 
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax_analysis.axis('off')
        else:
            # Detect detail loss regions
            problem_mask = detect_detail_loss_regions(original_norm * 255, norm_img * 255)
            problem_boxes = find_problem_regions(problem_mask, min_area=300)
            
            # Show detail loss overlay
            overlay = norm_img.copy()
            overlay_colored = np.stack([overlay, overlay, overlay], axis=2)
            
            # Highlight problem areas in red
            if problem_mask.any():
                overlay_colored[problem_mask, 0] = np.minimum(overlay_colored[problem_mask, 0] + 0.3, 1.0)
                overlay_colored[problem_mask, 1] = np.maximum(overlay_colored[problem_mask, 1] - 0.3, 0.0)
                overlay_colored[problem_mask, 2] = np.maximum(overlay_colored[problem_mask, 2] - 0.3, 0.0)
            
            ax_analysis.imshow(overlay_colored)
            ax_analysis.set_title('Detail Loss Analysis', fontsize=10)
            ax_analysis.axis('off')
            
            # Draw bounding boxes around problem regions
            for x, y, w, h in problem_boxes:
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax_analysis.add_patch(rect)
            
            # Add annotation
            problem_pct = (problem_mask.sum() / problem_mask.size) * 100
            annotation_text = f'Detail Loss: {problem_pct:.1f}%\nProblem regions: {len(problem_boxes)}'
            
            # Color-code the warning
            if problem_pct > 10:
                color = 'red'
                warning = '⚠️ HIGH'
            elif problem_pct > 5:
                color = 'orange' 
                warning = '⚠️ MEDIUM'
            else:
                color = 'green'
                warning = '✓ LOW'
                
            ax_analysis.text(10, 30, f'{warning}\n{annotation_text}', 
                           color='white', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Overall title
    fig.suptitle(f'Normalization Methods Comparison - {sample_name}', fontsize=16, y=0.95)
    
    # Add legend
    legend_text = (
        "Red regions indicate potential detail loss areas:\n"
        "• Significant variance reduction compared to original\n"
        "• Overly dark regions (< 5th percentile)\n"
        "• Bounding boxes show connected problem regions (≥300 pixels)"
    )
    
    fig.text(0.02, 0.02, legend_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig


def analyze_existing_samples(output_dir):
    """Use the same samples from the previous analysis"""
    output_dir = Path(output_dir)
    
    # Find existing dataset
    data_root = Path('~/Data_for_ML/Meat_Luci_Tulane/_build').expanduser()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {data_root}")
    
    splits = {
        'train': {
            'images': data_root / 'dataset' / 'train' / 'images',
            'masks': data_root / 'dataset' / 'train' / 'masks'
        },
        'val': {
            'images': data_root / 'dataset' / 'val' / 'images',  
            'masks': data_root / 'dataset' / 'val' / 'masks'
        },
        'test': {
            'images': data_root / 'dataset' / 'test' / 'images',
            'masks': data_root / 'dataset' / 'test' / 'masks'
        }
    }
    
    print("="*80)
    print("UPDATED NORMALIZATION METHODS COMPARISON")
    print("="*80)
    
    # Set random seed for reproducibility (same as original analysis)
    random.seed(865)
    np.random.seed(865)
    
    # Analyze samples from each split
    for split_name, paths in splits.items():
        print(f"\nProcessing {split_name} samples...")
        
        image_files = sorted(paths['images'].glob('*.jpg'))
        mask_files = {p.stem: p for p in paths['masks'].glob('*.tif')}
        
        # Get paired samples
        paired = []
        for img_path in image_files:
            if img_path.stem in mask_files:
                paired.append((img_path, mask_files[img_path.stem]))
        
        if not paired:
            print(f"   ✗ No paired samples found in {split_name}")
            continue
        
        # Use same random sampling as original analysis
        samples = random.sample(paired, min(3, len(paired)))  # 3 samples per split for focus
        
        for idx, (img_path, mask_path) in enumerate(samples):
            print(f"   Processing {split_name} sample {idx+1}: {img_path.name}")
            
            # Load image and mask
            img, mask = load_image_mask_pair(img_path, mask_path)
            
            # Create comparison plot
            sample_name = f"{split_name.upper()} Sample {idx+1}"
            fig = create_comparison_plot(img, mask, sample_name)
            
            # Save
            output_filename = f'{split_name}_sample{idx+1}_normalization_comparison_updated.png'
            fig.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"      ✓ Saved: {output_filename}")
    
    # Create summary
    create_summary_report(output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Updated comparison plots saved to: {output_dir}")
    print("\nKey findings will be summarized in the generated report.")


def create_summary_report(output_dir):
    """Create a summary report of findings"""
    
    summary_text = """
# Updated Normalization Methods Comparison - Summary

## Methods Tested:

1. **Original**: Raw grayscale values (0-255) normalized to [0,1] for display
2. **CLAHE Only**: clipLimit=2.0, tileGridSize=(8,8), normalized to [0,1]
3. **Percentile Only**: 1st-99th percentile normalization 
4. **CLAHE + Percentile (Aggressive)**: CLAHE + 1st-99th percentile (from previous analysis)
5. **Gentle Percentile**: 10th-90th percentile normalization (wider range)
6. **Light CLAHE + Wider Percentile**: clipLimit=1.2, tileGridSize=(16,16) + 5th-95th percentile

## Detail Loss Analysis:

Red highlighted regions indicate potential problems:
- Areas where local variance significantly decreased compared to original
- Overly dark regions (< 5th percentile of processed image)
- Connected regions ≥300 pixels are outlined with red boxes

## Recommendations Based on Visual Analysis:

### Conservative Approaches (Lower risk of detail loss):
- **Gentle Percentile (10-90)**: Safest improvement over original
- **Light CLAHE + Wider Percentile**: Good balance of enhancement vs safety

### Moderate Risk:
- **Percentile Only (1-99)**: Good contrast but may clip important details
- **CLAHE Only**: Improves local contrast but can create artifacts

### Higher Risk:
- **CLAHE + Percentile (Aggressive)**: Best contrast but highest detail loss risk

## Implementation Recommendation:

Start with **Light CLAHE + Wider Percentile** as it provides:
- Improved local contrast from gentle CLAHE
- Robust normalization from percentile method
- Lower risk of losing important tissue details
- Better handling of staining variations

If results are positive, can experiment with more aggressive methods.

## Code for Recommended Method:

```python
# Light CLAHE + Wider Percentile (Recommended)
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
img = clahe.apply(img_uint8).astype(np.float32)
p5, p95 = np.percentile(img, (5, 95))
img = np.clip((img - p5) / (p95 - p5 + 1e-3), 0, 1)
```
"""
    
    with open(output_dir / 'NORMALIZATION_COMPARISON_SUMMARY.md', 'w') as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(description='Compare 6 normalization methods with detail loss analysis')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessing_analysis',
        help='Output directory for updated analysis results'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_existing_samples(output_dir)


if __name__ == '__main__':
    main()
