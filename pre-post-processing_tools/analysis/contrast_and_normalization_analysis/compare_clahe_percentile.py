#!/usr/bin/env python3
"""
Focused CLAHE + Percentile Comparison for Adipose Tissue U-Net
Compares original vs CLAHE vs percentile combinations
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
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (24, 12)


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


def apply_six_methods(img):
    """Apply all 6 normalization methods"""
    
    # 1. Original (raw grayscale 0-255, display as 0-1)
    img_original = img / 255.0
    
    # 2. CLAHE only
    img_clahe_only = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img_clahe_only = img_clahe_only / 255.0  # Normalize to [0,1] for display
    
    # 3. Percentile (0.5-99.5) - very gentle
    img_percentile_gentle = normalize_percentile(img, p_low=0.5, p_high=99.5)
    
    # 4. CLAHE + Percentile (0.5-99.5)
    img_clahe_gentle = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img_clahe_perc_gentle = normalize_percentile(img_clahe_gentle, p_low=0.5, p_high=99.5)
    
    # 5. Percentile (0.2-99.8) - more aggressive
    img_percentile_agg = normalize_percentile(img, p_low=0.2, p_high=99.8)
    
    # 6. CLAHE + Percentile (0.2-99.8)
    img_clahe_agg = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img_clahe_perc_agg = normalize_percentile(img_clahe_agg, p_low=0.2, p_high=99.8)
    
    images = [
        img_original,
        img_clahe_only, 
        img_percentile_gentle,
        img_clahe_perc_gentle,
        img_percentile_agg,
        img_clahe_perc_agg
    ]
    
    titles = [
        'Original',
        'CLAHE Only',
        'Percentile (0.5-99.5)',
        'CLAHE + Percentile (0.5-99.5)',
        'Percentile (0.2-99.8)',
        'CLAHE + Percentile (0.2-99.8)'
    ]
    
    return images, titles


def create_comparison_plot(img, mask, sample_name):
    """Create clean comparison plot without detail loss analysis"""
    
    # Apply all normalization methods
    norm_images, norm_titles = apply_six_methods(img)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, (norm_img, title) in enumerate(zip(norm_images, norm_titles)):
        
        # Main image
        ax_img = fig.add_subplot(gs[0, idx])
        im = ax_img.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
        ax_img.set_title(f'{title}', fontsize=14, pad=10, weight='bold')
        ax_img.axis('off')
        
        # Add statistics
        stats_text = f'Mean: {norm_img.mean():.3f}\nStd: {norm_img.std():.3f}\nMin: {norm_img.min():.3f}\nMax: {norm_img.max():.3f}'
        ax_img.text(10, 40, stats_text, color='yellow', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Histogram
        ax_hist = fig.add_subplot(gs[1, idx])
        ax_hist.hist(norm_img.ravel(), bins=50, alpha=0.8, color='skyblue', edgecolor='black')
        ax_hist.set_title(f'Histogram', fontsize=12)
        ax_hist.set_xlabel('Intensity', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.grid(True, alpha=0.3)
        
        # Add range information
        if 'Original' in title:
            range_text = 'Raw [0-255] → [0-1]'
        elif 'CLAHE Only' in title:
            range_text = 'CLAHE → [0-1]'
        elif '0.5-99.5' in title:
            range_text = '0.5-99.5th percentile'
        elif '0.2-99.8' in title:
            range_text = '0.2-99.8th percentile'
        else:
            range_text = ''
            
        ax_hist.text(0.02, 0.95, range_text, transform=ax_hist.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Overall title
    fig.suptitle(f'CLAHE + Percentile Methods Comparison - {sample_name}', 
                 fontsize=18, y=0.95, weight='bold')
    
    return fig


def analyze_focused_samples(output_dir):
    """Generate focused CLAHE + percentile comparison"""
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
    print("FOCUSED CLAHE + PERCENTILE COMPARISON")
    print("="*80)
    print("Methods: Original | CLAHE Only | Percentile (0.5-99.5) | CLAHE+Percentile (0.5-99.5) | Percentile (0.2-99.8) | CLAHE+Percentile (0.2-99.8)")
    print("Samples: 2 per split (train/val/test)")
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
        
        # Use 2 samples per split
        samples = random.sample(paired, min(2, len(paired)))
        
        for idx, (img_path, mask_path) in enumerate(samples):
            print(f"   Processing {split_name} sample {idx+1}: {img_path.name}")
            
            # Load image and mask
            img, mask = load_image_mask_pair(img_path, mask_path)
            
            # Create comparison plot
            sample_name = f"{split_name.upper()} Sample {idx+1}"
            fig = create_comparison_plot(img, mask, sample_name)
            
            # Save
            output_filename = f'{split_name}_sample{idx+1}_clahe_percentile_comparison.png'
            fig.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"      ✓ Saved: {output_filename}")
    
    # Create summary
    create_summary_report(output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"CLAHE + percentile comparison plots saved to: {output_dir}")
    print("\nReview the generated plots to select your preferred normalization approach.")


def create_summary_report(output_dir):
    """Create a summary report of the focused analysis"""
    
    summary_text = """
# CLAHE + Percentile Methods Comparison - Summary

## Methods Compared:

1. **Original**: Raw grayscale values (0-255) normalized to [0,1] for display
2. **CLAHE Only**: clipLimit=2.0, tileGridSize=(8,8), normalized to [0,1]
3. **Percentile (0.5-99.5)**: Very gentle percentile normalization (keeps most data)
4. **CLAHE + Percentile (0.5-99.5)**: CLAHE followed by gentle percentile normalization
5. **Percentile (0.2-99.8)**: More aggressive percentile normalization
6. **CLAHE + Percentile (0.2-99.8)**: CLAHE followed by more aggressive percentile normalization

## What to Look For:

### Visual Quality Assessment:
- **Adipocyte boundary clarity**: Are cell edges well-defined?
- **Background contrast**: Can you distinguish tissue from background?
- **Detail preservation**: Are fine structures maintained?
- **Artifact presence**: Any unnatural enhancement or dark spots?

### Histogram Analysis:
- **Distribution shape**: Bell curve vs skewed vs bimodal
- **Dynamic range**: How well does the method use [0,1] range?
- **Outlier handling**: Are extreme values clipped appropriately?

## Expected Characteristics:

### CLAHE Only:
- Improves local contrast significantly
- May create some enhancement artifacts
- Good for boundary detection

### Gentle Percentile (0.5-99.5):
- Very conservative normalization
- Minimal data loss
- May not improve contrast much

### Aggressive Percentile (0.2-99.8):
- More contrast enhancement
- Some outlier clipping
- Better dynamic range utilization

### CLAHE + Percentile Combinations:
- Best of both: local contrast + robust normalization
- Should handle staining variations well
- May be optimal for segmentation

## Implementation:

Once you select your preferred method, update the normalization in `adipose_unet_2.py`:

```python
# Replace the current z-score normalization with your chosen method
# Example for CLAHE + Percentile (0.5-99.5):

img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img_uint8).astype(np.float32)
p_low, p_high = np.percentile(img, (0.5, 99.5))
img = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)
```

## Next Steps:

1. Review all 6 comparison plots
2. Select the method that provides the best visual quality
3. Test the chosen method in training to validate improvement
4. Consider test-time augmentation for further gains
"""
    
    with open(output_dir / 'CLAHE_PERCENTILE_COMPARISON_SUMMARY.md', 'w') as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(description='Focused CLAHE + percentile methods comparison')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessing_analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_focused_samples(output_dir)


if __name__ == '__main__':
    main()
