#!/usr/bin/env python3
"""
Focused Comparison of Requested Normalization Methods
Compares current z-score vs percentile variations vs mild CLAHE combinations
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
plt.rcParams['figure.figsize'] = (24, 16)


def load_image_mask_pair(img_path, mask_path):
    """Load image and mask pair"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mask = tiff.imread(str(mask_path)).astype(np.float32)
    if mask.ndim == 3:
        mask = mask.squeeze()
    return img, mask


def apply_clahe(img, clip_limit=1.5, tile_grid_size=(12, 12)):
    """Apply Mild CLAHE"""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8).astype(np.float32)


def normalize_zscore(img, global_mean=200.99, global_std=25.26):
    """Current z-score normalization (from analysis summary)"""
    return (img - global_mean) / (global_std + 1e-10)


def normalize_percentile(img, p_low=1, p_high=99):
    """Percentile-based normalization"""
    plow, phigh = np.percentile(img, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    return np.clip((img - plow) / scale, 0, 1)


def apply_requested_methods(img):
    """Apply the 6 requested normalization methods"""
    
    # 1. Current Z-score normalization
    img_zscore = normalize_zscore(img)
    
    # 2. Percentile (0.01-99.99)
    img_perc_01_9999 = normalize_percentile(img, p_low=0.01, p_high=99.99)
    
    # 3. Mild CLAHE + Percentile (0.01-99.99)
    img_clahe_01 = apply_clahe(img)
    img_clahe_perc_01_9999 = normalize_percentile(img_clahe_01, p_low=0.01, p_high=99.99)
    
    # 4. Percentile (0.05-99.95)
    img_perc_05_9995 = normalize_percentile(img, p_low=0.05, p_high=99.95)
    
    # 5. Mild CLAHE + Percentile (0.05-99.95)
    img_clahe_05 = apply_clahe(img)
    img_clahe_perc_05_9995 = normalize_percentile(img_clahe_05, p_low=0.05, p_high=99.95)
    
    # 6. Mild CLAHE + Percentile (0.001-99.999)
    img_clahe_001 = apply_clahe(img)
    img_clahe_perc_001_99999 = normalize_percentile(img_clahe_001, p_low=0.001, p_high=99.999)
    
    images = [
        img_zscore,
        img_perc_01_9999,
        img_clahe_perc_01_9999,
        img_perc_05_9995,
        img_clahe_perc_05_9995,
        img_clahe_perc_001_99999
    ]
    
    titles = [
        'Current Z-score',
        'Percentile (0.01-99.99)',
        'Mild CLAHE + Percentile (0.01-99.99)',
        'Percentile (0.05-99.95)',
        'Mild CLAHE + Percentile (0.05-99.95)',
        'Mild CLAHE + Percentile (0.001-99.999)'
    ]
    
    return images, titles


def create_comparison_plot(img, mask, sample_name):
    """Create comparison plot for requested methods"""
    
    # Apply all normalization methods
    norm_images, norm_titles = apply_requested_methods(img)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, (norm_img, title) in enumerate(zip(norm_images, norm_titles)):
        
        # Main image
        ax_img = fig.add_subplot(gs[0, idx])
        
        # Handle different value ranges for display
        if 'Z-score' in title:
            # Z-score can have negative values, so normalize for display
            display_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min() + 1e-6)
            im = ax_img.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        else:
            # Percentile methods are already in [0,1] range
            im = ax_img.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
            
        ax_img.set_title(f'{title}', fontsize=12, pad=10, weight='bold')
        ax_img.axis('off')
        
        # Add statistics
        stats_text = f'Mean: {norm_img.mean():.3f}\nStd: {norm_img.std():.3f}\nMin: {norm_img.min():.3f}\nMax: {norm_img.max():.3f}'
        ax_img.text(10, 40, stats_text, color='yellow', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Histogram
        ax_hist = fig.add_subplot(gs[1, idx])
        ax_hist.hist(norm_img.ravel(), bins=50, alpha=0.8, color='lightblue', edgecolor='black')
        ax_hist.set_title(f'Histogram', fontsize=10)
        ax_hist.set_xlabel('Intensity', fontsize=9)
        ax_hist.set_ylabel('Frequency', fontsize=9)
        ax_hist.grid(True, alpha=0.3)
        
        # Add method details
        if 'Z-score' in title:
            method_text = 'Current method\n(mean=0, std=1)'
        elif title == 'Percentile (0.01-99.99)':
            method_text = '0.01-99.99th percentile\nClips 0.02% outliers'
        elif title == 'Percentile (0.05-99.95)':
            method_text = '0.05-99.95th percentile\nClips 0.1% outliers'
        elif '0.01-99.99' in title and 'CLAHE' in title:
            method_text = 'Mild CLAHE +\n0.01-99.99th percentile'
        elif '0.05-99.95' in title and 'CLAHE' in title:
            method_text = 'Mild CLAHE +\n0.05-99.95th percentile'
        elif '0.001-99.999' in title:
            method_text = 'Mild CLAHE +\n0.001-99.999th percentile\n(minimal clipping)'
        else:
            method_text = ''
            
        ax_hist.text(0.02, 0.95, method_text, transform=ax_hist.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Overall title
    fig.suptitle(f'Requested Normalization Methods Comparison - {sample_name}', 
                 fontsize=16, y=0.95, weight='bold')
    
    return fig


def analyze_requested_methods(output_dir):
    """Generate comparison of requested normalization methods"""
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
    print("REQUESTED NORMALIZATION METHODS COMPARISON")
    print("="*80)
    print("Methods: Current Z-score | Percentile (0.01-99.99) | Mild CLAHE + (0.01-99.99) | Percentile (0.05-99.95) | Mild CLAHE + (0.05-99.95) | Mild CLAHE + (0.001-99.999)")
    print("Samples: 2 per split (train/val/test) - SAME IMAGES AS PREVIOUS ANALYSES")
    print("="*80)
    
    # Use same random seed as previous analyses for consistency
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
        
        # Use 2 samples per split (same as previous analyses)
        samples = random.sample(paired, min(2, len(paired)))
        
        for idx, (img_path, mask_path) in enumerate(samples):
            print(f"   Processing {split_name} sample {idx+1}: {img_path.name}")
            
            # Load image and mask
            img, mask = load_image_mask_pair(img_path, mask_path)
            
            # Create comparison plot
            sample_name = f"{split_name.upper()} Sample {idx+1}"
            fig = create_comparison_plot(img, mask, sample_name)
            
            # Save
            output_filename = f'{split_name}_sample{idx+1}_requested_comparison.png'
            fig.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"      ✓ Saved: {output_filename}")
    
    # Create summary
    create_requested_summary_report(output_dir)
    
    print("\n" + "="*80)
    print("REQUESTED COMPARISON COMPLETE!")
    print("="*80)
    print(f"Comparison plots saved to: {output_dir}")
    print("\nCompare current z-score vs percentile variations vs CLAHE combinations!")


def create_requested_summary_report(output_dir):
    """Create summary report for requested comparison"""
    
    summary_text = """
# Requested Normalization Methods Comparison - Summary

## 6 Methods Compared:

1. **Current Z-score**: (img - global_mean) / global_std → Output can be negative
2. **Percentile (0.01-99.99)**: Clips 0.02% of outliers → Output [0,1]
3. **Mild CLAHE + Percentile (0.01-99.99)**: Enhancement + clipping → Output [0,1]
4. **Percentile (0.05-99.95)**: Clips 0.1% of outliers → Output [0,1]
5. **Mild CLAHE + Percentile (0.05-99.95)**: Enhancement + moderate clipping → Output [0,1]
6. **Mild CLAHE + Percentile (0.001-99.999)**: Enhancement + minimal clipping → Output [0,1]

## Key Differences:

### **Range and Distribution:**
- **Z-score**: Can have negative values, centered at 0
- **Percentile methods**: Always [0,1] range, preserves relative intensities
- **CLAHE combinations**: Enhanced contrast + [0,1] range

### **Outlier Handling:**
- **0.001-99.999**: Virtually no clipping (keeps 99.998% of data)
- **0.01-99.99**: Minimal clipping (keeps 99.98% of data)
- **0.05-99.95**: Conservative clipping (keeps 99.9% of data)

### **Contrast Enhancement:**
- **Percentile only**: No contrast enhancement, just normalization
- **CLAHE + Percentile**: Local contrast enhancement + robust normalization

## Expected Characteristics:

### **Current Z-score:**
- Familiar method, maintains statistical properties
- Can produce negative values (may affect ReLU activations)
- Sensitive to outliers in mean/std calculation

### **Percentile (0.01-99.99):**
- Very conservative clipping
- Good dynamic range utilization
- Robust to outliers

### **Mild CLAHE + Percentile (0.01-99.99):**
- Enhanced boundaries + conservative normalization
- Best for poor contrast images
- Minimal data loss

### **Percentile (0.05-99.95):**
- Slightly more aggressive outlier clipping
- Good balance of robustness vs data preservation

### **Mild CLAHE + Percentile (0.05-99.95):**
- Enhanced boundaries + balanced normalization
- Likely optimal for most cases
- Good contrast without artifacts

### **Mild CLAHE + Percentile (0.001-99.999):**
- Enhanced boundaries + minimal clipping
- Preserves almost all original data
- Most conservative approach

## Implementation Code:

```python
# Method 1: Current Z-score
img_norm = (img - 200.99) / (25.26 + 1e-10)

# Method 2: Percentile (0.01-99.99)
p_low, p_high = np.percentile(img, (0.01, 99.99))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 3: Mild CLAHE + Percentile (0.01-99.99)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
p_low, p_high = np.percentile(img, (0.01, 99.99))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 4: Percentile (0.05-99.95)
p_low, p_high = np.percentile(img, (0.05, 99.95))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 5: Mild CLAHE + Percentile (0.05-99.95) [RECOMMENDED]
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
p_low, p_high = np.percentile(img, (0.05, 99.95))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 6: Mild CLAHE + Percentile (0.001-99.999)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
p_low, p_high = np.percentile(img, (0.001, 99.999))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)
```

## Selection Criteria:

**Choose based on:**
1. **Visual quality**: Clear adipocyte boundaries without artifacts
2. **Range compatibility**: [0,1] range works better with modern activations
3. **Robustness**: Handles intensity variations across samples
4. **Quantitative consistency**: Produces reliable measurements

**Likely best option**: Method 5 (Mild CLAHE + Percentile 0.05-99.95) balances all factors optimally.
"""
    
    with open(output_dir / 'REQUESTED_METHODS_COMPARISON_SUMMARY.md', 'w') as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(description='Compare requested normalization methods')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessing_analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_requested_methods(output_dir)


if __name__ == '__main__':
    main()
