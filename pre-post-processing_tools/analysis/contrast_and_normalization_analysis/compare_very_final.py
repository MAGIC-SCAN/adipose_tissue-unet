#!/usr/bin/env python3
"""
Very Final Normalization Methods Comparison for Adipose Tissue U-Net
Last comparison: Original, CLAHE, Mild CLAHE, and Mild CLAHE + percentile variations
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
plt.rcParams['figure.figsize'] = (20, 12)


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


def apply_very_final_methods(img):
    """Apply the very final 5 normalization methods"""
    
    # 1. Original (raw grayscale 0-255, display as 0-1)
    img_original = img / 255.0
    
    # 2. CLAHE (standard settings)
    img_clahe = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img_clahe = img_clahe / 255.0  # Normalize to [0,1] for display
    
    # 3. Mild CLAHE (lower clip limit, larger tiles)
    img_mild_clahe = apply_clahe(img, clip_limit=1.5, tile_grid_size=(12, 12))
    img_mild_clahe = img_mild_clahe / 255.0  # Normalize to [0,1] for display
    
    # 4. Mild CLAHE + Percentile (0.05-99.95)
    img_mild_clahe_raw1 = apply_clahe(img, clip_limit=1.5, tile_grid_size=(12, 12))
    img_mild_clahe_perc1 = normalize_percentile(img_mild_clahe_raw1, p_low=0.05, p_high=99.95)
    
    # 5. Mild CLAHE + Percentile (0.01-99.99)
    img_mild_clahe_raw2 = apply_clahe(img, clip_limit=1.5, tile_grid_size=(12, 12))
    img_mild_clahe_perc2 = normalize_percentile(img_mild_clahe_raw2, p_low=0.01, p_high=99.99)
    
    images = [
        img_original,
        img_clahe, 
        img_mild_clahe,
        img_mild_clahe_perc1,
        img_mild_clahe_perc2
    ]
    
    titles = [
        'Original',
        'CLAHE',
        'Mild CLAHE',
        'Mild CLAHE + Percentile (0.05-99.95)',
        'Mild CLAHE + Percentile (0.01-99.99)'
    ]
    
    return images, titles


def create_comparison_plot(img, mask, sample_name):
    """Create clean comparison plot for very final methods"""
    
    # Apply all normalization methods
    norm_images, norm_titles = apply_very_final_methods(img)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.2)
    
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
        ax_hist.hist(norm_img.ravel(), bins=50, alpha=0.8, color='lightgreen', edgecolor='black')
        ax_hist.set_title(f'Histogram', fontsize=12)
        ax_hist.set_xlabel('Intensity', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.grid(True, alpha=0.3)
        
        # Add method details
        if 'Original' in title:
            method_text = 'Raw [0-255] → [0-1]'
        elif title == 'CLAHE':
            method_text = 'clipLimit=2.0\ntileGrid=(8,8)'
        elif title == 'Mild CLAHE':
            method_text = 'clipLimit=1.5\ntileGrid=(12,12)'
        elif '0.05-99.95' in title:
            method_text = 'Mild CLAHE +\n0.05-99.95th percentile'
        elif '0.01-99.99' in title:
            method_text = 'Mild CLAHE +\n0.01-99.99th percentile'
        else:
            method_text = ''
            
        ax_hist.text(0.02, 0.95, method_text, transform=ax_hist.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # Overall title
    fig.suptitle(f'Very Final Normalization Methods Comparison - {sample_name}', 
                 fontsize=18, y=0.95, weight='bold')
    
    return fig


def analyze_very_final_samples(output_dir):
    """Generate very final normalization comparison with new sample selection"""
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
    print("VERY FINAL NORMALIZATION METHODS COMPARISON")
    print("="*80)
    print("Methods: Original | CLAHE | Mild CLAHE | Mild CLAHE + Percentile (0.05-99.95) | Mild CLAHE + Percentile (0.01-99.99)")
    print("Samples: 2 per split (train/val/test) - NEW RANDOM SELECTION")
    print("="*80)
    
    # Set same random seed as previous analyses for consistency
    random.seed(865)  # Same seed as previous analyses
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
        
        # Use 2 NEW samples per split
        samples = random.sample(paired, min(2, len(paired)))
        
        for idx, (img_path, mask_path) in enumerate(samples):
            print(f"   Processing {split_name} sample {idx+1}: {img_path.name}")
            
            # Load image and mask
            img, mask = load_image_mask_pair(img_path, mask_path)
            
            # Create comparison plot
            sample_name = f"{split_name.upper()} Sample {idx+1}"
            fig = create_comparison_plot(img, mask, sample_name)
            
            # Save
            output_filename = f'{split_name}_sample{idx+1}_very_final_comparison.png'
            fig.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"      ✓ Saved: {output_filename}")
    
    # Create summary
    create_very_final_summary_report(output_dir)
    
    print("\n" + "="*80)
    print("VERY FINAL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Very final comparison plots saved to: {output_dir}")
    print("\nThis is your absolute final set of normalization options!")


def create_very_final_summary_report(output_dir):
    """Create very final summary report"""
    
    summary_text = """
# Very Final Normalization Methods Comparison - Summary

## Very Final 5 Methods:

1. **Original**: Raw grayscale values (0-255) normalized to [0,1] for display
2. **CLAHE**: Standard CLAHE (clipLimit=2.0, tileGridSize=(8,8))
3. **Mild CLAHE**: Gentle CLAHE (clipLimit=1.5, tileGridSize=(12,12))
4. **Mild CLAHE + Percentile (0.05-99.95)**: Conservative combination
5. **Mild CLAHE + Percentile (0.01-99.99)**: Very conservative combination

## Method Focus:

This final comparison focuses on **mild CLAHE variations** since you preferred CLAHE but wanted to avoid artifacts:

### CLAHE Settings:
- **Standard CLAHE**: More aggressive enhancement (clipLimit=2.0, smaller tiles)
- **Mild CLAHE**: Gentler enhancement (clipLimit=1.5, larger tiles)

### Percentile Range Impact:
- **0.05-99.95**: Clips 0.1% of outliers (very conservative)
- **0.01-99.99**: Clips 0.02% of outliers (extremely conservative)

## Key Differences to Evaluate:

1. **Artifact reduction**: Mild CLAHE should have fewer enhancement artifacts
2. **Percentile sensitivity**: Compare 0.05-99.95 vs 0.01-99.99 clipping
3. **Boundary clarity**: Essential for adipocyte segmentation
4. **Dynamic range**: How well each method uses [0,1] range

## Implementation Code:

```python
# Method 2: CLAHE
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img_uint8).astype(np.float32) / 255.0

# Method 3: Mild CLAHE
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img_uint8).astype(np.float32) / 255.0

# Method 4: Mild CLAHE + Percentile (0.05-99.95)
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img_uint8).astype(np.float32)
p_low, p_high = np.percentile(img, (0.05, 99.95))
img = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 5: Mild CLAHE + Percentile (0.01-99.99) [MOST CONSERVATIVE]
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img_uint8).astype(np.float32)
p_low, p_high = np.percentile(img, (0.01, 99.99))
img = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)
```

## Final Selection Criteria:

Choose the method that gives you:
- **Clear adipocyte boundaries** without artifacts
- **Good contrast** between tissue and background  
- **Minimal enhancement artifacts** (dark spots, unnatural contrast)
- **Robust performance** across different tissue samples

The **Mild CLAHE + Percentile (0.05-99.95)** combination is likely optimal, 
balancing enhancement with robustness for medical imaging applications.
"""
    
    with open(output_dir / 'VERY_FINAL_NORMALIZATION_SUMMARY.md', 'w') as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(description='Very final normalization methods comparison')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessing_analysis',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_very_final_samples(output_dir)


if __name__ == '__main__':
    main()
