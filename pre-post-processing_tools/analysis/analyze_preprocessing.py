#!/usr/bin/env python3
"""
Comprehensive Preprocessing Analysis for Adipose Tissue U-Net
Samples and visualizes images to identify preprocessing improvements
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


def normalize_zscore(img, mean=None, std=None):
    """Z-score normalization"""
    if mean is None:
        mean = img.mean()
    if std is None:
        std = img.std()
    return (img - mean) / (std + 1e-10)


def normalize_percentile(img, p_low=1, p_high=99):
    """Percentile-based normalization"""
    plow, phigh = np.percentile(img, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    return np.clip((img - plow) / scale, 0, 1)


def normalize_minmax(img):
    """Min-max normalization"""
    imin, imax = img.min(), img.max()
    scale = max(imax - imin, 1e-3)
    return (img - imin) / scale


def analyze_intensity_distribution(images, titles, split_name):
    """Analyze and plot intensity distributions"""
    fig, axes = plt.subplots(2, len(images), figsize=(5*len(images), 10))
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        # Histogram
        axes[0, idx].hist(img.ravel(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, idx].set_title(f'{title}\nHistogram', fontsize=12)
        axes[0, idx].set_xlabel('Intensity')
        axes[0, idx].set_ylabel('Frequency')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Image
        axes[1, idx].imshow(img, cmap='gray')
        axes[1, idx].set_title(f'{title}\nImage', fontsize=12)
        axes[1, idx].axis('off')
        
        # Add statistics
        stats_text = f'Mean: {img.mean():.2f}\nStd: {img.std():.2f}\nMin: {img.min():.2f}\nMax: {img.max():.2f}'
        axes[1, idx].text(10, 30, stats_text, color='yellow', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    return fig


def analyze_normalization_methods(img, mask):
    """Compare different normalization methods"""
    # Original
    img_original = img.copy()
    
    # Z-score
    img_zscore = normalize_zscore(img)
    
    # Percentile
    img_percentile = normalize_percentile(img)
    
    # MinMax
    img_minmax = normalize_minmax(img)
    
    # CLAHE then normalize
    img_clahe = apply_clahe(img)
    img_clahe_norm = normalize_percentile(img_clahe)
    
    images = [img_original, img_zscore, img_percentile, img_minmax, img_clahe, img_clahe_norm]
    titles = [
        'Original',
        'Z-score',
        'Percentile (1-99)',
        'MinMax',
        'CLAHE',
        'CLAHE + Percentile'
    ]
    
    fig = analyze_intensity_distribution(images, titles, "Normalization Comparison")
    return fig, images, titles


def analyze_stain_consistency(image_paths, n_samples=10):
    """Analyze staining consistency across images"""
    sampled = random.sample(list(image_paths), min(n_samples, len(image_paths)))
    
    means = []
    stds = []
    p1s = []
    p99s = []
    
    for img_path in sampled:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        means.append(img.mean())
        stds.append(img.std())
        p1, p99 = np.percentile(img, (1, 99))
        p1s.append(p1)
        p99s.append(p99)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].bar(range(len(means)), means)
    axes[0, 0].set_title('Mean Intensity Across Images')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].axhline(np.mean(means), color='red', linestyle='--', label=f'Average: {np.mean(means):.2f}')
    axes[0, 0].legend()
    
    axes[0, 1].bar(range(len(stds)), stds)
    axes[0, 1].set_title('Std Dev Across Images')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].axhline(np.mean(stds), color='red', linestyle='--', label=f'Average: {np.mean(stds):.2f}')
    axes[0, 1].legend()
    
    axes[1, 0].bar(range(len(p1s)), p1s, label='1st percentile')
    axes[1, 0].bar(range(len(p99s)), p99s, alpha=0.7, label='99th percentile')
    axes[1, 0].set_title('Percentile Range Across Images')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend()
    
    # Coefficient of variation
    cv_mean = np.std(means) / np.mean(means) * 100
    cv_std = np.std(stds) / np.mean(stds) * 100
    
    stats_text = f'Stain Consistency Analysis:\n\n'
    stats_text += f'Mean CV: {cv_mean:.2f}%\n'
    stats_text += f'Std CV: {cv_std:.2f}%\n\n'
    stats_text += f'Interpretation:\n'
    if cv_mean < 10:
        stats_text += '✓ Good stain consistency\n'
    elif cv_mean < 20:
        stats_text += '⚠ Moderate variability\n'
    else:
        stats_text += '✗ High variability - consider stain normalization\n'
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig, cv_mean


def analyze_mask_quality(mask):
    """Analyze mask quality and characteristics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mask visualization
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Mask')
    axes[0].axis('off')
    
    # Positive pixel ratio
    pos_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
    axes[1].bar(['Negative', 'Positive'], [1-pos_ratio, pos_ratio])
    axes[1].set_title(f'Class Balance\nPositive: {pos_ratio*100:.2f}%')
    axes[1].set_ylabel('Fraction')
    
    # Connected components analysis
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        axes[2].hist(areas, bins=30, edgecolor='black')
        axes[2].set_title(f'Component Size Distribution\n{num_labels-1} components')
        axes[2].set_xlabel('Area (pixels)')
        axes[2].set_ylabel('Count')
        axes[2].set_yscale('log')
    else:
        axes[2].text(0.5, 0.5, 'No positive regions', ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig, pos_ratio, num_labels-1


def sample_and_analyze(data_root, output_dir, n_samples=5):
    """Main analysis function"""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ADIPOSE TISSUE U-NET PREPROCESSING ANALYSIS")
    print("="*80)
    
    # Dataset paths
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
    
    # Compute global statistics
    print("\n1. Computing global dataset statistics...")
    all_train_images = sorted(splits['train']['images'].glob('*.jpg'))
    
    train_means = []
    train_stds = []
    for img_path in all_train_images[:100]:  # Sample 100 for speed
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        train_means.append(img.mean())
        train_stds.append(img.std())
    
    global_mean = np.mean(train_means)
    global_std = np.mean(train_stds)
    
    print(f"   Global mean: {global_mean:.2f}")
    print(f"   Global std: {global_std:.2f}")
    
    # Stain consistency analysis
    print("\n2. Analyzing stain consistency...")
    fig_stain, cv_mean = analyze_stain_consistency(all_train_images, n_samples=20)
    fig_stain.savefig(output_dir / 'stain_consistency.png', dpi=150, bbox_inches='tight')
    plt.close(fig_stain)
    print(f"   ✓ Saved stain consistency plot (CV: {cv_mean:.2f}%)")
    
    # Sample and analyze each split
    for split_name, paths in splits.items():
        print(f"\n3. Analyzing {split_name} set...")
        
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
        
        # Random samples
        samples = random.sample(paired, min(n_samples, len(paired)))
        
        print(f"   Found {len(paired)} paired samples, analyzing {len(samples)}...")
        
        # Analyze each sample
        for idx, (img_path, mask_path) in enumerate(samples):
            print(f"   Processing sample {idx+1}/{len(samples)}: {img_path.name}")
            
            # Load
            img, mask = load_image_mask_pair(img_path, mask_path)
            
            # Normalization comparison
            fig_norm, norm_images, norm_titles = analyze_normalization_methods(img, mask)
            fig_norm.suptitle(f'{split_name.upper()} - Sample {idx+1} - Normalization Methods', 
                             fontsize=16, y=1.00)
            fig_norm.savefig(output_dir / f'{split_name}_sample{idx+1}_normalization.png', 
                           dpi=150, bbox_inches='tight')
            plt.close(fig_norm)
            
            # Mask quality
            fig_mask, pos_ratio, n_components = analyze_mask_quality(mask)
            fig_mask.suptitle(f'{split_name.upper()} - Sample {idx+1} - Mask Quality', 
                            fontsize=16)
            fig_mask.savefig(output_dir / f'{split_name}_sample{idx+1}_mask.png', 
                           dpi=150, bbox_inches='tight')
            plt.close(fig_mask)
            
            print(f"      Positive ratio: {pos_ratio*100:.2f}%, Components: {n_components}")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Based on stain consistency
    if cv_mean > 15:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Stain Normalization',
            'issue': f'High staining variability detected (CV: {cv_mean:.2f}%)',
            'recommendation': 'Implement Macenko or Reinhard stain normalization to reduce batch effects'
        })
    elif cv_mean > 10:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Stain Normalization',
            'issue': f'Moderate staining variability (CV: {cv_mean:.2f}%)',
            'recommendation': 'Consider per-image adaptive normalization or CLAHE'
        })
    
    # Normalization method
    recommendations.append({
        'priority': 'HIGH',
        'category': 'Normalization',
        'issue': 'Current z-score normalization may not handle intensity variations optimally',
        'recommendation': 'Test CLAHE + percentile normalization for better contrast and robustness'
    })
    
    # Augmentation
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Augmentation',
        'issue': 'Dataset size: 673 training tiles',
        'recommendation': 'Current moderate augmentation is appropriate. Consider heavy augmentation if overfitting occurs'
    })
    
    # Class balance
    recommendations.append({
        'priority': 'LOW',
        'category': 'Class Balance',
        'issue': '40% negative tiles - good balance',
        'recommendation': 'Current negative sampling ratio is good. Monitor for class imbalance during training'
    })
    
    # Write recommendations to file
    with open(output_dir / 'RECOMMENDATIONS.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("PREPROCESSING RECOMMENDATIONS FOR ADIPOSE TISSUE U-NET\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset Statistics:\n")
        f.write(f"  - Global mean: {global_mean:.2f}\n")
        f.write(f"  - Global std: {global_std:.2f}\n")
        f.write(f"  - Stain CV: {cv_mean:.2f}%\n\n")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. [{rec['priority']}] {rec['category']}\n")
            f.write(f"   Issue: {rec['issue']}\n")
            f.write(f"   Recommendation: {rec['recommendation']}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("IMPLEMENTATION SUGGESTIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Immediate Improvements (Easy to implement):\n")
        f.write("   - Test CLAHE preprocessing before normalization\n")
        f.write("   - Try percentile normalization instead of z-score\n")
        f.write("   - Add test-time augmentation for inference\n\n")
        
        f.write("2. Medium-term Improvements:\n")
        f.write("   - Implement stain normalization if CV > 15%\n")
        f.write("   - Add morphological post-processing to predictions\n")
        f.write("   - Experiment with different loss function weights\n\n")
        
        f.write("3. Advanced Techniques:\n")
        f.write("   - Multi-scale input pyramids\n")
        f.write("   - Attention mechanisms for boundary refinement\n")
        f.write("   - Ensemble predictions from multiple checkpoints\n")
    
    # Print to console
    print("\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. [{rec['priority']}] {rec['category']}")
        print(f"   {rec['recommendation']}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze preprocessing for adipose tissue dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        default='~/Data_for_ML/Meat_Luci_Tulane/_build',
        help='Root directory of built dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preprocessing_analysis',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of samples to analyze per split'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root).expanduser()
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    sample_and_analyze(data_root, args.output_dir, args.n_samples)


if __name__ == '__main__':
    main()
