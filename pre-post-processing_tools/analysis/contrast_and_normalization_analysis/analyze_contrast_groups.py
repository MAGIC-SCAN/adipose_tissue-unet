#!/usr/bin/env python3
"""
Analyze Sample Images for Contrast-Based Grouping
Determines cutoffs for adaptive CLAHE application based on image quality metrics
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
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def load_image_mask_pair(img_path, mask_path):
    """Load image and mask pair"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mask = tiff.imread(str(mask_path)).astype(np.float32)
    if mask.ndim == 3:
        mask = mask.squeeze()
    return img, mask


def calculate_image_quality_metrics(img):
    """Calculate comprehensive image quality metrics"""
    
    # Basic statistics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    
    # Contrast metrics
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    dynamic_range = img.max() - img.min()
    coefficient_variation = std_intensity / mean_intensity * 100
    
    # Sharpness metrics
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    laplacian_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    
    # Local contrast analysis
    # Calculate local standard deviation using sliding window
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_var = cv2.filter2D(img**2, -1, kernel) - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    avg_local_contrast = np.mean(local_std)
    std_local_contrast = np.std(local_std)
    local_contrast_variation = std_local_contrast / (avg_local_contrast + 1e-6)
    
    # Histogram analysis
    hist, bins = np.histogram(img, bins=256, range=(0, 255))
    hist_normalized = hist / np.sum(hist)
    
    # Entropy (measure of information content)
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    
    # Peak-to-valley ratio in histogram
    hist_1d = hist.astype(np.float32).reshape(-1, 1)
    hist_smooth = cv2.GaussianBlur(hist_1d, (1, 5), 1.0).flatten()
    peaks = []
    for i in range(1, len(hist_smooth)-1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            peaks.append(hist_smooth[i])
    
    peak_prominence = max(peaks) / (np.mean(hist_smooth) + 1e-6) if peaks else 0
    
    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'contrast_ratio': contrast_ratio,
        'dynamic_range': dynamic_range,
        'coefficient_variation': coefficient_variation,
        'laplacian_variance': laplacian_var,
        'avg_local_contrast': avg_local_contrast,
        'local_contrast_variation': local_contrast_variation,
        'entropy': entropy,
        'peak_prominence': peak_prominence
    }


def analyze_sample_images():
    """Analyze all sample images from preprocessing analysis"""
    
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
    
    # Use same random seed as previous analyses for consistency
    random.seed(865)
    np.random.seed(865)
    
    all_metrics = []
    
    print("="*80)
    print("ANALYZING SAMPLE IMAGES FOR CONTRAST GROUPING")
    print("="*80)
    
    for split_name, paths in splits.items():
        print(f"\nAnalyzing {split_name} samples...")
        
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
        
        # Use same 2 samples per split as previous analyses
        samples = random.sample(paired, min(2, len(paired)))
        
        for idx, (img_path, mask_path) in enumerate(samples):
            print(f"   Analyzing {split_name} sample {idx+1}: {img_path.name}")
            
            # Load image
            img, mask = load_image_mask_pair(img_path, mask_path)
            
            # Calculate metrics
            metrics = calculate_image_quality_metrics(img)
            metrics['split'] = split_name
            metrics['sample_id'] = f"{split_name}_sample{idx+1}"
            metrics['filename'] = img_path.name
            
            all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)


def determine_grouping_criteria(df):
    """Determine optimal grouping criteria and cutoffs"""
    
    print("\n" + "="*80)
    print("IMAGE QUALITY METRICS ANALYSIS")
    print("="*80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    key_metrics = ['contrast_ratio', 'laplacian_variance', 'avg_local_contrast', 'entropy']
    print(df[key_metrics].describe())
    
    # Calculate percentiles for cutoff determination
    print("\nKey Metrics Percentiles:")
    print("-" * 50)
    for metric in key_metrics:
        values = df[metric]
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        print(f"{metric:20s}: 25%={p25:.3f}, 50%={p50:.3f}, 75%={p75:.3f}")
    
    # Determine cutoffs based on analysis
    contrast_ratio_values = df['contrast_ratio'].values
    laplacian_var_values = df['laplacian_variance'].values
    local_contrast_values = df['avg_local_contrast'].values
    
    # Conservative cutoffs based on data distribution
    poor_contrast_cutoff = np.percentile(contrast_ratio_values, 33)  # Bottom third
    good_contrast_cutoff = np.percentile(contrast_ratio_values, 67)  # Top third
    
    poor_sharpness_cutoff = np.percentile(laplacian_var_values, 33)
    good_sharpness_cutoff = np.percentile(laplacian_var_values, 67)
    
    poor_local_contrast_cutoff = np.percentile(local_contrast_values, 33)
    good_local_contrast_cutoff = np.percentile(local_contrast_values, 67)
    
    print(f"\nProposed Cutoffs:")
    print("-" * 50)
    print(f"Contrast Ratio:")
    print(f"  Poor (needs CLAHE): < {poor_contrast_cutoff:.3f}")
    print(f"  Medium (needs mild CLAHE): {poor_contrast_cutoff:.3f} - {good_contrast_cutoff:.3f}")
    print(f"  Good (percentile only): > {good_contrast_cutoff:.3f}")
    
    print(f"\nSharpness (Laplacian Variance):")
    print(f"  Poor: < {poor_sharpness_cutoff:.1f}")
    print(f"  Medium: {poor_sharpness_cutoff:.1f} - {good_sharpness_cutoff:.1f}")
    print(f"  Good: > {good_sharpness_cutoff:.1f}")
    
    return {
        'contrast_ratio': {
            'poor_cutoff': poor_contrast_cutoff,
            'good_cutoff': good_contrast_cutoff
        },
        'laplacian_variance': {
            'poor_cutoff': poor_sharpness_cutoff,
            'good_cutoff': good_sharpness_cutoff
        },
        'avg_local_contrast': {
            'poor_cutoff': poor_local_contrast_cutoff,
            'good_cutoff': good_local_contrast_cutoff
        }
    }


def classify_images(df, cutoffs):
    """Classify images into quality groups"""
    
    def classify_image(row):
        contrast = row['contrast_ratio']
        sharpness = row['laplacian_variance']
        
        # Primary decision based on contrast ratio
        if contrast < cutoffs['contrast_ratio']['poor_cutoff']:
            return "Poor Quality (Needs CLAHE)"
        elif contrast > cutoffs['contrast_ratio']['good_cutoff']:
            # Check if it's really good quality
            if sharpness > cutoffs['laplacian_variance']['good_cutoff']:
                return "Good Quality (Percentile Only)"
            else:
                return "Medium Quality (Mild CLAHE)"
        else:
            return "Medium Quality (Mild CLAHE)"
    
    df['quality_group'] = df.apply(classify_image, axis=1)
    
    print(f"\nImage Classification Results:")
    print("-" * 50)
    print(df[['sample_id', 'contrast_ratio', 'laplacian_variance', 'quality_group']].to_string(index=False))
    
    print(f"\nGroup Distribution:")
    print("-" * 50)
    print(df['quality_group'].value_counts())
    
    return df


def create_adaptive_clahe_function(cutoffs, num_samples):
    """Generate adaptive CLAHE function based on determined cutoffs"""
    
    function_code = f"""
def adaptive_clahe_normalization(img):
    \"\"\"
    Adaptive CLAHE normalization based on image quality analysis
    Cutoffs determined from preprocessing analysis of {num_samples} sample images
    \"\"\"
    # Calculate image quality metrics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Determine preprocessing strategy
    if contrast_ratio < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}:
        # Poor quality - needs aggressive CLAHE
        print(f"Poor quality image (contrast_ratio={{contrast_ratio:.3f}}) - applying CLAHE")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img.astype(np.uint8)).astype(np.float32)
        p5, p95 = np.percentile(img_enhanced, (5, 95))
        return np.clip((img_enhanced - p5) / (p95 - p5 + 1e-3), 0, 1)
        
    elif contrast_ratio > {cutoffs['contrast_ratio']['good_cutoff']:.3f} and sharpness > {cutoffs['laplacian_variance']['good_cutoff']:.1f}:
        # Good quality - percentile normalization only
        print(f"Good quality image (contrast_ratio={{contrast_ratio:.3f}}, sharpness={{sharpness:.1f}}) - percentile only")
        p2, p98 = np.percentile(img, (2, 98))
        return np.clip((img - p2) / (p98 - p2 + 1e-3), 0, 1)
        
    else:
        # Medium quality - mild CLAHE
        print(f"Medium quality image (contrast_ratio={{contrast_ratio:.3f}}) - applying mild CLAHE")
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
        img_enhanced = clahe.apply(img.astype(np.uint8)).astype(np.float32)
        p5, p95 = np.percentile(img_enhanced, (5, 95))
        return np.clip((img_enhanced - p5) / (p95 - p5 + 1e-3), 0, 1)
"""
    
    return function_code


def create_visualization(df):
    """Create visualization of image quality metrics and grouping"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Contrast ratio vs laplacian variance
    scatter = axes[0, 0].scatter(df['contrast_ratio'], df['laplacian_variance'], 
                                c=df['quality_group'].astype('category').cat.codes, 
                                cmap='viridis', alpha=0.7, s=100)
    axes[0, 0].set_xlabel('Contrast Ratio (std/mean)')
    axes[0, 0].set_ylabel('Laplacian Variance (sharpness)')
    axes[0, 0].set_title('Image Quality Metrics')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add sample labels
    for idx, row in df.iterrows():
        axes[0, 0].annotate(row['sample_id'], 
                           (row['contrast_ratio'], row['laplacian_variance']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Contrast ratio distribution
    df['quality_group'].value_counts().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Quality Group Distribution')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Histogram of contrast ratios
    axes[1, 0].hist(df['contrast_ratio'], bins=8, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Contrast Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Contrast Ratio Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of sharpness
    axes[1, 1].hist(df['laplacian_variance'], bins=8, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Laplacian Variance (Sharpness)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Sharpness Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    output_dir = Path('preprocessing_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze sample images
    df = analyze_sample_images()
    
    # Determine grouping criteria
    cutoffs = determine_grouping_criteria(df)
    
    # Classify images
    df_classified = classify_images(df, cutoffs)
    
    # Create adaptive function
    adaptive_function_code = create_adaptive_clahe_function(cutoffs, len(df_classified))
    
    # Create visualization
    fig = create_visualization(df_classified)
    fig.savefig(output_dir / 'contrast_analysis_grouping.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save results
    df_classified.to_csv(output_dir / 'image_quality_analysis.csv', index=False)
    
    # Save adaptive function
    with open(output_dir / 'adaptive_clahe_function.py', 'w') as f:
        f.write("import numpy as np\nimport cv2\n\n")
        f.write(adaptive_function_code)
    
    # Create summary report
    summary_text = f"""
# Image Quality Analysis and Adaptive CLAHE Cutoffs

## Analysis Results

Based on analysis of {len(df_classified)} sample images from your preprocessing analysis:

### Determined Cutoffs:

**Contrast Ratio (std/mean):**
- Poor Quality (needs CLAHE): < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}
- Medium Quality (needs mild CLAHE): {cutoffs['contrast_ratio']['poor_cutoff']:.3f} - {cutoffs['contrast_ratio']['good_cutoff']:.3f}
- Good Quality (percentile only): > {cutoffs['contrast_ratio']['good_cutoff']:.3f}

**Sharpness (Laplacian Variance):**
- Poor: < {cutoffs['laplacian_variance']['poor_cutoff']:.1f}
- Medium: {cutoffs['laplacian_variance']['poor_cutoff']:.1f} - {cutoffs['laplacian_variance']['good_cutoff']:.1f}
- Good: > {cutoffs['laplacian_variance']['good_cutoff']:.1f}

### Image Classification:

{df_classified['quality_group'].value_counts().to_string()}

### Adaptive Strategy:

1. **Poor Quality Images**: Apply standard CLAHE (clipLimit=2.0, tileGridSize=(8,8)) + percentile normalization
2. **Medium Quality Images**: Apply mild CLAHE (clipLimit=1.5, tileGridSize=(12,12)) + percentile normalization  
3. **Good Quality Images**: Skip CLAHE, use percentile normalization only

### Implementation:

The adaptive function has been generated in `adaptive_clahe_function.py` and can be directly integrated into your preprocessing pipeline.

### Files Generated:

- `contrast_analysis_grouping.png`: Visualization of quality metrics and grouping
- `image_quality_analysis.csv`: Detailed metrics for all analyzed images
- `adaptive_clahe_function.py`: Ready-to-use adaptive function

This adaptive approach should provide optimal preprocessing for each image quality level while avoiding over-enhancement of already good quality images.
"""
    
    with open(output_dir / 'CONTRAST_GROUPING_ANALYSIS.md', 'w') as f:
        f.write(summary_text)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nFiles generated:")
    print("- contrast_analysis_grouping.png")
    print("- image_quality_analysis.csv") 
    print("- adaptive_clahe_function.py")
    print("- CONTRAST_GROUPING_ANALYSIS.md")


if __name__ == '__main__':
    main()
