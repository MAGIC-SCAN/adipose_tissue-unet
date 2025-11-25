#!/usr/bin/env python3
"""
Comprehensive Normalization Analysis for Adipose Tissue U-Net
Compares dataset tiles to adipocyte references across 4 normalization methods
Uses 6 quality metrics relevant to segmentation performance
"""

import os
import sys
from pathlib import Path
import random
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)


def calculate_comprehensive_metrics(img, method_name=""):
    """Calculate comprehensive image quality metrics for segmentation"""
    
    # Ensure image is in [0,1] range for consistent metrics
    if img.max() > 1.1:  # Likely [0,255] range
        img = img / 255.0
    
    # Basic statistics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    
    # 1. Contrast Ratio (primary grouping metric)
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    # 2. Laplacian Variance (sharpness/blur detection)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    laplacian_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    
    # 3. Entropy (information content)
    hist, _ = np.histogram(img, bins=256, range=(0, 1))
    hist_normalized = hist / (np.sum(hist) + 1e-10)
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    
    # 4. Edge Density (critical for boundary detection)
    edges = cv2.Canny(img_uint8, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 5. Dynamic Range Utilization
    dynamic_range = img.max() - img.min()
    
    # 6. Local Contrast Consistency
    # Calculate local standard deviation consistency
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_var = cv2.filter2D(img**2, -1, kernel) - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    local_contrast_consistency = 1.0 / (np.std(local_std) + 1e-6)  # Higher = more consistent
    
    return {
        'method': method_name,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'contrast_ratio': contrast_ratio,
        'laplacian_variance': laplacian_var,
        'entropy': entropy,
        'edge_density': edge_density,
        'dynamic_range': dynamic_range,
        'local_contrast_consistency': local_contrast_consistency
    }


def apply_zscore_normalization(img, global_mean=200.99, global_std=25.26):
    """Current z-score normalization"""
    return (img - global_mean) / (global_std + 1e-10)


def apply_clahe_percentile(img, clip_limit=2.0, tile_grid_size=(8, 8), p_low=0.01, p_high=99.99):
    """CLAHE + percentile normalization"""
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_enhanced = clahe.apply(img_uint8).astype(np.float32)
    
    p_low_val, p_high_val = np.percentile(img_enhanced, (p_low, p_high))
    return np.clip((img_enhanced - p_low_val) / (p_high_val - p_low_val + 1e-3), 0, 1)


def apply_mild_clahe_percentile(img, p_low=0.01, p_high=99.99):
    """Mild CLAHE + percentile normalization"""
    return apply_clahe_percentile(img, clip_limit=1.5, tile_grid_size=(12, 12), p_low=p_low, p_high=p_high)


def apply_percentile_only(img, p_low=0.01, p_high=99.99):
    """Percentile normalization only"""
    p_low_val, p_high_val = np.percentile(img, (p_low, p_high))
    return np.clip((img - p_low_val) / (p_high_val - p_low_val + 1e-3), 0, 1)


def analyze_adipocyte_references():
    """Analyze adipocyte reference images"""
    adipocyte_dir = Path('/home/luci/adipose_tissue-unet/example_class_tiles/test/adipocyte')
    
    if not adipocyte_dir.exists():
        print(f"⚠️  Adipocyte reference directory not found: {adipocyte_dir}")
        return pd.DataFrame()
    
    adipocyte_files = list(adipocyte_dir.glob('*.jpg'))
    
    if not adipocyte_files:
        print(f"⚠️  No adipocyte reference images found")
        return pd.DataFrame()
    
    print(f"\n🎯 Analyzing {len(adipocyte_files)} adipocyte reference images...")
    
    all_metrics = []
    
    for img_path in tqdm(adipocyte_files, desc="Processing adipocyte references"):
        try:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            if img is None:
                continue
            
            # Apply current z-score normalization
            img_normalized = apply_zscore_normalization(img)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(img_normalized, "adipocyte_reference")
            metrics['filename'] = img_path.name
            metrics['source'] = 'adipocyte_reference'
            
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path.name}: {e}")
            continue
    
    df = pd.DataFrame(all_metrics)
    print(f"✅ Analyzed {len(df)} adipocyte reference images")
    
    return df


def sample_dataset_tiles(dataset_path, n_per_split=100):
    """Sample representative tiles from dataset"""
    
    splits = ['train', 'val', 'test']
    all_samples = []
    
    random.seed(42)  # For reproducible sampling
    
    for split in splits:
        images_path = dataset_path / split / 'images'
        
        if not images_path.exists():
            print(f"⚠️  Split path not found: {images_path}")
            continue
        
        image_files = list(images_path.glob('*.jpg'))
        
        if not image_files:
            print(f"⚠️  No images found in {split}")
            continue
        
        # Sample n_per_split images
        sampled_files = random.sample(image_files, min(n_per_split, len(image_files)))
        
        for img_path in sampled_files:
            all_samples.append((img_path, split))
        
        print(f"📁 Sampled {len(sampled_files)} images from {split}")
    
    return all_samples


def analyze_normalization_methods(sampled_tiles):
    """Analyze all normalization methods on sampled tiles"""
    
    normalization_methods = {
        'current_zscore': lambda img: apply_zscore_normalization(img),
        'clahe_percentile': lambda img: apply_clahe_percentile(img),
        'mild_clahe_percentile': lambda img: apply_mild_clahe_percentile(img),
        'percentile_only': lambda img: apply_percentile_only(img)
    }
    
    all_results = []
    
    print(f"\n🔬 Analyzing {len(sampled_tiles)} tiles with {len(normalization_methods)} normalization methods...")
    
    for img_path, split in tqdm(sampled_tiles, desc="Processing tiles"):
        try:
            # Load raw image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            
            if img is None:
                continue
            
            # Apply each normalization method
            for method_name, method_func in normalization_methods.items():
                try:
                    img_normalized = method_func(img)
                    
                    # Calculate metrics
                    metrics = calculate_comprehensive_metrics(img_normalized, method_name)
                    metrics['filename'] = img_path.name
                    metrics['split'] = split
                    metrics['source'] = 'dataset_tile'
                    
                    all_results.append(metrics)
                    
                except Exception as e:
                    print(f"⚠️  Error with {method_name} on {img_path.name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error loading {img_path.name}: {e}")
            continue
    
    df = pd.DataFrame(all_results)
    print(f"✅ Analyzed {len(df)} method/tile combinations")
    
    return df


def compare_to_adipocyte_standards(dataset_df, adipocyte_df):
    """Compare dataset tiles to adipocyte reference standards"""
    
    if adipocyte_df.empty:
        print("⚠️  No adipocyte reference data for comparison")
        return None
    
    print(f"\n📊 Comparing dataset tiles to adipocyte standards...")
    
    # Calculate adipocyte reference statistics
    adipocyte_stats = {}
    quality_metrics = ['contrast_ratio', 'laplacian_variance', 'entropy', 'edge_density', 'dynamic_range', 'local_contrast_consistency']
    
    for metric in quality_metrics:
        adipocyte_stats[metric] = {
            'mean': adipocyte_df[metric].mean(),
            'std': adipocyte_df[metric].std(),
            'min': adipocyte_df[metric].min(),
            'max': adipocyte_df[metric].max()
        }
    
    # Calculate similarity scores for each tile/method combination
    similarity_results = []
    
    for _, row in dataset_df.iterrows():
        similarity_score = 0
        metric_scores = {}
        
        for metric in quality_metrics:
            tile_value = row[metric]
            ref_mean = adipocyte_stats[metric]['mean']
            ref_std = adipocyte_stats[metric]['std']
            
            # Calculate z-score distance from adipocyte mean
            z_distance = abs((tile_value - ref_mean) / (ref_std + 1e-6))
            
            # Convert to similarity score (closer = higher score)
            metric_score = np.exp(-z_distance / 2)  # Gaussian-like scoring
            metric_scores[f'{metric}_similarity'] = metric_score
            similarity_score += metric_score
        
        # Average similarity across all metrics
        overall_similarity = similarity_score / len(quality_metrics)
        
        similarity_result = {
            'filename': row['filename'],
            'split': row['split'],
            'method': row['method'],
            'overall_similarity': overall_similarity,
            **metric_scores
        }
        
        similarity_results.append(similarity_result)
    
    similarity_df = pd.DataFrame(similarity_results)
    
    return similarity_df, adipocyte_stats


def create_comprehensive_visualizations(dataset_df, adipocyte_df, similarity_df, adipocyte_stats):
    """Create comprehensive visualization dashboard"""
    
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    quality_metrics = ['contrast_ratio', 'laplacian_variance', 'entropy', 'edge_density', 'dynamic_range', 'local_contrast_consistency']
    methods = dataset_df['method'].unique()
    
    # 1. Method comparison - overall similarity
    ax1 = fig.add_subplot(gs[0, 0])
    similarity_by_method = similarity_df.groupby('method')['overall_similarity'].mean().sort_values(ascending=False)
    bars = ax1.bar(range(len(similarity_by_method)), similarity_by_method.values)
    ax1.set_xticks(range(len(similarity_by_method)))
    ax1.set_xticklabels(similarity_by_method.index, rotation=45)
    ax1.set_ylabel('Avg Similarity to Adipocytes')
    ax1.set_title('Method Performance: Similarity to Adipocyte References')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 2. Method comparison by split
    ax2 = fig.add_subplot(gs[0, 1])
    similarity_pivot = similarity_df.groupby(['method', 'split'])['overall_similarity'].mean().unstack()
    similarity_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title('Similarity by Method and Split')
    ax2.set_ylabel('Similarity Score')
    ax2.legend(title='Split')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3-8. Individual metric comparisons
    for i, metric in enumerate(quality_metrics):
        ax = fig.add_subplot(gs[(i+2)//4, (i+2)%4])
        
        # Box plot comparing methods for this metric
        metric_data = []
        metric_labels = []
        
        # Add adipocyte reference
        adipocyte_values = adipocyte_df[metric].values
        metric_data.append(adipocyte_values)
        metric_labels.append('Adipocyte\nRef')
        
        # Add each method
        for method in methods:
            method_values = dataset_df[dataset_df['method'] == method][metric].values
            metric_data.append(method_values)
            metric_labels.append(method.replace('_', '\n'))
        
        ax.boxplot(metric_data, labels=metric_labels)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add adipocyte reference line
        ref_mean = adipocyte_stats[metric]['mean']
        ax.axhline(y=ref_mean, color='red', linestyle='--', alpha=0.7, label='Adipocyte Mean')
    
    # Method improvement analysis
    ax_improvement = fig.add_subplot(gs[3, 2:])
    
    # Calculate improvement over current method
    current_similarity = similarity_df[similarity_df['method'] == 'current_zscore']['overall_similarity'].mean()
    
    improvements = []
    for method in methods:
        if method != 'current_zscore':
            method_similarity = similarity_df[similarity_df['method'] == method]['overall_similarity'].mean()
            improvement = ((method_similarity - current_similarity) / current_similarity) * 100
            improvements.append((method, improvement))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    method_names = [imp[0].replace('_', '\n') for imp in improvements]
    improvement_values = [imp[1] for imp in improvements]
    
    colors = ['green' if x > 0 else 'red' for x in improvement_values]
    bars = ax_improvement.bar(method_names, improvement_values, color=colors, alpha=0.7)
    ax_improvement.set_ylabel('Improvement over Current (%)')
    ax_improvement.set_title('Normalization Method Improvement over Current Z-Score')
    ax_improvement.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax_improvement.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax_improvement.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', 
                           va='bottom' if height >= 0 else 'top')
    
    plt.suptitle('Comprehensive Normalization Analysis: Dataset vs Adipocyte References', 
                 fontsize=16, y=0.98)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Comprehensive normalization analysis')
    parser.add_argument('--output-dir', type=str, default='preprocessing_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--samples-per-split', type=int, default=100,
                       help='Number of samples per split to analyze')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE NORMALIZATION ANALYSIS")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Samples per split: {args.samples_per_split}")
    
    # 1. Analyze adipocyte references
    adipocyte_df = analyze_adipocyte_references()
    
    # 2. Sample dataset tiles
    data_root = Path('~/Data_for_ML/Meat_Luci_Tulane/_build').expanduser()
    dataset_path = data_root / 'dataset'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    sampled_tiles = sample_dataset_tiles(dataset_path, args.samples_per_split)
    
    if not sampled_tiles:
        print("❌ No tiles sampled for analysis!")
        return
    
    # 3. Analyze normalization methods
    dataset_df = analyze_normalization_methods(sampled_tiles)
    
    # 4. Compare to adipocyte standards
    similarity_df, adipocyte_stats = compare_to_adipocyte_standards(dataset_df, adipocyte_df)
    
    # 5. Create visualizations
    print(f"\n📈 Creating comprehensive visualizations...")
    fig = create_comprehensive_visualizations(dataset_df, adipocyte_df, similarity_df, adipocyte_stats)
    fig.savefig(output_dir / 'comprehensive_normalization_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Save detailed results
    dataset_df.to_csv(output_dir / 'dataset_normalization_metrics.csv', index=False)
    adipocyte_df.to_csv(output_dir / 'adipocyte_reference_metrics.csv', index=False)
    similarity_df.to_csv(output_dir / 'similarity_to_adipocytes.csv', index=False)
    
    # 7. Generate summary report
    print(f"\n📋 Generating summary report...")
    
    # Calculate key statistics
    best_method = similarity_df.groupby('method')['overall_similarity'].mean().sort_values(ascending=False).index[0]
    current_similarity = similarity_df[similarity_df['method'] == 'current_zscore']['overall_similarity'].mean()
    best_similarity = similarity_df[similarity_df['method'] == best_method]['overall_similarity'].mean()
    improvement = ((best_similarity - current_similarity) / current_similarity) * 100
    
    report = f"""# Comprehensive Normalization Analysis Report

## Executive Summary

**Total Tiles Analyzed:** {len(sampled_tiles)}
**Adipocyte References:** {len(adipocyte_df)}
**Normalization Methods Compared:** 4

## Key Findings

### Best Performing Method: **{best_method.replace('_', ' ').title()}**
- Similarity to adipocytes: {best_similarity:.3f}
- Improvement over current: {improvement:.1f}%

### Method Rankings by Similarity to Adipocyte References:
"""
    
    method_rankings = similarity_df.groupby('method')['overall_similarity'].mean().sort_values(ascending=False)
    for i, (method, score) in enumerate(method_rankings.items(), 1):
        improvement_vs_current = ((score - current_similarity) / current_similarity) * 100 if method != 'current_zscore' else 0
        report += f"{i}. **{method.replace('_', ' ').title()}**: {score:.3f} ({improvement_vs_current:+.1f}%)\n"
    
    report += f"""
## Adipocyte Reference Standards

"""
    
    for metric in ['contrast_ratio', 'laplacian_variance', 'entropy', 'edge_density']:
        stats = adipocyte_stats[metric]
        report += f"- **{metric.replace('_', ' ').title()}**: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f}-{stats['max']:.3f})\n"
    
    report += f"""
## Recommendations

Based on this comprehensive analysis of {len(sampled_tiles)} dataset tiles compared to {len(adipocyte_df)} adipocyte references:

1. **Switch to {best_method.replace('_', ' ')}** for {improvement:.1f}% improvement in similarity to high-quality adipocytes
2. Focus on metrics that matter most for segmentation: edge density and local contrast consistency
3. Monitor performance on actual segmentation tasks to validate similarity improvements

## Files Generated

- `comprehensive_normalization_analysis.png`: Complete visualization dashboard
- `dataset_normalization_metrics.csv`: Detailed metrics for all tile/method combinations
- `adipocyte_reference_metrics.csv`: Reference standards from adipocyte examples
- `similarity_to_adipocytes.csv`: Similarity scores for each method

## Implementation

The optimal preprocessing pipeline should use **{best_method.replace('_', ' ')}** to achieve the closest similarity to high-quality adipocyte references.
"""
    
    with open(output_dir / 'COMPREHENSIVE_NORMALIZATION_REPORT.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Analyzed {len(sampled_tiles)} tiles with 4 normalization methods")
    print(f"🎯 Compared against {len(adipocyte_df)} adipocyte references")
    print(f"🏆 Best method: {best_method} ({improvement:.1f}% improvement)")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
