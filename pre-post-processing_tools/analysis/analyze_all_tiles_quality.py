#!/usr/bin/env python3
"""
Comprehensive Analysis of ALL Tiles in _build Dataset
Analyzes image quality metrics for all tiles including empty/blurry ones
Provides robust statistics for adaptive CLAHE grouping
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)


def calculate_image_quality_metrics(img):
    """Calculate key image quality metrics efficiently"""
    
    # Basic statistics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    
    # Primary metrics for grouping
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    # Sharpness metric (using uint8 for OpenCV compatibility)
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    laplacian_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    
    # Additional useful metrics
    dynamic_range = img.max() - img.min()
    
    # Simple local contrast metric (faster than full sliding window)
    # Use 3x3 local standard deviation
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(img, -1, kernel)
    local_var = cv2.filter2D(img**2, -1, kernel) - local_mean**2
    avg_local_contrast = np.mean(np.sqrt(np.maximum(local_var, 0)))
    
    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'contrast_ratio': contrast_ratio,
        'laplacian_variance': laplacian_var,
        'dynamic_range': dynamic_range,
        'avg_local_contrast': avg_local_contrast
    }


def analyze_dataset_folder(dataset_path, split_name):
    """Analyze all images in a dataset folder"""
    
    images_path = dataset_path / split_name / 'images'
    
    if not images_path.exists():
        print(f"⚠️  Images path not found: {images_path}")
        return pd.DataFrame()
    
    # Get all image files
    image_files = list(images_path.glob('*.jpg'))
    
    if not image_files:
        print(f"⚠️  No images found in {images_path}")
        return pd.DataFrame()
    
    print(f"\n📁 Analyzing {split_name} folder: {len(image_files)} images")
    
    all_metrics = []
    
    # Process images with progress bar
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        try:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"⚠️  Could not load: {img_path.name}")
                continue
                
            img = img.astype(np.float32)
            
            # Calculate metrics
            metrics = calculate_image_quality_metrics(img)
            metrics['split'] = split_name
            metrics['filename'] = img_path.name
            metrics['file_path'] = str(img_path)
            
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path.name}: {e}")
            continue
    
    df = pd.DataFrame(all_metrics)
    print(f"✅ Successfully analyzed {len(df)} images from {split_name}")
    
    return df


def perform_clustering_analysis(df, n_clusters=3):
    """Perform clustering analysis to identify natural groupings"""
    
    # Use primary metrics for clustering
    features = ['contrast_ratio', 'laplacian_variance']
    X = df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return df_clustered, cluster_centers, scaler


def analyze_clustering_results(df_clustered, cluster_centers):
    """Analyze and interpret clustering results"""
    
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS RESULTS")
    print("="*80)
    
    # Cluster statistics
    print("\nCluster Statistics:")
    print("-" * 50)
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        print(f"\nCluster {cluster_id} ({len(cluster_data)} images):")
        print(f"  Contrast Ratio: {cluster_data['contrast_ratio'].mean():.3f} ± {cluster_data['contrast_ratio'].std():.3f}")
        print(f"  Laplacian Var:  {cluster_data['laplacian_variance'].mean():.1f} ± {cluster_data['laplacian_variance'].std():.1f}")
        print(f"  Range CR: {cluster_data['contrast_ratio'].min():.3f} - {cluster_data['contrast_ratio'].max():.3f}")
        print(f"  Range LV: {cluster_data['laplacian_variance'].min():.1f} - {cluster_data['laplacian_variance'].max():.1f}")
    
    # Determine cluster characteristics
    cluster_chars = []
    for i, center in enumerate(cluster_centers):
        contrast, sharpness = center
        
        if contrast < 0.15 and sharpness < 20:
            char = "Poor Quality (Low contrast, Low sharpness)"
            recommendation = "Needs aggressive CLAHE"
        elif contrast > 0.25 and sharpness > 35:
            char = "Good Quality (High contrast, High sharpness)"
            recommendation = "Percentile normalization only"
        else:
            char = "Medium Quality (Moderate contrast/sharpness)"
            recommendation = "Needs mild CLAHE"
        
        cluster_chars.append({
            'cluster': i,
            'contrast_ratio': contrast,
            'laplacian_variance': sharpness,
            'characteristic': char,
            'recommendation': recommendation
        })
    
    print(f"\nCluster Interpretations:")
    print("-" * 50)
    for char in cluster_chars:
        print(f"Cluster {char['cluster']}: {char['characteristic']}")
        print(f"  Center: CR={char['contrast_ratio']:.3f}, LV={char['laplacian_variance']:.1f}")
        print(f"  Recommendation: {char['recommendation']}")
    
    return cluster_chars


def determine_robust_cutoffs(df):
    """Determine robust cutoffs based on all data"""
    
    print("\n" + "="*80)
    print("ROBUST CUTOFF DETERMINATION")
    print("="*80)
    
    # Calculate comprehensive statistics
    cr_values = df['contrast_ratio'].values
    lv_values = df['laplacian_variance'].values
    
    print(f"\nDataset Statistics (n={len(df)}):")
    print("-" * 50)
    print(f"Contrast Ratio:")
    print(f"  Mean: {cr_values.mean():.3f} ± {cr_values.std():.3f}")
    print(f"  Min: {cr_values.min():.3f}, Max: {cr_values.max():.3f}")
    print(f"  Percentiles: 10%={np.percentile(cr_values, 10):.3f}, 25%={np.percentile(cr_values, 25):.3f}, 50%={np.percentile(cr_values, 50):.3f}, 75%={np.percentile(cr_values, 75):.3f}, 90%={np.percentile(cr_values, 90):.3f}")
    
    print(f"\nLaplacian Variance:")
    print(f"  Mean: {lv_values.mean():.1f} ± {lv_values.std():.1f}")
    print(f"  Min: {lv_values.min():.1f}, Max: {lv_values.max():.1f}")
    print(f"  Percentiles: 10%={np.percentile(lv_values, 10):.1f}, 25%={np.percentile(lv_values, 25):.1f}, 50%={np.percentile(lv_values, 50):.1f}, 75%={np.percentile(lv_values, 75):.1f}, 90%={np.percentile(lv_values, 90):.1f}")
    
    # Conservative cutoffs based on data distribution
    # Use 25th and 75th percentiles as natural breaking points
    cr_poor_cutoff = np.percentile(cr_values, 25)
    cr_good_cutoff = np.percentile(cr_values, 75)
    
    lv_poor_cutoff = np.percentile(lv_values, 25)
    lv_good_cutoff = np.percentile(lv_values, 75)
    
    cutoffs = {
        'contrast_ratio': {
            'poor_cutoff': cr_poor_cutoff,
            'good_cutoff': cr_good_cutoff
        },
        'laplacian_variance': {
            'poor_cutoff': lv_poor_cutoff,
            'good_cutoff': lv_good_cutoff
        }
    }
    
    print(f"\nRecommended Cutoffs (based on quartiles):")
    print("-" * 50)
    print(f"Contrast Ratio:")
    print(f"  Poor (needs CLAHE): < {cr_poor_cutoff:.3f}")
    print(f"  Medium (needs mild CLAHE): {cr_poor_cutoff:.3f} - {cr_good_cutoff:.3f}")
    print(f"  Good (percentile only): > {cr_good_cutoff:.3f}")
    
    print(f"\nLaplacian Variance:")
    print(f"  Poor: < {lv_poor_cutoff:.1f}")
    print(f"  Medium: {lv_poor_cutoff:.1f} - {lv_good_cutoff:.1f}")
    print(f"  Good: > {lv_good_cutoff:.1f}")
    
    return cutoffs


def create_comprehensive_visualizations(all_data, df_clustered, cluster_centers, cutoffs):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall scatter plot with clusters
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        ax1.scatter(cluster_data['contrast_ratio'], cluster_data['laplacian_variance'], 
                   c=colors[cluster_id], alpha=0.6, s=10, label=f'Cluster {cluster_id}')
    
    # Plot cluster centers
    ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centers')
    
    ax1.set_xlabel('Contrast Ratio')
    ax1.set_ylabel('Laplacian Variance')
    ax1.set_title(f'All Tiles Clustered (n={len(df_clustered)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Split comparison
    ax2 = fig.add_subplot(gs[0, 1])
    splits = all_data['split'].unique()
    colors_split = ['blue', 'orange', 'green']
    for i, split in enumerate(splits):
        split_data = all_data[all_data['split'] == split]
        ax2.scatter(split_data['contrast_ratio'], split_data['laplacian_variance'], 
                   c=colors_split[i], alpha=0.6, s=10, label=f'{split} (n={len(split_data)})')
    
    ax2.set_xlabel('Contrast Ratio')
    ax2.set_ylabel('Laplacian Variance')
    ax2.set_title('Distribution by Split')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cutoff boundaries
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(all_data['contrast_ratio'], all_data['laplacian_variance'], 
               alpha=0.5, s=8, c='gray')
    
    # Add cutoff lines
    ax3.axvline(cutoffs['contrast_ratio']['poor_cutoff'], color='red', linestyle='--', 
               label=f"CR Poor: {cutoffs['contrast_ratio']['poor_cutoff']:.3f}")
    ax3.axvline(cutoffs['contrast_ratio']['good_cutoff'], color='green', linestyle='--',
               label=f"CR Good: {cutoffs['contrast_ratio']['good_cutoff']:.3f}")
    ax3.axhline(cutoffs['laplacian_variance']['poor_cutoff'], color='red', linestyle=':',
               label=f"LV Poor: {cutoffs['laplacian_variance']['poor_cutoff']:.1f}")
    ax3.axhline(cutoffs['laplacian_variance']['good_cutoff'], color='green', linestyle=':',
               label=f"LV Good: {cutoffs['laplacian_variance']['good_cutoff']:.1f}")
    
    ax3.set_xlabel('Contrast Ratio')
    ax3.set_ylabel('Laplacian Variance')
    ax3.set_title('Proposed Cutoff Boundaries')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Histograms by split
    metrics = ['contrast_ratio', 'laplacian_variance']
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i])
        
        for split in splits:
            split_data = all_data[all_data['split'] == split]
            ax.hist(split_data[metric], alpha=0.6, bins=30, label=f'{split}', density=True)
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution by Split')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Cluster size distribution
    ax7 = fig.add_subplot(gs[1, 2])
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    ax7.bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Number of Images')
    ax7.set_title('Cluster Size Distribution')
    ax7.grid(True, alpha=0.3)
    
    # 8-9. Box plots
    ax8 = fig.add_subplot(gs[2, 0])
    all_data.boxplot(column='contrast_ratio', by='split', ax=ax8)
    ax8.set_title('Contrast Ratio by Split')
    
    ax9 = fig.add_subplot(gs[2, 1])
    all_data.boxplot(column='laplacian_variance', by='split', ax=ax9)
    ax9.set_title('Laplacian Variance by Split')
    
    # 10. Summary statistics table
    ax10 = fig.add_subplot(gs[2, 2])
    ax10.axis('off')
    
    # Create summary table
    summary_text = f"""
COMPREHENSIVE ANALYSIS SUMMARY

Total Images Analyzed: {len(all_data)}

Split Distribution:
"""
    for split in splits:
        count = len(all_data[all_data['split'] == split])
        summary_text += f"  {split}: {count} images\n"
    
    summary_text += f"""
Clustering Results:
  {len(cluster_centers)} natural clusters identified
  
Recommended Cutoffs:
  CR Poor: < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}
  CR Good: > {cutoffs['contrast_ratio']['good_cutoff']:.3f}
  LV Poor: < {cutoffs['laplacian_variance']['poor_cutoff']:.1f}
  LV Good: > {cutoffs['laplacian_variance']['good_cutoff']:.1f}
"""
    
    ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Comprehensive Tile Quality Analysis', fontsize=16, y=0.98)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis of all tiles')
    parser.add_argument('--output-dir', type=str, default='preprocessing_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Find dataset
    data_root = Path('~/Data_for_ML/Meat_Luci_Tulane/_build').expanduser()
    dataset_path = data_root / 'dataset'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE TILE QUALITY ANALYSIS")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    
    # Analyze each split separately
    all_dataframes = []
    splits = ['train', 'val', 'test']
    
    for split in splits:
        df_split = analyze_dataset_folder(dataset_path, split)
        if not df_split.empty:
            all_dataframes.append(df_split)
    
    if not all_dataframes:
        print("❌ No data found to analyze!")
        return
    
    # Combine all data
    all_data = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\n📊 TOTAL DATASET SUMMARY:")
    print(f"   Total images analyzed: {len(all_data)}")
    for split in splits:
        count = len(all_data[all_data['split'] == split])
        print(f"   {split}: {count} images")
    
    # Perform clustering analysis
    print(f"\n🔍 Performing clustering analysis...")
    df_clustered, cluster_centers, scaler = perform_clustering_analysis(all_data, n_clusters=3)
    
    # Analyze clustering results
    cluster_chars = analyze_clustering_results(df_clustered, cluster_centers)
    
    # Determine robust cutoffs
    cutoffs = determine_robust_cutoffs(all_data)
    
    # Create visualizations
    print(f"\n📈 Creating comprehensive visualizations...")
    fig = create_comprehensive_visualizations(all_data, df_clustered, cluster_centers, cutoffs)
    fig.savefig(output_dir / 'comprehensive_tile_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save detailed results
    all_data.to_csv(output_dir / 'all_tiles_quality_metrics.csv', index=False)
    df_clustered.to_csv(output_dir / 'all_tiles_with_clusters.csv', index=False)
    
    # Create updated adaptive function
    adaptive_function_code = f"""import numpy as np
import cv2

def adaptive_clahe_normalization_robust(img):
    \"\"\"
    Robust adaptive CLAHE normalization based on analysis of {len(all_data)} tiles
    Cutoffs determined from comprehensive dataset analysis
    \"\"\"
    # Calculate image quality metrics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    sharpness = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    
    # Determine preprocessing strategy based on robust cutoffs
    if contrast_ratio < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}:
        # Poor quality - needs aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_uint8).astype(np.float32)
        p5, p95 = np.percentile(img_enhanced, (5, 95))
        return np.clip((img_enhanced - p5) / (p95 - p5 + 1e-3), 0, 1)
        
    elif contrast_ratio > {cutoffs['contrast_ratio']['good_cutoff']:.3f} and sharpness > {cutoffs['laplacian_variance']['good_cutoff']:.1f}:
        # Good quality - percentile normalization only
        p2, p98 = np.percentile(img, (2, 98))
        return np.clip((img - p2) / (p98 - p2 + 1e-3), 0, 1)
        
    else:
        # Medium quality - mild CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
        img_enhanced = clahe.apply(img_uint8).astype(np.float32)
        p5, p95 = np.percentile(img_enhanced, (5, 95))
        return np.clip((img_enhanced - p5) / (p95 - p5 + 1e-3), 0, 1)

def classify_image_quality(img):
    \"\"\"
    Classify image quality based on robust analysis
    Returns: 'poor', 'medium', or 'good'
    \"\"\"
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    sharpness = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    
    if contrast_ratio < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}:
        return 'poor'
    elif contrast_ratio > {cutoffs['contrast_ratio']['good_cutoff']:.3f} and sharpness > {cutoffs['laplacian_variance']['good_cutoff']:.1f}:
        return 'good'
    else:
        return 'medium'
"""
    
    with open(output_dir / 'adaptive_clahe_function_robust.py', 'w') as f:
        f.write(adaptive_function_code)
    
    # Create comprehensive report
    report = f"""# Comprehensive Tile Quality Analysis Report

## Dataset Overview

**Total Images Analyzed:** {len(all_data)}

**Split Distribution:**
"""
    for split in splits:
        count = len(all_data[all_data['split'] == split])
        percentage = count / len(all_data) * 100
        report += f"- {split}: {count} images ({percentage:.1f}%)\n"
    
    report += f"""
## Quality Metrics Statistics

### Contrast Ratio (std/mean):
- Mean: {all_data['contrast_ratio'].mean():.3f} ± {all_data['contrast_ratio'].std():.3f}
- Range: {all_data['contrast_ratio'].min():.3f} - {all_data['contrast_ratio'].max():.3f}
- Quartiles: Q1={np.percentile(all_data['contrast_ratio'], 25):.3f}, Q2={np.percentile(all_data['contrast_ratio'], 50):.3f}, Q3={np.percentile(all_data['contrast_ratio'], 75):.3f}

### Laplacian Variance (sharpness):
- Mean: {all_data['laplacian_variance'].mean():.1f} ± {all_data['laplacian_variance'].std():.1f}
- Range: {all_data['laplacian_variance'].min():.1f} - {all_data['laplacian_variance'].max():.1f}
- Quartiles: Q1={np.percentile(all_data['laplacian_variance'], 25):.1f}, Q2={np.percentile(all_data['laplacian_variance'], 50):.1f}, Q3={np.percentile(all_data['laplacian_variance'], 75):.1f}

## Clustering Analysis

**Number of Natural Clusters:** {len(cluster_centers)}

**Cluster Characteristics:**
"""
    
    for char in cluster_chars:
        cluster_data = df_clustered[df_clustered['cluster'] == char['cluster']]
        report += f"""
### Cluster {char['cluster']} ({len(cluster_data)} images):
- **Characteristic:** {char['characteristic']}
- **Center:** Contrast Ratio = {char['contrast_ratio']:.3f}, Laplacian Variance = {char['laplacian_variance']:.1f}
- **Recommendation:** {char['recommendation']}
"""
    
    report += f"""
## Robust Cutoffs (Based on Quartiles)

### Contrast Ratio Cutoffs:
- **Poor Quality (needs CLAHE):** < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}
- **Medium Quality (needs mild CLAHE):** {cutoffs['contrast_ratio']['poor_cutoff']:.3f} - {cutoffs['contrast_ratio']['good_cutoff']:.3f}
- **Good Quality (percentile only):** > {cutoffs['contrast_ratio']['good_cutoff']:.3f}

### Laplacian Variance Cutoffs:
- **Poor:** < {cutoffs['laplacian_variance']['poor_cutoff']:.1f}
- **Medium:** {cutoffs['laplacian_variance']['poor_cutoff']:.1f} - {cutoffs['laplacian_variance']['good_cutoff']:.1f}
- **Good:** > {cutoffs['laplacian_variance']['good_cutoff']:.1f}

## Implementation Strategy

Based on this comprehensive analysis of {len(all_data)} tiles:

1. **Poor Quality Images ({np.sum(all_data['contrast_ratio'] < cutoffs['contrast_ratio']['poor_cutoff'])} tiles, {np.sum(all_data['contrast_ratio'] < cutoffs['contrast_ratio']['poor_cutoff'])/len(all_data)*100:.1f}%)**
   - Apply standard CLAHE (clipLimit=2.0, tileGridSize=(8,8))
   - Use percentile normalization (5-95th percentile)

2. **Medium Quality Images**
   - Apply mild CLAHE (clipLimit=1.5, tileGridSize=(12,12))  
   - Use percentile normalization (5-95th percentile)

3. **Good Quality Images ({np.sum((all_data['contrast_ratio'] > cutoffs['contrast_ratio']['good_cutoff']) & (all_data['laplacian_variance'] > cutoffs['laplacian_variance']['good_cutoff']))} tiles, {np.sum((all_data['contrast_ratio'] > cutoffs['contrast_ratio']['good_cutoff']) & (all_data['laplacian_variance'] > cutoffs['laplacian_variance']['good_cutoff']))/len(all_data)*100:.1f}%)**
   - Skip CLAHE entirely
   - Use percentile normalization only (2-98th percentile)

## Files Generated

- `comprehensive_tile_quality_analysis.png`: Complete visualization dashboard
- `all_tiles_quality_metrics.csv`: Raw metrics for all {len(all_data)} tiles
- `all_tiles_with_clusters.csv`: Tiles with cluster assignments
- `adaptive_clahe_function_robust.py`: Updated adaptive function based on robust statistics

## Confidence Level

This analysis is based on **{len(all_data)} tiles** across all splits, providing high statistical confidence in the determined cutoffs and groupings.
"""
    
    with open(output_dir / 'COMPREHENSIVE_QUALITY_ANALYSIS_REPORT.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Analyzed {len(all_data)} total tiles")
    print(f"📁 Results saved to: {output_dir}")
    print("\n📋 Files generated:")
    print("- comprehensive_tile_quality_analysis.png")
    print("- all_tiles_quality_metrics.csv")
    print("- all_tiles_with_clusters.csv")
    print("- adaptive_clahe_function_robust.py")
    print("- COMPREHENSIVE_QUALITY_ANALYSIS_REPORT.md")
    
    # Print key findings
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   Robust cutoffs based on {len(all_data)} tiles:")
    print(f"   - Poor quality: CR < {cutoffs['contrast_ratio']['poor_cutoff']:.3f}")
    print(f"   - Good quality: CR > {cutoffs['contrast_ratio']['good_cutoff']:.3f} AND LV > {cutoffs['laplacian_variance']['good_cutoff']:.1f}")


if __name__ == '__main__':
    main()
