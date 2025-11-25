#!/usr/bin/env python3
"""
Preprocessing Pipeline Visualization Script
Shows the complete transformation: Original → Reinhard → Z-score → Percentile

Visualizes 5 sample tiles through all preprocessing stages to understand
the effect of stain normalization and intensity normalization.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple
import json

# Add project root to path for imports
sys.path.append('.')
from src.utils.data import (
    load_stain_normalizer, 
    apply_stain_normalization, 
    preprocess_image,
    compute_dataset_statistics
)

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

def find_sample_tiles(base_path: str, n_samples: int = 5) -> List[str]:
    """
    Find sample tiles from the dataset.
    
    Args:
        base_path: Base directory to search for tiles
        n_samples: Number of samples to return
        
    Returns:
        List of tile paths
    """
    print(f"Searching for tiles in: {base_path}")
    
    # Use the correct path provided by user
    dataset_path = Path(base_path).expanduser() / "_build/dataset/train/images"
    
    print(f"  Checking: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    tile_files = list(dataset_path.glob("*.jpg"))
    if not tile_files:
        raise FileNotFoundError(f"No .jpg tiles found in {dataset_path}")
    
    print(f"  Found {len(tile_files)} tiles in dataset")
    
    # Select diverse samples (spread across the dataset)
    if len(tile_files) > n_samples:
        step = len(tile_files) // n_samples
        selected = [tile_files[i * step] for i in range(n_samples)]
    else:
        selected = tile_files[:n_samples]
    
    print(f"Selected {len(selected)} tiles for visualization:")
    for tile in selected:
        print(f"  - {tile.name}")
    
    return [str(p) for p in selected]

def load_and_preprocess_tile_dual(tile_path: str, stain_normalizer, dataset_mean: float, 
                                 dataset_std: float) -> Tuple[dict, dict]:
    """
    Load a tile and apply all preprocessing stages in both RGB and grayscale.
    
    Args:
        tile_path: Path to tile image
        stain_normalizer: Stain normalizer object
        dataset_mean: Dataset mean for z-score normalization
        dataset_std: Dataset std for z-score normalization
        
    Returns:
        (rgb_stages, grayscale_stages) - dictionaries with 'original', 'reinhard', 'zscore', 'percentile'
    """
    # Load original image in RGB and grayscale
    original_rgb = cv2.imread(tile_path, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
    original_gray = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # RGB Processing Pipeline
    try:
        # Apply Reinhard normalization to RGB
        from reinhard_stain_normalization import load_best_reference
        reinhard_normalizer = load_best_reference()
        reinhard_rgb = reinhard_normalizer.normalize_image(original_rgb.astype(np.uint8))
        
        # Convert to grayscale for further processing
        reinhard_gray = cv2.cvtColor(reinhard_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Apply z-score normalization
        zscore_gray = preprocess_image(
            original_gray, 
            stain_normalizer=stain_normalizer,
            intensity_method='zscore_dataset',
            dataset_mean=dataset_mean,
            dataset_std=dataset_std
        )
        
        # Apply percentile normalization
        percentile_gray = preprocess_image(
            original_gray,
            stain_normalizer=stain_normalizer, 
            intensity_method='percentile',
            percentile_low=1,
            percentile_high=99
        )
        
        # For RGB display, convert normalized versions back to RGB representation
        zscore_rgb = np.stack([zscore_gray, zscore_gray, zscore_gray], axis=-1)
        percentile_rgb = np.stack([percentile_gray, percentile_gray, percentile_gray], axis=-1)
        
    except Exception as e:
        print(f"    Warning: Reinhard normalization failed, using grayscale only: {e}")
        # Fallback: use grayscale for both
        reinhard_rgb = np.stack([original_gray, original_gray, original_gray], axis=-1)
        reinhard_gray = original_gray.copy()
        
        zscore_gray = original_gray.copy()  # Fallback
        percentile_gray = original_gray.copy()  # Fallback
        zscore_rgb = np.stack([zscore_gray, zscore_gray, zscore_gray], axis=-1)
        percentile_rgb = np.stack([percentile_gray, percentile_gray, percentile_gray], axis=-1)
    
    # RGB stages
    rgb_stages = {
        'original': original_rgb,
        'reinhard': reinhard_rgb,
        'zscore': zscore_rgb,
        'percentile': percentile_rgb
    }
    
    # Grayscale stages (what the network sees)
    grayscale_stages = {
        'original': original_gray,
        'reinhard': reinhard_gray,
        'zscore': zscore_gray,
        'percentile': percentile_gray
    }
    
    return rgb_stages, grayscale_stages

def compute_image_stats(image: np.ndarray) -> dict:
    """Compute basic statistics for an image."""
    return {
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'range': float(np.max(image) - np.min(image))
    }

def create_dual_pipeline_visualization(tiles_data: List[Tuple], output_dir: str, version: str):
    """
    Create comprehensive visualization of preprocessing pipeline.
    
    Args:
        tiles_data: List of (tile_path, rgb_stages, gray_stages) tuples
        output_dir: Directory to save visualizations
        version: 'color' or 'grayscale'
    """
    n_tiles = len(tiles_data)
    
    # Create figure with images and histograms
    fig = plt.figure(figsize=(20, 4*n_tiles + 3))
    
    # Overall title
    version_title = "Original Colors" if version == 'color' else "Grayscale (Network View)"
    fig.suptitle(f'Preprocessing Pipeline - {version_title}: Original → Reinhard → Z-score → Percentile', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    stage_names = ['Original', 'Reinhard Normalized', 'Reinhard + Z-score', 'Reinhard + Percentile']
    stage_colors = ['red', 'green', 'blue', 'orange']
    
    for tile_idx, (tile_path, rgb_stages, gray_stages) in enumerate(tiles_data):
        tile_name = Path(tile_path).stem
        
        # Choose RGB or grayscale stages
        stages = rgb_stages if version == 'color' else gray_stages
        images = [stages['original'], stages['reinhard'], stages['zscore'], stages['percentile']]
        
        # Image row
        for stage_idx, (image, stage_name, color) in enumerate(zip(images, stage_names, stage_colors)):
            ax_img = plt.subplot(n_tiles*2, 4, tile_idx*8 + stage_idx + 1)
            
            # Display image
            if version == 'color' and len(image.shape) == 3:
                # RGB image
                display_img = np.clip(image, 0, 255).astype(np.uint8)
                ax_img.imshow(display_img)
            else:
                # Grayscale image
                if stage_idx <= 1:  # Original and Reinhard (0-255 range)
                    display_img = np.clip(image, 0, 255).astype(np.uint8)
                    ax_img.imshow(display_img, cmap='gray', vmin=0, vmax=255)
                else:  # Normalized images (may have negative values)
                    # Normalize for display
                    img_min, img_max = image.min(), image.max()
                    display_img = (image - img_min) / (img_max - img_min) if img_max > img_min else image
                    ax_img.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            
            ax_img.set_title(f'{stage_name}\n{tile_name}' if tile_idx == 0 else stage_name, 
                           fontsize=10, fontweight='bold', color=color)
            ax_img.axis('off')
            
            # Add colored border
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
        
        # Histogram row
        for stage_idx, (image, stage_name, color) in enumerate(zip(images, stage_names, stage_colors)):
            ax_hist = plt.subplot(n_tiles*2, 4, tile_idx*8 + stage_idx + 5)
            
            # For histograms, always use single channel data
            if len(image.shape) == 3:
                hist_data = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                hist_data = image
            
            stats = compute_image_stats(hist_data)
            
            ax_hist.hist(hist_data.flatten(), bins=50, alpha=0.7, density=True, color=color)
            ax_hist.set_xlabel('Pixel Value', fontsize=8)
            ax_hist.set_ylabel('Density', fontsize=8)
            ax_hist.tick_params(labelsize=7)
            
            # Add statistics
            stats_text = f"μ={stats['mean']:.2f}\nσ={stats['std']:.2f}\nRange=[{stats['min']:.2f}, {stats['max']:.2f}]"
            ax_hist.text(0.05, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=7,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend explaining the pipeline
    legend_text = f"""
    Pipeline Stages ({version_title}):
    1. Original: Raw tile from training dataset
    2. Reinhard: SYBR Gold + Eosin stain normalization  
    3. Z-score: Dataset statistics normalization (μ=0, σ=1)
    4. Percentile: 1-99% percentile normalization
    """
    
    plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save to output directory
    output_path = Path(output_dir) / f"preprocessing_pipeline_{version}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {version_title} visualization saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize preprocessing pipeline")
    parser.add_argument('--data-path', type=str, 
                       default='~/Data_for_ML/Meat_Luci_Tulane',
                       help='Path to base data directory')
    parser.add_argument('--n-samples', type=int, default=7,
                       help='Number of sample tiles to visualize')
    parser.add_argument('--output-dir', type=str, 
                       default='preprocessing_analysis',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("🎨 Preprocessing Pipeline Visualization")
    print("=" * 50)
    
    # Expand user path
    data_path = Path(args.data_path).expanduser()
    
    print(f"Data path: {data_path}")
    print(f"Sample count: {args.n_samples}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Find sample tiles
        print("1. Finding sample tiles...")
        tile_paths = find_sample_tiles(str(data_path), args.n_samples)
        
        # Load stain normalizer
        print("\n2. Loading stain normalizer...")
        stain_normalizer = load_stain_normalizer()
        if stain_normalizer:
            print(f"✓ Loaded stain normalizer: {stain_normalizer.reference_path.name}")
        else:
            print("⚠️  No stain normalizer available - will show identity transformation")
        
        # Compute dataset statistics for z-score normalization
        print("\n3. Computing dataset statistics...")
        dataset_mean, dataset_std = compute_dataset_statistics(tile_paths, max_samples=len(tile_paths))
        print(f"Dataset statistics: mean={dataset_mean:.2f}, std={dataset_std:.2f}")
        
        # Process all tiles
        print("\n4. Processing tiles through dual pipeline (RGB + Grayscale)...")
        tiles_data = []
        
        for i, tile_path in enumerate(tile_paths):
            print(f"  Processing tile {i+1}/{len(tile_paths)}: {Path(tile_path).name}")
            
            try:
                rgb_stages, grayscale_stages = load_and_preprocess_tile_dual(
                    tile_path, stain_normalizer, dataset_mean, dataset_std
                )
                tiles_data.append((tile_path, rgb_stages, grayscale_stages))
            except Exception as e:
                print(f"    ⚠️  Failed to process {Path(tile_path).name}: {e}")
                continue
        
        if not tiles_data:
            raise RuntimeError("No tiles were successfully processed")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create both visualizations
        print(f"\n5. Creating dual visualizations with {len(tiles_data)} tiles...")
        
        # Color version
        print("  Creating color version...")
        color_path = create_dual_pipeline_visualization(tiles_data, str(output_dir), 'color')
        
        # Grayscale version  
        print("  Creating grayscale version...")
        grayscale_path = create_dual_pipeline_visualization(tiles_data, str(output_dir), 'grayscale')
        
        print(f"\n✅ Complete! Dual visualizations saved:")
        print(f"  • Color version: {color_path}")
        print(f"  • Grayscale version: {grayscale_path}")
        print("\nVisualization features:")
        print("  • Row 1 per tile: Images at each preprocessing stage")
        print("  • Row 2 per tile: Pixel value histograms with statistics") 
        print("  • Color-coded borders: Red=Original, Green=Reinhard, Blue=Z-score, Orange=Percentile")
        print("  • Color version: Shows RGB stain normalization effects")
        print("  • Grayscale version: Shows exactly what the U-Net network processes")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
