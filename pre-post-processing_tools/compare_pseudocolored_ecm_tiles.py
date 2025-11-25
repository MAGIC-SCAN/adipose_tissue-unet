#!/usr/bin/env python3
"""
Compare Pseudocolored and ECM_channel tiles for visual and quantitative analysis.

Samples tiles from:
1. Perfect matches (5 WSI with same tile counts)
2. Mismatches (8 WSI with different tile counts)

Generates:
- CSV with comparison metrics
- Side-by-side visual comparisons with difference heatmaps
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Ensure reproducibility
RANDOM_SEED = 865


def load_image(path):
    """Load image as RGB numpy array."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def compute_metrics(img1, img2):
    """
    Compute comparison metrics between two images.
    
    Args:
        img1: First image (numpy array, RGB)
        img2: Second image (numpy array, RGB)
    
    Returns:
        Dictionary of metrics
    """
    # Ensure same size
    if img1.shape != img2.shape:
        return {
            'size_match': False,
            'size_diff': f"{img1.shape} vs {img2.shape}",
            'mse': None,
            'psnr': None,
            'ssim': None,
            'correlation': None,
            'mean_abs_diff': None,
            'max_abs_diff': None,
        }
    
    # Convert to float for calculations
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    
    # Mean Squared Error
    mse = np.mean((img1_f - img2_f) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Structural Similarity Index (simplified version)
    # Using correlation as a proxy for SSIM
    img1_flat = img1_f.flatten()
    img2_flat = img2_f.flatten()
    
    # Suppress warnings for correlation calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(img1_flat, img2_flat)
        correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
    
    # Simple SSIM calculation (per channel, then average)
    ssim_channels = []
    for i in range(3):
        c1 = img1_f[:, :, i]
        c2 = img2_f[:, :, i]
        
        mu1 = np.mean(c1)
        mu2 = np.mean(c2)
        sigma1 = np.std(c1)
        sigma2 = np.std(c2)
        sigma12 = np.mean((c1 - mu1) * (c2 - mu2))
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Avoid division by zero
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
        if denominator > 0:
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / denominator
        else:
            ssim = 1.0  # Perfect similarity if both are constant
        ssim_channels.append(ssim)
    
    ssim_avg = np.mean(ssim_channels)
    
    # Absolute differences
    abs_diff = np.abs(img1_f - img2_f)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    
    return {
        'size_match': True,
        'size_diff': 'match',
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_avg,
        'correlation': correlation,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'img1_mean': np.mean(img1_f),
        'img2_mean': np.mean(img2_f),
        'img1_std': np.std(img1_f),
        'img2_std': np.std(img2_f),
    }


def create_comparison_image(pseudo_img, ecm_img, output_path, tile_name, metrics):
    """
    Create side-by-side comparison with difference heatmap.
    
    Args:
        pseudo_img: Pseudocolored image (numpy array)
        ecm_img: ECM image (numpy array)
        output_path: Path to save comparison image
        tile_name: Name of the tile for title
        metrics: Dictionary of computed metrics
    """
    # Reduce figure size and DPI for faster generation
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top left: Pseudocolored (downsample for display if large)
    display_pseudo = pseudo_img[::2, ::2] if pseudo_img.shape[0] > 1024 else pseudo_img
    axes[0, 0].imshow(display_pseudo)
    axes[0, 0].set_title(f'Pseudocolored\n{tile_name}', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top right: ECM
    display_ecm = ecm_img[::2, ::2] if ecm_img.shape[0] > 1024 else ecm_img
    axes[0, 1].imshow(display_ecm)
    axes[0, 1].set_title(f'ECM Channel\n{tile_name}', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom left: Difference heatmap (grayscale difference)
    if metrics['size_match']:
        diff = np.abs(pseudo_img.astype(np.float32) - ecm_img.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)  # Average across RGB channels
        # Downsample difference map too
        display_diff = diff_gray[::2, ::2] if diff_gray.shape[0] > 1024 else diff_gray
        im = axes[1, 0].imshow(display_diff, cmap='hot', vmin=0, vmax=255)
        axes[1, 0].set_title('Absolute Difference (Grayscale)', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        axes[1, 0].text(0.5, 0.5, 'Size Mismatch', 
                       ha='center', va='center', fontsize=16, color='red')
        axes[1, 0].axis('off')
    
    # Bottom right: Metrics text
    axes[1, 1].axis('off')
    if metrics['size_match']:
        metrics_text = f"""
Comparison Metrics:

SSIM: {metrics['ssim']:.4f}
Correlation: {metrics['correlation']:.4f}
MSE: {metrics['mse']:.2f}
PSNR: {metrics['psnr']:.2f} dB

Mean Abs Diff: {metrics['mean_abs_diff']:.2f}
Max Abs Diff: {metrics['max_abs_diff']:.2f}

Pseudocolored:
  Mean: {metrics['img1_mean']:.2f}
  Std: {metrics['img1_std']:.2f}

ECM:
  Mean: {metrics['img2_mean']:.2f}
  Std: {metrics['img2_std']:.2f}
"""
    else:
        metrics_text = f"Size Mismatch:\n{metrics['size_diff']}"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, 
                    fontsize=10, family='monospace', 
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close('all')  # Close all figures to free memory


def collect_tiles_by_wsi(directory):
    """
    Collect tiles organized by WSI base name.
    
    Returns:
        dict: {wsi_base: [tile_filenames]}
    """
    tiles_by_wsi = defaultdict(list)
    
    for f in Path(directory).iterdir():
        if f.is_file() and '_grid_' in f.name and not f.name.startswith('.'):
            base = f.name.split('_grid_')[0]
            tiles_by_wsi[base].append(f.name)
    
    return tiles_by_wsi


def stratified_sample(tiles_by_wsi, n_total):
    """
    Sample tiles ensuring each WSI is represented.
    
    Args:
        tiles_by_wsi: Dictionary of {wsi_base: [tile_filenames]}
        n_total: Total number of tiles to sample
    
    Returns:
        List of sampled tile filenames
    """
    wsi_bases = sorted(tiles_by_wsi.keys())
    n_wsi = len(wsi_bases)
    
    if n_wsi == 0:
        return []
    
    # Calculate tiles per WSI (at least 1 per WSI)
    base_per_wsi = max(1, n_total // n_wsi)
    remainder = n_total - (base_per_wsi * n_wsi)
    
    sampled = []
    
    for i, wsi_base in enumerate(wsi_bases):
        available_tiles = tiles_by_wsi[wsi_base]
        n_to_sample = base_per_wsi + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(available_tiles))
        
        sampled.extend(random.sample(available_tiles, n_to_sample))
    
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description='Compare Pseudocolored and ECM_channel tiles'
    )
    parser.add_argument(
        '--pseudo-dir',
        type=Path,
        default=Path('/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/Pseudocolored'),
        help='Pseudocolored tiles directory'
    )
    parser.add_argument(
        '--ecm-dir',
        type=Path,
        default=Path('/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/ECM_channel'),
        help='ECM_channel tiles directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/home/luci/adipose_tissue-unet/pre-post-processing_tools/tile_comparison'),
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--n-perfect',
        type=int,
        default=30,
        help='Number of tiles to sample from perfect matches'
    )
    parser.add_argument(
        '--n-mismatch',
        type=int,
        default=30,
        help='Number of tiles to sample from mismatches'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    perfect_dir = output_dir / 'perfect_matches'
    mismatch_dir = output_dir / 'mismatches'
    perfect_dir.mkdir(exist_ok=True)
    mismatch_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Pseudocolored vs ECM Channel Tile Comparison")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Collect tiles from both directories
    print("\nCollecting tiles...")
    pseudo_tiles = collect_tiles_by_wsi(args.pseudo_dir)
    ecm_tiles = collect_tiles_by_wsi(args.ecm_dir)
    
    # Define perfect matches and mismatches based on previous analysis
    perfect_matches = [
        '12.12 Beef Shoulder 2',
        '2nd Meat',
        '4th Meat, 400 g, 3 wts',
        '5th meat, 400g, placeholder and black plate',
        'MeatLarge'
    ]
    
    mismatches = [
        '12.12 Beef Shoulder 3',
        '3rd Meat',
        '6 BEEF Shoulder -1',
        'B-1',
        'B. Shoulder 1.2',
        'B. Shoulder 1.3',
        'Beef Shoulder 4',
        'Meat_Dylan'
    ]
    
    # Filter tiles by category
    perfect_tiles_by_wsi = {k: v for k, v in pseudo_tiles.items() if k in perfect_matches}
    mismatch_tiles_by_wsi = {k: v for k, v in pseudo_tiles.items() if k in mismatches}
    
    print(f"\nPerfect match WSI: {len(perfect_tiles_by_wsi)}")
    print(f"Mismatch WSI: {len(mismatch_tiles_by_wsi)}")
    
    # Sample tiles
    print(f"\nSampling {args.n_perfect} tiles from perfect matches (stratified)...")
    perfect_samples = stratified_sample(perfect_tiles_by_wsi, args.n_perfect)
    
    print(f"Sampling {args.n_mismatch} tiles from mismatches (stratified)...")
    mismatch_samples = stratified_sample(mismatch_tiles_by_wsi, args.n_mismatch)
    
    print(f"\nTotal samples: {len(perfect_samples) + len(mismatch_samples)}")
    
    # Prepare CSV output
    csv_path = output_dir / 'comparison_metrics.csv'
    csv_fields = [
        'tile_name', 'category', 'wsi_base',
        'size_match', 'size_diff',
        'ssim', 'correlation', 'mse', 'psnr',
        'mean_abs_diff', 'max_abs_diff',
        'pseudo_mean', 'pseudo_std',
        'ecm_mean', 'ecm_std'
    ]
    
    # Process tiles
    print("\nProcessing tiles...")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        
        # Process perfect matches
        print(f"\n1. Processing {len(perfect_samples)} perfect match tiles...")
        for i, tile_name in enumerate(perfect_samples, 1):
            print(f"  [{i}/{len(perfect_samples)}] {tile_name}")
            
            wsi_base = tile_name.split('_grid_')[0]
            pseudo_path = args.pseudo_dir / tile_name
            ecm_path = args.ecm_dir / tile_name
            
            if not ecm_path.exists():
                print(f"    WARNING: ECM tile not found: {tile_name}")
                continue
            
            # Load images
            pseudo_img = load_image(pseudo_path)
            ecm_img = load_image(ecm_path)
            
            # Compute metrics
            metrics = compute_metrics(pseudo_img, ecm_img)
            
            # Create comparison image
            output_path = perfect_dir / f"{tile_name.replace('.jpg', '_comparison.png')}"
            create_comparison_image(pseudo_img, ecm_img, output_path, tile_name, metrics)
            
            # Write to CSV
            row = {
                'tile_name': tile_name,
                'category': 'perfect_match',
                'wsi_base': wsi_base,
                'size_match': metrics['size_match'],
                'size_diff': metrics['size_diff'],
                'ssim': metrics.get('ssim', ''),
                'correlation': metrics.get('correlation', ''),
                'mse': metrics.get('mse', ''),
                'psnr': metrics.get('psnr', ''),
                'mean_abs_diff': metrics.get('mean_abs_diff', ''),
                'max_abs_diff': metrics.get('max_abs_diff', ''),
                'pseudo_mean': metrics.get('img1_mean', ''),
                'pseudo_std': metrics.get('img1_std', ''),
                'ecm_mean': metrics.get('img2_mean', ''),
                'ecm_std': metrics.get('img2_std', ''),
            }
            writer.writerow(row)
        
        # Process mismatches
        print(f"\n2. Processing {len(mismatch_samples)} mismatch tiles...")
        for i, tile_name in enumerate(mismatch_samples, 1):
            print(f"  [{i}/{len(mismatch_samples)}] {tile_name}")
            
            wsi_base = tile_name.split('_grid_')[0]
            pseudo_path = args.pseudo_dir / tile_name
            ecm_path = args.ecm_dir / tile_name
            
            if not ecm_path.exists():
                print(f"    WARNING: ECM tile not found: {tile_name}")
                continue
            
            # Load images
            pseudo_img = load_image(pseudo_path)
            ecm_img = load_image(ecm_path)
            
            # Compute metrics
            metrics = compute_metrics(pseudo_img, ecm_img)
            
            # Create comparison image
            output_path = mismatch_dir / f"{tile_name.replace('.jpg', '_comparison.png')}"
            create_comparison_image(pseudo_img, ecm_img, output_path, tile_name, metrics)
            
            # Write to CSV
            row = {
                'tile_name': tile_name,
                'category': 'mismatch',
                'wsi_base': wsi_base,
                'size_match': metrics['size_match'],
                'size_diff': metrics['size_diff'],
                'ssim': metrics.get('ssim', ''),
                'correlation': metrics.get('correlation', ''),
                'mse': metrics.get('mse', ''),
                'psnr': metrics.get('psnr', ''),
                'mean_abs_diff': metrics.get('mean_abs_diff', ''),
                'max_abs_diff': metrics.get('max_abs_diff', ''),
                'pseudo_mean': metrics.get('img1_mean', ''),
                'pseudo_std': metrics.get('img1_std', ''),
                'ecm_mean': metrics.get('img2_mean', ''),
                'ecm_std': metrics.get('img2_std', ''),
            }
            writer.writerow(row)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"CSV metrics: {csv_path}")
    print(f"Perfect match comparisons: {perfect_dir}")
    print(f"Mismatch comparisons: {mismatch_dir}")
    print(f"Total comparisons generated: {len(perfect_samples) + len(mismatch_samples)}")


if __name__ == '__main__':
    main()
