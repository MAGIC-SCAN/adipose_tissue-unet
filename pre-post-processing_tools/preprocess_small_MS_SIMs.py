#!/usr/bin/env python3
"""
Preprocess ECM Channel Images from Meat_MS_Tulane Dataset

Addresses common issues in fluorescence microscopy images:
1. Vertical banding (scanner/compression artifacts)
2. Uneven illumination (gradient across frame)
3. Low contrast

USAGE EXAMPLES:

1. Basic preprocessing (FFT deband + rolling ball + contrast enhancement):
   python pre-post-processing_tools/preprocess_small_MS_SIMs.py \
     --input-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/ECM_channel \
     --output-dir /home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/ECM_channel/corrected \
     --banding-method fft \
     --illumination-method rolling-ball \
     --enhance-contrast

2. Only remove banding (no illumination correction):
   python pre-post-processing_tools/preprocess_small_MS_SIMs.py \
     --input-dir data/Meat_MS_Tulane/ECM_channel \
     --output-dir data/Meat_MS_Tulane/ECM_channel/corrected \
     --banding-method fft \
     --illumination-method none

3. Visualize corrections (saves before/after comparisons):
   python pre-post-processing_tools/preprocess_small_MS_SIMs.py \
     --input-dir data/Meat_MS_Tulane/ECM_channel \
     --output-dir data/Meat_MS_Tulane/ECM_channel/corrected \
     --banding-method fft \
     --illumination-method rolling-ball \
     --enhance-contrast \
     --visualize

4. Polynomial illumination correction (alternative to rolling ball):
   python pre-post-processing_tools/preprocess_small_MS_SIMs.py \
     --input-dir data/Meat_MS_Tulane/ECM_channel \
     --output-dir data/Meat_MS_Tulane/ECM_channel/corrected \
     --banding-method fft \
     --illumination-method polynomial \
     --enhance-contrast

5. Gaussian illumination correction (fastest):
   python pre-post-processing_tools/preprocess_small_MS_SIMs.py \
     --input-dir data/Meat_MS_Tulane/ECM_channel \
     --output-dir data/Meat_MS_Tulane/ECM_channel/corrected \
     --banding-method fft \
     --illumination-method gaussian \
     --enhance-contrast

METHODS:
  Banding Removal:
    - fft: FFT-based frequency filtering (recommended)
    - none: Skip banding correction
  
  Illumination Correction:
    - rolling-ball: Morphological rolling ball (best quality, slowest)
    - polynomial: 2D polynomial surface fitting (balanced)
    - gaussian: Gaussian blur subtraction (fastest)
    - none: Skip illumination correction
  
  Contrast Enhancement:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)

OUTPUT:
  output_dir/
    ├── corrected_*.jpg          # Preprocessed images
    ├── processing_log.json      # Settings and statistics
    └── visualization/           # Before/after comparisons (if --visualize)

Author: MAGIC-SCAN
Date: 2024-11-14
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import json
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ============================================================================
# NORMALIZATION METHODS (from large_wsi_to_small_wsi_2.py)
# ============================================================================

def normalize_zscore(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalization for contrast enhancement.
    Matches training pipeline normalization.
    
    Args:
        arr: Input image array (any bit depth)
        
    Returns:
        Enhanced 8-bit image
    """
    # Convert to float for processing
    arr_float = arr.astype(np.float32)
    
    # Z-score normalization
    mean = arr_float.mean()
    std = arr_float.std() + 1e-10
    normalized = (arr_float - mean) / std
    
    # Stretch to [0, 255] for visualization (±3 std devs covers ~99.7%)
    stretched = (normalized + 3) / 6 * 255
    
    return np.clip(stretched, 0, 255).astype(np.uint8)


def normalize_percentile(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Percentile-based normalization for robust contrast enhancement.
    Matches training pipeline percentile normalization.
    
    Args:
        arr: Input image array (any bit depth)
        p_low: Lower percentile (default: 1.0)
        p_high: Upper percentile (default: 99.0)
        
    Returns:
        Enhanced 8-bit image
    """
    # Convert to float for processing
    arr_float = arr.astype(np.float32)
    
    # Percentile normalization
    plow, phigh = np.percentile(arr_float, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    normalized = np.clip((arr_float - plow) / scale, 0, 1)
    
    return (normalized * 255).astype(np.uint8)


# ============================================================================
# BANDING REMOVAL METHODS
# ============================================================================

def remove_banding_fft(img: np.ndarray, band_freq_range: Tuple[float, float] = (0.01, 0.05),
                       vertical_width: int = 3, smooth_sigma_scale: float = 0.5,
                       blend: float = 1.0) -> np.ndarray:
    """
    Remove vertical banding using FFT-based notch filtering.
    
    Vertical banding in spatial domain = horizontal lines in frequency domain.
    We create a notch filter to suppress these specific frequencies.
    
    Args:
        img: Input grayscale image (uint8)
        band_freq_range: Relative frequency range to suppress (0-0.5)
                        Default (0.01, 0.05) catches typical scanner banding
        vertical_width: Width of notch filter in pixels (higher = more aggressive)
        smooth_sigma_scale: Scale factor that controls how gradual the notch edges are.
        blend: Blend filtered image with the original (1.0 = full filter).
    
    Returns:
        Filtered image (uint8)
    """
    # Convert to float for FFT
    img_float = img.astype(np.float32)
    
    # 2D FFT
    f_transform = np.fft.fft2(img_float)
    f_shift = np.fft.fftshift(f_transform)
    
    # Create notch filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Normalize target frequency range
    freq_low = max(min(band_freq_range[0], 0.5), 0.0)
    freq_high = max(min(band_freq_range[1], 0.5), freq_low + 1e-4)
    band_width = max(freq_high - freq_low, 1e-4)
    center_freq = (freq_low + freq_high) / 2.0
    
    # Smooth Gaussian notch filter (reduces ringing artifacts)
    mask = np.ones((rows, cols), dtype=np.float32)
    
    # Normalized coordinate grids relative to center
    y_idx = (np.arange(rows) - crow) / rows
    x_idx = (np.arange(cols) - ccol) / cols
    y_grid, x_grid = np.meshgrid(y_idx, x_idx, indexing='ij')
    
    sigma_x = max(band_width * smooth_sigma_scale, band_width * 0.25)
    sigma_y = max((vertical_width / rows) * smooth_sigma_scale, 1.0 / rows)
    
    gaussian_left = np.exp(-0.5 * (((x_grid + center_freq) / sigma_x) ** 2 + (y_grid / sigma_y) ** 2))
    gaussian_right = np.exp(-0.5 * (((x_grid - center_freq) / sigma_x) ** 2 + (y_grid / sigma_y) ** 2))
    
    notch = np.clip(gaussian_left + gaussian_right, 0.0, 1.0)
    mask -= notch
    mask = np.clip(mask, 0.0, 1.0)
    
    # Apply filter
    f_shift_filtered = f_shift * mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    
    # Clip and convert back to uint8
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    if blend < 1.0:
        img_filtered = cv2.addWeighted(img, 1.0 - blend, img_filtered, blend, 0)
    
    return img_filtered


def remove_banding_morphological(img: np.ndarray, kernel_width: int = 1,
                                  kernel_height: int = 512) -> np.ndarray:
    """
    Remove vertical banding by morphological background subtraction.
    
    Uses tall narrow kernel to estimate the vertical banding pattern,
    then subtracts it from the original image.
    
    Args:
        img: Input grayscale image (uint8)
        kernel_width: Horizontal kernel size (1 = pure vertical)
        kernel_height: Vertical kernel size (large for averaging)
                      Default 512 works well for 6144×6144 tiles
    
    Returns:
        Corrected image (uint8)
    """
    # Estimate background pattern using morphological opening
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Subtract background and add back mean to preserve overall intensity
    img_float = img.astype(np.float32)
    bg_float = background.astype(np.float32)
    bg_mean = bg_float.mean()
    
    corrected = img_float - bg_float + bg_mean
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected


def remove_banding_column_normalize(img: np.ndarray, preserve_global: bool = True) -> np.ndarray:
    """
    Remove vertical banding by normalizing each column independently.
    
    WARNING: This method may remove real vertical tissue structures!
    Use only if banding is severe and other methods fail.
    
    Args:
        img: Input grayscale image (uint8)
        preserve_global: If True, re-scale to preserve global mean/std
    
    Returns:
        Normalized image (uint8)
    """
    img_float = img.astype(np.float32)
    
    # Store global statistics
    global_mean = img_float.mean()
    global_std = img_float.std()
    
    # Compute per-column statistics
    col_means = img_float.mean(axis=0, keepdims=True)
    col_stds = img_float.std(axis=0, keepdims=True) + 1e-10
    
    # Normalize each column to zero mean, unit variance
    img_normalized = (img_float - col_means) / col_stds
    
    if preserve_global:
        # Re-scale to original global statistics
        img_normalized = img_normalized * global_std + global_mean
    else:
        # Scale to [0, 255] range
        img_normalized = ((img_normalized - img_normalized.min()) / 
                         (img_normalized.max() - img_normalized.min() + 1e-10) * 255)
    
    img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
    
    return img_normalized


# ============================================================================
# ILLUMINATION CORRECTION METHODS
# ============================================================================

def correct_illumination_rolling_ball(img: np.ndarray, radius: int = 100) -> np.ndarray:
    """
    Rolling ball background subtraction - excellent for gradient illumination.
    
    Simulates rolling a ball under the intensity surface to estimate background.
    Works well for gradual illumination changes (top darker, bottom brighter).
    
    Args:
        img: Input grayscale image (uint8)
        radius: Ball radius in pixels (50-200 typical for microscopy)
                Larger radius = smoother background estimation
    
    Returns:
        Background-corrected image (uint8)
    """
    img_float = img.astype(np.float32)
    
    # Create ball-shaped structuring element
    kernel_size = radius * 2 + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    ball = np.sqrt(x**2 + y**2) <= radius
    kernel = ball.astype(np.uint8)
    
    # Morphological opening with ball kernel estimates background
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Subtract background, preserve mean
    bg_float = background.astype(np.float32)
    bg_mean = bg_float.mean()
    
    corrected = img_float - bg_float + bg_mean
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected


def correct_illumination_polynomial(img: np.ndarray, smoothing_sigma: float = 150) -> np.ndarray:
    """
    Polynomial surface fitting for illumination correction.
    
    Estimates illumination field by heavy Gaussian blur, then normalizes by it.
    Good for gradual illumination gradients.
    
    Args:
        img: Input grayscale image (uint8)
        smoothing_sigma: Gaussian blur sigma (100-200 typical)
                        Larger = smoother background estimation
    
    Returns:
        Illumination-corrected image (uint8)
    """
    img_float = img.astype(np.float32)
    
    # Heavily blur to extract low-frequency illumination pattern
    background = cv2.GaussianBlur(img_float, (0, 0), smoothing_sigma)
    
    # Normalize by background (multiplicative correction)
    bg_mean = background.mean()
    corrected = img_float * (bg_mean / (background + 1.0))  # +1 avoids div by zero
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected


def correct_illumination_tophat(img: np.ndarray, kernel_size: int = 301) -> np.ndarray:
    """
    White top-hat transform for illumination correction.
    
    Top-hat = original - morphological opening
    Removes large-scale structures (illumination) while preserving details.
    
    Args:
        img: Input grayscale image (uint8)
        kernel_size: Size of structuring element (must be odd)
                    Larger = removes more global variation
    
    Returns:
        Corrected image (uint8)
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # White top-hat: img - opening(img)
    # Emphasizes bright structures smaller than kernel
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    # Add back to reduce overall darkening
    img_float = img.astype(np.float32)
    tophat_float = tophat.astype(np.float32)
    
    # Normalize to maintain brightness
    corrected = img_float + tophat_float * 0.5
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected


def correct_illumination_adaptive_histogram(img: np.ndarray, tile_size: int = 16,
                                             clip_limit: float = 2.0) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Local histogram equalization handles varying illumination by processing
    image in tiles. Also enhances contrast.
    
    Args:
        img: Input grayscale image (uint8)
        tile_size: Grid size for local processing (8-32 typical)
        clip_limit: Contrast limit to prevent over-amplification
    
    Returns:
        Locally normalized image (uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


# ============================================================================
# CONTRAST ENHANCEMENT
# ============================================================================

def enhance_contrast_clahe(img: np.ndarray, tile_size: int = 16,
                           clip_limit: float = 3.0) -> np.ndarray:
    """
    CLAHE for contrast enhancement.
    
    Args:
        img: Input grayscale image (uint8)
        tile_size: Grid size for local processing
        clip_limit: Contrast limit (2-4 typical, higher = more aggressive)
    
    Returns:
        Enhanced image (uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def sharpen_image(img: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    """
    Unsharp mask sharpening.
    
    Sharpened = original + amount * (original - blurred)
    
    Args:
        img: Input grayscale image (uint8)
        sigma: Gaussian blur sigma for unsharp mask
        amount: Sharpening strength (0.3-1.0 typical)
    
    Returns:
        Sharpened image (uint8)
    """
    img_float = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    
    # Unsharp mask
    sharpened = img_float + amount * (img_float - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened


# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_ecm_image(img: np.ndarray,
                         banding_method: Optional[str] = None,
                         banding_params: dict = {},
                         normalization_method: Optional[str] = None,
                         normalization_params: dict = {},
                         illumination_method: Optional[str] = None,
                         illumination_params: dict = {},
                         enhance_contrast: bool = False,
                         contrast_params: dict = {},
                         sharpen: bool = False,
                         sharpen_params: dict = {}) -> np.ndarray:
    """
    Complete preprocessing pipeline for ECM channel images.
    
    Args:
        img: Input image (can be RGB or grayscale, uint8)
        banding_method: 'fft', 'morphological', 'column', or None
        banding_params: Parameters for banding removal
        normalization_method: 'percentile', 'zscore', or None
        normalization_params: Parameters for normalization
        illumination_method: 'rolling-ball', 'polynomial', 'tophat', 'clahe', or None
        illumination_params: Parameters for illumination correction
        enhance_contrast: Apply CLAHE contrast enhancement
        contrast_params: Parameters for CLAHE
        sharpen: Apply unsharp mask sharpening
        sharpen_params: Parameters for sharpening
    
    Returns:
        Preprocessed image (uint8, grayscale)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 4:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img_gray = img[:, :, 0]  # Take first channel
    else:
        img_gray = img.copy()
    
    result = img_gray.copy()
    
    # Step 1: Banding removal
    if banding_method == 'fft':
        result = remove_banding_fft(result, **banding_params)
    elif banding_method == 'morphological':
        result = remove_banding_morphological(result, **banding_params)
    elif banding_method == 'column':
        result = remove_banding_column_normalize(result, **banding_params)
    
    # Step 2: Normalization (percentile or zscore)
    if normalization_method == 'percentile':
        result = normalize_percentile(result, **normalization_params)
    elif normalization_method == 'zscore':
        result = normalize_zscore(result)
    
    # Step 3: Illumination correction
    if illumination_method == 'rolling-ball':
        result = correct_illumination_rolling_ball(result, **illumination_params)
    elif illumination_method == 'polynomial':
        result = correct_illumination_polynomial(result, **illumination_params)
    elif illumination_method == 'tophat':
        result = correct_illumination_tophat(result, **illumination_params)
    elif illumination_method == 'clahe':
        result = correct_illumination_adaptive_histogram(result, **illumination_params)
    
    # Step 4: Enhance contrast
    if enhance_contrast:
        result = enhance_contrast_clahe(result, **contrast_params)
    
    # Step 5: Sharpen
    if sharpen:
        result = sharpen_image(result, **sharpen_params)
    
    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_visualization(original: np.ndarray, enhanced: np.ndarray, processed: np.ndarray,
                                    image_name: str, output_path: Path,
                                    banding_method: str = None,
                                    normalization_method: str = None,
                                    illumination_method: str = None) -> None:
    """
    Create three-way comparison visualization.
    
    Args:
        original: Original JPEG image from ECM_channel/
        enhanced: Enhanced image from ECM_channel/enhanced/ (matching normalization method)
        processed: Our processed output
        image_name: Name for the plot
        output_path: Where to save comparison
        banding_method: Banding removal method used
        normalization_method: Normalization method used
        illumination_method: Illumination correction method used
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Enhanced image (from large_wsi_to_small_wsi_2.py)
    axes[0].imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    enhanced_label = normalization_method if normalization_method else 'enhanced'
    axes[0].set_title(f'Enhanced ({enhanced_label})\nMean: {enhanced.mean():.1f}, Std: {enhanced.std():.1f}',
                     fontsize=12)
    axes[0].axis('off')
    
    # Processed image
    axes[1].imshow(processed, cmap='gray', vmin=0, vmax=255)
    
    # Build title for processed image
    title_parts = ['Processed']
    if banding_method:
        title_parts.append(f'Banding: {banding_method}')
    if normalization_method:
        title_parts.append(f'Norm: {normalization_method}')
    if illumination_method:
        title_parts.append(f'Illum: {illumination_method}')
    title_parts.append(f'Mean: {processed.mean():.1f}, Std: {processed.std():.1f}')
    
    axes[1].set_title('\n'.join(title_parts), fontsize=12)
    axes[1].axis('off')
    
    plt.suptitle(image_name, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================

def process_directory(input_dir: Path, output_dir: Path, args):
    """Process all images in input directory."""
    
    # Find all JPEG image files in main directory
    image_extensions = ('.jpg', '.jpeg')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    # Filter out enhanced versions (those with suffixes like _percentile, _zscore, _clahe)
    image_files = [f for f in image_files if not any(suffix in f.stem for suffix in ['_percentile', '_zscore', '_clahe'])]
    
    if not image_files:
        print(f"No JPEG images found in {input_dir}")
        return
    
    # Test mode: randomly sample images
    if args.test_mode:
        import random
        random.seed(865)  # Reproducible sampling
        
        # Always include this specific image
        fixed_image = input_dir / "Meat_Strip1_x0_y18432_w6144_h6144.jpg"
        
        # Get random sample
        available_images = [f for f in image_files if f != fixed_image]
        if len(available_images) > args.test_samples:
            sampled_images = random.sample(available_images, args.test_samples)
        else:
            sampled_images = available_images
        
        # Add fixed image if it exists
        if fixed_image.exists() and fixed_image in image_files:
            sampled_images.append(fixed_image)
            print(f"Test mode: Processing {len(sampled_images)} images ({args.test_samples} random + 1 fixed)")
        else:
            print(f"Test mode: Processing {len(sampled_images)} randomly sampled images")
            if fixed_image.exists():
                print(f"  Warning: Fixed image not in filtered list: {fixed_image.name}")
        
        image_files = sampled_images
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if requested
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    # Prepare preprocessing parameters
    banding_params = {}
    if args.banding_method == 'fft':
        banding_params = {
            'band_freq_range': (args.fft_freq_low, args.fft_freq_high),
            'vertical_width': args.fft_width,
            'smooth_sigma_scale': args.fft_sigma_scale,
            'blend': args.fft_blend
        }
    elif args.banding_method == 'morphological':
        banding_params = {
            'kernel_width': args.morph_width,
            'kernel_height': args.morph_height
        }
    elif args.banding_method == 'column':
        banding_params = {
            'preserve_global': args.column_preserve_global
        }
    
    normalization_params = {}
    if args.normalization_method == 'percentile':
        normalization_params = {
            'p_low': args.percentile_low,
            'p_high': args.percentile_high
        }
    
    illumination_params = {}
    if args.illumination_method == 'rolling-ball':
        illumination_params = {'radius': args.rolling_ball_radius}
    elif args.illumination_method == 'polynomial':
        illumination_params = {'smoothing_sigma': args.poly_sigma}
    elif args.illumination_method == 'tophat':
        illumination_params = {'kernel_size': args.tophat_kernel}
    elif args.illumination_method == 'clahe':
        illumination_params = {
            'tile_size': args.clahe_illum_tile,
            'clip_limit': args.clahe_illum_clip
        }
    
    contrast_params = {
        'tile_size': args.clahe_tile_size,
        'clip_limit': args.clahe_clip_limit
    }
    
    sharpen_params = {
        'sigma': args.sharpen_sigma,
        'amount': args.sharpen_amount
    }
    
    # Process each image
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
        'original_mean': [],
        'processed_mean': [],
        'original_std': [],
        'processed_std': []
    }
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to load: {img_path}")
                stats['failed'] += 1
                continue
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Store original for comparison
            if len(img.shape) == 3:
                original_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                original_gray = img.copy()
            
            # Preprocess
            processed = preprocess_ecm_image(
                img,
                banding_method=args.banding_method,
                banding_params=banding_params,
                normalization_method=args.normalization_method,
                normalization_params=normalization_params,
                illumination_method=args.illumination_method,
                illumination_params=illumination_params,
                enhance_contrast=args.enhance_contrast,
                contrast_params=contrast_params,
                sharpen=args.sharpen,
                sharpen_params=sharpen_params
            )
            
            # Save processed image
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), processed, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 9] if output_path.suffix.lower() == '.png' else [])
            
            # Collect statistics
            stats['processed'] += 1
            stats['original_mean'].append(original_gray.mean())
            stats['processed_mean'].append(processed.mean())
            stats['original_std'].append(original_gray.std())
            stats['processed_std'].append(processed.std())
            
            # Create visualization if requested
            if args.visualize and stats['processed'] <= args.max_visualizations:
                vis_path = vis_dir / f"{img_path.stem}_comparison.png"
                
                # Load corresponding enhanced image from enhanced/ subdirectory
                enhanced_dir = input_dir / "enhanced"
                enhanced_img = None
                if enhanced_dir.exists() and args.normalization_method:
                    # Match the enhanced file based on normalization method
                    enhanced_suffix = f"_{args.normalization_method}.jpg"
                    enhanced_path = enhanced_dir / f"{img_path.stem}{enhanced_suffix}"
                    if enhanced_path.exists():
                        enhanced_img = cv2.imread(str(enhanced_path), cv2.IMREAD_GRAYSCALE)
                
                # If no enhanced image found, use original for middle panel
                if enhanced_img is None:
                    enhanced_img = original_gray.copy()
                
                create_comparison_visualization(
                    original_gray, enhanced_img, processed,
                    img_path.name, vis_path,
                    args.banding_method, args.normalization_method, args.illumination_method
                )
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            stats['failed'] += 1
            continue
    
    # Save processing report
    report_path = output_dir / "processing_report.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'settings': {
            'banding_method': args.banding_method,
            'banding_params': banding_params,
            'normalization_method': args.normalization_method,
            'normalization_params': normalization_params if args.normalization_method else None,
            'illumination_method': args.illumination_method,
            'illumination_params': illumination_params,
            'enhance_contrast': args.enhance_contrast,
            'contrast_params': contrast_params if args.enhance_contrast else None,
            'sharpen': args.sharpen,
            'sharpen_params': sharpen_params if args.sharpen else None,
        },
        'statistics': {
            'total_images': stats['total'],
            'processed': stats['processed'],
            'failed': stats['failed'],
            'original_mean_avg': float(np.mean(stats['original_mean'])) if stats['original_mean'] else 0,
            'processed_mean_avg': float(np.mean(stats['processed_mean'])) if stats['processed_mean'] else 0,
            'original_std_avg': float(np.mean(stats['original_std'])) if stats['original_std'] else 0,
            'processed_std_avg': float(np.mean(stats['processed_std'])) if stats['processed_std'] else 0,
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"{'='*80}")
    print(f"Total images: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Processing report: {report_path}")
    if args.visualize:
        print(f"Visualizations: {vis_dir}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ECM channel images from Meat_MS_Tulane dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O arguments
    parser.add_argument('--input-dir', type=str,
                       default='/home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/ECM_channel',
                       help='Input directory containing original ECM images')
    parser.add_argument('--output-dir', type=str,
                       default='/home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/ECM_channel/corrected',
                       help='Output directory for processed images')
    
    # Banding removal options
    banding_group = parser.add_argument_group('Banding Removal')
    banding_group.add_argument('--banding-method', type=str, choices=['fft', 'morphological', 'column', 'none'],
                              default='none',
                              help='Method to remove vertical banding')
    
    # FFT parameters
    banding_group.add_argument('--fft-freq-low', type=float, default=0.01,
                              help='FFT: Lower frequency bound (0-0.5)')
    banding_group.add_argument('--fft-freq-high', type=float, default=0.05,
                              help='FFT: Upper frequency bound (0-0.5)')
    banding_group.add_argument('--fft-width', type=int, default=3,
                              help='FFT: Notch filter width in pixels')
    banding_group.add_argument('--fft-sigma-scale', type=float, default=0.5,
                              help='FFT: Controls softness of notch edges (higher=smoother)')
    banding_group.add_argument('--fft-blend', type=float, default=1.0,
                              help='FFT: Blend factor with original (1=full filter, 0.7 keeps detail)')
    
    # Morphological parameters
    banding_group.add_argument('--morph-width', type=int, default=1,
                              help='Morphological: Kernel width (1=pure vertical)')
    banding_group.add_argument('--morph-height', type=int, default=512,
                              help='Morphological: Kernel height (larger=more averaging)')
    
    # Column normalization parameters
    banding_group.add_argument('--column-preserve-global', action='store_true', default=True,
                              help='Column: Preserve global mean/std after normalization')
    
    # Normalization options (from large_wsi_to_small_wsi_2.py)
    norm_group = parser.add_argument_group('Normalization')
    norm_group.add_argument('--normalization-method', type=str,
                           choices=['percentile', 'zscore', 'none'],
                           default='none',
                           help='Normalization method (percentile or zscore from large_wsi_to_small_wsi_2.py)')
    norm_group.add_argument('--percentile-low', type=float, default=1.0,
                           help='Percentile: Lower percentile bound (0-100)')
    norm_group.add_argument('--percentile-high', type=float, default=99.0,
                           help='Percentile: Upper percentile bound (0-100)')
    
    # Illumination correction options
    illum_group = parser.add_argument_group('Illumination Correction')
    illum_group.add_argument('--illumination-method', type=str,
                            choices=['rolling-ball', 'polynomial', 'tophat', 'clahe', 'none'],
                            default='none',
                            help='Method to correct uneven illumination')
    
    # Rolling ball parameters
    illum_group.add_argument('--rolling-ball-radius', type=int, default=100,
                            help='Rolling ball: Ball radius in pixels (50-200 typical)')
    
    # Polynomial parameters
    illum_group.add_argument('--poly-sigma', type=float, default=150,
                            help='Polynomial: Gaussian blur sigma for background estimation')
    
    # Top-hat parameters
    illum_group.add_argument('--tophat-kernel', type=int, default=301,
                            help='Top-hat: Kernel size (must be odd, larger=more correction)')
    
    # CLAHE for illumination parameters
    illum_group.add_argument('--clahe-illum-tile', type=int, default=16,
                            help='CLAHE illumination: Tile grid size')
    illum_group.add_argument('--clahe-illum-clip', type=float, default=2.0,
                            help='CLAHE illumination: Clip limit')
    
    # Contrast enhancement options
    contrast_group = parser.add_argument_group('Contrast Enhancement')
    contrast_group.add_argument('--enhance-contrast', action='store_true', default=False,
                               help='Apply CLAHE contrast enhancement')
    contrast_group.add_argument('--clahe-tile-size', type=int, default=16,
                               help='CLAHE: Tile grid size for contrast enhancement')
    contrast_group.add_argument('--clahe-clip-limit', type=float, default=3.0,
                               help='CLAHE: Clip limit for contrast enhancement (2-4 typical)')
    
    # Sharpening options
    sharpen_group = parser.add_argument_group('Sharpening')
    sharpen_group.add_argument('--sharpen', action='store_true', default=False,
                              help='Apply unsharp mask sharpening')
    sharpen_group.add_argument('--sharpen-sigma', type=float, default=1.0,
                              help='Sharpen: Gaussian blur sigma for unsharp mask')
    sharpen_group.add_argument('--sharpen-amount', type=float, default=0.5,
                              help='Sharpen: Sharpening strength (0.3-1.0 typical)')
    
    # Visualization options
    vis_group = parser.add_argument_group('Visualization')
    vis_group.add_argument('--visualize', action='store_true', default=False,
                          help='Generate before/after comparison images')
    vis_group.add_argument('--max-visualizations', type=int, default=10,
                          help='Maximum number of visualizations to generate')
    
    # Test mode options
    test_group = parser.add_argument_group('Test Mode')
    test_group.add_argument('--test-mode', action='store_true', default=False,
                           help='Test mode: randomly sample subset of images')
    test_group.add_argument('--test-samples', type=int, default=5,
                           help='Number of images to randomly sample in test mode (plus 1 fixed image)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Convert 'none' to None
    if args.banding_method == 'none':
        args.banding_method = None
    if args.illumination_method == 'none':
        args.illumination_method = None
    
    # Print configuration
    print("="*80)
    print("ECM CHANNEL IMAGE PREPROCESSING")
    print("="*80)
    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nMode: {'TEST MODE (sampling {0} images)'.format(args.test_samples) if args.test_mode else 'FULL PROCESSING'}")
    print(f"Target: Original JPEG files from ECM_channel/")
    print(f"\nProcessing pipeline:")
    print(f"  1. Banding removal:       {args.banding_method}")
    print(f"  2. Normalization:         {args.normalization_method}")
    print(f"  3. Illumination correct:  {args.illumination_method}")
    print(f"  4. Contrast enhancement:  {'CLAHE' if args.enhance_contrast else 'SKIP'}")
    print(f"  5. Sharpening:            {'Unsharp mask' if args.sharpen else 'SKIP'}")
    print(f"\nVisualization: {args.visualize}")
    if args.visualize and args.normalization_method != 'none':
        print(f"  - Comparing against: enhanced/*_{args.normalization_method}.jpg")
    print("="*80)
    print()
    
    # Process images
    process_directory(input_dir, output_dir, args)


if __name__ == '__main__':
    main()
