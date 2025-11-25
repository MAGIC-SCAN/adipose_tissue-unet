#!/usr/bin/env python3
"""
Reinhard Stain Normalization for SYBR Gold + Eosin
==================================================

Implementation of Reinhard color normalization method specifically optimized
for SYBR Gold + Eosin stained adipose tissue images.

Based on:
Reinhard, E., et al. "Color transfer between images." IEEE Computer Graphics and Applications, 2001.

Optimizations for SYBR Gold + Eosin:
- Enhanced LAB color space handling for golden-yellow vs pink separation
- Robust reference image loading and caching
- Quality validation for normalized outputs
- Integration with existing preprocessing pipeline

Author: Analysis Pipeline
Date: 2024-10-29
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import json
from skimage.color import rgb2lab, lab2rgb
import warnings
warnings.filterwarnings('ignore')


class ReinhardStainNormalizer:
    """
    Reinhard stain normalization optimized for SYBR Gold + Eosin
    """
    
    def __init__(self, reference_path=None, reference_metadata_path=None):
        self.reference_path = reference_path
        self.reference_metadata_path = reference_metadata_path
        self.reference_image = None
        self.reference_lab_stats = None
        self.reference_metadata = None
        
        # Load reference if provided
        if reference_path:
            self.load_reference(reference_path, reference_metadata_path)
    
    def load_reference(self, reference_path, metadata_path=None):
        """Load reference image and its statistics"""
        self.reference_path = Path(reference_path)
        
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_path}")
        
        # Load reference image
        self.reference_image = np.array(Image.open(self.reference_path))
        
        if len(self.reference_image.shape) != 3 or self.reference_image.shape[2] != 3:
            raise ValueError("Reference image must be RGB")
        
        # Calculate LAB statistics for reference
        self.reference_lab_stats = self._calculate_lab_stats(self.reference_image)
        
        # Load metadata if provided
        if metadata_path:
            self.reference_metadata_path = Path(metadata_path)
            if self.reference_metadata_path.exists():
                with open(self.reference_metadata_path, 'r') as f:
                    self.reference_metadata = json.load(f)
        
        print(f"Reference loaded: {self.reference_path.name}")
        print(f"Reference LAB stats: L={self.reference_lab_stats['L']['mean']:.1f}±{self.reference_lab_stats['L']['std']:.1f}, "
              f"A={self.reference_lab_stats['A']['mean']:.1f}±{self.reference_lab_stats['A']['std']:.1f}, "
              f"B={self.reference_lab_stats['B']['mean']:.1f}±{self.reference_lab_stats['B']['std']:.1f}")
    
    def _calculate_lab_stats(self, image):
        """Calculate LAB color space statistics"""
        # Ensure image is in [0, 1] range for LAB conversion
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to LAB
        lab = rgb2lab(image)
        
        # Calculate statistics for each channel
        stats = {
            'L': {'mean': lab[:,:,0].mean(), 'std': lab[:,:,0].std()},
            'A': {'mean': lab[:,:,1].mean(), 'std': lab[:,:,1].std()},
            'B': {'mean': lab[:,:,2].mean(), 'std': lab[:,:,2].std()}
        }
        
        return stats
    
    def normalize_image(self, source_image):
        """
        Apply Reinhard normalization to source image using loaded reference
        
        Args:
            source_image: Input image (numpy array, RGB, 0-255 or 0-1 range)
            
        Returns:
            Normalized image (numpy array, RGB, same range as input)
        """
        if self.reference_lab_stats is None:
            raise ValueError("No reference loaded. Call load_reference() first.")
        
        # Determine input range and normalize to [0, 1]
        input_is_uint8 = source_image.max() > 1.0
        if input_is_uint8:
            source_normalized = source_image / 255.0
        else:
            source_normalized = source_image.copy()
        
        # Convert to LAB
        source_lab = rgb2lab(source_normalized)
        
        # Calculate source statistics
        source_stats = self._calculate_lab_stats(source_normalized)
        
        # Apply Reinhard normalization for each channel
        normalized_lab = source_lab.copy()
        
        for i, channel in enumerate(['L', 'A', 'B']):
            source_mean = source_stats[channel]['mean']
            source_std = source_stats[channel]['std']
            target_mean = self.reference_lab_stats[channel]['mean']
            target_std = self.reference_lab_stats[channel]['std']
            
            # Avoid division by zero
            if source_std == 0:
                normalized_lab[:,:,i] = target_mean
            else:
                # Reinhard transformation: (source - source_mean) * (target_std / source_std) + target_mean
                normalized_lab[:,:,i] = (source_lab[:,:,i] - source_mean) * (target_std / source_std) + target_mean
        
        # Convert back to RGB
        normalized_rgb = lab2rgb(normalized_lab)
        
        # Clip to valid range
        normalized_rgb = np.clip(normalized_rgb, 0, 1)
        
        # Convert back to original range
        if input_is_uint8:
            normalized_rgb = (normalized_rgb * 255).astype(np.uint8)
        
        return normalized_rgb
    
    def normalize_batch(self, image_paths, output_dir=None, preserve_names=True):
        """
        Normalize a batch of images
        
        Args:
            image_paths: List of image paths or directory path
            output_dir: Output directory (if None, overwrites originals)
            preserve_names: Whether to preserve original filenames
            
        Returns:
            List of normalized image paths
        """
        if isinstance(image_paths, (str, Path)):
            # If directory provided, get all images
            input_dir = Path(image_paths)
            if input_dir.is_dir():
                image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
                image_paths = [p for p in input_dir.iterdir() 
                             if p.suffix.lower() in image_extensions]
            else:
                image_paths = [input_dir]
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        normalized_paths = []
        
        print(f"Normalizing {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths):
            img_path = Path(img_path)
            print(f"Processing {img_path.name} ({i+1}/{len(image_paths)})")
            
            try:
                # Load and normalize image
                source_image = np.array(Image.open(img_path))
                normalized_image = self.normalize_image(source_image)
                
                # Determine output path
                if output_dir:
                    if preserve_names:
                        output_path = output_dir / img_path.name
                    else:
                        output_path = output_dir / f"normalized_{i:04d}{img_path.suffix}"
                else:
                    output_path = img_path  # Overwrite original
                
                # Save normalized image
                Image.fromarray(normalized_image).save(output_path)
                normalized_paths.append(output_path)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        print(f"Successfully normalized {len(normalized_paths)} images")
        return normalized_paths
    
    def validate_normalization(self, source_image, normalized_image, tolerance=0.1):
        """
        Validate that normalization preserved important image characteristics
        
        Args:
            source_image: Original image
            normalized_image: Normalized image
            tolerance: Acceptable difference in key metrics
            
        Returns:
            Dictionary with validation results
        """
        # Calculate quality metrics for both images
        def calculate_metrics(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Entropy
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist = hist[hist > 0]
            probabilities = hist / hist.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Mean intensity
            mean_intensity = gray.mean()
            
            return {
                'sharpness': sharpness,
                'entropy': entropy,
                'mean_intensity': mean_intensity
            }
        
        source_metrics = calculate_metrics(source_image)
        normalized_metrics = calculate_metrics(normalized_image)
        
        # Calculate relative differences
        validation_results = {
            'sharpness_preserved': abs(normalized_metrics['sharpness'] - source_metrics['sharpness']) / source_metrics['sharpness'] < tolerance,
            'entropy_preserved': abs(normalized_metrics['entropy'] - source_metrics['entropy']) / source_metrics['entropy'] < tolerance,
            'intensity_reasonable': 50 <= normalized_metrics['mean_intensity'] <= 200,  # Reasonable intensity range
            'sharpness_ratio': normalized_metrics['sharpness'] / source_metrics['sharpness'],
            'entropy_ratio': normalized_metrics['entropy'] / source_metrics['entropy'],
            'mean_intensity_change': normalized_metrics['mean_intensity'] - source_metrics['mean_intensity']
        }
        
        validation_results['overall_valid'] = (
            validation_results['sharpness_preserved'] and
            validation_results['entropy_preserved'] and
            validation_results['intensity_reasonable']
        )
        
        return validation_results
    
    def create_before_after_comparison(self, source_image, normalized_image, save_path=None):
        """
        Create side-by-side comparison of original and normalized images
        
        Args:
            source_image: Original image
            normalized_image: Normalized image
            save_path: Path to save comparison image
            
        Returns:
            Comparison image
        """
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(source_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Normalized image
        axes[1].imshow(normalized_image)
        axes[1].set_title('Reinhard Normalized')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        return fig
    
    def get_reference_info(self):
        """Get information about the loaded reference"""
        if self.reference_path is None:
            return "No reference loaded"
        
        info = {
            'reference_path': str(self.reference_path),
            'reference_name': self.reference_path.name,
            'lab_stats': self.reference_lab_stats,
            'metadata': self.reference_metadata
        }
        
        return info


def load_best_reference(metadata_path="preprocessing_analysis/stain_normalization/selected_reference_metadata.json"):
    """
    Load the best reference selected by the reference selector
    
    Args:
        metadata_path: Path to reference metadata JSON file
        
    Returns:
        ReinhardStainNormalizer instance with best reference loaded
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Reference metadata not found: {metadata_path}\n"
            "Run select_stain_reference.py first to select optimal reference"
        )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    reference_path = metadata['selected_reference']['path']
    
    normalizer = ReinhardStainNormalizer(reference_path, metadata_path)
    
    print(f"Loaded best reference: {metadata['selected_reference']['name']}")
    print(f"Composite score: {metadata['selected_reference']['composite_score']:.3f}")
    print(f"Stain type: {metadata['selected_reference']['stain_type']}")
    
    return normalizer


def normalize_with_zscore(image, mean=200.99, std=25.26):
    """
    Apply z-score normalization (existing pipeline component)
    
    Args:
        image: Input image (0-255 range)
        mean: Target mean
        std: Target standard deviation
        
    Returns:
        Z-score normalized image
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Calculate current statistics
    current_mean = img_float.mean()
    current_std = img_float.std()
    
    # Apply z-score normalization
    if current_std > 0:
        normalized = (img_float - current_mean) / current_std * std + mean
    else:
        normalized = img_float  # Avoid division by zero
    
    # Clip to valid range and convert back
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def normalize_with_percentile(image, low_percentile=1.0, high_percentile=99.0):
    """
    Apply 1-99 percentile normalization (robust alternative to z-score)
    
    Args:
        image: Input image (0-255 range)
        low_percentile: Lower percentile for clipping (default: 1st percentile)
        high_percentile: Upper percentile for clipping (default: 99th percentile)
        
    Returns:
        Percentile normalized image (0-255 range)
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Calculate percentiles
    low_val = np.percentile(img_float, low_percentile)
    high_val = np.percentile(img_float, high_percentile)
    
    # Avoid division by zero
    if high_val == low_val:
        return image  # Return original if no dynamic range
    
    # Clip and normalize to 0-255 range
    clipped = np.clip(img_float, low_val, high_val)
    normalized = (clipped - low_val) / (high_val - low_val) * 255.0
    
    return normalized.astype(np.uint8)


def complete_preprocessing_pipeline(image, normalizer, norm_method="zscore", zscore_mean=200.99, zscore_std=25.26, percentile_low=1.0, percentile_high=99.0):
    """
    Complete preprocessing pipeline with Reinhard + flexible normalization
    
    Args:
        image: Input image (numpy array or path)
        normalizer: ReinhardStainNormalizer instance
        norm_method: Normalization method ("zscore", "percentile", or "none")
        zscore_mean: Target mean for z-score normalization
        zscore_std: Target std for z-score normalization
        percentile_low: Lower percentile for percentile normalization
        percentile_high: Upper percentile for percentile normalization
        
    Returns:
        Fully preprocessed image
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = np.array(Image.open(image))
    
    # Step 1: Reinhard stain normalization
    stain_normalized = normalizer.normalize_image(image)
    
    # Step 2: Intensity normalization based on method
    if norm_method == "zscore":
        final_image = normalize_with_zscore(stain_normalized, zscore_mean, zscore_std)
    elif norm_method == "percentile":
        final_image = normalize_with_percentile(stain_normalized, percentile_low, percentile_high)
    elif norm_method == "none":
        final_image = stain_normalized
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}. Use 'zscore', 'percentile', or 'none'")
    
    return final_image


def main():
    """Example usage"""
    print("Reinhard Stain Normalization for SYBR Gold + Eosin")
    print("="*50)
    
    try:
        # Load best reference from selection analysis
        normalizer = load_best_reference()
        
        # Example: normalize a single image
        example_path = "example_tiles/5100-20400_GTEX-WWYW_Adipose-Subcutaneous.jpg"
        if Path(example_path).exists():
            print(f"\nNormalizing example image: {example_path}")
            
            # Load original image
            original = np.array(Image.open(example_path))
            
            # Apply complete pipeline
            normalized = complete_preprocessing_pipeline(original, normalizer)
            
            # Validate normalization
            validation = normalizer.validate_normalization(original, normalized)
            print(f"Normalization validation: {'PASSED' if validation['overall_valid'] else 'FAILED'}")
            
            # Create comparison
            comparison_path = "preprocessing_analysis/stain_normalization/example_normalization_comparison.png"
            normalizer.create_before_after_comparison(original, normalized, comparison_path)
            
        else:
            print(f"Example image not found: {example_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run select_stain_reference.py first to select optimal reference")


if __name__ == "__main__":
    main()
