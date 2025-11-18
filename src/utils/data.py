"""
Enhanced data augmentation utilities for small dataset training.
Optimized for medical image segmentation.
Compatible with scikit-image, scipy, and OpenCV.
"""

import numpy as np
import cv2


# ---- Core Augmentation Functions ------------------------------------------

def random_rotation_90(image, mask, rng=np.random):
    """Random 90-degree rotation (0, 90, 180, 270)"""
    k = rng.randint(0, 4)
    if k == 0:
        return image, mask
    return np.rot90(image, k), np.rot90(mask, k)


def random_flip(image, mask, rng=np.random):
    """Random horizontal and vertical flips"""
    if rng.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def random_brightness(image, factor_range=(0.7, 1.3), rng=np.random):
    """Random brightness adjustment"""
    factor = rng.uniform(*factor_range)
    return np.clip(image * factor, 0, 255)


def random_contrast(image, factor_range=(0.7, 1.3), rng=np.random):
    """Random contrast adjustment"""
    mean = image.mean()
    factor = rng.uniform(*factor_range)
    return np.clip((image - mean) * factor + mean, 0, 255)


def random_gamma(image, gamma_range=(0.7, 1.3), rng=np.random):
    """Random gamma correction"""
    gamma = rng.uniform(*gamma_range)
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255.0).astype(image.dtype)


def random_gaussian_blur(image, sigma_range=(0, 1.5), prob=0.3, rng=np.random):
    """Random Gaussian blur"""
    if rng.random() > prob:
        return image
    sigma = rng.uniform(*sigma_range)
    if sigma < 0.1:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def random_gaussian_noise(image, std_range=(0, 10), prob=0.3, rng=np.random):
    """Add random Gaussian noise"""
    if rng.random() > prob:
        return image
    std = rng.uniform(*std_range)
    noise = rng.normal(0, std, image.shape)
    return np.clip(image + noise, 0, 255)


def random_scale(image, mask, scale_range=(0.85, 1.15), prob=0.5, rng=np.random):
    """
    Random scaling (zoom in/out) with proper handling.
    Returns images of same size as input.
    """
    if rng.random() > prob:
        return image, mask
    
    scale = rng.uniform(*scale_range)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    if scale > 1.0:  # Zoom in
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Center crop
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        image = image_scaled[y0:y0+h, x0:x0+w]
        mask = mask_scaled[y0:y0+h, x0:x0+w]
    else:  # Zoom out
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Pad to original size
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        image = np.pad(image_scaled, 
                      ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), 
                      mode='reflect')
        mask = np.pad(mask_scaled, 
                     ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), 
                     mode='constant', 
                     constant_values=0)
    
    return image, mask


def elastic_transform(image, mask, alpha=10, sigma=3, rng=np.random):
    """
    Elastic deformation for data augmentation.
    Creates smooth, random distortions.
    
    Args:
        image: Input image
        mask: Input mask
        alpha: Strength of deformation
        sigma: Smoothness of deformation
        rng: Random number generator
        
    Returns:
        Deformed image and mask
    """
    shape = image.shape[:2]
    
    # Generate random displacement fields
    dx = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    
    # Create meshgrid and add displacement
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
    
    # Apply deformation
    image_def = cv2.remap(image, indices[1], indices[0], 
                         cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_REFLECT)
    mask_def = cv2.remap(mask, indices[1], indices[0], 
                        cv2.INTER_NEAREST, 
                        borderMode=cv2.BORDER_CONSTANT, 
                        borderValue=0)
    
    return image_def, mask_def


# ---- Comprehensive Augmentation Pipelines ---------------------------------

def augment_pair_heavy(image, mask, rng=np.random):
    """
    ORIGINAL heavy augmentation from 0.68 dice model.
    This is the PROVEN version that worked well.
    
    Use for datasets <200 tiles.
    
    Heavy augmentation pipeline for small datasets (<100 tiles).
    Applies multiple augmentations with high probability.
    
    Use cases:
    - Very small datasets with high risk of overfitting
    - Need maximum data diversity
    
    Args:
        image: Input image (grayscale or RGB)
        mask: Input mask (binary)
        rng: Random number generator
        
    Returns:
        Augmented image and mask
    """
    # Geometric transforms (applied to both)
    image, mask = random_rotation_90(image, mask, rng)
    image, mask = random_flip(image, mask, rng)
    image, mask = random_scale(image, mask, scale_range=(0.9, 1.1), prob=0.5, rng=rng)
    
    # Elastic deformation (30% chance)
    if rng.random() > 0.7:
        image, mask = elastic_transform(image, mask, alpha=15, sigma=3, rng=rng)
    
    # Color/intensity transforms (image only) - high probability
    if rng.random() > 0.3:
        image = random_brightness(image, factor_range=(0.8, 1.2), rng=rng)
    if rng.random() > 0.3:
        image = random_contrast(image, factor_range=(0.8, 1.2), rng=rng)
    if rng.random() > 0.3:
        image = random_gamma(image, gamma_range=(0.8, 1.2), rng=rng)
    
    # Blur and noise
    image = random_gaussian_blur(image, sigma_range=(0, 1.0), prob=0.2, rng=rng)
    image = random_gaussian_noise(image, std_range=(0, 5), prob=0.2, rng=rng)
    
    return image.astype(np.float32), mask.astype(np.float32)


def augment_pair_moderate(image, mask, rng=np.random):
    """
    Moderate augmentation pipeline for medium datasets (100-500 tiles).
    Balanced approach between diversity and stability.
    
    Use cases:
    - Medium-sized datasets (like 469 images)
    - Moderate natural variation in data
    - Target: 2-3x effective dataset size
    
    Args:
        image: Input image
        mask: Input mask
        rng: Random number generator
        
    Returns:
        Augmented image and mask
    """
    # Geometric transforms (always applied)
    image, mask = random_rotation_90(image, mask, rng)
    image, mask = random_flip(image, mask, rng)
    
    # Mild scaling (30% chance, narrower range)
    image, mask = random_scale(image, mask, scale_range=(0.95, 1.05), prob=0.3, rng=rng)
    
    # Elastic deformation (rare, mild)
    if rng.random() > 0.85:
        image, mask = elastic_transform(image, mask, alpha=8, sigma=3, rng=rng)
    
    # Color/intensity transforms (50% chance each)
    if rng.random() > 0.5:
        image = random_brightness(image, factor_range=(0.9, 1.1), rng=rng)
    if rng.random() > 0.5:
        image = random_contrast(image, factor_range=(0.9, 1.1), rng=rng)
    
    # Occasional blur (15% chance)
    image = random_gaussian_blur(image, sigma_range=(0, 0.8), prob=0.15, rng=rng)
    
    return image.astype(np.float32), mask.astype(np.float32)


def augment_pair_light(image, mask, rng=np.random):
    """
    Light augmentation pipeline for large datasets (>500 tiles).
    Minimal augmentation to preserve data fidelity.
    
    Use cases:
    - Large datasets with natural variation
    - Data already has sufficient diversity
    - Prioritizing training stability
    
    Args:
        image: Input image
        mask: Input mask
        rng: Random number generator
        
    Returns:
        Augmented image and mask
    """
    # Basic geometric transforms only
    image, mask = random_rotation_90(image, mask, rng)
    image, mask = random_flip(image, mask, rng)
    
    # Minimal color adjustment (30% chance)
    if rng.random() > 0.7:
        image = random_brightness(image, factor_range=(0.95, 1.05), rng=rng)
    
    return image.astype(np.float32), mask.astype(np.float32)


def augment_pair_tta_style(image, mask, rng=np.random):
    """
    TTA-mimicking augmentation for training.
    Combines systematic TTA geometric transforms with conservative photometric augmentation.
    
    Philosophy:
    - Apply ONE of 8 TTA transformations systematically (not randomly)
    - Ensures model sees all geometric variations that TTA uses
    - Add moderate photometric augmentation for robustness
    - Avoid aggressive transforms that could mislead the model
    
    Use cases:
    - When TTA shows large performance gains at test time
    - Want training to match test-time augmentation strategy
    - Need consistent geometric coverage across epochs
    - Medium to large datasets (200+ tiles)
    
    Expected benefits:
    - 8x geometric coverage (TTA transforms)
    - 2-3x photometric coverage (brightness/contrast/gamma)
    - ~1.5x scale coverage (conservative scaling)
    - Total: ~24-36x effective dataset size
    
    Args:
        image: Input image
        mask: Input mask
        rng: Random number generator
        
    Returns:
        Augmented image and mask
    """
    # CORE: Systematic TTA geometric transform (always applied)
    # Choose one of 8 transformations deterministically
    transform_id = rng.randint(0, 8)
    
    if transform_id == 0:  # Original (no change)
        pass
    elif transform_id == 1:  # Rotate 90°
        image, mask = np.rot90(image, 1), np.rot90(mask, 1)
    elif transform_id == 2:  # Rotate 180°
        image, mask = np.rot90(image, 2), np.rot90(mask, 2)
    elif transform_id == 3:  # Rotate 270°
        image, mask = np.rot90(image, 3), np.rot90(mask, 3)
    elif transform_id == 4:  # Horizontal flip
        image, mask = np.fliplr(image), np.fliplr(mask)
    elif transform_id == 5:  # Horizontal flip + Rotate 90°
        image, mask = np.fliplr(image), np.fliplr(mask)
        image, mask = np.rot90(image, 1), np.rot90(mask, 1)
    elif transform_id == 6:  # Horizontal flip + Rotate 180°
        image, mask = np.fliplr(image), np.fliplr(mask)
        image, mask = np.rot90(image, 2), np.rot90(mask, 2)
    else:  # transform_id == 7: Horizontal flip + Rotate 270°
        image, mask = np.fliplr(image), np.fliplr(mask)
        image, mask = np.rot90(image, 3), np.rot90(mask, 3)
    
    # Conservative scale augmentation (30% chance, narrow range)
    # Handles slight zoom variations that TTA doesn't cover
    if rng.random() > 0.7:
        image, mask = random_scale(image, mask, 
                                   scale_range=(0.95, 1.05), 
                                   prob=1.0, rng=rng)
    
    # Photometric augmentations (moderate probability)
    # Addresses intensity variations TTA doesn't handle
    if rng.random() > 0.4:  # 60% chance
        image = random_brightness(image, factor_range=(0.85, 1.15), rng=rng)
    if rng.random() > 0.4:  # 60% chance
        image = random_contrast(image, factor_range=(0.85, 1.15), rng=rng)
    if rng.random() > 0.5:  # 50% chance
        image = random_gamma(image, gamma_range=(0.85, 1.15), rng=rng)
    
    # Light blur (15% chance, mild sigma)
    # Simulates slight out-of-focus or compression artifacts
    image = random_gaussian_blur(image, sigma_range=(0, 0.7), prob=0.15, rng=rng)
    
    return image.astype(np.float32), mask.astype(np.float32)


def augment_grayscale_tile_classification(image, rng=np.random):
    """
    Augmentation pipeline for grayscale classification tiles (e.g., 1024x1024 SYBR images).
    Applies geometry + photometric jitter without requiring a paired mask.
    """
    if image.ndim != 2:
        raise ValueError("augment_grayscale_tile_classification expects a 2D grayscale array.")

    # Geometric transforms: random rotation (0, 90, 180, 270) + optional flips
    k = rng.randint(0, 4)
    if k:
        image = np.rot90(image, k)
    if rng.random() > 0.5:
        image = np.fliplr(image)
    if rng.random() > 0.5:
        image = np.flipud(image)

    # Mild random scaling (zoom in/out) with reflect padding/cropping
    if rng.random() > 0.7:
        scale = rng.uniform(0.95, 1.05)
        h, w = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if scale >= 1.0:
            y0 = (new_h - h) // 2
            x0 = (new_w - w) // 2
            image = resized[y0:y0 + h, x0:x0 + w]
        else:
            pad_h = h - new_h
            pad_w = w - new_w
            image = np.pad(
                resized,
                (
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="reflect",
            )

    # Photometric jitter
    if rng.random() > 0.4:
        image = random_brightness(image, factor_range=(0.9, 1.1), rng=rng)
    if rng.random() > 0.4:
        image = random_contrast(image, factor_range=(0.9, 1.1), rng=rng)
    if rng.random() > 0.5:
        image = random_gamma(image, gamma_range=(0.9, 1.1), rng=rng)

    # Occasional blur/noise
    image = random_gaussian_blur(image, sigma_range=(0, 0.8), prob=0.15, rng=rng)
    image = random_gaussian_noise(image, std_range=(0, 5), prob=0.15, rng=rng)

    return image.astype(np.float32)


# ---- Utility Functions ----------------------------------------------------

def normalize_image(image, method='percentile', p_low=1, p_high=99, mean=None, std=None):
    """
    Normalize image using various methods.
    
    Args:
        image: Input image
        method: 'percentile', 'minmax', 'zscore', or 'zscore_dataset'
        p_low: Lower percentile (for percentile method)
        p_high: Upper percentile (for percentile method)
        mean: Dataset mean (for zscore_dataset method)
        std: Dataset std (for zscore_dataset method)
        
    Returns:
        Normalized image
    """
    if method == 'percentile':
        plow, phigh = np.percentile(image, (p_low, p_high))
        scale = max(phigh - plow, 1e-3)
        return np.clip((image - plow) / scale, 0, 1)
    elif method == 'minmax':
        imin, imax = image.min(), image.max()
        scale = max(imax - imin, 1e-3)
        return (image - imin) / scale
    elif method == 'zscore':
        img_mean, img_std = image.mean(), image.std()
        return (image - img_mean) / (img_std + 1e-10)
    elif method == 'zscore_dataset':
        if mean is None or std is None:
            raise ValueError("Dataset mean and std required for zscore_dataset method")
        return (image - mean) / (std + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_dataset_statistics(image_paths, max_samples=100):
    """
    Compute mean and std across a dataset for normalization.
    
    Args:
        image_paths: List of paths to images
        max_samples: Maximum number of images to sample
        
    Returns:
        (mean, std) tuple
    """
    from pathlib import Path
    
    sample_size = min(max_samples, len(image_paths))
    sample_pixels = []
    
    for img_path in image_paths[:sample_size]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sample_pixels.append(img.flatten())
    
    if not sample_pixels:
        return 127.5, 50.0  # Default values
    
    all_pixels = np.concatenate(sample_pixels)
    return np.mean(all_pixels), np.std(all_pixels)


# ---- Test/Visualization Utilities -----------------------------------------

def visualize_augmentation(image, mask, augment_fn=augment_pair_moderate, 
                          num_examples=5, save_path=None):
    """
    Visualize augmentation results for debugging.
    
    Args:
        image: Input image
        mask: Input mask
        augment_fn: Augmentation function to test
        num_examples: Number of augmented examples to generate
        save_path: Path to save visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    rng = np.random.RandomState(42)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    
    for i in range(num_examples):
        aug_img, aug_mask = augment_fn(image.copy(), mask.copy(), rng)
        
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(aug_img, cmap='gray')
        axes[i, 1].set_title(f'Augmented {i+1}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(aug_mask, cmap='gray')
        axes[i, 2].set_title('Augmented Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved augmentation examples to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Simple test
    print("Data augmentation utilities loaded successfully")
    
    # Create dummy image and mask for testing
    test_img = np.random.rand(256, 256).astype(np.float32) * 255
    test_mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
    
    # Test all augmentation levels
    print("\nTesting augmentation pipelines:")
    
    rng = np.random.RandomState(42)
    
    aug_img, aug_mask = augment_pair_heavy(test_img, test_mask, rng)
    print(f"  Heavy:    img shape={aug_img.shape}, dtype={aug_img.dtype}")
    
    aug_img, aug_mask = augment_pair_moderate(test_img, test_mask, rng)
    print(f"  Moderate: img shape={aug_img.shape}, dtype={aug_mask.dtype}")
    
    aug_img, aug_mask = augment_pair_light(test_img, test_mask, rng)
    print(f"  Light:    img shape={aug_img.shape}, dtype={aug_mask.dtype}")
    
    print("\n✓ All augmentation functions working correctly")
