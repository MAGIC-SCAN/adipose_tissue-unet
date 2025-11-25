import numpy as np
import cv2


def adaptive_clahe_normalization(img):
    """
    Adaptive CLAHE normalization based on image quality analysis
    Cutoffs determined from preprocessing analysis of 6 sample images
    """
    # Calculate image quality metrics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    contrast_ratio = std_intensity / (mean_intensity + 1e-6)
    
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Determine preprocessing strategy
    if contrast_ratio < 0.183:
        # Poor quality - needs aggressive CLAHE
        print(f"Poor quality image (contrast_ratio={contrast_ratio:.3f}) - applying CLAHE")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img.astype(np.uint8)).astype(np.float32)
        p5, p95 = np.percentile(img_enhanced, (5, 95))
        return np.clip((img_enhanced - p5) / (p95 - p5 + 1e-3), 0, 1)
        
    elif contrast_ratio > 0.267 and sharpness > 38.2:
        # Good quality - percentile normalization only
        print(f"Good quality image (contrast_ratio={contrast_ratio:.3f}, sharpness={sharpness:.1f}) - percentile only")
        p2, p98 = np.percentile(img, (2, 98))
        return np.clip((img - p2) / (p98 - p2 + 1e-3), 0, 1)
        
    else:
        # Medium quality - mild CLAHE
        print(f"Medium quality image (contrast_ratio={contrast_ratio:.3f}) - applying mild CLAHE")
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
        img_enhanced = clahe.apply(img.astype(np.uint8)).astype(np.float32)
        p5, p95 = np.percentile(img_enhanced, (5, 95))
        return np.clip((img_enhanced - p5) / (p95 - p5 + 1e-3), 0, 1)
