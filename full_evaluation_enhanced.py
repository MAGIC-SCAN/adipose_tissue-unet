#!/usr/bin/env python3
"""
Full Evaluation Script for Adipose U-Net
Extends full_evaluation.py with advanced post-processing for improved performance

FEATURES:
- Training statistics isolation (no data leakage)
- Slide-level metrics aggregation 
- Comprehensive segmentation metrics with confidence intervals
- TP/FP/FN categorical error analysis
- F1-optimized threshold selection
- Statistical reporting
- Adaptive threshold optimization (0.1-0.9 two-stage grid search)
- Sliding window inference with Gaussian blending (configurable overlap)
- Morphological boundary refinement (bilateral filtering + morphology)

Usage:
    # All enhancements enabled
    python full_evaluation_enhanced.py \
        --weights checkpoints/best/phase_best.weights.h5 \
        --clean-test --stain \
        --use-tta --tta-mode full \
        --sliding-window --overlap 0.5 \
        --boundary-refine \
        --adaptive-threshold
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Any, Optional
import tifffile as tiff
import time
import glob
import re
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
import warnings
import math

# Scientific computing imports
import scipy.stats as stats
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

# Machine learning imports
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Image processing imports
from skimage import morphology, measure
import seaborn as sns

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,
    Concatenate, Lambda, Reshape, Add,
)

# Set non-interactive backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ====================================================================
# ENHANCED POST-PROCESSING CLASSES
# ====================================================================

class GaussianBlender:
    """
    Weighted blending with 2D Gaussian kernel for seamless tile reconstruction
    
    Gaussian weight: w(x,y) = exp(-((x-cx)² + (y-cy)²) / (2σ²))
    where (cx, cy) is the tile center and σ controls falloff
    """
    
    def __init__(self, tile_size: int = 1024, sigma_factor: float = 0.25):
        """
        Args:
            tile_size: Size of tiles
            sigma_factor: Controls Gaussian falloff (0.25 = smooth, 0.5 = aggressive)
        """
        self.tile_size = tile_size
        self.sigma = tile_size * sigma_factor
        self.weight_map = self._create_gaussian_weight_map()
    
    def _create_gaussian_weight_map(self) -> np.ndarray:
        """Create 2D Gaussian weight map centered on tile"""
        center = self.tile_size / 2
        y, x = np.ogrid[0:self.tile_size, 0:self.tile_size]
        
        # Distance from center
        dist_sq = (x - center)**2 + (y - center)**2
        
        # Gaussian kernel
        weights = np.exp(-dist_sq / (2 * self.sigma**2))
        
        # Normalize to [0, 1]
        weights = weights / weights.max()
        
        return weights.astype(np.float32)
    
    def reconstruct(self, tiles: List[np.ndarray], positions: List[Tuple[int, int]], 
                   output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct full image from overlapping tiles using Gaussian blending
        
        Args:
            tiles: List of tile predictions [0-1]
            positions: List of (y, x) top-left positions
            output_shape: (height, width) of output
            
        Returns:
            Blended full prediction [0-1]
        """
        h, w = output_shape
        accumulator = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        for tile, (y, x) in zip(tiles, positions):
            # Determine actual tile size (handle edge cases)
            th, tw = tile.shape[:2]
            
            # Get corresponding weight map slice
            weight_slice = self.weight_map[:th, :tw]
            
            # Accumulate weighted predictions
            accumulator[y:y+th, x:x+tw] += tile * weight_slice
            weight_sum[y:y+th, x:x+tw] += weight_slice
        
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)
        
        # Normalize
        result = accumulator / weight_sum
        
        return result.astype(np.float32)


class LinearBlender:
    """Simple averaging blender for overlapping tiles (baseline)"""
    
    def reconstruct(self, tiles: List[np.ndarray], positions: List[Tuple[int, int]], 
                   output_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct using simple averaging"""
        h, w = output_shape
        accumulator = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.int32)
        
        for tile, (y, x) in zip(tiles, positions):
            th, tw = tile.shape[:2]
            accumulator[y:y+th, x:x+tw] += tile
            count[y:y+th, x:x+tw] += 1
        
        # Avoid division by zero
        count = np.maximum(count, 1)
        
        return (accumulator / count).astype(np.float32)


class SlidingWindowInference:
    """
    Sliding window inference with configurable overlap and blending
    
    Handles overlapping tile extraction, prediction, and reconstruction
    """
    
    def __init__(self, tile_size: int = 1024, overlap: float = 0.5, 
                 blend_mode: str = 'gaussian'):
        """
        Args:
            tile_size: Size of tiles
            overlap: Overlap ratio (0.0 - 0.75)
            blend_mode: 'gaussian', 'linear', or 'none'
        """
        self.tile_size = tile_size
        self.overlap = max(0.0, min(overlap, 0.75))
        self.stride = int(tile_size * (1 - self.overlap))
        
        # Initialize blender
        if blend_mode == 'gaussian':
            self.blender = GaussianBlender(tile_size)
        elif blend_mode == 'linear':
            self.blender = LinearBlender()
        else:
            self.blender = None  # No blending
        
        print(f"[SlidingWindow] Initialized: tile={tile_size}, stride={self.stride}, "
              f"overlap={overlap:.1%}, blend={blend_mode}")
    
    def extract_tile_positions(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Calculate tile positions for sliding window
        
        Args:
            image_shape: (height, width)
            
        Returns:
            List of (y, x) top-left positions
        """
        h, w = image_shape[:2]
        
        positions = []
        
        # Calculate number of tiles needed
        y_steps = max(1, math.ceil((h - self.tile_size) / self.stride) + 1)
        x_steps = max(1, math.ceil((w - self.tile_size) / self.stride) + 1)
        
        for yi in range(y_steps):
            for xi in range(x_steps):
                # Calculate position, ensuring we don't exceed image bounds
                y = min(yi * self.stride, h - self.tile_size)
                x = min(xi * self.stride, w - self.tile_size)
                
                # Ensure valid position
                if y >= 0 and x >= 0 and y + self.tile_size <= h and x + self.tile_size <= w:
                    positions.append((y, x))
        
        return positions
    
    def extract_tiles(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract overlapping tiles from image
        
        Args:
            image: Input image
            
        Returns:
            (tiles, positions) - List of tiles and their positions
        """
        positions = self.extract_tile_positions(image.shape)
        
        tiles = []
        for y, x in positions:
            tile = image[y:y+self.tile_size, x:x+self.tile_size]
            tiles.append(tile)
        
        return tiles, positions
    
    def predict_with_sliding_window(self, image: np.ndarray, model, mean: float, std: float,
                                   use_tta: bool = False, tta_mode: str = 'basic') -> np.ndarray:
        """
        Perform sliding window inference on full image
        
        Args:
            image: Full input image
            model: Model with predict_single method
            mean, std: Normalization parameters
            use_tta: Whether to use TTA for each tile
            tta_mode: TTA mode if enabled
            
        Returns:
            Full prediction map [0-1]
        """
        # Extract tiles
        tiles, positions = self.extract_tiles(image)
        
        print(f"[SlidingWindow] Processing {len(tiles)} overlapping tiles...")
        
        # Predict each tile
        predictions = []
        for i, tile in enumerate(tiles):
            if use_tta:
                # Use TTA for this tile
                tta = TestTimeAugmentation(mode=tta_mode)
                pred, _ = tta.predict_with_tta(model, tile, mean, std)
            else:
                # Standard prediction
                pred = model.predict_single(tile, mean, std)
            
            predictions.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(tiles)} tiles")
        
        # Reconstruct with blending
        if self.blender is not None:
            full_pred = self.blender.reconstruct(predictions, positions, image.shape[:2])
        else:
            # Simple averaging fallback
            full_pred = LinearBlender().reconstruct(predictions, positions, image.shape[:2])
        
        return full_pred


class BoundaryRefiner:
    """
    Morphological boundary refinement for segmentation masks
    
    Uses bilateral filtering + morphological operations to smooth boundaries
    while preserving important structures
    """
    
    def __init__(self, kernel_size: int = 5, bilateral_d: int = 5, 
                 bilateral_sigma_color: float = 50, bilateral_sigma_space: float = 50):
        """
        Args:
            kernel_size: Size of morphological kernels
            bilateral_d: Diameter of bilateral filter
            bilateral_sigma_color: Color space sigma for bilateral
            bilateral_sigma_space: Coordinate space sigma for bilateral
        """
        self.kernel_size = kernel_size
        self.bilateral_d = bilateral_d
        self.sigma_color = bilateral_sigma_color
        self.sigma_space = bilateral_sigma_space
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (kernel_size, kernel_size))
    
    def refine(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply boundary refinement to binary mask
        
        Args:
            mask: Binary prediction mask [0-1]
            image: Optional original image for image-guided filtering
            
        Returns:
            Refined mask [0-1]
        """
        # Convert to uint8 for OpenCV
        mask_u8 = (mask * 255).astype(np.uint8)
        
        # 1. Identify boundary region using XOR to avoid uint8 wraparound
        eroded = cv2.erode(mask_u8, self.kernel, iterations=1)
        dilated = cv2.dilate(mask_u8, self.kernel, iterations=1)
        boundary = np.logical_xor(dilated > 0, eroded > 0).astype(np.uint8)
        
        # 2. Apply bilateral filtering to smooth boundaries
        # This preserves edges while smoothing noise
        filtered = cv2.bilateralFilter(
            mask_u8, 
            self.bilateral_d, 
            self.sigma_color, 
            self.sigma_space
        )
        
        # 3. Blend filtered result only in boundary regions
        refined = np.where(boundary > 0, filtered, mask_u8)
        
        # 4. Remove small holes and components
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, self.kernel, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        
        # Convert back to float [0-1]
        return (refined / 255.0).astype(np.float32)


# ====================================================================
# ORIGINAL FULL_EVALUATION.PY CODE (with enhancements integrated)
# ====================================================================


def resolve_checkpoint_paths(
    weights_arg: str = "",
    output_arg: str = "",
    data_root_arg: str = "",
    use_test_timestamp: bool = False,
) -> Tuple[str, str, str]:
    """
    Simplified path resolution with new logic:
    - weights_path: absolute/relative path to a .h5 (required)
    - output_path: base folder for publication_evaluation_{dataset} directories
    - data_root_path: dataset root (default: /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_test/stain_normalized)
    """
    # 1) Weights (required)
    if not weights_arg:
        raise ValueError(
            "❌ --weights argument is required.\n"
            "Please specify path to trained model weights."
        )
    
    weights_path = weights_arg
    # Handle both direct weights files and checkpoint directories
    if Path(weights_path).is_dir():
        # Directory provided - find best weights file
        ckpt_dir = weights_path
        weights_file = _find_best_weights_in_dir(Path(ckpt_dir))
        if weights_file is None:
            raise FileNotFoundError(f"No weights files found in directory: {ckpt_dir}")
        weights_path = str(weights_file)
    else:
        # Weights file provided directly
        ckpt_dir = str(Path(weights_path).parent)

    # 2) Output base (honor CLI if provided; else default near the checkpoint)
    if output_arg:
        output_path = output_arg
    else:
        output_path = str(Path(ckpt_dir) / "evaluation")

    # 3) Data root
    if data_root_arg:
        data_root_path = data_root_arg
        print(f"📂 Using provided data root: {data_root_path}")
    elif use_test_timestamp:
        # Special case for --test flag: extract timestamp and find matching build
        data_root_base = Path("/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane")
        
        # Extract timestamp from checkpoint directory name
        checkpoint_dir_name = Path(ckpt_dir).name
        checkpoint_timestamp = _extract_ts_from_name(checkpoint_dir_name)
        
        if checkpoint_timestamp is not None:
            # Convert to exactly 14 digits: YYYYMMDDHHMMSS
            ts_num = int(checkpoint_timestamp)
            ts_str = f"{ts_num:014d}"
            # Format as YYYYMMDD_HHMMSS
            if len(ts_str) >= 14:
                formatted_ts = f"{ts_str[:8]}_{ts_str[8:14]}"
                
                # Look for matching dataset build
                matching_build = data_root_base / f"_build_{formatted_ts}" / "dataset"
                
                if matching_build.exists():
                    data_root_path = str(matching_build)
                    print(f"🔗 Matched checkpoint timestamp {formatted_ts} to dataset: {data_root_path}")
                else:
                    # Build list of available builds for error message
                    available_builds = []
                    for p in data_root_base.glob("_build_*"):
                        if (p / "dataset").exists():
                            available_builds.append(p.name)
                    
                    available_builds.sort(reverse=True)
                    
                    raise FileNotFoundError(
                        f"❌ No matching dataset found for checkpoint timestamp {formatted_ts}\n"
                        f"Expected: {matching_build}\n"
                        f"Available builds: {available_builds}\n"
                        f"Please build the dataset with the correct timestamp or specify --data-root manually."
                    )
            else:
                raise ValueError(f"Invalid timestamp format extracted: {ts_str}")
        else:
            # No timestamp found in checkpoint name - cannot auto-match
            raise FileNotFoundError(
                f"❌ Cannot extract timestamp from checkpoint directory: {checkpoint_dir_name}\n"
                f"Unable to auto-match dataset build for --test flag. Please specify --data-root manually."
            )
    else:
        # Default data root for human-test and clean-test
        data_root_path = "/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_test/stain_normalized"
        print(f"📂 Using default data root: {data_root_path}")

    return weights_path, output_path, data_root_path

def find_most_recent_checkpoint(checkpoints_dir: str = "checkpoints") -> Tuple[str, str]:
    """
    Returns (weights_path_str, checkpoint_dir_str).
    Chooses the newest checkpoint folder by timestamp in name;
    falls back to folder mtime if no timestamp is present.
    Raises FileNotFoundError if nothing usable is found.
    """
    root = Path(checkpoints_dir)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {root}")

    # candidate dirs
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No checkpoint subdirectories in: {root}")

    # sort: timestamp in name desc, then mtime desc
    def sort_key(d: Path):
        ts = _extract_ts_from_name(d.name)
        return (1, ts) if ts is not None else (0, d.stat().st_mtime)

    dirs.sort(key=sort_key, reverse=True)

    for d in dirs:
        weights = _find_best_weights_in_dir(d)
        if weights:
            return str(weights), str(d)

    raise FileNotFoundError("No .h5 / .weights.h5 files found in any checkpoint directory.")


_WEIGHT_CANDIDATES = [
    "weights_best_overall.weights.h5",
    "phase2_best.weights.h5",
    "phase1_best.weights.h5",
    "best_model.weights.h5",
    "model_best.weights.h5",
    "weights_best.weights.h5",
]

def _find_best_weights_in_dir(ckpt_dir: Path) -> Optional[Path]:
    for name in _WEIGHT_CANDIDATES:
        p = ckpt_dir / name
        if p.exists():
            return p
    # fallback to any .weights.h5, or .h5
    files = list(ckpt_dir.glob("*.weights.h5")) + list(ckpt_dir.glob("*.h5"))
    return files[0] if files else None

def _extract_ts_from_name(s: str) -> Optional[float]:
    """
    Expect pattern like 20251031_164636 in the directory name.
    Return a sortable float timestamp; fallback to mtime via None.
    """
    m = re.search(r'(\d{8}_\d{6})', s)
    if not m:
        return None
    # convert to "YYYYMMDD_HHMMSS" -> epoch-ish weight
    ts = m.group(1)
    # Make a comparable float, not actual epoch (we just need ordering)
    return float(ts.replace('_', ''))


class TestTimeAugmentation:
    """
    Simple TTA with D4-like flips/rotations.
    Expects a model wrapper that exposes predict_single(image, mean, std) -> prob_map.
    """
    def __init__(self, mode: str = "basic"):
        mode = (mode or "basic").lower()
        if mode not in {"minimal", "basic", "full"}:
            mode = "basic"
        self.mode = mode

        # Define transforms as pairs of (aug_fn, deaug_fn)
        # aug_fn: apply to input image
        # deaug_fn: invert transform on predicted prob map
        def ident(x): return x
        def rot90(x): return np.rot90(x, 1)
        def rot180(x): return np.rot90(x, 2)
        def rot270(x): return np.rot90(x, 3)
        def r90_inv(x): return np.rot90(x, 3)
        def r180_inv(x): return np.rot90(x, 2)
        def r270_inv(x): return np.rot90(x, 1)
        def flip_h(x): return np.flip(x, axis=1)
        def flip_v(x): return np.flip(x, axis=0)

        T_minimal = [
            (ident, ident),
            (flip_h, flip_h),
        ]

        T_basic = [
            (ident, ident),
            (flip_h, flip_h),
            (flip_v, flip_v),
            (rot90, r90_inv),
        ]

        # 8-way (subset of D4)
        T_full = [
            (ident, ident),
            (rot90, r90_inv),
            (rot180, r180_inv),
            (rot270, r270_inv),
            (flip_h, flip_h),
            (flip_v, flip_v),
            (lambda x: flip_h(rot90(x)), lambda x: r90_inv(flip_h(x))),
            (lambda x: flip_v(rot90(x)), lambda x: r90_inv(flip_v(x))),
        ]

        if self.mode == "minimal":
            self.transforms = T_minimal
        elif self.mode == "basic":
            self.transforms = T_basic
        else:
            self.transforms = T_full

    def predict_with_tta(
        self,
        model_wrapper: Any,
        image: np.ndarray,
        mean: float,
        std: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs TTA by averaging de-augmented predictions.
        Returns (avg_pred, timing_info).
        """
        start = time.time()
        preds = []
        for aug_fn, deaug_fn in self.transforms:
            aug_img = aug_fn(image)
            pred = model_wrapper.predict_single(aug_img, mean, std)
            pred = deaug_fn(pred)
            preds.append(pred.astype(np.float32))
        avg_pred = np.mean(preds, axis=0).astype(np.float32)
        timing = {
            "num_augmentations": len(self.transforms),
            "total_time": time.time() - start,
        }
        return avg_pred, timing

@dataclass
class ComprehensiveMetrics:
    """Container for comprehensive evaluation metrics with confidence intervals"""
    # Pixel-level metrics
    dice_score: float
    dice_ci: Tuple[float, float]
    
    jaccard_index: float
    jaccard_ci: Tuple[float, float]
    
    sensitivity: float
    sensitivity_ci: Tuple[float, float]
    
    specificity: float
    specificity_ci: Tuple[float, float]
    
    precision: float
    precision_ci: Tuple[float, float]
    
    f1_score: float
    f1_ci: Tuple[float, float]
    
    accuracy: float
    accuracy_ci: Tuple[float, float]
    
    # AUC metrics
    roc_auc: float
    roc_auc_ci: Tuple[float, float]
    
    pr_auc: float
    pr_auc_ci: Tuple[float, float]
    
    # Boundary metrics
    hausdorff95: float
    hausdorff95_ci: Tuple[float, float]
    
    assd: float
    assd_ci: Tuple[float, float]
    
    # Sample information
    n_slides: int
    n_tiles: int
    optimal_threshold: float


def set_deterministic_seeds(seed: int = 1337):
    """Set seeds for reproducible evaluation with enhanced determinism"""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def extract_slide_id(tile_path: str) -> str:
    """
    Extract slide ID from tile path
    
    Args:
        tile_path: Path like "6 BEEF Shoulder -1_grid_5x5_r1_c2_r0_c1.jpg"
        
    Returns:
        Slide ID like "6 BEEF Shoulder -1_grid_5x5_r1_c2"
    """
    stem = Path(tile_path).stem
    parts = stem.split("_")
    
    # Remove the last part if it matches pattern rX_cY
    if len(parts) >= 2 and parts[-2].startswith('r') and parts[-1].startswith('c'):
        return "_".join(parts[:-2])
    else:
        # Fallback: remove last part only if it starts with 'r' or 'c'
        if parts[-1].startswith(('r', 'c')):
            return "_".join(parts[:-1])
        return stem


def load_training_stats(checkpoint_dir: str) -> Tuple[float, float]:
    """
    Load training normalization statistics from checkpoint directory
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Tuple of (mean, std) from training data
        
    Raises:
        FileNotFoundError: If normalization stats not found
    """
    stats_path = Path(checkpoint_dir) / "normalization_stats.json"
    
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Training normalization statistics not found: {stats_path}\n"
            f"Make sure training was completed with the updated training script."
        )
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    mean = float(stats["mean"])
    std = float(stats["std"])
    
    print(f"✓ Loaded training normalization statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Source: {stats_path}")
    
    return mean, std


def binarize_prediction(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probability prediction to binary mask"""
    return (pred > threshold).astype(np.uint8)


def calculate_pixel_metrics(pred: np.ndarray, true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive pixel-level segmentation metrics
    
    Args:
        pred: Prediction probability map [0, 1]
        true: Ground truth binary mask {0, 1}
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary with all pixel-level metrics
    """
    # Convert to boolean arrays for proper logical operations
    pred_bin = (pred > threshold)
    true_bin = (true > 0.5)
    
    # Handle both-empty masks case (perfect agreement for background tiles)
    if not true_bin.any() and not pred_bin.any():
        n = true_bin.size
        return {
            'dice_score': 1.0,
            'jaccard_index': 1.0,
            'sensitivity': 1.0,
            'specificity': 1.0,
            'precision': 1.0,
            'f1_score': 1.0,
            'accuracy': 1.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': int(n)
        }
    
    # Basic counts using logical operations on boolean arrays
    tp = np.sum(pred_bin & true_bin)
    fp = np.sum(pred_bin & ~true_bin)
    fn = np.sum(~pred_bin & true_bin)
    tn = np.sum(~pred_bin & ~true_bin)
    
    # Derived metrics
    precision = tp / (tp + fp + 1e-10)
    sensitivity = tp / (tp + fn + 1e-10)  # recall
    specificity = tn / (tn + fp + 1e-10)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    
    # F1 and Dice
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
    dice = f1  # Dice coefficient equals F1 score for binary segmentation
    
    # Jaccard (IoU)
    jaccard = tp / (tp + fp + fn + 1e-10)
    
    return {
        'dice_score': float(dice),
        'jaccard_index': float(jaccard),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def calculate_boundary_metrics(pred: np.ndarray, true: np.ndarray, threshold: float = 0.5, 
                             spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, float]:
    """
    Calculate boundary-based segmentation metrics
    
    Args:
        pred: Prediction probability map [0, 1]
        true: Ground truth binary mask {0, 1}
        threshold: Threshold for binarizing predictions
        spacing: Pixel spacing in (row, col) format
        
    Returns:
        Dictionary with boundary metrics
    """
    # Convert to boolean arrays for proper logical operations
    pred_bin = (pred > threshold)
    true_bin = (true > 0.5)
    
    # Handle edge cases
    if not pred_bin.any() and not true_bin.any():
        return {'hausdorff95': 0.0, 'assd': 0.0}
    
    if not pred_bin.any() or not true_bin.any():
        return {'hausdorff95': float('inf'), 'assd': float('inf')}
    
    try:
        # Distance transforms using boolean negation
        pred_dt = ndimage.distance_transform_edt(~pred_bin, sampling=spacing)
        true_dt = ndimage.distance_transform_edt(~true_bin, sampling=spacing)
        
        # Surface extraction using morphological operations on boolean arrays
        pred_surface = pred_bin & ~morphology.binary_erosion(pred_bin)
        true_surface = true_bin & ~morphology.binary_erosion(true_bin)
        
        # Distances from surfaces
        if pred_surface.sum() > 0 and true_surface.sum() > 0:
            pred_to_true = pred_dt[pred_surface]
            true_to_pred = true_dt[true_surface]
            
            all_distances = np.concatenate([pred_to_true, true_to_pred])
            
            # Hausdorff distance (95th percentile for robustness)
            hausdorff95 = np.percentile(all_distances, 95)
            
            # Average symmetric surface distance
            assd = np.mean(all_distances)
            
            return {
                'hausdorff95': float(hausdorff95),
                'assd': float(assd)
            }
        else:
            return {'hausdorff95': float('inf'), 'assd': float('inf')}
            
    except Exception as e:
        warnings.warn(f"Error calculating boundary metrics: {e}")
        return {'hausdorff95': float('nan'), 'assd': float('nan')}


def calculate_auc_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Calculate ROC AUC and PR AUC metrics for segmentation
    
    Args:
        pred: Prediction probability map [0, 1]
        true: Ground truth binary mask {0, 1}
        
    Returns:
        Dictionary with AUC metrics
    """
    # Flatten arrays for pixel-level classification
    pred_flat = pred.flatten()
    true_flat = (true > 0.5).astype(int).flatten()
    
    # Handle edge cases - need at least two classes for meaningful AUC
    unique_classes = np.unique(true_flat)
    if len(unique_classes) < 2:
        # Only one class present - cannot compute meaningful AUC
        return {
            'roc_auc': np.nan,
            'pr_auc': np.nan
        }
    
    try:
        # ROC AUC - Area under Receiver Operating Characteristic curve
        roc_auc = roc_auc_score(true_flat, pred_flat)
        
        # Precision-Recall AUC - More informative for imbalanced segmentation
        pr_auc = average_precision_score(true_flat, pred_flat)
        
        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        }
        
    except Exception as e:
        warnings.warn(f"Error calculating AUC metrics: {e}")
        return {
            'roc_auc': np.nan,
            'pr_auc': np.nan
        }


def optimize_threshold_f1_slide_level(predictions: List[np.ndarray], ground_truths: List[np.ndarray], 
                                     tile_paths: List[str], threshold_range: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """
    Find optimal threshold that maximizes slide-level F1 score on validation set
    
    Args:
        predictions: List of prediction probability maps
        ground_truths: List of ground truth binary masks
        tile_paths: List of tile paths for grouping by slide
        threshold_range: Range of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, f1_scores_array)
    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.95, 0.05)
    
    print("Optimizing threshold using slide-level F1 scores...")
    
    best_threshold = 0.5
    best_mean_f1 = -1.0
    f1_scores = []
    
    for threshold in threshold_range:
        # Group tiles by slide ID
        slide_f1_scores = defaultdict(list)
        
        for pred, true, tile_path in zip(predictions, ground_truths, tile_paths):
            slide_id = extract_slide_id(tile_path)
            metrics = calculate_pixel_metrics(pred, true, threshold)
            slide_f1_scores[slide_id].append(metrics['f1_score'])
        
        # Calculate slide-level macro F1 (average F1 per slide, then average across slides)
        slide_macro_f1s = [np.mean(f1_list) for f1_list in slide_f1_scores.values()]
        slide_macro_f1 = np.mean(slide_macro_f1s)
        
        f1_scores.append(slide_macro_f1)
        
        if slide_macro_f1 > best_mean_f1:
            best_mean_f1 = slide_macro_f1
            best_threshold = threshold
        
        print(f"  Threshold {threshold:.2f}: Slide-Macro F1 = {slide_macro_f1:.4f}")
    
    f1_scores = np.array(f1_scores)
    
    print(f"✓ Optimal threshold: {best_threshold:.2f} (Slide-Macro F1 = {best_mean_f1:.4f})")
    
    return best_threshold, f1_scores


def optimize_threshold_f1(predictions: List[np.ndarray], ground_truths: List[np.ndarray], 
                         threshold_range: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """
    Find optimal threshold that maximizes F1 score on validation set (tile-level for backward compatibility)
    
    Args:
        predictions: List of prediction probability maps
        ground_truths: List of ground truth binary masks
        threshold_range: Range of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, f1_scores_array)
    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.95, 0.05)
    
    f1_scores = []
    
    print("Optimizing threshold on validation set (tile-level)...")
    
    for threshold in threshold_range:
        tile_f1_scores = []
        
        for pred, true in zip(predictions, ground_truths):
            metrics = calculate_pixel_metrics(pred, true, threshold)
            tile_f1_scores.append(metrics['f1_score'])
        
        mean_f1 = np.mean(tile_f1_scores)
        f1_scores.append(mean_f1)
        
        print(f"  Threshold {threshold:.2f}: F1 = {mean_f1:.4f}")
    
    f1_scores = np.array(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = threshold_range[optimal_idx]
    
    print(f"✓ Optimal threshold: {optimal_threshold:.2f} (F1 = {f1_scores[optimal_idx]:.4f})")
    
    return optimal_threshold, f1_scores


def bootstrap_confidence_interval(data: np.ndarray, statistic_func=np.mean, 
                                n_bootstrap: int = 10000, alpha: float = 0.05, 
                                seed: int = 42) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic
    
    Args:
        data: Input data array
        statistic_func: Function to calculate statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (0.05 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    
    # Use list comprehension for efficiency
    bootstrap_stats = [statistic_func(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
    bootstrap_stats = np.asarray(bootstrap_stats)
    
    point_estimate = statistic_func(data)
    ci_lower, ci_upper = np.percentile(bootstrap_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    
    return float(point_estimate), float(ci_lower), float(ci_upper)


def safe_bootstrap_ci(data, func=np.mean):
    """Calculate bootstrap CI, handling NaN values with correct unpacking"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return np.nan, (np.nan, np.nan)
    point, lo, hi = bootstrap_confidence_interval(valid_data, func)
    return point, (lo, hi)


def create_4panel_visualization(original: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, 
                                dice_score: float, output_path: str) -> None:
    """
    Create 4-panel visualization with unique colors (matches visualize_test_predictions.py)
    
    Panel 1: Original RGB image
    Panel 2: Ground truth overlay (yellow on grayscale)
    Panel 3: Prediction overlay (magenta on grayscale)
    Panel 4: Discrepancy map (green=TP, red=FP, blue=FN, black=TN)
    """
    # Handle grayscale images - convert to RGB for visualization
    if original.ndim == 2:
        grayscale = original.astype(np.uint8)
        original_rgb = np.stack([grayscale]*3, axis=-1)
    else:
        original_rgb = original
        grayscale = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Panel 1: Original Image (RGB)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: Ground Truth Overlay (Yellow on grayscale)
    gt_overlay = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.float32)
    
    # Create yellow overlay for ground truth
    yellow_mask = np.zeros_like(gt_overlay)
    yellow_mask[gt_mask > 0] = [255, 255, 0]  # Yellow
    
    # Blend with alpha
    gt_blend = cv2.addWeighted(gt_overlay.astype(np.uint8), 0.6, 
                                yellow_mask.astype(np.uint8), 0.4, 0)
    
    axes[0, 1].imshow(gt_blend)
    axes[0, 1].set_title('Ground Truth (Yellow)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Panel 3: Prediction Overlay (Magenta on grayscale)
    pred_overlay = np.stack([grayscale, grayscale, grayscale], axis=-1).astype(np.float32)
    
    # Create magenta overlay for prediction
    magenta_mask = np.zeros_like(pred_overlay)
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    magenta_mask[pred_binary > 0] = [255, 0, 255]  # Magenta
    
    # Blend with alpha
    pred_blend = cv2.addWeighted(pred_overlay.astype(np.uint8), 0.6,
                                  magenta_mask.astype(np.uint8), 0.4, 0)
    
    axes[1, 0].imshow(pred_blend)
    axes[1, 0].set_title(f'Prediction (Magenta) - Dice: {dice_score:.3f}', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Panel 4: Discrepancy Map
    # Green=TP, Red=FP, Blue=FN, Black=TN
    discrepancy = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
    
    gt_binary = (gt_mask > 0).astype(bool)
    pred_binary_bool = (pred_mask > 0.5).astype(bool)
    
    # True Positive (Green)
    tp_mask = gt_binary & pred_binary_bool
    discrepancy[tp_mask] = [0, 255, 0]
    
    # False Positive (Red)
    fp_mask = (~gt_binary) & pred_binary_bool
    discrepancy[fp_mask] = [255, 0, 0]
    
    # False Negative (Blue)
    fn_mask = gt_binary & (~pred_binary_bool)
    discrepancy[fn_mask] = [0, 0, 255]
    
    # True Negative (Black) - already zero
    
    axes[1, 1].imshow(discrepancy)
    axes[1, 1].set_title('Discrepancy (Green=TP, Red=FP, Blue=FN, Black=TN)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def sample_tiles(predictions: List[np.ndarray], ground_truths: List[np.ndarray], 
                tile_paths: List[str], n_positive: int = 120, n_negative: int = 30) -> List[int]:
    """Sample tiles stratified by positive/negative status"""
    positive_indices = []
    negative_indices = []
    
    print("[Sampling] Categorizing tiles as positive/negative...")
    for idx, gt in enumerate(ground_truths):
        if gt.sum() > 0:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)
    
    print(f"[Sampling] Found {len(positive_indices)} positive and {len(negative_indices)} negative tiles")
    
    # Sample
    if len(positive_indices) < n_positive:
        print(f"[WARN] Only {len(positive_indices)} positive tiles available, sampling all")
        sampled_pos = positive_indices
    else:
        sampled_pos = np.random.choice(positive_indices, n_positive, replace=False).tolist()
    
    if len(negative_indices) < n_negative:
        print(f"[WARN] Only {len(negative_indices)} negative tiles available, sampling all")
        sampled_neg = negative_indices
    else:
        sampled_neg = np.random.choice(negative_indices, n_negative, replace=False).tolist()
    
    sampled_indices = sampled_pos + sampled_neg
    np.random.shuffle(sampled_indices)
    
    return sampled_indices


def categorize_by_dice(dice_score: float) -> str:
    """Categorize into performance buckets"""
    if dice_score < 0.25:
        return 'poor'
    elif dice_score < 0.50:
        return 'medium'
    elif dice_score < 0.75:
        return 'good'
    else:
        return 'excellent'


class AdiposeUNet:
    """Recreate U-Net model for inference with publication-quality evaluation"""
    
    def __init__(self):
        self.net = None
    
    def build_model(self, init_nb: int = 44, dropout_rate: float = 0.3):
        """Build U-Net with exact same architecture as training"""
        K.set_image_data_format('channels_last')
        
        inputs = Input(shape=(1024, 1024), dtype='float32')
        x = Reshape((1024, 1024, 1))(inputs)
        
        # Encoder
        down1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='down1_conv1')(x)
        down1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='down1_conv2')(down1)
        down1pool = MaxPooling2D((2, 2), strides=(2, 2), name='down1_pool')(down1)
        
        down2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='down2_conv1')(down1pool)
        down2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='down2_conv2')(down2)
        down2pool = MaxPooling2D((2, 2), strides=(2, 2), name='down2_pool')(down2)
        
        down3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='down3_conv1')(down2pool)
        down3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='down3_conv2')(down3)
        down3pool = MaxPooling2D((2, 2), strides=(2, 2), name='down3_pool')(down3)
        
        # Bottleneck with dilated convolutions
        dilate1 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=1, name='dilate1')(down3pool)
        dilate1 = Dropout(dropout_rate, name='dropout_dilate1')(dilate1)
        dilate2 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=2, name='dilate2')(dilate1)
        dilate3 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=4, name='dilate3')(dilate2)
        dilate4 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=8, name='dilate4')(dilate3)
        dilate5 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=16, name='dilate5')(dilate4)
        dilate6 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=32, name='dilate6')(dilate5)
        dilate_all_added = Add(name='dilate_add')([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
        
        # Decoder
        up3 = UpSampling2D((2, 2), name='up3_upsample')(dilate_all_added)
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='up3_conv1')(up3)
        up3 = Concatenate(axis=-1, name='up3_concat')([down3, up3])
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='up3_conv2')(up3)
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='up3_conv3')(up3)
        up3 = Dropout(dropout_rate, name='dropout_up3')(up3)
        
        up2 = UpSampling2D((2, 2), name='up2_upsample')(up3)
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='up2_conv1')(up2)
        up2 = Concatenate(axis=-1, name='up2_concat')([down2, up2])
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='up2_conv2')(up2)
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='up2_conv3')(up2)
        up2 = Dropout(dropout_rate, name='dropout_up2')(up2)
        
        up1 = UpSampling2D((2, 2), name='up1_upsample')(up2)
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='up1_conv1')(up1)
        up1 = Concatenate(axis=-1, name='up1_concat')([down1, up1])
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='up1_conv2')(up1)
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='up1_conv3')(up1)
        up1 = Dropout(dropout_rate, name='dropout_up1')(up1)
        
        # Output: 2-channel softmax (same as training)
        x = Conv2D(2, 1, activation='softmax', name='output_softmax')(up1)
        x = Lambda(lambda z: z[:, :, :, 1:2], output_shape=(1024, 1024, 1), name='output_class1')(x)
        x = Lambda(lambda z: K.squeeze(z, axis=-1), output_shape=(1024, 1024), name='squeeze')(x)
        
        self.net = Model(inputs=inputs, outputs=x)
        return self.net
    
    def load_weights(self, weights_path: str):
        """Load trained weights with backward compatibility for legacy formats"""
        try:
            # Try modern TF2 format first
            self.net.load_weights(weights_path)
            print(f"✓ Loaded weights from {weights_path} (modern format)")
        except KeyError as e:
            if "vars" in str(e):
                # This is likely a legacy weight format, try legacy loading
                print(f"⚠️  Modern loading failed ({e}), attempting legacy format...")
                self.load_legacy_weights(weights_path)
            else:
                # Some other KeyError, re-raise it
                raise
        except Exception as e:
            # Try legacy loading as fallback for any other loading error
            print(f"⚠️  Standard loading failed ({e}), attempting legacy format...")
            self.load_legacy_weights(weights_path)
    
    def load_legacy_weights(self, h5_path: str):
        """Load weights from legacy TF format (backward compatibility)"""
        import h5py
        from tensorflow.python.keras.saving import hdf5_format
        
        try:
            with h5py.File(h5_path, 'r') as f:
                group = f['model_weights'] if 'model_weights' in f else f
                try:
                    hdf5_format.load_weights_from_hdf5_group(group, self.net.layers)
                    print(f"✓ Loaded legacy weights from {h5_path} (strict topology match)")
                except Exception as e:
                    print(f"Strict legacy load failed: {e}")
                    hdf5_format.load_weights_from_hdf5_group_by_name(group, self.net.layers)
                    print(f"✓ Loaded legacy weights from {h5_path} (by layer name - skipped mismatches)")
        except Exception as e:
            raise RuntimeError(f"Failed to load legacy weights from {h5_path}: {e}")
    
    def predict_single(self, image: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Run inference on a single image without TTA (used by TTA class)"""
        # Normalize using training statistics (no data leakage!)
        normalized_img = (image - mean) / (std + 1e-10)
        
        # Add batch dimension and predict
        batch_img = np.expand_dims(normalized_img, axis=0)
        prediction = self.net.predict(batch_img, verbose=0)
        
        return prediction[0]
    
    def predict(self, image: np.ndarray, mean: float, std: float, 
                use_tta: bool = False, tta_mode: str = 'basic') -> Tuple[np.ndarray, dict]:
        """Run inference on a single image with optional TTA
        
        Args:
            image: Input image array
            mean: Normalization mean
            std: Normalization std
            use_tta: Whether to use Test Time Augmentation
            tta_mode: TTA mode ('minimal', 'basic', 'full')
            
        Returns:
            Tuple of (prediction, timing_info)
        """
        if not use_tta:
            # Standard prediction
            start_time = time.time()
            pred = self.predict_single(image, mean, std)
            timing_info = {
                'num_augmentations': 1,
                'total_time': time.time() - start_time,
                'tta_enabled': False
            }
            return pred, timing_info
        else:
            # TTA prediction
            tta = TestTimeAugmentation(mode=tta_mode)
            pred, timing_info = tta.predict_with_tta(self, image, mean, std)
            timing_info['tta_enabled'] = True
            timing_info['tta_mode'] = tta_mode
            return pred, timing_info


def read_image_gray(path: str) -> np.ndarray:
    """
    Load image in grayscale with proper bit-depth handling for TIFFs
    
    Args:
        path: Path to image file
        
    Returns:
        Grayscale image as float32 array
        
    Note:
        Uses tifffile for .tif/.tiff to preserve 12/16-bit depth.
        Uses cv2 for other formats (.jpg, .png, etc.)
    """
    p = Path(path)
    
    if p.suffix.lower() in {'.tif', '.tiff'}:
        # Use tifffile to preserve bit depth
        arr = tiff.imread(str(p))
        
        # Handle RGB TIFFs - convert to grayscale
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        return arr.astype(np.float32)
    else:
        # Use OpenCV for standard formats
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)


def load_validation_data(val_root: str) -> List[Tuple[str, str]]:
    """
    Flexible loader:
    - Recurses under <val_root>/images and <val_root>/masks
    - Supports image exts: .jpg .jpeg .png .tif .tiff
    - Supports mask  exts: .tif .tiff .png .jpg .jpeg
    - Pairs by stem; also tolerates a single '_mask' suffix on masks
    """
    val_root = Path(val_root)
    images_dir = val_root / "images"
    masks_dir  = val_root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Image/mask dirs not found:\n  {images_dir}\n  {masks_dir}")

    img_exts  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    mask_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    # recurse to be robust to nested layout
    image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in img_exts]
    mask_files  = [p for p in masks_dir.rglob("*") if p.suffix.lower() in mask_exts]

    if not image_files:
        raise FileNotFoundError(f"No image files under {images_dir} (looked for {sorted(img_exts)})")
    if not mask_files:
        raise FileNotFoundError(f"No mask files under  {masks_dir} (looked for {sorted(mask_exts)})")

    # index masks by stem and by stem without a single trailing '_mask'
    masks_by_stem: Dict[str, Path] = {}
    for m in mask_files:
        stem = m.stem
        masks_by_stem.setdefault(stem, m)
        if stem.endswith("_mask"):
            masks_by_stem.setdefault(stem[:-5], m)  # tolerate '_mask' suffix

    paired: List[Tuple[str, str]] = []
    missing = 0
    for img in sorted(image_files):
        stem = img.stem
        m = masks_by_stem.get(stem)
        if m is not None:
            paired.append((str(img), str(m)))
        else:
            missing += 1

    if not paired:
        # helpful debug: show a few examples
        sample_imgs  = [p.name for p in image_files[:5]]
        sample_masks = [p.name for p in mask_files[:5]]
        raise FileNotFoundError(
            "No paired image-mask files found.\n"
            f"Sample images: {sample_imgs}\n"
            f"Sample masks:  {sample_masks}\n"
            "Ensure stems match (optionally with '_mask' on masks)."
        )

    print(f"Found {len(paired)} pairs (images: {len(image_files)}, masks: {len(mask_files)}, unpaired images: {missing})")
    return paired


def run_publication_evaluation(val_data_root: str, weights_path: str, output_dir: str,
                             dataset_name: str = "test", optimize_threshold: bool = True,
                             save_visualizations: bool = True, n_vis_samples: int = 20,
                             use_tta: bool = False, tta_mode: str = 'basic',
                             use_sliding_window: bool = False, overlap: float = 0.5,
                             blend_mode: str = 'gaussian', use_boundary_refine: bool = False,
                             refine_kernel: int = 5, adaptive_threshold: bool = False,
                             save_overlays: bool = False, n_positive: int = 120, 
                             n_negative: int = 30) -> ComprehensiveMetrics:
    """
    Run publication-quality evaluation with all methodological improvements
    
    Args:
        val_data_root: Path to validation/test dataset
        weights_path: Path to trained model weights
        output_dir: Output directory for results
        dataset_name: Name of dataset being evaluated
        optimize_threshold: Whether to optimize threshold (only for validation)
        save_visualizations: Whether to save visualization images
        n_vis_samples: Number of sample visualizations to save
        use_tta: Whether to use Test Time Augmentation
        tta_mode: TTA mode ('minimal', 'basic', 'full')
        use_sliding_window: Whether to use sliding window inference
        overlap: Overlap ratio for sliding window (0.0-0.75)
        blend_mode: Blending mode ('gaussian', 'linear', 'none')
        use_boundary_refine: Whether to apply boundary refinement
        refine_kernel: Kernel size for boundary refinement
        adaptive_threshold: Whether to use adaptive threshold optimization
        
    Returns:
        ComprehensiveMetrics object with all results and confidence intervals
    """
    print(f"\n{'='*80}")
    print(f"PUBLICATION-QUALITY EVALUATION: {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    
    # Set deterministic environment
    set_deterministic_seeds(1337)
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log environment info
    print(f"Environment: Python {sys.version.split()[0]}, TensorFlow {tf.__version__}")
    
    # Load training normalization statistics (NO DATA LEAKAGE)
    checkpoint_dir = Path(weights_path).parent
    train_mean, train_std = load_training_stats(str(checkpoint_dir))
    
    # Load validation data
    paired_files = load_validation_data(val_data_root)
    n_files = len(paired_files)
    
    # Load model
    print("\nLoading trained model...")
    model = AdiposeUNet()
    model.build_model()
    model.load_weights(weights_path)
    print("✓ Model loaded successfully")
    
    # Initialize optional post-processing components
    sliding_window = None
    boundary_refiner = None
    
    if use_sliding_window:
        sliding_window = SlidingWindowInference(
            tile_size=1024,
            overlap=overlap,
            blend_mode=blend_mode
        )
        print(f"✓ Sliding window inference enabled")
    
    if use_boundary_refine:
        boundary_refiner = BoundaryRefiner(
            kernel_size=refine_kernel,
            bilateral_d=5,
            bilateral_sigma_color=50,
            bilateral_sigma_space=50
        )
        print(f"✓ Boundary refinement enabled (kernel={refine_kernel})")
    
    # Collect all predictions and ground truths for analysis
    print(f"\nRunning inference on {n_files} samples...")
    
    tile_predictions = []
    tile_ground_truths = []
    tile_images = []
    tile_paths = []
    
    start_time = time.time()
    
    for i, (img_path, mask_path) in enumerate(paired_files):
        # Load image and mask with proper bit-depth handling
        image = read_image_gray(img_path)
        true_mask = tiff.imread(mask_path)
        
        if true_mask.ndim == 3:
            true_mask = true_mask.squeeze()
        
        # Normalize true mask to uint8 [0, 255] for memory efficiency
        true_mask = (true_mask > 0).astype(np.uint8)
        
        # Run prediction with optional enhancements
        if use_sliding_window and sliding_window is not None:
            # Use sliding window inference (handles TTA internally)
            pred_mask = sliding_window.predict_with_sliding_window(
                image, model, train_mean, train_std, 
                use_tta=use_tta, tta_mode=tta_mode
            )
        else:
            # Standard prediction with optional TTA
            pred_mask, timing_info = model.predict(
                image, train_mean, train_std, 
                use_tta=use_tta, tta_mode=tta_mode
            )
        
        pred_mask = pred_mask.astype(np.float32)
        
        # Apply boundary refinement if enabled
        if use_boundary_refine and boundary_refiner is not None:
            pred_mask = boundary_refiner.refine(pred_mask, image)
        
        # Store for analysis (memory-efficient storage)
        tile_predictions.append(pred_mask)
        tile_ground_truths.append(true_mask)
        tile_images.append(image)
        tile_paths.append(img_path)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_files - (i + 1)) / rate if rate > 0 else 0
            print(f"  Processed {i+1}/{n_files} samples | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min")
    
    print(f"✓ Inference completed in {(time.time() - start_time)/60:.1f} minutes")
    
    # Optimize threshold if requested (typically for validation set only)
    if optimize_threshold:
        print(f"\nOptimizing threshold on {dataset_name} set...")
        
        if adaptive_threshold:
            # Two-stage adaptive threshold optimization
            print("Using adaptive two-stage threshold optimization...")
            
            # Stage 1: Coarse search (0.1 to 0.9, step 0.1)
            print("Stage 1: Coarse grid search...")
            coarse_range = np.arange(0.1, 1.0, 0.1)
            coarse_threshold, _ = optimize_threshold_f1_slide_level(
                tile_predictions, tile_ground_truths, tile_paths, coarse_range
            )
            
            # Stage 2: Fine search around best value (±0.1, step 0.01)
            print(f"Stage 2: Fine search around {coarse_threshold:.2f}...")
            fine_min = max(0.1, coarse_threshold - 0.1)
            fine_max = min(0.9, coarse_threshold + 0.1)
            fine_range = np.arange(fine_min, fine_max + 0.01, 0.01)
            optimal_threshold, f1_scores = optimize_threshold_f1_slide_level(
                tile_predictions, tile_ground_truths, tile_paths, fine_range
            )
            del f1_scores
            print(f"✓ Adaptive optimization complete: {optimal_threshold:.3f}")
        else:
            # Standard single-stage optimization
            optimal_threshold, f1_scores = optimize_threshold_f1_slide_level(
                tile_predictions, tile_ground_truths, tile_paths
            )
            # Explicit memory cleanup after threshold optimization
            del f1_scores
    else:
        # Use default threshold or load from previous optimization
        optimal_threshold = 0.5
        print(f"Using fixed threshold: {optimal_threshold}")
    
    # Group tiles by slide for slide-level analysis
    print(f"\nGrouping tiles by slide for slide-level analysis...")
    slide_data = defaultdict(list)
    
    for i, (img_path, pred, true) in enumerate(zip(tile_paths, tile_predictions, tile_ground_truths)):
        slide_id = extract_slide_id(img_path)
        slide_data[slide_id].append({
            'prediction': pred,
            'ground_truth': true,
            'image': tile_images[i],
            'path': img_path
        })
    
    n_slides = len(slide_data)
    print(f"✓ Grouped {n_files} tiles into {n_slides} slides")
    
    # Calculate slide-level metrics
    print(f"\nCalculating slide-level metrics with threshold {optimal_threshold:.2f}...")
    
    slide_metrics = {
        'dice_scores': [],
        'jaccard_indices': [],
        'sensitivities': [],
        'specificities': [],
        'precisions': [],
        'f1_scores': [],
        'accuracies': [],
        'roc_aucs': [],
        'pr_aucs': [],
        'hausdorff95s': [],
        'assds': []
    }
    
    slide_names = []
    
    for slide_id, tiles in slide_data.items():
        slide_names.append(slide_id)
        
        # Aggregate tile metrics for this slide
        slide_tile_metrics = []
        slide_boundary_metrics = []
        slide_auc_metrics = []
        
        for tile in tiles:
            # Calculate pixel-level metrics for this tile
            tile_metrics = calculate_pixel_metrics(tile['prediction'], tile['ground_truth'], optimal_threshold)
            slide_tile_metrics.append(tile_metrics)
            
            # Calculate boundary metrics for this tile
            boundary_metrics = calculate_boundary_metrics(tile['prediction'], tile['ground_truth'], optimal_threshold)
            slide_boundary_metrics.append(boundary_metrics)
            
            # Calculate AUC metrics for this tile
            auc_metrics = calculate_auc_metrics(tile['prediction'], tile['ground_truth'])
            slide_auc_metrics.append(auc_metrics)
        
        # Average metrics across tiles in this slide
        slide_dice = np.mean([m['dice_score'] for m in slide_tile_metrics])
        slide_jaccard = np.mean([m['jaccard_index'] for m in slide_tile_metrics])
        slide_sensitivity = np.mean([m['sensitivity'] for m in slide_tile_metrics])
        slide_specificity = np.mean([m['specificity'] for m in slide_tile_metrics])
        slide_precision = np.mean([m['precision'] for m in slide_tile_metrics])
        slide_f1 = np.mean([m['f1_score'] for m in slide_tile_metrics])
        slide_accuracy = np.mean([m['accuracy'] for m in slide_tile_metrics])
        
        # Handle AUC metrics (filter out NaN values)
        valid_roc_auc = [m['roc_auc'] for m in slide_auc_metrics if np.isfinite(m['roc_auc'])]
        valid_pr_auc = [m['pr_auc'] for m in slide_auc_metrics if np.isfinite(m['pr_auc'])]
        
        slide_roc_auc = np.mean(valid_roc_auc) if valid_roc_auc else np.nan
        slide_pr_auc = np.mean(valid_pr_auc) if valid_pr_auc else np.nan
        
        # Handle boundary metrics (filter out inf/nan values)
        valid_hausdorff = [m['hausdorff95'] for m in slide_boundary_metrics 
                          if np.isfinite(m['hausdorff95'])]
        valid_assd = [m['assd'] for m in slide_boundary_metrics 
                     if np.isfinite(m['assd'])]
        
        slide_hausdorff95 = np.mean(valid_hausdorff) if valid_hausdorff else np.nan
        slide_assd = np.mean(valid_assd) if valid_assd else np.nan
        
        # Store slide-level metrics
        slide_metrics['dice_scores'].append(slide_dice)
        slide_metrics['jaccard_indices'].append(slide_jaccard)
        slide_metrics['sensitivities'].append(slide_sensitivity)
        slide_metrics['specificities'].append(slide_specificity)
        slide_metrics['precisions'].append(slide_precision)
        slide_metrics['f1_scores'].append(slide_f1)
        slide_metrics['accuracies'].append(slide_accuracy)
        slide_metrics['roc_aucs'].append(slide_roc_auc)
        slide_metrics['pr_aucs'].append(slide_pr_auc)
        slide_metrics['hausdorff95s'].append(slide_hausdorff95)
        slide_metrics['assds'].append(slide_assd)
    
    # Convert to numpy arrays for statistical analysis
    for key, values in slide_metrics.items():
        slide_metrics[key] = np.array(values)
    
    print(f"✓ Calculated slide-level metrics for {n_slides} slides")
    
    # Calculate confidence intervals using bootstrap resampling over slides
    print(f"\nCalculating bootstrap confidence intervals (n=10000)...")
    
    # Calculate CIs for each metric (using the top-level safe_bootstrap_ci function)
    dice_mean, dice_ci = safe_bootstrap_ci(slide_metrics['dice_scores'])
    jaccard_mean, jaccard_ci = safe_bootstrap_ci(slide_metrics['jaccard_indices'])
    sensitivity_mean, sensitivity_ci = safe_bootstrap_ci(slide_metrics['sensitivities'])
    specificity_mean, specificity_ci = safe_bootstrap_ci(slide_metrics['specificities'])
    precision_mean, precision_ci = safe_bootstrap_ci(slide_metrics['precisions'])
    f1_mean, f1_ci = safe_bootstrap_ci(slide_metrics['f1_scores'])
    accuracy_mean, accuracy_ci = safe_bootstrap_ci(slide_metrics['accuracies'])
    roc_auc_mean, roc_auc_ci = safe_bootstrap_ci(slide_metrics['roc_aucs'])
    pr_auc_mean, pr_auc_ci = safe_bootstrap_ci(slide_metrics['pr_aucs'])
    hausdorff95_mean, hausdorff95_ci = safe_bootstrap_ci(slide_metrics['hausdorff95s'])
    assd_mean, assd_ci = safe_bootstrap_ci(slide_metrics['assds'])
    
    print(f"✓ Bootstrap confidence intervals calculated")
    
    # Create comprehensive results object
    results = ComprehensiveMetrics(
        dice_score=dice_mean,
        dice_ci=dice_ci,
        jaccard_index=jaccard_mean,
        jaccard_ci=jaccard_ci,
        sensitivity=sensitivity_mean,
        sensitivity_ci=sensitivity_ci,
        specificity=specificity_mean,
        specificity_ci=specificity_ci,
        precision=precision_mean,
        precision_ci=precision_ci,
        f1_score=f1_mean,
        f1_ci=f1_ci,
        accuracy=accuracy_mean,
        accuracy_ci=accuracy_ci,
        roc_auc=roc_auc_mean,
        roc_auc_ci=roc_auc_ci,
        pr_auc=pr_auc_mean,
        pr_auc_ci=pr_auc_ci,
        hausdorff95=hausdorff95_mean,
        hausdorff95_ci=hausdorff95_ci,
        assd=assd_mean,
        assd_ci=assd_ci,
        n_slides=n_slides,
        n_tiles=n_files,
        optimal_threshold=optimal_threshold
    )
    
    # Save comprehensive results table
    results_df = pd.DataFrame({
        'Metric': ['Dice Score', 'Jaccard Index (IoU)', 'Sensitivity (Recall)', 'Specificity', 
                   'Precision', 'F1 Score', 'Accuracy', 'ROC AUC', 'PR AUC', 'Hausdorff95', 'ASSD'],
        'Mean': [dice_mean, jaccard_mean, sensitivity_mean, specificity_mean, 
                precision_mean, f1_mean, accuracy_mean, roc_auc_mean, pr_auc_mean, hausdorff95_mean, assd_mean],
        'CI_Lower': [dice_ci[0], jaccard_ci[0], sensitivity_ci[0], specificity_ci[0], 
                    precision_ci[0], f1_ci[0], accuracy_ci[0], roc_auc_ci[0], pr_auc_ci[0], hausdorff95_ci[0], assd_ci[0]],
        'CI_Upper': [dice_ci[1], jaccard_ci[1], sensitivity_ci[1], specificity_ci[1], 
                    precision_ci[1], f1_ci[1], accuracy_ci[1], roc_auc_ci[1], pr_auc_ci[1], hausdorff95_ci[1], assd_ci[1]],
        'N_Slides': [n_slides] * 11,
        'N_Tiles': [n_files] * 11
    })
    
    # Format results for display
    results_df['Mean_CI'] = results_df.apply(
        lambda row: f"{row['Mean']:.4f} [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]", axis=1
    )
    
    # Save results table
    results_table_path = output_dir / f"{dataset_name}_comprehensive_results.csv"
    results_df.to_csv(results_table_path, index=False)
    print(f"✓ Saved results table: {results_table_path}")
    
    # Create 4-panel overlays organized by Dice buckets if requested
    if save_overlays:
        print(f"\nGenerating 4-panel overlay visualizations...")
        
        # Sample tiles (80/20 positive/negative split)
        sampled_indices = sample_tiles(tile_predictions, tile_ground_truths, tile_paths, 
                                       n_positive=n_positive, n_negative=n_negative)
        
        print(f"\n[Overlays] Processing {len(sampled_indices)} sampled tiles...")
        
        # Create overlay directories
        overlays_dir = output_dir / "overlays"
        for bucket in ['poor', 'medium', 'good', 'excellent']:
            (overlays_dir / bucket).mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        bucket_counts = {'poor': 0, 'medium': 0, 'good': 0, 'excellent': 0}
        overlay_dice_scores = []
        
        # Generate overlays for sampled tiles
        from tqdm import tqdm
        for i, idx in enumerate(tqdm(sampled_indices, desc="Creating overlays")):
            # Get data for this tile
            pred = tile_predictions[idx]
            true = tile_ground_truths[idx]
            img_gray = tile_images[idx]
            img_path = tile_paths[idx]
            
            # Load RGB version of image for visualization
            img_rgb = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
            # Calculate Dice score
            dice = calculate_pixel_metrics(pred, true, optimal_threshold)['dice_score']
            overlay_dice_scores.append(dice)
            
            # Categorize by Dice score
            bucket = categorize_by_dice(dice)
            bucket_counts[bucket] += 1
            
            # Create output filename
            image_name = Path(img_path).stem
            output_filename = f"{bucket}_{i+1:03d}_{image_name}_dice_{dice:.3f}.png"
            output_path = overlays_dir / bucket / output_filename
            
            # Generate 4-panel visualization
            create_4panel_visualization(img_rgb, true, pred, dice, str(output_path))
        
        # Generate summary
        summary_path = overlays_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"OVERLAY VISUALIZATION SUMMARY: {dataset_name.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples: {len(sampled_indices)}\n")
            f.write(f"Positive tiles: {n_positive}\n")
            f.write(f"Negative tiles: {n_negative}\n")
            f.write(f"Threshold: {optimal_threshold:.3f}\n\n")
            
            f.write("DICE SCORE STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean Dice: {np.mean(overlay_dice_scores):.4f}\n")
            f.write(f"Median Dice: {np.median(overlay_dice_scores):.4f}\n")
            f.write(f"Std Dice: {np.std(overlay_dice_scores):.4f}\n")
            f.write(f"Min Dice: {np.min(overlay_dice_scores):.4f}\n")
            f.write(f"Max Dice: {np.max(overlay_dice_scores):.4f}\n\n")
            
            f.write("BUCKET DISTRIBUTION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Poor (< 25%):       {bucket_counts['poor']:3d} images\n")
            f.write(f"Medium (25-50%):    {bucket_counts['medium']:3d} images\n")
            f.write(f"Good (50-75%):      {bucket_counts['good']:3d} images\n")
            f.write(f"Excellent (75-100%): {bucket_counts['excellent']:3d} images\n")
        
        print(f"✓ Saved {len(sampled_indices)} overlay visualizations to: {overlays_dir}")
        print(f"  Bucket distribution: Poor={bucket_counts['poor']}, Medium={bucket_counts['medium']}, " 
              f"Good={bucket_counts['good']}, Excellent={bucket_counts['excellent']}")
        print(f"✓ Summary saved to: {summary_path}")
    
    # Create sample visualizations if requested
    elif save_visualizations:
        print(f"\nCreating sample visualizations (n={min(n_vis_samples, n_files)})...")
        
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Select diverse samples for visualization
        n_viz = min(n_vis_samples, n_files)
        viz_indices = np.linspace(0, n_files - 1, n_viz, dtype=int)
        
        for i, idx in enumerate(viz_indices):
            img_path = tile_paths[idx]
            image_name = Path(img_path).stem
            
            output_path = viz_dir / f"{dataset_name}_sample_{i+1:02d}_{image_name}.png"

            # Calculate dice score for this tile
            dice = calculate_pixel_metrics(tile_predictions[idx],
                                          tile_ground_truths[idx],
                                          optimal_threshold)['dice_score']
            
            create_4panel_visualization(
                tile_images[idx],
                tile_ground_truths[idx],    # GT in correct position
                tile_predictions[idx],      # PRED in correct position
                dice,                       # dice_score, not threshold
                str(output_path)            # removed extra image_name
            )
            
            if (i + 1) % 5 == 0:
                print(f"  Created {i+1}/{n_viz} visualizations")
        
        print(f"✓ Saved visualizations to: {viz_dir}")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"PUBLICATION-QUALITY RESULTS SUMMARY: {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {n_slides} slides, {n_files} tiles")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Bootstrap Samples: 10,000")
    print(f"")
    print(f"{'Metric':<20} {'Mean (95% CI)':<30} {'Range':<25}")
    print(f"{'-'*75}")
    
    metrics_display = [
        ('Dice Score', dice_mean, dice_ci, np.min(slide_metrics['dice_scores']), np.max(slide_metrics['dice_scores'])),
        ('Jaccard (IoU)', jaccard_mean, jaccard_ci, np.min(slide_metrics['jaccard_indices']), np.max(slide_metrics['jaccard_indices'])),
        ('Sensitivity', sensitivity_mean, sensitivity_ci, np.min(slide_metrics['sensitivities']), np.max(slide_metrics['sensitivities'])),
        ('Specificity', specificity_mean, specificity_ci, np.min(slide_metrics['specificities']), np.max(slide_metrics['specificities'])),
        ('Precision', precision_mean, precision_ci, np.min(slide_metrics['precisions']), np.max(slide_metrics['precisions'])),
        ('F1 Score', f1_mean, f1_ci, np.min(slide_metrics['f1_scores']), np.max(slide_metrics['f1_scores'])),
        ('Accuracy', accuracy_mean, accuracy_ci, np.min(slide_metrics['accuracies']), np.max(slide_metrics['accuracies'])),
    ]
    
    for name, mean_val, ci, min_val, max_val in metrics_display:
        ci_str = f"{mean_val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        range_str = f"[{min_val:.4f}, {max_val:.4f}]"
        print(f"{name:<20} {ci_str:<30} {range_str:<25}")
    
    # Handle boundary metrics separately (may have NaN values)
    if not np.isnan(hausdorff95_mean):
        valid_hausdorff = slide_metrics['hausdorff95s'][np.isfinite(slide_metrics['hausdorff95s'])]
        ci_str = f"{hausdorff95_mean:.2f} [{hausdorff95_ci[0]:.2f}, {hausdorff95_ci[1]:.2f}]"
        range_str = f"[{np.min(valid_hausdorff):.2f}, {np.max(valid_hausdorff):.2f}]"
        print(f"{'Hausdorff95 (px)':<20} {ci_str:<30} {range_str:<25}")
    
    if not np.isnan(assd_mean):
        valid_assd = slide_metrics['assds'][np.isfinite(slide_metrics['assds'])]
        ci_str = f"{assd_mean:.2f} [{assd_ci[0]:.2f}, {assd_ci[1]:.2f}]"
        range_str = f"[{np.min(valid_assd):.2f}, {np.max(valid_assd):.2f}]"
        print(f"{'ASSD (px)':<20} {ci_str:<30} {range_str:<25}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"✓ Results table: {results_table_path}")
    
    print(f"{'='*80}\n")
    
    return results


def main():
    """Main execution function with redesigned flag-based evaluation"""
    parser = argparse.ArgumentParser(
        description="Publication-Quality Full Evaluation for Adipose U-Net",
        formatter_class=esRawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate human-test with checkpoint
  python full_evaluation.py --weights checkpoints/20251024_150723_adipose_sybreosin_1024_finetune --human-test
  
  # Evaluate multiple test sets
  python full_evaluation.py --weights checkpoints/20251024_150723_adipose_sybreosin_1024_finetune --human-test --clean-test
  
  # Evaluate timestamp-matched test set
  python full_evaluation.py --weights checkpoints/20251024_150723_adipose_sybreosin_1024_finetune --test
  
  # Custom data root
  python full_evaluation.py --weights /path/to/weights.h5 --data-root /custom/path --human-test
        """
    )
    
    # Required
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights (required)')
    
    # Optional paths
    parser.add_argument('--data-root', type=str, default='',
                       help='Path to dataset root directory (default: /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_test/stain_normalized)')
    parser.add_argument('--output', type=str, default='',
                       help='Output directory for results (auto-places in checkpoint folder if not specified)')
    
    # Dataset selection flags (combinable)
    parser.add_argument('--val', action='store_true',
                       help='Evaluate val dataset using timestamp-matched build')
    parser.add_argument('--test', action='store_true',
                       help='Evaluate test dataset using timestamp-matched build')
    parser.add_argument('--human-test', action='store_true',
                       help='Evaluate human_test dataset')
    parser.add_argument('--clean-test', action='store_true',
                       help='Evaluate clean_test dataset')
    parser.add_argument('--clean-test-50-overlap', action='store_true',
                       help='Evaluate clean_test_50_overlap dataset')
    
    # Data source selection (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--stain', action='store_true',
                       help='Use stain normalized data')
    data_group.add_argument('--original', action='store_true',
                       help='Use original data')
    
    # Evaluation options
    parser.add_argument('--optimize-threshold', action='store_true',
                       help='Optimize threshold on validation set (not typically used with test flags)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip saving sample visualizations (faster execution)')
    parser.add_argument('--n-vis-samples', type=int, default=10,
                       help='Number of sample visualizations to create (default: 10)')
    
    # TTA configuration
    parser.add_argument('--use-tta', action='store_true', default=False,
                       help='Enable Test Time Augmentation for improved predictions (default: False)')
    parser.add_argument('--tta-mode', type=str, default='basic', choices=['minimal', 'basic', 'full'],
                       help='TTA mode: minimal (2 augs), basic (4 augs), full (8 augs)')
    
    # Enhanced post-processing options
    parser.add_argument('--sliding-window', action='store_true', default=False,
                       help='Enable sliding window inference with overlapping tiles')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for sliding window inference (0.0-0.75, default: 0.5)')
    parser.add_argument('--blend-mode', type=str, default='gaussian', choices=['gaussian', 'linear', 'none'],
                       help='Blending mode for sliding window: gaussian (default), linear, none')
    parser.add_argument('--boundary-refine', action='store_true', default=False,
                       help='Enable morphological boundary refinement')
    parser.add_argument('--refine-kernel', type=int, default=5,
                       help='Kernel size for boundary refinement (default: 5)')
    parser.add_argument('--adaptive-threshold', action='store_true', default=False,
                       help='Use adaptive two-stage threshold optimization (0.1-0.9)')
    
    # Overlay visualization options
    parser.add_argument('--save-overlays', action='store_true', default=False,
                       help='Save 4-panel overlay visualizations organized by Dice score buckets')
    parser.add_argument('--n-positive', type=int, default=120,
                       help='Number of positive tiles to sample for overlays (default: 120)')
    parser.add_argument('--n-negative', type=int, default=30,
                       help='Number of negative tiles to sample for overlays (default: 30)')
    
    args = parser.parse_args()
    
    # Determine which datasets to evaluate
    datasets_to_evaluate = []
    
    if args.val:
        datasets_to_evaluate.append('val')
        print("🔗 --val flag: Adding timestamp-matched validation dataset")
    
    if args.test:
        datasets_to_evaluate.append('test')
        print("🔗 --test flag: Adding timestamp-matched test dataset")
    
    if args.human_test:
        if 'human_test' not in datasets_to_evaluate:
            datasets_to_evaluate.append('human_test')
            print("🔗 --human-test flag: Adding human_test dataset")
    
    if args.clean_test:
        if 'clean_test' not in datasets_to_evaluate:
            datasets_to_evaluate.append('clean_test')
            print("🔗 --clean-test flag: Adding clean_test dataset")
    
    if args.clean_test_50_overlap:
        if 'clean_test_50_overlap' not in datasets_to_evaluate:
            datasets_to_evaluate.append('clean_test_50_overlap')
            print("🔗 --clean-test-50-overlap flag: Adding clean_test_50_overlap dataset")
    
    # Validate at least one dataset is selected
    if not datasets_to_evaluate:
        print("❌ No datasets specified. Please use one or more of: --val, --test, --human-test, --clean-test")
        return 1
    
    # Validate data source selection for human-test/clean-test datasets
    requires_data_source = [d for d in datasets_to_evaluate if d in ['human_test', 'clean_test', 'clean_test_50_overlap']]
    if requires_data_source and not args.stain and not args.original:
        print(f"❌ Data source must be specified for {requires_data_source}.")
        print("Please use either --stain or --original flag.")
        return 1
    
    # Fix: Make adaptive-threshold imply optimize-threshold
    # If user passes --adaptive-threshold, enable optimization automatically
    opt_thresh = args.optimize_threshold or args.adaptive_threshold
    
    # Setup pipeline
    print(f"\n{'='*80}")
    print("PUBLICATION-QUALITY EVALUATION PIPELINE")
    print(f"{'='*80}")
    print(f"🎯 Datasets to evaluate: {datasets_to_evaluate}")
    
    # Track results from all datasets and their actual output directories
    all_results = {}
    all_output_dirs = {}
    
    for i, dataset_name in enumerate(datasets_to_evaluate):
        print(f"\n{'='*80}")
        print(f"EVALUATING DATASET {i+1}/{len(datasets_to_evaluate)}: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Resolve paths for this dataset
            use_timestamp_matching = (dataset_name in ['val', 'test'])
            
            # For timestamp-matched datasets (--test, --val), ignore stain flags and let resolve_checkpoint_paths handle it
            if use_timestamp_matching:
                # Force timestamp matching by passing empty data_root_arg
                default_dataroot = "" if not args.data_root else args.data_root
                if not args.data_root:
                    print(f"🔗 Timestamp matching enabled for {dataset_name} → Will auto-detect matching dataset build")
            else:
                # Apply stain flag logic only for --human-test and --clean-test datasets
                if not args.data_root:
                    if args.stain:
                        default_dataroot = "/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_test/stain_normalized"
                    else:
                        default_dataroot = "/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_test/original"
                    print(f"🧪 Stain flag: {'enabled' if args.stain else 'disabled'} → Using {default_dataroot}")
                else:
                    default_dataroot = args.data_root
            
            # Debug: Print what we're about to pass
            print(f"[DEBUG] Calling resolve_checkpoint_paths with:")
            print(f"  data_root_arg={repr(default_dataroot)}")
            print(f"  use_test_timestamp={use_timestamp_matching}")
            
            weights_path, output_path, data_root_path = resolve_checkpoint_paths(
                weights_arg=args.weights,
                output_arg=args.output,
                data_root_arg=default_dataroot,
                use_test_timestamp=use_timestamp_matching
            )
            
            # Determine dataset path
            dataset_path = Path(data_root_path) / dataset_name
            
            # Validate dataset exists
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                print(f"   Available datasets in {data_root_path}:")
                available = [d.name for d in Path(data_root_path).iterdir() if d.is_dir()]
                print(f"   {available}")
                continue
            
            print(f"📂 Dataset path: {dataset_path}")
            
            # Build output directory name with data source and TTA configuration
            # Format: {dataset}_{stain|original}_tta_{mode} or {dataset}_{stain|original}
            
            # Detect data source from actual dataset path for timestamp-matched or from flags
            if use_timestamp_matching:
                # For val/test, infer from build path (contains stain_normalized or not)
                data_source_suffix = 'stain' if 'stain' in str(data_root_path).lower() else 'original'
            else:
                # For human-test/clean-test, use explicit flags
                data_source_suffix = 'stain' if args.stain else 'original'
            
            # Build comprehensive output folder name with all enhancement flags
            enhancement_suffixes = []
            
            # TTA
            if args.use_tta:
                enhancement_suffixes.append(f"tta_{args.tta_mode}")
            
            # Sliding window
            if args.sliding_window:
                sw_suffix = f"sw_{args.blend_mode}"
                if args.overlap != 0.5:
                    sw_suffix += f"_o{int(args.overlap*100)}"
                enhancement_suffixes.append(sw_suffix)
            
            # Boundary refinement
            if args.boundary_refine:
                refine_suffix = "refine"
                if args.refine_kernel != 5:
                    refine_suffix += f"{args.refine_kernel}"
                enhancement_suffixes.append(refine_suffix)
            
            # Adaptive threshold
            if args.adaptive_threshold:
                enhancement_suffixes.append("adaptive")
            
            # Build final folder name
            if enhancement_suffixes:
                output_folder_name = f"{dataset_name}_{data_source_suffix}_{'_'.join(enhancement_suffixes)}"
            else:
                output_folder_name = f"{dataset_name}_{data_source_suffix}"
            
            # Setup output directory
            output_dir = Path(output_path) / output_folder_name
            
            # Configuration logging
            print(f"\n📊 Configuration:")
            print(f"  Weights: {weights_path}")
            print(f"  Dataset: {dataset_path}")
            print(f"  Output: {output_dir}")
            print(f"  Threshold optimization: {args.optimize_threshold}")
            print(f"  Visualizations: {not args.no_visualizations}")
            print(f"  TTA enabled: {args.use_tta}")
            if args.use_tta:
                print(f"  TTA mode: {args.tta_mode}")
            
            # Run evaluation for this dataset with enhanced post-processing
            results = run_publication_evaluation(
                val_data_root=str(dataset_path),
                weights_path=weights_path,
                output_dir=str(output_dir),
                dataset_name=dataset_name,
                optimize_threshold=args.optimize_threshold,
                save_visualizations=not args.no_visualizations,
                n_vis_samples=args.n_vis_samples,
                use_tta=args.use_tta,
                tta_mode=args.tta_mode,
                use_sliding_window=args.sliding_window,
                overlap=args.overlap,
                blend_mode=args.blend_mode,
                use_boundary_refine=args.boundary_refine,
                refine_kernel=args.refine_kernel,
                adaptive_threshold=args.adaptive_threshold,
                save_overlays=args.save_overlays,
                n_positive=args.n_positive,
                n_negative=args.n_negative
            )
            
            all_results[dataset_name] = results
            all_output_dirs[dataset_name] = str(output_dir)
            
            print(f"✅ {dataset_name.upper()} evaluation completed!")
            print(f"   Dice = {results.dice_score:.4f} [{results.dice_ci[0]:.4f}, {results.dice_ci[1]:.4f}]")
            print(f"   Slides: {results.n_slides}, Tiles: {results.n_tiles}")
            print(f"   Threshold: {results.optimal_threshold:.3f}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary with comprehensive error handling
    if all_results:
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY: {len(all_results)} DATASETS EVALUATED")
        print(f"{'='*80}")
        
        for dataset_name, results in all_results.items():
            print(f"{dataset_name.upper():<12} | "
                  f"Dice: {results.dice_score:.4f} [{results.dice_ci[0]:.4f}, {results.dice_ci[1]:.4f}] | "
                  f"Slides: {results.n_slides:3d} | "
                  f"Thresh: {results.optimal_threshold:.3f}")
        
        print(f"\n📂 Results saved in evaluation directories:")
        for dataset_name in all_results.keys():
            # Use the actual tracked output directory
            result_dir = all_output_dirs[dataset_name]
            print(f"   {dataset_name}: {result_dir}")
        
        # Check if some datasets failed
        failed_datasets = [d for d in datasets_to_evaluate if d not in all_results]
        if failed_datasets:
            print(f"\n⚠️  Failed datasets: {failed_datasets}")
            print(f"   Successful: {len(all_results)}/{len(datasets_to_evaluate)}")
        
        return 0
    else:
        # All datasets failed - provide comprehensive error message
        print(f"\n{'='*80}")
        print("❌ ALL DATASET EVALUATIONS FAILED")
        print(f"{'='*80}")
        print(f"None of the {len(datasets_to_evaluate)} specified datasets could be evaluated successfully.")
        print(f"Failed datasets: {datasets_to_evaluate}")
        print("\nCommon causes:")
        print("1. Dataset directories not found in the specified data root")
        print("2. Missing or corrupt image/mask files")
        print("3. Incompatible data source selection (missing --stain/--original flags)")
        print("4. Invalid checkpoint or weights files")
        print("5. Timestamp mismatch between checkpoint and dataset build")
        print("\nCheck the individual error messages above for specific details.")
        return 1


if __name__ == "__main__":
    # GPU memory growth setup
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    
    exit(main())
