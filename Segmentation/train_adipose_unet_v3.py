"""
Advanced U-Net Training for Sybr Gold + Eosin Adipose Segmentation
TensorFlow 2.13 / Python 3.10

VERSION 3: Advanced training techniques
- Deep supervision (auxiliary outputs at decoder stages)
- Online hard example mining (focus on difficult pixels)
- EMA weights (exponential moving average for better generalization)
- Optional AdamW optimizer
- Smart pretrained weight loading from v2
- All v2 features retained (stain norm, TTA-style aug, 2-phase training)

USAGE EXAMPLES:

1. Standard two-phase training (recommended):
   python Segmentation/train_adipose_unet_v3.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
     --epochs-phase1 50 --epochs-phase2 100 \
     --batch-size 2

2. Train with deep supervision and hard example mining:
   python Segmentation/train_adipose_unet_v3.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
     --epochs-phase1 50 --epochs-phase2 100 \
     --use-deep-supervision \
     --use-hard-example-mining \
     --ohem-ratio 0.25

3. Train with EMA and AdamW optimizer:
   python Segmentation/train_adipose_unet_v3.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
     --epochs-phase1 50 --epochs-phase2 100 \
     --use-ema --ema-decay 0.999 \
     --use-adamw --weight-decay 1e-4

4. Resume from checkpoint:
   python Segmentation/train_adipose_unet_v3.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
     --resume-from checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/phase1_best.weights.h5 \
     --epochs-phase2 100

5. All advanced features enabled:
   python Segmentation/train_adipose_unet_v3.py \
     --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
     --epochs-phase1 50 --epochs-phase2 100 \
     --use-deep-supervision \
     --use-hard-example-mining --ohem-ratio 0.25 \
     --use-ema --ema-decay 0.999 \
     --use-adamw --weight-decay 1e-4 \
     --batch-size 2

OUTPUT STRUCTURE:
  checkpoints/
    └── YYYYMMDD_HHMMSS_adipose_sybreosin{suffix}_1024_finetune/
        ├── normalization_stats.json      # Mean/std for inference
        ├── phase1_best.weights.h5        # Best Phase 1 weights
        ├── phase2_best.weights.h5        # Best Phase 2 weights
        ├── weights_best_overall.weights.h5  # FINAL MODEL (use this)
        ├── phase1_training.log           # CSV metrics (Phase 1)
        ├── phase2_training.log           # CSV metrics (Phase 2)
        └── training_settings.log         # Hyperparameters
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime
import glob
import re
import random

# Configure determinism BEFORE importing TensorFlow
# NOTE: TF_DETERMINISTIC_OPS=1 causes issues with UpSampling2D gradient on GPU
# Disabled to allow training with U-Net architecture
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(865)

import numpy as np
import tensorflow as tf

# Add parent directory to path for src module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load seed from seed.csv for reproducibility
from src.utils.seed_utils import get_project_seed
GLOBAL_SEED = get_project_seed()
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# NOTE: tf.config.experimental.enable_op_determinism() disabled
# Reason: ResizeNearestNeighborGrad (used in UpSampling2D) has no deterministic GPU implementation
# Seeds above provide sufficient reproducibility for most purposes
print(f"⚙️ Random seeds set: {GLOBAL_SEED} (deterministic ops disabled for GPU compatibility)")
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,
    Concatenate, Lambda, Reshape, Add,
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, Callback, LearningRateScheduler
)
from tensorflow.keras.losses import binary_crossentropy

import cv2
import tifffile as tiff
import platform
import subprocess

# Local utils
from src.utils.runtime import funcname
from src.utils.model import dice_coef
from src.utils.data import (
    augment_pair_light, augment_pair_moderate, augment_pair_heavy,
    augment_pair_tta_style, normalize_image
)


# ---- Build Directory Management (from v2) -------------------------------------------

def find_most_recent_build_dir(base_data_root: Path) -> Path:
    """Find the most recent _build_* directory or fall back to _build."""
    base_data_root = Path(base_data_root).expanduser()
    
    build_pattern = str(base_data_root / "_build_*")
    timestamped_builds = glob.glob(build_pattern)
    
    if timestamped_builds:
        build_dirs = []
        for build_path in timestamped_builds:
            build_dir = Path(build_path)
            match = re.search(r'_build_(\d{8}_\d{6})$', build_dir.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    build_dirs.append((timestamp, build_dir))
                except ValueError:
                    continue
        
        if build_dirs:
            build_dirs.sort(key=lambda x: x[0], reverse=True)
            most_recent = build_dirs[0][1]
            print(f"🔍 Found {len(build_dirs)} timestamped build directories")
            print(f"📂 Using most recent: {most_recent}")
            return most_recent
    
    original_build = base_data_root / "_build"
    if original_build.exists():
        print(f"📂 Using original build directory: {original_build}")
        return original_build
    
    raise FileNotFoundError(
        f"No build directories found in {base_data_root}. "
        f"Available options:\n"
        f"  1. Run build_dataset.py to create a new dataset\n"
        f"  2. Specify a specific build directory with --data-root"
    )


def resolve_data_root(data_root_arg: str) -> Tuple[Path, str]:
    """Resolve the data root argument to a valid build directory and extract timestamp."""
    data_root = Path(data_root_arg).expanduser()
    
    def extract_timestamp_from_build_dir(build_path: Path) -> str:
        match = re.search(r'_build_(\d{8}_\d{6})$', build_path.name)
        if match:
            return match.group(1)
        return None
    
    if data_root.name.startswith("_build"):
        if data_root.exists() and (data_root / "dataset").exists():
            timestamp = extract_timestamp_from_build_dir(data_root)
            print(f"📂 Using specified build directory: {data_root}")
            if timestamp:
                print(f"📅 Extracted build timestamp: {timestamp}")
            return data_root, timestamp
        else:
            raise FileNotFoundError(
                f"Specified build directory not found or incomplete: {data_root}\n"
                f"Expected structure: {data_root}/dataset/train/, {data_root}/dataset/val/"
            )
    
    elif (data_root / "Pseudocolored").exists() or (data_root / "Masks").exists():
        recent_build = find_most_recent_build_dir(data_root)
        timestamp = extract_timestamp_from_build_dir(recent_build)
        if timestamp:
            print(f"📅 Extracted build timestamp: {timestamp}")
        return recent_build, timestamp
    
    elif (data_root / "dataset").exists():
        print(f"📂 Using specified directory with dataset: {data_root}")
        timestamp = extract_timestamp_from_build_dir(data_root)
        if timestamp:
            print(f"📅 Extracted build timestamp: {timestamp}")
        return data_root, timestamp
    
    else:
        raise FileNotFoundError(
            f"Could not find valid dataset at: {data_root}\n"
            f"Please specify one of:\n"
            f"  1. Base data directory (e.g., /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane)\n"
            f"  2. Specific build directory (e.g., /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20241031_124305)\n"
            f"  3. Any directory containing dataset/train/ and dataset/val/"
        )


# ---- Loss Functions ----------------------

def dice_loss(y_true, y_pred):
    """Dice loss for better boundary detection"""
    smooth = 1.0
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - score


def combined_loss_standard(y_true, y_pred):
    """
    Standard combined loss (PROVEN default for adipocyte segmentation).
    BCE + Dice
    
    Stable and effective for most medical image segmentation tasks.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    
    return bce + dice


def combined_loss_with_label_smoothing(y_true, y_pred, epsilon_pos=0.03, epsilon_neg=0.07):
    """
    NEW in v3.1: Asymmetric label smoothing for conservative annotations.
    
    Addresses:
    - Conservative annotation (only clear adipose labeled) → more uncertainty on background
    - Fuzzy/out-of-focus boundaries → graduated confidence
    - Model overconfidence → forces calibrated probabilities
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        epsilon_pos: Smoothing for positive class (adipose). Lower = more confident.
                    Recommended: 0.02-0.05 (default: 0.03)
        epsilon_neg: Smoothing for negative class (background). Higher = more uncertainty.
                    Recommended: 0.05-0.10 (default: 0.07)
                    
    Math:
        Without smoothing: adipose=1.0, background=0.0
        With smoothing:    adipose→(1-ε_pos-ε_neg)=0.90, background→ε_neg=0.07
        
    Returns:
        Combined BCE + Dice loss on smoothed labels
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Asymmetric smoothing: background gets more uncertainty
    # positive (1) → (1 - epsilon_pos - epsilon_neg)
    # negative (0) → epsilon_neg
    y_smooth = y_true * (1.0 - epsilon_pos - epsilon_neg) + epsilon_neg
    
    bce = binary_crossentropy(y_smooth, y_pred)
    dice = dice_loss(y_smooth, y_pred)
    
    return bce + dice


def online_hard_example_mining_loss(y_true, y_pred, keep_ratio=0.7):
    """
    NEW in v3: Online hard example mining loss.
    
    Focuses training on the hardest pixels (highest loss).
    Perfect for blob segmentation where some regions are trickier.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        keep_ratio: Ratio of hardest pixels to keep (0.5-0.9)
        
    Returns:
        Combined loss focusing on hard examples
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute per-pixel BCE
    per_pixel_bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Flatten per batch
    batch_size = tf.shape(per_pixel_bce)[0]
    flat_loss = tf.reshape(per_pixel_bce, [batch_size, -1])
    
    # Keep top-k hardest pixels per image
    num_pixels = tf.shape(flat_loss)[1]
    k = tf.cast(tf.cast(num_pixels, tf.float32) * keep_ratio, tf.int32)
    
    # Get top-k hardest pixels
    top_k_loss, _ = tf.nn.top_k(flat_loss, k=k, sorted=False)
    hard_bce = tf.reduce_mean(top_k_loss)
    
    # Combine with standard dice (computed globally for shape consistency)
    dice = dice_loss(y_true, y_pred)
    
    return hard_bce + dice


def online_hard_example_mining_loss_with_smoothing(y_true, y_pred, keep_ratio=0.7,
                                                    epsilon_pos=0.03, epsilon_neg=0.07):
    """
    NEW in v3.1: Hard mining + Label smoothing combined.
    
    Best of both worlds:
    - Hard mining: Focus on difficult pixels
    - Label smoothing: Reduce overconfidence, handle annotation uncertainty
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        keep_ratio: Ratio of hardest pixels to keep (0.5-0.9)
        epsilon_pos: Smoothing for positive class (0.02-0.05)
        epsilon_neg: Smoothing for negative class (0.05-0.10)
        
    Returns:
        Combined hard mining loss on smoothed labels
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Apply asymmetric label smoothing
    y_smooth = y_true * (1.0 - epsilon_pos - epsilon_neg) + epsilon_neg
    
    # Compute per-pixel BCE on smoothed labels
    per_pixel_bce = tf.keras.losses.binary_crossentropy(y_smooth, y_pred)
    
    # Flatten per batch
    batch_size = tf.shape(per_pixel_bce)[0]
    flat_loss = tf.reshape(per_pixel_bce, [batch_size, -1])
    
    # Keep top-k hardest pixels
    num_pixels = tf.shape(flat_loss)[1]
    k = tf.cast(tf.cast(num_pixels, tf.float32) * keep_ratio, tf.int32)
    
    top_k_loss, _ = tf.nn.top_k(flat_loss, k=k, sorted=False)
    hard_bce = tf.reduce_mean(top_k_loss)
    
    # Dice on smoothed labels
    dice = dice_loss(y_smooth, y_pred)
    
    return hard_bce + dice


# ---- EMA Callback (NEW in v3) ----------------------

class CosineAnnealingWithWarmup(Callback):
    """
    NEW in v3.2: Cosine annealing learning rate schedule with linear warmup.
    
    Provides smoother convergence than step-based ReduceLROnPlateau:
    - Warmup phase: Linear increase from 0 to max_lr (first few epochs)
    - Annealing phase: Cosine decay from max_lr to min_lr
    
    Benefits:
    - Warmup stabilizes early training (especially with frozen encoder)
    - Cosine decay provides smooth exploration-exploitation transition
    - No plateau detection needed (deterministic schedule)
    
    Expected impact: +1-3% Dice improvement vs ReduceLROnPlateau
    """
    
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, verbose=1):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = (self.max_lr / self.warmup_epochs) * (epoch + 1)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        K.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose and epoch % 5 == 0:
            print(f"\n📈 Epoch {epoch + 1}: Learning rate = {lr:.2e}")


class EMACallback(Callback):
    """
    NEW in v3: Exponential Moving Average of model weights.
    
    Maintains a running average of model weights for better generalization.
    At inference time, use EMA weights instead of final weights.
    
    This is a "free lunch" - no training cost, better performance!
    """
    
    def __init__(
        self,
        decay=0.995,
        save_ema_weights=True,
        checkpoint_dir=None,
        monitor: str | None = None,
        mode: str = "max",
        save_best_only: bool = False,
    ):
        super().__init__()
        self.decay = decay
        self.save_ema_weights = save_ema_weights
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.ema_weights = None
        self.monitor = monitor
        self.mode = mode.lower()
        self.save_best_only = save_best_only
        if self.mode not in ("min", "max"):
            raise ValueError("EMACallback mode must be 'min' or 'max'")
        self.best = -np.inf if self.mode == "max" else np.inf
        self._best_saved = False
        self._missing_monitor_warned = False
        
    def on_train_begin(self, logs=None):
        print(f"\n🔄 EMA Callback initialized (decay={self.decay})")
        
    def on_epoch_end(self, epoch, logs=None):
        # Get current weights
        current_weights = self.model.get_weights()
        
        if self.ema_weights is None:
            # Initialize EMA weights
            self.ema_weights = [w.copy() for w in current_weights]
            print(f"  EMA weights initialized at epoch {epoch + 1}")
        else:
            # Update EMA: ema = decay * ema + (1 - decay) * current
            self.ema_weights = [
                self.decay * ema + (1 - self.decay) * curr
                for ema, curr in zip(self.ema_weights, current_weights)
            ]
        
        if (
            self.save_best_only
            and self.save_ema_weights
            and self.monitor
            and self.ema_weights is not None
        ):
            metric = logs.get(self.monitor) if logs else None
            if metric is None:
                if not self._missing_monitor_warned:
                    print(
                        f"⚠️  EMACallback: monitor '{self.monitor}' "
                        "not found in logs; skipping best-EMA save."
                    )
                    self._missing_monitor_warned = True
            else:
                is_better = (
                    metric > self.best if self.mode == "max" else metric < self.best
                )
                if is_better:
                    self.best = metric
                    self._best_saved = True
                    self._save_ema_weights(best_snapshot=True)
            
    def on_train_end(self, logs=None):
        if (
            self.save_ema_weights
            and self.ema_weights
            and self.checkpoint_dir
            and not self._best_saved
        ):
            self._save_ema_weights()
    
    def _save_ema_weights(self, best_snapshot: bool = False):
        if not self.checkpoint_dir:
            return
        ema_path = self.checkpoint_dir / "weights_ema.weights.h5"
        current_weights = self.model.get_weights()
        self.model.set_weights(self.ema_weights)
        self.model.save_weights(str(ema_path))
        self.model.set_weights(current_weights)
        print(f"\n✓ Saved EMA weights to: {ema_path}")
        if best_snapshot:
            print("  (best EMA snapshot)")
        else:
            print("  Use these weights for inference (typically +0.5-1% Dice improvement)")


# ---- Data Pipeline (from v2) ----------------------------------------------

class TileDataset:
    """Efficient data loading with caching and prefetching"""
    
    def __init__(self, images_dir, masks_dir, batch_size, augment=True, cache_size=100, mean=None, std=None, 
                 normalization_method='zscore', percentile_low=1.0, percentile_high=99.0, augment_fn=None, augment_level: str = 'moderate'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.augment_fn = augment_fn
        self.augment_level = augment_level
        self.cache_size = cache_size
        self.mean = mean
        self.std = std
        self.normalization_method = normalization_method
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        print("📊 Using pre-processed dataset (stain normalization applied during dataset building)")
        print(f"📊 Augmentation: {('off' if not self.augment or self.augment_fn is None else augment_level)}")
        print(f"📊 Intensity normalization: {normalization_method.upper()}")
        if normalization_method == 'percentile':
            print(f"   Percentile range: {percentile_low}-{percentile_high}")
        elif normalization_method == 'zscore':
            print(f"   Dataset stats: mean={mean:.2f}, std={std:.2f}")
        
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        self.mask_files = {p.stem: p for p in self.masks_dir.glob("*.tif")}
        
        self.pairs = []
        for img_path in self.image_files:
            if img_path.stem in self.mask_files:
                self.pairs.append((img_path, self.mask_files[img_path.stem]))
        
        print(f"Found {len(self.pairs)} paired tiles in {images_dir.name}")
        
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    def load_pair(self, img_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache a single image-mask pair"""
        cache_key = img_path.stem
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = tiff.imread(str(mask_path)).astype(np.float32)
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = (img.copy(), mask.copy())
        
        return img, mask
    
    def __len__(self):
        return len(self.pairs)
    
    def generator(self):
        """Python generator for tf.data.Dataset"""
        rng = np.random.RandomState()
        indices = np.arange(len(self.pairs))
        
        while True:
            rng.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                images = []
                masks = []
                
                for idx in batch_indices:
                    img_path, mask_path = self.pairs[idx]
                    img, mask = self.load_pair(img_path, mask_path)
                    
                    if self.augment and self.augment_fn is not None:
                        img, mask = self.augment_fn(img, mask, rng)
                    
                    if self.normalization_method == 'zscore':
                        img = (img - self.mean) / (self.std + 1e-10)
                    elif self.normalization_method == 'percentile':
                        img = normalize_image(img, method='percentile', 
                                            p_low=self.percentile_low, p_high=self.percentile_high)
                    else:
                        raise ValueError(f"Unknown normalization method: {self.normalization_method}")
                    
                    images.append(img)
                    masks.append(mask)
                
                while len(images) < self.batch_size:
                    images.append(images[-1])
                    masks.append(masks[-1])
                
                yield (
                    np.array(images, dtype=np.float32),
                    np.array(masks, dtype=np.float32)
                )
    
    def create_dataset(self):
        """Create tf.data.Dataset with prefetching"""
        output_signature = (
            tf.TensorSpec(shape=(self.batch_size, 1024, 1024), dtype=tf.float32),
            tf.TensorSpec(shape=(self.batch_size, 1024, 1024), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ---- Model V3 with Deep Supervision ------------------

class AdiposeUNetV3:
    """
    NEW in v3: U-Net with optional deep supervision.
    
    Deep supervision adds auxiliary outputs at intermediate decoder stages,
    providing better gradient flow and multi-scale learning.
    """
    
    def __init__(self, checkpoint_name: str, freeze_encoder: bool = True, 
                 build_timestamp: str = None, use_deep_supervision: bool = True):
        self.checkpoint_name = checkpoint_name
        self.freeze_encoder = freeze_encoder
        self.use_deep_supervision = use_deep_supervision
        self.net: Model | None = None
        
        if build_timestamp:
            timestamp = build_timestamp
            print(f"📅 Using build timestamp: {timestamp}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"📅 Generated new timestamp: {timestamp}")
        
        timestamped_name = f"{timestamp}_{checkpoint_name}_1024_finetune_v3"
        self.checkpoint_dir = Path(f"checkpoints/segmentation/{timestamped_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
        if use_deep_supervision:
            print(f"🔬 Deep supervision: ENABLED")
        else:
            print(f"🔬 Deep supervision: DISABLED (v2-compatible mode)")
    
    def build_model(self, init_nb: int = 44, dropout_rate: float = 0.3):
        """Build U-Net model with optional deep supervision"""
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
        
        if self.use_deep_supervision:
            # Deep supervision: auxiliary outputs at up3 and up2
            # Auxiliary output 1 (from up3, 128x128 -> upsampled to 1024x1024)
            aux_out1 = Conv2D(1, 1, activation='sigmoid', name='aux_out1')(up3)
            aux_out1_upsampled = Lambda(
                lambda x: tf.image.resize(x, [1024, 1024], method='bilinear'),
                name='aux_out1_upsample'
            )(aux_out1)
            
            # Auxiliary output 2 (from up2, 512x512 -> upsampled to 1024x1024)
            aux_out2 = Conv2D(1, 1, activation='sigmoid', name='aux_out2')(up2)
            aux_out2_upsampled = Lambda(
                lambda x: tf.image.resize(x, [1024, 1024], method='bilinear'),
                name='aux_out2_upsample'
            )(aux_out2)
            
            # Main output (softmax as in v2)
            main_out = Conv2D(2, 1, activation='softmax', name='output_softmax')(up1)
            main_out = Lambda(lambda z: z[:, :, :, 1:2], output_shape=(1024, 1024, 1), name='output_class1')(main_out)
            main_out = Lambda(lambda z: K.squeeze(z, axis=-1), output_shape=(1024, 1024), name='main_out')(main_out)
            
            # Squeeze auxiliary outputs
            aux_out1_final = Lambda(lambda z: K.squeeze(z, axis=-1), output_shape=(1024, 1024), name='aux1_squeeze')(aux_out1_upsampled)
            aux_out2_final = Lambda(lambda z: K.squeeze(z, axis=-1), output_shape=(1024, 1024), name='aux2_squeeze')(aux_out2_upsampled)
            
            # Multi-output model
            self.net = Model(
                inputs=inputs,
                outputs={
                    'main_out': main_out,
                    'aux_out1': aux_out1_final,
                    'aux_out2': aux_out2_final
                }
            )
        else:
            # Standard output (v2-compatible)
            x = Conv2D(2, 1, activation='softmax', name='output_softmax')(up1)
            x = Lambda(lambda z: z[:, :, :, 1:2], output_shape=(1024, 1024, 1), name='output_class1')(x)
            x = Lambda(lambda z: K.squeeze(z, axis=-1), output_shape=(1024, 1024), name='squeeze')(x)
            
            self.net = Model(inputs=inputs, outputs=x)
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            self.freeze_encoder_layers()
        
        return self.net
    
    def freeze_encoder_layers(self):
        """Freeze encoder for transfer learning"""
        frozen_layers = [
            'down1_conv1', 'down1_conv2', 'down1_pool',
            'down2_conv1', 'down2_conv2', 'down2_pool',
            'down3_conv1', 'down3_conv2', 'down3_pool',
        ]
        
        for layer in self.net.layers:
            if layer.name in frozen_layers:
                layer.trainable = False
        
        print(f"Frozen {len(frozen_layers)} encoder layers for transfer learning")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for layer in self.net.layers:
            layer.trainable = True
        print("Unfrozen all layers for fine-tuning")
    
    def compile_model(self, lr: float = 1e-4, use_hard_mining: bool = True, 
                     hard_example_ratio: float = 0.7, optimizer_type: str = 'adam',
                     use_label_smoothing: bool = False, epsilon_pos: float = 0.03,
                     epsilon_neg: float = 0.07, ds_weight_main: float = 1.0,
                     ds_weight_aux1: float = 0.4, ds_weight_aux2: float = 0.3):
        """
        Compile model with chosen optimizer and loss.
        
        Args:
            lr: Learning rate
            use_hard_mining: Use online hard example mining loss
            hard_example_ratio: Ratio of hard pixels to keep (0.5-0.9)
            optimizer_type: 'adam' or 'adamw'
            use_label_smoothing: Enable asymmetric label smoothing (v3.1)
            epsilon_pos: Smoothing for positive class (0.02-0.05)
            epsilon_neg: Smoothing for negative class (0.05-0.10)
            ds_weight_main: Loss weight for main output (default: 1.0)
            ds_weight_aux1: Loss weight for aux_out1 (default: 0.4)
            ds_weight_aux2: Loss weight for aux_out2 (default: 0.3)
        """
        # Select optimizer
        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(learning_rate=lr, weight_decay=0.01)
            print(f"Using AdamW optimizer (lr={lr}, weight_decay=0.01)")
        else:
            optimizer = Adam(learning_rate=lr)
            print(f"Using Adam optimizer (lr={lr})")
        
        if self.use_deep_supervision:
            # Multi-output loss with deep supervision
            if use_label_smoothing and use_hard_mining:
                # Hard mining + label smoothing (v3.1)
                loss_fn_main = lambda y_true, y_pred: online_hard_example_mining_loss_with_smoothing(
                    y_true, y_pred, hard_example_ratio, epsilon_pos, epsilon_neg
                )
                loss_fn_aux = lambda y_true, y_pred: combined_loss_with_label_smoothing(
                    y_true, y_pred, epsilon_pos, epsilon_neg
                )
                print(f"Loss: Hard mining (ratio={hard_example_ratio}) + Label smoothing (ε_pos={epsilon_pos}, ε_neg={epsilon_neg}) + Deep supervision")
            elif use_label_smoothing:
                # Label smoothing only
                loss_fn_main = lambda y_true, y_pred: combined_loss_with_label_smoothing(
                    y_true, y_pred, epsilon_pos, epsilon_neg
                )
                loss_fn_aux = lambda y_true, y_pred: combined_loss_with_label_smoothing(
                    y_true, y_pred, epsilon_pos, epsilon_neg
                )
                print(f"Loss: Label smoothing (ε_pos={epsilon_pos}, ε_neg={epsilon_neg}) + Deep supervision")
            elif use_hard_mining:
                # Hard mining only (default v3)
                loss_fn_main = lambda y_true, y_pred: online_hard_example_mining_loss(y_true, y_pred, hard_example_ratio)
                loss_fn_aux = combined_loss_standard
                print(f"Loss: Hard mining (ratio={hard_example_ratio}) + Deep supervision")
            else:
                # Standard (no advanced techniques)
                loss_fn_main = combined_loss_standard
                loss_fn_aux = combined_loss_standard
                print(f"Loss: BCE + Dice + Deep supervision")
            
            print(f"Deep supervision weights: main={ds_weight_main}, aux1={ds_weight_aux1}, aux2={ds_weight_aux2}")
            self.net.compile(
                optimizer=optimizer,
                loss={
                    'main_out': loss_fn_main,
                    'aux_out1': loss_fn_aux,
                    'aux_out2': loss_fn_aux
                },
                loss_weights={
                    'main_out': ds_weight_main,
                    'aux_out1': ds_weight_aux1,
                    'aux_out2': ds_weight_aux2
                },
                metrics={
                    'main_out': [dice_coef, 'binary_accuracy']
                }
            )
        else:
            # Single output (v2-compatible)
            if use_label_smoothing and use_hard_mining:
                loss_fn = lambda y_true, y_pred: online_hard_example_mining_loss_with_smoothing(
                    y_true, y_pred, hard_example_ratio, epsilon_pos, epsilon_neg
                )
                print(f"Loss: Hard mining (ratio={hard_example_ratio}) + Label smoothing (ε_pos={epsilon_pos}, ε_neg={epsilon_neg})")
            elif use_label_smoothing:
                loss_fn = lambda y_true, y_pred: combined_loss_with_label_smoothing(
                    y_true, y_pred, epsilon_pos, epsilon_neg
                )
                print(f"Loss: Label smoothing (ε_pos={epsilon_pos}, ε_neg={epsilon_neg})")
            elif use_hard_mining:
                loss_fn = lambda y_true, y_pred: online_hard_example_mining_loss(y_true, y_pred, hard_example_ratio)
                print(f"Loss: Hard mining (ratio={hard_example_ratio})")
            else:
                loss_fn = combined_loss_standard
                print(f"Loss: BCE + Dice (standard)")
            
            self.net.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=[dice_coef, 'binary_accuracy']
            )
    
    def load_pretrained_weights(self, h5_path: str):
        """
        Smart weight loading from v2 models.
        
        If using deep supervision, loads by name and skips auxiliary heads.
        Otherwise, loads all weights strictly.
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format
        
        with h5py.File(h5_path, 'r') as f:
            group = f['model_weights'] if 'model_weights' in f else f
            
            if self.use_deep_supervision:
                # Load by name, skip auxiliary heads (they're new)
                try:
                    hdf5_format.load_weights_from_hdf5_group_by_name(
                        group, self.net.layers, skip_mismatch=True
                    )
                    print(f"✓ Loaded pretrained weights from {h5_path} (by name, skipped aux heads)")
                    print(f"  Encoder: ✓ loaded")
                    print(f"  Decoder: ✓ loaded")
                    print(f"  Main output: ✓ loaded")
                    print(f"  Aux outputs: ⚡ random init (will train quickly)")
                except Exception as e:
                    print(f"Warning: Partial weight loading failed: {e}")
                    print(f"Training from scratch!")
            else:
                # Standard loading (v2-compatible)
                try:
                    hdf5_format.load_weights_from_hdf5_group(group, self.net.layers)
                    print(f"✓ Loaded weights from {h5_path} (strict topology match)")
                except Exception as e:
                    print(f"Strict load failed: {e}")
                    hdf5_format.load_weights_from_hdf5_group_by_name(group, self.net.layers)
                    print(f"✓ Loaded weights from {h5_path} by layer name (skipped mismatches)")
    
    def save_weights_modern(self, suffix: str = "finetuned"):
        """Save weights in modern TF2 format"""
        weights_path = self.checkpoint_dir / f"weights_{suffix}.weights.h5"
        self.net.save_weights(str(weights_path))
        print(f"✓ Saved modern weights to {weights_path}")


# ---- Training Settings Logging (from v2) --------------------------------------------

def capture_system_info() -> Dict:
    """Capture comprehensive system and environment information."""
    system_info = {}
    
    system_info["platform"] = platform.platform()
    system_info["system"] = platform.system()
    system_info["machine"] = platform.machine()
    system_info["processor"] = platform.processor()
    system_info["python_version"] = platform.python_version()
    system_info["tensorflow_version"] = tf.__version__
    
    try:
        system_info["keras_version"] = tf.__version__
    except:
        system_info["keras_version"] = "integrated with TensorFlow"
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = []
            for i, gpu in enumerate(gpus):
                gpu_details = {
                    "device": str(gpu),
                    "name": gpu.name,
                    "device_type": gpu.device_type
                }
                gpu_info.append(gpu_details)
            system_info["gpus"] = gpu_info
        else:
            system_info["gpus"] = []
    except Exception as e:
        system_info["gpus"] = f"Error getting GPU info: {str(e)}"
    
    try:
        system_info["cpu_count"] = os.cpu_count()
    except:
        system_info["cpu_count"] = "Unknown"
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["memory_total_gb"] = round(memory.total / (1024**3), 2)
        system_info["memory_available_gb"] = round(memory.available / (1024**3), 2)
    except:
        system_info["memory_info"] = "psutil not available"
    
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.').decode().strip()
        system_info["git_commit"] = git_commit
        git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd='.').decode().strip()
        system_info["git_dirty"] = len(git_status) > 0
    except:
        system_info["git_info"] = "Git repository information not available"
    
    return system_info


def log_training_settings(checkpoint_dir: Path, command_line_args: Dict,
                          data_config: Dict, model_config: Dict, training_config: Dict) -> None:
    """Create comprehensive training settings log file."""
    settings_log_path = checkpoint_dir / "training_settings.log"
    system_info = capture_system_info()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    settings = {
        "training_session": {
            "timestamp": timestamp,
            "script_name": "train_adipose_unet_3.py",
            "working_directory": str(Path.cwd()),
            "checkpoint_directory": str(checkpoint_dir),
            "version": "3.0 (deep supervision + hard mining + EMA)"
        },
        "command_line": command_line_args,
        "system_environment": system_info,
        "data_configuration": data_config,
        "model_architecture": model_config,
        "training_parameters": training_config
    }
    
    with open(settings_log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SETTINGS LOG - VERSION 3\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Script: train_adipose_unet_3.py\n")
        f.write(f"Checkpoint Directory: {checkpoint_dir}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("COMMAND LINE ARGUMENTS\n")
        f.write("-" * 60 + "\n")
        for key, value in command_line_args.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("DATA CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        for key, value in data_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 60 + "\n")
        for key, value in model_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 60 + "\n")
        for key, value in training_config.items():
            if isinstance(value, dict):
                f.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"    {sub_key}: {sub_value}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("MACHINE READABLE FORMAT (JSON)\n")
        f.write("-" * 60 + "\n")
        f.write(json.dumps(settings, indent=2, default=str))
        f.write("\n")
    
    print(f"✓ Saved training settings log to: {settings_log_path}")


def _select_augment_fn(level: str):
    """Select augmentation function based on level"""
    lvl = (level or 'moderate').lower()
    if lvl == 'light':
        return augment_pair_light, 'light'
    if lvl == 'heavy':
        return augment_pair_heavy, 'heavy'
    if lvl == 'tta-style' or lvl == 'tta_style':
        return augment_pair_tta_style, 'tta-style'
    if lvl in ('none', 'off', 'disable'):
        return None, 'none'
    return augment_pair_moderate, 'moderate'


# ---- Training Function -------------------------------------

def train_model(
    data_root: Path,
    pretrained_weights: str,
    batch_size: int = 2,
    epochs_phase1: int = 75,
    epochs_phase2: int = 150,
    normalization_method: str = 'percentile',
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    build_timestamp: str = None,
    augmentation_level: str = 'moderate',
    checkpoint_suffix: str = '',
    use_deep_supervision: bool = True,
    use_hard_mining: bool = True,
    hard_example_ratio: float = 0.7,
    ema_decay: float = 0.995,
    optimizer_type: str = 'adam',
    use_label_smoothing: bool = False,
    epsilon_pos: float = 0.03,
    epsilon_neg: float = 0.07,
    use_cosine_schedule: bool = True,
    warmup_epochs_phase1: int = 5,
    warmup_epochs_phase2: int = 3,
    ds_weight_main: float = 1.0,
    ds_weight_aux1: float = 0.4,
    ds_weight_aux2: float = 0.3,
):
    """
    V3 training with deep supervision, hard mining, and EMA.
    V3.1 adds label smoothing for conservative annotations.
    """
    
    print("="*80)
    print("TRAIN ADIPOSE U-NET VERSION 3.1")
    print("="*80)
    print("✓ Deep supervision: ", "ENABLED" if use_deep_supervision else "DISABLED")
    print("✓ Hard example mining:", "ENABLED" if use_hard_mining else "DISABLED")
    print(f"✓ EMA weights (decay={ema_decay})")
    print(f"✓ Optimizer: {optimizer_type.upper()}")
    if use_label_smoothing:
        print(f"✓ Label smoothing: ENABLED (ε_pos={epsilon_pos}, ε_neg={epsilon_neg})")
    else:
        print("✓ Label smoothing: DISABLED")
    print("="*80 + "\n")
    
    # Setup data
    data_root = Path(data_root)
    train_images = data_root / "dataset" / "train" / "images"
    train_masks = data_root / "dataset" / "train" / "masks"
    val_images = data_root / "dataset" / "val" / "images"
    val_masks = data_root / "dataset" / "val" / "masks"
    
    # Compute normalization stats
    def compute_mean_std(image_paths, max_n=None):
        vals = []
        for i, p in enumerate(image_paths):
            if max_n and i >= max_n:
                break
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            vals.append(img.reshape(-1))
        vals = np.concatenate(vals)
        return float(vals.mean()), float(vals.std() + 1e-10)

    train_image_paths = sorted(train_images.glob("*.jpg"))
    train_mean, train_std = compute_mean_std(train_image_paths)
    print(f"Global normalization stats: mean={train_mean:.2f}, std={train_std:.2f}")

    augment_fn, augment_label = _select_augment_fn(augmentation_level)
    use_augment = augment_label != 'none'

    # Build datasets
    train_dataset = TileDataset(train_images, train_masks, batch_size,
                                augment=use_augment, mean=train_mean, std=train_std,
                                normalization_method=normalization_method,
                                percentile_low=percentile_low, percentile_high=percentile_high,
                                augment_fn=augment_fn, augment_level=augment_label)
    val_dataset   = TileDataset(val_images, val_masks, batch_size,
                                augment=False, mean=train_mean, std=train_std,
                                normalization_method=normalization_method,
                                percentile_low=percentile_low, percentile_high=percentile_high,
                                augment_fn=None, augment_level='none')
    
    train_ds = train_dataset.create_dataset()
    val_ds = val_dataset.create_dataset()
    
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    validation_steps = max(1, math.ceil(len(val_dataset) / batch_size))
    
    print(f"\n{'='*60}")
    print(f"Dataset Configuration:")
    print(f"{'='*60}")
    print(f"  Training:   {len(train_dataset)} tiles ({steps_per_epoch} steps/epoch)")
    print(f"  Validation: {len(val_dataset)} tiles ({validation_steps} steps/epoch)")
    print(f"  Augmentation: {augment_label}")
    print(f"  Normalization: {normalization_method.upper()}")
    print(f"{'='*60}\n")
    
    # Build model
    checkpoint_base = "adipose_sybreosin"
    if checkpoint_suffix:
        checkpoint_name = f"{checkpoint_base}_{checkpoint_suffix}"
    else:
        checkpoint_name = checkpoint_base
    
    model = AdiposeUNetV3(checkpoint_name, freeze_encoder=True, 
                         build_timestamp=build_timestamp,
                         use_deep_supervision=use_deep_supervision)
    model.build_model(init_nb=44, dropout_rate=0.3)
    model.compile_model(lr=1e-4, use_hard_mining=use_hard_mining,
                       hard_example_ratio=hard_example_ratio,
                       optimizer_type=optimizer_type,
                       use_label_smoothing=use_label_smoothing,
                       epsilon_pos=epsilon_pos,
                       epsilon_neg=epsilon_neg,
                       ds_weight_main=ds_weight_main,
                       ds_weight_aux1=ds_weight_aux1,
                       ds_weight_aux2=ds_weight_aux2)    # Load pretrained weights
    if pretrained_weights and Path(pretrained_weights).exists():
        model.load_pretrained_weights(pretrained_weights)
    else:
        print("WARNING: No pretrained weights found, training from scratch")
    
    # Save normalization stats
    normalization_stats = {
        "mean": float(train_mean),
        "std": float(train_std),
        "normalization_method": normalization_method,
        "dataset_path": str(data_root),
        "num_training_images": len(train_image_paths),
        "build_timestamp": build_timestamp,
        "version": "3.0"
    }
    
    norm_stats_path = model.checkpoint_dir / "normalization_stats.json"
    with open(norm_stats_path, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    # Log settings
    command_line_args = {
        "data_root": str(data_root),
        "pretrained_weights": pretrained_weights,
        "batch_size": batch_size,
        "use_deep_supervision": use_deep_supervision,
        "use_hard_mining": use_hard_mining,
        "hard_example_ratio": hard_example_ratio,
        "ema_decay": ema_decay,
        "optimizer": optimizer_type
    }
    
    data_config = {
        "train_images_count": len(train_dataset),
        "validation_images_count": len(val_dataset),
        "normalization_mean": train_mean,
        "normalization_std": train_std,
        "augmentation_type": augment_label
    }
    
    trainable_params = int(np.sum([K.count_params(w) for w in model.net.trainable_weights]))
    non_trainable_params = int(np.sum([K.count_params(w) for w in model.net.non_trainable_weights]))
    
    model_config = {
        "architecture": "U-Net V3 with deep supervision",
        "deep_supervision": use_deep_supervision,
        "total_parameters": model.net.count_params(),
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }
    
    training_config = {
        "total_epochs": epochs_phase1 + epochs_phase2,
        "batch_size": batch_size,
        "hard_mining": use_hard_mining,
        "ema_decay": ema_decay,
        "optimizer": optimizer_type
    }
    
    log_training_settings(model.checkpoint_dir, command_line_args, 
                         data_config, model_config, training_config)
    
    model.net.summary()
    
    # Phase 1: Frozen Encoder
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training decoder only ({epochs_phase1} epochs)")
    print(f"{'='*60}")
    print(f"EMA decay: 0.999 (slow/smooth - frozen encoder learning)")
    print(f"{'='*60}\n")
    
    # Prepare for multi-output if needed
    if use_deep_supervision:
        train_ds_phase1 = train_ds.map(lambda x, y: (x, {'main_out': y, 'aux_out1': y, 'aux_out2': y}))
        val_ds_phase1 = val_ds.map(lambda x, y: (x, {'main_out': y, 'aux_out1': y, 'aux_out2': y}))
    else:
        train_ds_phase1 = train_ds
        val_ds_phase1 = val_ds
    monitor_metric = 'val_main_out_dice_coef' if use_deep_supervision else 'val_dice_coef'
    
    # Build Phase 1 callbacks
    callbacks_phase1 = [
        ModelCheckpoint(
            filepath=str(model.checkpoint_dir / "phase1_best.weights.h5"),
            monitor=monitor_metric,
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode='max',
            patience=15,
            restore_best_weights=False,
            verbose=1
        ),
        # Phase 1: Slower decay (0.999) for smoother averaging during frozen encoder training
        EMACallback(decay=0.999, save_ema_weights=False, checkpoint_dir=model.checkpoint_dir),
        CSVLogger(str(model.checkpoint_dir / "phase1_training.log"))
    ]
    
    # Add learning rate schedule (Tier 1 optimization)
    if use_cosine_schedule:
        callbacks_phase1.append(
            CosineAnnealingWithWarmup(
                max_lr=1e-4,
                min_lr=1e-7,
                warmup_epochs=warmup_epochs_phase1,
                total_epochs=epochs_phase1,
                verbose=1
            )
        )
        print(f"📈 Using Cosine Annealing with {warmup_epochs_phase1}-epoch warmup")
    else:
        callbacks_phase1.append(
            ReduceLROnPlateau(
                monitor=monitor_metric,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
        print(f"📉 Using ReduceLROnPlateau (legacy mode)")
    
    model.net.fit(
        train_ds_phase1,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_phase1,
        validation_data=val_ds_phase1,
        validation_steps=validation_steps,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    model.save_weights_modern("phase1_final")
    
    # Phase 2: Full Fine-tuning
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning all layers ({epochs_phase2} epochs)")
    print(f"{'='*60}")
    print(f"EMA decay: {ema_decay} (fast/responsive - full network learning)")
    print(f"EMA saves best validation snapshot only")
    print(f"{'='*60}\n")
    
    best_phase1_path = model.checkpoint_dir / "phase1_best.weights.h5"
    if best_phase1_path.exists():
        model.net.load_weights(str(best_phase1_path))
        print("✓ Loaded best Phase 1 weights\n")
    
    model.unfreeze_encoder()
    model.compile_model(lr=1e-5, use_hard_mining=use_hard_mining,
                       hard_example_ratio=hard_example_ratio,
                       optimizer_type=optimizer_type,
                       use_label_smoothing=use_label_smoothing,
                       epsilon_pos=epsilon_pos,
                       epsilon_neg=epsilon_neg,
                       ds_weight_main=ds_weight_main,
                       ds_weight_aux1=ds_weight_aux1,
                       ds_weight_aux2=ds_weight_aux2)
    
    if use_deep_supervision:
        train_ds_phase2 = train_ds.map(lambda x, y: (x, {'main_out': y, 'aux_out1': y, 'aux_out2': y}))
        val_ds_phase2 = val_ds.map(lambda x, y: (x, {'main_out': y, 'aux_out1': y, 'aux_out2': y}))
    else:
        train_ds_phase2 = train_ds
        val_ds_phase2 = val_ds
    
    # Build Phase 2 callbacks
    callbacks_phase2 = [
        ModelCheckpoint(
            filepath=str(model.checkpoint_dir / "phase2_best.weights.h5"),
            monitor=monitor_metric,
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode='max',
            patience=15,
            restore_best_weights=False,
            verbose=1
        ),
        # Phase 2: Faster decay (default 0.995) for responsive tracking during full fine-tuning
        EMACallback(
            decay=ema_decay,  # Uses --ema-decay flag (default: 0.995)
            save_ema_weights=True,
            checkpoint_dir=model.checkpoint_dir,
            monitor=monitor_metric,
            mode='max',
            save_best_only=True,
        ),
        CSVLogger(str(model.checkpoint_dir / "phase2_training.log"))
    ]
    
    # Add learning rate schedule (Tier 1 optimization)
    if use_cosine_schedule:
        callbacks_phase2.append(
            CosineAnnealingWithWarmup(
                max_lr=1e-5,
                min_lr=1e-8,
                warmup_epochs=warmup_epochs_phase2,
                total_epochs=epochs_phase2,
                verbose=1
            )
        )
        print(f"📈 Using Cosine Annealing with {warmup_epochs_phase2}-epoch warmup")
    else:
        callbacks_phase2.append(
            ReduceLROnPlateau(
                monitor=monitor_metric,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-8,
                verbose=1
            )
        )
        print(f"📉 Using ReduceLROnPlateau (legacy mode)")
    
    model.net.fit(
        train_ds_phase2,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_phase2,
        validation_data=val_ds_phase2,
        validation_steps=validation_steps,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    model.save_weights_modern("phase2_final")
    
    best_phase2_path = model.checkpoint_dir / "phase2_best.weights.h5"
    if best_phase2_path.exists():
        model.net.load_weights(str(best_phase2_path))
        model.save_weights_modern("best_overall")
    
    print(f"\n{'='*60}")
    print("✓ Training Complete!")
    print(f"{'='*60}")
    print(f"Checkpoint directory: {model.checkpoint_dir}")
    print(f"\nBest models saved:")
    print(f"  - phase1_best.weights.h5")
    print(f"  - phase2_best.weights.h5")
    print(f"  - weights_best_overall.weights.h5")
    print(f"  - weights_ema.weights.h5 (EMA - use this for inference!)")
    print(f"{'='*60}\n")
    
    return model


# ---- CLI ------------------------------------------------------------------

def main():
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Train U-Net V3 for adipose segmentation (deep supervision + hard mining + EMA)"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane',
        help='Base data directory or specific build directory'
    )
    parser.add_argument(
        '--pretrained-weights',
        type=str,
        default='checkpoints/unet_1024_dilation/weights_loss_val.weights.h5',
        help='Path to pretrained weights (v2 compatible)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size (2-4 recommended)'
    )
    parser.add_argument(
        '--epochs-phase1',
        type=int,
        default=75,
        help='Epochs for phase 1 (frozen encoder, +20 for aux heads)'
    )
    parser.add_argument(
        '--epochs-phase2',
        type=int,
        default=150,
        help='Epochs for phase 2 (full fine-tuning, +30 recommended)'
    )
    parser.add_argument(
        '--normalization-method',
        type=str,
        default='percentile',
        choices=['zscore', 'percentile'],
        help='Intensity normalization method (percentile default for robustness)'
    )
    parser.add_argument(
        '--percentile-low',
        type=float,
        default=1.0,
        help='Lower percentile for percentile normalization'
    )
    parser.add_argument(
        '--percentile-high',
        type=float,
        default=99.0,
        help='Upper percentile for percentile normalization'
    )
    parser.add_argument(
        '--augmentation-level',
        type=str,
        default='moderate',
        choices=['none', 'light', 'moderate', 'heavy', 'tta-style'],
        help='Augmentation level (moderate default, tta-style for advanced training)'
    )
    parser.add_argument(
        '--checkpoint-suffix',
        type=str,
        default='',
        help='Optional suffix for checkpoint folder name'
    )
    parser.add_argument(
        '--use-deep-supervision',
        action='store_true',
        default=True,
        help='Enable deep supervision (default: True)'
    )
    parser.add_argument(
        '--no-deep-supervision',
        action='store_false',
        dest='use_deep_supervision',
        help='Disable deep supervision (v2-compatible mode)'
    )
    parser.add_argument(
        '--use-hard-mining',
        action='store_true',
        default=True,
        help='Enable online hard example mining (default: True)'
    )
    parser.add_argument(
        '--no-hard-mining',
        action='store_false',
        dest='use_hard_mining',
        help='Disable hard example mining'
    )
    parser.add_argument(
        '--hard-example-ratio',
        type=float,
        default=0.7,
        help='Ratio of hard pixels to keep (0.5-0.9, default: 0.7)'
    )
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=0.995,
        help='EMA decay rate (default: 0.995 for faster tracking)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw'],
        help='Optimizer: adam (proven) or adamw (for testing)'
    )
    parser.add_argument(
        '--label-smoothing',
        action='store_true',
        default=False,
        help='Enable asymmetric label smoothing (v3.1, addresses overconfidence)'
    )
    parser.add_argument(
        '--no-label-smoothing',
        action='store_false',
        dest='label_smoothing',
        help='Disable label smoothing (recommended for clean annotations)'
    )
    parser.add_argument(
        '--label-smooth-epsilon-pos',
        type=float,
        default=0.03,
        help='Label smoothing for adipose class (0.02-0.05, default: 0.03)'
    )
    parser.add_argument(
        '--label-smooth-epsilon-neg',
        type=float,
        default=0.07,
        help='Label smoothing for background class (0.05-0.10, default: 0.07)'
    )
    
    # Tier 1 Optimizations
    parser.add_argument(
        '--use-cosine-schedule',
        action='store_true',
        default=True,
        help='Use cosine annealing LR schedule with warmup (default: True, +1-3%% Dice)'
    )
    parser.add_argument(
        '--no-cosine-schedule',
        action='store_false',
        dest='use_cosine_schedule',
        help='Disable cosine schedule, use ReduceLROnPlateau instead'
    )
    parser.add_argument(
        '--warmup-epochs-phase1',
        type=int,
        default=5,
        help='Warmup epochs for Phase 1 (default: 5)'
    )
    parser.add_argument(
        '--warmup-epochs-phase2',
        type=int,
        default=3,
        help='Warmup epochs for Phase 2 (default: 3)'
    )
    parser.add_argument(
        '--ds-weight-main',
        type=float,
        default=1.0,
        help='Deep supervision weight for main output (default: 1.0)'
    )
    parser.add_argument(
        '--ds-weight-aux1',
        type=float,
        default=0.4,
        help='Deep supervision weight for aux_out1 (default: 0.4)'
    )
    parser.add_argument(
        '--ds-weight-aux2',
        type=float,
        default=0.3,
        help='Deep supervision weight for aux_out2 (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    try:
        data_root, build_timestamp = resolve_data_root(args.data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    model = train_model(
        data_root=data_root,
        pretrained_weights=args.pretrained_weights,
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        normalization_method=args.normalization_method,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        build_timestamp=build_timestamp,
        augmentation_level=args.augmentation_level,
        checkpoint_suffix=args.checkpoint_suffix,
        use_deep_supervision=args.use_deep_supervision,
        use_hard_mining=args.use_hard_mining,
        hard_example_ratio=args.hard_example_ratio,
        ema_decay=args.ema_decay,
        optimizer_type=args.optimizer,
        use_label_smoothing=args.label_smoothing,
        epsilon_pos=args.label_smooth_epsilon_pos,
        epsilon_neg=args.label_smooth_epsilon_neg,
        use_cosine_schedule=args.use_cosine_schedule,
        warmup_epochs_phase1=args.warmup_epochs_phase1,
        warmup_epochs_phase2=args.warmup_epochs_phase2,
        ds_weight_main=args.ds_weight_main,
        ds_weight_aux1=args.ds_weight_aux1,
        ds_weight_aux2=args.ds_weight_aux2,
    )
    
    print("\n✓ Training complete!")
    print(f"  Best regular: {model.checkpoint_dir}/weights_best_overall.weights.h5")
    print(f"  Best EMA:     {model.checkpoint_dir}/weights_ema.weights.h5 ⭐ USE THIS!")


if __name__ == "__main__":
    main()
