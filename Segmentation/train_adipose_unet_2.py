"""
Optimized U-Net Training for Sybr Gold + Eosin Adipose Segmentation
TensorFlow 2.13 / Python 3.10

HYBRID VERSION: Combines established architecture with improvements
- Uses original softmax output 
- Uses z-score normalization 
- Uses original loss weights 
- Adds improved callbacks and checkpoint management
- Uses moderate augmentation from data.py
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

# Configure seeding WITHOUT forcing GPU determinism (ResizeNearestNeighborGrad lacks deterministic kernel)
os.environ['PYTHONHASHSEED'] = str(865)
os.environ.pop('TF_DETERMINISTIC_OPS', None)

import numpy as np
import tensorflow as tf

# Load seed from seed.csv for reproducibility
from src.utils.seed_utils import get_project_seed
GLOBAL_SEED = get_project_seed()
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,
    Concatenate, Lambda, Reshape, Add,
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.losses import binary_crossentropy

import cv2
import tifffile as tiff
import platform
import subprocess

# Local utils
sys.path.append('.')
from src.utils.runtime import funcname
from src.utils.model import dice_coef
from src.utils.data import (
    augment_pair_light, augment_pair_moderate, augment_pair_heavy,
    augment_pair_tta_style, normalize_image
)


# ---- Build Directory Management -------------------------------------------

def find_most_recent_build_dir(base_data_root: Path) -> Path:
    """
    Find the most recent _build_* directory or fall back to _build.
    
    Args:
        base_data_root: The root data directory (e.g., /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane)
        
    Returns:
        Path to the most recent build directory
    """
    base_data_root = Path(base_data_root).expanduser()
    
    # Look for timestamped build directories
    build_pattern = str(base_data_root / "_build_*")
    timestamped_builds = glob.glob(build_pattern)
    
    if timestamped_builds:
        # Extract timestamps and find the most recent
        build_dirs = []
        for build_path in timestamped_builds:
            build_dir = Path(build_path)
            # Extract timestamp from directory name like _build_20241031_124305
            match = re.search(r'_build_(\d{8}_\d{6})$', build_dir.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Parse timestamp: YYYYMMDD_HHMMSS
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    build_dirs.append((timestamp, build_dir))
                except ValueError:
                    continue
        
        if build_dirs:
            # Sort by timestamp (most recent first) and return the most recent
            build_dirs.sort(key=lambda x: x[0], reverse=True)
            most_recent = build_dirs[0][1]
            print(f"🔍 Found {len(build_dirs)} timestamped build directories")
            print(f"📂 Using most recent: {most_recent}")
            return most_recent
    
    # Fall back to original _build directory
    original_build = base_data_root / "_build"
    if original_build.exists():
        print(f"📂 Using original build directory: {original_build}")
        return original_build
    
    # If nothing exists, raise an error
    raise FileNotFoundError(
        f"No build directories found in {base_data_root}. "
        f"Available options:\n"
        f"  1. Run build_dataset.py to create a new dataset\n"
        f"  2. Specify a specific build directory with --data-root"
    )


def resolve_data_root(data_root_arg: str) -> Tuple[Path, str]:
    """
    Resolve the data root argument to a valid build directory and extract timestamp.
    
    Args:
        data_root_arg: Command line argument for data root
        
    Returns:
        Tuple of (Path to resolved build directory, extracted timestamp or None)
    """
    data_root = Path(data_root_arg).expanduser()
    
    def extract_timestamp_from_build_dir(build_path: Path) -> str:
        """Extract timestamp from build directory name like _build_20241031_124942"""
        match = re.search(r'_build_(\d{8}_\d{6})$', build_path.name)
        if match:
            return match.group(1)
        return None
    
    # Case 1: User specified a specific _build directory (timestamped or not)
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
    
    # Case 2: User specified base data directory - find most recent build
    elif (data_root / "Pseudocolored").exists() or (data_root / "Masks").exists():
        # This looks like a base data directory, find the most recent build
        recent_build = find_most_recent_build_dir(data_root)
        timestamp = extract_timestamp_from_build_dir(recent_build)
        if timestamp:
            print(f"📅 Extracted build timestamp: {timestamp}")
        return recent_build, timestamp
    
    # Case 3: User specified some other directory - check if it contains dataset/
    elif (data_root / "dataset").exists():
        print(f"📂 Using specified directory with dataset: {data_root}")
        # Try to extract timestamp if it's a build-like directory
        timestamp = extract_timestamp_from_build_dir(data_root)
        if timestamp:
            print(f"📅 Extracted build timestamp: {timestamp}")
        return data_root, timestamp
    
    # Case 4: Nothing found - provide helpful error
    else:
        raise FileNotFoundError(
            f"Could not find valid dataset at: {data_root}\n"
            f"Please specify one of:\n"
            f"  1. Base data directory (e.g., /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane)\n"
            f"  2. Specific build directory (e.g., /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20241031_124305)\n"
            f"  3. Any directory containing dataset/train/ and dataset/val/"
        )


# ---- Loss Functions ----------------------

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute cross entropy
    ce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    
    # Compute focal weight
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = K.pow(1.0 - p_t, gamma)
    
    # Apply alpha weighting
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    return K.mean(alpha_t * focal_weight * ce)


def dice_loss(y_true, y_pred):
    """Dice loss for better boundary detection"""
    smooth = 1.0
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - score


def combined_loss_with_focal(y_true, y_pred):
    """
    Combined loss with focal component for hard example mining.
    BCE + Dice + Focal (0.5x weight)
    
    Use for challenging datasets with class imbalance or difficult boundaries.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    
    return bce + dice + 0.5 * focal

def combined_loss_standard(y_true, y_pred):
    """
    Standard combined loss.
    BCE + Dice
    
    Stable and effective for most medical image segmentation tasks.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    
    return bce + dice


# ---- Efficient Data Pipeline ----------------------------------------------

class TileDataset:
    """Efficient data loading with caching and prefetching"""
    
    def __init__(self, images_dir, masks_dir, batch_size, augment=True, cache_size=100, mean=None, std=None, 
                 normalization_method='zscore', percentile_low=1.0, percentile_high=99.0, augment_fn=None, augment_level: str = 'moderate'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.augment_fn = augment_fn     # ⬅️ NEW
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
        
        # Load file lists
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        self.mask_files = {p.stem: p for p in self.masks_dir.glob("*.tif")}
        
        # Filter to paired files only
        self.pairs = []
        for img_path in self.image_files:
            if img_path.stem in self.mask_files:
                self.pairs.append((img_path, self.mask_files[img_path.stem]))
        
        print(f"Found {len(self.pairs)} paired tiles in {images_dir.name}")
        
        # Cache for frequently accessed tiles
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    def load_pair(self, img_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache a single image-mask pair"""
        cache_key = img_path.stem
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Load mask
        mask = tiff.imread(str(mask_path)).astype(np.float32)
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # Cache if space available
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
                    
                    # Augmentation using moderate pipeline from data.py
                    if self.augment and self.augment_fn is not None:
                        img, mask = self.augment_fn(img, mask, rng)
                    
                    # Apply selected normalization method (stain normalization already applied during dataset building)
                    if self.normalization_method == 'zscore':
                        # Z-score normalization using dataset statistics 
                        img = (img - self.mean) / (self.std + 1e-10)
                    elif self.normalization_method == 'percentile':
                        # 1-99 percentile normalization 
                        img = normalize_image(img, method='percentile', 
                                            p_low=self.percentile_low, p_high=self.percentile_high)
                    else:
                        raise ValueError(f"Unknown normalization method: {self.normalization_method}")
                    
                    images.append(img)
                    masks.append(mask)
                
                # Pad last batch if needed
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


# ---- Model ------------------

class AdiposeUNet:
    def __init__(self, checkpoint_name: str, freeze_encoder: bool = True, build_timestamp: str = None):
        self.checkpoint_name = checkpoint_name
        self.freeze_encoder = freeze_encoder
        self.net: Model | None = None
        
        # Use build timestamp if provided, otherwise generate new one
        if build_timestamp:
            timestamp = build_timestamp
            print(f"📅 Using build timestamp: {timestamp}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"📅 Generated new timestamp: {timestamp}")
        
        timestamped_name = f"{timestamp}_{checkpoint_name}_1024_finetune"
        self.checkpoint_dir = Path(f"checkpoints/segmentation/{timestamped_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
    
    def build_model(self, init_nb: int = 44, dropout_rate: float = 0.3):
        """
        DO NOT change to sigmoid - softmax worked better!
        """
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
    
    def compile_model(self, lr: float = 1e-4, use_focal_loss: bool = False, optimizer_type: str = 'adam'):
        """
        Compile model with selected optimizer (Adam or AdamW).
        """
        opt = optimizer_type.lower()
        if opt == 'adamw':
            optimizer = AdamW(learning_rate=lr)
        elif opt == 'adam':
            optimizer = Adam(learning_rate=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        loss_fn = combined_loss_with_focal if use_focal_loss else combined_loss_standard
        self.net.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[dice_coef, 'binary_accuracy']
        )
    
    def load_legacy_weights(self, h5_path: str):
        """Load weights from legacy TF format"""
        import h5py
        from tensorflow.python.keras.saving import hdf5_format
        
        with h5py.File(h5_path, 'r') as f:
            group = f['model_weights'] if 'model_weights' in f else f
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


# ---- Training Settings Logging --------------------------------------------

def capture_system_info() -> Dict:
    """
    Capture comprehensive system and environment information.
    
    Returns:
        Dictionary with system information
    """
    system_info = {}
    
    # Basic system information
    system_info["platform"] = platform.platform()
    system_info["system"] = platform.system()
    system_info["machine"] = platform.machine()
    system_info["processor"] = platform.processor()
    system_info["python_version"] = platform.python_version()
    
    # TensorFlow information
    system_info["tensorflow_version"] = tf.__version__
    
    # Try to get Keras version (integrated with TF since TF 2.0)
    try:
        # Keras is integrated into TensorFlow, so we use TF version
        system_info["keras_version"] = tf.__version__
    except:
        system_info["keras_version"] = "integrated with TensorFlow"
    
    # GPU information
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
                # Try to get GPU details if available
                try:
                    gpu_details["memory_limit"] = tf.config.experimental.get_memory_info(gpu.name)
                except:
                    pass
                gpu_info.append(gpu_details)
            system_info["gpus"] = gpu_info
        else:
            system_info["gpus"] = []
    except Exception as e:
        system_info["gpus"] = f"Error getting GPU info: {str(e)}"
    
    # CPU information
    try:
        system_info["cpu_count"] = os.cpu_count()
    except:
        system_info["cpu_count"] = "Unknown"
    
    # Memory information (if available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["memory_total_gb"] = round(memory.total / (1024**3), 2)
        system_info["memory_available_gb"] = round(memory.available / (1024**3), 2)
    except ImportError:
        system_info["memory_info"] = "psutil not available"
    except Exception as e:
        system_info["memory_info"] = f"Error: {str(e)}"
    
    # Git repository information (if available)
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.').decode().strip()
        system_info["git_commit"] = git_commit
        
        # Get git status
        git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd='.').decode().strip()
        system_info["git_dirty"] = len(git_status) > 0
        if git_status:
            system_info["git_modifications"] = git_status.split('\n')
    except:
        system_info["git_info"] = "Git repository information not available"
    
    return system_info


def log_training_settings(
    checkpoint_dir: Path,
    command_line_args: Dict,
    data_config: Dict,
    model_config: Dict,
    training_config: Dict
) -> None:
    """
    Create comprehensive training settings log file.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        command_line_args: Command line arguments used
        data_config: Data configuration settings
        model_config: Model architecture configuration
        training_config: Training hyperparameters
    """
    settings_log_path = checkpoint_dir / "training_settings.log"
    
    # Capture system information
    system_info = capture_system_info()
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create comprehensive settings dictionary
    settings = {
        "training_session": {
            "timestamp": timestamp,
            "script_name": "train_adipose_unet_2.py",
            "working_directory": str(Path.cwd()),
            "checkpoint_directory": str(checkpoint_dir)
        },
        "command_line": command_line_args,
        "system_environment": system_info,
        "data_configuration": data_config,
        "model_architecture": model_config,
        "training_parameters": training_config
    }
    
    # Write to log file in both JSON and human-readable format
    with open(settings_log_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TRAINING SETTINGS LOG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Script: train_adipose_unet_2.py\n")
        f.write(f"Checkpoint Directory: {checkpoint_dir}\n\n")
        
        # Command line section
        f.write("-" * 60 + "\n")
        f.write("COMMAND LINE ARGUMENTS\n")
        f.write("-" * 60 + "\n")
        for key, value in command_line_args.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # System information section
        f.write("-" * 60 + "\n")
        f.write("SYSTEM ENVIRONMENT\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Platform: {system_info.get('platform', 'Unknown')}\n")
        f.write(f"  Python Version: {system_info.get('python_version', 'Unknown')}\n")
        f.write(f"  TensorFlow Version: {system_info.get('tensorflow_version', 'Unknown')}\n")
        f.write(f"  CPU Count: {system_info.get('cpu_count', 'Unknown')}\n")
        
        if 'memory_total_gb' in system_info:
            f.write(f"  Memory Total: {system_info['memory_total_gb']} GB\n")
            f.write(f"  Memory Available: {system_info['memory_available_gb']} GB\n")
        
        if 'gpus' in system_info and isinstance(system_info['gpus'], list):
            f.write(f"  GPUs: {len(system_info['gpus'])} detected\n")
            for i, gpu in enumerate(system_info['gpus']):
                f.write(f"    GPU {i}: {gpu.get('name', 'Unknown')}\n")
        
        if 'git_commit' in system_info:
            f.write(f"  Git Commit: {system_info['git_commit']}\n")
            f.write(f"  Git Clean: {not system_info.get('git_dirty', True)}\n")
        
        f.write("\n")
        
        # Data configuration section
        f.write("-" * 60 + "\n")
        f.write("DATA CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        for key, value in data_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Model architecture section
        f.write("-" * 60 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 60 + "\n")
        for key, value in model_config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Training parameters section
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
        
        # JSON section for programmatic parsing
        f.write("-" * 60 + "\n")
        f.write("MACHINE READABLE FORMAT (JSON)\n")
        f.write("-" * 60 + "\n")
        f.write(json.dumps(settings, indent=2, default=str))
        f.write("\n")
    
    print(f"✓ Saved training settings log to: {settings_log_path}")


def _select_augment_fn(level: str):
    lvl = (level or 'moderate').lower()
    if lvl == 'light':
        return augment_pair_light, 'light'
    if lvl == 'heavy':
        return augment_pair_heavy, 'heavy'
    if lvl == 'tta-style' or lvl == 'tta_style':
        return augment_pair_tta_style, 'tta-style'
    if lvl in ('none', 'off', 'disable'):
        return None, 'none'
    # default
    return augment_pair_moderate, 'moderate'

# ---- Training with Improved Callbacks -------------------------------------

def train_model(
    data_root: Path,
    pretrained_weights: str,
    batch_size: int = 2,
    epochs_phase1: int = 75,
    epochs_phase2: int = 150,
    normalization_method: str = 'zscore',
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    use_focal_loss: bool = False,
    build_timestamp: str = None,
    augmentation_level: str = 'moderate',
    checkpoint_suffix: str = '',
    optimizer_type: str = 'adam',
):
    """
    Two-phase training
    """
    
    print("✓ Using float32 precision")
    
    # Setup data
    data_root = Path(data_root)
    train_images = data_root / "dataset" / "train" / "images"
    train_masks = data_root / "dataset" / "train" / "masks"
    val_images = data_root / "dataset" / "val" / "images"
    val_masks = data_root / "dataset" / "val" / "masks"
    
    # --- Compute shared train statistics for z-score normalization ---
    def compute_mean_std(image_paths, max_n=None):
        vals = []
        for i, p in enumerate(image_paths):
            if max_n and i >= max_n:
                break
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            vals.append(img.reshape(-1))
        vals = np.concatenate(vals)
        return float(vals.mean()), float(vals.std() + 1e-10)

    # Compute mean/std from all (or many) train images
    train_image_paths = sorted(train_images.glob("*.jpg"))
    train_mean, train_std = compute_mean_std(train_image_paths)
    print(f"Global normalization stats: mean={train_mean:.2f}, std={train_std:.2f}")

    augment_fn, augment_label = _select_augment_fn(augmentation_level)
    use_augment = augment_label != 'none'

    # --- Build datasets with shared normalization stats ---
    # Use pre-processed dataset (stain normalization applied during dataset building)
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
    
    # IMPROVED: Use math.ceil for correct step calculation
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    validation_steps = max(1, math.ceil(len(val_dataset) / batch_size))
    
    print(f"\n{'='*60}")
    print(f"Dataset Configuration:")
    print(f"{'='*60}")
    print(f"  Training:   {len(train_dataset)} tiles ({steps_per_epoch} steps/epoch)")
    print(f"  Validation: {len(val_dataset)} tiles ({validation_steps} steps/epoch)")
    print(f"  Augmentation: {augment_label}")
    print(f"  Preprocessing: Applied during dataset building")
    norm_display = f"{normalization_method.upper()}"
    if normalization_method == 'percentile':
        norm_display += f" ({percentile_low}-{percentile_high})"
    elif normalization_method == 'zscore':
        norm_display
    print(f"  Normalization: {norm_display}")
    print(f"  Architecture: 2-ch softmax")
    loss_display = "BCE + Dice + 0.5*Focal" if use_focal_loss else "BCE + Dice"
    print(f"  Loss: {loss_display}")
    print(f"{'='*60}\n")
    
    # Build model
    checkpoint_base = "adipose_sybreosin"
    if checkpoint_suffix:
        checkpoint_name = f"{checkpoint_base}_{checkpoint_suffix}"
    else:
        checkpoint_name = checkpoint_base
    
    model = AdiposeUNet(checkpoint_name, freeze_encoder=True, build_timestamp=build_timestamp)
    model.build_model(init_nb=44, dropout_rate=0.3)
    model.compile_model(lr=1e-4, use_focal_loss=use_focal_loss, optimizer_type=optimizer_type)
    
    # --- Save normalization statistics for inference reproducibility ---
    normalization_stats = {
        "mean": float(train_mean),
        "std": float(train_std),
        "normalization_method": normalization_method,
        "dataset_path": str(data_root),
        "num_training_images": len(train_image_paths),
        "build_timestamp": build_timestamp,
        "timestamp_saved": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "percentile_low": float(percentile_low) if normalization_method == 'percentile' else None,
        "percentile_high": float(percentile_high) if normalization_method == 'percentile' else None,
        "batch_size": batch_size,
        "image_size": [1024, 1024],
        "augmentation": "augment_label",
        "preprocessing_applied": "stain_normalization"
    }
    
    # Save normalization stats to checkpoint directory
    normalization_stats_path = model.checkpoint_dir / "normalization_stats.json"
    with open(normalization_stats_path, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    print(f"✓ Saved normalization statistics to: {normalization_stats_path}")
    print(f"  Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    print(f"  Method: {normalization_method}")
    if normalization_method == 'percentile':
        print(f"  Percentile range: {percentile_low}-{percentile_high}")
    
    # Load pretrained weights
    if pretrained_weights and Path(pretrained_weights).exists():
        model.load_legacy_weights(pretrained_weights)
    else:
        print("WARNING: No pretrained weights found, training from scratch")
    
    # --- Log comprehensive training settings ---
    # Prepare configuration dictionaries for logging
    command_line_args = {
        "data_root": str(data_root),
        "pretrained_weights": pretrained_weights,
        "batch_size": batch_size,
        "epochs_phase1": epochs_phase1,
        "epochs_phase2": epochs_phase2,
        "normalization_method": normalization_method,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "use_focal_loss": use_focal_loss,
        "build_timestamp": build_timestamp,
        "augmentation_level": augmentation_level,
        "optimizer": optimizer_type,
    }
    
    data_config = {
        "dataset_path": str(data_root),
        "train_images_count": len(train_dataset),
        "validation_images_count": len(val_dataset),
        "train_images_dir": str(train_images),
        "validation_images_dir": str(val_images),
        "normalization_mean": train_mean,
        "normalization_std": train_std,
        "normalization_method": normalization_method,
        "percentile_range": f"{percentile_low}-{percentile_high}" if normalization_method == 'percentile' else None,
        "augmentation_type": augment_label,
        "preprocessing_applied": "stain_normalization",
        "image_size": [1024, 1024], 
        "cache_size": 100
    }
    
    trainable_params = int(np.sum([K.count_params(w) for w in model.net.trainable_weights]))
    non_trainable_params = int(np.sum([K.count_params(w) for w in model.net.non_trainable_weights]))

    model_config = {
        "architecture": "U-Net with dilated convolutions",
        "input_shape": [1024, 1024],
        "output_activation": "softmax (2-channel)",
        "init_filters": 44,
        "dropout_rate": 0.3,
        "encoder_frozen": True,
        "total_parameters": model.net.count_params(),
        "trainable_parameters": trainable_params,            # ← fixed
        "non_trainable_parameters": non_trainable_params,
        "optimizer": optimizer_type.upper(),
        "loss_function": "combined_loss_with_focal" if use_focal_loss else "combined_loss_standard",
        "metrics": ["dice_coefficient", "binary_accuracy"]
    }
    
    training_config = {
        "total_epochs": epochs_phase1 + epochs_phase2,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "phase1": {
            "epochs": epochs_phase1,
            "learning_rate": 1e-4,
            "encoder_frozen": True,
            "callbacks": ["ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping", "CSVLogger"],
            "monitor_metric": "val_dice_coef",
            "early_stopping_patience": 15,
            "reduce_lr_patience": 5
        },
        "phase2": {
            "epochs": epochs_phase2,
            "learning_rate": 1e-5,
            "encoder_frozen": False,
            "callbacks": ["ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping", "CSVLogger"],
            "monitor_metric": "val_dice_coef", 
            "early_stopping_patience": 15,
            "reduce_lr_patience": 5
        },
        "precision": "float32",
        "seed": 865
    }
    
    # Log all training settings before training starts
    log_training_settings(
        checkpoint_dir=model.checkpoint_dir,
        command_line_args=command_line_args,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config
    )
    
    model.net.summary()
    
    # ==== PHASE 1: Frozen Encoder ====
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training decoder only ({epochs_phase1} epochs)")
    print(f"Learning rate: 1e-4")
    print(f"{'='*60}\n")
    
    callbacks_phase1 = [
        ModelCheckpoint(
            filepath=str(model.checkpoint_dir / "phase1_best.weights.h5"),
            monitor='val_dice_coef',  
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coef',  
            mode='max',
            factor=0.5,
            patience=5,  
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coef',  
            mode='max',
            patience=15,  
            restore_best_weights=False,  # We load manually
            verbose=1
        ),
        CSVLogger(str(model.checkpoint_dir / "phase1_training.log"))
    ]
    
    model.net.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_phase1,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Save final phase 1 weights
    model.save_weights_modern("phase1_final")
    
    # ==== PHASE 2: Full Fine-tuning ====
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning all layers ({epochs_phase2} epochs)")
    print(f"Learning rate: 1e-5 (10x lower)")
    print(f"{'='*60}\n")
    
    # IMPROVED: Load BEST Phase 1 model (not last epoch)
    best_phase1_path = model.checkpoint_dir / "phase1_best.weights.h5"
    if best_phase1_path.exists():
        print(f"Loading BEST Phase 1 model from: {best_phase1_path}")
        model.net.load_weights(str(best_phase1_path))
        print("✓ Loaded best Phase 1 weights\n")
    else:
        print("WARNING: Best Phase 1 weights not found, using last epoch weights\n")
    
    model.unfreeze_encoder()
    model.compile_model(lr=1e-5, optimizer_type=optimizer_type)  # FIXED: 10x lower LR
    
    callbacks_phase2 = [
        ModelCheckpoint(
            filepath=str(model.checkpoint_dir / "phase2_best.weights.h5"),
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coef',  
            mode='max',
            factor=0.5,
            patience=5,  
            min_lr=1e-8,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coef',  
            mode='max',
            patience=15,  
            restore_best_weights=False,
            verbose=1
        ),
        CSVLogger(str(model.checkpoint_dir / "phase2_training.log"))
    ]
    
    model.net.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_phase2,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Save final weights
    model.save_weights_modern("phase2_final")
    
    # Load and save best overall model
    best_phase2_path = model.checkpoint_dir / "phase2_best.weights.h5"
    if best_phase2_path.exists():
        model.net.load_weights(str(best_phase2_path))
        model.save_weights_modern("best_overall")
    
    print(f"\n{'='*60}")
    print("✓ Training Complete!")
    print(f"{'='*60}")
    print(f"Checkpoint directory: {model.checkpoint_dir}")
    print(f"\nBest models saved:")
    print(f"  - phase1_best.weights.h5  (best Phase 1 by val_dice)")
    print(f"  - phase2_best.weights.h5  (best Phase 2 by val_dice)")
    print(f"  - weights_best_overall.weights.h5  (final best)")
    print(f"{'='*60}\n")
    
    return model


# ---- CLI ------------------------------------------------------------------

def main():
    # GPU memory growth
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Train U-Net for adipose segmentation"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane',
        help='Base data directory (auto-finds most recent _build_*) or specific build directory'
    )
    parser.add_argument(
        '--pretrained-weights',
        type=str,
        default='checkpoints/unet_1024_dilation/weights_loss_val.weights.h5',
        help='Path to pretrained weights'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size (2-4 recommended for 1024x1024 tiles)'
    )
    parser.add_argument(
        '--epochs-phase1',
        type=int,
        default=50,
        help='Epochs for phase 1 (frozen encoder)'
    )
    parser.add_argument(
        '--epochs-phase2',
        type=int,
        default=100,
        help='Epochs for phase 2 (full fine-tuning)'
    )
    parser.add_argument(
        '--normalization-method',
        type=str,
        default='zscore',
        choices=['zscore', 'percentile'],
        help='Intensity normalization method: zscore (default) or percentile (1-99 robust)'
    )
    parser.add_argument(
        '--percentile-low',
        type=float,
        default=1.0,
        help='Lower percentile for percentile normalization (default: 1.0)'
    )
    parser.add_argument(
        '--percentile-high',
        type=float,
        default=99.0,
        help='Upper percentile for percentile normalization (default: 99.0)'
    )
    parser.add_argument(
        '--use-focal-loss',
        action='store_true',
        default=False,
        help='Use focal loss component for hard example mining (default: False, uses BCE+Dice only)'
    )
    parser.add_argument(
        '--augmentation-level',
        type=str,
        default='moderate',
        choices=['none', 'light', 'moderate', 'heavy', 'tta-style'],
        help='Augmentation intensity for training tiles. tta-style mimics TTA transformations.'
    )
    parser.add_argument(
        '--checkpoint-suffix',
        type=str,
        default='',
        help='Optional suffix for checkpoint folder name (e.g., "_perc" for percentile normalization)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw'],
        help='Optimizer to use (default: adam)'
    )
    
    args = parser.parse_args()
    
    # Resolve data root to actual build directory and extract timestamp
    try:
        data_root, build_timestamp = resolve_data_root(args.data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Train model
    model = train_model(
        data_root=data_root,
        pretrained_weights=args.pretrained_weights,
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        normalization_method=args.normalization_method,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        use_focal_loss=args.use_focal_loss,
        build_timestamp=build_timestamp,
        augmentation_level=args.augmentation_level,
        checkpoint_suffix=args.checkpoint_suffix,
        optimizer_type=args.optimizer,
    )
    
    print("\n✓ Training complete! Use the best model for inference:")
    print(f"  Best model: {model.checkpoint_dir}/weights_best_overall.weights.h5")


if __name__ == "__main__":
    main()
