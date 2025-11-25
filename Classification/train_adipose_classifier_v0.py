#!/usr/bin/env python3
"""
Tile-level adipose classifier training.

Uses grayscale tiles from build_class_dataset.py, resizes to 299x299, and fine-tunes
an InceptionV3 head with transfer learning from legacy checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from src.utils.data import augment_grayscale_tile_classification
from src.utils.seed_utils import get_project_seed

GLOBAL_SEED = get_project_seed()
tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEFAULT_WEIGHTS = "/home/luci/adipose_tissue-unet/checkpoints/classifier_runs/tile_classifier_InceptionV3/tile_adipocyte.weights.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train adipose tile classifier (binary)")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to dataset directory with train/ and val/ splits (Keras-style).")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Subdirectory name for training split (default: train).")
    parser.add_argument("--val-split", type=str, default="val",
                        help="Subdirectory name for validation split (default: val).")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/classification",
                        help="Directory to store training artifacts.")
    parser.add_argument("--pretrained-weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to legacy weights (.h5 or SavedModel). Used with by_name loading.")
    parser.add_argument("--warmup-epochs", type=int, default=6,
                        help="Epochs with frozen backbone.")
    parser.add_argument("--finetune-epochs", type=int, default=20,
                        help="Epochs after unfreezing top Inception blocks.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-lr", type=float, default=1e-3,
                        help="Learning rate during warmup.")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="Learning rate during fine-tuning.")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate for classification head.")
    parser.add_argument("--unfreeze-from", type=str, default="mixed7",
                        help="Name of Inception layer to start unfreezing during fine-tuning.")
    parser.add_argument("--patience", type=int, default=4,
                        help="Patience for EarlyStopping/ReduceLROnPlateau (monitors val_auc).")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for loss (default: 0.1).")
    parser.add_argument("--use-class-weights", action="store_true", default=False,
                        help="Enable image-level class weighting (disabled by default for balanced datasets).")
    parser.add_argument("--pos-weight-multiplier", type=float, default=1.0,
                        help="Multiplier for positive class weight (e.g., 1.5 to emphasize adipose detection).")
    parser.add_argument("--save-best-only", action="store_true", default=True)
    parser.add_argument("--no-save-best-only", dest="save_best_only", action="store_false",
                        help="Disable save_best_only on checkpoints.")
    parser.add_argument("--percentile-norm", type=lambda x: str(x).lower() == 'true', required=True,
                        help="Apply percentile normalization to enhance contrast. Required: true or false.")
    parser.add_argument("--percentile-low", type=float, default=1.0,
                        help="Lower percentile for percentile normalization (default: 1.0).")
    parser.add_argument("--percentile-high", type=float, default=99.0,
                        help="Upper percentile for percentile normalization (default: 99.0).")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to add to checkpoint directory name (e.g., '_v2', '_experiment1').")
    return parser.parse_args()


def list_split_files(split_dir: Path) -> Tuple[List[str], List[int]]:
    pos_dir = split_dir / "adipose"
    neg_dir = split_dir / "not_adipose"
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(f"Expected directories {pos_dir} and {neg_dir}.")
    file_paths: List[str] = []
    labels: List[int] = []
    for path in sorted(pos_dir.glob("*.jpg")):
        file_paths.append(str(path))
        labels.append(1)
    for path in sorted(neg_dir.glob("*.jpg")):
        file_paths.append(str(path))
        labels.append(0)
    if not file_paths:
        raise RuntimeError(f"No images discovered under {split_dir}.")
    return file_paths, labels


def extract_slide_base(filename: str) -> str:
    """
    Extract slide base name from tile filename.
    
    Example: 'slide_001_r5_c3.jpg' → 'slide_001'
    
    Args:
        filename: Full path or filename of tile
        
    Returns:
        Base slide name without row/column suffix
    """
    path = Path(filename)
    stem = path.stem  # Remove .jpg extension
    
    # Remove _rX_cY pattern
    parts = stem.split('_')
    
    # Find where _rX_cY pattern starts (look backwards for r and c pattern)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].startswith('r') and parts[i+1].startswith('c'):
            return '_'.join(parts[:i])
    
    # Fallback: return full stem if pattern not found
    return stem


def compute_image_level_class_weights(file_paths: List[str], 
                                      labels: List[int],
                                      pos_weight_multiplier: float = 1.0) -> dict:
    """
    Compute class weights based on SLIDE-level contribution, not tile counts.
    
    This prevents individual slides from dominating the class weight calculation.
    
    Args:
        file_paths: List of tile file paths
        labels: Corresponding labels (0=not_adipose, 1=adipose)
        pos_weight_multiplier: Extra weight multiplier for positive class (default: 1.0)
        
    Returns:
        Class weights dictionary {0: weight, 1: weight}
    """
    # Group tiles by slide
    slide_labels = {}  # {slide_base: set of labels from that slide}
    
    for path, label in zip(file_paths, labels):
        slide_base = extract_slide_base(path)
        if slide_base not in slide_labels:
            slide_labels[slide_base] = set()
        slide_labels[slide_base].add(label)
    
    # Count how many slides contribute to each class
    slides_per_class = {0: 0, 1: 0}
    for slide_base, label_set in slide_labels.items():
        if 0 in label_set:
            slides_per_class[0] += 1
        if 1 in label_set:
            slides_per_class[1] += 1
    
    total_slides = len(slide_labels)
    
    print(f"[Image-Level Class Distribution]")
    print(f"  Total slides: {total_slides}")
    print(f"  Slides with not_adipose (0): {slides_per_class[0]}")
    print(f"  Slides with adipose (1):     {slides_per_class[1]}")
    
    # Compute weights based on slide counts (inverse frequency weighting)
    weights = {}
    for cls in (0, 1):
        if slides_per_class[cls] == 0:
            weights[cls] = 0.0
        else:
            # Standard balanced weighting: total / (num_classes * count_for_class)
            weights[cls] = total_slides / (2.0 * slides_per_class[cls])
    
    # Apply multiplier to positive class
    weights[1] *= pos_weight_multiplier
    
    print(f"  Image-level class weights: {weights}")
    return weights


def _percentile_normalize(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Apply percentile normalization to enhance contrast."""
    img = np.asarray(img, dtype=np.float32)
    plow, phigh = np.percentile(img, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    # Normalize to [0, 1] then scale to [0, 255] for InceptionV3 preprocessing
    normalized = np.clip((img - plow) / scale, 0, 1)
    return (normalized * 255.0).astype(np.float32)


def _numpy_augment(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    return augment_grayscale_tile_classification(img)


def _preprocess(path: tf.Tensor, label: tf.Tensor, training: bool, percentile_norm: bool, p_low: float, p_high: float) -> Tuple[tf.Tensor, tf.Tensor]:
    
    def _load_and_process_all(path_bytes, training, percentile_norm, p_low, p_high):
        """Single Python function to handle all preprocessing to avoid nested py_function calls."""
        import cv2
        
        # Read JPEG image
        path_str = path_bytes.numpy().decode('utf-8')
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path_str}")
        img = img.astype(np.float32)
        
        # Apply percentile normalization before augmentation if enabled
        if percentile_norm:
            plow, phigh = np.percentile(img, (p_low, p_high))
            scale = max(phigh - plow, 1e-3)
            # Normalize to [0, 1] then scale to [0, 255] for InceptionV3 preprocessing
            img = np.clip((img - plow) / scale, 0, 1)
            img = (img * 255.0).astype(np.float32)
        
        # Apply augmentation if training
        if training:
            img = augment_grayscale_tile_classification(img)
        
        return img.astype(np.float32)
    
    # Single py_function call for all preprocessing
    image = tf.py_function(
        func=lambda p: _load_and_process_all(p, training, percentile_norm, p_low, p_high),
        inp=[path],
        Tout=tf.float32
    )
    image.set_shape([None, None])
    
    # Expand to add channel dimension
    image = tf.expand_dims(image, axis=-1)
    
    # Resize to InceptionV3 input size
    image = tf.image.resize(image, [299, 299], method="bilinear")
    
    # Convert grayscale to RGB by tiling
    image = tf.tile(image, [1, 1, 3])
    
    # Apply InceptionV3 preprocessing
    image = preprocess_input(image)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(files: List[str], labels: List[int], batch_size: int, training: bool, 
                 percentile_norm: bool = True, p_low: float = 1.0, p_high: float = 99.0) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(files), seed=GLOBAL_SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _preprocess(p, y, training, percentile_norm, p_low, p_high),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(dropout_rate: float) -> Tuple[Model, Model]:
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation="sigmoid", name="adipose_score")(x)
    model = Model(inputs=base.input, outputs=out)
    return model, base


def load_transfer_weights(model: Model, weights_path: str | None):
    """
    Load transfer learning weights on top of ImageNet initialization.
    
    The model is first initialized with ImageNet weights (in build_model),
    then this function optionally loads task-specific weights (e.g., adipocyte detector)
    on top, using by_name matching to transfer compatible layers.
    
    Args:
        model: Model with ImageNet-initialized backbone
        weights_path: Path to additional pretrained weights (optional)
    """
    if not weights_path:
        print("[Init] No pretrained weights specified; using ImageNet initialization only.")
        return
    
    weight_path = Path(weights_path)
    if not weight_path.exists():
        print(f"[Init] WARNING: Pretrained weights not found at {weights_path}")
        print(f"[Init] Using ImageNet initialization only (no additional transfer weights).")
        return
    
    try:
        print(f"[Init] Loading additional pretrained weights from: {weights_path}")
        print(f"[Init] Strategy: by_name matching (loads compatible layers on top of ImageNet weights)")
        model.load_weights(str(weight_path), by_name=True, skip_mismatch=True)
        print(f"[Init] ✓ Successfully loaded pretrained weights")
        print(f"[Init] Model initialized with: ImageNet backbone + task-specific weights from {weight_path.name}")
    except Exception as exc:
        print(f"[Init] ERROR: Failed to load weights from {weights_path}")
        print(f"[Init] Error details: {exc}")
        print(f"[Init] Falling back to ImageNet initialization only")


def freeze_backbone(base_model: Model):
    for layer in base_model.layers:
        layer.trainable = False


def unfreeze_from_layer(base_model: Model, layer_name: str):
    unfreeze = False
    for layer in base_model.layers:
        if layer.name == layer_name:
            unfreeze = True
        layer.trainable = unfreeze


def compile_model(model: Model, lr: float, label_smoothing: float):
    optimizer = Adam(learning_rate=lr)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def create_callbacks(out_dir: Path, monitor: str, patience: int, save_best_only: bool) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = []
    callbacks.append(ModelCheckpoint(
        filepath=str(out_dir / "weights_best.weights.h5"),
        monitor=monitor,
        mode="max",
        save_best_only=save_best_only,
        save_weights_only=True,
        verbose=1,
    ))
    callbacks.append(ReduceLROnPlateau(
        monitor=monitor,
        mode="max",
        factor=0.5,
        patience=patience,
        min_lr=1e-6,
        verbose=1,
    ))
    callbacks.append(EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=patience + 2,
        restore_best_weights=True,
        verbose=1,
    ))
    callbacks.append(CSVLogger(str(out_dir / "training.log")))
    return callbacks


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / args.train_split
    val_dir = dataset_root / args.val_split

    # Extract timestamp and channel from dataset build folder name
    # Pattern: _build_class[_ecm][_MS]_YYYYMMDD_HHMMSS
    timestamp = None
    channel_suffix = ""
    build_folder = dataset_root.parent.name  # e.g., "_build_class_ecm_MS_20251111_130505"
    import re
    match = re.search(r'_(\d{8}_\d{6})$', build_folder)
    if match:
        timestamp = match.group(1)
        print(f"[Config] Using dataset timestamp from build folder: {timestamp}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[Config] Could not extract timestamp from '{build_folder}', using current: {timestamp}")
    
    # Check if build folder contains 'ecm' to add channel suffix
    if '_ecm_' in build_folder.lower():
        channel_suffix = "_ecm"
        print(f"[Config] Detected ECM channel in build folder, adding channel suffix")
        
        # Check if build folder also contains 'MS' to add MS suffix
        if '_MS_' in build_folder or 'MS_' in build_folder:
            channel_suffix = "_ecm_MS"
            print(f"[Config] Detected MS in build folder, adding MS suffix")

    train_files, train_labels = list_split_files(train_dir)
    val_files, val_labels = list_split_files(val_dir)

    print(f"[Data] Train samples: {len(train_files)} | Val samples: {len(val_files)}")
    print(f"[Normalization] Percentile normalization: {'ENABLED' if args.percentile_norm else 'DISABLED'}")
    if args.percentile_norm:
        print(f"[Normalization]   Percentile range: {args.percentile_low}-{args.percentile_high}")
    
    # Compute image-level class weights if requested
    if args.use_class_weights:
        class_weights = compute_image_level_class_weights(
            train_files,
            train_labels,
            pos_weight_multiplier=args.pos_weight_multiplier
        )
        print(f"[Weighting] Using image-level class weights: {json.dumps(class_weights, indent=2)}")
    else:
        class_weights = None
        print(f"[Weighting] Class weights disabled (using balanced dataset)")

    train_ds = make_dataset(train_files, train_labels, args.batch_size, training=True,
                           percentile_norm=args.percentile_norm, 
                           p_low=args.percentile_low, p_high=args.percentile_high)
    val_ds = make_dataset(val_files, val_labels, args.batch_size, training=False,
                         percentile_norm=args.percentile_norm,
                         p_low=args.percentile_low, p_high=args.percentile_high)

    model, base = build_model(dropout_rate=args.dropout)
    load_transfer_weights(model, args.pretrained_weights)

    # Build checkpoint directory name: timestamp_classifier[_ecm]_adipose_sybreosin[_percentile][suffix]
    norm_suffix = "_percentile" if args.percentile_norm else ""
    user_suffix = args.suffix if args.suffix else ""
    run_dir = Path(args.checkpoint_dir) / f"{timestamp}_classifier{channel_suffix}_adipose_sybreosin{norm_suffix}{user_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Phase 1: warm-up (freeze backbone)
    freeze_backbone(base)
    compile_model(model, lr=args.base_lr, label_smoothing=args.label_smoothing)
    callbacks = create_callbacks(run_dir, monitor="val_auc", patience=args.patience, save_best_only=args.save_best_only)

    print("\n=== Phase 1: Training classification head (frozen backbone) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.warmup_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Phase 2: fine-tune top layers
    print("\n=== Phase 2: Fine-tuning upper backbone layers ===")
    unfreeze_from_layer(base, args.unfreeze_from)
    compile_model(model, lr=args.finetune_lr, label_smoothing=args.label_smoothing)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    final_path = run_dir / "weights_final.weights.h5"
    model.save_weights(final_path)
    print(f"\n[Done] Saved final weights to {final_path}")
    print(f"[Info] Best checkpoint + logs available under {run_dir}")


if __name__ == "__main__":
    main()
