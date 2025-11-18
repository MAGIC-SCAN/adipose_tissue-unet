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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/classifier_runs",
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
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--use-class-weights", action="store_true", default=False,
                        help="Enable image-level class weighting (disabled by default for balanced datasets).")
    parser.add_argument("--pos-weight-multiplier", type=float, default=1.0,
                        help="Multiplier for positive class weight (e.g., 1.5 to emphasize adipose detection).")
    parser.add_argument("--save-best-only", action="store_true", default=True)
    parser.add_argument("--no-save-best-only", dest="save_best_only", action="store_false",
                        help="Disable save_best_only on checkpoints.")
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


def _numpy_augment(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    return augment_grayscale_tile_classification(img)


def _preprocess(path: tf.Tensor, label: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    if training:
        image = tf.squeeze(image, axis=-1)
        image = tf.numpy_function(_numpy_augment, [image], tf.float32)
        image.set_shape([None, None])
        image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [299, 299], method="bilinear")
    image = tf.tile(image, [1, 1, 3])
    image = preprocess_input(image)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(files: List[str], labels: List[int], batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(files), seed=GLOBAL_SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _preprocess(p, y, training),
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
    if not weights_path:
        return
    weight_path = Path(weights_path)
    if not weight_path.exists():
        print(f"[WARN] Pretrained weights not found at {weights_path}; using ImageNet initialization.")
        return
    try:
        print(f"[Init] Loading weights (by_name, skip mismatches) from {weights_path}")
        model.load_weights(str(weight_path), by_name=True, skip_mismatch=True)
        print("[Init] Weight loading completed.")
    except Exception as exc:
        print(f"[WARN] Failed to load weights from {weights_path}: {exc}")


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

    train_files, train_labels = list_split_files(train_dir)
    val_files, val_labels = list_split_files(val_dir)

    print(f"[Data] Train samples: {len(train_files)} | Val samples: {len(val_files)}")
    
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

    train_ds = make_dataset(train_files, train_labels, args.batch_size, training=True)
    val_ds = make_dataset(val_files, val_labels, args.batch_size, training=False)

    model, base = build_model(dropout_rate=args.dropout)
    load_transfer_weights(model, args.pretrained_weights)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.checkpoint_dir) / f"classifier_{timestamp}"
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
