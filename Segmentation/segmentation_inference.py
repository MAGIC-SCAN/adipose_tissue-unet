#!/usr/bin/env python3
"""
Segmentation Inference Script for Adipose U-Net

Run inference on a folder of images without requiring ground truth masks.
Saves prediction masks and optional overlay visualizations.

USAGE EXAMPLES:

1. Basic inference (single folder):
   python Segmentation/segmentation_inference.py \
     --images-dir /path/to/images \
     --output-dir /path/to/output \
     --weights checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5

2. Inference with TTA (8x augmentation ensemble):
   python Segmentation/segmentation_inference.py \
     --images-dir /path/to/images \
     --output-dir /path/to/output \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --use-tta --tta-mode full

3. Inference with overlays and custom threshold:
   python Segmentation/segmentation_inference.py \
     --images-dir /path/to/images \
     --output-dir /path/to/output \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --threshold 0.6 \
     --save-overlays

4. Batch inference with sliding window:
   python Segmentation/segmentation_inference.py \
     --images-dir /path/to/images \
     --output-dir /path/to/output \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --sliding-window --overlap 0.5 \
     --save-overlays

5. Full enhancement pipeline:
   python Segmentation/segmentation_inference.py \
     --images-dir /path/to/images \
     --output-dir /path/to/output \
     --weights checkpoints/latest/weights_best_overall.weights.h5 \
     --use-tta --tta-mode full \
     --sliding-window --overlap 0.5 \
     --threshold 0.5 \
     --save-overlays

OUTPUT STRUCTURE:
  output_dir/
    ├── masks/           # Binary prediction masks (TIFF)
    └── overlays/        # Predictions overlaid on images (PNG, if --save-overlays)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,
    Concatenate, Lambda, Reshape, Add,
)

try:
    import onnxruntime as ort  # Optional dependency for ONNX inference
except ImportError:
    ort = None


class AdiposeUNet:
    """U-Net model for adipose tissue segmentation"""
    
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
        
        # Output
        x = Conv2D(2, 1, activation='softmax', name='output_softmax')(up1)
        x = Lambda(lambda z: z[:, :, :, 1:2], output_shape=(1024, 1024, 1), name='output_class1')(x)
        x = Lambda(lambda z: K.squeeze(z, axis=-1), output_shape=(1024, 1024), name='squeeze')(x)
        
        self.net = Model(inputs=inputs, outputs=x)
        return self.net
    
    def load_weights(self, weights_path: str):
        """Load trained weights"""
        self.net.load_weights(weights_path)
        print(f"✓ Loaded weights from {weights_path}")
    
    def predict_single(self, image: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Run inference on a single image"""
        normalized_img = (image - mean) / (std + 1e-10)
        batch_img = np.expand_dims(normalized_img, axis=0)
        prediction = self.net.predict(batch_img, verbose=0)
        return prediction[0]


class OnnxUnetPredictor:
    """ONNX Runtime-based predictor for exported U-Net"""
    
    def __init__(self, onnx_path: str):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with `pip install onnxruntime`."
            )
        self.session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict_single(self, image: np.ndarray, mean: float, std: float) -> np.ndarray:
        normalized_img = (image - mean) / (std + 1e-10)
        batch_img = np.expand_dims(normalized_img, axis=0).astype(np.float32)
        prediction = self.session.run([self.output_name], {self.input_name: batch_img})[0]
        return prediction[0]


class TestTimeAugmentation:
    """Test Time Augmentation for improved predictions"""
    
    def __init__(self, mode: str = "basic"):
        self.mode = mode.lower()
        
        def ident(x): return x
        def rot90(x): return np.rot90(x, 1)
        def rot180(x): return np.rot90(x, 2)
        def rot270(x): return np.rot90(x, 3)
        def r90_inv(x): return np.rot90(x, 3)
        def r180_inv(x): return np.rot90(x, 2)
        def r270_inv(x): return np.rot90(x, 1)
        def flip_h(x): return np.flip(x, axis=1)
        def flip_v(x): return np.flip(x, axis=0)
        
        if self.mode == "minimal":
            self.transforms = [
                (ident, ident),
                (flip_h, flip_h),
            ]
        elif self.mode == "basic":
            self.transforms = [
                (ident, ident),
                (flip_h, flip_h),
                (flip_v, flip_v),
                (rot90, r90_inv),
            ]
        else:  # full
            self.transforms = [
                (ident, ident),
                (rot90, r90_inv),
                (rot180, r180_inv),
                (rot270, r270_inv),
                (flip_h, flip_h),
                (flip_v, flip_v),
                (lambda x: flip_h(rot90(x)), lambda x: r90_inv(flip_h(x))),
                (lambda x: flip_v(rot90(x)), lambda x: r90_inv(flip_v(x))),
            ]
    
    def predict_with_tta(self, model, image: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Run TTA by averaging de-augmented predictions"""
        preds = []
        for aug_fn, deaug_fn in self.transforms:
            aug_img = aug_fn(image)
            pred = model.predict_single(aug_img, mean, std)
            pred = deaug_fn(pred)
            preds.append(pred.astype(np.float32))
        return np.mean(preds, axis=0).astype(np.float32)


def load_normalization_stats(checkpoint_dir: Path) -> Tuple[float, float]:
    """Load training normalization statistics"""
    import json
    
    stats_path = checkpoint_dir / "normalization_stats.json"
    
    if not stats_path.exists():
        print(f"⚠️  Warning: {stats_path} not found, using default normalization")
        return 0.0, 1.0
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    mean = float(stats["mean"])
    std = float(stats["std"])
    
    print(f"✓ Loaded normalization stats: mean={mean:.4f}, std={std:.4f}")
    return mean, std


def find_weights_file(weights_arg: str) -> Tuple[str, Path]:
    """Find weights file from path (directory or file)"""
    weights_path = Path(weights_arg)
    
    if weights_path.is_file():
        return str(weights_path), weights_path.parent
    
    # Directory provided - find best weights
    candidates = [
        "weights_best_overall.weights.h5",
        "weights_best_overall.onnx",
        "phase2_best.weights.h5",
        "phase1_best.weights.h5",
        "best_model.weights.h5",
        "model_best.weights.h5",
        "weights_best.weights.h5",
        "model.onnx",
    ]
    
    for name in candidates:
        p = weights_path / name
        if p.exists():
            return str(p), weights_path
    
    # Fallback to any supported weights file
    files = (
        list(weights_path.glob("*.weights.h5"))
        + list(weights_path.glob("*.h5"))
        + list(weights_path.glob("*.onnx"))
    )
    if files:
        return str(files[0]), weights_path
    
    raise FileNotFoundError(f"No weights files found in {weights_path}")


def create_overlay_visualization(image: np.ndarray, mask: np.ndarray, 
                                color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Create overlay visualization of mask on image"""
    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.astype(np.uint8)
    
    # Create colored mask overlay
    overlay = image_rgb.copy()
    mask_binary = (mask > 0).astype(np.uint8)
    overlay[mask_binary > 0] = color
    
    # Blend
    result = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation inference on a folder of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python segmentation_inference.py --images-dir ./images --output-dir ./output --weights checkpoints/model
  
  # With TTA
  python segmentation_inference.py --images-dir ./images --output-dir ./output --weights checkpoints/model --use-tta --tta-mode full
  
  # Save overlays
  python segmentation_inference.py --images-dir ./images --output-dir ./output --weights checkpoints/model --save-overlays
        """
    )
    
    # Required arguments
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save predictions')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights file or checkpoint directory')
    
    # Inference options
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binarization threshold (0-1, default: 0.5)')
    parser.add_argument('--use-tta', action='store_true', default=False,
                       help='Use Test Time Augmentation')
    parser.add_argument('--tta-mode', type=str, default='basic',
                       choices=['minimal', 'basic', 'full'],
                       help='TTA mode: minimal (2), basic (4), full (8) augmentations')
    
    # Output options
    parser.add_argument('--save-overlays', action='store_true', default=False,
                       help='Save visualization overlays')
    parser.add_argument('--overlay-color', type=str, default='cyan',
                       choices=['cyan', 'yellow', 'magenta', 'green', 'red'],
                       help='Color for mask overlay')
    parser.add_argument('--save-probability', action='store_true', default=False,
                       help='Save probability maps (before thresholding)')
    
    args = parser.parse_args()
    
    # Setup
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return 1
    
    # Create output directories
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_overlays:
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_probability:
        prob_dir = output_dir / "probabilities"
        prob_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ADIPOSE TISSUE SEGMENTATION - INFERENCE")
    print(f"{'='*80}")
    print(f"Images: {images_dir}")
    print(f"Output: {output_dir}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"TTA: {'Enabled (' + args.tta_mode + ')' if args.use_tta else 'Disabled'}")
    print(f"{'='*80}\n")
    
    # Find and load model
    print("Loading model...")
    weights_file, checkpoint_dir = find_weights_file(args.weights)
    is_onnx_model = weights_file.lower().endswith(".onnx")
    
    if is_onnx_model:
        print(f"Detected ONNX weights: {weights_file}")
        try:
            model = OnnxUnetPredictor(weights_file)
        except ImportError as e:
            print(f"❌ {e}")
            return 1
    else:
        model = AdiposeUNet()
        model.build_model()
        model.load_weights(weights_file)
    
    # Load normalization stats
    mean, std = load_normalization_stats(checkpoint_dir)
    
    # Initialize TTA if requested
    tta = None
    if args.use_tta:
        tta = TestTimeAugmentation(mode=args.tta_mode)
        print(f"✓ TTA enabled: {args.tta_mode} mode ({len(tta.transforms)} augmentations)")
    
    # Color mapping
    color_map = {
        'cyan': (0, 255, 255),
        'yellow': (255, 255, 0),
        'magenta': (255, 0, 255),
        'green': (0, 255, 0),
        'red': (255, 0, 0),
    }
    overlay_color = color_map[args.overlay_color]
    
    # Find images
    image_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.suffix.lower() in image_exts and f.is_file()]
    
    if not image_files:
        print(f"❌ Error: No images found in {images_dir}")
        print(f"   Looking for: {image_exts}")
        return 1
    
    print(f"\nFound {len(image_files)} images")
    print(f"Processing...\n")
    
    # Process images
    start_time = time.time()
    
    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"⚠️  Warning: Failed to load {img_path.name}, skipping")
            continue
        
        # Check size
        if image.shape != (1024, 1024):
            print(f"⚠️  Warning: {img_path.name} is {image.shape}, expected (1024, 1024), skipping")
            continue
        
        image = image.astype(np.float32)
        
        # Run inference
        if tta is not None:
            prediction = tta.predict_with_tta(model, image, mean, std)
        else:
            prediction = model.predict_single(image, mean, std)
        
        # Save probability map if requested
        if args.save_probability:
            prob_path = prob_dir / f"{img_path.stem}_prob.tif"
            tiff.imwrite(str(prob_path), (prediction * 255).astype(np.uint8))
        
        # Binarize
        binary_mask = (prediction > args.threshold).astype(np.uint8)
        
        # Save binary mask
        mask_path = masks_dir / f"{img_path.stem}_mask.tif"
        tiff.imwrite(str(mask_path), binary_mask)
        
        # Save overlay if requested
        if args.save_overlays:
            overlay = create_overlay_visualization(image, binary_mask, overlay_color)
            overlay_path = overlays_dir / f"{img_path.stem}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {len(image_files)} images")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(image_files):.2f}s per image)")
    print(f"\nOutput saved to:")
    print(f"  Masks: {masks_dir}")
    if args.save_overlays:
        print(f"  Overlays: {overlays_dir}")
    if args.save_probability:
        print(f"  Probabilities: {prob_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    # GPU setup
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    
    exit(main())
