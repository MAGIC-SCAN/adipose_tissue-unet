#!/usr/bin/env python3
"""
Tile-level adipose classifier inference.

Matches the preprocessing pipeline from train_adipose_classifier_v0.py:
- Read RGB JPEG → (optionally convert to grayscale) → Resize to 299×299 
- (optionally tile to 3 channels if grayscale) → preprocess_input → predict

Supports both RGB and grayscale modes to test which performs better.

IMPORTANT - WEIGHTS FILE:
  This script requires weights-only files (.weights.h5) from training.
  The model architecture is rebuilt programmatically, then weights are loaded.
  
  Expected checkpoint structure:
    checkpoints/classifier_YYYYMMDD_HHMMSS/
    ├── best.weights.h5          # Use this file
    ├── weights_best.weights.h5  # Or this file
    └── training_settings.json   # Contains dropout rate (default: 0.4)

USAGE EXAMPLES:

1. Batch inference on directory (grayscale mode, no percentile norm):
   python Classification/classification_inference.py \
     --weights checkpoints/classification/*/best.weights.h5 \
     --input-dir data/Meat_MS_Tulane/ECM_channel \
     --output-csv predictions.csv \
     --mode grayscale \
     --percentile-norm false

2. Single tile inference with TTA:
   python Classification/classification_inference.py \
     --weights checkpoints/classification/*/best.weights.h5 \
     --input-tile path/to/tile.jpg \
     --mode grayscale \
     --percentile-norm false \
     --tta-mode full

3. RGB mode with percentile normalization:
   python Classification/classification_inference.py \
     --weights checkpoints/classification/*_percentile/best.weights.h5 \
     --input-dir data/tiles/ \
     --output-csv predictions.csv \
     --mode rgb \
     --percentile-norm true \
     --percentile-low 1.0 \
     --percentile-high 99.0

4. ONNX inference (faster, requires ONNX export):
   python Classification/classification_inference.py \
     --onnx-model model.onnx \
     --input-dir data/tiles/ \
     --output-csv predictions.csv \
     --mode grayscale \
     --percentile-norm false

OUTPUT CSV FORMAT:
  path,probability,prediction
  tile_001.jpg,0.8523,1
  tile_002.jpg,0.1234,0
  ...

NOTES:
  - Always match --percentile-norm to training settings
  - Use --mode grayscale for ECM channel data
  - Use --mode rgb for pseudocolored data
  - TTA modes: none, basic (4x), full (8x)
  - Match --dropout to training settings (default: 0.4)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tqdm import tqdm

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed_utils import get_project_seed

try:
    import onnxruntime as ort  # Optional dependency for ONNX inference
except ImportError:
    ort = None

GLOBAL_SEED = get_project_seed()
tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adipose classifier inference with RGB or grayscale preprocessing"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.weights.h5 or .h5)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing RGB JPEG tiles to classify."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="classification_outputs",
        help="Directory to save predictions CSV and optional visualizations."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.jpg",
        help="Glob pattern to match image files (default: **/*.jpg for recursive search)."
    )
    parser.add_argument(
        "--use-rgb",
        action="store_false",
        dest="use_grayscale",
        default=True,
        help="Use RGB images directly (matches legacy classifier and may perform better)."
    )
    parser.add_argument(
        "--use-grayscale",
        action="store_true",
        dest="use_grayscale",
        help="Convert to grayscale before prediction (matches current training, default: True)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for binary predictions (default: 0.5)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate (must match training, default: 0.4)."
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        default=False,
        help="Enable test-time augmentation (8x predictions averaged)."
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        default="basic",
        choices=["basic", "full"],
        help="TTA mode: basic (4x rotations) or full (8x rotations + flips)."
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        default=False,
        help="Save overlay visualizations for positive predictions."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID (default: 0)."
    )
    return parser.parse_args()


class OnnxClassifierPredictor:
    """ONNX Runtime-based predictor for exported InceptionV3 classifier"""
    
    def __init__(self, onnx_path: str):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with `pip install onnxruntime`."
            )
        self.session = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Run inference on batch of images.
        
        Args:
            x: Input array of shape (batch, 299, 299, 3)
            verbose: Ignored (for API compatibility)
            
        Returns:
            Predictions of shape (batch, 1)
        """
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        
        # ONNX expects float32
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: x})
        return outputs[0]


def build_model(dropout_rate: float = 0.4) -> Model:
    """
    Rebuild InceptionV3 classifier architecture matching training.
    
    Args:
        dropout_rate: Dropout rate for classification head
        
    Returns:
        Keras Model (not compiled, for inference only)
    """
    base = InceptionV3(
        include_top=False,
        weights="imagenet",  # Not used, just for architecture
        input_shape=(299, 299, 3)
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation="sigmoid", name="adipose_score")(x)
    
    model = Model(inputs=base.input, outputs=out)
    return model


def diagnose_weights_file(weights_path: Path) -> None:
    """
    Diagnose weights file to check if it's valid.
    
    Args:
        weights_path: Path to weights file
    """
    try:
        import h5py
        print(f"\n[Diagnosis] Inspecting weights file: {weights_path}")
        print(f"[Diagnosis] File size: {weights_path.stat().st_size / (1024*1024):.2f} MB")
        
        with h5py.File(str(weights_path), 'r') as f:
            # Check file structure
            print(f"[Diagnosis] HDF5 keys: {list(f.keys())}")
            
            # For weights-only files, expect a layer_names attribute
            if 'layer_names' in f.attrs:
                layer_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                              for name in f.attrs['layer_names']]
                print(f"[Diagnosis] Found {len(layer_names)} layers")
                print(f"[Diagnosis] First few layers: {layer_names[:5]}")
            else:
                print("[Diagnosis] WARNING: 'layer_names' attribute not found")
            
            # Check if this is a full model (has 'model_weights' group)
            if 'model_weights' in f.keys():
                print("[Diagnosis] This appears to be a full model file (.h5)")
                print("[Diagnosis] Suggestion: Use load_model() instead of load_weights()")
            else:
                print("[Diagnosis] This appears to be a weights-only file")
                
    except Exception as e:
        print(f"[Diagnosis] Error inspecting file: {e}")
        print(f"[Diagnosis] The file may be corrupted or in an unexpected format")


def load_and_preprocess_image(
    image_path: Path,
    use_grayscale: bool = True
) -> tf.Tensor:
    """
    Load and preprocess image matching training pipeline.
    
    Args:
        image_path: Path to RGB JPEG tile
        use_grayscale: If True, convert to grayscale and tile to 3 channels
        
    Returns:
        Preprocessed image tensor (299, 299, 3)
    """
    # Read as RGB JPEG (matches training)
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_jpeg(image, channels=3)
    
    if use_grayscale:
        # Convert to grayscale and tile to 3 channels (matches current training)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [299, 299], method="bilinear")
        image = tf.tile(image, [1, 1, 3])  # Replicate to 3 channels for InceptionV3
    else:
        # Use RGB directly (matches legacy classifier)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [299, 299], method="bilinear")
    
    # Apply InceptionV3 preprocessing (scales to [-1, 1])
    image = preprocess_input(image)
    
    return image


def apply_tta_transforms(image: tf.Tensor, mode: str = "basic") -> List[tf.Tensor]:
    """
    Generate test-time augmentation variants.
    
    Args:
        image: Preprocessed image tensor (299, 299, 3)
        mode: "basic" (4x rotations) or "full" (8x with flips)
        
    Returns:
        List of augmented image tensors
    """
    augmented = []
    
    # 4 rotations (0°, 90°, 180°, 270°)
    for k in range(4):
        rotated = tf.image.rot90(image, k=k)
        augmented.append(rotated)
    
    if mode == "full":
        # Add horizontal flips of all rotations
        for k in range(4):
            rotated = tf.image.rot90(image, k=k)
            flipped = tf.image.flip_left_right(rotated)
            augmented.append(flipped)
    
    return augmented


def predict_single_image(
    model: Model,
    image_path: Path,
    use_grayscale: bool = True,
    use_tta: bool = False,
    tta_mode: str = "basic"
) -> Tuple[float, int]:
    """
    Predict adipose probability for a single image.
    
    Args:
        model: Trained classifier model
        image_path: Path to image file
        use_grayscale: Convert to grayscale before prediction
        use_tta: Enable test-time augmentation
        tta_mode: TTA mode ("basic" or "full")
        
    Returns:
        (probability, binary_prediction) tuple
    """
    # Load and preprocess
    image = load_and_preprocess_image(image_path, use_grayscale)
    
    if use_tta:
        # Generate augmented variants
        augmented_images = apply_tta_transforms(image, mode=tta_mode)
        augmented_batch = tf.stack(augmented_images)  # Shape: (N, 299, 299, 3)
        
        # Predict on all variants
        predictions = model.predict(augmented_batch, verbose=0)
        probability = float(np.mean(predictions))
    else:
        # Single prediction
        image_batch = tf.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image_batch, verbose=0)
        probability = float(prediction[0, 0])
    
    return probability


def predict_directory(
    model: Model,
    input_dir: Path,
    pattern: str,
    use_grayscale: bool,
    use_tta: bool,
    tta_mode: str,
    batch_size: int
) -> List[Tuple[Path, float]]:
    """
    Predict on all images in directory.
    
    Args:
        model: Trained classifier model
        input_dir: Directory containing images
        pattern: Glob pattern for image files
        use_grayscale: Convert to grayscale
        use_tta: Enable TTA
        tta_mode: TTA mode
        batch_size: Batch size for inference
        
    Returns:
        List of (image_path, probability) tuples
    """
    # Find all matching images
    image_paths = sorted(input_dir.glob(pattern))
    
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_dir} matching pattern '{pattern}'"
        )
    
    print(f"Found {len(image_paths)} images to classify")
    
    results = []
    
    if use_tta:
        # TTA requires per-image processing
        print(f"Running inference with TTA mode: {tta_mode}")
        for img_path in tqdm(image_paths, desc="Classifying"):
            prob = predict_single_image(
                model, img_path, use_grayscale, use_tta, tta_mode
            )
            results.append((img_path, prob))
    else:
        # Batch processing without TTA
        print(f"Running inference with batch size: {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Classifying"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch
            batch_images = []
            for img_path in batch_paths:
                img = load_and_preprocess_image(img_path, use_grayscale)
                batch_images.append(img)
            
            batch_tensor = tf.stack(batch_images)
            
            # Predict
            predictions = model.predict(batch_tensor, verbose=0)
            
            # Store results
            for img_path, pred in zip(batch_paths, predictions):
                probability = float(pred[0])
                results.append((img_path, probability))
    
    return results


def save_results(
    results: List[Tuple[Path, float]],
    output_dir: Path,
    threshold: float,
    use_grayscale: bool,
    use_tta: bool
) -> None:
    """
    Save predictions to CSV.
    
    Args:
        results: List of (image_path, probability) tuples
        output_dir: Output directory
        threshold: Classification threshold
        use_grayscale: Whether grayscale mode was used
        use_tta: Whether TTA was used
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with settings
    mode_str = "grayscale" if use_grayscale else "rgb"
    tta_str = "_tta" if use_tta else ""
    csv_path = output_dir / f"predictions_{mode_str}{tta_str}.csv"
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path',
            'adipose_probability',
            'binary_prediction',
            'is_adipose'
        ])
        
        for img_path, prob in results:
            binary_pred = 1 if prob >= threshold else 0
            is_adipose = "adipose" if binary_pred == 1 else "not_adipose"
            writer.writerow([
                str(img_path),
                f"{prob:.6f}",
                binary_pred,
                is_adipose
            ])
    
    print(f"\nSaved predictions to: {csv_path}")
    
    # Print summary statistics
    probabilities = [prob for _, prob in results]
    binary_preds = [1 if p >= threshold else 0 for p in probabilities]
    
    n_adipose = sum(binary_preds)
    n_total = len(results)
    pct_adipose = 100 * n_adipose / n_total if n_total > 0 else 0
    
    print(f"\nSummary Statistics:")
    print(f"  Total images:     {n_total}")
    print(f"  Adipose:          {n_adipose} ({pct_adipose:.1f}%)")
    print(f"  Not adipose:      {n_total - n_adipose} ({100-pct_adipose:.1f}%)")
    print(f"  Mean probability: {np.mean(probabilities):.4f}")
    print(f"  Std probability:  {np.std(probabilities):.4f}")
    print(f"  Min probability:  {np.min(probabilities):.4f}")
    print(f"  Max probability:  {np.max(probabilities):.4f}")
    
    # Save summary JSON
    summary = {
        "total_images": n_total,
        "adipose_count": n_adipose,
        "not_adipose_count": n_total - n_adipose,
        "adipose_percentage": pct_adipose,
        "threshold": threshold,
        "mode": mode_str,
        "use_tta": use_tta,
        "statistics": {
            "mean": float(np.mean(probabilities)),
            "std": float(np.std(probabilities)),
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
            "median": float(np.median(probabilities))
        }
    }
    
    summary_path = output_dir / f"summary_{mode_str}{tta_str}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_path}")


def main():
    args = parse_args()
    
    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {gpus}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Validate paths
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir = Path(args.output_dir)
    
    # Print configuration
    print("=" * 80)
    print("Classification Inference Configuration")
    print("=" * 80)
    print(f"Weights:       {weights_path}")
    print(f"Input dir:     {input_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Pattern:       {args.pattern}")
    print(f"Mode:          {'Grayscale (tiled to 3ch)' if args.use_grayscale else 'RGB'}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Threshold:     {args.threshold}")
    print(f"Dropout:       {args.dropout}")
    print(f"TTA:           {args.use_tta} ({args.tta_mode if args.use_tta else 'N/A'})")
    print("=" * 80)
    
    # Detect ONNX model
    is_onnx = str(weights_path).lower().endswith(".onnx")
    
    if is_onnx:
        # Load ONNX model
        print(f"\nDetected ONNX weights: {weights_path}")
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install it with `pip install onnxruntime`."
            )
        print("Loading ONNX model...")
        model = OnnxClassifierPredictor(str(weights_path))
        print("✓ ONNX model loaded successfully")
    else:
        # Build and load Keras model
        print("\nBuilding Keras model...")
        model = build_model(dropout_rate=args.dropout)
        
        # Load weights
        print(f"Loading weights from {weights_path}...")
        
        # Check if weights-only file (.weights.h5) or full model (.h5)
        if '.weights.h5' in str(weights_path) or 'weights_best' in str(weights_path):
            # Weights-only file - use load_weights()
            try:
                model.load_weights(str(weights_path))
                print("✓ Weights loaded successfully (weights-only file)")
            except Exception as e:
                # Run diagnostics
                print(f"\n❌ Error loading weights: {e}")
                diagnose_weights_file(weights_path)
                
                raise ValueError(
                    f"\nFailed to load weights from {weights_path}.\n"
                    f"Error: {e}\n\n"
                    f"Possible causes:\n"
                    f"1. File may be corrupted (incomplete download/transfer)\n"
                    f"2. File was saved with a different TensorFlow/Keras version\n"
                    f"3. File might be a full model (.h5) but named .weights.h5\n"
                    f"4. Model architecture doesn't match (dropout={args.dropout})\n\n"
                    f"Solutions:\n"
                    f"- Verify file integrity (check file size matches source)\n"
                    f"- Re-download or re-copy the weights file\n"
                    f"- Ensure TensorFlow 2.13.x is installed\n"
                    f"- Try using the original checkpoint from training"
                )
        else:
            # Try loading as full model (.h5 or SavedModel)
            try:
                from tensorflow.keras.models import load_model
                model = load_model(str(weights_path), compile=False)
                print("✓ Loaded as full model")
            except Exception as e:
                raise ValueError(
                    f"Failed to load model from {weights_path}.\n"
                    f"Error: {e}\n\n"
                    f"If this is a weights-only file, rename it to include '.weights.h5'"
                )
    
    # Run inference
    print("\nRunning inference...")
    results = predict_directory(
        model=model,
        input_dir=input_dir,
        pattern=args.pattern,
        use_grayscale=args.use_grayscale,
        use_tta=args.use_tta,
        tta_mode=args.tta_mode,
        batch_size=args.batch_size
    )
    
    # Save results
    print("\nSaving results...")
    save_results(
        results=results,
        output_dir=output_dir,
        threshold=args.threshold,
        use_grayscale=args.use_grayscale,
        use_tta=args.use_tta
    )
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    import os
    main()
