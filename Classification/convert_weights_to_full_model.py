#!/usr/bin/env python3
"""
Convert weights-only (.weights.h5) file to full model (.h5) file.

This script rebuilds the model architecture and loads the weights,
then saves as a complete model file that can be loaded with load_model().

USAGE:
    python Classification/convert_weights_to_full_model.py \
      --weights checkpoints/classification/*/weights_best.weights.h5 \
      --output classifier_full_model.h5 \
      --dropout 0.4

OUTPUT:
    A complete .h5 file containing both architecture and weights.
    Can be loaded with: tf.keras.models.load_model('classifier_full_model.h5')
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed_utils import get_project_seed

GLOBAL_SEED = get_project_seed()
tf.random.set_seed(GLOBAL_SEED)


def build_model(dropout_rate: float = 0.4) -> Model:
    """Build InceptionV3 classifier architecture."""
    base = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3)
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation="sigmoid", name="adipose_score")(x)
    
    model = Model(inputs=base.input, outputs=out)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Convert .weights.h5 to full .h5 model file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to weights-only file (.weights.h5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for full model file (.h5)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate (must match training, default: 0.4)"
    )
    
    args = parser.parse_args()
    
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print("=" * 80)
    print("Convert Weights-Only to Full Model")
    print("=" * 80)
    print(f"Input weights:  {weights_path}")
    print(f"Output model:   {output_path}")
    print(f"Dropout:        {args.dropout}")
    print("=" * 80)
    
    # Build model
    print("\n[1/3] Building model architecture...")
    model = build_model(dropout_rate=args.dropout)
    print(f"✓ Model built with {len(model.layers)} layers")
    
    # Load weights
    print(f"\n[2/3] Loading weights from {weights_path.name}...")
    try:
        model.load_weights(str(weights_path))
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("\nPossible issues:")
        print("  - Weights file is corrupted")
        print("  - Dropout rate doesn't match training")
        print("  - TensorFlow version mismatch")
        raise
    
    # Save as full model
    print(f"\n[3/3] Saving full model to {output_path}...")
    model.save(str(output_path), save_format='h5')
    print("✓ Full model saved successfully")
    
    # Verify
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 80}")
    print("Conversion Complete!")
    print(f"{'=' * 80}")
    print(f"Output file: {output_path}")
    print(f"File size:   {file_size_mb:.1f} MB")
    print(f"\nYou can now load this model with:")
    print(f"  model = tf.keras.models.load_model('{output_path}')")
    print(f"  predictions = model.predict(images)")


if __name__ == "__main__":
    main()
