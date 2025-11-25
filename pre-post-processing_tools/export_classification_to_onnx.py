#!/usr/bin/env python3
"""
Export Adipose Classifier (InceptionV3) TF weights to ONNX.

Example:
    python pre-post-processing_tools/export_classification_to_onnx.py \
        --weights checkpoints/classification/classifier_20251111_164323/weights_best.weights.h5 \
        --output checkpoints/classification/classifier_20251111_164323/weights_best.onnx \
        --opset 17
"""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import tf2onnx


def build_model(dropout_rate: float = 0.4) -> Model:
    """
    Rebuild InceptionV3 classifier architecture matching training.
    
    The training pipeline uses ImageNet pretrained weights for the InceptionV3 base,
    then fine-tunes only the classification head. The saved .weights.h5 file
    contains only the trainable layers (classification head).
    
    Args:
        dropout_rate: Dropout rate for classification head
        
    Returns:
        Keras Model (not compiled, for inference only)
    """
    base = InceptionV3(
        include_top=False,
        weights="imagenet",  # Base model uses ImageNet weights (matches training)
        input_shape=(299, 299, 3)
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation="sigmoid", name="adipose_score")(x)
    
    model = Model(inputs=base.input, outputs=out)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Adipose Classifier weights to ONNX")
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to the .weights.h5 file produced by training",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .onnx file",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate (must match training, default: 0.4)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13 for TensorRT compatibility)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fixed batch size for TensorRT (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        raise SystemExit(f"❌ Weights file not found: {args.weights}")

    print(f"🔹 Building InceptionV3 classifier model (dropout={args.dropout})...")
    keras_model = build_model(dropout_rate=args.dropout)

    print(f"🔹 Loading fine-tuned weights from {args.weights}")
    print(f"    Note: Only trainable layers (classification head) will be loaded.")
    print(f"    InceptionV3 base uses ImageNet weights (as in training).")
    keras_model.load_weights(str(args.weights), by_name=True, skip_mismatch=True)

    print(f"🔹 Converting to ONNX (opset {args.opset}, batch_size={args.batch_size})...")
    
    # Create input signature with fixed batch size for TensorRT
    input_signature = [tf.TensorSpec(
        shape=(args.batch_size, 299, 299, 3),
        dtype=tf.float32,
        name='input_1'
    )]
    
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=args.opset,
        custom_ops=None,
        custom_op_handlers=None,
        custom_rewriter=None,
        inputs_as_nchw=['input_1']  # TensorRT prefers NCHW format
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ Saved ONNX model to {args.output}")
    print(f"\nModel Info:")
    print(f"  Input shape:  ({args.batch_size}, 299, 299, 3) - NHWC")
    print(f"  Output shape: ({args.batch_size}, 1) - adipose probability [0-1]")
    print(f"  Dropout:      {args.dropout}")
    print(f"  Opset:        {args.opset} (TensorRT compatible)")
    print(f"\n💡 For TensorRT inference:")
    print(f"  - Batch size is fixed at {args.batch_size}")
    print(f"  - Use opset 11-13 for best compatibility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
