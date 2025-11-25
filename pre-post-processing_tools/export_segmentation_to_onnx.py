#!/usr/bin/env python3
"""
Export Adipose U-Net TF weights to ONNX.

Example:
    python tools/export_weights_to_onnx.py \
        --weights checkpoints/.../weights_best_overall.weights.h5 \
        --output checkpoints/.../weights_best_overall.onnx \
        --opset 17
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
import tf2onnx

# Ensure project root is on sys.path so segmentation_inference can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact model definition from segmentation_inference.py
from segmentation_inference import AdiposeUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Adipose U-Net weights to ONNX")
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
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        raise SystemExit(f"❌ Weights file not found: {args.weights}")

    model_wrapper = AdiposeUNet()
    keras_model = model_wrapper.build_model()

    print(f"🔹 Loading weights from {args.weights}")
    keras_model.load_weights(str(args.weights))

    print(f"🔹 Converting to ONNX (opset {args.opset})...")
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, opset=args.opset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ Saved ONNX model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
