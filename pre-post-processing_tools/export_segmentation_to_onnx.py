#!/usr/bin/env python3
"""
Export Adipose U-Net TF weights to ONNX.

Converts TensorFlow .weights.h5 files to ONNX format for deployment on
non-Python platforms or for use with ONNX Runtime for faster inference.

USAGE EXAMPLES:

1. Basic export (opset 17, recommended):
   python pre-post-processing_tools/export_segmentation_to_onnx.py \
     --weights checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5 \
     --output checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.onnx

2. Export with custom opset (older ONNX Runtime versions):
   python pre-post-processing_tools/export_segmentation_to_onnx.py \
     --weights checkpoints/*/weights_best_overall.weights.h5 \
     --output checkpoints/*/weights_best_overall.onnx \
     --opset 15

3. Export phase 1 weights (frozen encoder):
   python pre-post-processing_tools/export_segmentation_to_onnx.py \
     --weights checkpoints/*/phase1_best.weights.h5 \
     --output checkpoints/*/phase1_best.onnx

INPUT FORMAT:
  - Shape: (batch, 1024, 1024, 3)
  - Type: float32
  - Range: Normalized according to training stats (z-score or percentile)

OUTPUT FORMAT:
  - Shape: (batch, 1024, 1024, 1)
  - Type: float32
  - Range: [0, 1] (sigmoid probability per pixel)

NOTE: The exported model uses the exact U-Net architecture from
      segmentation_inference.py (AdiposeUNet class).

VERIFY EXPORT:
  Use ONNX Runtime to test:
  
  import onnxruntime as ort
  import numpy as np
  
  session = ort.InferenceSession('weights_best_overall.onnx')
  dummy_input = np.random.randn(1, 1024, 1024, 3).astype(np.float32)
  output = session.run(None, {'input': dummy_input})
  print(output[0].shape)  # Should be (1, 1024, 1024, 1)
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
