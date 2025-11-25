#!/usr/bin/env python3
"""
Evaluation script for the adipose tile classifier.

Loads a trained checkpoint, runs inference on the test split of a Keras-style dataset,
optionally applies geometric TTA, and reports publication-ready metrics (ROC/PR AUC,
confusion matrices at default + optimized thresholds, F1/precision/recall, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tifffile
from sklearn import metrics
from sklearn.calibration import calibration_curve
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed_utils import get_project_seed

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not available; --save-plots will be disabled.")

GLOBAL_SEED = get_project_seed()
tf.random.set_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

TTA_MODES: Dict[str, Sequence[int]] = {
    "none": (0,),
    "basic": (0, 1, 2, 3),           # rotations 0/90/180/270
    "full": tuple(range(8)),         # rotations + horizontal flips
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate adipose tile classifier")
    
    # Test dataset selection
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Direct path to test dataset directory containing adipose/ and not_adipose/ folders")
    
    # Model weights
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to classifier weights (.weights.h5).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", type=str, default="none", choices=list(TTA_MODES.keys()),
                        help="Geometric TTA mode (averages predictions across transforms).")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate used when building the classification head (must match training).")
    parser.add_argument("--output-dir", type=str, default="eval_outputs",
                        help="Directory to store metrics + CSV outputs.")
    parser.add_argument("--calibration", type=str, default=None, choices=["temperature", "platt", "isotonic"],
                        help="Optional probability calibration (fit on a validation split).")
    parser.add_argument("--calibration-val-root", type=str, default=None,
                        help="Path to dataset root containing the validation split used for calibration.")
    parser.add_argument("--calibration-val-split", type=str, default="val",
                        help="Split name under calibration-val-root to use (default: val).")
    parser.add_argument("--snapshot", action="append", default=[],
                        help="Additional weight files to ensemble with --weights. Repeat flag for multiple files.")
    parser.add_argument("--slide-map", type=str, default=None,
                        help="Optional CSV with columns tile,slide_id to aggregate probabilities per slide.")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Generate visualization plots (ROC, PR, calibration, histograms, confusion matrix). Enabled by default.")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false",
                        help="Disable plot generation")
    parser.add_argument("--save-examples", action="store_true", default=True,
                        help="Save example images for TP/TN/FP/FN categories with preprocessing applied. Enabled by default.")
    parser.add_argument("--no-examples", dest="save_examples", action="store_false",
                        help="Disable example image generation")
    parser.add_argument("--num-examples", type=int, default=10,
                        help="Number of example images to save per category (default: 10).")
    parser.add_argument("--percentile-norm-examples", type=lambda x: str(x).lower() == 'true', default=True,
                        help="Apply percentile normalization to example images for better visualization (default: true).")
    parser.add_argument("--percentile-norm", type=lambda x: str(x).lower() == 'true', required=True,
                        help="Apply percentile normalization to enhance contrast. Required: true or false.")
    parser.add_argument("--percentile-low", type=float, default=1.0,
                        help="Lower percentile for percentile normalization (default: 1.0).")
    parser.add_argument("--percentile-high", type=float, default=99.0,
                        help="Upper percentile for percentile normalization (default: 99.0).")
    return parser.parse_args()


def list_files(split_dir: Path) -> Tuple[List[str], np.ndarray]:
    pos_dir = split_dir / "adipose"
    neg_dir = split_dir / "not_adipose"
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(f"Expected directories {pos_dir} and {neg_dir}.")

    files: List[str] = []
    labels: List[int] = []

    # Support multiple image formats
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    
    for ext in extensions:
        for path in sorted(pos_dir.glob(ext)):
            files.append(str(path))
            labels.append(1)
    
    for ext in extensions:
        for path in sorted(neg_dir.glob(ext)):
            files.append(str(path))
            labels.append(0)

    if not files:
        raise RuntimeError(f"No image files found under {split_dir}.")

    return files, np.asarray(labels, dtype=np.int32)


def apply_tta_transform(image: np.ndarray, transform_id: int) -> np.ndarray:
    """Deterministic geometric transforms matching TTA pipeline."""
    if transform_id in (0, 1, 2, 3):
        k = transform_id
        if k:
            image = np.rot90(image, k)
    else:
        # flip + rotation combos
        image = np.fliplr(image)
        if transform_id == 5:
            image = np.rot90(image, 1)
        elif transform_id == 6:
            image = np.rot90(image, 2)
        elif transform_id == 7:
            image = np.rot90(image, 3)
    return image


def percentile_normalize_numpy(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Apply percentile normalization to enhance contrast."""
    img = img.astype(np.float32)
    plow, phigh = np.percentile(img, (p_low, p_high))
    scale = max(phigh - plow, 1e-3)
    # Normalize to [0, 1] then scale to [0, 255] for InceptionV3 preprocessing
    normalized = np.clip((img - plow) / scale, 0, 1)
    return (normalized * 255.0).astype(np.float32)


def preprocess_image(path: tf.Tensor, transform_id: int, percentile_norm: bool, p_low: float, p_high: float) -> tf.Tensor:
    """Load and preprocess image with support for JPEG, PNG, and TIFF formats."""
    
    def _load_and_process_all(path_bytes, transform_id, percentile_norm, p_low, p_high):
        """Single Python function to handle all preprocessing to avoid nested py_function calls."""
        import cv2
        from PIL import Image
        
        path_str = path_bytes.numpy().decode('utf-8')
        file_ext = Path(path_str).suffix.lower()
        
        # Load image based on extension
        if file_ext in ['.tif', '.tiff']:
            # Use tifffile for TIFF files
            try:
                img = tifffile.imread(path_str)
                # Normalize if 16-bit
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
            except Exception as e:
                print(f"[WARN] tifffile failed for {path_str}, using PIL: {e}")
                with Image.open(path_str) as pil_img:
                    img = np.array(pil_img)
        else:
            # Use PIL for PNG, JPEG
            with Image.open(path_str) as pil_img:
                img = np.array(pil_img)
        
        # Convert to grayscale if needed
        if img.ndim == 2:
            # Already grayscale
            pass
        elif img.ndim == 3 and img.shape[2] == 3:
            # RGB to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            # RGBA to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        img = img.astype(np.float32)
        
        # Apply percentile normalization if enabled
        if percentile_norm:
            plow, phigh = np.percentile(img, (p_low, p_high))
            scale = max(phigh - plow, 1e-3)
            # Normalize to [0, 1] then scale to [0, 255] for InceptionV3 preprocessing
            img = np.clip((img - plow) / scale, 0, 1)
            img = (img * 255.0).astype(np.float32)
        
        # Apply TTA transform if needed
        if transform_id != 0:
            img = apply_tta_transform(img, transform_id)
        
        return img.astype(np.float32)
    
    # Single py_function call for all preprocessing
    image = tf.py_function(
        func=lambda p: _load_and_process_all(p, transform_id, percentile_norm, p_low, p_high),
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
    return image


def make_dataset(files: List[str], labels: np.ndarray, batch_size: int, transform_id: int,
                 percentile_norm: bool, p_low: float, p_high: float) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(
        lambda p, y: (preprocess_image(p, transform_id, percentile_norm, p_low, p_high), tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(dropout_rate: float) -> Model:
    base = tf.keras.applications.InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation="sigmoid", name="adipose_score")(x)
    return Model(inputs=base.input, outputs=out)


def load_weights(model: Model, weights_path: str):
    model.load_weights(weights_path)
    print(f"[Eval] Loaded weights from {weights_path}")


def predict_with_tta(model: Model, files: List[str], labels: np.ndarray, batch_size: int, transform_ids: Sequence[int],
                     percentile_norm: bool, p_low: float, p_high: float) -> np.ndarray:
    agg = np.zeros(len(files), dtype=np.float32)
    for idx, t_id in enumerate(transform_ids):
        print(f"[Eval] Running TTA pass {idx + 1}/{len(transform_ids)} (transform_id={t_id})")
        ds = make_dataset(files, labels, batch_size, transform_id=t_id,
                         percentile_norm=percentile_norm, p_low=p_low, p_high=p_high)
        preds = model.predict(ds, verbose=1).squeeze()
        agg += preds
    agg /= len(transform_ids)
    return agg


def average_snapshots(primary: np.ndarray, extra_models: List[Model], files: List[str], labels: np.ndarray,
                      batch_size: int, transform_ids: Sequence[int],
                      percentile_norm: bool, p_low: float, p_high: float) -> np.ndarray:
    if not extra_models:
        return primary
    logits = [np.log(primary / np.clip(1 - primary, 1e-7, 1))]
    for idx, m in enumerate(extra_models):
        print(f"[Eval] Snapshot ensemble member {idx + 1}")
        probs = predict_with_tta(m, files, labels, batch_size, transform_ids,
                               percentile_norm, p_low, p_high)
        logits.append(np.log(probs / np.clip(1 - probs, 1e-7, 1)))
    mean_logits = np.mean(logits, axis=0)
    return 1.0 / (1.0 + np.exp(-mean_logits))


def fit_calibrator(probs: np.ndarray, labels: np.ndarray, method: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    if method == "temperature":
        logits = np.log(probs / np.clip(1 - probs, 1e-7, 1))
        clf = LogisticRegression()
        clf.fit(logits.reshape(-1, 1), labels)
        info = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
        return ("temperature", clf, info)
    if method == "platt":
        clf = LogisticRegression()
        clf.fit(probs.reshape(-1, 1), labels)
        info = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
        return ("platt", clf, info)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    info = {"y_min": float(np.min(iso.transform(probs))), "y_max": float(np.max(iso.transform(probs)))}
    return ("isotonic", iso, info)


def apply_calibrator(probs: np.ndarray, calibrator):
    if calibrator is None:
        return probs
    method, model, _ = calibrator
    if method == "temperature":
        logits = np.log(probs / np.clip(1 - probs, 1e-7, 1))
        return model.predict_proba(logits.reshape(-1, 1))[:, 1]
    if method == "platt":
        return model.predict_proba(probs.reshape(-1, 1))[:, 1]
    # isotonic
    return model.transform(probs)


def evaluate_predictions(labels: np.ndarray, probs: np.ndarray) -> Dict:
    metrics_dict: Dict = {}
    metrics_dict["roc_auc"] = float(metrics.roc_auc_score(labels, probs))
    metrics_dict["pr_auc"] = float(metrics.average_precision_score(labels, probs))

    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1 = -1.0
    best_thresh = 0.5
    per_thresh = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision = metrics.precision_score(labels, preds, zero_division=0)
        recall = metrics.recall_score(labels, preds, zero_division=0)
        f1 = metrics.f1_score(labels, preds, zero_division=0)
        per_thresh.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    default_preds = (probs >= 0.5).astype(int)
    best_preds = (probs >= best_thresh).astype(int)

    def summarize(name: str, preds: np.ndarray) -> Dict:
        cm = metrics.confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        return {
            "threshold": float(name),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "precision": float(metrics.precision_score(labels, preds, zero_division=0)),
            "recall": float(metrics.recall_score(labels, preds, zero_division=0)),
            "f1": float(metrics.f1_score(labels, preds, zero_division=0)),
            "specificity": float(tn / (tn + fp + 1e-7)),
        }

    metrics_dict["threshold_metrics"] = {
        "default_0.5": summarize("0.5", default_preds),
        "best_f1": summarize(f"{best_thresh:.2f}", best_preds),
        "per_threshold": per_thresh,
    }

    return metrics_dict


def compute_class_statistics(labels: np.ndarray, probs: np.ndarray) -> Dict:
    """Compute per-class probability statistics."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    stats = {
        "adipose": {
            "count": int(pos_mask.sum()),
            "mean_prob": float(probs[pos_mask].mean()) if pos_mask.any() else 0.0,
            "std_prob": float(probs[pos_mask].std()) if pos_mask.any() else 0.0,
            "median_prob": float(np.median(probs[pos_mask])) if pos_mask.any() else 0.0,
            "min_prob": float(probs[pos_mask].min()) if pos_mask.any() else 0.0,
            "max_prob": float(probs[pos_mask].max()) if pos_mask.any() else 0.0,
        },
        "not_adipose": {
            "count": int(neg_mask.sum()),
            "mean_prob": float(probs[neg_mask].mean()) if neg_mask.any() else 0.0,
            "std_prob": float(probs[neg_mask].std()) if neg_mask.any() else 0.0,
            "median_prob": float(np.median(probs[neg_mask])) if neg_mask.any() else 0.0,
            "min_prob": float(probs[neg_mask].min()) if neg_mask.any() else 0.0,
            "max_prob": float(probs[neg_mask].max()) if neg_mask.any() else 0.0,
        }
    }
    return stats


def plot_confusion_matrix(labels: np.ndarray, probs: np.ndarray, threshold: float, output_dir: Path):
    """Generate confusion matrix heatmap"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        import seaborn as sns
    except ImportError:
        print("[WARN] seaborn not available, skipping confusion matrix plot.")
        return
    
    from sklearn.metrics import confusion_matrix
    
    # Create predictions at the given threshold
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    
    # Extract values for annotation
    tn, fp, fn, tp = cm.ravel()
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Adipose', 'Adipose'],
                yticklabels=['Not Adipose', 'Adipose'],
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix (threshold={threshold:.2f})', fontsize=14, fontweight='bold')
    
    # Add text summary
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f} | TP={tp}, TN={tn}, FP={fp}, FN={fn}',
             ha='center', va='top', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_visualizations(labels: np.ndarray, probs: np.ndarray, roc_auc: float, pr_auc: float, output_dir: Path):
    """Generate ROC, PR, calibration, probability histogram, and confusion matrix plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available, skipping plots.")
        return
    
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR (AUC={pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calibration Plot
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Probability', fontsize=12)
    plt.title('Calibration Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Probability Histograms by Class
    plt.figure(figsize=(10, 5))
    plt.hist(probs[labels == 1], bins=50, alpha=0.7, label='Adipose', color='#e74c3c', edgecolor='black')
    plt.hist(probs[labels == 0], bins=50, alpha=0.7, label='Not Adipose', color='#3498db', edgecolor='black')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'prob_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix (at default threshold of 0.5)
    plot_confusion_matrix(labels, probs, threshold=0.5, output_dir=output_dir)
    
    print(f"[Plots] Saved 5 visualization plots to {output_dir}")


def aggregate_by_slide(files: List[str], probs: np.ndarray, slide_map_csv: str | None) -> Dict | None:
    if not slide_map_csv:
        return None
    import csv
    stem_to_slide = {}
    with open(slide_map_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem_to_slide[row["tile"]] = row["slide_id"]
    slide_scores: Dict[str, List[float]] = {}
    for path, prob in zip(files, probs):
        stem = Path(path).stem
        slide = stem_to_slide.get(stem)
        if not slide:
            continue
        slide_scores.setdefault(slide, []).append(float(prob))
    slide_summary = {}
    for slide, vals in slide_scores.items():
        arr = np.asarray(vals, dtype=np.float32)
        slide_summary[slide] = {
            "mean_prob": float(arr.mean()),
            "median_prob": float(np.median(arr)),
            "max_prob": float(arr.max()),
            "tile_count": len(vals),
        }
    return slide_summary


def save_example_images(output_dir: Path, files: List[str], labels: np.ndarray, probs: np.ndarray, 
                        threshold: float = 0.5, num_examples: int = 10,
                        percentile_norm_examples: bool = True, inference_percentile_norm: bool = False,
                        p_low: float = 1.0, p_high: float = 99.0):
    """
    Save example images for each prediction category (TP/TN/FP/FN) with preprocessing applied.
    
    Args:
        output_dir: Directory to save examples
        files: List of image file paths
        labels: True labels
        probs: Predicted probabilities
        threshold: Classification threshold (default: 0.5)
        num_examples: Number of examples per category
        percentile_norm_examples: Whether to apply percentile normalization to examples
        inference_percentile_norm: Whether inference used percentile normalization
        p_low: Lower percentile for normalization
        p_high: Upper percentile for normalization
        
    Note: If percentile_norm_examples=True, examples will be normalized regardless of inference settings.
          This allows visualization of normalized images even when model was trained/tested without normalization.
    """
    import cv2
    from PIL import Image
    
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions at threshold
    preds = (probs >= threshold).astype(int)
    
    # Categorize predictions
    tp_idx = np.where((preds == 1) & (labels == 1))[0]
    tn_idx = np.where((preds == 0) & (labels == 0))[0]
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]
    
    categories = {
        "true_positive": tp_idx,
        "true_negative": tn_idx,
        "false_positive": fp_idx,
        "false_negative": fn_idx
    }
    
    rng = np.random.default_rng(GLOBAL_SEED)
    
    for category, indices in categories.items():
        if len(indices) == 0:
            print(f"[Examples] No {category} examples found")
            continue
            
        category_dir = examples_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Randomly sample examples
        n_samples = min(num_examples, len(indices))
        sampled_idx = rng.choice(indices, size=n_samples, replace=False)
        
        for i, idx in enumerate(sampled_idx):
            file_path = files[idx]
            file_ext = Path(file_path).suffix.lower()
            
            # Load and preprocess image (same as inference)
            if file_ext in ['.tif', '.tiff']:
                try:
                    img = tifffile.imread(file_path)
                    if img.dtype == np.uint16:
                        img = (img / 256).astype(np.uint8)
                except Exception:
                    with Image.open(file_path) as pil_img:
                        img = np.array(pil_img)
            else:
                with Image.open(file_path) as pil_img:
                    img = np.array(pil_img)
            
            # Handle grayscale and convert to float32
            if img.ndim == 2:
                img_gray = img.astype(np.float32)
            elif img.ndim == 3 and img.shape[2] == 1:
                img_gray = img.squeeze().astype(np.float32)
            elif img.ndim == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                img_gray = img.astype(np.float32)
            
            # Apply percentile normalization if requested for examples
            # (independent of whether inference used it - we're loading raw images)
            if percentile_norm_examples:
                plow, phigh = np.percentile(img_gray, (p_low, p_high))
                scale = max(phigh - plow, 1e-3)
                img_gray = np.clip((img_gray - plow) / scale, 0, 1)
                img_gray = (img_gray * 255.0).astype(np.uint8)
            else:
                img_gray = img_gray.astype(np.uint8)
            
            # Convert to RGB
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            
            # Resize to 299x299 (InceptionV3 input size)
            img_resized = cv2.resize(img, (299, 299), interpolation=cv2.IMWRITE_JPEG_OPTIMIZE)
            
            # Apply InceptionV3 preprocessing (scale to [-1, 1])
            img_preprocessed = preprocess_input(img_resized.astype(np.float32))
            
            # Convert back to displayable range [0, 255]
            img_display = ((img_preprocessed + 1.0) * 127.5).astype(np.uint8)
            
            # Save with metadata in filename
            prob = probs[idx]
            label = labels[idx]
            filename = f"{i:03d}_prob{prob:.3f}_label{label}.jpg"
            save_path = category_dir / filename
            
            # Convert RGB to BGR for cv2.imwrite
            img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"[Examples] Saved {n_samples} {category} examples to {category_dir}")
    
    print(f"[Examples] All examples saved to {examples_dir}")


def dump_outputs(output_dir: Path, files: List[str], labels: np.ndarray, probs: np.ndarray, metrics_dict: Dict, slide_summary: Dict | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w") as f:
        f.write("path,label,prob\n")
        for path, label, prob in zip(files, labels, probs):
            f.write(f"{path},{label},{prob:.6f}\n")
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    if slide_summary:
        slide_path = output_dir / "slide_summary.json"
        with open(slide_path, "w") as f:
            json.dump(slide_summary, f, indent=2)
        print(f"[Eval] Wrote slide summary to {slide_path}")
    print(f"[Eval] Wrote predictions to {csv_path}")
    print(f"[Eval] Wrote metrics to {metrics_path}")


def main():
    args = parse_args()
    
    # Auto-set output directory with structured naming
    if args.output_dir == "eval_outputs":
        checkpoint_dir = Path(args.weights).parent
        
        # Extract test folder name from path
        test_dir_path = Path(args.test_dir)
        test_folder_name = test_dir_path.name
        
        # Add enhancement suffixes
        enhancement_suffixes = []
        if args.tta != "none":
            enhancement_suffixes.append(f"tta_{args.tta}")
        if args.percentile_norm:
            enhancement_suffixes.append("percentile")
        
        # Build final folder name
        if enhancement_suffixes:
            folder_name = f"{test_folder_name}_{'_'.join(enhancement_suffixes)}"
        else:
            folder_name = test_folder_name
        
        # Set structured output directory
        args.output_dir = str(checkpoint_dir / "evaluation" / folder_name)
        print(f"[Output] Structured output directory: {args.output_dir}")
        print(f"[Output]   Dataset: {test_folder_name}")
        if enhancement_suffixes:
            print(f"[Output]   Enhancements: {', '.join(enhancement_suffixes)}")

    # Use direct test directory path
    split_dir = Path(args.test_dir)
    
    # Validate path exists
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {split_dir}\n"
            f"Please ensure the directory exists and has been built with build_test_class_dataset.py"
        )
    
    # Validate dataset structure (adipose and not_adipose folders)
    if not (split_dir / "adipose").exists() or not (split_dir / "not_adipose").exists():
        raise FileNotFoundError(
            f"Invalid dataset structure at {split_dir}\n"
            f"Expected folders: adipose/ and not_adipose/"
        )
    
    print(f"[Dataset] Using test set: {split_dir}")

    files, labels = list_files(split_dir)
    print(f"[Eval] Loaded {len(files)} tiles ({labels.sum()} positive / {(labels == 0).sum()} negative)")
    print(f"[Normalization] Percentile normalization: {'ENABLED' if args.percentile_norm else 'DISABLED'}")
    if args.percentile_norm:
        print(f"[Normalization]   Percentile range: {args.percentile_low}-{args.percentile_high}")

    model = build_model(dropout_rate=args.dropout)
    load_weights(model, args.weights)

    extra_models: List[Model] = []
    for snap_path in args.snapshot:
        snap_model = build_model(dropout_rate=args.dropout)
        load_weights(snap_model, snap_path)
        extra_models.append(snap_model)

    calibrator = None
    calibration_info = {}
    if args.calibration:
        if not args.calibration_val_root:
            raise SystemExit("--calibration-val-root is required when --calibration is set.")
        val_root = Path(args.calibration_val_root)
        val_split_dir = val_root / args.calibration_val_split
        val_files, val_labels = list_files(val_split_dir)
        print(f"[Calibration] Using {len(val_files)} tiles from {val_split_dir} for calibration ({args.calibration}).")
        val_base_probs = predict_with_tta(model, val_files, val_labels, args.batch_size, TTA_MODES[args.tta],
                                         args.percentile_norm, args.percentile_low, args.percentile_high)
        val_probs = average_snapshots(val_base_probs, extra_models, val_files, val_labels, args.batch_size, TTA_MODES[args.tta],
                                     args.percentile_norm, args.percentile_low, args.percentile_high)
        calibrator = fit_calibrator(val_probs, val_labels, args.calibration)
        calibration_info = {"method": args.calibration}
        calibration_info.update(calibrator[2])
        val_probs = apply_calibrator(val_probs, calibrator)
        calibration_info["val_calibrated_auc"] = float(metrics.roc_auc_score(val_labels, val_probs))
        calibration_info["val_calibrated_pr_auc"] = float(metrics.average_precision_score(val_labels, val_probs))

    tta_ids = TTA_MODES[args.tta]
    base_probs = predict_with_tta(model, files, labels, args.batch_size, tta_ids,
                                  args.percentile_norm, args.percentile_low, args.percentile_high)
    probs = average_snapshots(base_probs, extra_models, files, labels, args.batch_size, tta_ids,
                             args.percentile_norm, args.percentile_low, args.percentile_high)
    probs = apply_calibrator(probs, calibrator)

    metrics_dict = evaluate_predictions(labels, probs)
    if calibration_info:
        metrics_dict["calibration"] = calibration_info
    
    # Compute per-class statistics
    class_stats = compute_class_statistics(labels, probs)
    metrics_dict["class_statistics"] = class_stats

    print("\n[Metrics]")
    print(f"ROC AUC: {metrics_dict['roc_auc']:.4f}")
    print(f"PR  AUC: {metrics_dict['pr_auc']:.4f}")
    for name, summary in metrics_dict["threshold_metrics"].items():
        if name == "per_threshold":
            continue
        cm = summary["confusion_matrix"]
        print(f"{name} -> threshold {summary['threshold']}: precision={summary['precision']:.3f} recall={summary['recall']:.3f} "
              f"f1={summary['f1']:.3f} (TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']})")
    
    print("\n[Per-Class Statistics]")
    for class_name, stats in class_stats.items():
        print(f"{class_name.capitalize()}:")
        print(f"  Count:  {stats['count']:6d}")
        print(f"  Mean:   {stats['mean_prob']:.4f}")
        print(f"  Median: {stats['median_prob']:.4f}")
        print(f"  Std:    {stats['std_prob']:.4f}")
        print(f"  Range:  [{stats['min_prob']:.4f}, {stats['max_prob']:.4f}]")

    slide_summary = aggregate_by_slide(files, probs, args.slide_map)
    output_dir = Path(args.output_dir)
    dump_outputs(output_dir, files, labels, probs, metrics_dict, slide_summary=slide_summary)
    
    # Generate visualization plots if requested
    if args.save_plots:
        plot_visualizations(labels, probs, metrics_dict['roc_auc'], metrics_dict['pr_auc'], output_dir)
    
    # Save example images if requested
    if args.save_examples:
        save_example_images(
            files=files,
            labels=labels,
            probs=probs,
            output_dir=output_dir,
            num_examples=args.num_examples,
            percentile_norm_examples=args.percentile_norm_examples,
            inference_percentile_norm=args.percentile_norm,
            p_low=args.percentile_low,
            p_high=args.percentile_high
        )

    print("\n[Post-processing options]")
    print("- Use --save-plots to generate ROC/PR/calibration curves and probability histograms.")
    print("- Use --calibration temperature/platt/isotonic to calibrate probabilities (fit on validation set).")
    print("- Provide --snapshot paths to ensemble multiple checkpoints offline.")
    print("- Supply --slide-map CSV to produce slide-level adipose probabilities.")


if __name__ == "__main__":
    main()
