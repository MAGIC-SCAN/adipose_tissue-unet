#!/usr/bin/env python3
"""
Tile-Level Binary Classification Evaluation

Evaluates binary classification performance: "Has Fat" vs "No Fat" per tile.
Uses a configurable fat percentage threshold to determine class labels.

Key Features:
- Binary classification with confusion matrix
- Accuracy, Precision, Recall, F1-Score metrics
- Configurable fat percentage threshold (default: 10%)
- Optional TTA and boundary refinement from full_evaluation_enhanced
- Per-tile results CSV and dataset-level summary
- Threshold sensitivity analysis

Output Structure:
    output_dir/
    ├── per_tile_results.csv           # Individual tile predictions
    ├── summary_metrics.json           # Overall performance metrics
    ├── confusion_matrix.png           # Confusion matrix visualization
    ├── threshold_analysis.csv         # Multi-threshold comparison (optional)
    └── misclassified_tiles.txt        # List of misclassified tiles

Usage:
    # Basic evaluation (10% threshold)
    python tile_classification_evaluation.py \
        --weights checkpoints/best.weights.h5 \
        --data-root /path/to/test_dataset \
        --output-dir classification_results

    # With all enhancements
    python tile_classification_evaluation.py \
        --weights checkpoints/best.weights.h5 \
        --data-root /path/to/test_dataset \
        --output-dir classification_results \
        --use-tta --tta-mode full \
        --boundary-refine \
        --threshold 0.10

    # Threshold sensitivity analysis
    python tile_classification_evaluation.py \
        --weights checkpoints/best.weights.h5 \
        --data-root /path/to/test_dataset \
        --output-dir classification_results \
        --multi-threshold 0.01,0.05,0.10,0.15,0.25
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from full_evaluation_enhanced
from full_evaluation_enhanced import (
    AdiposeUNet, BoundaryRefiner, TestTimeAugmentation,
    load_training_stats, set_deterministic_seeds
)

# TensorFlow setup
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ====================================================================
# OUTPUT DIRECTORY NAMING (matching full_evaluation_enhanced pattern)
# ====================================================================

def extract_dataset_name(data_root: str) -> str:
    """
    Extract dataset name from data root path.
    
    Examples:
        /path/to/clean_test -> clean_test
        /path/to/clean_test_50_overlap -> clean_test_50_overlap
        /path/to/stain_normalized/clean_test -> clean_test
    
    Args:
        data_root: Path to dataset
    
    Returns:
        Dataset name
    """
    data_path = Path(data_root)
    
    # Get the last non-empty directory name
    parts = [p for p in data_path.parts if p]
    
    # Check last part first
    last_part = parts[-1] if parts else "unknown"
    
    # If it's a known dataset name, use it
    known_datasets = ['clean_test', 'clean_test_50_overlap', 'test', 'val', 'human_test']
    if last_part in known_datasets:
        return last_part
    
    # Otherwise check for these patterns in the path
    for dataset in known_datasets:
        if dataset in str(data_path):
            return dataset
    
    # Fallback to last directory name
    return last_part


def detect_data_source(data_root: str) -> str:
    """
    Detect whether data is stain-normalized or original.
    
    Args:
        data_root: Path to dataset
    
    Returns:
        'stain' or 'original'
    """
    data_path_str = str(data_root).lower()
    
    if 'stain' in data_path_str or 'normalized' in data_path_str:
        return 'stain'
    else:
        return 'original'


def build_enhancement_suffixes(args) -> List[str]:
    """
    Build list of enhancement suffixes based on enabled flags.
    
    Matches the pattern from full_evaluation_enhanced.py for consistency.
    
    Args:
        args: Command line arguments
    
    Returns:
        List of suffix strings
    """
    suffixes = []
    
    # TTA
    if args.use_tta:
        suffixes.append(f"tta_{args.tta_mode}")
    
    # Note: Tile classification doesn't use sliding window or blending
    # (those are only for reconstruction from overlapping tiles)
    
    # Boundary refinement - use consistent naming
    if args.boundary_refine:
        # Match full_evaluation_enhanced pattern: "refine_adaptive" or "refine"
        refine_type = getattr(args, 'refine_type', 'adaptive')  # Default to adaptive
        if refine_type == 'adaptive':
            suffixes.append('refine_adaptive')
        else:
            suffixes.append(f'refine_{refine_type}')
    
    return suffixes


def build_output_directory(args) -> Path:
    """
    Build output directory path matching full_evaluation_enhanced pattern.
    
    Format: checkpoint_dir/evaluation/{dataset}_{source}_{enhancements}/binary_classification_{threshold}/
    
    Args:
        args: Command line arguments
    
    Returns:
        Path object for output directory
    """
    # Get checkpoint directory
    checkpoint_dir = Path(args.weights).parent
    
    # Extract dataset name
    dataset_name = extract_dataset_name(args.data_root)
    
    # Detect data source
    data_source = detect_data_source(args.data_root)
    
    # Build enhancement suffixes
    enhancement_suffixes = build_enhancement_suffixes(args)
    
    # Build evaluation folder name (same as full_evaluation_enhanced)
    if enhancement_suffixes:
        eval_folder_name = f"{dataset_name}_{data_source}_{'_'.join(enhancement_suffixes)}"
    else:
        eval_folder_name = f"{dataset_name}_{data_source}"
    
    # Build binary classification subfolder with threshold
    threshold_folder = f"binary_classification_{args.threshold}"
    
    # Complete path
    output_dir = checkpoint_dir / "evaluation" / eval_folder_name / threshold_folder
    
    return output_dir


# ====================================================================
# BINARY CLASSIFICATION
# ====================================================================

def calculate_fat_percentage(mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate percentage of pixels classified as fat.
    
    Args:
        mask: Prediction mask [0-1]
        threshold: Threshold for binary classification
    
    Returns:
        Percentage of fat pixels (0-100)
    """
    binary_mask = (mask > threshold).astype(np.uint8)
    fat_pixels = binary_mask.sum()
    total_pixels = mask.size
    
    return (fat_pixels / total_pixels) * 100.0


def classify_tile(fat_percentage: float, classification_threshold: float) -> str:
    """
    Binary classification: Has Fat or No Fat.
    
    Args:
        fat_percentage: Percentage of fat in tile (0-100)
        classification_threshold: Threshold percentage (0-100)
    
    Returns:
        "Has Fat" or "No Fat"
    """
    return "Has Fat" if fat_percentage >= classification_threshold else "No Fat"


def calculate_confusion_matrix(predictions: List[str], ground_truth: List[str]) -> Dict:
    """
    Calculate confusion matrix for binary classification.
    
    Args:
        predictions: List of predictions ("Has Fat" or "No Fat")
        ground_truth: List of ground truth labels
    
    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    assert len(predictions) == len(ground_truth), "Mismatch in list lengths"
    
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == "Has Fat" and g == "Has Fat")
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == "No Fat" and g == "No Fat")
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "Has Fat" and g == "No Fat")
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == "No Fat" and g == "Has Fat")
    
    return {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'total': len(predictions)
    }


def calculate_classification_metrics(cm: Dict) -> Dict:
    """
    Calculate classification metrics from confusion matrix.
    
    Args:
        cm: Confusion matrix dictionary
    
    Returns:
        Dictionary with accuracy, precision, recall, F1-score, etc.
    """
    tp = cm['true_positive']
    tn = cm['true_negative']
    fp = cm['false_positive']
    fn = cm['false_negative']
    total = cm['total']
    
    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Precision (PPV)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall (Sensitivity, TPR)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (TNR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # NPV (Negative Predictive Value)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,  # Same as recall
        'specificity': specificity,
        'f1_score': f1_score,
        'npv': npv,
        'balanced_accuracy': (recall + specificity) / 2.0
    }


# ====================================================================
# VISUALIZATION
# ====================================================================

def plot_confusion_matrix(cm: Dict, output_path: Path, title: str = "Confusion Matrix"):
    """
    Create and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix dictionary
        output_path: Path to save PNG
        title: Plot title
    """
    # Create matrix
    matrix = np.array([
        [cm['true_positive'], cm['false_negative']],
        [cm['false_positive'], cm['true_negative']]
    ])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted: Has Fat', 'Predicted: No Fat'],
                yticklabels=['Actual: Has Fat', 'Actual: No Fat'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics text
    total = cm['total']
    accuracy = (cm['true_positive'] + cm['true_negative']) / total
    
    metrics_text = (
        f"Total Tiles: {total}\n"
        f"Accuracy: {accuracy:.1%}\n"
        f"TP: {cm['true_positive']}  FN: {cm['false_negative']}\n"
        f"FP: {cm['false_positive']}  TN: {cm['true_negative']}"
    )
    
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(results: List[Dict], output_path: Path):
    """
    Plot metrics across different thresholds.
    
    Args:
        results: List of result dictionaries for each threshold
        output_path: Path to save PNG
    """
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['metrics']['accuracy'] for r in results]
    precisions = [r['metrics']['precision'] for r in results]
    recalls = [r['metrics']['recall'] for r in results]
    f1_scores = [r['metrics']['f1_score'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
    ax.plot(thresholds, precisions, 's-', label='Precision', linewidth=2, markersize=8)
    ax.plot(thresholds, recalls, '^-', label='Recall', linewidth=2, markersize=8)
    ax.plot(thresholds, f1_scores, 'd-', label='F1-Score', linewidth=2, markersize=8)
    
    ax.set_xlabel('Fat Percentage Threshold (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ====================================================================
# MAIN EVALUATION
# ====================================================================

def evaluate_tiles(args, threshold_pct: float) -> Dict:
    """
    Evaluate binary classification for all tiles at a given threshold.
    
    Args:
        args: Command line arguments
        threshold_pct: Fat percentage threshold (0-100)
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING WITH {threshold_pct:.1f}% THRESHOLD")
    print(f"{'='*80}")
    
    # Setup paths
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Load model
    checkpoint_dir = Path(args.weights).parent
    train_mean, train_std = load_training_stats(str(checkpoint_dir))
    
    model = AdiposeUNet()
    model.build_model()
    model.load_weights(args.weights)
    
    # Initialize post-processing
    boundary_refiner = None
    if args.boundary_refine:
        boundary_refiner = BoundaryRefiner(
            kernel_size=args.refine_kernel,
            bilateral_d=5,
            bilateral_sigma_color=50,
            bilateral_sigma_space=50
        )
    
    # Find all tiles
    image_files = sorted(images_dir.glob("*.jpg"))
    mask_files = {p.stem: p for p in masks_dir.glob("*.tif")}
    
    # Filter to tiles with both image and mask
    paired_tiles = []
    for img_path in image_files:
        if img_path.stem in mask_files:
            paired_tiles.append((img_path, mask_files[img_path.stem]))
    
    print(f"Found {len(paired_tiles)} paired tiles")
    
    # Evaluate each tile
    results = []
    predictions = []
    ground_truths = []
    
    for img_path, mask_path in tqdm(paired_tiles, desc="Evaluating tiles"):
        # Load image
        tile = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Predict with optional enhancements
        if args.use_tta:
            tta = TestTimeAugmentation(mode=args.tta_mode)
            pred_mask, _ = tta.predict_with_tta(model, tile, train_mean, train_std)
        else:
            pred_mask = model.predict_single(tile, train_mean, train_std)
        
        # Apply boundary refinement if enabled
        if boundary_refiner is not None:
            pred_mask = boundary_refiner.refine(pred_mask, tile)
        
        # Load ground truth
        gt_mask = tiff.imread(str(mask_path)).astype(np.float32)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.squeeze()
        
        # Smart normalization - only divide by 255 if needed
        if gt_mask.max() > 1.0:
            gt_mask = gt_mask / 255.0  # Masks in [0-255] range
        # else: already in [0-1] range
        
        # Calculate fat percentages
        pred_fat_pct = calculate_fat_percentage(pred_mask, args.mask_threshold)
        gt_fat_pct = calculate_fat_percentage(gt_mask, args.mask_threshold)
        
        # Classify
        pred_class = classify_tile(pred_fat_pct, threshold_pct)
        gt_class = classify_tile(gt_fat_pct, threshold_pct)
        
        predictions.append(pred_class)
        ground_truths.append(gt_class)
        
        # Store result
        results.append({
            'tile_name': img_path.stem,
            'predicted_fat_pct': pred_fat_pct,
            'ground_truth_fat_pct': gt_fat_pct,
            'predicted_class': pred_class,
            'ground_truth_class': gt_class,
            'correct': pred_class == gt_class
        })
    
    # Calculate confusion matrix and metrics
    cm = calculate_confusion_matrix(predictions, ground_truths)
    metrics = calculate_classification_metrics(cm)
    
    # Print results
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Has Fat → Has Fat):   {cm['true_positive']}")
    print(f"  True Negatives (No Fat → No Fat):     {cm['true_negative']}")
    print(f"  False Positives (No Fat → Has Fat):   {cm['false_positive']}")
    print(f"  False Negatives (Has Fat → No Fat):   {cm['false_negative']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.1%}")
    print(f"  Precision:   {metrics['precision']:.1%}")
    print(f"  Recall:      {metrics['recall']:.1%}")
    print(f"  F1-Score:    {metrics['f1_score']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
    
    return {
        'threshold': threshold_pct,
        'confusion_matrix': cm,
        'metrics': metrics,
        'per_tile_results': results
    }


def save_results(results: Dict, output_dir: Path, args):
    """
    Save evaluation results to files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        args: Command line arguments
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-tile results
    df = pd.DataFrame(results['per_tile_results'])
    csv_path = output_dir / "per_tile_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Per-tile results saved: {csv_path}")
    
    # Save summary metrics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'weights': args.weights,
            'data_root': args.data_root,
            'classification_threshold_pct': results['threshold'],
            'mask_threshold': args.mask_threshold,
            'use_tta': args.use_tta,
            'tta_mode': args.tta_mode if args.use_tta else None,
            'boundary_refine': args.boundary_refine
        },
        'confusion_matrix': results['confusion_matrix'],
        'metrics': results['metrics'],
        'total_tiles': results['confusion_matrix']['total']
    }
    
    json_path = output_dir / "summary_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary metrics saved: {json_path}")
    
    # Plot confusion matrix
    cm_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        results['confusion_matrix'],
        cm_plot_path,
        title=f"Confusion Matrix ({results['threshold']:.1f}% Threshold)"
    )
    print(f"✓ Confusion matrix saved: {cm_plot_path}")
    
    # Save misclassified tiles
    misclassified = [r for r in results['per_tile_results'] if not r['correct']]
    if misclassified:
        misc_path = output_dir / "misclassified_tiles.txt"
        with open(misc_path, 'w') as f:
            f.write(f"Misclassified Tiles ({len(misclassified)} total)\n")
            f.write("="*80 + "\n\n")
            for tile in misclassified:
                f.write(f"{tile['tile_name']}\n")
                f.write(f"  Predicted: {tile['predicted_class']} ({tile['predicted_fat_pct']:.2f}%)\n")
                f.write(f"  Actual:    {tile['ground_truth_class']} ({tile['ground_truth_fat_pct']:.2f}%)\n\n")
        print(f"✓ Misclassified tiles saved: {misc_path}")


# ====================================================================
# CLI
# ====================================================================

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Tile-level binary classification evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to test dataset (contains images/ and masks/)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (optional - auto-generated if not provided)')
    
    # Classification parameters
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Fat percentage threshold for "Has Fat" classification (default: 10.0%%)')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                       help='Pixel threshold for binary mask (default: 0.5)')
    parser.add_argument('--multi-threshold', type=str, default=None,
                       help='Evaluate multiple thresholds (comma-separated, e.g., "1,5,10,15,25")')
    
    # Post-processing (from full_evaluation_enhanced)
    parser.add_argument('--use-tta', action='store_true', default=False,
                       help='Enable Test Time Augmentation')
    parser.add_argument('--tta-mode', type=str, default='basic',
                       choices=['minimal', 'basic', 'full'],
                       help='TTA mode (default: basic)')
    parser.add_argument('--boundary-refine', action='store_true', default=False,
                       help='Enable boundary refinement')
    parser.add_argument('--refine-kernel', type=int, default=5,
                       help='Kernel size for boundary refinement (default: 5)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    args = parse_args()
    
    # GPU setup
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    
    # Set deterministic seeds
    set_deterministic_seeds(1337)
    
    # Auto-generate output directory if not provided
    if args.output_dir is None:
        args.output_dir = str(build_output_directory(args))
        print(f"\n📁 Auto-generated output directory:")
        print(f"   {args.output_dir}")
    
    print(f"\n{'='*80}")
    print(f"TILE-LEVEL BINARY CLASSIFICATION EVALUATION")
    print(f"{'='*80}")
    print(f"Weights: {args.weights}")
    print(f"Data: {args.data_root}")
    print(f"Output: {args.output_dir}")
    
    try:
        if args.multi_threshold:
            # Multi-threshold analysis
            thresholds = [float(t) for t in args.multi_threshold.split(',')]
            print(f"\nRunning threshold analysis: {thresholds}")
            
            # For multi-threshold, create parent directory without threshold subfolder
            # Each threshold gets its own subfolder
            checkpoint_dir = Path(args.weights).parent
            dataset_name = extract_dataset_name(args.data_root)
            data_source = detect_data_source(args.data_root)
            enhancement_suffixes = build_enhancement_suffixes(args)
            
            if enhancement_suffixes:
                eval_folder_name = f"{dataset_name}_{data_source}_{'_'.join(enhancement_suffixes)}"
            else:
                eval_folder_name = f"{dataset_name}_{data_source}"
            
            base_output_dir = checkpoint_dir / "evaluation" / eval_folder_name
            
            all_results = []
            for threshold in thresholds:
                # Create modified args with current threshold
                threshold_args = argparse.Namespace(**vars(args))
                threshold_args.threshold = threshold
                
                results = evaluate_tiles(threshold_args, threshold)
                all_results.append(results)
                
                # Save individual threshold results
                threshold_dir = base_output_dir / f"binary_classification_{threshold}"
                save_results(results, threshold_dir, threshold_args)
            
            # Create threshold comparison in base directory
            output_dir = base_output_dir
            
            # Save comparison CSV
            comparison_data = []
            for r in all_results:
                comparison_data.append({
                    'threshold': r['threshold'],
                    'accuracy': r['metrics']['accuracy'],
                    'precision': r['metrics']['precision'],
                    'recall': r['metrics']['recall'],
                    'f1_score': r['metrics']['f1_score'],
                    'specificity': r['metrics']['specificity']
                })
            
            df = pd.DataFrame(comparison_data)
            csv_path = output_dir / "threshold_analysis.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Threshold analysis saved: {csv_path}")
            
            # Plot threshold analysis
            plot_path = output_dir / "threshold_analysis.png"
            plot_threshold_analysis(all_results, plot_path)
            print(f"✓ Threshold plot saved: {plot_path}")
            
        else:
            # Single threshold evaluation
            results = evaluate_tiles(args, args.threshold)
            save_results(results, Path(args.output_dir), args)
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Output: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
