#!/usr/bin/env python3
"""
Checkpoint Metrics Visualization Script
Extracts and visualizes metrics from checkpoint evaluation results

Creates publication-quality bar plots comparing:
- Dice scores with confidence intervals
- Performance metrics (Accuracy, Sensitivity, Specificity)

Metadata sources:
- Checkpoint: normalization_stats.json
- Build: build_summary.txt (if available)
- Metrics: comprehensive_results.csv

IMPORTANT: Flags MUST match the exact configuration used during evaluation with full_evaluation_enhanced.py
The script will throw an error if the expected directory doesn't exist and show available options.

Usage:
    # Basic evaluation (no enhancements)
    python visualize_checkpoint_metrics.py --clean-test --stain
    
    # With TTA
    python visualize_checkpoint_metrics.py --clean-test --stain --tta-mode full
    
    # With all enhancements (must match evaluation flags exactly)
    python visualize_checkpoint_metrics.py --clean-test --stain \
        --tta-mode full --sliding-window --boundary-refine --adaptive-threshold
    
    # Specific checkpoints with named subfolder
    python visualize_checkpoint_metrics.py --checkpoints 20251024_150723 20251101_023332 \
        --clean-test --stain --name best_models
    
    # Non-standard sliding window config
    python visualize_checkpoint_metrics.py --clean-test --original \
        --sliding-window --overlap 0.25 --blend-mode linear

NOTE: If you get a directory not found error, the script will show you which evaluation 
      directories actually exist and suggest the correct flags to use.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns


@dataclass
class CheckpointMetadata:
    """Container for checkpoint metadata"""
    timestamp: str
    normalization_method: str
    num_training_images: int
    preprocessing_applied: str
    stain_normalized: Optional[bool]
    
    # Build config (may be None if build_summary.txt not found)
    min_mask_ratio: Optional[float] = None
    overlap_percent: Optional[float] = None
    tile_size: Optional[int] = None
    stride: Optional[int] = None
    
    def get_label(self) -> str:
        """Generate multi-line label for Y-axis"""
        lines = [self.timestamp]
        
        # Line 2: Stain and image count
        if self.stain_normalized is None:
            stain_str = "Stain: N/A"
        elif self.stain_normalized:
            stain_str = "Stain: Yes"
        else:
            stain_str = "Stain: No"
        lines.append(f"{stain_str} | {self.num_training_images} imgs")
        
        # Line 3: MinMask and Overlap
        if self.min_mask_ratio is not None and self.overlap_percent is not None:
            lines.append(f"MinMask: {self.min_mask_ratio*100:.0f}% | Ovlp: {self.overlap_percent:.0f}%")
        elif self.min_mask_ratio is not None:
            lines.append(f"MinMask: {self.min_mask_ratio*100:.0f}% | Ovlp: N/A")
        else:
            lines.append("Config: N/A")
        
        # Line 4: Normalization method
        lines.append(f"Norm: {self.normalization_method}")
        
        return "\n".join(lines)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    dice_score: float
    dice_ci_lower: float
    dice_ci_upper: float
    accuracy: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    sensitivity: float
    sensitivity_ci_lower: float
    sensitivity_ci_upper: float
    specificity: float
    specificity_ci_lower: float
    specificity_ci_upper: float
    n_slides: int
    n_tiles: int


class CheckpointMetricsExtractor:
    """Extracts metrics and metadata from checkpoints"""
    
    def __init__(self, checkpoints_dir: str = "checkpoints",
                 data_root: str = "data/Meat_Luci_Tulane"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.data_root = Path(data_root)
        
        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {self.checkpoints_dir}")
    
    def discover_checkpoints(self, checkpoint_names: Optional[List[str]] = None) -> List[Path]:
        """
        Discover checkpoint directories
        
        Args:
            checkpoint_names: Optional list of specific checkpoint names/timestamps
            
        Returns:
            List of checkpoint directory paths
        """
        if checkpoint_names:
            # Specific checkpoints requested
            checkpoints = []
            for name in checkpoint_names:
                # Try exact match first
                checkpoint_path = self.checkpoints_dir / name
                if checkpoint_path.exists() and checkpoint_path.is_dir():
                    checkpoints.append(checkpoint_path)
                    continue
                
                # Try with suffix
                checkpoint_path = self.checkpoints_dir / f"{name}_adipose_sybreosin_1024_finetune"
                if checkpoint_path.exists() and checkpoint_path.is_dir():
                    checkpoints.append(checkpoint_path)
                    continue
                
                # Try pattern match
                matches = list(self.checkpoints_dir.glob(f"*{name}*"))
                if matches:
                    checkpoints.extend([m for m in matches if m.is_dir()])
                else:
                    print(f"⚠️  Checkpoint not found: {name}")
        else:
            # Discover all checkpoints
            checkpoints = [d for d in self.checkpoints_dir.glob("*adipose*") if d.is_dir()]
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        
        # Sort by timestamp (extract from name)
        def extract_timestamp(path: Path) -> str:
            match = re.search(r'(\d{8}_\d{6})', path.name)
            return match.group(1) if match else path.name
        
        checkpoints.sort(key=extract_timestamp)
        
        print(f"✓ Found {len(checkpoints)} checkpoint(s)")
        return checkpoints
    
    def extract_checkpoint_metadata(self, checkpoint_dir: Path) -> CheckpointMetadata:
        """
        Extract metadata from checkpoint's normalization_stats.json
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            CheckpointMetadata object
        """
        stats_file = checkpoint_dir / "normalization_stats.json"
        
        if not stats_file.exists():
            raise FileNotFoundError(f"normalization_stats.json not found in {checkpoint_dir}")
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Extract timestamp from directory name or from stats
        timestamp = stats.get('build_timestamp')
        if not timestamp:
            match = re.search(r'(\d{8}_\d{6})', checkpoint_dir.name)
            timestamp = match.group(1) if match else checkpoint_dir.name
        
        # Initialize metadata (stain_normalized will be set from build_summary.txt)
        metadata = CheckpointMetadata(
            timestamp=timestamp,
            normalization_method=stats.get('normalization_method', 'unknown'),
            num_training_images=stats.get('num_training_images', 0),
            preprocessing_applied=stats.get('preprocessing_applied', ''),
            stain_normalized=None  # Will be updated from build_summary.txt (None = unknown/not found)
        )
        
        # Extract build config including stain normalization status
        self._extract_build_config(metadata, timestamp)
        
        return metadata
    
    def _extract_build_config(self, metadata: CheckpointMetadata, timestamp: str) -> None:
        """
        Extract build configuration from build_summary.txt
        
        Args:
            metadata: CheckpointMetadata to populate
            timestamp: Build timestamp
        """
        build_dir = self.data_root / f"_build_{timestamp}"
        summary_file = build_dir / "build_summary.txt"
        
        if not summary_file.exists():
            print(f"  ℹ️  build_summary.txt not found for {timestamp} (using defaults)")
            return
        
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            
            # Extract stain normalization status (primary method)
            match = re.search(r'Stain Normalization Actually Used:\s*(True|False)', content, re.IGNORECASE)
            if match:
                metadata.stain_normalized = match.group(1).lower() == 'true'
            else:
                # Fallback: check "Actual Status" field
                match = re.search(r'Actual Status:\s*ENABLED_AND_AVAILABLE', content, re.IGNORECASE)
                if match:
                    metadata.stain_normalized = True
                else:
                    # Second fallback: check if "Requested: YES"
                    match = re.search(r'Requested:\s*YES', content, re.IGNORECASE)
                    if match:
                        metadata.stain_normalized = True
            
            # Extract minimum mask ratio
            match = re.search(r'Minimum Mask Ratio:\s*([\d.]+)', content)
            if match:
                metadata.min_mask_ratio = float(match.group(1))
            
            # Extract tile size
            match = re.search(r'Tile Size:\s*(\d+)x(\d+)', content)
            if match:
                metadata.tile_size = int(match.group(1))
            
            # Extract stride
            match = re.search(r'Stride:\s*(\d+)', content)
            if match:
                metadata.stride = int(match.group(1))
            
            # Calculate overlap percentage
            if metadata.tile_size and metadata.stride:
                overlap = metadata.tile_size - metadata.stride
                metadata.overlap_percent = (overlap / metadata.tile_size) * 100
            
        except Exception as e:
            print(f"  ⚠️  Error parsing build_summary.txt for {timestamp}: {e}")
    
    def extract_evaluation_metrics(self, checkpoint_dir: Path, 
                                   eval_config: str) -> Optional[EvaluationMetrics]:
        """
        Extract metrics from evaluation CSV file with strict directory validation
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            eval_config: Evaluation configuration string (e.g., "clean_test_stain_tta_full")
            
        Returns:
            EvaluationMetrics object or None if not found
            
        Raises:
            FileNotFoundError: If expected evaluation directory doesn't exist with helpful suggestions
        """
        # Build evaluation path
        eval_dir = checkpoint_dir / "evaluation" / eval_config
        
        # STRICT VALIDATION: Check if directory exists
        if not eval_dir.exists():
            # List available evaluation directories for this checkpoint
            eval_base = checkpoint_dir / "evaluation"
            
            if not eval_base.exists():
                raise FileNotFoundError(
                    f"❌ No evaluation directory found for checkpoint: {checkpoint_dir.name}\n"
                    f"   Expected base: {eval_base}\n"
                    f"   This checkpoint has not been evaluated yet."
                )
            
            # Find what evaluation directories actually exist
            available_dirs = [d.name for d in eval_base.iterdir() if d.is_dir()]
            
            if not available_dirs:
                raise FileNotFoundError(
                    f"❌ Evaluation directory exists but is empty: {eval_base}\n"
                    f"   No evaluation configurations found."
                )
            
            # Build helpful error message
            error_msg = (
                f"❌ Evaluation directory not found for checkpoint: {checkpoint_dir.name}\n\n"
                f"   Expected: {eval_config}\n"
                f"   Location: {eval_dir}\n\n"
                f"   Available evaluation configurations ({len(available_dirs)}):\n"
            )
            
            for available_dir in sorted(available_dirs):
                error_msg += f"     - {available_dir}\n"
            
            # Parse expected config to suggest corrections
            error_msg += f"\n   Suggestion:\n"
            
            # Check if a similar config exists (without some flags)
            base_config_parts = eval_config.split('_')
            if len(base_config_parts) >= 2:
                base_config = '_'.join(base_config_parts[:2])  # e.g., "clean_test_stain"
                
                # Check if base config exists
                if base_config in available_dirs:
                    error_msg += f"     ✓ Found base config: {base_config}\n"
                    error_msg += f"     → Remove enhancement flags (--tta-mode, --sliding-window, etc.)\n"
                else:
                    # Check if any config matches the dataset but different data source
                    dataset_part = base_config_parts[0]  # e.g., "clean_test"
                    matching = [d for d in available_dirs if d.startswith(dataset_part)]
                    
                    if matching:
                        error_msg += f"     ✓ Found {len(matching)} config(s) for dataset '{dataset_part}':\n"
                        for m in matching[:3]:  # Show up to 3 examples
                            error_msg += f"       → {m}\n"
                        error_msg += f"     → Check your --stain/--original flag and enhancement flags\n"
                    else:
                        error_msg += f"     → No configurations found for dataset '{dataset_part}'\n"
                        error_msg += f"     → Check your dataset flag (--clean-test, --human-test, etc.)\n"
            
            raise FileNotFoundError(error_msg)
        
        # Look for comprehensive results CSV
        csv_files = list(eval_dir.glob("*comprehensive_results.csv"))
        
        if not csv_files:
            raise FileNotFoundError(
                f"❌ Evaluation directory exists but no results CSV found\n"
                f"   Directory: {eval_dir}\n"
                f"   Expected file pattern: *comprehensive_results.csv\n"
                f"   The evaluation may have failed or not completed."
            )
        
        csv_file = csv_files[0]
        
        try:
            df = pd.read_csv(csv_file)
            
            # Extract metrics by row
            metrics_dict = {}
            for _, row in df.iterrows():
                metric_name = row['Metric'].lower().replace(' ', '_').replace('(', '').replace(')', '')
                metrics_dict[metric_name] = {
                    'mean': row['Mean'],
                    'ci_lower': row['CI_Lower'],
                    'ci_upper': row['CI_Upper']
                }
            
            # Get sample counts from first row
            n_slides = int(df.iloc[0]['N_Slides'])
            n_tiles = int(df.iloc[0]['N_Tiles'])
            
            # Build EvaluationMetrics object
            metrics = EvaluationMetrics(
                dice_score=metrics_dict['dice_score']['mean'],
                dice_ci_lower=metrics_dict['dice_score']['ci_lower'],
                dice_ci_upper=metrics_dict['dice_score']['ci_upper'],
                accuracy=metrics_dict['accuracy']['mean'],
                accuracy_ci_lower=metrics_dict['accuracy']['ci_lower'],
                accuracy_ci_upper=metrics_dict['accuracy']['ci_upper'],
                sensitivity=metrics_dict['sensitivity_recall']['mean'],
                sensitivity_ci_lower=metrics_dict['sensitivity_recall']['ci_lower'],
                sensitivity_ci_upper=metrics_dict['sensitivity_recall']['ci_upper'],
                specificity=metrics_dict['specificity']['mean'],
                specificity_ci_lower=metrics_dict['specificity']['ci_lower'],
                specificity_ci_upper=metrics_dict['specificity']['ci_upper'],
                n_slides=n_slides,
                n_tiles=n_tiles
            )
            
            return metrics
            
        except Exception as e:
            print(f"  ❌ Error parsing metrics CSV {csv_file}: {e}")
            return None


class MetricsVisualizer:
    """Creates publication-quality visualization plots"""
    
    def __init__(self, output_dir: str = "model_comparison_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
    
    def plot_dice_comparison(self, data: List[Tuple[CheckpointMetadata, EvaluationMetrics]],
                            config_name: str, collection_name: str = "") -> Path:
        """
        Create Dice score comparison plot with confidence intervals
        
        Args:
            data: List of (metadata, metrics) tuples
            config_name: Evaluation configuration name
            collection_name: Name of the checkpoint collection (e.g., "all" or custom name)
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.5)))
        
        # Extract data
        labels = [metadata.get_label() for metadata, _ in data]
        dice_scores = [metrics.dice_score for _, metrics in data]
        ci_lower = [metrics.dice_ci_lower for _, metrics in data]
        ci_upper = [metrics.dice_ci_upper for _, metrics in data]
        
        # Calculate error bars (distance from mean)
        errors_lower = [dice - lower for dice, lower in zip(dice_scores, ci_lower)]
        errors_upper = [upper - dice for dice, upper in zip(dice_scores, ci_upper)]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, dice_scores, height=0.6, 
                      color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
        
        # Add error bars (confidence intervals)
        ax.errorbar(dice_scores, y_pos, 
                   xerr=[errors_lower, errors_upper],
                   fmt='none', ecolor='darkred', elinewidth=2, capsize=5, capthick=2)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, dice_scores)):
            ax.text(score + 0.02, i, f'{score:.4f}', 
                   va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Dice Score', fontweight='bold')
        
        # Build title with collection name
        title = f'Dice Score Comparison\n{config_name}'
        if collection_name:
            title = f'Dice Score Comparison\n{config_name} - {collection_name}'
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add reference lines
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0.5)')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"dice_comparison_{config_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Dice comparison plot: {output_path}")
        return output_path
    
    def plot_performance_metrics(self, data: List[Tuple[CheckpointMetadata, EvaluationMetrics]],
                                config_name: str, collection_name: str = "") -> Path:
        """
        Create grouped performance metrics comparison
        
        Args:
            data: List of (metadata, metrics) tuples
            config_name: Evaluation configuration name
            collection_name: Name of the checkpoint collection (e.g., "all" or custom name)
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, max(8, len(data) * 0.8)))
        
        # Extract data
        labels = [metadata.get_label() for metadata, _ in data]
        accuracy = [metrics.accuracy for _, metrics in data]
        sensitivity = [metrics.sensitivity for _, metrics in data]
        specificity = [metrics.specificity for _, metrics in data]
        
        # Bar positions
        y_pos = np.arange(len(labels))
        bar_height = 0.25
        
        # Create grouped bars
        bars1 = ax.barh(y_pos - bar_height, accuracy, bar_height, 
                       label='Accuracy', color='#2E86AB', alpha=0.9, edgecolor='black', linewidth=0.8)
        bars2 = ax.barh(y_pos, sensitivity, bar_height,
                       label='Sensitivity', color='#A23B72', alpha=0.9, edgecolor='black', linewidth=0.8)
        bars3 = ax.barh(y_pos + bar_height, specificity, bar_height,
                       label='Specificity', color='#F18F01', alpha=0.9, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', va='center', ha='left', fontsize=8)
        
        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Score', fontweight='bold')
        
        # Build title with collection name
        title = f'Performance Metrics Comparison\n{config_name}'
        if collection_name:
            title = f'Performance Metrics Comparison\n{config_name} - {collection_name}'
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.05)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"performance_metrics_{config_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved performance metrics plot: {output_path}")
        return output_path
    
    def save_summary_csv(self, data: List[Tuple[CheckpointMetadata, EvaluationMetrics]],
                        config_name: str) -> Path:
        """
        Save metrics summary to CSV
        
        Args:
            data: List of (metadata, metrics) tuples
            config_name: Evaluation configuration name
            
        Returns:
            Path to saved CSV
        """
        rows = []
        
        for metadata, metrics in data:
            row = {
                'checkpoint': metadata.timestamp,
                'stain_normalized': metadata.stain_normalized,
                'num_images': metadata.num_training_images,
                'normalization_method': metadata.normalization_method,
                'min_mask_ratio': metadata.min_mask_ratio if metadata.min_mask_ratio else 'N/A',
                'overlap_percent': f"{metadata.overlap_percent:.1f}" if metadata.overlap_percent else 'N/A',
                'dice_score': metrics.dice_score,
                'dice_ci_lower': metrics.dice_ci_lower,
                'dice_ci_upper': metrics.dice_ci_upper,
                'accuracy': metrics.accuracy,
                'accuracy_ci_lower': metrics.accuracy_ci_lower,
                'accuracy_ci_upper': metrics.accuracy_ci_upper,
                'sensitivity': metrics.sensitivity,
                'sensitivity_ci_lower': metrics.sensitivity_ci_lower,
                'sensitivity_ci_upper': metrics.sensitivity_ci_upper,
                'specificity': metrics.specificity,
                'specificity_ci_lower': metrics.specificity_ci_lower,
                'specificity_ci_upper': metrics.specificity_ci_upper,
                'n_slides': metrics.n_slides,
                'n_tiles': metrics.n_tiles
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by dice score descending
        df = df.sort_values('dice_score', ascending=False)
        
        # Save
        output_path = self.output_dir / f"metrics_summary_{config_name}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved metrics summary CSV: {output_path}")
        return output_path


def build_eval_config_string(args: argparse.Namespace) -> str:
    """
    Build evaluation configuration string from CLI arguments
    MUST MATCH the exact naming scheme from full_evaluation_enhanced.py
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configuration string (e.g., "clean_test_stain_tta_full_sw_gaussian_refine_adaptive")
    """
    parts = []
    
    # Dataset selection
    if args.val:
        parts.append('val')
    elif args.test:
        parts.append('test')
    elif args.human_test:
        parts.append('human_test')
    elif args.clean_test:
        parts.append('clean_test')
    elif args.clean_test_50_overlap:
        parts.append('clean_test_50_overlap')
    else:
        raise ValueError("No dataset specified. Use --val, --test, --human-test, --clean-test, or --clean-test-50-overlap")
    
    # Data source
    if args.stain:
        parts.append('stain')
    elif args.original:
        parts.append('original')
    else:
        raise ValueError("No data source specified. Use --stain or --original")
    
    # Enhancement suffixes (MUST match full_evaluation_enhanced.py logic)
    enhancement_suffixes = []
    
    # TTA
    if args.tta_mode:
        enhancement_suffixes.append(f'tta_{args.tta_mode}')
    
    # Sliding window
    if args.sliding_window:
        sw_suffix = f'sw_{args.blend_mode}'
        if args.overlap != 0.5:
            sw_suffix += f'_o{int(args.overlap*100)}'
        enhancement_suffixes.append(sw_suffix)
    
    # Boundary refinement
    if args.boundary_refine:
        refine_suffix = 'refine'
        if args.refine_kernel != 5:
            refine_suffix += f'{args.refine_kernel}'
        enhancement_suffixes.append(refine_suffix)
    
    # Adaptive threshold
    if args.adaptive_threshold:
        enhancement_suffixes.append('adaptive')
    
    # Combine all parts
    if enhancement_suffixes:
        parts.extend(enhancement_suffixes)
    
    return '_'.join(parts)


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Visualize checkpoint evaluation metrics with publication-quality plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All checkpoints (saves to model_comparison_plots/all/)
  python visualize_checkpoint_metrics.py --clean-test --stain --tta-full
  
  # Specific checkpoints with named subfolder (saves to model_comparison_plots/best_models/)
  python visualize_checkpoint_metrics.py --checkpoints 20251024_150723 20251101_023332 \
      --clean-test --stain --name best_models
  
  # Custom output directory with all checkpoints (saves to my_plots/all/)
  python visualize_checkpoint_metrics.py --clean-test --stain --output ./my_plots/
  
  # ERROR: --name required when using --checkpoints
  python visualize_checkpoint_metrics.py --checkpoints 20251024_150723 --clean-test --stain
        """
    )
    
    # Checkpoint selection
    parser.add_argument('--checkpoints', nargs='+', 
                       help='Specific checkpoint names/timestamps to visualize (default: all)')
    parser.add_argument('--name', type=str,
                       help='Subfolder name for output (required when using --checkpoints)')
    
    # Dataset selection (mutually exclusive, required)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--val', action='store_true',
                              help='Use validation set evaluation')
    dataset_group.add_argument('--test', action='store_true',
                              help='Use test set evaluation')
    dataset_group.add_argument('--human-test', action='store_true',
                              help='Use human_test set evaluation')
    dataset_group.add_argument('--clean-test', action='store_true',
                              help='Use clean_test set evaluation')
    
    # Data source (mutually exclusive, required)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--stain', action='store_true',
                             help='Use stain normalized data')
    source_group.add_argument('--original', action='store_true',
                             help='Use original data')
    
    # Enhancement flags (must match evaluation configuration)
    parser.add_argument('--tta-mode', type=str, choices=['minimal', 'basic', 'full'],
                       help='TTA mode (if evaluation used TTA)')
    parser.add_argument('--sliding-window', action='store_true', default=False,
                       help='Evaluation used sliding window inference')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for sliding window (0.0-0.75, default: 0.5)')
    parser.add_argument('--blend-mode', type=str, default='gaussian',
                       choices=['gaussian', 'linear', 'none'],
                       help='Blending mode for sliding window (default: gaussian)')
    parser.add_argument('--boundary-refine', action='store_true', default=False,
                       help='Evaluation used morphological boundary refinement')
    parser.add_argument('--refine-kernel', type=int, default=5,
                       help='Kernel size for boundary refinement (default: 5)')
    parser.add_argument('--adaptive-threshold', action='store_true', default=False,
                       help='Evaluation used adaptive threshold optimization')
    
    # Output options
    parser.add_argument('--output', type=str, default='model_comparison_plots',
                       help='Output directory for plots (default: model_comparison_plots)')
    
    args = parser.parse_args()
    
    # Validate: --name is required when using --checkpoints
    if args.checkpoints and not args.name:
        parser.error("--name is required when using --checkpoints to specify a subfolder name")
    
    print("="*80)
    print("📊 CHECKPOINT METRICS VISUALIZATION")
    print("="*80)
    
    try:
        # Determine output subfolder
        if args.checkpoints:
            subfolder = args.name  # Required by validation above
        else:
            subfolder = "all"
        
        # Build full output path
        output_path = Path(args.output) / subfolder
        
        # Build evaluation configuration string
        eval_config = build_eval_config_string(args)
        print(f"\n🔍 Evaluation configuration: {eval_config}")
        print(f"📁 Output subfolder: {subfolder}")
        
        # Initialize extractor
        extractor = CheckpointMetricsExtractor()
        
        # Discover checkpoints
        checkpoints = extractor.discover_checkpoints(args.checkpoints)
        print(f"📁 Checkpoints: {[c.name for c in checkpoints]}")
        
        # Extract data for each checkpoint
        data = []
        print(f"\n📊 Extracting metrics...")
        
        for checkpoint_dir in checkpoints:
            print(f"\n  Processing: {checkpoint_dir.name}")
            
            try:
                # Extract metadata
                metadata = extractor.extract_checkpoint_metadata(checkpoint_dir)
                
                # Extract metrics
                metrics = extractor.extract_evaluation_metrics(checkpoint_dir, eval_config)
                
                if metrics is None:
                    print(f"    ⚠️  Skipping (no metrics found)")
                    continue
                
                data.append((metadata, metrics))
                print(f"    ✓ Dice: {metrics.dice_score:.4f}, Slides: {metrics.n_slides}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        if not data:
            print("\n❌ No valid data found. Check that evaluations exist for the specified configuration.")
            return 1
        
        print(f"\n✓ Successfully extracted data from {len(data)} checkpoint(s)")
        
        # Sort by dice score (descending)
        data.sort(key=lambda x: x[1].dice_score, reverse=True)
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        visualizer = MetricsVisualizer(str(output_path))
        
        # Plot 1: Dice comparison
        dice_plot = visualizer.plot_dice_comparison(data, eval_config, subfolder)
        
        # Plot 2: Performance metrics
        perf_plot = visualizer.plot_performance_metrics(data, eval_config, subfolder)
        
        # Save summary CSV
        csv_file = visualizer.save_summary_csv(data, eval_config)
        
        # Summary
        print("\n" + "="*80)
        print("✅ VISUALIZATION COMPLETE")
        print("="*80)
        print(f"📊 Checkpoints visualized: {len(data)}")
        print(f"📁 Output directory: {visualizer.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - {dice_plot.name}")
        print(f"  - {perf_plot.name}")
        print(f"  - {csv_file.name}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
