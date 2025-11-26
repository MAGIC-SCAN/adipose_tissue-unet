#!/usr/bin/env python3
"""
Batch Evaluation Script for All Adipose U-Net Checkpoints
Runs full evaluation on all checkpoint directories, then generates comparative visualizations

Key Features:
- Auto-discovers all checkpoint directories
- Runs evaluation on specified datasets
- Supports TTA configuration
- Robust error handling
- Automatically generates comparison plots via visualize_checkpoint_metrics.py

USAGE EXAMPLES:

1. Evaluate all checkpoints on clean test set:
   python Segmentation/evaluate_all_checkpoints.py \
     --clean-test --stain

2. Evaluate with TTA (basic 4x augmentation):
   python Segmentation/evaluate_all_checkpoints.py \
     --clean-test --stain \
     --use-tta --tta-mode basic

3. Evaluate with full TTA (8x augmentation):
   python Segmentation/evaluate_all_checkpoints.py \
     --clean-test --stain \
     --use-tta --tta-mode full

4. Parallel evaluation (faster on multi-core systems):
   python Segmentation/evaluate_all_checkpoints.py \
     --clean-test --stain \
     --parallel --max-workers 4

5. Evaluate specific checkpoints directory:
   python Segmentation/evaluate_all_checkpoints.py \
     --checkpoints-dir /path/to/custom/checkpoints \
     --clean-test --stain \
     --use-tta --tta-mode full

6. Evaluate on original test set (no stain normalization):
   python Segmentation/evaluate_all_checkpoints.py \
     --clean-test --original

OUTPUT:
  - Evaluation results for each checkpoint in its evaluation/ subdirectory
  - Comparative plots in model_comparison_plots/
  - Summary CSV with all checkpoint metrics
  - Execution log with timing and error information
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import subprocess

from tqdm import tqdm

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class CheckpointBatchEvaluator:
    """Manager class for batch evaluation of all checkpoints"""
    
    def __init__(self, dataset_names: List[str] = None, data_source: str = 'original',
                 use_tta: bool = False, tta_mode: str = 'basic'):
        """
        Initialize batch evaluator
        
        Args:
            dataset_names: List of dataset names being evaluated
            data_source: Data source ('stain' or 'original')
            use_tta: Whether TTA is enabled
            tta_mode: TTA mode if enabled
        """
        self.checkpoints_dir = Path("checkpoints")
        self.results = {}
        self.failed_evaluations = []
        
        print(f"🎯 Batch Checkpoint Evaluator Initialized")
        print(f"🔍 Searching for checkpoints in: {self.checkpoints_dir}")
    
    def discover_checkpoints(self, filter_pattern: str = "*adipose*") -> List[Tuple[Path, str]]:
        """
        Discover all valid checkpoint directories
        
        Args:
            filter_pattern: Pattern to filter checkpoint directories
            
        Returns:
            List of (checkpoint_path, checkpoint_id) tuples
        """
        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {self.checkpoints_dir}")
        
        # Find all potential checkpoint directories
        checkpoint_candidates = list(self.checkpoints_dir.glob(filter_pattern))
        
        valid_checkpoints = []
        
        for checkpoint_dir in checkpoint_candidates:
            if not checkpoint_dir.is_dir():
                continue
            
            # Check if it contains weight files
            weight_files = list(checkpoint_dir.glob("*.weights.h5")) + list(checkpoint_dir.glob("*.h5"))
            
            if weight_files:
                # Extract timestamp or create ID from directory name
                checkpoint_id = self._extract_checkpoint_id(checkpoint_dir)
                valid_checkpoints.append((checkpoint_dir, checkpoint_id))
                print(f"  ✓ Found checkpoint: {checkpoint_dir.name}")
            else:
                print(f"  ⚠️  Skipping {checkpoint_dir.name} (no .h5/.weights.h5 files)")
        
        if not valid_checkpoints:
            raise FileNotFoundError(f"No valid checkpoints found in {self.checkpoints_dir}")
        
        # Sort by timestamp if available
        valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Discovered {len(valid_checkpoints)} valid checkpoint(s)")
        return valid_checkpoints
    
    def _extract_checkpoint_id(self, checkpoint_dir: Path) -> str:
        """Extract meaningful ID from checkpoint directory name"""
        dir_name = checkpoint_dir.name
        
        # Try to extract timestamp (YYYYMMDD_HHMMSS format)
        timestamp_match = re.search(r'(\d{8}_\d{6})', dir_name)
        if timestamp_match:
            return timestamp_match.group(1)
        
        # Fall back to directory name
        return dir_name.replace("_adipose_sybreosin_1024_finetune", "").replace("_", "")
    
    def _find_best_weights(self, checkpoint_dir: Path) -> Optional[Path]:
        """Find the best weights file in a checkpoint directory"""
        
        # Priority order for weight file selection
        weight_candidates = [
            "weights_best_overall.weights.h5",
            "phase2_best.weights.h5", 
            "phase1_best.weights.h5",
            "best_model.weights.h5",
            "model_best.weights.h5",
            "weights_best.weights.h5",
        ]
        
        for candidate in weight_candidates:
            weight_path = checkpoint_dir / candidate
            if weight_path.exists():
                return weight_path
        
        # Fall back to any .h5 file
        weight_files = list(checkpoint_dir.glob("*.weights.h5")) + list(checkpoint_dir.glob("*.h5"))
        if weight_files:
            return weight_files[0]
        
        return None
    
    def evaluate_single_checkpoint(self, checkpoint_dir: Path, checkpoint_id: str, 
                                 datasets_to_eval: List[str],
                                 use_tta: bool = False, tta_mode: str = 'basic',
                                 save_images: bool = True,
                                 sliding_window: bool = False, overlap: float = 0.5,
                                 blend_mode: str = 'gaussian', boundary_refine: bool = False,
                                 refine_kernel: int = 5, adaptive_threshold: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single checkpoint on specified datasets
        
        Returns:
            Dictionary with evaluation results or error information
        """
        result = {
            'checkpoint_id': checkpoint_id,
            'checkpoint_dir': str(checkpoint_dir),
            'status': 'pending',
            'error': None,
            'evaluation_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Find best weights file
            weights_path = self._find_best_weights(checkpoint_dir)
            if not weights_path:
                result['status'] = 'failed'
                result['error'] = 'No suitable weights file found'
                return result
            
            result['weights_path'] = str(weights_path)
            
            # Run evaluation with provided dataset flags
            print(f"    🔄 Running evaluation with flags: {datasets_to_eval}")
            
            eval_result = self._run_evaluation(
                weights_path=weights_path,
                dataset_flags=datasets_to_eval,
                use_tta=use_tta,
                tta_mode=tta_mode,
                save_images=save_images,
                sliding_window=sliding_window,
                overlap=overlap,
                blend_mode=blend_mode,
                boundary_refine=boundary_refine,
                refine_kernel=refine_kernel,
                adaptive_threshold=adaptive_threshold
            )
            
            result['status'] = eval_result['status']
            result['error'] = eval_result.get('error')
            result['evaluation_time'] = time.time() - start_time
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['evaluation_time'] = time.time() - start_time
            print(f"    ❌ Error: {e}")
        
        return result
    
    def _run_evaluation(self, weights_path: Path, dataset_flags: List[str], 
                       use_tta: bool, tta_mode: str, save_images: bool,
                       sliding_window: bool = False, overlap: float = 0.5,
                       blend_mode: str = 'gaussian', boundary_refine: bool = False,
                       refine_kernel: int = 5, adaptive_threshold: bool = False) -> Dict[str, Any]:
        """Run evaluation using full_evaluation_enhanced.py"""
        
        result = {'status': 'pending', 'error': None}
        
        try:
            # Build command for full_evaluation_enhanced.py
            cmd = [
                'conda', 'run', '-n', 'adipose-tf2', 'python', 'full_evaluation_enhanced.py',
                '--weights', str(weights_path)
            ]
            
            # Add dataset flags
            cmd.extend(dataset_flags)
            
            # Add TTA options
            if use_tta:
                cmd.extend(['--use-tta', '--tta-mode', tta_mode])
            
            # Add enhanced post-processing options
            if sliding_window:
                cmd.append('--sliding-window')
                cmd.extend(['--overlap', str(overlap)])
                cmd.extend(['--blend-mode', blend_mode])
            
            if boundary_refine:
                cmd.append('--boundary-refine')
                cmd.extend(['--refine-kernel', str(refine_kernel)])
            
            if adaptive_threshold:
                cmd.append('--adaptive-threshold')
            
            # Add visualization option
            if not save_images:
                cmd.append('--no-visualizations')
            
            print(f"      🔧 Running: {' '.join(cmd)}")
            print(f"      📊 Streaming output in real-time...")
            print("-" * 80)
            
            # Run evaluation with proper output handling using Popen
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                cwd=os.getcwd()
            )
            
            # Stream output line by line in real-time
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line, end='')  # Already has newline
                
                # Wait for process to complete
                return_code = process.wait(timeout=3600)  # 1 hour timeout
                
                print("-" * 80)
                if return_code == 0:
                    print(f"      ✅ Evaluation completed successfully")
                    result['status'] = 'completed'
                else:
                    print(f"      ❌ Evaluation failed (return code {return_code})")
                    result['status'] = 'failed'
                    result['error'] = f"Process failed with return code {return_code}"
                    
            finally:
                # Ensure streams are closed
                if process.stdout:
                    process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
        
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error'] = 'Evaluation timeout (>1 hour)'
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def run_batch_evaluation(self, datasets: List[str], use_tta: bool = False, 
                           tta_mode: str = 'basic', save_images: bool = True, 
                           parallel: bool = False, max_workers: int = 2,
                           sliding_window: bool = False, overlap: float = 0.5,
                           blend_mode: str = 'gaussian', boundary_refine: bool = False,
                           refine_kernel: int = 5, adaptive_threshold: bool = False) -> bool:
        """
        Run evaluation on all discovered checkpoints
        
        Returns:
            True if at least one evaluation succeeded, False otherwise
        """
        
        print("\n" + "="*80)
        print("🚀 BATCH CHECKPOINT EVALUATION")
        print("="*80)
        
        # Discover checkpoints
        checkpoints = self.discover_checkpoints()
        
        # Show configuration
        dataset_names = [flag.replace('--', '') for flag in datasets]
        print(f"\n🔍 Configuration:")
        print(f"   📊 Datasets: {', '.join(datasets)}")
        print(f"   🔄 TTA: {use_tta} ({tta_mode if use_tta else 'N/A'})")
        print(f"   🖼️  Save images: {save_images}")
        print(f"   🔬 Sliding window: {sliding_window}")
        if sliding_window:
            print(f"   📐 Overlap: {overlap:.1%}, Blend: {blend_mode}")
        print(f"   🎨 Boundary refine: {boundary_refine}")
        if boundary_refine:
            print(f"   🔧 Kernel size: {refine_kernel}")
        print(f"   📊 Adaptive threshold: {adaptive_threshold}")
        print(f"   ⚡ Parallel: {parallel} ({max_workers} workers)")
        print()
        
        # Start evaluation
        start_time = time.time()
        
        if parallel and len(checkpoints) > 1:
            self._run_parallel_evaluation(checkpoints, datasets, use_tta, tta_mode, 
                                        save_images, max_workers, sliding_window, overlap,
                                        blend_mode, boundary_refine, refine_kernel, adaptive_threshold)
        else:
            self._run_sequential_evaluation(checkpoints, datasets, use_tta, tta_mode, 
                                          save_images, sliding_window, overlap, blend_mode,
                                          boundary_refine, refine_kernel, adaptive_threshold)
        
        total_time = time.time() - start_time
        
        # Summary
        successful = len([r for r in self.results.values() if r['status'] == 'completed'])
        failed = len([r for r in self.results.values() if r['status'] != 'completed'])
        
        print("\n" + "="*80)
        print("📊 BATCH EVALUATION COMPLETE")
        print("="*80)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print("="*80)
        
        return successful > 0
    
    def _run_sequential_evaluation(self, checkpoints: List[Tuple[Path, str]], 
                                  datasets: List[str], use_tta: bool, tta_mode: str, 
                                  save_images: bool, sliding_window: bool, overlap: float,
                                  blend_mode: str, boundary_refine: bool, refine_kernel: int,
                                  adaptive_threshold: bool) -> None:
        """Run evaluations sequentially with progress tracking"""
        
        print(f"🔄 Running sequential evaluation on {len(checkpoints)} checkpoint(s)...")
        
        with tqdm(total=len(checkpoints), desc="📊 Evaluating", unit="checkpoint") as pbar:
            for i, (checkpoint_dir, checkpoint_id) in enumerate(checkpoints):
                pbar.set_description(f"📊 [{i+1}/{len(checkpoints)}] {checkpoint_dir.name}")
                
                print(f"\n[{i+1}/{len(checkpoints)}] 🔍 {checkpoint_dir.name}")
                
                result = self.evaluate_single_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_id=checkpoint_id,
                    datasets_to_eval=datasets,
                    use_tta=use_tta,
                    tta_mode=tta_mode,
                    save_images=save_images,
                    sliding_window=sliding_window,
                    overlap=overlap,
                    blend_mode=blend_mode,
                    boundary_refine=boundary_refine,
                    refine_kernel=refine_kernel,
                    adaptive_threshold=adaptive_threshold
                )
                
                self.results[checkpoint_id] = result
                
                if result['status'] == 'completed':
                    print(f"  ✅ Completed in {result['evaluation_time']/60:.1f} min")
                else:
                    print(f"  ❌ Failed: {result['error']}")
                    self.failed_evaluations.append((checkpoint_id, result['error']))
                
                pbar.update(1)
    
    def _run_parallel_evaluation(self, checkpoints: List[Tuple[Path, str]], 
                                datasets: List[str], use_tta: bool, tta_mode: str,
                                save_images: bool, max_workers: int, sliding_window: bool,
                                overlap: float, blend_mode: str, boundary_refine: bool,
                                refine_kernel: int, adaptive_threshold: bool) -> None:
        """Run evaluations in parallel"""
        
        print(f"⚡ Running parallel evaluation ({max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_checkpoint = {
                executor.submit(
                    self.evaluate_single_checkpoint,
                    checkpoint_dir, checkpoint_id, datasets, use_tta, tta_mode, save_images,
                    sliding_window, overlap, blend_mode, boundary_refine, refine_kernel,
                    adaptive_threshold
                ): (checkpoint_dir, checkpoint_id) 
                for checkpoint_dir, checkpoint_id in checkpoints
            }
            
            completed = 0
            for future in as_completed(future_to_checkpoint):
                checkpoint_dir, checkpoint_id = future_to_checkpoint[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results[checkpoint_id] = result
                    
                    status = "✅" if result['status'] == 'completed' else "❌"
                    print(f"[{completed}/{len(checkpoints)}] {status} {checkpoint_dir.name} "
                          f"({result['evaluation_time']/60:.1f}min)")
                    
                except Exception as e:
                    print(f"[{completed}/{len(checkpoints)}] ❌ {checkpoint_dir.name} - {e}")
                    self.failed_evaluations.append((checkpoint_id, str(e)))


def run_visualization(dataset_flags: List[str], use_tta: bool, tta_mode: str) -> bool:
    """
    Trigger visualize_checkpoint_metrics.py with the same flags
    
    Args:
        dataset_flags: Dataset flags from evaluation
        use_tta: Whether TTA was used
        tta_mode: TTA mode if used
        
    Returns:
        True if visualization succeeded, False otherwise
    """
    print("\n" + "="*80)
    print("📊 GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    try:
        # Build command for visualize_checkpoint_metrics.py
        cmd = ['python', 'visualize_checkpoint_metrics.py']
        
        # Add dataset flags
        cmd.extend(dataset_flags)
        
        # Add TTA flag if used
        if use_tta:
            cmd.extend(['--tta-mode', tta_mode])
        
        print(f"🔧 Running: {' '.join(cmd)}")
        print()
        
        # Run visualization
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            check=True
        )
        
        print("\n✅ Visualization completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Visualization failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Visualization error: {e}")
        return False


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Batch evaluation of all Adipose U-Net checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoints on clean-test with stain + TTA
  python evaluate_all_checkpoints.py --clean-test --stain --use-tta --tta-mode full

  # Quick evaluation without TTA
  python evaluate_all_checkpoints.py --clean-test --original

  # Parallel processing
  python evaluate_all_checkpoints.py --clean-test --stain --parallel --max-workers 3

After evaluation, automatically runs visualize_checkpoint_metrics.py to generate plots.
        """
    )
    
    # Dataset selection flags
    parser.add_argument('--val', action='store_true',
                       help='Evaluate val dataset')
    parser.add_argument('--test', action='store_true',
                       help='Evaluate test dataset')
    parser.add_argument('--human-test', action='store_true',
                       help='Evaluate human_test dataset')
    parser.add_argument('--clean-test', action='store_true',
                       help='Evaluate clean_test dataset')
    
    # Data source (required)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--stain', action='store_true',
                       help='Use stain normalized data')
    data_group.add_argument('--original', action='store_true',
                       help='Use original data')
    
    # TTA configuration
    parser.add_argument('--use-tta', action='store_true', default=False,
                       help='Enable Test Time Augmentation')
    parser.add_argument('--tta-mode', type=str, default='basic',
                       choices=['minimal', 'basic', 'full'],
                       help='TTA mode (default: basic)')
    
    # Enhanced post-processing options
    parser.add_argument('--sliding-window', action='store_true', default=False,
                       help='Enable sliding window inference with overlapping tiles')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for sliding window (0.0-0.75, default: 0.5)')
    parser.add_argument('--blend-mode', type=str, default='gaussian',
                       choices=['gaussian', 'linear', 'none'],
                       help='Blending mode for sliding window (default: gaussian)')
    parser.add_argument('--boundary-refine', action='store_true', default=False,
                       help='Enable morphological boundary refinement')
    parser.add_argument('--refine-kernel', type=int, default=5,
                       help='Kernel size for boundary refinement (default: 5)')
    parser.add_argument('--adaptive-threshold', action='store_true', default=False,
                       help='Use adaptive two-stage threshold optimization')
    
    # Output options
    parser.add_argument('--no-images', action='store_true',
                       help='Skip saving individual visualization images')
    
    # Processing options
    parser.add_argument('--parallel', action='store_true', default=False,
                       help='Enable parallel processing (default: False for stability)')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Max parallel workers (default: 2)')
    
    args = parser.parse_args()
    
    # Build dataset flags
    dataset_flags = []
    dataset_names = []
    
    if args.val:
        dataset_flags.append('--val')
        dataset_names.append('val')
    if args.test:
        dataset_flags.append('--test')
        dataset_names.append('test')
    if args.human_test:
        dataset_flags.append('--human-test')
        dataset_names.append('human_test')
    if args.clean_test:
        dataset_flags.append('--clean-test')
        dataset_names.append('clean_test')
    
    if not dataset_flags:
        print("❌ No dataset specified. Use --val, --test, --human-test, or --clean-test")
        return 1
    
    # Add data source
    data_source = 'stain' if args.stain else 'original'
    dataset_flags.append('--stain' if args.stain else '--original')
    
    # Setup GPU
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    
    print("🧬 Adipose U-Net Batch Checkpoint Evaluator")
    print("=" * 80)
    
    try:
        # Initialize and run evaluations
        evaluator = CheckpointBatchEvaluator(
            dataset_names=dataset_names,
            data_source=data_source,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode
        )
        
        success = evaluator.run_batch_evaluation(
            datasets=dataset_flags,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode,
            save_images=not args.no_images,
            parallel=args.parallel,
            max_workers=args.max_workers,
            sliding_window=args.sliding_window,
            overlap=args.overlap,
            blend_mode=args.blend_mode,
            boundary_refine=args.boundary_refine,
            refine_kernel=args.refine_kernel,
            adaptive_threshold=args.adaptive_threshold
        )
        
        if not success:
            print("\n⚠️  No successful evaluations, skipping visualization")
            return 1
        
        # Run visualization
        viz_success = run_visualization(dataset_flags, args.use_tta, args.tta_mode)
        
        if viz_success:
            print("\n🎉 Pipeline completed successfully!")
            return 0
        else:
            print("\n⚠️  Evaluations succeeded but visualization failed")
            return 0  # Still return success since evaluations worked
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
