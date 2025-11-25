#!/usr/bin/env python3
"""
Stain Normalization Validation Framework
========================================

This script validates the performance of selected reference images by testing
Reinhard normalization on diverse samples and measuring preservation of
critical metrics for adipocyte segmentation.

Features:
- Cross-validation testing on diverse dataset samples
- Metric preservation analysis
- Similarity to adipocyte reference standards
- Performance comparison across different references
- Comprehensive validation reports

Author: Analysis Pipeline
Date: 2024-10-29
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from reinhard_stain_normalization import ReinhardStainNormalizer, complete_preprocessing_pipeline
from select_stain_reference import SybrGoldEosinReferenceSelector


class StainNormalizationValidator:
    """
    Comprehensive validation framework for stain normalization
    """
    
    def __init__(self, 
                 test_dataset_dir="example_unet_tiles/gtex/trn_imgs",
                 reference_standards_path="preprocessing_analysis/adipocyte_reference_metrics.csv",
                 output_dir="preprocessing_analysis/stain_normalization"):
        
        self.test_dataset_dir = Path(test_dataset_dir)
        self.reference_standards_path = Path(reference_standards_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load adipocyte reference standards
        self.adipocyte_standards = self._load_adipocyte_standards()
        
    def _load_adipocyte_standards(self):
        """Load adipocyte reference standards from previous analysis"""
        try:
            if self.reference_standards_path.exists():
                standards_df = pd.read_csv(self.reference_standards_path)
                standards = {
                    'laplacian_variance': standards_df['laplacian_variance'].mean(),
                    'local_contrast_consistency': standards_df['local_contrast_consistency'].mean(),
                    'entropy': standards_df['entropy'].mean(),
                    'edge_density': standards_df['edge_density'].mean()
                }
                print(f"Loaded adipocyte standards from {self.reference_standards_path}")
                return standards
            else:
                # Fallback standards based on our previous analysis
                print("Using fallback adipocyte standards")
                return {
                    'laplacian_variance': 0.20,
                    'local_contrast_consistency': 0.15,
                    'entropy': 0.30,
                    'edge_density': 0.03
                }
        except Exception as e:
            print(f"Error loading standards: {e}")
            return {
                'laplacian_variance': 0.20,
                'local_contrast_consistency': 0.15,
                'entropy': 0.30,
                'edge_density': 0.03
            }
    
    def sample_diverse_test_images(self, n_samples=20):
        """Sample diverse test images from dataset"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        
        # Get all available images
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(self.test_dataset_dir.glob(f"*{ext}")))
            all_images.extend(list(self.test_dataset_dir.glob(f"*{ext.upper()}")))
        
        if len(all_images) == 0:
            # Fallback to example tiles
            fallback_dir = Path("example_tiles")
            if fallback_dir.exists():
                for ext in image_extensions:
                    all_images.extend(list(fallback_dir.glob(f"*{ext}")))
        
        # Sample diverse subset
        if len(all_images) > n_samples:
            # Select images with diverse names to get variety
            step = len(all_images) // n_samples
            sampled_images = all_images[::step][:n_samples]
        else:
            sampled_images = all_images
        
        print(f"Sampled {len(sampled_images)} test images from {len(all_images)} available")
        return sampled_images
    
    def calculate_image_metrics(self, image):
        """Calculate comprehensive image quality metrics"""
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        metrics = {}
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['laplacian_variance'] = laplacian.var()
        
        # Entropy (information content)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        probabilities = hist / hist.sum()
        metrics['entropy'] = -np.sum(probabilities * np.log2(probabilities))
        
        # Local contrast consistency
        metrics['local_contrast_consistency'] = self._calculate_local_contrast_consistency(gray)
        
        # Edge density
        from skimage.feature import canny
        edges = canny(gray, sigma=1.0)
        metrics['edge_density'] = np.sum(edges) / edges.size
        
        # Mean and std intensity
        metrics['mean_intensity'] = gray.mean()
        metrics['std_intensity'] = gray.std()
        
        return metrics
    
    def _calculate_local_contrast_consistency(self, gray_image):
        """Calculate local contrast consistency"""
        gray = gray_image.astype(np.float32)
        
        # Calculate local contrast in overlapping patches
        patch_size = 64
        step_size = 32
        contrasts = []
        
        h, w = gray.shape
        for i in range(0, h - patch_size + 1, step_size):
            for j in range(0, w - patch_size + 1, step_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                if patch.std() > 0:
                    local_contrast = (patch.max() - patch.min()) / (patch.max() + patch.min() + 1e-10)
                    contrasts.append(local_contrast)
        
        if len(contrasts) > 0:
            consistency = 1.0 / (np.std(contrasts) / (np.mean(contrasts) + 1e-10) + 1e-10)
            return min(consistency, 1000)
        else:
            return 0.0
    
    def calculate_similarity_to_adipocyte_standards(self, metrics):
        """Calculate similarity to adipocyte reference standards"""
        similarities = {}
        overall_similarity = 0
        
        for metric_name, standard_value in self.adipocyte_standards.items():
            if metric_name in metrics:
                # Calculate relative difference
                image_value = metrics[metric_name]
                if standard_value != 0:
                    relative_diff = abs(image_value - standard_value) / standard_value
                    similarity = max(0, 1 - relative_diff)  # 1 = perfect match, 0 = very different
                else:
                    similarity = 1.0 if image_value == 0 else 0.0
                
                similarities[metric_name] = similarity
                overall_similarity += similarity
        
        # Average similarity
        if len(similarities) > 0:
            overall_similarity /= len(similarities)
        
        similarities['overall_similarity'] = overall_similarity
        return similarities
    
    def validate_single_reference(self, reference_path, test_images):
        """Validate a single reference image on test set"""
        print(f"\nValidating reference: {Path(reference_path).name}")
        
        # Create normalizer with this reference
        normalizer = ReinhardStainNormalizer(reference_path)
        
        results = []
        
        for i, test_image_path in enumerate(test_images):
            print(f"Processing test image {i+1}/{len(test_images)}")
            
            try:
                # Load original image
                original_image = np.array(Image.open(test_image_path))
                
                # Apply normalization
                normalized_image = normalizer.normalize_image(original_image)
                
                # Calculate metrics for both images
                original_metrics = self.calculate_image_metrics(original_image)
                normalized_metrics = self.calculate_image_metrics(normalized_image)
                
                # Calculate preservation ratios
                preservation = {}
                for metric in ['laplacian_variance', 'entropy', 'local_contrast_consistency']:
                    if original_metrics[metric] != 0:
                        preservation[f'{metric}_preservation'] = normalized_metrics[metric] / original_metrics[metric]
                    else:
                        preservation[f'{metric}_preservation'] = 1.0
                
                # Calculate similarity to adipocyte standards
                original_similarity = self.calculate_similarity_to_adipocyte_standards(original_metrics)
                normalized_similarity = self.calculate_similarity_to_adipocyte_standards(normalized_metrics)
                
                # Store results
                result = {
                    'test_image': test_image_path.name,
                    'original_metrics': original_metrics,
                    'normalized_metrics': normalized_metrics,
                    'preservation': preservation,
                    'original_similarity': original_similarity,
                    'normalized_similarity': normalized_similarity,
                    'similarity_improvement': normalized_similarity['overall_similarity'] - original_similarity['overall_similarity']
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {test_image_path.name}: {e}")
                continue
        
        return results
    
    def compare_multiple_references(self, reference_candidates, test_images):
        """Compare performance of multiple reference candidates"""
        all_results = {}
        
        for ref_path in reference_candidates:
            ref_name = Path(ref_path).name
            results = self.validate_single_reference(ref_path, test_images)
            all_results[ref_name] = results
        
        return all_results
    
    def generate_validation_report(self, validation_results, reference_metadata=None):
        """Generate comprehensive validation report"""
        
        # Calculate aggregate statistics for each reference
        reference_performance = {}
        
        for ref_name, results in validation_results.items():
            if not results:
                continue
                
            # Aggregate metrics
            preservation_metrics = []
            similarity_improvements = []
            final_similarities = []
            
            for result in results:
                # Preservation metrics
                preservation_metrics.append([
                    result['preservation']['laplacian_variance_preservation'],
                    result['preservation']['entropy_preservation'],
                    result['preservation']['local_contrast_consistency_preservation']
                ])
                
                # Similarity metrics
                similarity_improvements.append(result['similarity_improvement'])
                final_similarities.append(result['normalized_similarity']['overall_similarity'])
            
            preservation_array = np.array(preservation_metrics)
            
            # Calculate summary statistics
            reference_performance[ref_name] = {
                'n_samples': len(results),
                'avg_sharpness_preservation': preservation_array[:, 0].mean(),
                'avg_entropy_preservation': preservation_array[:, 1].mean(),
                'avg_contrast_preservation': preservation_array[:, 2].mean(),
                'overall_preservation': preservation_array.mean(),
                'avg_similarity_improvement': np.mean(similarity_improvements),
                'avg_final_similarity': np.mean(final_similarities),
                'preservation_stability': 1.0 / (preservation_array.std() + 1e-10)  # Lower std = more stable
            }
            
            # Composite performance score
            perf = reference_performance[ref_name]
            composite_score = (
                perf['overall_preservation'] * 0.4 +
                perf['avg_final_similarity'] * 0.4 +
                min(perf['preservation_stability'] / 10, 1.0) * 0.2
            )
            reference_performance[ref_name]['composite_performance'] = composite_score
        
        # Rank references by performance
        ranked_references = sorted(reference_performance.items(), 
                                 key=lambda x: x[1]['composite_performance'], 
                                 reverse=True)
        
        return reference_performance, ranked_references
    
    def create_validation_visualizations(self, validation_results, reference_performance):
        """Create comprehensive visualization of validation results"""
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SYBR Gold + Eosin Stain Normalization Validation', fontsize=16)
        
        # 1. Reference performance comparison
        ref_names = list(reference_performance.keys())
        composite_scores = [reference_performance[name]['composite_performance'] for name in ref_names]
        
        axes[0, 0].bar(range(len(ref_names)), composite_scores, color='steelblue')
        axes[0, 0].set_xlabel('Reference Images')
        axes[0, 0].set_ylabel('Composite Performance Score')
        axes[0, 0].set_title('Overall Reference Performance')
        axes[0, 0].set_xticks(range(len(ref_names)))
        axes[0, 0].set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in ref_names], 
                                  rotation=45, ha='right')
        
        # 2. Metric preservation comparison
        preservation_metrics = ['avg_sharpness_preservation', 'avg_entropy_preservation', 'avg_contrast_preservation']
        preservation_data = []
        
        for ref_name in ref_names:
            for metric in preservation_metrics:
                preservation_data.append({
                    'Reference': ref_name[:15] + '...' if len(ref_name) > 15 else ref_name,
                    'Metric': metric.replace('avg_', '').replace('_preservation', ''),
                    'Preservation': reference_performance[ref_name][metric]
                })
        
        preservation_df = pd.DataFrame(preservation_data)
        
        # Create heatmap
        pivot_df = preservation_df.pivot(index='Reference', columns='Metric', values='Preservation')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', center=1.0, ax=axes[0, 1])
        axes[0, 1].set_title('Metric Preservation (1.0 = Perfect)')
        
        # 3. Similarity improvement
        similarity_improvements = [reference_performance[name]['avg_similarity_improvement'] for name in ref_names]
        colors = ['green' if x > 0 else 'red' for x in similarity_improvements]
        
        axes[1, 0].bar(range(len(ref_names)), similarity_improvements, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Reference Images')
        axes[1, 0].set_ylabel('Avg Similarity Improvement')
        axes[1, 0].set_title('Improvement in Adipocyte Similarity')
        axes[1, 0].set_xticks(range(len(ref_names)))
        axes[1, 0].set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in ref_names], 
                                  rotation=45, ha='right')
        
        # 4. Final similarity scores
        final_similarities = [reference_performance[name]['avg_final_similarity'] for name in ref_names]
        
        axes[1, 1].bar(range(len(ref_names)), final_similarities, color='lightcoral', alpha=0.7)
        axes[1, 1].set_xlabel('Reference Images')
        axes[1, 1].set_ylabel('Final Similarity Score')
        axes[1, 1].set_title('Final Adipocyte Similarity')
        axes[1, 1].set_xticks(range(len(ref_names)))
        axes[1, 1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in ref_names], 
                                  rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'stain_normalization_validation.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Validation visualization saved to: {viz_path}")
        
        return fig, viz_path
    
    def save_validation_results(self, validation_results, reference_performance, ranked_references):
        """Save comprehensive validation results"""
        
        # Save performance summary
        performance_df = pd.DataFrame.from_dict(reference_performance, orient='index')
        performance_path = self.output_dir / 'reference_performance_summary.csv'
        performance_df.to_csv(performance_path)
        
        # Save detailed results
        detailed_results = []
        for ref_name, results in validation_results.items():
            for result in results:
                row = {
                    'reference': ref_name,
                    'test_image': result['test_image'],
                    'original_sharpness': result['original_metrics']['laplacian_variance'],
                    'normalized_sharpness': result['normalized_metrics']['laplacian_variance'],
                    'sharpness_preservation': result['preservation']['laplacian_variance_preservation'],
                    'original_entropy': result['original_metrics']['entropy'],
                    'normalized_entropy': result['normalized_metrics']['entropy'],
                    'entropy_preservation': result['preservation']['entropy_preservation'],
                    'original_similarity': result['original_similarity']['overall_similarity'],
                    'normalized_similarity': result['normalized_similarity']['overall_similarity'],
                    'similarity_improvement': result['similarity_improvement']
                }
                detailed_results.append(row)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = self.output_dir / 'detailed_validation_results.csv'
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save validation report
        report_path = self.output_dir / 'stain_normalization_validation_report.md'
        with open(report_path, 'w') as f:
            f.write("# SYBR Gold + Eosin Stain Normalization Validation Report\n\n")
            f.write(f"Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Validation Summary\n")
            f.write(f"- Total references tested: {len(validation_results)}\n")
            f.write(f"- Test images per reference: {len(list(validation_results.values())[0]) if validation_results else 0}\n")
            f.write(f"- Best performing reference: **{ranked_references[0][0]}**\n")
            f.write(f"- Best composite performance: {ranked_references[0][1]['composite_performance']:.3f}\n\n")
            
            f.write("## Reference Performance Ranking\n")
            for i, (ref_name, performance) in enumerate(ranked_references):
                f.write(f"{i+1}. **{ref_name}** (Score: {performance['composite_performance']:.3f})\n")
                f.write(f"   - Preservation: {performance['overall_preservation']:.3f}\n")
                f.write(f"   - Final Similarity: {performance['avg_final_similarity']:.3f}\n")
                f.write(f"   - Similarity Improvement: {performance['avg_similarity_improvement']:.3f}\n\n")
            
            f.write("## Key Findings\n")
            best_ref = ranked_references[0][1]
            f.write(f"- Average metric preservation: {best_ref['overall_preservation']:.1%}\n")
            f.write(f"- Average similarity to adipocyte standards: {best_ref['avg_final_similarity']:.1%}\n")
            f.write(f"- Average improvement over no normalization: {best_ref['avg_similarity_improvement']:.3f}\n")
            
            if best_ref['avg_similarity_improvement'] > 0:
                f.write("- **Recommendation**: Stain normalization improves adipocyte similarity\n")
            else:
                f.write("- **Recommendation**: Consider alternative normalization approaches\n")
        
        print(f"Validation results saved to {self.output_dir}")
        return report_path, performance_path, detailed_path
    
    def run_validation(self, reference_candidates=None, n_test_samples=15):
        """Run complete validation pipeline"""
        print("Starting SYBR Gold + Eosin stain normalization validation...")
        
        # Get test images
        test_images = self.sample_diverse_test_images(n_test_samples)
        
        if not test_images:
            print("No test images found!")
            return None
        
        # Get reference candidates if not provided
        if reference_candidates is None:
            # Use all example tiles as candidates
            candidate_dir = Path("example_tiles")
            if candidate_dir.exists():
                image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
                reference_candidates = []
                for ext in image_extensions:
                    reference_candidates.extend(list(candidate_dir.glob(f"*{ext}")))
                reference_candidates = reference_candidates[:5]  # Limit to top 5
            else:
                print("No reference candidates found!")
                return None
        
        # Run validation on all references
        validation_results = self.compare_multiple_references(reference_candidates, test_images)
        
        # Generate performance analysis
        reference_performance, ranked_references = self.generate_validation_report(validation_results)
        
        # Create visualizations
        fig, viz_path = self.create_validation_visualizations(validation_results, reference_performance)
        
        # Save results
        report_path, performance_path, detailed_path = self.save_validation_results(
            validation_results, reference_performance, ranked_references)
        
        # Print summary
        print("\n" + "="*60)
        print("SYBR Gold + Eosin Stain Normalization Validation Complete!")
        print("="*60)
        if ranked_references:
            best_ref = ranked_references[0]
            print(f"Best reference: {best_ref[0]}")
            print(f"Composite performance: {best_ref[1]['composite_performance']:.3f}")
            print(f"Metric preservation: {best_ref[1]['overall_preservation']:.1%}")
            print(f"Adipocyte similarity: {best_ref[1]['avg_final_similarity']:.1%}")
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return {
            'validation_results': validation_results,
            'reference_performance': reference_performance,
            'ranked_references': ranked_references,
            'best_reference': ranked_references[0][0] if ranked_references else None
        }


def main():
    """Main execution function"""
    print("SYBR Gold + Eosin Stain Normalization Validation")
    print("="*50)
    
    validator = StainNormalizationValidator()
    results = validator.run_validation()
    
    if results and results['best_reference']:
        print(f"\nRecommended reference for stain normalization: {results['best_reference']}")
        return results['best_reference']
    else:
        print("Validation failed or no suitable reference found!")
        return None


if __name__ == "__main__":
    main()
