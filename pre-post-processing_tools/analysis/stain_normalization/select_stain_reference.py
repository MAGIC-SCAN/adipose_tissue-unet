#!/usr/bin/env python3
"""
SYBR Gold + Eosin Stain Reference Selection for Adipocyte Segmentation
=====================================================================

This script analyzes potential reference images for Reinhard stain normalization
specifically optimized for SYBR Gold + Eosin staining protocol.

Key differences from H&E:
- SYBR Gold: Golden/yellow fluorescence (nuclei) vs. Purple/blue hematoxylin
- Eosin: Pink/red (cytoplasm) - same as H&E
- Result: Yellow-to-pink color scheme instead of purple-to-pink

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
from skimage import filters, feature, measure
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class SybrGoldEosinReferenceSelector:
    """
    SYBR Gold + Eosin specific reference image selection for stain normalization
    """
    
    def __init__(self, candidate_dir="/home/luci/Data_for_ML/Meat_Luci_Tulane/Pseudocolored", output_dir="preprocessing_analysis/stain_normalization"):
        self.candidate_dir = Path(candidate_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds based on our previous analysis
        self.quality_thresholds = {
            'min_laplacian_variance': 0.15,
            'min_entropy': 0.25,
            'min_local_contrast': 0.1,
            'max_edge_density': 0.05
        }
        
        # SYBR Gold + Eosin specific color ranges
        self.stain_colors = {
            'sybr_gold': {
                'hue_range': (40, 80),  # Golden/yellow range in HSV
                'description': 'SYBR Gold (nuclei, golden-yellow)'
            },
            'eosin': {
                'hue_range': [(0, 20), (160, 180)],  # Pink/red range in HSV
                'description': 'Eosin (cytoplasm, pink-red)'
            }
        }
        
    def load_candidates(self):
        """Load all candidate images from the example tiles directory"""
        candidates = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        
        for img_path in self.candidate_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                try:
                    img = np.array(Image.open(img_path))
                    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
                        candidates.append({
                            'path': str(img_path),
                            'name': img_path.name,
                            'image': img
                        })
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    
        print(f"Loaded {len(candidates)} candidate images for SYBR Gold + Eosin analysis")
        return candidates
    
    def calculate_sharpness(self, image):
        """Calculate Laplacian variance (sharpness metric)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def calculate_entropy(self, image):
        """Calculate Shannon entropy (information content)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Remove zero bins
        probabilities = hist / hist.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def calculate_local_contrast_consistency(self, image):
        """Calculate local contrast consistency across image regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Calculate local contrast in overlapping patches
        patch_size = 64
        step_size = 32
        contrasts = []
        
        h, w = gray.shape
        for i in range(0, h - patch_size + 1, step_size):
            for j in range(0, w - patch_size + 1, step_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                if patch.std() > 0:  # Avoid empty patches
                    local_contrast = (patch.max() - patch.min()) / (patch.max() + patch.min() + 1e-10)
                    contrasts.append(local_contrast)
        
        if len(contrasts) > 0:
            # Consistency is inverse of coefficient of variation
            consistency = 1.0 / (np.std(contrasts) / (np.mean(contrasts) + 1e-10) + 1e-10)
            return min(consistency, 1000)  # Cap at reasonable value
        else:
            return 0.0
    
    def calculate_edge_density(self, image):
        """Calculate edge density using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = feature.canny(gray, sigma=1.0)
        return np.sum(edges) / edges.size
    
    def analyze_lab_colorspace(self, image):
        """Analyze color characteristics in LAB space for SYBR Gold + Eosin"""
        # Convert to LAB
        lab = rgb2lab(image / 255.0)  # skimage expects [0,1] range
        
        # Calculate statistics for each channel
        l_stats = {'mean': lab[:,:,0].mean(), 'std': lab[:,:,0].std()}
        a_stats = {'mean': lab[:,:,1].mean(), 'std': lab[:,:,1].std()}
        b_stats = {'mean': lab[:,:,2].mean(), 'std': lab[:,:,2].std()}
        
        return {
            'L': l_stats,  # Lightness
            'A': a_stats,  # Green-Red
            'B': b_stats   # Blue-Yellow (important for SYBR Gold detection)
        }
    
    def assess_sybr_eosin_separation(self, image):
        """Assess SYBR Gold + Eosin stain separation quality"""
        lab = rgb2lab(image / 255.0)
        
        # SYBR Gold + Eosin should create good separation in A and B channels
        # SYBR Gold (yellow) has positive B values
        # Eosin (pink/red) has positive A values
        a_channel = lab[:,:,1]  # Green-Red axis
        b_channel = lab[:,:,2]  # Blue-Yellow axis
        
        # Calculate separation metrics
        a_range = a_channel.max() - a_channel.min()
        b_range = b_channel.max() - b_channel.min()
        a_var = a_channel.var()
        b_var = b_channel.var()
        
        # For SYBR Gold + Eosin, we expect:
        # - High B channel variance (SYBR Gold creates yellow)
        # - Good A channel variance (Eosin creates pink/red)
        # - B channel should have positive bias (yellow dominance)
        b_bias = b_channel.mean()  # Should be positive for SYBR Gold
        
        # Calculate separation score emphasizing B channel for SYBR Gold
        separation_score = (a_range * b_range) * (a_var * b_var) * (1 + max(0, b_bias))
        
        return {
            'a_range': a_range,
            'b_range': b_range,
            'a_variance': a_var,
            'b_variance': b_var,
            'b_bias': b_bias,
            'separation_score': separation_score
        }
    
    def evaluate_sybr_eosin_balance(self, image):
        """Evaluate color balance for SYBR Gold + Eosin staining"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Analyze hue distribution
        hue = hsv[:,:,0]
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # SYBR Gold + Eosin specific hue ranges:
        # SYBR Gold (golden/yellow): ~45-75 degrees -> ~25-42 in OpenCV (180 scale)
        # Eosin (pink/red): ~300-360 and 0-30 degrees -> ~166-180 and 0-17 in OpenCV
        
        hue_hist, _ = np.histogram(hue, bins=180, range=(0, 180))
        
        # Calculate balance metrics for SYBR Gold + Eosin
        golden_region = np.sum(hue_hist[25:42])   # Golden/yellow region (SYBR Gold)
        pink_region = np.sum(hue_hist[0:17]) + np.sum(hue_hist[166:180])  # Pink/red region (Eosin)
        
        total_pixels = hue.size
        golden_ratio = golden_region / total_pixels
        pink_ratio = pink_region / total_pixels
        
        # Good balance should have both SYBR Gold and Eosin present
        balance_score = min(golden_ratio, pink_ratio) * 2  # Multiply by 2 to normalize
        
        # Calculate color separation quality
        color_separation = abs(golden_ratio - pink_ratio)  # Lower is better for balance
        
        return {
            'golden_ratio': golden_ratio,
            'pink_ratio': pink_ratio,
            'balance_score': balance_score,
            'color_separation': color_separation,
            'saturation_mean': saturation.mean(),
            'value_mean': value.mean()
        }
    
    def estimate_adipocyte_coverage(self, image):
        """Estimate percentage of image covered by adipocytes"""
        # Adipocytes appear as large, light, circular regions with thin boundaries
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold for light regions (adipocytes are typically lighter)
        light_threshold = np.percentile(gray, 70)
        light_mask = gray > light_threshold
        
        # Apply morphological operations to get blob-like regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        cleaned_mask = cv2.morphologyEx(light_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Estimate coverage
        coverage = np.sum(cleaned_mask > 0) / cleaned_mask.size
        
        return coverage
    
    def calculate_structure_variety(self, image):
        """Calculate variety of structures in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local binary patterns for texture variety
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            # Normalize histogram
            lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-10)
            # Calculate entropy of LBP histogram as measure of variety
            variety = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        except ImportError:
            # Fallback: use gradient variance
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            variety = gradient_magnitude.var()
            
        return variety
    
    def assess_background_quality(self, image):
        """Assess background cleanliness (absence of artifacts)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect potential background (very light or very dark regions)
        background_mask = (gray < 30) | (gray > 220)
        
        if np.sum(background_mask) == 0:
            return 1.0  # No clear background
        
        # Calculate variance in background regions
        background_variance = gray[background_mask].var()
        
        # Good background should have low variance (uniform)
        # Normalize by expected variance for clean background
        quality_score = max(0, 1 - background_variance / 100)
        
        return quality_score
    
    def check_quality_threshold(self, metrics):
        """Check if image meets minimum quality thresholds"""
        return (
            metrics['laplacian_variance'] >= self.quality_thresholds['min_laplacian_variance'] and
            metrics['entropy'] >= self.quality_thresholds['min_entropy'] and
            metrics['local_contrast_consistency'] >= self.quality_thresholds['min_local_contrast'] and
            metrics['edge_density'] <= self.quality_thresholds['max_edge_density']
        )
    
    def calculate_composite_score(self, metrics):
        """Calculate composite score for SYBR Gold + Eosin reference quality"""
        weights = {
            'technical_quality': 0.4,  # Sharpness, contrast, etc.
            'color_characteristics': 0.35,  # LAB stats, stain separation
            'biological_relevance': 0.25   # Tissue content, structure variety
        }
        
        # Technical quality (normalized to 0-1)
        technical = (
            min(metrics['laplacian_variance'] / 0.3, 1.0) * 0.3 +
            min(metrics['entropy'] / 8.0, 1.0) * 0.3 +
            min(metrics['local_contrast_consistency'] / 1.0, 1.0) * 0.2 +
            max(0, 1 - metrics['edge_density'] / 0.05) * 0.2
        )
        
        # Color characteristics (updated for SYBR Gold + Eosin)
        color_quality = (
            min(metrics['lab_stats']['B']['std'] / 15.0, 1.0) * 0.4 +  # B channel important for SYBR Gold
            min(metrics['stain_separation']['separation_score'] / 2000.0, 1.0) * 0.4 +
            metrics['color_balance']['balance_score'] * 0.2
        )
        
        # Biological relevance
        biological = (
            metrics['adipocyte_coverage'] * 0.4 +
            min(metrics['structure_variety'] / 3.0, 1.0) * 0.3 +
            metrics['background_quality'] * 0.3
        )
        
        # Combine with weights
        composite_score = (
            technical * weights['technical_quality'] +
            color_quality * weights['color_characteristics'] +
            biological * weights['biological_relevance']
        )
        
        return {
            'composite_score': composite_score,
            'technical_quality': technical,
            'color_characteristics': color_quality,
            'biological_relevance': biological
        }
    
    def analyze_all_candidates(self):
        """Analyze all candidate images and rank them"""
        candidates = self.load_candidates()
        results = []
        
        print("Analyzing candidate images for SYBR Gold + Eosin stain normalization...")
        for i, candidate in enumerate(candidates):
            print(f"Processing {candidate['name']} ({i+1}/{len(candidates)})")
            
            try:
                metrics = {
                    'laplacian_variance': self.calculate_sharpness(candidate['image']),
                    'entropy': self.calculate_entropy(candidate['image']),
                    'local_contrast_consistency': self.calculate_local_contrast_consistency(candidate['image']),
                    'edge_density': self.calculate_edge_density(candidate['image']),
                    'lab_stats': self.analyze_lab_colorspace(candidate['image']),
                    'stain_separation': self.assess_sybr_eosin_separation(candidate['image']),
                    'color_balance': self.evaluate_sybr_eosin_balance(candidate['image']),
                    'adipocyte_coverage': self.estimate_adipocyte_coverage(candidate['image']),
                    'structure_variety': self.calculate_structure_variety(candidate['image']),
                    'background_quality': self.assess_background_quality(candidate['image'])
                }
                
                scores = self.calculate_composite_score(metrics)
                meets_threshold = self.check_quality_threshold(metrics)
                
                results.append({
                    'name': candidate['name'],
                    'path': candidate['path'],
                    'metrics': metrics,
                    'scores': scores,
                    'meets_quality_threshold': meets_threshold
                })
                
            except Exception as e:
                print(f"Error processing {candidate['name']}: {e}")
                continue
        
        # Sort by composite score
        results.sort(key=lambda x: x['scores']['composite_score'], reverse=True)
        
        return results
    
    def create_analysis_report(self, results):
        """Create comprehensive analysis report"""
        # Create summary statistics
        summary_stats = {
            'total_candidates': len(results),
            'passed_quality_threshold': sum(1 for r in results if r['meets_quality_threshold']),
            'avg_composite_score': np.mean([r['scores']['composite_score'] for r in results]),
            'best_candidate': results[0]['name'] if results else None,
            'best_score': results[0]['scores']['composite_score'] if results else None
        }
        
        # Create detailed results DataFrame
        detailed_data = []
        for result in results:
            row = {
                'name': result['name'],
                'composite_score': result['scores']['composite_score'],
                'technical_quality': result['scores']['technical_quality'],
                'color_characteristics': result['scores']['color_characteristics'],
                'biological_relevance': result['scores']['biological_relevance'],
                'laplacian_variance': result['metrics']['laplacian_variance'],
                'entropy': result['metrics']['entropy'],
                'local_contrast': result['metrics']['local_contrast_consistency'],
                'edge_density': result['metrics']['edge_density'],
                'golden_ratio': result['metrics']['color_balance']['golden_ratio'],
                'pink_ratio': result['metrics']['color_balance']['pink_ratio'],
                'balance_score': result['metrics']['color_balance']['balance_score'],
                'adipocyte_coverage': result['metrics']['adipocyte_coverage'],
                'background_quality': result['metrics']['background_quality'],
                'meets_threshold': result['meets_quality_threshold']
            }
            detailed_data.append(row)
        
        df = pd.DataFrame(detailed_data)
        
        return summary_stats, df
    
    def save_results(self, results, summary_stats, df):
        """Save analysis results to files"""
        # Save summary report
        report_path = self.output_dir / 'stain_reference_selection_report.md'
        with open(report_path, 'w') as f:
            f.write("# SYBR Gold + Eosin Stain Reference Selection Report\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Statistics\n")
            f.write(f"- Total candidates analyzed: {summary_stats['total_candidates']}\n")
            f.write(f"- Candidates passing quality threshold: {summary_stats['passed_quality_threshold']}\n")
            f.write(f"- Average composite score: {summary_stats['avg_composite_score']:.3f}\n")
            f.write(f"- **Recommended reference**: {summary_stats['best_candidate']}\n")
            f.write(f"- **Best composite score**: {summary_stats['best_score']:.3f}\n\n")
            
            f.write("## Top 5 Candidates\n")
            for i, result in enumerate(results[:5]):
                f.write(f"{i+1}. **{result['name']}** (Score: {result['scores']['composite_score']:.3f})\n")
                f.write(f"   - Technical Quality: {result['scores']['technical_quality']:.3f}\n")
                f.write(f"   - Color Characteristics: {result['scores']['color_characteristics']:.3f}\n")
                f.write(f"   - Biological Relevance: {result['scores']['biological_relevance']:.3f}\n")
                f.write(f"   - SYBR Gold Ratio: {result['metrics']['color_balance']['golden_ratio']:.3f}\n")
                f.write(f"   - Eosin Ratio: {result['metrics']['color_balance']['pink_ratio']:.3f}\n\n")
        
        # Save detailed CSV
        csv_path = self.output_dir / 'stain_reference_candidates.csv'
        df.to_csv(csv_path, index=False)
        
        # Save selected reference metadata
        if results:
            best_result = results[0]
            reference_metadata = {
                'selected_reference': {
                    'name': best_result['name'],
                    'path': best_result['path'],
                    'selection_date': pd.Timestamp.now().isoformat(),
                    'composite_score': best_result['scores']['composite_score'],
                    'stain_type': 'SYBR_Gold_Eosin'
                },
                'metrics': best_result['metrics'],
                'scores': best_result['scores'],
                'quality_thresholds': self.quality_thresholds
            }
            
            metadata_path = self.output_dir / 'selected_reference_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(reference_metadata, f, indent=2, default=str)
        
        print(f"Results saved to {self.output_dir}")
        return report_path, csv_path
    
    def run_analysis(self):
        """Run complete reference selection analysis"""
        print("Starting SYBR Gold + Eosin reference selection analysis...")
        
        # Analyze all candidates
        results = self.analyze_all_candidates()
        
        if not results:
            print("No valid candidates found!")
            return None
        
        # Create analysis report
        summary_stats, df = self.create_analysis_report(results)
        
        # Save results
        report_path, csv_path = self.save_results(results, summary_stats, df)
        
        # Print summary
        print("\n" + "="*60)
        print("SYBR Gold + Eosin Reference Selection Complete!")
        print("="*60)
        print(f"Recommended reference: {summary_stats['best_candidate']}")
        print(f"Composite score: {summary_stats['best_score']:.3f}")
        print(f"Report saved to: {report_path}")
        print(f"Detailed results: {csv_path}")
        
        return results[0]['path'] if results else None


def main():
    """Main execution function"""
    selector = SybrGoldEosinReferenceSelector()
    selected_reference = selector.run_analysis()
    
    if selected_reference:
        print(f"\nSelected reference image: {selected_reference}")
        return selected_reference
    else:
        print("No suitable reference image found!")
        return None


if __name__ == "__main__":
    main()
