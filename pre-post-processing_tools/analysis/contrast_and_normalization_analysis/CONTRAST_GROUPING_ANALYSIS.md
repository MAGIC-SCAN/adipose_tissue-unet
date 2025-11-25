
# Image Quality Analysis and Adaptive CLAHE Cutoffs

## Analysis Results

Based on analysis of 6 sample images from your preprocessing analysis:

### Determined Cutoffs:

**Contrast Ratio (std/mean):**
- Poor Quality (needs CLAHE): < 0.183
- Medium Quality (needs mild CLAHE): 0.183 - 0.267
- Good Quality (percentile only): > 0.267

**Sharpness (Laplacian Variance):**
- Poor: < 15.6
- Medium: 15.6 - 38.2
- Good: > 38.2

### Image Classification:

quality_group
Poor Quality (Needs CLAHE)        2
Good Quality (Percentile Only)    2
Medium Quality (Mild CLAHE)       2

### Adaptive Strategy:

1. **Poor Quality Images**: Apply standard CLAHE (clipLimit=2.0, tileGridSize=(8,8)) + percentile normalization
2. **Medium Quality Images**: Apply mild CLAHE (clipLimit=1.5, tileGridSize=(12,12)) + percentile normalization  
3. **Good Quality Images**: Skip CLAHE, use percentile normalization only

### Implementation:

The adaptive function has been generated in `adaptive_clahe_function.py` and can be directly integrated into your preprocessing pipeline.

### Files Generated:

- `contrast_analysis_grouping.png`: Visualization of quality metrics and grouping
- `image_quality_analysis.csv`: Detailed metrics for all analyzed images
- `adaptive_clahe_function.py`: Ready-to-use adaptive function

This adaptive approach should provide optimal preprocessing for each image quality level while avoiding over-enhancement of already good quality images.
