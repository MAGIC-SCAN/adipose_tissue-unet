
# Updated Normalization Methods Comparison - Summary

## Methods Tested:

1. **Original**: Raw grayscale values (0-255) normalized to [0,1] for display
2. **CLAHE Only**: clipLimit=2.0, tileGridSize=(8,8), normalized to [0,1]
3. **Percentile Only**: 1st-99th percentile normalization 
4. **CLAHE + Percentile (Aggressive)**: CLAHE + 1st-99th percentile (from previous analysis)
5. **Gentle Percentile**: 10th-90th percentile normalization (wider range)
6. **Light CLAHE + Wider Percentile**: clipLimit=1.2, tileGridSize=(16,16) + 5th-95th percentile

## Detail Loss Analysis:

Red highlighted regions indicate potential problems:
- Areas where local variance significantly decreased compared to original
- Overly dark regions (< 5th percentile of processed image)
- Connected regions ≥300 pixels are outlined with red boxes

## Recommendations Based on Visual Analysis:

### Conservative Approaches (Lower risk of detail loss):
- **Gentle Percentile (10-90)**: Safest improvement over original
- **Light CLAHE + Wider Percentile**: Good balance of enhancement vs safety

### Moderate Risk:
- **Percentile Only (1-99)**: Good contrast but may clip important details
- **CLAHE Only**: Improves local contrast but can create artifacts

### Higher Risk:
- **CLAHE + Percentile (Aggressive)**: Best contrast but highest detail loss risk

## Implementation Recommendation:

Start with **Light CLAHE + Wider Percentile** as it provides:
- Improved local contrast from gentle CLAHE
- Robust normalization from percentile method
- Lower risk of losing important tissue details
- Better handling of staining variations

If results are positive, can experiment with more aggressive methods.

## Code for Recommended Method:

```python
# Light CLAHE + Wider Percentile (Recommended)
img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
img = clahe.apply(img_uint8).astype(np.float32)
p5, p95 = np.percentile(img, (5, 95))
img = np.clip((img - p5) / (p95 - p5 + 1e-3), 0, 1)
```
