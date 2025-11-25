
# Requested Normalization Methods Comparison - Summary

## 6 Methods Compared:

1. **Current Z-score**: (img - global_mean) / global_std → Output can be negative
2. **Percentile (0.01-99.99)**: Clips 0.02% of outliers → Output [0,1]
3. **Mild CLAHE + Percentile (0.01-99.99)**: Enhancement + clipping → Output [0,1]
4. **Percentile (0.05-99.95)**: Clips 0.1% of outliers → Output [0,1]
5. **Mild CLAHE + Percentile (0.05-99.95)**: Enhancement + moderate clipping → Output [0,1]
6. **Mild CLAHE + Percentile (0.001-99.999)**: Enhancement + minimal clipping → Output [0,1]

## Key Differences:

### **Range and Distribution:**
- **Z-score**: Can have negative values, centered at 0
- **Percentile methods**: Always [0,1] range, preserves relative intensities
- **CLAHE combinations**: Enhanced contrast + [0,1] range

### **Outlier Handling:**
- **0.001-99.999**: Virtually no clipping (keeps 99.998% of data)
- **0.01-99.99**: Minimal clipping (keeps 99.98% of data)
- **0.05-99.95**: Conservative clipping (keeps 99.9% of data)

### **Contrast Enhancement:**
- **Percentile only**: No contrast enhancement, just normalization
- **CLAHE + Percentile**: Local contrast enhancement + robust normalization

## Expected Characteristics:

### **Current Z-score:**
- Familiar method, maintains statistical properties
- Can produce negative values (may affect ReLU activations)
- Sensitive to outliers in mean/std calculation

### **Percentile (0.01-99.99):**
- Very conservative clipping
- Good dynamic range utilization
- Robust to outliers

### **Mild CLAHE + Percentile (0.01-99.99):**
- Enhanced boundaries + conservative normalization
- Best for poor contrast images
- Minimal data loss

### **Percentile (0.05-99.95):**
- Slightly more aggressive outlier clipping
- Good balance of robustness vs data preservation

### **Mild CLAHE + Percentile (0.05-99.95):**
- Enhanced boundaries + balanced normalization
- Likely optimal for most cases
- Good contrast without artifacts

### **Mild CLAHE + Percentile (0.001-99.999):**
- Enhanced boundaries + minimal clipping
- Preserves almost all original data
- Most conservative approach

## Implementation Code:

```python
# Method 1: Current Z-score
img_norm = (img - 200.99) / (25.26 + 1e-10)

# Method 2: Percentile (0.01-99.99)
p_low, p_high = np.percentile(img, (0.01, 99.99))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 3: Mild CLAHE + Percentile (0.01-99.99)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
p_low, p_high = np.percentile(img, (0.01, 99.99))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 4: Percentile (0.05-99.95)
p_low, p_high = np.percentile(img, (0.05, 99.95))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 5: Mild CLAHE + Percentile (0.05-99.95) [RECOMMENDED]
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
p_low, p_high = np.percentile(img, (0.05, 99.95))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)

# Method 6: Mild CLAHE + Percentile (0.001-99.999)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
p_low, p_high = np.percentile(img, (0.001, 99.999))
img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-3), 0, 1)
```

## Selection Criteria:

**Choose based on:**
1. **Visual quality**: Clear adipocyte boundaries without artifacts
2. **Range compatibility**: [0,1] range works better with modern activations
3. **Robustness**: Handles intensity variations across samples
4. **Quantitative consistency**: Produces reliable measurements

**Likely best option**: Method 5 (Mild CLAHE + Percentile 0.05-99.95) balances all factors optimally.
