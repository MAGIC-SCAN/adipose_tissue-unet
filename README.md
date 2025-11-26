# Adipose Tissue Analysis Pipeline

Dual-model deep learning pipeline for automated adipose tissue detection and segmentation from fluorescent histology images.

**Architecture:**
- **Classification**: InceptionV3 transfer learning for tile-level adipose detection
- **Segmentation**: U-Net for pixel-level adipose tissue segmentation

![Original Adipocyte Segmentation Overview](overview.png)

---

## 🎯 Project Goal

Automated analysis of adipose tissue in meat samples using **SYBR Gold + Eosin** fluorescent staining. Analysis is done either using or **ECM channel** or **Pseudocolored** combinations of nuclear channel and ECM channel. The dual-model pipeline first classifies tiles as adipose/non-adipose, then performs detailed pixel-level segmentation for quantitative morphology analysis.

**Key Applications:**
- Adipose tissue distribution mapping in meat samples
- High-throughput histological analysis
- Transfer learning demonstration for tissue-specific deep learning

---

## 📊 Current Status

### ✅ Completed
- **Dual-Model Architecture**: Classification (InceptionV3) + Segmentation (U-Net) pipelines
- **Dataset Pipeline**: Automated preprocessing with stain normalization, quality filtering, test isolation
- **Training Framework**: TF2.13 two-phase fine-tuning with reproducible seeding
- **Evaluation System**: Test-time augmentation (TTA), sliding window inference, comprehensive metrics
- **Quality Control**: Blur detection, white tile filtering, confidence scoring, bubble subtraction
- **Visualization**: WSI reconstruction with color-coded overlays, checkpoint comparison plots
- **Documentation**: Complete usage examples for all 21 scripts across Classification, Segmentation, and preprocessing

### 🔄 In Progress
- Training U-Net segmentation on latest dataset with 75% tile overlap
- Training InceptionV3 classifier on Luci ECM dataset with 50% overlap (pending dataset creation)
- Training InceptionV3 classifier on MS ECM dataset (pending dataset creation)
- Hyperparameter optimization and cross-validation

### 📋 Roadmap
- Automated batch inference pipeline for production deployment
- Integration with downstream morphometry analysis tools
- Multi-channel fusion (ECM + pseudocolored) for enhanced segmentation

---

## 🏗️ Repository Structure

```
adipose_tissue-unet/
├── Classification/               # InceptionV3 tile classifier pipeline
│   ├── build_class_dataset.py    # Binary dataset builder (adipose/not-adipose)
│   ├── build_test_class_dataset.py
│   ├── train_adipose_classifier_v0.py  # Transfer learning trainer
│   ├── eval_adipose_classifier.py      # Evaluation with ROC/PR curves
│   ├── classification_inference.py     # Single/batch inference
│   └── reconstruct_wsi_classification.py  # WSI visualization (TP/FP/FN/TN)
│
├── Segmentation/                 # U-Net segmentation pipeline (root-level scripts)
│   ├── build_dataset.py          # Dataset builder with stain normalization
│   ├── build_test_dataset.py     # Test-specific dataset builder
│   ├── train_adipose_unet_2.py   # Two-phase U-Net training (TF2.13)
│   ├── train_adipose_unet_3.py   # Alternative training with advanced features
│   ├── full_evaluation_enhanced.py    # TTA + sliding window evaluation
│   ├── evaluate_all_checkpoints.py    # Batch checkpoint comparison
│   ├── segmentation_inference.py      # Single/batch tile inference
│   ├── reconstruct_full_images.py     # WSI reassembly with blending
│   └── visualize_checkpoint_metrics.py
│
├── tools/                        # Preprocessing utilities
│   ├── large_wsi_to_small_wsi_Lucy.py  # Grid-based WSI tiling
│   ├── large_wsi_to_small_wsi_MS.py    # Adaptive WSI tiling
│   ├── preprocess_small_MS_SIMs.py     # ECM channel preprocessing (FFT)
│   ├── ECM_scaling.py                  # Dimension matching
│   ├── compare_pseudocolored_ecm_tiles.py
│   ├── export_classification_to_onnx.py
│   ├── export_segmentation_to_onnx.py
│   └── convert_tif_to_jpg.py
│
├── src/
│   └── utils/
│       ├── data.py               # Augmentation pipeline (light/moderate/heavy)
│       ├── model.py              # Metrics (Dice, IoU, BCE+Dice loss)
│       ├── runtime.py            # TF2 GPU configuration
│       ├── seed_utils.py         # Centralized seeding (seed.csv)
│       ├── stain_normalization.py # Reinhard LAB normalization
│       └── stain_reference_metadata.json
│
├── data/                         # Dataset storage
│   ├── Meat_Luci_Tulane/         # Pseudocolored training data
│   │   ├── Pseudocolored/
│   │   ├── Masks/{fat,bubbles,muscle}/
│   │   ├── _build_YYYYMMDD_HHMMSS/    # Timestamped builds
│   │   └── Classification_Build_*/
│   └── Meat_MS_Tulane/           # ECM channel data
│       └── ECM_channel/
│
├── checkpoints/                  # Model weights (not in git)
│   ├── 20251104_152203_adipose_sybreosin_1024_finetune/  # U-Net
│   └── classifier_runs/          # InceptionV3
│
├── seed.csv                      # Global random seed (865)
├── run_complete_pipeline.sh      # End-to-end automation
└── PIPELINE_README.md            # Detailed pipeline documentation
```

---

## 🚀 Quick Start

### Installation

**Requirements:**
- Ubuntu 22.04 (or similar Linux)
- Python 3.10
- CUDA-capable GPU (recommended)
- TensorFlow 2.13.1

```bash
# Create conda environment
conda create -n adipose-tf2 python=3.10
conda activate adipose-tf2

# Install core dependencies
pip install tensorflow==2.13.1 keras==2.13.1
pip install opencv-python scikit-image tifffile
pip install matplotlib seaborn pandas tqdm

# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Complete Workflows

For detailed step-by-step instructions with all parameters and outputs, see the **Classification Pipeline** and **Segmentation Pipeline** sections below.

**Quick Classification Workflow:**
```bash
# 1. Build dataset → 2. Train → 3. Evaluate → 4. Reconstruct WSI
python Classification/build_class_dataset.py --data-root data/Meat_Luci_Tulane --stain-normalize
python Classification/train_adipose_classifier_v0.py --dataset-root <build_dir>
python Classification/eval_adipose_classifier.py --weights <checkpoint> --tta-mode full
python Classification/reconstruct_wsi_classification.py --predictions <csv> --image-dir <wsi_dir>
```

**Quick Segmentation Workflow:**
```bash
# 1. Build dataset → 2. Train → 3. Evaluate → 4. Reconstruct WSI
python build_dataset.py --data-root data/Meat_Luci_Tulane --stain-normalize --target-mask fat --subtract
python train_adipose_unet_2.py --data-root <build_dir> --epochs-phase1 50 --epochs-phase2 100
python full_evaluation_enhanced.py --weights <checkpoint> --use-tta --tta-mode full
python reconstruct_full_images.py --predictions-dir <pred_dir> --original-wsi <wsi_path>
```

---

## 🔄 Complete Workflows

### 🎨 Classification Pipeline (InceptionV3)

**Overview:** Tile-level binary classification (adipose vs non-adipose) using transfer learning from InceptionV3.

**Step 1: Build Classification Dataset**
```bash
# For pseudocolored SYBR Gold + Eosin images
python Classification/build_class_dataset.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --adipose-threshold 0.10 \
  --balance-classes \
  --stain-normalize \
  --quality-filter
```

**Outputs:**
```
Classification_Build_YYYYMMDD_HHMMSS/
├── train/
│   ├── adipose/     # Tiles with ≥10% fat masks
│   └── not_adipose/ # Tiles with <10% fat masks
├── val/
│   ├── adipose/
│   └── not_adipose/
└── build_summary.json
```

**Step 2: Train Binary Classifier**
```bash
# Two-phase training: warmup (frozen encoder) + fine-tuning
python Classification/train_adipose_classifier_v0.py \
  --dataset-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/Classification_Build_20241111_164323 \
  --warmup-epochs 6 \
  --finetune-epochs 20 \
  --batch-size 16
```

**Outputs:**
```
checkpoints/classifier_20241111_164323/
├── best.weights.h5          # Best model weights
├── warmup_training.log      # CSV metrics (warmup phase)
├── finetune_training.log    # CSV metrics (fine-tuning phase)
└── training_settings.json   # Hyperparameters
```

**Step 3: Build Test Dataset**
```bash
# Build isolated test set from test subdirectories
python Classification/build_test_class_dataset.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --adipose-threshold 0.10 \
  --stain-normalize \
  --include-ambiguous --confidence-threshold 0.15
```

**Step 4: Evaluate Classifier**
```bash
# With test-time augmentation (8× ensemble)
python Classification/eval_adipose_classifier.py \
  --weights checkpoints/classifier_20241111_164323/best.weights.h5 \
  --test-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/Classification_Test_Build_20241112_093045 \
  --tta-mode full \
  --save-examples 50
```

**Outputs:**
```
eval_outputs/
├── metrics.json             # Accuracy, precision, recall, F1, AUC
├── confusion_matrix.png     # Confusion matrix heatmap
├── roc_curve.png           # ROC curve with AUC
├── precision_recall_curve.png
├── predictions.csv          # Per-tile predictions with confidence
└── examples/               # Correct/incorrect classification examples
    ├── correct_adipose/
    ├── correct_not_adipose/
    ├── false_positive/
    └── false_negative/
```

**Step 5: Inference on New Images**
```bash
# Single image inference
python Classification/classification_inference.py \
  --weights checkpoints/classifier_20241111_164323/best.weights.h5 \
  --image /path/to/new_tile.jpg \
  --use-tta --tta-mode full

# Batch inference on directory
python Classification/classification_inference.py \
  --weights checkpoints/classifier_20241111_164323/best.weights.h5 \
  --input-dir /path/to/tiles/ \
  --output predictions.csv \
  --use-tta
```

**Step 6: Reconstruct WSI with Classification Results**
```bash
# Visualize predictions on full whole-slide images
python Classification/reconstruct_wsi_classification.py \
  --predictions eval_outputs/predictions.csv \
  --image-dir /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/Pseudocolored/test \
  --output-dir wsi_reconstructions/ \
  --combine-patches 3 \
  --downsample 8
```

**Outputs:**
```
wsi_reconstructions/
└── Meat_11_14_S5_2_classification.png  # Color-coded overlay
    # Green = True Positive (adipose correctly classified)
    # Red = False Positive (non-adipose classified as adipose)
    # Orange = False Negative (adipose missed)
    # Cyan = True Negative (non-adipose correctly classified)
```

---

### 🧩 Segmentation Pipeline (U-Net)

**Overview:** Pixel-level adipose tissue segmentation using U-Net with two-phase fine-tuning.

**Step 1: Build Segmentation Dataset**
```bash
# With stain normalization and bubble subtraction
python build_dataset.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --stain-normalize \
  --target-mask fat --subtract --subtract-class bubbles \
  --min-mask-ratio 0.05 \
  --workers 8
```

**Outputs:**
```
data/Meat_Luci_Tulane/_build_YYYYMMDD_HHMMSS/
├── dataset/
│   ├── train/              # 60% from main directories
│   │   ├── images/         # JPEG tiles (1024×1024)
│   │   └── masks/          # TIFF binary masks
│   ├── val/                # 20% from main directories
│   │   ├── images/
│   │   └── masks/
│   └── test/               # 20% from */test/ subdirs only
│       ├── images/
│       └── masks/
├── masks/                  # Intermediate binary masks
│   ├── fat/
│   ├── bubbles/
│   └── muscle/
└── overlays/               # QA visualization (optional)
```

**Step 2: Train U-Net Segmentation Model**
```bash
# Two-phase fine-tuning: frozen encoder + full training
python train_adipose_unet_2.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
  --epochs-phase1 50 --epochs-phase2 100 \
  --batch-size 2
```

**Outputs:**
```
checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/
├── normalization_stats.json        # Training mean/std (CRITICAL for inference)
├── phase1_best.weights.h5          # Best Phase 1 (frozen encoder)
├── phase2_best.weights.h5          # Best Phase 2 (full fine-tuning)
├── weights_best_overall.weights.h5 # FINAL MODEL (use this)
├── phase1_training.log             # CSV metrics (loss, dice, lr)
├── phase2_training.log
└── training_settings.log           # Hyperparameters, system info
```

**Step 3: Build Test Dataset (if not using build_dataset.py test split)**
```bash
# Build isolated test set from test subdirectories
python build_test_dataset.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --stain-normalize \
  --target-mask fat --subtract --subtract-class bubbles \
  --min-mask-ratio 0.05
```

**Step 4: Evaluate U-Net with Test-Time Augmentation**
```bash
# Full evaluation with TTA, sliding window, and metrics
python full_evaluation_enhanced.py \
  --weights checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5 \
  --clean-test --stain \
  --use-tta --tta-mode full \
  --sliding-window --overlap 0.5
```

**Outputs:**
```
eval_outputs/
├── metrics.json            # Dice, IoU, precision, recall, Hausdorff
├── confusion_matrix.png
├── dice_histogram.png
├── predictions/            # Predicted masks
├── overlays/              # Ground truth vs predictions
└── boundary_analysis/     # Boundary refinement results
```

**Step 5: Batch Evaluate All Checkpoints**
```bash
# Compare multiple checkpoints in parallel
python evaluate_all_checkpoints.py \
  --checkpoints-dir checkpoints/ \
  --test-dir /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203/dataset/test \
  --use-tta --tta-mode full \
  --workers 4
```

**Outputs:**
```
eval_outputs/
├── checkpoint_comparison.csv       # All checkpoints ranked by metrics
├── checkpoint_comparison_plots/
│   ├── dice_comparison.png
│   ├── iou_comparison.png
│   └── precision_recall_comparison.png
└── individual_evals/
    └── {checkpoint_name}/         # Per-checkpoint detailed results
```

**Step 6: Inference on New Tiles**
```bash
# Single tile inference
python segmentation_inference.py \
  --weights checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5 \
  --image /path/to/new_tile.jpg \
  --output predicted_mask.png \
  --use-tta --tta-mode full \
  --save-overlay

# Batch inference on directory
python segmentation_inference.py \
  --weights checkpoints/20251104_152203_adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5 \
  --input-dir /path/to/tiles/ \
  --output-dir predictions/ \
  --use-tta --sliding-window
```

**Step 7: Reconstruct Full WSI from Tiles**
```bash
# Reassemble tiles into full whole-slide images with blending
python reconstruct_full_images.py \
  --predictions-dir predictions/ \
  --original-wsi /path/to/original_wsi.tif \
  --output reconstructed_wsi.png \
  --blending gaussian \
  --use-tta --boundary-refinement
```

**Outputs:**
```
reconstructed_wsi_outputs/
├── Meat_11_14_S5_2_segmentation.png    # Full segmentation overlay
├── Meat_11_14_S5_2_mask.tif           # Binary mask
└── Meat_11_14_S5_2_boundary.png       # Boundary-refined version
```

---

## 🔬 Key Features

### 1. **Stain Normalization**
- SYBR Gold + Eosin color correction using Reinhard method
- Consistent preprocessing across batches
- Optional - can be disabled for raw data training

### 2. **Quality Filtering**
- Blur detection via Laplacian variance
- White tile removal (empty regions)
- Confidence score filtering for annotations

### 3. **Data Integrity**
- Test set completely isolated in separate directory
- No data leakage between train/val/test
- Identical preprocessing for all splits
- Reproducible builds with seed management

### 4. **Bubble Subtraction**
- Removes bubble annotations from fat masks
- Ensures clean training targets
- Morphological cleanup options

### 5. **Comprehensive Evaluation**
- Test-time augmentation (TTA) support
- Multi-metric evaluation (Dice, IoU, Precision, Recall)
- Visualization of predictions
- Checkpoint comparison plots

---

## 📈 Dataset Statistics

Typical build produces:
- **Training**: ~60% of tiles from main directory
- **Validation**: ~20% of tiles from main directory  
- **Test**: ~7 separate slide images from test subdirectory
- **Tile size**: 1024×1024 pixels
- **Format**: JPEG (images), TIFF (masks)

---

## 🔧 Configuration Files

### seed.csv
```csv
seed
865
```
Central random seed for all operations ensuring reproducibility.

### stain_reference_metadata.json
Metadata for stain normalization reference images including color statistics for SYBR Gold + Eosin normalization.

---

## 📚 References

**Original Implementation:**
- Glastonbury et al., "Automatic Adipocyte Detection and Quantification in Histology Images Using Deep Learning"
- GitHub: [GlastonburyC/Adipocyte-U-net](https://github.com/GlastonburyC/Adipocyte-U-net)

**Architecture:**
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)

**Stain Normalization:**
- Reinhard et al., "Color Transfer between Images" (2001)

---

## 📝 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

This is an active research project. For questions or collaboration inquiries, please open an issue.

---

## 📧 Contact

Repository maintained by InstaFlo.

**Binder Demo:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/InstaFlo/adipose_tissue-unet/main)
