# Adipose Tissue U-Net - AI Coding Agent Instructions

## Project Overview

Dual-model pipeline for **adipose tissue analysis** in fluorescent histology images. Originally trained on adipocyte (single cell) detection, **retrained via transfer learning** to detect whole adipose tissue regions in meat samples.

**Two Model Architecture:**
1. **U-Net Segmentation:** Pixel-level adipose region masks (1024×1024 tiles)
2. **InceptionV3 Classification:** Tile-level binary classifier (adipose vs non-adipose)

**Tech Stack:** TensorFlow 2.13, Keras 2.13, Python 3.10, OpenCV, scikit-image

**Data Transition:** Models trained on SYBR Gold + Eosin pseudocolored data (`/home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane`). Future retraining planned for ECM channel data (`/home/luci/adipose_tissue-unet/data/Meat_MS_Tulane/ECM_channel/`).

**Note on Directory Names:** The directory `pre-post-processing_tools/` is referred to as the "tools folder" in conversation and documentation.

## Data Organization

### Current Training Data: `data/Meat_Luci_Tulane/`
```
Meat_Luci_Tulane/
├── Pseudocolored/              # RGB fluorescent images (SYBR Gold + Eosin)
│   ├── *.tif                   # Full-resolution slides
│   └── test/                   # Isolated test slides
├── Masks/                      # JSON annotations (LabelMe format)
│   ├── fat/                    # Fat cell annotations
│   │   ├── *.json
│   │   └── test/               # Test annotations (isolated)
│   ├── bubbles/                # Bubble artifacts
│   │   └── test/
│   └── muscle/                 # Muscle tissue
│       └── test/
└── _build_YYYYMMDD_HHMMSS/    # Timestamped dataset builds
    ├── dataset/
    │   ├── train/              # 60% from main directories
    │   │   ├── images/         # JPEG tiles
    │   │   └── masks/          # TIFF binary masks
    │   ├── val/                # 20% from main directories
    │   └── test/               # 20% from */test/ subdirs ONLY
    ├── masks/                  # Intermediate binary masks
    │   ├── fat/
    │   ├── bubbles/
    │   └── muscle/
    └── overlays/               # QA visualization (optional)
```

### Future Data: `data/Meat_MS_Tulane/`
```
Meat_MS_Tulane/
├── ECM_channel/                # ECM fluorescence channel (for retraining)
└── SIMs/                       # Structured illumination microscopy
```

**Migration Note:** When retraining on ECM data, update `--data-root` flags in all scripts from `Meat_Luci_Tulane` → `Meat_MS_Tulane/ECM_channel`.

## Critical Architecture Patterns

### 1. Two-Stage Color Normalization Pipeline

**Stage 1 (Dataset Building):** Stain normalization via Reinhard method applied during tiling
- `build_dataset.py` applies SYBR Gold + Eosin color correction (optional `--stain-normalize`)
- Reference: `src/utils/stain_normalization.py` (LAB color space transformation)
- Output: Pre-normalized JPEG tiles in timestamped `_build_YYYYMMDD_HHMMSS/` directories

**Stage 2 (Training):** Intensity normalization using training set statistics
- `train_adipose_unet_2.py` computes mean/std from training tiles
- Default: z-score normalization (`normalization_method='zscore'`)
- Statistics saved to `checkpoints/.../normalization_stats.json` for inference consistency
- **Critical:** Inference must use identical normalization stats (no data leakage)

```python
# Training pattern (from train_adipose_unet_2.py lines 776-790)
train_mean, train_std = compute_mean_std(train_image_paths)
img_normalized = (img - train_mean) / train_std
```

### 2. Timestamped Build Isolation

All builds are **timestamped and isolated** to prevent overwrites:
- Dataset builds: `_build_20241104_152203/` (not `_build/`)
- Training checkpoints: `20241104_152203_adipose_sybreosin_1024_finetune/`
- Scripts auto-detect most recent build via `find_most_recent_build_dir()` (line 68 in train script)

### 3. Reproducibility via Centralized Seeding

**All scripts load seed from `seed.csv`** using `src/utils/seed_utils.py`:
```python
from src.utils.seed_utils import get_project_seed
GLOBAL_SEED = get_project_seed()  # Reads seed.csv or defaults to 865
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
```

**Do NOT hardcode seeds** - always use `get_project_seed()` for consistency.

### 4. Test Set Isolation Architecture

Three-tier data organization prevents leakage:
```
data/Meat_Luci_Tulane/
├── Pseudocolored/          # Training images
├── Masks/                  # Training annotations
│   ├── fat/test/          # Test annotations isolated
│   ├── bubbles/test/
│   └── muscle/test/
└── _build_*/
    ├── dataset/
    │   ├── train/ (60%)   # From main directories only
    │   ├── val/ (20%)     # From main directories only
    │   └── test/ (20%)    # From */test/ subdirectories only
```

**Critical:** Test tiles sourced from `Pseudocolored/test/` and `Masks/*/test/` subdirectories, ensuring zero overlap with training data.

### 5. Multi-Mask Training Target Construction

Fat segmentation uses **subtraction pipeline** to remove bubble artifacts:
```python
# build_dataset.py pattern (lines 384-426)
fat_mask = load_mask("fat")
bubbles_mask = load_mask("bubbles")
target_mask = fat_mask - bubbles_mask  # Remove bubbles from fat
target_mask = morph_close(target_mask, k=3)  # Light cleanup
target_mask = remove_small_components(target_mask, min_px=100)
```

Controlled via: `--target-mask fat --subtract --subtract-class bubbles`

## Key Development Workflows

### Dataset Building (Parallel Optimized)

```bash
python build_dataset.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --stain-normalize \                    # Apply Reinhard normalization
  --target-mask fat --subtract \         # fat - bubbles
  --subtract-class bubbles \
  --min-mask-ratio 0.05 \               # 5% minimum mask coverage
  --workers 8                            # Parallel mask generation
```

**Output Structure:**
- Timestamped `_build_YYYYMMDD_HHMMSS/dataset/{train,val,test}/`
- Normalization stats, quality overlays, build summary

### Training (Two-Phase Fine-Tuning)

```bash
python train_adipose_unet_2.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203 \
  --epochs-phase1 50 --epochs-phase2 100 \
  --batch-size 2                         # For 1024x1024 tiles
```

**Two-Phase Pattern (lines 968-1074):**
1. **Phase 1:** Frozen encoder (50 epochs, lr=1e-4)
2. **Phase 2:** Full fine-tuning (100 epochs, lr=1e-5)
3. Saves `phase1_best.weights.h5`, `phase2_best.weights.h5`, `weights_best_overall.weights.h5`

### Evaluation with Test-Time Augmentation

```bash
python full_evaluation_enhanced.py \
  --weights checkpoints/20251104_152203_*/weights_best_overall.weights.h5 \
  --clean-test --stain \                # Use clean test set with stain norm
  --use-tta --tta-mode full \           # 8x augmentation ensemble
  --sliding-window --overlap 0.5        # Gaussian blending
```

**TTA Modes:** `basic` (4x: rotations), `full` (8x: + flips), `extreme` (16x: + scales)

## Project-Specific Conventions

### File Naming Patterns

- **Checkpoints:** `YYYYMMDD_HHMMSS_adipose_sybreosin{suffix}_1024_finetune/`
- **Tiles:** `{slide_stem}_r{row:04d}_c{col:04d}.jpg` (images), `.tif` (masks)
- **Logs:** `phase1_training.log`, `phase2_training.log`, `training_settings.log`

### Quality Filtering Defaults

```python
# build_dataset.py defaults (lines 81-85)
white_threshold = 240       # Pixel brightness for "empty"
white_ratio_limit = 0.98    # Max 98% white pixels
blurry_threshold = 55.0     # Laplacian variance minimum
min_mask_ratio = 0.05       # 5% minimum target coverage
```

### U-Net Architecture Specifics

- **Input:** 1024×1024×3 (pseudocolored fluorescent images)
- **Output:** 1024×1024×1 (sigmoid, class 1 = fat cells)
- **Loss:** BCE + Dice combined (`combined_loss_standard` at line 232)
- **Encoder:** Frozen in Phase 1, unfrozen in Phase 2
- **Dilation:** Uses dilated convolutions in bottleneck (legacy from `adipocyte_unet.py`)

### Augmentation Levels

`src/utils/data.py` provides three preset functions:
- `augment_pair_light`: Rotations + flips only
- `augment_pair_moderate`: + brightness/contrast (default for training)
- `augment_pair_heavy`: + elastic deformations, blur, noise

Selected via `--augment-level {light,moderate,heavy}` (default: moderate)

## Common Pitfalls & Gotchas

### 1. Normalization Stats Mismatch
**Problem:** Inference fails if using different normalization than training  
**Solution:** Always load `normalization_stats.json` from checkpoint directory:
```python
stats_path = checkpoint_dir / "normalization_stats.json"
with open(stats_path) as f:
    stats = json.load(f)
```

### 2. TensorFlow 2.13 Compatibility
**Legacy imports cause failures:**
```python
# ❌ WRONG (TF1 style)
from keras.layers import Conv2D

# ✅ CORRECT (TF2 style)
from tensorflow.keras.layers import Conv2D
```

### 3. Build Directory Auto-Detection
Scripts auto-find most recent `_build_*` directory. To use specific build:
```bash
--data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane/_build_20251104_152203
```

### 4. Checkpoint Loading
Use `.weights.h5` format (not `.h5` or SavedModel):
```python
model.load_weights("weights_best_overall.weights.h5")  # ✅ Correct
```

## Integration Points

### Stain Normalization Module
- **File:** `src/utils/stain_normalization.py`
- **Reference:** `src/utils/stain_reference_metadata.json` (LAB stats for reference image)
- **Function:** `complete_preprocessing_pipeline(image)` - full Reinhard normalization

### Data Augmentation
- **File:** `src/utils/data.py`
- **Key Functions:** `augment_pair_moderate(img, mask)` (default), `normalize_image(img, method='zscore')`

### Model Definitions
- **Legacy:** `src/models/adipocyte_unet.py` (original implementation, archived)
- **Current:** Inline in `train_adipose_unet_2.py` (lines 400-465)

### Evaluation Metrics
- **File:** `src/utils/model.py`
- **Functions:** `dice_coef`, `jaccard_coef`, `weighted_bce_dice_loss`

## Testing & Validation

**Run full evaluation:**
```bash
python full_evaluation_enhanced.py \
  --weights checkpoints/latest/weights_best_overall.weights.h5 \
  --clean-test --stain --use-tta
```

**Batch checkpoint comparison:**
```bash
python evaluate_all_checkpoints.py --checkpoints-dir checkpoints/
```

**Outputs:** Dice scores, IoU, precision/recall, Hausdorff distance, confusion matrices

## Dual-Model Workflows

### U-Net Segmentation Pipeline
```bash
# 1. Build dataset (60/20/20 train/val/test split)
python build_dataset.py --stain-normalize --target-mask fat --subtract

# 2. Train U-Net (two-phase fine-tuning)
python train_adipose_unet_2.py --epochs-phase1 50 --epochs-phase2 100

# 3. Evaluate with TTA
python full_evaluation_enhanced.py --weights checkpoints/latest/*.weights.h5 \
  --clean-test --stain --use-tta --tta-mode full
```

### InceptionV3 Classification Pipeline
```bash
# 1. Build classification dataset (binary: adipose vs not-adipose)
python Classification/build_class_dataset.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --adipose-threshold 0.10 --balance-classes

# 2. Train classifier (warmup + fine-tuning)
python Classification/train_adipose_classifier_v0.py \
  --dataset-root <build_dir> \
  --warmup-epochs 6 --finetune-epochs 20

# 3. Evaluate classifier
python Classification/eval_adipose_classifier.py \
  --weights checkpoints/classifier_runs/*/best.weights.h5 \
  --test-root <test_dir> --tta-mode full
```

**Key Differences:**
- **Segmentation:** Pixel-wise masks, U-Net architecture, BCE+Dice loss
- **Classification:** Tile-level labels, InceptionV3 transfer learning, binary cross-entropy

## Checkpoint Structure

### U-Net Checkpoints (`checkpoints/YYYYMMDD_HHMMSS_adipose_sybreosin{suffix}_1024_finetune/`)
```
├── normalization_stats.json      # Training mean/std (CRITICAL for inference)
├── phase1_best.weights.h5        # Best Phase 1 (frozen encoder)
├── phase2_best.weights.h5        # Best Phase 2 (full fine-tuning)
├── weights_best_overall.weights.h5  # FINAL MODEL (use this)
├── phase1_training.log           # CSV metrics (loss, dice, lr)
├── phase2_training.log
├── training_settings.log         # Hyperparameters, system info
└── evaluation/                    # Post-training eval results
```

**Checkpoint Naming Convention:**
- `20251104_152203` = Build timestamp (links to dataset)
- `adipose_sybreosin` = Base model name
- `{suffix}` = Optional experiment identifier (`_perc`, `_tta`, `adamw`, etc.)
- `_1024_finetune` = Fixed suffix

### Classifier Checkpoints (`checkpoints/classifier_runs/`)
```
tile_classifier_InceptionV3/
└── tile_adipocyte.weights.h5     # Pretrained on adipocyte detection

classifier_20251111_164323/        # Transfer learning runs
└── best.weights.h5                # Fine-tuned for adipose regions
```

## Available Tools (`tools/`)

### 1. WSI Preprocessing (`large_wsi_to_small_wsi_2.py`)
**Purpose:** Split massive whole-slide images (>13k px) into 6144×6144 tiles for annotation

```bash
python tools/large_wsi_to_small_wsi_2.py \
  --input-dir /path/to/raw_slides \
  --output-dir /path/to/tiles \
  --save-enhanced --enhancement-method clahe  # Dual output for annotation
```

**Features:**
- Adaptive tiling (6144×6144 primary, 3072×3072 edge fallback)
- Bit-depth conversion (`--output-bit-depth 8|16|32f`)
- Intensity inversion (`--invert`) for dark-background stains
- Enhancement modes: `zscore`, `percentile`, `clahe` (annotation-friendly)
- Format preservation (TIFF metadata retained)

**Enhancement Methods:**
- `zscore`: Matches training normalization (±3 std → [0,255])
- `percentile`: Robust 1-99% stretching
- `clahe`: Adaptive histogram equalization (best for annotation)

### 2. Normalization Stats Generator (`generate_checkpoint_normalization_stats.py`)
**Purpose:** Backfill `normalization_stats.json` for old checkpoints

```bash
python tools/generate_checkpoint_normalization_stats.py \
  --data-root /home/luci/adipose_tissue-unet/data/Meat_Luci_Tulane \
  --checkpoints-dir checkpoints/ \
  --overwrite  # Force regenerate
```

**Auto-matches:** Checkpoint timestamp → dataset `_build_{timestamp}/` → compute stats

### 3. ONNX Export (`export_weights_to_onnx.py`)
**Purpose:** Convert `.weights.h5` to ONNX for deployment

```bash
python tools/export_weights_to_onnx.py \
  --weights checkpoints/.../weights_best_overall.weights.h5 \
  --output model.onnx
```

## Utility Modules (`src/utils/`)

### Utility Modules (`src/utils/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `seed_utils.py` | Centralized randomization | `get_project_seed()` - always use this |
| `stain_normalization.py` | Reinhard LAB normalization | `ReinhardStainNormalizer`, `complete_preprocessing_pipeline()` |
| `stain_reference_metadata.json` | Reference image statistics | LAB color space stats for SYBR Gold + Eosin |
| `data.py` | Augmentation pipeline | `augment_pair_{light,moderate,heavy}()`, `normalize_image()` |
| `model.py` | Loss functions & metrics | `dice_coef()`, `jaccard_coef()`, `weighted_bce_dice_loss()` |
| `runtime.py` | TF2 GPU setup | `gpu_selection(memory_growth=True)` |
| `clr_callback.py` | Cyclic learning rate | `CyclicLR` callback (used in legacy code) |
| `multi_gpu.py` | Multi-GPU training | Model parallelization utilities |
| `isbi_utils.py` | ISBI challenge utils | Legacy utilities (not actively used) |

## Project Structure (Reorganized)

```
adipose_tissue-unet/
├── Segmentation/              # U-Net segmentation pipeline
│   ├── build_dataset.py       # Parallel dataset builder with stain normalization
│   ├── build_dataset_v2.py    # Alternative dataset builder
│   ├── build_test_dataset.py  # Test set builder
│   ├── train_adipose_unet_2.py    # Two-phase U-Net training (TF2.13)
│   ├── train_adipose_unet_3.py    # Alternative training script
│   ├── full_evaluation_enhanced.py # Evaluation with TTA, sliding window
│   ├── evaluate_all_checkpoints.py # Batch checkpoint comparison
│   ├── segmentation_inference.py   # Inference script
│   ├── reconstruct_full_images.py  # Tile reassembly
│   ├── tile_classification_evaluation.py
│   ├── visualize_checkpoint_metrics.py
│   └── run_complete_pipeline.sh    # End-to-end automation
├── Classification/            # InceptionV3 tile classifier pipeline
│   ├── build_class_dataset.py      # Binary dataset builder
│   ├── build_test_class_dataset.py # Test set builder
│   ├── train_adipose_classifier_v0.py # Transfer learning
│   └── eval_adipose_classifier.py     # Evaluation with ROC/PR curves
├── tools/                     # Preprocessing and utilities
│   ├── preprocess_small_MS_SIMs.py    # ECM channel preprocessing (FFT, normalization)
│   ├── large_wsi_to_small_wsi_Lucy.py # WSI splitter for Lucy data
│   ├── large_wsi_to_small_wsi_MS.py   # WSI splitter for MS data
│   ├── generate_checkpoint_normalization_stats.py
│   └── export_weights_to_onnx.py
├── src/                       # Shared utilities
│   └── utils/
│       ├── seed_utils.py      # Centralized seeding
│       ├── stain_normalization.py  # Reinhard LAB color normalization
│       ├── stain_reference_metadata.json  # Reference image LAB statistics
│       ├── data.py            # Augmentation pipeline
│       ├── model.py           # Metrics/losses (dice, IoU, BCE+Dice)
│       ├── runtime.py         # GPU setup
│       ├── clr_callback.py    # Cyclic learning rate callback
│       ├── multi_gpu.py       # Multi-GPU training utilities
│       └── isbi_utils.py      # ISBI challenge utilities (legacy)
├── data/                      # Dataset storage
│   ├── Meat_Luci_Tulane/      # Current training data (pseudocolored)
│   └── Meat_MS_Tulane/        # Future ECM channel data
│       └── ECM_channel/
│           ├── *.jpg          # Original JPEGs from WSI splitter
│           ├── enhanced/      # Normalized versions (*_percentile.jpg, *_zscore.jpg)
│           └── corrected/     # Preprocessed output
├── checkpoints/               # Model weights
├── .github/
│   └── copilot-instructions.md # This file
├── seed.csv                   # Global random seed (865)
└── README.md
```

## Key Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `build_dataset.py` | Parallel U-Net dataset builder with stain normalization | `Segmentation/` |
| `build_class_dataset.py` | Classification dataset builder (balanced binary) | `Classification/` |
| `train_adipose_unet_2.py` | Two-phase U-Net training (TF2.13) | `Segmentation/` |
| `train_adipose_classifier_v0.py` | InceptionV3 transfer learning | `Classification/` |
| `full_evaluation_enhanced.py` | U-Net evaluation with TTA, sliding window | `Segmentation/` |
| `eval_adipose_classifier.py` | Classifier evaluation with ROC/PR curves | `Classification/` |
| `run_complete_pipeline.sh` | End-to-end U-Net automation | `Segmentation/` |
| `preprocess_small_MS_SIMs.py` | ECM channel preprocessing (FFT, normalization) | `tools/` |
| `large_wsi_to_small_wsi_Lucy.py` | WSI splitter for Lucy data | `tools/` |
| `large_wsi_to_small_wsi_MS.py` | WSI splitter for MS data | `tools/` |
| `generate_checkpoint_normalization_stats.py` | Backfill normalization stats | `tools/` |
| `export_weights_to_onnx.py` | Convert weights to ONNX format | `tools/` |
