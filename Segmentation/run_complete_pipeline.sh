#!/bin/bash

# ================================================================
# Complete Adipose U-Net Pipeline Automation Script
# ================================================================
# Phases:
#   1. Dataset Building (build_dataset.py --clean-build) - OPTIONAL
#   2. Model Training (train_adipose_unet_2.py)
#   3. Publication-Quality Evaluation (full_evaluation.py) - with TTA support
#
# Usage: 
#   bash run_complete_pipeline.sh                                    # Full pipeline
#   bash run_complete_pipeline.sh --skip-dataset-build --existing-dataset /path/to/dataset
#   bash run_complete_pipeline.sh --existing-dataset /path/to/dataset --enable-tta
# ================================================================

set -e  # Exit on any error

# Parse command line arguments
SKIP_DATASET_BUILD=false
EXISTING_DATASET=""
ENABLE_TTA=false
TTA_MODE="basic"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dataset-build)
            SKIP_DATASET_BUILD=true
            shift
            ;;
        --existing-dataset)
            EXISTING_DATASET="$2"
            SKIP_DATASET_BUILD=true  # Auto-enable skip when existing dataset provided
            shift 2
            ;;
        --enable-tta)
            ENABLE_TTA=true
            shift
            ;;
        --disable-tta)
            ENABLE_TTA=false
            shift
            ;;
        --tta-mode)
            TTA_MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-dataset-build        Skip Phase 1 (dataset building)"
            echo "  --existing-dataset PATH     Use existing dataset at PATH (auto-skips building)"
            echo "  --enable-tta                Enable Test Time Augmentation in evaluation"
            echo "  --disable-tta               Disable Test Time Augmentation (default)"
            echo "  --tta-mode MODE             TTA mode: minimal, basic, full (default: basic)"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                                    # Full pipeline"
            echo "  $0 --skip-dataset-build --existing-dataset /path    # Skip dataset, use existing"
            echo "  $0 --existing-dataset /path --enable-tta            # Use existing + TTA"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_ROOT="$HOME/Data_for_ML/Meat_Luci_Tulane"
CONDA_ENV="adipose-tf2"
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

check_conda_env() {
    log "Checking conda environment: $CONDA_ENV"
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        error "Conda environment '$CONDA_ENV' not found. Please create it first."
    fi
    success "Conda environment '$CONDA_ENV' found"
}

validate_existing_dataset() {
    local dataset_path="$1"
    log "🔍 Validating existing dataset: $dataset_path"
    
    # Check if dataset directory exists
    if [ ! -d "$dataset_path" ]; then
        error "Existing dataset directory not found: $dataset_path"
    fi
    
    # Check required subdirectories exist
    for subdir in "train/images" "train/masks" "val/images" "val/masks" "test/images" "test/masks"; do
        if [ ! -d "$dataset_path/$subdir" ]; then
            error "Required dataset subdirectory missing: $dataset_path/$subdir"
        fi
    done
    
    # Count tiles and validate minimum requirements
    local train_count=$(find "$dataset_path/train/images" -name "*.jpg" 2>/dev/null | wc -l)
    local val_count=$(find "$dataset_path/val/images" -name "*.jpg" 2>/dev/null | wc -l)
    local test_count=$(find "$dataset_path/test/images" -name "*.jpg" 2>/dev/null | wc -l)
    
    # Validate minimum dataset size requirements
    if [ "$train_count" -lt 10 ]; then
        error "Insufficient training data in existing dataset: $train_count tiles (minimum: 10)"
    fi
    
    if [ "$val_count" -lt 5 ]; then
        error "Insufficient validation data in existing dataset: $val_count tiles (minimum: 5)"
    fi
    
    # Verify image-mask pairs exist
    local train_mask_count=$(find "$dataset_path/train/masks" -name "*.tif" 2>/dev/null | wc -l)
    local val_mask_count=$(find "$dataset_path/val/masks" -name "*.tif" 2>/dev/null | wc -l)
    
    if [ "$train_count" -ne "$train_mask_count" ]; then
        warning "Image-mask count mismatch in existing training data: $train_count images, $train_mask_count masks"
    fi
    
    if [ "$val_count" -ne "$val_mask_count" ]; then
        warning "Image-mask count mismatch in existing validation data: $val_count images, $val_mask_count masks"
    fi
    
    log "📊 Existing Dataset Statistics:"
    log "  • Training tiles: $train_count (masks: $train_mask_count)"
    log "  • Validation tiles: $val_count (masks: $val_mask_count)"
    log "  • Test tiles: $test_count"
    log "  • Total tiles: $((train_count + val_count + test_count))"
    
    success "Existing dataset validation complete - dataset ready for training"
    
    # Set global variables for later use
    TRAIN_COUNT=$train_count
    VAL_COUNT=$val_count
    TEST_COUNT=$test_count
    TRAIN_MASK_COUNT=$train_mask_count
    VAL_MASK_COUNT=$val_mask_count
}

# Print header
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}🧬 COMPLETE ADIPOSE U-NET PIPELINE${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "📅 Started: $(date)"
echo -e "📁 Data root: $DATA_ROOT"
echo -e "🐍 Conda env: $CONDA_ENV"
echo -e "📝 Log file: $LOG_FILE"
echo -e "${BLUE}================================================================${NC}\n"

# Verify prerequisites
log "🔍 Verifying prerequisites..."
check_conda_env

if [ ! -d "$DATA_ROOT" ]; then
    error "Data root directory not found: $DATA_ROOT"
fi
success "Data root directory found: $DATA_ROOT"

# Activate conda environment
log "🐍 Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
success "Conda environment activated"

# ================================================================
# PHASE 1: Dataset Building (OPTIONAL)
# ================================================================
log ""
PHASE1_START=$(date +%s)

if [ "$SKIP_DATASET_BUILD" = true ]; then
    log "🏗️  PHASE 1: Dataset Building - SKIPPED"
    log "================================================================"
    
    if [ -n "$EXISTING_DATASET" ]; then
        log "Using existing dataset: $EXISTING_DATASET"
        DATASET_DIR="$EXISTING_DATASET"
        
        # If existing dataset path is absolute, use it directly
        if [[ "$EXISTING_DATASET" = /* ]]; then
            DATASET_DIR="$EXISTING_DATASET"
        else
            # If relative path, resolve relative to current directory
            DATASET_DIR="$(pwd)/$EXISTING_DATASET"
        fi
        
        # Validate existing dataset
        validate_existing_dataset "$DATASET_DIR"
        
        # For existing datasets, we don't have a specific build directory structure
        # Set LATEST_BUILD to the parent of the dataset for compatibility
        LATEST_BUILD="$(dirname "$DATASET_DIR")"
        BUILD_TIMESTAMP="existing"
        
        PHASE1_END=$(date +%s)
        PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
        success "Phase 1 (Dataset Validation) completed in $((PHASE1_DURATION / 60))m $((PHASE1_DURATION % 60))s"
    else
        # Skip dataset build but look for most recent existing build
        log "🔍 Locating most recent existing build directory..."
        LATEST_BUILD=$(find "$DATA_ROOT" -maxdepth 1 -type d -name "_build_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        
        # Fallback to original _build if no timestamped build found
        if [ -z "$LATEST_BUILD" ] && [ -d "$DATA_ROOT/_build" ]; then
            LATEST_BUILD="$DATA_ROOT/_build"
            warning "No timestamped build found, using fallback: $LATEST_BUILD"
        elif [ -z "$LATEST_BUILD" ]; then
            error "No existing build directory found. Either build a dataset first or provide --existing-dataset path."
        fi
        
        DATASET_DIR="$LATEST_BUILD/dataset"
        BUILD_TIMESTAMP=$(basename "$LATEST_BUILD" | sed 's/_build_//' || echo "unknown")
        
        success "Using existing build directory: $LATEST_BUILD"
        if [ "$BUILD_TIMESTAMP" != "unknown" ] && [ "$BUILD_TIMESTAMP" != "_build" ]; then
            log "📅 Build timestamp: $BUILD_TIMESTAMP"
        fi
        
        # Validate existing dataset
        validate_existing_dataset "$DATASET_DIR"
        
        PHASE1_END=$(date +%s)
        PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
        success "Phase 1 (Existing Dataset Validation) completed in $((PHASE1_DURATION / 60))m $((PHASE1_DURATION % 60))s"
    fi
else
    log "🏗️  PHASE 1: Dataset Building"
    log "================================================================"
    log "Building fresh dataset with enhanced features:"
    log "  • 70/20/10 train/val/test split"
    log "  • SYBR Gold + Eosin stain normalization"
    log "  • Quality overlays for QA"
    log "  • Fat target with bubbles subtraction"
    
    if ! python build_dataset.py 2>&1 | tee -a "$LOG_FILE"; then
        error "Phase 1 (Dataset Building) failed"
    fi
    
    PHASE1_END=$(date +%s)
    PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
    
    success "Phase 1 completed in $((PHASE1_DURATION / 60))m $((PHASE1_DURATION % 60))s"
    
    # Find the most recent build directory (timestamped)
    log "🔍 Locating timestamped build directory..."
    LATEST_BUILD=$(find "$DATA_ROOT" -maxdepth 1 -type d -name "_build_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    # Fallback to original _build if no timestamped build found
    if [ -z "$LATEST_BUILD" ] && [ -d "$DATA_ROOT/_build" ]; then
        LATEST_BUILD="$DATA_ROOT/_build"
        warning "No timestamped build found, using fallback: $LATEST_BUILD"
    elif [ -z "$LATEST_BUILD" ]; then
        error "No build directory found after dataset creation"
    fi
    
    DATASET_DIR="$LATEST_BUILD/dataset"
    BUILD_TIMESTAMP=$(basename "$LATEST_BUILD" | sed 's/_build_//' || echo "unknown")
    
    success "Using build directory: $LATEST_BUILD"
    if [ "$BUILD_TIMESTAMP" != "unknown" ] && [ "$BUILD_TIMESTAMP" != "_build" ]; then
        log "📅 Build timestamp: $BUILD_TIMESTAMP"
    fi
    
    # Verify dataset was created
    if [ ! -d "$DATASET_DIR/train" ] || [ ! -d "$DATASET_DIR/val" ] || [ ! -d "$DATASET_DIR/test" ]; then
        error "Dataset directories not found at: $DATASET_DIR"
    fi
    
    # Verify required subdirectories exist
    for subdir in "train/images" "train/masks" "val/images" "val/masks" "test/images" "test/masks"; do
        if [ ! -d "$DATASET_DIR/$subdir" ]; then
            error "Required directory missing: $DATASET_DIR/$subdir"
        fi
    done
    
    # Count tiles and validate minimum requirements
    TRAIN_COUNT=$(find "$DATASET_DIR/train/images" -name "*.jpg" 2>/dev/null | wc -l)
    VAL_COUNT=$(find "$DATASET_DIR/val/images" -name "*.jpg" 2>/dev/null | wc -l)
    TEST_COUNT=$(find "$DATASET_DIR/test/images" -name "*.jpg" 2>/dev/null | wc -l)
    
    # Validate minimum dataset size requirements
    if [ "$TRAIN_COUNT" -lt 10 ]; then
        error "Insufficient training data: $TRAIN_COUNT tiles (minimum: 10)"
    fi
    
    if [ "$VAL_COUNT" -lt 5 ]; then
        error "Insufficient validation data: $VAL_COUNT tiles (minimum: 5)"
    fi
    
    # Verify image-mask pairs exist
    TRAIN_MASK_COUNT=$(find "$DATASET_DIR/train/masks" -name "*.tif" 2>/dev/null | wc -l)
    VAL_MASK_COUNT=$(find "$DATASET_DIR/val/masks" -name "*.tif" 2>/dev/null | wc -l)
    
    if [ "$TRAIN_COUNT" -ne "$TRAIN_MASK_COUNT" ]; then
        warning "Image-mask count mismatch in training: $TRAIN_COUNT images, $TRAIN_MASK_COUNT masks"
    fi
    
    if [ "$VAL_COUNT" -ne "$VAL_MASK_COUNT" ]; then
        warning "Image-mask count mismatch in validation: $VAL_COUNT images, $VAL_MASK_COUNT masks"
    fi
    
    log "📊 Dataset Statistics:"
    log "  • Training tiles: $TRAIN_COUNT (masks: $TRAIN_MASK_COUNT)"
    log "  • Validation tiles: $VAL_COUNT (masks: $VAL_MASK_COUNT)"
    log "  • Test tiles: $TEST_COUNT"
    log "  • Total tiles: $((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))"
    
    success "Phase 1 validation complete - dataset ready for training"
fi

# ================================================================
# PHASE 2: Model Training
# ================================================================
log ""
log "🧠 PHASE 2: Model Training"
log "================================================================"
log "Training with enhanced features:"
log "  • 2-phase training (50 + 100 epochs)"
log "  • BCE + Dice loss (proven)"
log "  • Timestamped checkpoints (no overwrites)"
log "  • Best model selection by validation Dice"

PHASE2_START=$(date +%s)

# Capture checkpoint directory before training
CHECKPOINTS_BEFORE=$(find checkpoints -maxdepth 1 -type d -name "*adipose_sybreosin_1024_finetune" 2>/dev/null | wc -l)

if ! python train_adipose_unet_2.py --data-root "$LATEST_BUILD" 2>&1 | tee -a "$LOG_FILE"; then
    error "Phase 2 (Model Training) failed"
fi

PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))

success "Phase 2 completed in $((PHASE2_DURATION / 60))m $((PHASE2_DURATION % 60))s"

# Find the newly created checkpoint directory with robust fallback logic
log "🔍 Locating newly created checkpoint directory..."
CHECKPOINT_DIR=""
BEST_WEIGHTS=""

# First try: Find most recent checkpoint with build timestamp if available
if [ "$BUILD_TIMESTAMP" != "unknown" ] && [ "$BUILD_TIMESTAMP" != "_build" ]; then
    CHECKPOINT_DIR=$(find checkpoints -maxdepth 1 -type d -name "${BUILD_TIMESTAMP}_adipose_sybreosin_1024_finetune" 2>/dev/null | head -1)
    
    if [ -n "$CHECKPOINT_DIR" ]; then
        log "Found timestamped checkpoint directory: $CHECKPOINT_DIR"
    fi
fi

# Second try: Find most recent checkpoint directory by modification time
if [ -z "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR=$(find checkpoints -maxdepth 1 -type d -name "*adipose_sybreosin_1024_finetune" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$CHECKPOINT_DIR" ]; then
        log "Found most recent checkpoint directory: $CHECKPOINT_DIR"
    fi
fi

# Third try: Look for any checkpoint directory
if [ -z "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR=$(find checkpoints -maxdepth 1 -type d -name "*adipose*" -o -name "*unet*" | head -1 2>/dev/null)
    
    if [ -n "$CHECKPOINT_DIR" ]; then
        warning "Using fallback checkpoint directory: $CHECKPOINT_DIR"
    fi
fi

if [ -z "$CHECKPOINT_DIR" ]; then
    error "Could not find any checkpoint directory in ./checkpoints/"
fi

# Try to find the best weights with multiple naming patterns
WEIGHT_CANDIDATES=(
    "$CHECKPOINT_DIR/weights_best_overall.weights.h5"
    "$CHECKPOINT_DIR/phase2_best.weights.h5"
    "$CHECKPOINT_DIR/phase1_best.weights.h5"
    "$CHECKPOINT_DIR/weights_best.weights.h5"
    "$CHECKPOINT_DIR/best_model.weights.h5"
    "$CHECKPOINT_DIR/model_best.weights.h5"
)

for weight_file in "${WEIGHT_CANDIDATES[@]}"; do
    if [ -f "$weight_file" ]; then
        BEST_WEIGHTS="$weight_file"
        break
    fi
done

if [ -z "$BEST_WEIGHTS" ]; then
    # List available weights files for debugging
    log "Available weight files in $CHECKPOINT_DIR:"
    find "$CHECKPOINT_DIR" -name "*.h5" -o -name "*.weights*" | head -10 | while read -r f; do
        log "  - $(basename "$f")"
    done
    error "No suitable weights file found in checkpoint directory: $CHECKPOINT_DIR"
fi

# Validate training phase completion
if [ ! -f "$CHECKPOINT_DIR/normalization_stats.json" ]; then
    warning "Training normalization statistics not found - this may cause evaluation issues"
fi

# Check if training logs exist
TRAINING_LOGS=$(find "$CHECKPOINT_DIR" -name "*.log" -o -name "*training*.csv" 2>/dev/null | wc -l)
if [ "$TRAINING_LOGS" -gt 0 ]; then
    log "Found $TRAINING_LOGS training log file(s) in checkpoint directory"
else
    warning "No training logs found - training may not have completed normally"
fi

# Verify checkpoint file size (should be > 50MB for full model)
WEIGHTS_SIZE=$(du -m "$BEST_WEIGHTS" 2>/dev/null | cut -f1)
if [ -n "$WEIGHTS_SIZE" ] && [ "$WEIGHTS_SIZE" -lt 50 ]; then
    warning "Weights file unusually small: ${WEIGHTS_SIZE}MB - training may have failed"
elif [ -n "$WEIGHTS_SIZE" ]; then
    log "Weights file size: ${WEIGHTS_SIZE}MB (appears normal)"
fi

success "Found checkpoint directory: $CHECKPOINT_DIR"
success "Best weights located: $BEST_WEIGHTS"
success "Phase 2 validation complete - model ready for evaluation"

# ================================================================
# PHASE 3: Publication-Quality Evaluation
# ================================================================
log ""
log "📈 PHASE 3: Publication-Quality Evaluation"
log "================================================================"
log "Running comprehensive publication-ready evaluation:"
log "  • Slide-level metrics aggregation"
log "  • Bootstrap confidence intervals (10,000 samples)"
log "  • TP/FP/FN categorical error analysis"
if [ "$ENABLE_TTA" = true ]; then
    log "  • Test Time Augmentation enabled (mode: $TTA_MODE)"
else
    log "  • Test Time Augmentation disabled"
fi
log "  • Publication-ready statistical reporting"
log "  • Comprehensive visualizations"

PHASE3_START=$(date +%s)

# Set up output location for evaluation results
if [ "$BUILD_TIMESTAMP" = "existing" ]; then
    # For existing datasets, create evaluation dir in current location
    EVAL_OUTPUT="evaluation_$(date +%Y%m%d_%H%M%S)"
else
    # Use timestamped build directory structure
    EVAL_OUTPUT="$LATEST_BUILD/evaluation"
fi

# Build TTA arguments if enabled
TTA_ARGS=""
if [ "$ENABLE_TTA" = true ]; then
    TTA_ARGS="--use-tta --tta-mode $TTA_MODE"
fi

# Run evaluation on validation set first (with threshold optimization)
log "🔬 Running evaluation on VALIDATION set (with threshold optimization)..."
if ! python full_evaluation.py \
    --weights "$BEST_WEIGHTS" \
    --data-root "$DATASET_DIR" \
    --output "$EVAL_OUTPUT" \
    --dataset val \
    --optimize-threshold \
    $TTA_ARGS \
    2>&1 | tee -a "$LOG_FILE"; then
    error "Phase 3 (Validation Evaluation) failed"
fi

# Run evaluation on test set (using optimized threshold)
log "🔬 Running evaluation on TEST set (using optimized threshold)..."
if ! python full_evaluation.py \
    --weights "$BEST_WEIGHTS" \
    --data-root "$DATASET_DIR" \
    --output "$EVAL_OUTPUT" \
    --dataset test \
    $TTA_ARGS \
    2>&1 | tee -a "$LOG_FILE"; then
    error "Phase 3 (Test Evaluation) failed"
fi

PHASE3_END=$(date +%s)
PHASE3_DURATION=$((PHASE3_END - PHASE3_START))

success "Phase 3 completed in $((PHASE3_DURATION / 60))m $((PHASE3_DURATION % 60))s"

# Check for evaluation outputs
VAL_RESULTS="$EVAL_OUTPUT/publication_evaluation_val"
TEST_RESULTS="$EVAL_OUTPUT/publication_evaluation_test"

if [ -d "$VAL_RESULTS" ]; then
    log "✓ Validation results available at: $VAL_RESULTS"
    
    # Look for results CSV
    VAL_CSV=$(find "$VAL_RESULTS" -name "*comprehensive_results.csv" | head -1)
    if [ -n "$VAL_CSV" ]; then
        log "✓ Validation metrics: $(basename "$VAL_CSV")"
    fi
fi

if [ -d "$TEST_RESULTS" ]; then
    log "✓ Test results available at: $TEST_RESULTS"
    
    # Look for results CSV
    TEST_CSV=$(find "$TEST_RESULTS" -name "*comprehensive_results.csv" | head -1)
    if [ -n "$TEST_CSV" ]; then
        log "✓ Test metrics: $(basename "$TEST_CSV")"
    fi
fi

# ================================================================
# FINAL SUMMARY
# ================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( $(date +%s) - PHASE1_START ))

log ""
log "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
log "================================================================"
log "⏱️  Total execution time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
log ""
log "📂 Key Output Locations:"
log "  📊 Dataset: $DATASET_DIR"
log "  🧠 Model: $CHECKPOINT_DIR"
log "  📈 Validation: $EVAL_OUTPUT"
log "  📝 Pipeline log: $LOG_FILE"
log ""
log "🎯 Phase Breakdown:"
log "  Phase 1 (Dataset):   $((PHASE1_DURATION / 60))m $((PHASE1_DURATION % 60))s"
log "  Phase 2 (Training):  $((PHASE2_DURATION / 60))m $((PHASE2_DURATION % 60))s" 
log "  Phase 3 (Validation): $((PHASE3_DURATION / 60))m $((PHASE3_DURATION % 60))s"
log ""
log "📋 Dataset Statistics:"
log "  • Training: $TRAIN_COUNT tiles"
log "  • Validation: $VAL_COUNT tiles"
log "  • Test: $TEST_COUNT tiles"
log ""
log "🏆 Best Model:"
log "  • Weights: $BEST_WEIGHTS"
log "  • Checkpoint: $CHECKPOINT_DIR"
log ""
log "✅ All phases completed successfully!"
log "✅ Ready for inference and analysis!"

echo -e "\n${GREEN}🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉${NC}"
echo -e "📝 Check $LOG_FILE for detailed logs"
echo -e "🏆 Best model: $BEST_WEIGHTS"
echo -e "📊 Results: $EVAL_OUTPUT"
