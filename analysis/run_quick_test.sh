#!/bin/bash
# Quick test script for Emo-FiLM HMM analysis
# Run this on the lab server to test with a few subjects

echo "============================================"
echo "Emo-FiLM HMM Analysis - Quick Test"
echo "============================================"

# Configuration
DATA_ROOT="/storage/bigdata/Emo-FiLM/brain_data"
N_TEST_SUBJECTS=5
SESSION_ID="ses-1"
TASK="Rest"
ATLAS="aal"
OUTPUT_DIR="../results/quick_test_$(date +%Y%m%d_%H%M%S)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Warning: conda not found. Using system python."
    PYTHON="python3"
else
    echo "Activating conda environment: emofilm-hmm"
    eval "$(conda shell.bash hook)"
    conda activate emofilm-hmm || {
        echo "Environment not found. Creating..."
        conda create -n emofilm-hmm python=3.9 -y
        conda activate emofilm-hmm
        pip install -r requirements_labserver.txt
    }
    PYTHON="python"
fi

# Check data access
echo ""
echo "Checking data access..."
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Emo-FiLM data not found at $DATA_ROOT"
    echo "Please update DATA_ROOT in this script"
    exit 1
fi

echo "✓ Data directory found: $DATA_ROOT"
echo ""

# Run analysis
echo "Running HMM analysis on $N_TEST_SUBJECTS subjects..."
echo "Output directory: $OUTPUT_DIR"
echo ""

$PYTHON run_hmm_emofilm.py \
    --data_root "$DATA_ROOT" \
    --session_id "$SESSION_ID" \
    --task "$TASK" \
    --atlas "$ATLAS" \
    --n_subjects $N_TEST_SUBJECTS \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================"
    echo "✓ Analysis completed successfully!"
    echo "============================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/emofilm_subject_metrics.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Check output files in $OUTPUT_DIR"
    echo "  2. If test successful, run full analysis without --n_subjects limit"
else
    echo "============================================"
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "============================================"
    echo "Check error messages above"
fi
