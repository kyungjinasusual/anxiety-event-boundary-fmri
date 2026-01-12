#!/bin/bash
#SBATCH --job-name=llava-caption
#SBATCH --output=logs/caption_%j.out
#SBATCH --error=logs/caption_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# ============================================
# LLaVA-NeXT Video Captioning - 2 GPU Version
# ============================================

# Configuration
VIDEO_PATH="${1:?Error: VIDEO_PATH required as first argument}"
OUTPUT_DIR="${2:-./outputs}"

# Parameters - adjusted for 2 GPUs (more memory available)
SEGMENT_DURATION="${SEGMENT_DURATION:-30}"
MAX_FRAMES="${MAX_FRAMES:-32}"
TARGET_FPS="${TARGET_FPS:-2.0}"
FULL_VIDEO_FRAMES="${FULL_VIDEO_FRAMES:-48}"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "============================================"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llava-video

# Environment variables
export PYTHONNOUSERSITE=1
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Video filename
VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
OUTPUT_PATH="${OUTPUT_DIR}/${VIDEO_NAME}_captions.json"

echo ""
echo "Configuration:"
echo "  Video: $VIDEO_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Segment Duration: ${SEGMENT_DURATION}s"
echo "  Max Frames/Segment: $MAX_FRAMES"
echo "  Full Video Frames: $FULL_VIDEO_FRAMES"
echo ""

# GPU Info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# ============================================
# Run with 2 GPUs (device_map="auto" will split model)
# ============================================

echo "Starting video captioning with 2 GPUs..."
echo ""

python video_captioner.py \
    "$VIDEO_PATH" \
    --output "$OUTPUT_PATH" \
    --segment-duration "$SEGMENT_DURATION" \
    --max-frames "$MAX_FRAMES" \
    --target-fps "$TARGET_FPS" \
    --full-video-frames "$FULL_VIDEO_FRAMES"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS!"
    echo "Results: $OUTPUT_PATH"
else
    echo "FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
