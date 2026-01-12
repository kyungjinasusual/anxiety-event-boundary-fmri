#!/bin/bash
#SBATCH --job-name=llava-caption
#SBATCH --output=logs/caption_%j.out
#SBATCH --error=logs/caption_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu           # Adjust to your cluster's GPU partition name

# ============================================
# LLaVA-NeXT Video Captioning SLURM Script
# ============================================

# Configuration - MODIFY THESE
VIDEO_PATH="${1:-/path/to/your/animation.mp4}"
OUTPUT_DIR="${2:-./outputs}"
MODEL="${3:-lmms-lab/LLaVA-Video-7B-Qwen2}"

# Advanced options
SEGMENT_DURATION="${SEGMENT_DURATION:-30}"        # seconds per segment
MAX_FRAMES="${MAX_FRAMES:-32}"                    # frames per segment
TARGET_FPS="${TARGET_FPS:-2.0}"                   # sampling FPS
FULL_VIDEO_FRAMES="${FULL_VIDEO_FRAMES:-64}"      # frames for full video analysis

# ============================================
# Environment Setup
# ============================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

# Load modules (adjust for your cluster)
# module purge
# module load cuda/12.1
# module load anaconda3

# Activate conda environment
# source activate llava  # or: conda activate llava

# Set up environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Get video filename for output naming
VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
OUTPUT_PATH="${OUTPUT_DIR}/${VIDEO_NAME}_captions.json"

echo "============================================"
echo "Configuration:"
echo "  Video: $VIDEO_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Model: $MODEL"
echo "  Segment Duration: ${SEGMENT_DURATION}s"
echo "  Max Frames/Segment: $MAX_FRAMES"
echo "  Target FPS: $TARGET_FPS"
echo "  Full Video Frames: $FULL_VIDEO_FRAMES"
echo "============================================"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================
# Run Captioning
# ============================================

echo "Starting video captioning..."
echo ""

python video_captioner.py \
    "$VIDEO_PATH" \
    --output "$OUTPUT_PATH" \
    --segment-duration "$SEGMENT_DURATION" \
    --max-frames "$MAX_FRAMES" \
    --target-fps "$TARGET_FPS" \
    --full-video-frames "$FULL_VIDEO_FRAMES" \
    --model "$MODEL"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Captioning completed!"
    echo "Results saved to:"
    echo "  JSON: $OUTPUT_PATH"
    echo "  Text: ${OUTPUT_PATH%.json}.txt"
else
    echo "ERROR: Captioning failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
