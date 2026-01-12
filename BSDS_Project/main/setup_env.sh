#!/bin/bash
# ============================================
# Environment Setup Script for LLaVA-NeXT Video Captioning
# Clean isolated installation (CUDA 12.4)
# Using pip for PyTorch to avoid MKL conflicts
# ============================================

set -e

ENV_NAME="${1:-llava-video}"
CONDA_BASE=$(conda info --base)

echo "============================================"
echo "Setting up clean conda environment: $ENV_NAME"
echo "============================================"

# ============================================
# Step 1: Complete cleanup
# ============================================
echo "[1/7] Cleaning up existing environments..."

conda deactivate 2>/dev/null || true
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
rm -rf "$CONDA_BASE/envs/$ENV_NAME" 2>/dev/null || true
rm -rf "$HOME/.conda/envs/$ENV_NAME" 2>/dev/null || true

conda clean --all -y
echo "Cleanup complete."

# ============================================
# Step 2: Create fresh environment (minimal)
# ============================================
echo "[2/7] Creating fresh conda environment..."

# Only install Python - avoid conda's MKL
conda create -n "$ENV_NAME" python=3.10 pip -y --no-default-packages

# Activate properly
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ============================================
# Step 3: Block system packages & MKL issues
# ============================================
echo "[3/7] Configuring isolated environment..."

export PYTHONNOUSERSITE=1
export PIP_USER=false
export PATH="$CONDA_PREFIX/bin:$CONDA_BASE/condabin:/usr/bin:/bin"

# Disable MKL to avoid symbol conflicts
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

hash -r

echo "Using pip: $(which pip)"
echo "Using python: $(which python)"

# ============================================
# Step 4: Install PyTorch via pip (not conda!)
# ============================================
echo "[4/7] Installing PyTorch via pip (CUDA 12.4)..."

# Use pip wheel - avoids MKL conflicts from conda
"$CONDA_PREFIX/bin/pip" install --no-cache-dir --no-user \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ============================================
# Step 5: Install basic dependencies
# ============================================
echo "[5/7] Installing dependencies..."

"$CONDA_PREFIX/bin/pip" install --no-cache-dir --no-user \
    numpy \
    pillow \
    tqdm \
    decord \
    av \
    opencv-python \
    accelerate \
    "transformers==4.44.2" \
    sentencepiece \
    protobuf \
    einops

# ============================================
# Step 6: Install LLaVA-NeXT
# ============================================
echo "[6/7] Installing LLaVA-NeXT..."

"$CONDA_PREFIX/bin/pip" install --no-cache-dir --no-user --force-reinstall \
    git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# Flash Attention disabled - causes build issues and not required
# Speed impact: ~1.5-2x slower, but RTX 3090 has enough VRAM
echo "Skipping Flash Attention (disabled for compatibility)"

# ============================================
# Step 7: Verify installation
# ============================================
echo "[7/7] Verifying installation..."

# Set env vars for verification too
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

"$CONDA_PREFIX/bin/python" << 'VERIFY_EOF'
import sys
print(f"Python: {sys.executable}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
print("LLaVA-NeXT imported successfully!")
VERIFY_EOF

# Create activation script with env vars (auto-set on conda activate)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'ENVEOF'
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export PYTHONNOUSERSITE=1
ENVEOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

echo ""
echo "============================================"
echo "SUCCESS! Environment ready."
echo ""
echo "To use:"
echo "  conda activate $ENV_NAME"
echo "  python video_captioner.py your_video.mp4"
echo "============================================"
