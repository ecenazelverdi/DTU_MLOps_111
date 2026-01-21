#!/bin/bash
set -e

# Create cache directories with write permissions
mkdir -p /tmp/matplotlib /tmp/torch_cache
chmod -R 777 /tmp/matplotlib /tmp/torch_cache 2>/dev/null || true

# Load Kaggle credentials (only KAGGLE_* variables)
export $(grep -v '^#' .env | grep '^KAGGLE_' | xargs)

# Set nnUNet environment variables
export nnUNet_raw="/app/nnUNet_raw"
export nnUNet_preprocessed="/app/nnUNet_preprocessed"
export nnUNet_results="/app/nnUNet_results"

# --- NEW ADDITIONS ---
# 1. Fallback for Mac GPU support
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 2. PYTHONPATH setting:
# Add the current directory (.) so nnU-Net can find my_mac_trainer.py.
export PYTHONPATH=$PYTHONPATH:.
# -----------------------------

# Set cache directories
export HOME="/tmp"
export MPLCONFIGDIR="/tmp/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"

echo "--- Step 1: Data Preparation ---"
python3 -m dtu_mlops_111.data main

echo "--- Step 2: nnU-Net Planning ---"
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity

echo "--- Step 3: Training (5 Epochs on Mac/Universal) ---"
# NOTE: We call our custom '_Mac' trainer here, not the one from the library.
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_5epochs_Mac --npz

echo "--- Process Completed Successfully! ---"