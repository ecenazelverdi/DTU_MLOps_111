#!/bin/bash
set -e

# Create cache directories with write permissions
mkdir -p /tmp/matplotlib /tmp/torch_cache
chmod -R 777 /tmp/matplotlib /tmp/torch_cache 2>/dev/null || true

# Load Kaggle credentials from .env file (but not nnUNet paths)
# Only export KAGGLE_* variables
export $(grep -v '^#' .env | grep '^KAGGLE_' | xargs)

# Set nnUNet environment variables to container paths
export nnUNet_raw="/app/nnUNet_raw"
export nnUNet_preprocessed="/app/nnUNet_preprocessed"
export nnUNet_results="/app/nnUNet_results"

# Set cache directories
export HOME="/tmp"
export MPLCONFIGDIR="/tmp/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"

echo "--- Step 1: Data Preparation ---"
# Run the 'main' command from data.py (Download + Export)
python3 -m dtu_mlops_111.data main

echo "--- Step 2: nnU-Net Planning ---"
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity

echo "--- Step 3: Training (1 Epoch) ---"
# Call custom trainer class using -tr parameter
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_1epoch --npz

echo "--- Process Completed Successfully! ---"