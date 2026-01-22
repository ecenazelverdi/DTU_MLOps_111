#!/bin/bash
set -e

# Create cache directories with write permissions
mkdir -p /tmp/matplotlib /tmp/torch_cache
chmod -R 777 /tmp/matplotlib /tmp/torch_cache 2>/dev/null || true

# Load Kaggle credentials and W&B API key from .env file (but not nnUNet paths)
# Only export KAGGLE_* and WANDB_* variables
set -a  # Automatically export all variables
source <(grep -E '^(KAGGLE_|WANDB_)' .env | sed 's/^/export /')
set +a

# Debug: Check if WANDB_API_KEY is set
echo "DEBUG: WANDB_API_KEY is set: ${WANDB_API_KEY:+yes}"
echo "DEBUG: WANDB_API_KEY length: ${#WANDB_API_KEY}"

# Set nnUNet environment variables to container paths
export nnUNet_raw="/app/nnUNet_raw"
export nnUNet_preprocessed="/app/nnUNet_preprocessed"
export nnUNet_results="/app/nnUNet_results"

# Set cache directories
export HOME="/tmp"
export MPLCONFIGDIR="/tmp/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"

# W&B Login (if WANDB_API_KEY is set, this will use it)
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "--- W&B Login ---"
    wandb login "$WANDB_API_KEY"
fi

echo "--- Step 1: Data Preparation ---"
# Run the 'main' command from data.py (Download + Export)
python3 -m dtu_mlops_111.data main

echo "--- Step 2: nnU-Net Planning ---"
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity

echo "--- Step 3: Training (5 Epochs) ---"
# Call custom trainer class using -tr parameter
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_5epochs_custom --npz

echo "--- Process Completed Successfully! ---"