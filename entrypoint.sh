#!/bin/bash
set -e  # Exit on error

echo "========================================"
echo "nnUNet v2 Training Pipeline"
echo "========================================"
echo "Dataset ID: ${DATASET_ID}"
echo "Config: ${CONFIG}"
echo "Fold: ${FOLD}"
echo "nnUNet_raw: ${nnUNet_raw}"
echo "nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "nnUNet_results: ${nnUNet_results}"
echo "========================================"

# Check if dataset exists
if [ ! -d "${nnUNet_raw}/Dataset${DATASET_ID}_"* ]; then
    echo "ERROR: Dataset not found in ${nnUNet_raw}"
    echo "Please mount your preprocessed dataset to ${nnUNet_raw}"
    exit 1
fi

echo ""
echo "--- Step 1: Planning and Preprocessing ---"
uv run nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

echo ""
echo "--- Step 2: Training (Fold: ${FOLD}) ---"
uv run nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} --npz

echo ""
echo "========================================"
echo "Training Complete!"
echo "Results saved to: ${nnUNet_results}"
echo "========================================"