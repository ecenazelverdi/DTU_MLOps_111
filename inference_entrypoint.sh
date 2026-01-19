#!/bin/bash
set -e

echo "=== nnU-Net Inference Pipeline ==="

# Set paths
RESULTS_BASE="/nnUnet_results"
MODEL_DIR="$RESULTS_BASE/Dataset101_DroneSeg/nnUNetTrainer_1epoch__nnUNetPlans__2d/fold_0"
OUTPUT_DIR="$RESULTS_BASE/inference_outputs"
IMAGES_RAW="/images_raw"
INPUT_DIR="/input"
VIZ_DIR="/visualizations"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model not found at $MODEL_DIR"
    echo "Please mount your trained model using: -v \$(pwd)/nnUNet_results:/nnUnet_results"
    exit 1
fi

echo "✅ Model found: $MODEL_DIR"

# Check if images_raw directory has files
if [ ! "$(ls -A $IMAGES_RAW 2>/dev/null)" ]; then
    echo "❌ Error: No input images found in $IMAGES_RAW"
    echo "Please add your test images to images_raw/ folder"
    echo "Example: cp your_image.jpg images_raw/"
    exit 1
fi

echo "✅ Input images found: $(ls -1 $IMAGES_RAW | wc -l) files"

# Step 1: Preprocessing - Convert RGB to nnU-Net format
echo ""
echo "📝 Step 1/3: Preprocessing (RGB → R/G/B channels)..."
python3 /app/prepare_inference_input.py $IMAGES_RAW $INPUT_DIR

if [ ! "$(ls -A $INPUT_DIR 2>/dev/null)" ]; then
    echo "❌ Error: Preprocessing failed, no channel files created"
    exit 1
fi

echo "✅ Preprocessing complete: $(ls -1 $INPUT_DIR | wc -l) channel files"

# Step 2: Inference - Run segmentation
echo ""
echo "🚀 Step 2/3: Running inference..."
mkdir -p $OUTPUT_DIR

nnUNetv2_predict \
    -i $INPUT_DIR \
    -o $OUTPUT_DIR \
    -d 101 \
    -c 2d \
    -tr nnUNetTrainer_1epoch \
    -f 0 \
    -chk checkpoint_best.pth \
    --disable_tta

echo "✅ Inference complete: $(ls -1 $OUTPUT_DIR/*.png 2>/dev/null | wc -l) segmentation masks"

# Step 3: Visualization - Create colored overlays
echo ""
echo "🎨 Step 3/3: Creating visualizations..."
python3 /app/visualize_results.py $IMAGES_RAW $OUTPUT_DIR $VIZ_DIR --combined-only

echo "✅ Visualization complete: $(ls -1 $VIZ_DIR/*.png 2>/dev/null | wc -l) visualization files"

# Summary
echo ""
echo "════════════════════════════════════"
echo "✅ Pipeline completed successfully!"
echo "════════════════════════════════════"
echo "📂 Segmentation masks: $OUTPUT_DIR"
echo "📂 Visualizations: $VIZ_DIR"
echo "📊 Processed images: $(ls -1 $IMAGES_RAW | wc -l)"
echo "════════════════════════════════════"
