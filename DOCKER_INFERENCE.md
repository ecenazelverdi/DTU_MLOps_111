# Docker Inference Guide

This Docker container performs end-to-end inference using a trained [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) model. It automatically handles preprocessing, segmentation, and visualization.

## Prerequisites

- **Trained model** in `nnUNet_results/` folder (from training container)
- **Docker with GPU support** (NVIDIA Docker runtime)
- **Test images** to segment

## Usage

### 1. Prepare Test Images

Put your test images (JPG, PNG, etc.) in the `images_raw/` folder:

```bash
mkdir -p images_raw
cp your_image1.jpg images_raw/
cp your_image2.png images_raw/
```

### 2. Build Inference Container

```bash
docker build -f inference.dockerfile -t droneseg-inference .
```

### 3. Run Inference (Single Command)

```bash
docker run --gpus all --ipc=host \
  -v $(pwd)/images_raw:/images_raw \
  -v $(pwd)/nnUNet_results:/nnUnet_results \
  -v $(pwd)/visualizations:/visualizations \
  droneseg-inference
```

**That's it!** The container automatically:
1. ✅ Converts RGB images to nnU-Net format (separates R/G/B channels)
2. ✅ Runs segmentation using the trained model
3. ✅ Creates colored visualizations

## Inference Pipeline

When the container runs, these steps execute automatically:

1. **Preprocessing**: RGB images → separate R/G/B channel files (nnU-Net format)
2. **Segmentation**: Model processes images and generates segmentation masks
3. **Visualization**: Creates colored overlays and side-by-side comparisons

**Duration:** ~1-5 seconds per image (depending on GPU)

## Outputs

After inference completes, results are saved in two locations:

```
nnUNet_results/
└── inference_outputs/
    ├── case_0000.png              # Segmentation mask (grayscale)
    ├── case_0001.png
    ├── case_0002.png
    ├── dataset.json               # Metadata
    └── plans.json

visualizations/
├── case_0000_visualization.png    # ✅ Side-by-side view (original | mask | overlay)
├── case_0001_visualization.png
└── case_0002_visualization.png
```

### Understanding the Outputs

**Segmentation Masks** (`nnUNet_results/inference_outputs/`):
- Grayscale PNG images
- Pixel values represent class IDs (0-5)

**Visualizations** (`visualizations/`):
- RGB images showing three panels:
  - **Left**: Original image
  - **Center**: Colored segmentation mask
  - **Right**: Overlay on original image

### Class Colors

| Class ID | Class Name      | Color  |
| -------- | --------------- | ------ |
| 0        | Background      | Black  |
| 1        | Obstacles       | Purple |
| 2        | Water           | Blue   |
| 3        | Soft-surfaces   | Green  |
| 4        | Moving-objects  | Pink   |
| 5        | Landing-zones   | Gray   |

## Troubleshooting

**GPU not found:**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Model not found error:**
- Make sure you've run training first (see [DOCKER_TRAINING.md](DOCKER_TRAINING.md))
- Verify `nnUNet_results/Dataset101_DroneSeg/.../checkpoint_best.pth` exists

**No images in images_raw:**
- Add at least one image to `images_raw/` folder before running inference

**Permission errors:**
```bash
# Make output files owned by your user
sudo chown -R $USER:$USER nnUNet_results/ visualizations/
```

## Complete Workflow (Training → Inference)

If you just finished training:

```bash
# 1. Training completed ✅
# Model saved to: nnUNet_results/Dataset101_DroneSeg/.../checkpoint_best.pth

# 2. Add test images
mkdir -p images_raw
cp test_image.jpg images_raw/

# 3. Build inference container
docker build -f inference.dockerfile -t droneseg-inference .

# 4. Run inference
docker run --gpus all --ipc=host \
  -v $(pwd)/images_raw:/images_raw \
  -v $(pwd)/nnUNet_results:/nnUnet_results \
  -v $(pwd)/visualizations:/visualizations \
  droneseg-inference

# 5. View results
ls visualizations/  # Colored visualizations ready!
```

**Note:** Both training and inference containers share the same `nnUNet_results/` folder for seamless workflow!
