# Docker Inference Guide

Runs segmentation inference using trained nnU-Net model. Automatically handles preprocessing, segmentation, and visualization.

## Requirements

- Trained model (in `nnUNet_results/` folder)
- Docker + GPU support
- W&B API key (in `.env` file as `WANDB_API_KEY`)

## Usage

### 1. Prepare Test Images

Put test images in `images_raw/` folder:

```bash
# Auto-copied during training (test split)
# Or manually add:
cp your_image.jpg images_raw/
```

### 2. Build Docker Image

```bash
docker build -f inference.dockerfile -t droneseg-inference .
```

### 3. Run Inference

```bash
docker run --gpus all --shm-size=2g \
  --env-file .env \
  -v $(pwd)/images_raw:/images_raw \
  -v $(pwd)/nnUNet_results:/nnUnet_results \
  -v $(pwd)/visualizations:/visualizations \
  droneseg-inference
```

**Note:** `--shm-size=2g` is required! Otherwise you'll get "No space left on device" error.

## Pipeline

Container automatically runs 3 steps:
1. **Preprocessing**: RGB → R/G/B channels (nnU-Net format)
2. **Inference**: Segmentation with custom predictor (W&B + Loguru logging)
3. **Visualization**: Colored overlay images

## Outputs

**Segmentation masks:** `nnUNet_results/inference_outputs/`
- Grayscale PNG files
- Pixel values = class ID (0-5)

**Visualizations:** `visualizations/`
- 3 panels: Original | Mask | Overlay
- First 10 images also uploaded to W&B

**Loguru Logs:**
- **Training:** `nnUNet_results/Dataset101_DroneSeg/nnUNetTrainer_5epochs_custom__nnUNetPlans__2d/fold_*/training_loguru_<timestamp>.log`
- **Inference:** `nnUNet_results/inference_outputs/inference_loguru_<timestamp>.log`
- All metrics also tracked on W&B dashboard

## Class Colors

| ID | Class           | Color  |
|----|-----------------|--------|
| 0  | Background      | Black  |
| 1  | Obstacles       | Purple |
| 2  | Water           | Blue   |
| 3  | Soft-surfaces   | Green  |
| 4  | Moving-objects  | Pink   |
| 5  | Landing-zones   | Gray   |

## Troubleshooting

**"No space left on device" error:**
```bash
# Add --shm-size=2g
docker run --gpus all --shm-size=2g ...
```

**Model not found:**
```bash
# Run training first
docker run --gpus all ... droneseg-trainer
```

**GPU not detected:**
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Features

- ✅ Custom predictor (W&B + Loguru logging)
- ✅ Auto device detection (CUDA > MPS > CPU)
- ✅ Custom trainer support (nnUNetTrainer_5epochs_custom)
- ✅ Train/test split integration (20% test auto-separated)
- ✅ Metric tracking with W&B

