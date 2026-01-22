# Docker Model Training Guide

This Docker container trains a model on the [Semantic Drone Dataset](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset/data) using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) with **W&B tracking** and **Loguru logging**.

## Setup

Create a `.env` file in the project root directory and add your Kaggle credentials and W&B API key:

```bash
cat > .env << 'EOF'
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
WANDB_API_KEY=your_wandb_api_key
EOF
```

**Note:** 
- Get your Kaggle API key from [kaggle.com/account](https://www.kaggle.com/account) by clicking "Create New API Token"
- Get your W&B API key from [wandb.ai/authorize](https://wandb.ai/authorize)

## Features

✅ **Custom Trainer** (`nnUNetTrainer_5epochs_custom`):
- 5-epoch training (reduced from default 1000)
- W&B logging (loss, epoch metrics)
- Loguru file logging (`training_loguru_*.log`)
- Auto device detection (CUDA > MPS > CPU)

✅ **Data Split**:
- Train: 80% (~320 images)
- Test: 20% (~80 images, saved to `images_raw/` for inference)
- 5-fold cross-validation on train set

✅ **Monitoring**:
- Real-time metrics on [wandb.ai](https://wandb.ai)
- Local logs in `nnUNet_results/.../fold_0/training_loguru_*.log`

## Usage

### 1. Build Docker Image

```bash
docker build -f train.dockerfile -t droneseg-training .
```

### 2. Start Training

```bash
docker run --gpus all --ipc=host \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/images_raw:/app/images_raw \
  -v $(pwd)/nnUNet_raw:/app/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/app/nnUNet_preprocessed \
  -v $(pwd)/nnUNet_results:/app/nnUNet_results \
  droneseg-training
```

**Important:** Added `-v $(pwd)/images_raw:/app/images_raw` to save test images for inference.

## Training Process

When the container runs, the following steps are executed automatically:

1. **W&B Login**: Authenticates with Weights & Biases using `WANDB_API_KEY`
2. **Data Download** (if not present): Dataset is downloaded from Kaggle
3. **Train/Test Split**: 80% train (~320 images), 20% test (~80 images, no data leakage)
4. **Format Conversion**: Data is converted to nnU-Net format (`nnUNet_raw/Dataset101_DroneSeg/`)
   - Training set: `imagesTr/`, `labelsTr/`
   - Test set: `imagesTs/` (no labels), originals copied to `images_raw/`
5. **Preprocessing**: Data is analyzed and preprocessed
6. **Planning**: Training parameters (batch size, patch size, etc.) are automatically determined
7. **Training**: Model is trained for **5 epochs** using **fold 0** (~256 train, ~64 validation)
   - Custom trainer: `nnUNetTrainer_5epochs_custom`
   - Auto device detection (CUDA/MPS/CPU)
   - Metrics logged to W&B and local files
8. **Model Saving**: Best and final checkpoints are saved

**Duration:** ~10-15 minutes (depending on GPU)

## Outputs

Folder structure after training:

```
data/
├── raw/classes_dataset/              # Downloaded Kaggle dataset (400 images)
└── test_images/                      # Test set backup (80 images)

images_raw/                           # Test images ready for inference (80 images)

nnUNet_raw/
└── Dataset101_DroneSeg/
    ├── dataset.json                  # Dataset metadata
    ├── imagesTr/                     # Training images (320 samples, 80% split)
    ├── labelsTr/                     # Training labels (320 samples)
    └── imagesTs/                     # Test images (80 samples, 20% split, no labels)

nnUNet_preprocessed/
└── Dataset101_DroneSeg/
    ├── nnUNetPlans.json              # Preprocessing plan
    ├── dataset.json
    ├── splits_final.json             # 5-fold cross-validation splits
    └── nnUNetPlans_2d/               # Preprocessed data
        ├── case_000.b2nd
        └── ...

nnUNet_results/
└── Dataset101_DroneSeg/
    └── nnUNetTrainer_5epochs_custom__nnUNetPlans__2d/
        └── fold_0/
            ├── checkpoint_best.pth        # ✅ Best model (354MB)
            ├── checkpoint_final.pth       # ✅ Final model (354MB)
            ├── progress.png               # Training curve
            ├── training_log_*.txt         # nnU-Net logs
            ├── training_loguru_*.log      # ✅ Custom loguru logs
            └── validation/                # Validation predictions
```

**Model files:** Checkpoint files in `nnUNet_results/` are used for inference.

**W&B Dashboard:** View real-time metrics at [wandb.ai](https://wandb.ai) under project `semantic_segmentation_nnunet`

## Troubleshooting

**GPU not found:**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**W&B not logging:**
- Check `.env` file contains `WANDB_API_KEY=your_key`
- Verify key at [wandb.ai/authorize](https://wandb.ai/authorize)

**Data leakage concerns:**
- Test set (20%) is completely separated before training
- Saved to `images_raw/` for inference
- Never used in training or validation

**Shared memory error:**
- `--ipc=host` is already added to the command

**Permission error:**
```bash
# Make files owned by your user after training
sudo chown -R $USER:$USER nnUNet_results/ nnUNet_preprocessed/
```

**Retraining after data changes:**
```bash
# Delete old preprocessed data and results
rm -rf nnUNet_preprocessed/Dataset101_DroneSeg
rm -rf nnUNet_results/Dataset101_DroneSeg
# Run training again
```

## Advanced: Training All 5 Folds

For ensemble predictions (5x slower but better accuracy):

Edit `train_entrypoint.sh` line 43:
```bash
# Replace:
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_5epochs_custom --npz

# With loop:
for fold in 0 1 2 3 4; do
    nnUNetv2_train 101 2d $fold -tr nnUNetTrainer_5epochs_custom --npz
done
```

This trains 5 models (one per fold). Total time: ~50-75 minutes.

## Platform Compatibility

✅ **CUDA GPUs** (NVIDIA): Auto-detected, uses CUDA acceleration
✅ **Apple Silicon** (M1/M2/M3): Auto-detected, uses MPS backend
✅ **CPU only**: Auto-detected, falls back to CPU (slower)

Device detection is automatic in `trainers.py`.

## Next Step: Inference

After training completes, see [DOCKER_INFERENCE.md](DOCKER_INFERENCE.md) for inference.
