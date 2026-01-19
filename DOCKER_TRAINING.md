# Docker Model Training Guide

This Docker container trains a model on the [Semantic Drone Dataset](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset/data) using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

## Setup

Create a `.env` file in the project root directory and add your Kaggle credentials:

```bash
cat > .env << 'EOF'
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
EOF
```

**Note:** Get your Kaggle API key from [kaggle.com/account](https://www.kaggle.com/account) by clicking "Create New API Token".

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
  -v $(pwd)/nnUNet_raw:/app/nnUNet_raw \
  -v $(pwd)/nnUNet_preprocessed:/app/nnUNet_preprocessed \
  -v $(pwd)/nnUNet_results:/app/nnUNet_results \
  droneseg-training
```

## Training Process

When the container runs, the following steps are executed automatically:

1. **Data Download** (if not present): Dataset is downloaded from Kaggle
2. **Format Conversion**: Data is converted to nnU-Net format (`nnUNet_raw/Dataset101_DroneSeg/`)
3. **Preprocessing**: Data is analyzed and preprocessed
4. **Planning**: Training parameters (batch size, patch size, etc.) are automatically determined
5. **Training**: Model is trained for **1 epoch** (`nnUNetTrainer_1epoch`)
6. **Model Saving**: Best and final checkpoints are saved

**Duration:** ~15-30 minutes (depending on GPU)

## Outputs

Folder structure after training:

```
data/
└── raw/classes_dataset/              # Downloaded Kaggle dataset

nnUNet_raw/
└── Dataset101_DroneSeg/
    ├── dataset.json                  # Dataset metadata
    ├── imagesTr/                     # Training images (400 samples)
    └── labelsTr/                     # Training labels (400 samples)

nnUNet_preprocessed/
└── Dataset101_DroneSeg/
    ├── nnUNetPlans.json              # Preprocessing plan
    ├── dataset.json
    └── nnUNetPlans_2d/               # Preprocessed data
        ├── case_000.b2nd
        └── ...

nnUNet_results/
└── Dataset101_DroneSeg/
    └── nnUNetTrainer_1epoch__nnUNetPlans__2d/
        └── fold_0/
            ├── checkpoint_best.pth        # ✅ Best model (354MB)
            ├── checkpoint_final.pth       # ✅ Final model (354MB)
            ├── progress.png               # Training curve
            └── validation_raw_postprocessed/
```

**Model files:** Checkpoint files in `nnUNet_results/` are used for inference.

## Troubleshooting

**GPU not found:**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Shared memory error:**
- `--ipc=host` is already added to the command

**Permission error:**
```bash
# Make files owned by your user after training
sudo chown -R $USER:$USER nnUNet_results/
```

## Next Step: Inference

After training completes, see [DOCKER_INFERENCE.md](DOCKER_INFERENCE.md) for inference.
