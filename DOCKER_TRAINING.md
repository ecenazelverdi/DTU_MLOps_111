# Docker Model Training Guide

This documentation explains the steps required to train the nnU-Net model inside a Docker container.

## Overview

The Docker setup in this project is designed for:
- Model training in a CUDA-enabled GPU environment
- Consistent and isolated working environment
- Automation of data download, preprocessing, and model training processes

## File Structure

### `train.dockerfile`
Docker image definition. This file:
- Uses NVIDIA CUDA 12.1 base image 
- Installs system dependencies (Python, OpenCV, etc.)
- Installs `uv` package manager
- Installs required Python packages (nnunetv2, PyTorch CUDA 12.1, etc.)
- Copies project source code
- Creates nnU-Net and cache directories
- Sets environment variables

### `train_entrypoint.sh`
Script that runs when the container starts. It performs these steps:

1. **Prepares cache directories**: Temporary directories for PyTorch and Matplotlib
2. **Sets environment variables**: Kaggle credentials and nnU-Net paths
3. **Step 1 - Data Preparation**: Downloads data from Kaggle and converts to nnU-Net format
4. **Step 2 - nnU-Net Planning**: Analyzes dataset and creates preprocessing plan
5. **Step 3 - Training**: Trains for 1 epoch (custom trainer: `nnUNetTrainer_1epoch`)

## Usage

### 1. Prerequisites

Make sure the following files are ready:

**`.env` file** (in project root directory):
```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

> 💡 Kaggle API credentials'larını [Kaggle Account Settings](https://www.kaggle.com/settings/account) sayfasından alabilirsiniz.

### 2. Docker Image Build Etme

```bash
docker build -f train.dockerfile -t drone-train .
```

This command:
- Creates image using `train.dockerfile`
- Tags the image as `drone-train`
- Installs all dependencies (~5-10 minutes)

### 3. Running the Training Container

**Basic command (recommended):**
```bash
docker run --gpus all \
    --ipc=host \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/nnUNet_raw:/app/nnUNet_raw \
    -v $(pwd)/nnUNet_preprocessed:/app/nnUNet_preprocessed \
    -v $(pwd)/nnUNet_results:/app/nnUNet_results \
    --env-file .env \
    drone-train
```

**Parameter explanations:**
- `--gpus all`: Gives all GPUs to the container
- `--ipc=host`: Removes shared memory limit (required for PyTorch multiprocessing)
- `-v $(pwd)/data:/app/data`: Mounts local data folder
- `-v $(pwd)/nnUNet_raw:/app/nnUNet_raw`: Mounts raw data folder
- `-v $(pwd)/nnUNet_preprocessed:/app/nnUNet_preprocessed`: Mounts preprocessed data folder
- `-v $(pwd)/nnUNet_results:/app/nnUNet_results`: Mounts folder where model checkpoints will be saved
- `--env-file .env`: Loads Kaggle credentials

### 4. Training Process

When the container runs, these steps happen automatically:

#### Step 1: Data Preparation (~2-3 minutes)
- Data is downloaded from Kaggle (if not present)
- Data is converted to nnU-Net format
- 400 training samples are exported

#### Step 2: nnU-Net Planning (~5-10 minutes)
- Dataset fingerprint is extracted
- Model architecture is planned
- Preprocessing strategy is determined
- Data is preprocessed and cached

#### Step 3: Training (~10-20 minutes, depends on GPU)
- 1 epoch training with custom trainer (`nnUNetTrainer_1epoch`)
- Model checkpoints are automatically saved
- Best model is written to `nnUNet_results/` folder

## Outputs

After training completes, data will be found in these folders:

```
nnUNet_raw/Dataset101_DroneSeg/
├── dataset.json              # Dataset metadata
├── imagesTr/                 # Training images
└── labelsTr/                 # Training labels

nnUNet_preprocessed/Dataset101_DroneSeg/
├── nnUNetPlans.json          # Preprocessing plans
├── dataset.json              # Dataset info
└── nnUNetPlans_2d/           # Preprocessed data

nnUNet_results/Dataset101_DroneSeg/
└── nnUNetTrainer_1epoch__nnUNetPlans__2d/
    └── fold_0/
        ├── checkpoint_final.pth       # Final model
        ├── checkpoint_best.pth        # Best model
        ├── progress.png               # Training curve
        └── validation_raw_postprocessed/
```

## Sorun Giderme

### GPU Bulunmuyor Hatası
```bash
# NVIDIA Docker runtime'ın yüklü olduğunu kontrol edin
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Shared Memory Hatası (`No space left on device`)
- `--ipc=host` parametresini ekleyin
- Or use `--shm-size=8g`

### Permission Denied Errors
After training, the `nnUNet_results/` folder may become root-owned:
```bash
sudo chown -R $USER:$USER nnUNet_results/
```

### Data Re-downloading
If data already exists but is still being downloaded:
- Check that `.env` file is in the correct location
- Verify that Kaggle credentials are correct

## Advanced Usage

### Training with More Epochs

To use a different trainer, modify this line in `train_entrypoint.sh`:
```bash
# For 100 epochs
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_100epochs --npz

# For 250 epochs (default)
nnUNetv2_train 101 2d 0 --npz
```

### Resume from Checkpoint

If training is interrupted, you can resume by running the same command. nnU-Net will automatically find the last checkpoint and continue.

### Multi-GPU Training

To use multiple GPUs:
```bash
nnUNetv2_train 101 2d 0 --npz -num_gpus 2
```

## Performance Tips

1. **First run is slow**: Preprocessing and cache creation takes time initially
2. **Subsequent runs are fast**: Preprocessed data is read from cache
3. **GPU memory**: batch_size=4 is optimal for RTX 4060 8GB
4. **Disk space**: Total ~5-10GB space required

## Cleanup

To clean up containers and images:
```bash
# Clean stopped containers
docker container prune

# Clean unused images
docker image prune

# Remove all drone-train containers
docker ps -a | grep drone-train | awk '{print $1}' | xargs docker rm

# Remove drone-train image
docker rmi drone-train
```

## Notes

- Training time varies by GPU (RTX 4060: ~15-20 minutes)
- Initial build may take long (PyTorch and nnunetv2 installation)
- Don't commit `.env` file to Git (contains API key)
- While container is running, you can follow logs with `docker logs <container_id>`
