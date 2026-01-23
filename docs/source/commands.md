# Commands Reference

Complete reference for all `tasks.py` invoke commands available in the project.

## Overview

All project commands are defined in `tasks.py` and executed using the `invoke` library. You run them with:

```bash
uv run invoke <command-name> [options]
```

## List All Commands

To see all available commands:

```bash
uv run invoke --list
```

## Data Commands

### download-data

Downloads the drone imagery dataset from Kaggle.

```bash
uv run invoke download-data
```

**Prerequisites:**

- Kaggle credentials configured in `.env`
- `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables set

**Output:**

- Downloads dataset to `data/raw/classes_dataset/`
- Creates directories automatically if they don't exist

**What it does:**

- Uses Kaggle API to download the semantic segmentation drone dataset
- Extracts the archive
- Organizes images into `original_images/` and `label_images_semantic/`

---

### export-data

Converts the downloaded Kaggle dataset into nnU-Net format.

```bash
uv run invoke export-data
```

**Prerequisites:**

- Dataset already downloaded with `download-data`
- nnU-Net environment variables set in `.env`

**Output:**

- Creates `nnUNet_raw/Dataset101_DroneSeg/`
- Populates `imagesTr/` and `labelsTr/` directories
- Generates `dataset.json` metadata file

**What it does:**

- Reads images from `data/raw/classes_dataset/`
- Converts to nnU-Net naming convention (`<id>_0000.png`)
- Creates train/test splits
- Generates nnU-Net dataset configuration

---

### download-and-export-data

Combines `download-data` and `export-data` into a single command.

```bash
uv run invoke download-and-export-data
```

**Equivalent to:**

```bash
uv run invoke download-data
uv run invoke export-data
```

---

### download-models

Downloads pre-trained models and checkpoints from DVC (Data Version Control).

```bash
uv run invoke download-models
```

**Prerequisites:**

- DVC configured (comes with `uv sync`)
- Access to the GCS bucket (for team members)
- Google Cloud credentials set in `.env` (if required)

**Output:**

- Downloads tracked files: `nnUNet_results.dvc`, `nnUNet_preprocessed.dvc`, `data.dvc`
- Populates directories with versioned data and models

**What it does:**

- Runs `dvc pull` to fetch all DVC-tracked resources
- Downloads model checkpoints from cloud storage
- Syncs preprocessed data if needed

---

## Preprocessing Commands

### preprocess

Runs nnU-Net preprocessing pipeline.

```bash
uv run invoke preprocess [--dataset-id DATASET_ID]
```

**Options:**

- `--dataset-id` (default: `101`) - Dataset ID to preprocess

**Examples:**

```bash
# Preprocess default dataset (101)
uv run invoke preprocess

# Preprocess specific dataset
uv run invoke preprocess --dataset-id 102
```

**Prerequisites:**

- Data exported to `nnUNet_raw/` with `export-data`
- nnU-Net environment variables set

**Output:**

- Creates `nnUNet_preprocessed/Dataset101_DroneSeg/`
- Generates preprocessing plans
- Creates `gt_segmentations/` and `nnUNetPlans_2d/` directories

**What it does:**

- Analyzes dataset statistics
- Generates preprocessing plans
- Preprocesses images for training
- Verifies dataset integrity

---

## Training Commands

### train

Trains the nnU-Net model.

```bash
uv run invoke train [OPTIONS]
```

**Options:**

- `--dataset-id` (default: `101`) - Dataset ID
- `--fold` (default: `0`) - Cross-validation fold (0-4)
- `--dim` (default: `2d`) - Model dimension (`2d` or `3d_fullres`)
- `--device` (default: `auto`) - Device to use (`auto`, `cuda`, `mps`, `cpu`)

**Examples:**

```bash
# Train with defaults (2D, fold 0, auto-detect device)
uv run invoke train

# Train on GPU with CUDA
uv run invoke train --device cuda

# Train on Apple Silicon
uv run invoke train --device mps

# Train on CPU
uv run invoke train --device cpu

# Train specific fold
uv run invoke train --fold 1

# Train 3D model
uv run invoke train --dim 3d_fullres

# Combine options
uv run invoke train --dataset-id 101 --fold 2 --dim 2d --device cuda
```

**Prerequisites:**

- Data preprocessed with `preprocess`
- Sufficient disk space for checkpoints (~500MB)
- GPU recommended (training is slow on CPU)

**Output:**

- Model checkpoints in `nnUNet_results/Dataset101_DroneSeg/nnUNetTrainer__nnUNetPlans__2d/fold_X/`
- Training logs and metrics
- Can be monitored with Weights & Biases if configured

**What it does:**

- Initializes nnU-Net trainer
- Trains segmentation model
- Saves checkpoints periodically
- Logs metrics to console and W&B

**Device Selection:**

- `auto` - Automatically selects CUDA > MPS > CPU
- `cuda` - Use NVIDIA GPU (requires CUDA toolkit)
- `mps` - Use Apple Metal Performance Shaders (M1/M2 Macs)
- `cpu` - Use CPU only (slow, for testing)

---

## Testing Commands

### test

Runs the test suite with coverage reporting.

```bash
uv run invoke test
```

**What it does:**

- Runs all tests in `tests/` directory with pytest
- Generates coverage report
- Displays coverage statistics

**Output:**

```
tests/integrationtests/test_apis.py ...
tests/test_data.py ...

---------- coverage: platform darwin, python 3.12.x -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src/dtu_mlops_111/__init__.py        0      0   100%
src/dtu_mlops_111/api.py           120     10    92%
src/dtu_mlops_111/data.py           85      5    94%
-----------------------------------------------------
TOTAL                               205     15    93%
```

**Manual Testing:**

```bash
# Run specific test file (integration tests for API)
uv run pytest tests/integrationtests/test_apis.py

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_data.py::test_download_data

# Run without coverage
uv run pytest tests/
```

---

## API Commands

### app

Starts the FastAPI development server.

```bash
uv run invoke app
```

**What it does:**

- Starts Uvicorn server on `http://localhost:8000`
- Enables auto-reload on code changes
- Serves API endpoints for prediction

**Output:**

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Available Endpoints:**

- `GET /` - Health check & model status
- `POST /predict/` - Image segmentation (single)
- `POST /batch_predict/` - Image segmentation (batch)
- `GET /drift/` - Data drift report
- `GET /model_info/` - Model metadata
- `GET /metrics` - Prometheus metrics

**Testing the API:**

```bash
# In another terminal
curl http://localhost:8000/

# Make a prediction
curl --location 'http://localhost:8000/predict/' \
  --form 'data=@"path/to/image.png"'
```

## Start manually with Uvicorn

```bash
uv run uvicorn main:app --port 8000 --reload

```

---

## BentoML Serving (Alternative)

### bento-build

Builds the BentoML bundle containing model and code.

```bash
uv run invoke bento-build
```

### bento-serve

Serves the BentoML service locally with auto-reload.

```bash
uv run invoke bento-serve [--port 3000]
```

**Output:**

```
INFO [cli] Starting 1 BentoServer instances..
INFO [cli] Service "drone-seg-service" is versioned: ...
INFO [cli] Serving at http://localhost:3000
```

---

## Docker Commands

### docker-build

Builds all project Docker images (train, inference, api).

```bash
uv run invoke docker-build [--progress PROGRESS]
```

**What it does:**

- Builds `train:latest`
- Builds `inference:latest`
- Builds `api:latest`
- Builds `bento:latest`

### Individual Build Tasks

- `uv run invoke docker-build-train` - Build training image
- `uv run invoke docker-build-inference` - Build inference image
- `uv run invoke docker-build-api` - Build API image
- `uv run invoke docker-build-bento` - Build BentoML image

### docker-train

Runs the training pipeline inside a Docker container with all necessary volume mounts and environment variables.

```bash
uv run invoke docker-train
```

**What it does:**

- Runs `train:latest` with `--gpus all` and `--ipc=host`
- Mounts local `data/`, `nnUNet_raw/`, `nnUNet_preprocessed/`, and `nnUNet_results/`
- Loads environment variables from `.env`

### docker-inference

Runs the nnU-Net prediction pipeline inside a Docker container.

```bash
uv run invoke docker-inference
```

**What it does:**

- Runs `inference:latest` with volume mounts for `nnUNet_results/`, `images_raw/`, and `visualizations/`

### docker-run-api

Runs the FastAPI inference server in a container.

```bash
uv run invoke docker-run-api
```

**What it does:**

- Runs `api:latest` on port 8080
- Automatically mounts Google Cloud ADC credentials if found locally

### docker-run-bento

Runs the BentoML container.

```bash
uv run invoke docker-run-bento
```

**What it does:**

- Runs `bento:latest` on port 8080
- Automatically mounts Google Cloud ADC credentials if found locally
- Enables DVC pull if env var is set

---

## Cloud Run Commands

### cloud-run-update-env

Enables DVC model pull in Cloud Run.

```bash
uv run invoke cloud-run-update-env [--service SERVICE] [--region REGION]
```

### cloud-run-update-resources

Updates Cloud Run resource settings (CPU, Memory, Timeout).

```bash
uv run invoke cloud-run-update-resources [--cpu 4] [--memory 8Gi]
```

### cloud-run-logs

Reads logs from a Cloud Run service.

```bash
uv run invoke cloud-run-logs [--limit 200]
```

---

## BentoML Utility Commands

### bento-smoke-test

Runs a quick health check sequence (`livez`, `readyz`, `model_info`) against a running service.

```bash
uv run invoke bento-smoke-test --service-url http://localhost:3000
```

### bento-predict-base64

Helper to send a base64 encoded image prediction request (avoids shell argument limits).

```bash
uv run invoke bento-predict-base64 --image-path path/to/image.png --service-url http://localhost:3000
```

---

## Command Chaining

You can chain multiple commands if needed:

```bash
# Download, export, and preprocess in one go
uv run invoke download-data && \
uv run invoke export-data && \
uv run invoke preprocess
```

## Next Steps

- [Workflows](workflows.md) - Learn how to combine these commands into complete workflows
- [Troubleshooting](troubleshooting.md) - Solutions if commands fail
