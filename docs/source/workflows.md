# Workflows

Detailed step-by-step workflows for common tasks in the project.

## Data Workflow

Complete workflow from downloading data to having it ready for training.

### Step 1: Download Dataset

```bash
uv run invoke download-data
```

**What happens:**

- Connects to Kaggle API
- Downloads semantic segmentation drone dataset
- Extracts to `data/raw/classes_dataset/`

**Expected output:**

```
data/raw/classes_dataset/classes_dataset/
├── original_images/        (400 RGB drone images)
└── label_images_semantic/  (400 segmentation masks)
```

### Step 2: Export to nnU-Net Format

```bash
uv run invoke export-data
```

**What happens:**

- Reads images from `data/raw/`
- Converts to nnU-Net naming convention
- Creates dataset metadata

**Expected output:**

```
nnUNet_raw/Dataset101_DroneSeg/
├── imagesTr/       (training images)
├── labelsTr/       (training labels)
└── dataset.json    (metadata)
```

### Step 3: Preprocess Data

```bash
uv run invoke preprocess
```

**What happens:**

- Analyzes dataset statistics
- Generates preprocessing plans
- Creates training-ready data

**Expected output:**

```
nnUNet_preprocessed/Dataset101_DroneSeg/
├── gt_segmentations/
├── nnUNetPlans_2d/
└── nnUNetPlans.json
```

**Time:** 10-30 minutes depending on your machine

---

## Training Workflow

End-to-end model training workflow.

### Prerequisites

- [x] Data downloaded and preprocessed (see Data Workflow)
- [x] Environment variables configured
- [x] GPU available (recommended)

### Step 1: Verify Data

```bash
# Check that preprocessed data exists
ls nnUNet_preprocessed/Dataset101_DroneSeg/

# Should see:
# gt_segmentations/  nnUNetPlans_2d/  nnUNetPlans.json
```

### Step 2: Start Training

```bash
# Train with auto device detection
uv run invoke train

# Or specify device explicitly
uv run invoke train --device cuda  # For NVIDIA GPU
uv run invoke train --device mps   # For Apple Silicon
```

### Step 3: Monitor Training

Training progress is logged to console:

```
Epoch 1/250
Current loss: 0.4532
Validation Dice: 0.7821
```

If you configured Weights & Biases:

- Visit [wandb.ai](https://wandb.ai/) to see live metrics
- Charts for loss, Dice score, learning rate
- System metrics (GPU usage, memory)

### Step 4: Verify Checkpoints

```bash
# Check training outputs
ls nnUNet_results/Dataset101_DroneSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/

# Should contain:
# checkpoint_final.pth
# checkpoint_best.pth
# training.log
# progress.png
```

**Training time:**

- GPU: ~2-6 hours for full training
- CPU: ~24-48 hours (not recommended)

---

## API Workflow

Running and testing the inference API.

### Step 1: Ensure Model Available

You need a trained model checkpoint. Either:

**Option A: Use pre-trained model (recommended for testing)**

```bash
# Download from DVC
uv run invoke download-models
```

**Option B: Train your own model**

```bash
# Follow the Training Workflow above
uv run invoke train
```

### Step 2: Start API Server

```bash
uv run invoke app
```

**Output:**

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Step 3: Test API Health

In another terminal:

```bash
# Check health endpoint (the root / behaves as health check)
curl http://localhost:8000/

# Response: {"status": "ok", "model_status": "model_loaded"}
```

### Step 4: Make a Prediction

```bash
# Predict on a sample image
curl --location 'http://localhost:8000/predict/' \
  --form 'data=@"path/to/drone_image.png"'
```

**Response (example):**

```json
{
  "prediction_map_url": "gs://bucket/predictions/image_segmentation.png",
  "class_percentages": {
    "obstacles": 32.4,
    "water": 8.1,
    "soft_surfaces": 25.3,
    "moving_objects": 2.8,
    "landing_zones": 31.4
  }
}
```

### Step 5: View Interactive Documentation

Open in your browser:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

You can test all endpoints interactively from Swagger UI.

---

## Docker Workflow

Using Docker for training and inference.

### Training with Docker

#### Step 1: Build Training Container

```bash
uv run invoke docker-build-train
```

#### Step 2: Run Training

```bash
uv run invoke docker-train
```

**What it does:**

- Runs the container with GPU support (`--gpus all`)
- Mounts local directories (`data/`, `nnUNet_raw/`, etc.) for persistence
- Executes the full training pipeline defined in `train_entrypoint.sh`

### API with Docker

#### Step 1: Build API Container

```bash
uv run invoke docker-build-api
```

#### Step 2: Run API

```bash
uv run invoke docker-run-api
```

API available at: `http://localhost:8080`

---

### Inference with Docker

#### Step 1: Build Inference Container

```bash
uv run invoke docker-build-inference
```

#### Step 2: Run Inference

```bash
uv run invoke docker-inference
```

**What it does:**

- Runs `inference:latest` to generate masks from images in `images_raw/`
- Saves results to `nnUNet_results/` and `visualizations/`

---

## BentoML Deployment Workflow

Alternative high-performance serving with BentoML. See [BentoML Commands](commands.md#bentoml-serving-alternative).

### Step 1: Build the Bento

```bash
uv run invoke bento-build
```

### Step 2: Serve Locally

```bash
uv run invoke bento-serve
```

### Step 3: Run with Docker

```bash
uv run invoke docker-build-bento
uv run invoke docker-run-bento
```

**Note:** The `docker-run-bento` command will automatically mount your local Google Cloud credentials if they are available, enabling `dvc pull` to work inside the container.

**Quick start:**

```bash
# 1. Prepare input images
python prepare_inference_input.py images_raw/ input/

# 2. Build inference container
uv run invoke docker_build_inference

# 3. Run inference
uv run invoke docker_inference

# 4. Visualize results
python visualize_results.py images_raw/ nnUNet_results/inference_outputs/ visualizations/
```

---

## Testing & Verification

Comprehensive testing to ensure project integrity.

### 1. Unit Tests & Coverage

```bash
# Run tests with coverage report
uv run invoke test
```

### 2. Pipeline Verification (Integration)

Verify the full data pipeline from download to preprocessing:

```bash
uv run invoke download-data && uv run invoke export-data && uv run invoke preprocess
```

### 3. API Verification

```bash
# Run API integration tests
uv run pytest tests/integrationtests/test_apis.py -v
```

---

## Next Steps

- [Commands Reference](commands.md) - Detailed command documentation
- [Troubleshooting](troubleshooting.md) - Solutions to common issues
