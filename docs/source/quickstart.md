# Quick Start Guide

This guide will help you get up and running quickly with the most common workflows.

## Using the Application

This guide focuses on the specific commands used to run the Drone Segmentation project. All commands are executed using `uv run invoke`, which automatically handles environment activation and dependencies.

### 1. Data Management

Commands for handling the dataset:

```bash
# Download the dataset from Kaggle
uv run invoke download-data

# Convert downloaded data to nnU-Net format
uv run invoke export-data

# Run both steps (download + export)
uv run invoke download-and-export-data
```

### 2. Training Pipeline

Commands to preprocess data and train models:

```bash
# Preprocess data (required before training)
uv run invoke preprocess

# Train the model (automatically detects GPU/MPS/CPU)
uv run invoke train
```

### 3. API & Inference

Commands to serve the model and make predictions:

```bash
# Start the FastAPI server locally
uv run invoke app

# The API will be available at http://localhost:8000
```

### 4. Development & Testing

Commands for maintaining the codebase:

```bash
# Run the test suite with coverage
uv run invoke test
```

## Complete Workflow

Here's the complete workflow from installation to running inference:

### 1. Initial Setup

```bash
# Clone and install
git clone https://github.com/ecenazelverdi/DTU_MLOps_111.git
cd DTU_MLOps_111
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your credentials (see Environment Setup guide)
source .env
```

### 2. Download and Prepare Data

```bash
# Download data from Kaggle
uv run invoke download-data

# Export to nnU-Net format
uv run invoke export-data

# Or do both at once
uv run invoke download-and-export-data
```

### 3. Preprocess Data

```bash
# Preprocess with nnU-Net
uv run invoke preprocess
```

This creates the `nnUNet_preprocessed/` directory with training-ready data.

### 4. Train the Model

```bash
# Train with default settings (auto-detect device)
uv run invoke train

# Train on specific device
uv run invoke train --device cuda    # GPU with CUDA
uv run invoke train --device mps     # Apple Silicon GPU
uv run invoke train --device cpu     # CPU only
```

### 5. Run the API

```bash
# Start the FastAPI server
uv run invoke app

# API will be available at http://localhost:8000
```

### 6. Make Predictions

```bash
# Using curl
curl --location 'http://localhost:8000/predict/' \
  --form 'data=@"path/to/your/image.png"'

# Or use the deployed API from our side
curl --location 'https://model-api-32512441443.europe-west1.run.app/predict/' \
  --form 'data=@"path/to/your/image.png"'
```

## First-Time Setup Checklist

Use this checklist to ensure you've completed all setup steps:

- [ ] Install UV package manager
- [ ] Clone the repository
- [ ] Run `uv sync` to install dependencies
- [ ] Create `.env` file from `.env.example`
- [ ] Add Kaggle credentials to `.env`
- [ ] Add Weights & Biases API key to `.env` (optional)
- [ ] Configure nnU-Net paths in `.env`
- [ ] Load environment variables with `source .env`
- [ ] Download data with `uv run invoke download-data`
- [ ] Export data with `uv run invoke export-data`
- [ ] Preprocess data with `uv run invoke preprocess`
- [ ] Verify installation with `uv run invoke test`

## Advanced Workflows

### Docker Workflow

```bash
# Build both API and Training images
uv run invoke docker-build

# Run the API container locally (available at http://localhost:8080)
uv run invoke docker-run-api
```

> The training image built with `docker-build` is fully **CUDA-compatible**, allowing for high-performance training on NVIDIA GPUs.

For detailed instructions on running the training container with GPU support and volume mounts, see the [Detailed Docker Guide](workflows.md#docker-workflow).

## Quick Command Reference

Most common commands you'll use:

| Command                       | Description                     |
| ----------------------------- | ------------------------------- |
| `uv sync`                     | Install/update all dependencies |
| `uv run invoke download-data` | Download dataset from Kaggle    |
| `uv run invoke export-data`   | Convert data to nnU-Net format  |
| `uv run invoke preprocess`    | Preprocess data for training    |
| `uv run invoke train`         | Train the model                 |
| `uv run invoke test`          | Run tests with coverage         |
| `uv run invoke app`           | Start the API server            |
| `uv run invoke build-docs`    | Build documentation             |
| `uv run invoke serve-docs`    | Serve docs locally              |

For a complete command reference, see [Commands Reference](commands.md).

## Next Steps

Now that you're familiar with the basics:

- **[Commands Reference](commands.md)** - Learn about all available commands and their options
- **[Workflows](workflows.md)** - Detailed step-by-step workflows for specific tasks
- **[Troubleshooting](troubleshooting.md)** - Solutions to common issues

## Getting Help

If you run into issues:

1. Check the [Troubleshooting](troubleshooting.md) guide
2. Review error messages carefully
3. Check that your `.env` file is configured correctly
4. Verify all dependencies are installed with `uv sync`
5. Contact the team members listed on the homepage
