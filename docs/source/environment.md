# Environment Setup

After installing the project, you need to configure environment variables for various services and tools. This page guides you through setting up your `.env` file.

## Overview

The project uses a `.env` file to store sensitive credentials and configuration paths. This file is **not tracked in Git** for security reasons.

## Step 1: Create Your .env File

Copy the example environment file:

```bash
cp .env.example .env
```

Now open `.env` in your text editor. You'll need to fill in several values.

## Step 2: Configure Kaggle Credentials

The project downloads data from Kaggle, so you need a Kaggle API key.

### Get Your Kaggle username

1. Go to [kaggle.com](https://www.kaggle.com/) and log in
2. Click on your profile picture → **Your Profile**
3. The username is displayed on top of your name.

### Get Your Kaggle API Key

1. Go to [kaggle.com](https://www.kaggle.com/) and log in
2. Click on your profile picture → **Settings**
3. Scroll down to **API** section
4. Click **Create New Token**
5. Copy the generated token.

### Add to .env

Copy these values into your `.env` file:

```bash
# In your .env file
KAGGLE_USERNAME="your_username_here"
KAGGLE_KEY="your_api_key_here"
```

!!! warning "Keep Credentials Private"
Never commit your `.env` file or share your API keys publicly!

## Step 3: Configure Weights & Biases (Optional)

[Weights & Biases](https://wandb.ai/) is used for experiment tracking.

### Get Your Wandb API Key

1. Go to [wandb.ai](https://wandb.ai/) and sign up/log in
2. Go to [Settings → API Keys](https://wandb.ai/settings)
3. Scroll to **API Keys** section and generate a new API key or copy your existing API key

### Add to .env

```bash
# In your .env file
WANDB_API_KEY="your_wandb_api_key_here"
```

!!! tip "Skip if Not Using W&B"
If you're not using Weights & Biases for experiment tracking, you can leave this empty or comment it out.

## Step 4: Configure nnU-Net Paths

nnU-Net requires three environment variables pointing to data directories. These should be **absolute paths** on your system.

### Understanding the Directories

- `nnUNet_raw` - Raw dataset in nnU-Net format
- `nnUNet_preprocessed` - Preprocessed data ready for training
- `nnUNet_results` - Model checkpoints and training outputs

### Add to .env

There are two ways to define variables in your `.env` file, depending on how you load them.

#### Option 1: Standard Usage (Recommended)

If you use `python-dotenv` (which this project does) or Docker, you generally don't need `export`. This is the standard format:

```bash
# In your .env file
nnUNet_raw="/absolute/path/to/DTU_MLOps_111/nnUNet_raw"
nnUNet_preprocessed="/absolute/path/to/DTU_MLOps_111/nnUNet_preprocessed"
nnUNet_results="/absolute/path/to/DTU_MLOps_111/nnUNet_results"
```

#### Option 2: Shell Loading (With Export)

If you plan to load these variables directly into your shell using `source .env`, you might need the `export` keyword so they become available to child processes:

```bash
# In your .env file
export nnUNet_raw="/absolute/path/to/DTU_MLOps_111/nnUNet_raw"
export nnUNet_preprocessed="/absolute/path/to/DTU_MLOps_111/nnUNet_preprocessed"
export nnUNet_results="/absolute/path/to/DTU_MLOps_111/nnUNet_results"
```

!!! example "Example for macOS/Linux"
`bash
    nnUNet_raw="/Users/yourname/Documents/Github/DTU_MLOps_111/nnUNet_raw"
    nnUNet_preprocessed="/Users/yourname/Documents/Github/DTU_MLOps_111/nnUNet_preprocessed"
    nnUNet_results="/Users/yourname/Documents/Github/DTU_MLOps_111/nnUNet_results"
    `

!!! example "Example for Windows"
`bash
    nnUNet_raw="C:/Users/yourname/Documents/Github/DTU_MLOps_111/nnUNet_raw"
    nnUNet_preprocessed="C:/Users/yourname/Documents/Github/DTU_MLOps_111/nnUNet_preprocessed"
    nnUNet_results="C:/Users/yourname/Documents/Github/DTU_MLOps_111/nnUNet_results"
    `

!!! note "Directories Created Automatically"
These environment variables tell the code **where** to look for or create these directories. The directories themselves will be created by the data processing commands (e.g., `make_dataset` or `export_dataset`), but defining them here ensures the code knows the correct locations. You don't need to manually create the folders before running the commands.

## Step 5: Configure Cloud Storage (Optional)

[Google Cloud Storage (GCS)](https://cloud.google.com/storage) is used for data versioning with DVC and storing data drift reports.

### Set Up Bucket Name

Add your GCS bucket name to your `.env` file:

```bash
# In your .env file
BUCKET_NAME="your-gcs-bucket-name"
```

### Set Up Authentication

If you're using GCS for data versioning with DVC, you need to set up credentials.

1.  **Run the login command**:
    ```bash
    uv run gcloud auth application-default login
    ```
2.  **Follow the browser prompt** to authenticate.
3.  **Done!** The project's automated tasks (e.g., `uv run invoke docker-run-api`) are pre-configured to find these credentials automatically. You don't need to add anything to your `.env` file for authentication.

## Complete .env Example

Here's what your complete `.env` file should look like:

### Option 1: Standard Usage (Recommended for Python/Docker)

```bash
# Kaggle credentials
KAGGLE_USERNAME="john_doe"
KAGGLE_KEY="abc123def456ghi789"

# Weights & Biases API key
WANDB_API_KEY="1234567890abcdef"

# nnU-Net environment variables (use absolute paths!)
nnUNet_raw="/Users/john/Documents/Github/DTU_MLOps_111/nnUNet_raw"
nnUNet_preprocessed="/Users/john/Documents/Github/DTU_MLOps_111/nnUNet_preprocessed"
nnUNet_results="/Users/john/Documents/Github/DTU_MLOps_111/nnUNet_results"

# Google Cloud Storage bucket name
BUCKET_NAME="my-mlops-bucket"

```

!!! note "Automatic Loading"
The project's Python code uses `python-dotenv` to automatically load `.env` when running tasks. You typically don't need to manually source the file when using `uv run invoke` commands.

### Option 2: Shell Loading (With Export)

Use this if you manually source your `.env` file (`source .env`).

```bash
# Kaggle credentials
KAGGLE_USERNAME="john_doe"
KAGGLE_KEY="abc123def456ghi789"

# Weights & Biases API key
WANDB_API_KEY="1234567890abcdef"

# nnU-Net environment variables (use absolute paths!)
export nnUNet_raw="/Users/john/Documents/Github/DTU_MLOps_111/nnUNet_raw"
export nnUNet_preprocessed="/Users/john/Documents/Github/DTU_MLOps_111/nnUNet_preprocessed"
export nnUNet_results="/Users/john/Documents/Github/DTU_MLOps_111/nnUNet_results"

# Google Cloud Storage bucket name
export BUCKET_NAME="my-mlops-bucket"

```

## Verification

### 1. Verify Standard Usage

If you used **Option 1**, the variables are strictly for the Python application. Verify they work by running a project command:

!!! note "This command runs a Python script that loads .env and prints a variable"

```bash
uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f\"nnUNet_raw: {os.environ.get('nnUNet_raw', 'NOT SET')}\")"
```

Expected output:

```bash
nnUNet_raw: /Users/john/Documents/Github/DTU_MLOps_111/nnUNet_raw
```

!!! note "If it says 'NOT SET', your .env file is not loading correcty"

### 2. Verify Shell Loading

If you used **Option 2** (manual `export` or `source .env`), the variables are available in your terminal. You can check them directly:

```bash
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

## Next Steps

With your environment configured, you're ready to:

1. [**Quick Start**](quickstart.md) - Run your first workflow
2. [**Commands Reference**](commands.md) - Learn about available commands
3. [**Workflows**](workflows.md) - Step-by-step guides

## Troubleshooting

### Environment Variables Not Set

If you get errors about missing environment variables:

```bash
# Make sure to source your .env file
source .env

# Or on Windows, reload the variables
```

### Permission Denied on nnU-Net Directories

Make sure the paths exist and you have write permissions:

```bash
mkdir -p nnUNet_raw nnUNet_preprocessed nnUNet_results
chmod 755 nnUNet_*
```

For more help, see [Troubleshooting](troubleshooting.md).
