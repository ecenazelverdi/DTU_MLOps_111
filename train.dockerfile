# Up-to-date image with CUDA 12 support for RTX 4060
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System dependencies (required for SimpleITK and OpenCV)
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from official image (fastest method)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# First install only dependencies (speeds up build)
# --system flag installs packages directly to system python
RUN uv pip install --system \
    nnunetv2 \
    SimpleITK \
    pandas \
    typer \
    wandb \
    loguru \
    scikit-learn \
    python-dotenv \
    kaggle \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY src/ /app/src/
RUN cp /app/src/dtu_mlops_111/trainers.py $(python3 -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/custom_trainer.py
COPY train_entrypoint.sh /app/train_entrypoint.sh
COPY .env /app/.env

RUN chmod +x /app/train_entrypoint.sh

# Create nnU-Net directories and cache directories
RUN mkdir -p /app/nnUNet_raw /app/nnUNet_preprocessed /app/nnUNet_results /tmp/cache

# Environment Variables
ENV nnUNet_raw="/app/nnUNet_raw"
ENV nnUNet_preprocessed="/app/nnUNet_preprocessed"
ENV nnUNet_results="/app/nnUNet_results"
# Add src directory to path so nnU-Net can find our Custom Trainer
ENV PYTHONPATH="/app/src"
# Set cache directories to avoid permission issues
ENV HOME="/tmp"
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"

ENTRYPOINT ["/app/train_entrypoint.sh"]