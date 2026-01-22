# Inference container for nnU-Net model (CLI-based)
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Increase timeout for large packages like torch (744MB)
ENV UV_HTTP_TIMEOUT=600

# Install dependencies
RUN uv pip install --system \
    nnunetv2 \
    SimpleITK \
    Pillow \
    numpy \
    wandb \
    loguru \
    python-dotenv \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Create necessary directories
RUN mkdir -p /input /nnUnet_results /images_raw /visualizations

# Copy preprocessing and visualization scripts
COPY prepare_inference_input.py /app/prepare_inference_input.py
COPY visualize_results.py /app/visualize_results.py

# Copy project source (for custom predictor)
COPY src/ /app/src/
RUN cp /app/src/dtu_mlops_111/trainers.py $(python3 -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/custom_trainer.py
COPY .env /app/.env

# Add src to Python path
ENV PYTHONPATH="/app/src"

# Environment Variables
ENV nnUNet_results="/nnUnet_results"
ENV HOME="/tmp"

# Entrypoint script for inference
COPY inference_entrypoint.sh /app/inference_entrypoint.sh
RUN chmod +x /app/inference_entrypoint.sh

ENTRYPOINT ["/app/inference_entrypoint.sh"]
