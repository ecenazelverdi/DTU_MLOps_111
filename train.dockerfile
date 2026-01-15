FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Install Python dependencies (PyTorch already in base image)
RUN pip install --no-cache-dir \
    nnunetv2 \
    scikit-learn \
    python-dotenv \
    typer \
    tqdm \
    Pillow \
    numpy

# Create nnUNet directories
RUN mkdir -p /app/data/nnUNet_raw /app/data/nnUNet_preprocessed /app/data/nnUNet_results

# Environment variables
ENV nnUNet_raw="/app/data/nnUNet_raw"
ENV nnUNet_preprocessed="/app/data/nnUNet_preprocessed"
ENV nnUNet_results="/app/data/nnUNet_results"
ENV DATASET_ID=101
ENV CONFIG="2d"
ENV FOLD=0
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["/app/entrypoint.sh"]