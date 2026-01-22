FROM python:3.12-slim

# runtime configuration (Cloud Run uses PORT; nnU-Net paths are used by inference code)
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    nnUNet_raw="/app/nnUNet_raw" \
    nnUNet_preprocessed="/app/nnUNet_preprocessed" \
    nnUNet_results="/app/nnUNet_results"

# System dependencies:
# - build-essential: required for compiling some Python packages
# - git: required by DVC (and optionally for dependency resolution)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python runtime dependencies required by the BentoML service
COPY requirements.runtime.txt .
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY src/ src/

# Copy DVC metadata and model artifact pointer file
# The actual artifacts are pulled at runtime by the entrypoint if ENABLE_DVC_PULL=1
COPY .dvc/ .dvc/
COPY nnUNet_results.dvc .

# Container entrypoint
COPY bento_entrypoint.sh .
RUN chmod +x bento_entrypoint.sh

EXPOSE $PORT
CMD ["./bento_entrypoint.sh"]
