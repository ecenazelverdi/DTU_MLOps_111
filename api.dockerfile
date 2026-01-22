FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    nnUNet_raw="/app/nnUNet_raw" \
    nnUNet_preprocessed="/app/nnUNet_preprocessed" \
    nnUNet_results="/app/nnUNet_results" \
    BUCKET_NAME=dtumlops-111-data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency definition
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
# Copy lock file if it exists
COPY uv.lock . 

# Copy application code
COPY src/ src/
COPY main.py .

# Install dependencies
RUN uv pip install --system -e .

# Copy DVC configuration and pointer file
COPY .dvc/ .dvc/
COPY nnUNet_results.dvc .

# Copy entrypoint script
# Copy entrypoint script
COPY api_entrypoint.sh .
RUN chmod +x api_entrypoint.sh

# Expose the port
EXPOSE $PORT

# Run the entrypoint script (Pulls data at runtime)
CMD ["./api_entrypoint.sh"]
