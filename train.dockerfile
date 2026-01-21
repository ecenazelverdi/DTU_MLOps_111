# Base 'uv' image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Boilerplate essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copying essential components of application to container
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY LICENSE LICENSE
COPY entrypoint.sh entrypoint.sh

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# set working directory in our container to root `corrupt-mnist/`
WORKDIR /

# #  install dependencies
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# *Note*: data should be dealt with different when building on the cloud
COPY data_tiny/nnUNet_preprocessed/ data/nnUNet_preprocessed/

# Create nnUNet directories
RUN mkdir -p data/nnUNet_results

# Environment variables
ENV nnUNet_raw="/data/raw"
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
ENV nnUNet_results="/data/nnUNet_results"
ENV DATASET_ID=101
ENV CONFIG="2d"
ENV FOLD=0
ENV PYTHONPATH="/:${PYTHONPATH}"

ENTRYPOINT ["/entrypoint.sh"]