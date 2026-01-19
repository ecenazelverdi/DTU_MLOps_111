#!/bin/bash
set -e

echo "Pulling models from DVC..."
# Configure dvc to not ask for confirmation
dvc config core.no_scm true
dvc pull nnUNet_results.dvc

echo "Starting application..."
# Use the installed system uvicorn directly to avoid 'uv run' creating a new env
exec uvicorn main:app --host 0.0.0.0 --port $PORT