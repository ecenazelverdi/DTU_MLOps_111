#!/usr/bin/env bash
set -euo pipefail

log() { echo "[bento] $*"; }

PORT="${PORT:-8080}"
log "PORT=${PORT}"

export PYTHONPATH="${PYTHONPATH:-/app/src}"
log "PYTHONPATH=${PYTHONPATH}"

# Configure nnU-Net data directories
export nnUNet_raw="${nnUNet_raw:-/app/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/app/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/app/nnUNet_results}"
log "nnUNet_raw=${nnUNet_raw}"
log "nnUNet_preprocessed=${nnUNet_preprocessed}"
log "nnUNet_results=${nnUNet_results}"

# Pull model artifacts via DVC at container startup.
# ENABLE_DVC_PULL=1 enables this behavior.
# DVC_STRICT=1 causes the container to fail fast if pull fails
DVC_STRICT="${DVC_STRICT:-0}"

if [[ "${ENABLE_DVC_PULL:-0}" == "1" ]]; then
  log "Pulling model artifacts from DVC..."
  dvc config core.no_scm true

  if [[ "${DVC_STRICT}" == "1" ]]; then
    dvc pull nnUNet_results.dvc
    log "DVC pull completed successfully."
  else
    if dvc pull nnUNet_results.dvc; then
      log "DVC pull completed successfully."
    else
      log "DVC pull failed (DVC_STRICT=0). Continuing startup."
    fi
  fi
else
  log "Skipping DVC pull (ENABLE_DVC_PULL!=1)."
fi

log "Starting BentoML service..."
exec bentoml serve dtu_mlops_111.bento_service:DroneSegService \
  --host 0.0.0.0 \
  --port "${PORT}"
