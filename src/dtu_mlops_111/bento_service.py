from __future__ import annotations

import base64
import io
import os
import subprocess
from typing import Any, Dict

import bentoml
import numpy as np
from bentoml import Context
from PIL import Image
from pydantic import BaseModel

from dtu_mlops_111.model import Model
from dtu_mlops_111.utils import array_to_base64


# -------------------------------------------------------------------
# Artifact management (DVC)
# -------------------------------------------------------------------
def _checkpoint_exists() -> bool:
    """Return True if the trained nnU-Net checkpoint is present locally."""
    return os.path.exists(
        os.path.join(
            "nnUNet_results",
            "Dataset101_DroneSeg",
            "nnUNetTrainer__nnUNetPlans__2d",
            "fold_0",
            "checkpoint_best.pth",
        )
    )


def _try_dvc_pull() -> None:
    try:
        subprocess.run(["dvc", "pull", "nnUNet_results.dvc"], check=True)
        print("[BentoML] Model artifacts pulled via DVC.")
    except Exception as e:
        print(f"[BentoML] DVC pull failed (continuing): {e}")


# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------
class PredictBase64Request(BaseModel):
    """Request payload for base64-encoded image inference."""

    image_b64: str
    content_type: str = "image/jpeg"


# -------------------------------------------------------------------
# BentoML service
# -------------------------------------------------------------------
@bentoml.service(
    name="drone-seg-service",
    resources={"cpu": "2"},
    traffic={"timeout": 120},
)
class DroneSegService:
    """
    Semantic segmentation inference service for the DroneSeg nnU-Net model.

    Exposed endpoints:
      - POST /healthz
      - POST /model_info
      - POST /predict_base64
      - POST /predict_image_bytes (best-effort; see notes)
    """

    def __init__(self) -> None:
        if os.environ.get("ENABLE_DVC_PULL", "0") == "1" and not _checkpoint_exists():
            _try_dvc_pull()

        self.model = Model()

    def _predict_from_hwc(self, arr_hwc: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an RGB image in HWC layout.

        Args:
            arr_hwc: RGB image as uint8 numpy array with shape (H, W, 3).

        Returns:
            JSON-serializable dictionary containing segmentation mask and metadata.
        """
        if arr_hwc.ndim != 3 or arr_hwc.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H,W,3), got {arr_hwc.shape}")

        img_chw = np.transpose(arr_hwc, (2, 0, 1)).astype(np.uint8, copy=False)
        pred = self.model.predict(img_chw)

        # Normalize to (H, W) if the model returns (1, H, W)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]

        return {
            "prediction_shape": list(pred.shape),
            "classes_found": [int(x) for x in np.unique(pred)],
            "segmentation_mask": array_to_base64(pred),
            "model_loaded": bool(getattr(self.model, "loaded", False)),
        }

    @bentoml.api
    def healthz(self, ctx: Context) -> Dict[str, Any]:
        """Service health check."""
        return {
            "status": "ok",
            "model_loaded": bool(getattr(self.model, "loaded", False)),
        }

    @bentoml.api
    def model_info(self, ctx: Context) -> Dict[str, Any]:
        """Return model metadata."""
        return self.model.metadata

    @bentoml.api
    def predict_base64(self, req: PredictBase64Request) -> Dict[str, Any]:
        """
        Predict from base64-encoded image bytes in a JSON payload.
        """
        raw = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return self._predict_from_hwc(arr)

    @bentoml.api
    def predict_image_bytes(self, image: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(image)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return self._predict_from_hwc(arr)
