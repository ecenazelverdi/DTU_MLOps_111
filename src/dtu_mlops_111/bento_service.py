from __future__ import annotations

import os
import subprocess
from typing import Any, Dict

import bentoml
import numpy as np
from PIL import Image

from dtu_mlops_111.model import Model
from dtu_mlops_111.utils import array_to_base64


def _checkpoint_exists() -> bool:
    return os.path.exists(
        os.path.join(
            "nnUNet_results",
            "Dataset101_DroneSeg",
            "nnUNetTrainer__nnUNetPlans__2d",
            "fold_0",
            "checkpoint_best.pth",
        )
    )


@bentoml.service(
    resources={"cpu": "2"},          
    traffic={"timeout": 60},         
)
class DroneSegService:
    """
    Drone semantic segmentation service using BentoML.

    Input: image as numpy array in CHW format (uint8) or HWC (uint8)
    Output: base64-encoded PNG mask (color-coded) + metadata
    """

    def __init__(self) -> None:
        if not _checkpoint_exists():
            try:
                subprocess.run(["dvc", "pull", "nnUNet_results.dvc"], check=True)
                print("[BentoML] Pulled nnUNet_results via DVC.")
            except Exception as e:
                print(f"[BentoML] DVC pull failed (continuing; model may run dummy): {e}")

        # Model init 
        self.model = Model()

    @bentoml.api
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict segmentation mask from an input image array.

        Expected:
          - HWC uint8 (H, W, 3)  OR
          - CHW uint8 (3, H, W)

        Returns:
          - base64 encoded PNG of colorized segmentation mask
        """
        if image.ndim == 3 and image.shape[-1] == 3:
            # HWC -> CHW
            img_np = np.transpose(image, (2, 0, 1))
        elif image.ndim == 3 and image.shape[0] == 3:
            img_np = image
        else:
            raise ValueError(f"Expected image with shape (H,W,3) or (3,H,W), got {image.shape}")

        pred = self.model.predict(img_np)

        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]

        return {
            "prediction_shape": list(pred.shape),
            "classes_found": [int(x) for x in np.unique(pred)],
            "segmentation_mask": array_to_base64(pred),
        }
