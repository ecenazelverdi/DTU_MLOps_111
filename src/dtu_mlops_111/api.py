import http
import io
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from dtu_mlops_111.model import Model
from dtu_mlops_111.utils import array_to_base64

api_errors = Counter(
    "api_request_errors_total",
    "Total number of API request errors",
    ["endpoint", "reason"],
)
request_counter = Counter("prediction_requests", "Number of prediction requests")
request_latency = Histogram(
    "api_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
)

app = FastAPI()
app.mount("/metrics", make_asgi_app())

# Global model instance
# Ideally we load this once.
# We wrap it in a try-except block just in case initialization fails during import (e.g. CI/CD without weights)
try:
    model = Model()
except Exception as e:
    api_errors.labels(
        endpoint="startup",
        reason="model_init_failed"
    ).inc()
    print(f"Failed to initialize model: {e}")
    model = None

class PredictionResponse(BaseModel):
    filename: str               # The original filename of the uploaded image
    prediction_shape: list[int] # Dimensions of the predicted mask (Height, Width)
    classes_found: list[int]    # Unique class indices identified in the prediction (e.g., [0, 1, 3])
    segmentation_mask: str      # Base64 encoded PNG string of the color-coded segmentation visualization

@app.get("/", status_code=http.HTTPStatus.OK)
def read_root():
    """Health check."""
    status = "model_loaded" if model else "model_failed"
    return {"status": "ok", "model_status": status}

async def process_single_image(file: UploadFile, endpoint: str) -> PredictionResponse:
    """Helper to process a single image file."""
    with request_latency.labels(endpoint=endpoint).time():
        if not model:
            api_errors.labels(
                endpoint=endpoint,
                reason="model_not_loaded"
            ).inc()
            raise http.HTTPException(status_code=500, detail="Model not loaded")

        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(image).transpose(2, 0, 1)

        prediction = model.predict(img_np)
        mask_b64 = array_to_base64(prediction)

        return PredictionResponse(
            filename=file.filename,
            prediction_shape=list(prediction.shape),
            classes_found=list(np.unique(prediction)),
            segmentation_mask=mask_b64
        )

@app.post("/predict/", response_model=PredictionResponse)
async def predict(data: UploadFile = File(...)):
    """
    Run inference on an uploaded image.
    
    Args:
        data: The image file to process.
        
    Returns:
        PredictionResponse containing filename, shape, classes found, and Base64 mask.
    """
    request_counter.inc()
    return await process_single_image(data, endpoint="predict")

@app.post("/batch_predict/", response_model=List[PredictionResponse])
async def batch_predict(data: List[UploadFile] = File(...)):
    """
    Run inference on multiple uploaded images.

    Args:
        data: List of image files to process.
        
    Returns:
        List[PredictionResponse] containing results for each image.
    """
    request_counter.inc(len(data))
    results = []
    for file in data:
        result = await process_single_image(file, endpoint="batch_predict")
        results.append(result)
    return results

class ModelInfo(BaseModel):
    name: str
    version: str
    description: str
    input_shape: str
    output_shape: str
    framework: str
    license: str

@app.get("/model_info/", response_model=ModelInfo, status_code=http.HTTPStatus.OK)
def model_info():
    """
    Get model information.
    """
    if not model:
        return ModelInfo(
            name="nnU-Net (Failed)",
            version="0.0.0",
            description="Failed to load",
            input_shape="N/A",
            output_shape="N/A",
            framework="PyTorch",
            license="MIT"
        )
    return ModelInfo(**model.metadata)
