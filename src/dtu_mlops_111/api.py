import http
import io
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from PIL import Image
from pydantic import BaseModel

import csv
from datetime import datetime
from pathlib import Path

from dtu_mlops_111.model import Model
from dtu_mlops_111.utils import array_to_base64
from dtu_mlops_111.data import LABELS

app = FastAPI()

# Global model instance
try:
    model = Model()
except Exception as e:
    print(f"Failed to initialize model: {e}")
    model = None

class PredictionResponse(BaseModel):
    filename: str
    prediction_shape: list[int]
    classes_found: list[int]
    segmentation_mask: str

def log_inference(prediction: np.ndarray, filename: str):
    """
    Background task to log inference statistics to CSV.
    """
    try:
        # Calculate pixel counts
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        counts_dict = dict(zip(unique, counts))
        
        # Prepare log entry
        timestamp = datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "filename": filename}
        
        # LABELS indices are 0..5 (from data.py)
        for class_idx in LABELS.keys():
            count = counts_dict.get(class_idx, 0)
            percentage = count / total_pixels
            class_name = LABELS[class_idx]
            log_entry[class_name] = percentage
            
        # Write to CSV
        log_path = Path("inference_logs.csv")
        file_exists = log_path.exists()
        
        with open(log_path, mode='a', newline='') as f:
            fieldnames = ["timestamp", "filename"] + list(LABELS.values())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(log_entry)
            
    except Exception as e:
        print(f"Logging failed: {e}")

@app.get("/", status_code=http.HTTPStatus.OK)
def read_root():
    """Health check."""
    status = "model_loaded" if model else "model_failed"
    return {"status": "ok", "model_status": status}

async def process_single_image(file: UploadFile, background_tasks: BackgroundTasks = None) -> PredictionResponse:
    """Helper to process a single image file."""
    if not model:
        raise http.HTTPException(status_code=500, detail="Model not loaded")

    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(image).transpose(2, 0, 1)
    
    prediction = model.predict(img_np)
    mask_b64 = array_to_base64(prediction)

    # Schedule logging as a background task if available
    if background_tasks:
        background_tasks.add_task(log_inference, prediction, file.filename)
    
    return PredictionResponse(
        filename=file.filename,
        prediction_shape=list(prediction.shape),
        classes_found=list(np.unique(prediction)),
        segmentation_mask=mask_b64
    )

@app.post("/predict/", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, data: UploadFile = File(...)):
    """
    Run inference on an uploaded image.
    
    Args:
        data: The image file to process.
        
    Returns:
        PredictionResponse containing filename, shape, classes found, and Base64 mask.
    """
    return await process_single_image(data, background_tasks)

@app.post("/batch_predict/", response_model=List[PredictionResponse])
async def batch_predict(data: List[UploadFile] = File(...)):
    """
    Run inference on multiple uploaded images.

    Args:
        data: List of image files to process.
        
    Returns:
        List[PredictionResponse] containing results for each image.
    """
    results = []
    for file in data:
        result = await process_single_image(file)
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
