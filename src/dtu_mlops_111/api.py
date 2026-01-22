import http
import io
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from datetime import datetime

from dtu_mlops_111.model import Model
from dtu_mlops_111.utils import array_to_base64
from dtu_mlops_111.data import LABELS

import json
import uuid
import os
from dotenv import load_dotenv
from google.cloud import storage

from fastapi.responses import JSONResponse
from dtu_mlops_111.data_drift import get_drift_report_html, upload_drift_report

load_dotenv()

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

# GCS Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Global model instance
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
    filename: str
    prediction_shape: list[int]
    classes_found: list[int]
    segmentation_mask: str


def log_inference(prediction: np.ndarray, filename: str):
    """
    Background task to log inference statistics to Google Cloud Storage.
    """
    try:
        # Calculate pixel counts
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        # Create a dict of counts for present classes
        counts_dict = dict(zip(unique, counts))

        # Prepare log entry (JSON friendly)
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "filename": filename,
            "classes": {}
        }

        # LABELS indices are 0..5 (from data.py)
        for class_idx in LABELS.keys():
            count = counts_dict.get(class_idx, 0)
            percentage = float(count / total_pixels) # Ensure float for JSON
            class_name = LABELS[class_idx]
            log_entry["classes"][class_name] = percentage

        # Upload to GCS
        try:
            if not BUCKET_NAME:
                print("BUCKET_NAME not set in environment. Skipping GCS upload.")
                return

            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            # Use timestamp and uuid to ensure uniqueness
            blob_name = f"inference_logs/{timestamp}_{uuid.uuid4()}.json"
            blob = bucket.blob(blob_name)
            
            blob.upload_from_string(
                json.dumps(log_entry),
                content_type='application/json'
            )
            print(f"Logged to gs://{BUCKET_NAME}/{blob_name}")
            
        except Exception as e:
            print(f"GCS Upload failed: {e}")

    except Exception as e:
        print(f"Logging calculation failed: {e}")

@app.get("/", status_code=http.HTTPStatus.OK)
def read_root():
    """Health check."""
    request_counter.inc()
    status = "model_loaded" if model else "model_failed"
    return {"status": "ok", "model_status": status}

async def process_single_image(file: UploadFile, background_tasks: BackgroundTasks = None, endpoint: str = "predict") -> PredictionResponse:
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
async def predict(background_tasks: BackgroundTasks = None, data: UploadFile = File(...)):
    """
    Run inference on an uploaded image.
    
    Args:
        data: The image file to process.
        
    Returns:
        PredictionResponse containing filename, shape, classes found, and Base64 mask.
    """
    request_counter.inc()
    return await process_single_image(data, background_tasks, endpoint="predict")

@app.post("/batch_predict/", response_model=List[PredictionResponse])
async def batch_predict(background_tasks: BackgroundTasks = None, data: List[UploadFile] = File(...)):
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
        result = await process_single_image(file, background_tasks, endpoint="batch_predict")
        results.append(result)
    return results

@app.get("/drift/")
async def drift():
    """
    Generate and save the data drift report.
    Returns:
        JSON object containing the signed URL to the saved report.
    """
    request_counter.inc()
    try:
        html_content = get_drift_report_html()
        url = upload_drift_report(html_content, bucket_name=BUCKET_NAME)
        return JSONResponse(content={"url": url, "message": "Report saved to GCS."})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate report: {str(e)}"})

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
    request_counter.inc()
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
