import html
import json
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.metrics import DatasetMissingValueCount
from evidently.presets import DataDriftPreset, DataSummaryPreset
from google.cloud import storage
from PIL import Image
from tqdm import tqdm

load_dotenv()

# To prevent error that is caused by background class in the training data with evidently
import warnings

warnings.filterwarnings("ignore")

from dtu_mlops_111.data import LABELS


def calculate_label_distribution(mask_path: Path) -> dict:
    """
    Calculates the percentage of pixels for each class in the mask.
    Returns a dictionary {class_name: percentage}.
    """
    # Masks in nnUNet_preprocessed are already class indices (L mode)
    mask_img = Image.open(mask_path) 
    class_mask = np.array(mask_img)
    
    # Count pixels for each class
    total_pixels = class_mask.size
    counts = np.bincount(class_mask.flatten(), minlength=len(LABELS))
    
    # Calculate percentages
    distribution = {}
    for idx, count in enumerate(counts):
        if idx in LABELS:
            class_name = LABELS[idx]
            distribution[class_name] = count / total_pixels
            
    return distribution

def load_reference_data(data_path: Path, limit: int = None) -> pd.DataFrame:
    print("Loading reference dataset from masks...")
    
    # Directly glob the masks since we are using the preprocessed gt_segmentations folder
    mask_files = sorted(list(data_path.glob("*.png")))
    
    if limit:
        print(f"Limiting reference data to {limit} samples.")
        mask_files = mask_files[:limit]

    data_stats = []
    for mask_file in tqdm(mask_files, desc="Processing reference masks"):
        stats = calculate_label_distribution(mask_file)
        data_stats.append(stats)
        
    return pd.DataFrame(data_stats)

def get_drift_report_html(bucket_name: str = None, limit_ref: int = 200) -> str:
    # 1. Load Reference Data (Training Set)
    # Using nnUNet_preprocessed ground truth segmentations
    train_data_path = Path("nnUNet_preprocessed/Dataset101_DroneSeg/gt_segmentations")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data path {train_data_path} not found. Ensure volume is mounted.")

    reference_data = load_reference_data(train_data_path, limit=limit_ref)
    
    # 2. Load Current Data (Inference Logs from GCS)
    if not bucket_name:
        bucket_name = os.getenv("BUCKET_NAME")
    
    if not bucket_name:
         raise ValueError("BUCKET_NAME not set.")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix="inference_logs/"))
        
        current_data_list = []
        for blob in tqdm(blobs, desc="Downloading logs"):
            if blob.name.endswith(".json"):
                json_str = blob.download_as_string()
                log_entry = json.loads(json_str)
                # Flatten the 'classes' dictionary
                flat_entry = {
                    "timestamp": log_entry["timestamp"],
                    "filename": log_entry["filename"],
                    **log_entry["classes"]
                }
                current_data_list.append(flat_entry)
                
        if not current_data_list:
             raise ValueError(f"No inference logs found in gs://{bucket_name}/inference_logs/")
             
        current_data = pd.DataFrame(current_data_list)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load logs from GCS: {e}") from e
    
    # Ensure columns match
    target_columns = list(LABELS.values())
    reference_data = reference_data[target_columns]
    
    missing_cols = [c for c in target_columns if c not in current_data.columns]
    if missing_cols:
        for c in missing_cols:
            current_data[c] = 0.0
        
    current_data = current_data[target_columns]
    
    if len(current_data) < 5:
        print("Warning: Very few inference logs found.")

    # 3. Run Report
    print("Running data drift report...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(), 
        DatasetMissingValueCount()
    ])
    
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    # Save to temp file and read back 
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=True) as tmp:
        snapshot.save_html(tmp.name)
        tmp.seek(0)
        content = tmp.read()
    
    return content

def upload_drift_report(html_content: str, bucket_name: str = None) -> str:
    """
    Uploads the HTML report to GCS and returns a signed URL.
    """
    if not bucket_name:
        raise ValueError("BUCKET_NAME not set via argument.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"reports/drift_report_{timestamp}.html"
    blob = bucket.blob(blob_name)
    
    # Upload
    blob.upload_from_string(html_content, content_type='text/html')
    print(f"Uploaded report to gs://{bucket_name}/{blob_name}")
    
    # Return standard storage URL
    return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

def main():
    html_content = get_drift_report_html()
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "data_drift.html"
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()
