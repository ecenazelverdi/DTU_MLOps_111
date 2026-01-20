import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import DatasetMissingValueCount
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

def main():
    # 1. Load Reference Data (Training Set)
    # Using nnUNet_preprocessed ground truth segmentations
    train_data_path = Path("nnUNet_preprocessed/Dataset101_DroneSeg/gt_segmentations")
    if not train_data_path.exists():
        print(f"Training data path {train_data_path} not found.")
        return

    reference_data = load_reference_data(train_data_path, limit=200)
    print(f"Reference data shape: {reference_data.shape}")

    # 2. Load Current Data (Inference Logs)
    log_path = Path("inference_logs.csv")
    if not log_path.exists():
        print(f"Inference log {log_path} not found. Run the app and make predictions first.")
        return
        
    current_data = pd.read_csv(log_path)
    
    # Ensure columns match (drop extra columns from logs like timestamp/filename for drift calc)
    target_columns = list(LABELS.values())
    
    # Filter only relevant columns for drift detection
    reference_data = reference_data[target_columns]
    
    # Check if current_data has these columns
    missing_cols = [c for c in target_columns if c not in current_data.columns]
    if missing_cols:
        print(f"Error: Log file missing columns: {missing_cols}")
        return
        
    current_data = current_data[target_columns]
    print(f"Current data shape: {current_data.shape}")
    
    if len(current_data) < 5:
        print("Warning: Very few inference logs found. Report may not be statistically significant.")

    # 3. Run Report
    print("Running data drift report...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(), 
        DatasetMissingValueCount()
    ])
    
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "data_drift.html"
    
    snapshot.save_html(str(output_path))
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()
