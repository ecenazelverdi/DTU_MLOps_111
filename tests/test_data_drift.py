import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dtu_mlops_111.data_drift import calculate_label_distribution, load_reference_data

# Constants for testing
GCS_PREFIX = "nnUNet_preprocessed/Dataset101_DroneSeg/gt_segmentations/"


def create_test_mask_bytes(size=(10, 10), value=0):
    """Helper function to create test mask data as bytes."""
    mask_arr = np.zeros(size, dtype=np.uint8)
    mask_arr[:] = value
    mask_img = Image.fromarray(mask_arr, mode="L")
    bio = io.BytesIO()
    mask_img.save(bio, format="PNG")
    return bio.getvalue()


def test_calculate_label_distribution_basic(tmp_path):
    """Test basic label distribution calculation with a simple mask."""
    # Create a simple 10x10 mask with 3 classes
    # Class 0: 30 pixels, Class 1: 20 pixels, Class 2: 50 pixels
    mask_arr = np.zeros((10, 10), dtype=np.uint8)
    mask_arr[:5, :6] = 0  # 30 pixels (5x6) of class 0 (background)
    mask_arr[:5, 6:] = 1  # 20 pixels (5x4) of class 1
    mask_arr[5:, :] = 2   # 50 pixels (5x10) of class 2
    
    # Save as L mode image
    mask_path = tmp_path / "test_mask.png"
    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_img.save(mask_path)
    
    # Calculate distribution
    distribution = calculate_label_distribution(mask_path)
    
    # Verify distribution - should have percentages for each class
    assert isinstance(distribution, dict)
    assert "background" in distribution  # class 0
    assert distribution["background"] == pytest.approx(0.3, abs=0.01)


def test_calculate_label_distribution_with_bytesio():
    """Test label distribution calculation with BytesIO (simulating GCS download)."""
    # Create a simple mask
    mask_arr = np.zeros((10, 10), dtype=np.uint8)
    mask_arr[:5, :] = 0
    mask_arr[5:, :] = 1
    
    # Save to BytesIO
    mask_img = Image.fromarray(mask_arr, mode="L")
    bio = io.BytesIO()
    mask_img.save(bio, format="PNG")
    bio.seek(0)
    
    # Calculate distribution
    distribution = calculate_label_distribution(bio)
    
    assert isinstance(distribution, dict)
    assert "background" in distribution
    assert distribution["background"] == pytest.approx(0.5, abs=0.01)


def test_load_reference_data_local_path(tmp_path):
    """Test loading reference data from local path."""
    # Create a local directory with test masks
    data_path = tmp_path / "gt_segmentations"
    data_path.mkdir()
    
    # Create 3 test masks
    for i in range(3):
        mask_arr = np.zeros((10, 10), dtype=np.uint8)
        mask_arr[:5, :] = i % 2
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_img.save(data_path / f"mask_{i}.png")
    
    # Load reference data
    df = load_reference_data(data_path=data_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "background" in df.columns


def test_load_reference_data_local_path_with_limit(tmp_path):
    """Test loading reference data with a limit on number of samples."""
    data_path = tmp_path / "gt_segmentations"
    data_path.mkdir()
    
    # Create 5 test masks
    for i in range(5):
        mask_arr = np.zeros((10, 10), dtype=np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_img.save(data_path / f"mask_{i}.png")
    
    # Load with limit of 2
    df = load_reference_data(data_path=data_path, limit=2)
    
    assert len(df) == 2


def test_load_reference_data_local_path_not_exists():
    """Test that local path that doesn't exist falls back properly."""
    non_existent_path = Path("/nonexistent/path")
    
    # Should raise ValueError when no bucket_name is provided
    with pytest.raises(ValueError, match="Either data_path .* or bucket_name .* must be provided/valid"):
        load_reference_data(data_path=non_existent_path)


def test_load_reference_data_no_masks_found(tmp_path):
    """Test handling when no mask files are found."""
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    
    df = load_reference_data(data_path=empty_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_success(mock_storage_client):
    """Test successful loading of reference data from GCS bucket."""
    # Setup mock GCS client
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    
    # Create mock blobs
    mock_blobs = []
    for i in range(3):
        mock_blob = MagicMock()
        mock_blob.name = f"{GCS_PREFIX}mask_{i}.png"
        mock_blob.download_as_bytes.return_value = create_test_mask_bytes()
        mock_blobs.append(mock_blob)
    
    mock_bucket.list_blobs.return_value = mock_blobs
    
    # Load reference data from GCS
    df = load_reference_data(bucket_name="test-bucket")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    mock_client.bucket.assert_called_once_with("test-bucket")
    mock_bucket.list_blobs.assert_called_once_with(prefix=GCS_PREFIX)


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_no_blobs(mock_storage_client):
    """Test GCS loading when no blobs are found (missing data case)."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    
    # Return empty list of blobs
    mock_bucket.list_blobs.return_value = []
    
    df = load_reference_data(bucket_name="test-bucket")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_connection_failure(mock_storage_client):
    """Test handling of GCS connection failures."""
    # Simulate connection failure
    mock_storage_client.side_effect = Exception("Connection failed")
    
    # Should handle the exception gracefully and return empty DataFrame
    df = load_reference_data(bucket_name="test-bucket")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_list_blobs_failure(mock_storage_client):
    """Test handling when list_blobs operation fails."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    
    # Simulate list_blobs failure
    mock_bucket.list_blobs.side_effect = Exception("List blobs failed")
    
    # Should handle the exception and return empty DataFrame
    df = load_reference_data(bucket_name="test-bucket")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_filters_png_only(mock_storage_client):
    """Test that only .png files are loaded from GCS."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    
    # Create mock blobs with different extensions
    mock_blob_png = MagicMock()
    mock_blob_png.name = f"{GCS_PREFIX}mask.png"
    mock_blob_png.download_as_bytes.return_value = create_test_mask_bytes()
    
    mock_blob_jpg = MagicMock()
    mock_blob_jpg.name = f"{GCS_PREFIX}mask.jpg"
    
    mock_bucket.list_blobs.return_value = [mock_blob_png, mock_blob_jpg]
    
    df = load_reference_data(bucket_name="test-bucket")
    
    # Only PNG should be processed
    assert len(df) == 1


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_with_limit(mock_storage_client):
    """Test GCS loading with limit parameter."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    
    # Create 5 mock blobs
    mock_blobs = []
    for i in range(5):
        mock_blob = MagicMock()
        mock_blob.name = f"{GCS_PREFIX}mask_{i}.png"
        mock_blob.download_as_bytes.return_value = create_test_mask_bytes()
        mock_blobs.append(mock_blob)
    
    mock_bucket.list_blobs.return_value = mock_blobs
    
    # Load with limit
    df = load_reference_data(bucket_name="test-bucket", limit=2)
    
    assert len(df) == 2


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_local_preferred_over_gcs(mock_storage_client, tmp_path):
    """Test that local path is preferred over GCS when both are provided and local exists."""
    # Create local data
    data_path = tmp_path / "gt_segmentations"
    data_path.mkdir()
    
    mask_arr = np.zeros((10, 10), dtype=np.uint8)
    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_img.save(data_path / "mask.png")
    
    # Load - should use local path and not call GCS
    df = load_reference_data(data_path=data_path, bucket_name="test-bucket")
    
    assert len(df) == 1
    # Storage client should not be called when local path exists
    mock_storage_client.assert_not_called()


@patch("dtu_mlops_111.data_drift.storage.Client")
def test_load_reference_data_from_gcs_download_failure(mock_storage_client):
    """Test handling when individual blob download fails."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    
    # Create mock blobs - one succeeds, one fails
    mock_blob_success = MagicMock()
    mock_blob_success.name = f"{GCS_PREFIX}mask_1.png"
    mock_blob_success.download_as_bytes.return_value = create_test_mask_bytes()
    
    mock_blob_fail = MagicMock()
    mock_blob_fail.name = f"{GCS_PREFIX}mask_2.png"
    mock_blob_fail.download_as_bytes.side_effect = Exception("Download failed")
    
    mock_bucket.list_blobs.return_value = [mock_blob_success, mock_blob_fail]
    
    # Should continue processing and skip failed downloads
    df = load_reference_data(bucket_name="test-bucket")
    
    # Only the successful blob should be in the result
    assert len(df) == 1
