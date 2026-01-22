import io
from unittest.mock import patch

import numpy as np
import pytest
from dtu_mlops_111.api import app
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(app)


def test_read_root():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_status" in response.json()


def test_model_info_no_model():
    """Test model info when model is not loaded."""
    # Ensure model is None for this test
    with patch("dtu_mlops_111.api.model", None):
        response = client.get("/model_info/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "nnU-Net (Failed)"


@patch("dtu_mlops_111.api.model")
def test_model_info_loaded(mock_model):
    """Test model info when model is successfully loaded."""
    # Setup mock metadata
    mock_model.metadata = {
        "name": "nnU-Net",
        "version": "2.6.2",
        "description": "Drone semantic segmentation",
        "input_shape": "(1, 3, H, W)",
        "output_shape": "(1, 1, H, W)",
        "framework": "PyTorch",
        "license": "MIT",
    }

    response = client.get("/model_info/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "nnU-Net"
    assert data["version"] == "2.6.2"


@patch("dtu_mlops_111.api.model")
def test_predict(mock_model):
    """Test /predict/ endpoint with a mocked model."""
    # Setup mock return value
    # (H, W) output from model.predict (it squeezes internally)
    mock_prediction = np.zeros((100, 100), dtype=np.uint8)
    mock_prediction[50, 50] = 1  # One pixel of class 1
    mock_model.predict.return_value = mock_prediction
    mock_model.metadata = {"name": "Test Model"}

    # Create dummy image
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Send request
    response = client.post("/predict/", files={"data": ("test.png", img_byte_arr, "image/png")})

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.png"
    assert data["classes_found"] == [0, 1]
    assert "segmentation_mask" in data

    # Verify model called
    mock_model.predict.assert_called_once()


@patch("dtu_mlops_111.api.model")
def test_batch_predict(mock_model):
    """Test /batch_predict/ endpoint with mocked model."""
    # Setup mock
    mock_prediction = np.zeros((50, 50), dtype=np.uint8)
    mock_model.predict.return_value = mock_prediction

    # Create dummy image
    img = Image.new("RGB", (50, 50), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Send request with 2 files
    files = [
        ("data", ("img1.png", io.BytesIO(img_byte_arr.getvalue()), "image/png")),
        ("data", ("img2.png", io.BytesIO(img_byte_arr.getvalue()), "image/png")),
    ]

    response = client.post("/batch_predict/", files=files)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["filename"] == "img1.png"
    assert data[1]["filename"] == "img2.png"

    # Verify model called twice
    assert mock_model.predict.call_count == 2
