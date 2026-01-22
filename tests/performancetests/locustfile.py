import io
import os
import random

from locust import HttpUser, between, task
from PIL import Image


def make_png_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a random RGB PNG in-memory for testing the prediction endpoint"""
    # Random solid color; cheap to generate and still exercises the upload + decode path.
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()



class MyUser(HttpUser):
    """A simple Locust user class that can access the FastAPI endpoints."""

    wait_time = between(1, 2)

    def on_start(self):
        # Generate a few images to use in tests
        img_w = int(os.getenv("LOCUST_IMG_W", "256"))
        img_h = int(os.getenv("LOCUST_IMG_H", "256"))
        self.images = [make_png_bytes(img_w, img_h) for _ in range(8)]

        # Set batch size range
        self.batch_min = int(os.getenv("LOCUST_BATCH_MIN", "2"))
        self.batch_max = int(os.getenv("LOCUST_BATCH_MAX", "6"))

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")
    
    @task()
    def predict_single(self):
        img_bytes = random.choice(self.images)

        # FastAPI expects parameter name "data"
        files = {
            "data": ("test.png", img_bytes, "image/png"),
        }

        with self.client.post("/predict/", files=files, name="POST /predict/", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"Unexpected status {r.status_code}: {r.text}")
                return

            # Light sanity checks on response shape
            try:
                js = r.json()
                if "segmentation_mask" not in js:
                    r.failure("Missing 'segmentation_mask' in response")
            except Exception as e:
                r.failure(f"Failed to parse JSON: {e}")


    @task()
    def predict_batch(self):
        batch_n = random.randint(self.batch_min, self.batch_max)
        chosen = random.choices(self.images, k=batch_n)

        # For multiple files under same field name, use list of tuples.
        # FastAPI expects parameter name "data" for List[UploadFile].
        files = [("data", (f"img{i}.png", b, "image/png")) for i, b in enumerate(chosen)]

        with self.client.post(
            "/batch_predict/",
            files=files,
            name="POST /batch_predict/",
            catch_response=True,
        ) as r:
            if r.status_code != 200:
                r.failure(f"Unexpected status {r.status_code}: {r.text}")
                return

            try:
                js = r.json()
                if not isinstance(js, list):
                    r.failure("Expected a list response for batch_predict")
                elif len(js) != batch_n:
                    r.failure(f"Expected {batch_n} items, got {len(js)}")
            except Exception as e:
                r.failure(f"Failed to parse JSON: {e}")