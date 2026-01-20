import bentoml
import numpy as np
from PIL import Image


if __name__ == "__main__":
    # kendi test görselini koy
    img = Image.open("data/sample.png").convert("RGB")
    img = np.array(img, dtype=np.uint8)  # HWC

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        resp = client.predict(image=img)
        print(resp.keys())
        print("classes_found:", resp["classes_found"])
        print("prediction_shape:", resp["prediction_shape"])
