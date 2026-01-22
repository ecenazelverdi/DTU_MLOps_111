import base64
import io
import bentoml
from PIL import Image

if __name__ == "__main__":
    img = Image.open("data/sample.png").convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        resp = client.predict_base64(image_b64=b64, content_type="image/png")
        print(resp.keys())
        print("classes_found:", resp["classes_found"])
        print("prediction_shape:", resp["prediction_shape"])
