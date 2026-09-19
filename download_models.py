"""
download_models.py
Utility script to ensure YOLOv8-Face and ArcFace MobileFaceNet ONNX models
are present in the models/ directory.
Automatically invoked during Docker image build or on application startup.
"""

import os
import urllib.request

MODELS = {
    "yolov8n-face.onnx": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx",
    "w600k_mbf.onnx": "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000000:
            print(f"[Model Loader] Downloading {filename} from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"[Model Loader] Successfully downloaded {filename} ({os.path.getsize(filepath)} bytes).")
        else:
            print(f"[Model Loader] Model {filename} already present ({os.path.getsize(filepath)} bytes).")


if __name__ == "__main__":
    ensure_models()
