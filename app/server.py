from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = APP_DIR / "frontend"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

sys.path.insert(0, str(APP_DIR))

from predict import predict_image
from gradcam import generate_gradcam

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def image_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1100, 1100))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=86, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(FRONTEND_DIR, filename)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": "ResNet18"})


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image before starting the analysis."}), 400

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        return jsonify({"error": "Please upload a JPG, JPEG or PNG image."}), 400

    raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        return jsonify({"error": "The selected image is empty."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "The image is larger than 10 MB."}), 413

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "The selected file is not a readable image."}), 400

    try:
        prediction = predict_image(image)
        explanation = generate_gradcam(image)
        return jsonify({
            "predictedClass": prediction["predicted_class"],
            "confidence": float(prediction["confidence"]),
            "probabilities": prediction["probabilities"],
            "originalImage": image_data_url(image),
            "gradcamImage": image_data_url(explanation),
        })
    except Exception:
        app.logger.exception("Image analysis failed")
        return jsonify({"error": "Analysis could not be completed. Check the model files and try again."}), 500


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "The image is larger than 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
