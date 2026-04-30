"""
app.py — Flask backend
Receives image/video from frontend, runs pipeline, returns results.

Place this file at: backend/app.py
Run: python backend/app.py
"""

import sys
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Add project root to path so we can import src/pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Load pipeline once at startup (loads all available models)
pipeline = Pipeline()


def encode_image(frame):
    """Convert OpenCV BGR frame to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Check which models are loaded."""
    from src.pipeline import POTHOLE_MODEL_PATH, TRAFFIC_MODEL_PATH, GATE_MODEL_PATH
    return jsonify({
        "status"        : "ok",
        "pothole_model" : POTHOLE_MODEL_PATH.exists(),
        "traffic_model" : TRAFFIC_MODEL_PATH.exists(),
        "gate_model"    : GATE_MODEL_PATH.exists(),
    })


@app.route("/detect/image", methods=["POST"])
def detect_image():
    """
    Accepts: multipart form with field 'image' (image file)
    Returns: JSON with annotated image (base64), warnings, detections
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Decode image from bytes
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = pipeline.run(frame)

    return jsonify({
        "annotated_image": encode_image(result["annotated_frame"]),
        "warnings"       : result["warnings"],
        "weather"        : result["weather"],
        "traffic_lights" : [
            {
                "color"     : d["color"],
                "confidence": round(d["conf"], 3),
                "box"       : [round(v, 1) for v in d["box"]],
            }
            for d in result["traffic_lights"]
        ],
        "potholes": [
            {
                "severity"  : d["severity"],
                "distance"  : d["distance"],
                "confidence": round(d["conf"], 3),
                "box"       : [round(v, 1) for v in d["box"]],
            }
            for d in result["potholes"]
        ],
    })


@app.route("/detect/video", methods=["POST"])
def detect_video():
    """
    Accepts: multipart form with field 'video' (video file)
    Returns: JSON with path to processed video
    """
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    file     = request.files["video"]
    in_path  = str(UPLOAD_DIR / "input.mp4")
    out_path = str(UPLOAD_DIR / "output.mp4")

    file.save(in_path)
    pipeline.run_video(in_path, output_path=out_path)

    return jsonify({"message": "Video processed", "output": "/video/output"})


@app.route("/video/output", methods=["GET"])
def serve_video():
    out_path = UPLOAD_DIR / "output.mp4"
    if not out_path.exists():
        return jsonify({"error": "No video available"}), 404
    return send_file(str(out_path), mimetype="video/mp4")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print(" Smart Road Assistant — Backend")
    print(f" Running at: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)