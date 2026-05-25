"""
app.py — Flask backend for Smart Road Assistant

Endpoints:
    GET  /health          — model status
    POST /detect/image    — single image detection (multipart, field: image)
    POST /detect/video    — video detection       (multipart, field: video)
    GET  /video/output/<job_id>  — retrieve processed video
"""

import sys
import uuid
import base64
import threading
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
from pipeline import Pipeline

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Load pipeline once at startup
pipeline = Pipeline()

# In-memory job tracker for async video processing
_video_jobs: dict = {}   # job_id → {"status": "processing"|"done"|"error", "path": Path}
_jobs_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(frame) -> str:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    from pipeline import DETECTOR_MODEL_PATH
    from analyzers.pothole_analyzer import SEVERITY_MODEL_PATH
    return jsonify({
        "status":          "ok",
        "detector_model":  DETECTOR_MODEL_PATH.exists(),
        "severity_model":  SEVERITY_MODEL_PATH.exists(),
    })


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = pipeline.run(frame)

    return jsonify({
        "annotated_image": encode_image(result["annotated_frame"]),
        "warnings":        result["warnings"],
        "potholes": [
            {
                "severity":    d["severity"],
                "distance":    d["distance"],
                "confidence":  round(d["conf"], 3),
                "instructions": d["instructions"],
                "box":         [round(v, 1) for v in d["box"]],
                "cv_scores":   d.get("cv_scores", {}),
            }
            for d in result["potholes"]
        ],
        "traffic_lights": [
            {
                "color":        d["color"],
                "lane_relevant": d["lane_relevant"],
                "confidence":   round(d["conf"], 3),
                "instructions": d["instructions"],
                "box":          [round(v, 1) for v in d["box"]],
            }
            for d in result["traffic_lights"]
        ],
    })


@app.route("/detect/video", methods=["POST"])
def detect_video():
    """
    Accepts a video file, starts processing in background.
    Returns a job_id immediately; poll /video/status/<job_id>.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    file   = request.files["video"]
    job_id = str(uuid.uuid4())

    job_dir  = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path  = job_dir / "input.mp4"
    out_path = job_dir / "output.mp4"

    file.save(str(in_path))

    with _jobs_lock:
        _video_jobs[job_id] = {"status": "processing", "path": out_path}

    def _process():
        try:
            pipeline.run_video(str(in_path), str(out_path))
            with _jobs_lock:
                _video_jobs[job_id]["status"] = "done"
        except Exception as exc:
            with _jobs_lock:
                _video_jobs[job_id]["status"] = "error"
                _video_jobs[job_id]["error"]  = str(exc)

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/video/status/<job_id>", methods=["GET"])
def video_status(job_id: str):
    with _jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify({"job_id": job_id, "status": job["status"]})


@app.route("/video/output/<job_id>", methods=["GET"])
def serve_video(job_id: str):
    with _jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"Video not ready — status: {job['status']}"}), 202
    out_path = job["path"]
    if not out_path.exists():
        return jsonify({"error": "Output file missing"}), 500
    return send_file(str(out_path), mimetype="video/mp4")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Road Assistant — Backend")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
