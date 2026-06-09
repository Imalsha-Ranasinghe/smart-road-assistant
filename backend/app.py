"""
app.py — Flask backend for Smart Road Assistant

Endpoints:
    GET  /health                  — model load status
    POST /detect/image            — single image detection   (field: image)
    POST /detect/video            — start async video job    (field: video)
    GET  /video/status/<job_id>   — poll job progress
    GET  /video/output/<job_id>   — download annotated video
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

# Job tracker  {job_id: {status, path, progress, frames_done, frames_total, error}}
_jobs: dict = {}
_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def frame_to_b64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")


def serialize_pothole(d: dict) -> dict:
    return {
        "severity":     d["severity"],
        "distance":     d["distance"],
        "confidence":   round(d["conf"], 3),
        "instructions": d["instructions"],
        "box":          [round(v, 1) for v in d["box"]],
        "area_px":      d.get("area_px", 0),
        "cv_scores":    d.get("cv_scores", {}),
    }


def serialize_traffic(d: dict) -> dict:
    return {
        "color":            d["color"],
        "lane_relevant":    d["lane_relevant"],
        "confidence":       round(d["conf"], 3),
        "instructions":     d["instructions"],
        "box":              [round(v, 1) for v in d["box"]],
        "color_confidence": d.get("color_confidence", 0),
        "position_vote":    d.get("position_vote", ""),
        "color_vote":       d.get("color_vote", ""),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    from pipeline import DETECTOR_MODEL_PATH, DASHCAM_MODEL_PATH
    from analyzers.pothole_analyzer import SEVERITY_MODEL_PATH
    import yaml as _yaml
    severity_method = "auto"
    cfg_path = BASE_DIR / "configs" / "pipeline_config.yaml"
    if cfg_path.exists():
        cfg = _yaml.safe_load(cfg_path.read_text()) or {}
        severity_method = cfg.get("severity_method", "auto")
    return jsonify({
        "status":           "ok",
        "detector_model":   DETECTOR_MODEL_PATH.exists(),
        "dashcam_model":    DASHCAM_MODEL_PATH.exists(),
        "severity_model":   SEVERITY_MODEL_PATH.exists(),
        "severity_method":  severity_method,
    })


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    # "default" (photo page) or "dashcam" (dashboard); falls back if untrained
    detector = request.form.get("model", "default")
    result = pipeline.run(frame, detector=detector)

    return jsonify({
        "annotated_image": frame_to_b64(result["annotated_frame"]),
        "warnings":        result["warnings"],
        "potholes":        [serialize_pothole(d) for d in result["potholes"]],
        "traffic_lights":  [serialize_traffic(d) for d in result["traffic_lights"]],
        "lane":            result.get("lane"),
        "counts": {
            "potholes":       len(result["potholes"]),
            "traffic_lights": len(result["traffic_lights"]),
            "lane_relevant":  sum(1 for t in result["traffic_lights"] if t["lane_relevant"]),
        },
    })


@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video field in request"}), 400

    file   = request.files["video"]
    job_id = str(uuid.uuid4())

    job_dir  = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path  = job_dir / "input.mp4"
    out_path = job_dir / "output.mp4"
    file.save(str(in_path))

    with _lock:
        _jobs[job_id] = {
            "status":       "processing",
            "path":         out_path,
            "progress":     0.0,
            "frames_done":  0,
            "frames_total": 0,
        }

    def _on_progress(done, total, pct):
        with _lock:
            _jobs[job_id]["frames_done"]  = done
            _jobs[job_id]["frames_total"] = total
            _jobs[job_id]["progress"]     = round(pct, 1)

    def _process():
        try:
            pipeline.run_video(str(in_path), str(out_path),
                               progress_callback=_on_progress)
            with _lock:
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["progress"] = 100.0
        except Exception as exc:
            with _lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"]  = str(exc)

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/video/status/<job_id>", methods=["GET"])
def video_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify({
        "job_id":       job_id,
        "status":       job["status"],
        "progress":     job["progress"],
        "frames_done":  job["frames_done"],
        "frames_total": job["frames_total"],
        "error":        job.get("error"),
    })


@app.route("/video/output/<job_id>", methods=["GET"])
def video_output(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"Not ready — status: {job['status']}",
                        "progress": job["progress"]}), 202
    path = job["path"]
    if not path.exists():
        return jsonify({"error": "Output file missing"}), 500
    return send_file(str(path), mimetype="video/mp4",
                     download_name="road_analysis.mp4", as_attachment=True)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Road Assistant — Backend")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
