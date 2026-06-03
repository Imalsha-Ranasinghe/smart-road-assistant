"""
pipeline.py — Smart Road Assistant inference pipeline (two-stage cascade)

Stage 1: YOLOv8s detector  → detects potholes and traffic lights
Stage 2a: PotholeAnalyzer  → severity (CNN or CV), distance, instructions
Stage 2b: TrafficAnalyzer  → color (spatial+HSV), lane relevance, instructions

Usage:
    from src.pipeline import Pipeline
    pipe   = Pipeline()
    result = pipe.run(frame)          # BGR numpy array
    result = pipe.run_image("img.jpg")
    result = pipe.run_video("in.mp4", "out.mp4")
    pipe.run_webcam()
"""

import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

# Ensure src/ is importable when running from project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzers.pothole_analyzer import PotholeAnalyzer
from analyzers.traffic_analyzer import TrafficLightAnalyzer

# ── Paths ──────────────────────────────────────────────
MODELS_DIR          = BASE_DIR / "models"
DETECTOR_MODEL_PATH = MODELS_DIR / "detector_model.pt"           # photo page
DASHCAM_MODEL_PATH  = MODELS_DIR / "detector_dashcam.pt"          # dashboard (video) — optional
CONFIG_PATH         = BASE_DIR / "configs" / "pipeline_config.yaml"

# ── Class IDs ──────────────────────────────────────────
CLS_POTHOLE = 0
CLS_TRAFFIC = 1

# ── Draw colors ────────────────────────────────────────
COLOR_POTHOLE = (0, 140, 255)    # orange
COLOR_TL = {
    "Red":     (0, 0, 220),
    "Yellow":  (0, 200, 220),
    "Green":   (0, 200, 0),
    "Unknown": (200, 200, 200),
}
COLOR_TL_FAINT = {k: tuple(int(c * 0.5) for c in v) for k, v in COLOR_TL.items()}


# ── Load pipeline config ────────────────────────────────

def _load_config() -> dict:
    defaults = {
        "severity_method":     "auto",
        "detector_conf":       0.40,
        "lane_x_min":          0.20,
        "lane_x_max":          0.80,
        "lane_min_size_ratio": 0.001,
        "tl_min_aspect_ratio": 1.2,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    return defaults


# ══════════════════════════════════════════════════════
# DRAWING + OUTPUT
# ══════════════════════════════════════════════════════

def _draw_pothole(frame, det):
    x1, y1, x2, y2 = [int(v) for v in det["box"]]
    sev  = det["severity"]
    dist = det["distance"]
    conf = det["conf"]
    label = f"Pothole|{sev}|{dist} {conf:.0%}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_POTHOLE, 2)
    lx = x1 + len(label) * 9
    cv2.rectangle(frame, (x1, y1 - 24), (lx, y1), COLOR_POTHOLE, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _draw_traffic(frame, det):
    x1, y1, x2, y2 = [int(v) for v in det["box"]]
    color  = det["color"]
    conf   = det["conf"]
    relevant = det["lane_relevant"]
    draw_color = COLOR_TL.get(color, COLOR_TL["Unknown"])
    if not relevant:
        draw_color = COLOR_TL_FAINT.get(color, (100, 100, 100))

    label = f"TL:{color} {conf:.0%}" + ("" if relevant else " [adj]")
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
    lx = x1 + len(label) * 9
    cv2.rectangle(frame, (x1, y1 - 24), (lx, y1), draw_color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def draw_detections(frame, potholes, traffic_lights):
    out = frame.copy()
    for det in potholes:
        _draw_pothole(out, det)
    for det in traffic_lights:
        _draw_traffic(out, det)

    # Count summary (top-right)
    relevant_tl = sum(1 for t in traffic_lights if t["lane_relevant"])
    count_text  = f"TL:{relevant_tl}({len(traffic_lights)})  PH:{len(potholes)}"
    cv2.putText(out, count_text,
                (out.shape[1] - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return out


def generate_warnings(potholes, traffic_lights):
    warnings = []
    for det in traffic_lights:
        if det["lane_relevant"]:
            warnings.append(det["instructions"])
    for det in potholes:
        warnings.append(det["instructions"])
    return warnings


# ══════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════

class Pipeline:
    """
    Two-stage cascade inference pipeline.

    Stage 1 — YOLOv8s Detector  (detector_model.pt)
      Classes: pothole (0), traffic_light (1)

    Stage 2a — PotholeAnalyzer  (severity_model.pt or CV fallback)
    Stage 2b — TrafficLightAnalyzer  (pure classical CV)
    """

    def __init__(self):
        cfg = _load_config()

        print("\nLoading Smart Road Assistant pipeline...")
        print("-" * 50)

        # Stage 1 — default detector (photo page)
        if not DETECTOR_MODEL_PATH.exists():
            print(f"  Detector model not found: {DETECTOR_MODEL_PATH}")
            print("  Run: python src/train.py --stage detector")
            self.detector = None
        else:
            self.detector = YOLO(str(DETECTOR_MODEL_PATH))
            print(f"  Stage 1 detector loaded  ({DETECTOR_MODEL_PATH.name})")

        # Stage 1 — optional dashcam detector (dashboard / video page).
        # Trained via notebooks/06-dashcam-detector.ipynb. Falls back to the
        # default detector if it hasn't been trained yet.
        if DASHCAM_MODEL_PATH.exists():
            self.detector_dashcam = YOLO(str(DASHCAM_MODEL_PATH))
            print(f"  Stage 1 dashcam detector loaded  ({DASHCAM_MODEL_PATH.name})")
        else:
            self.detector_dashcam = None
            print("  Stage 1 dashcam detector not found — dashboard will use the default detector")

        self.detector_conf = float(cfg.get("detector_conf", 0.40))

        # Stage 2a
        severity_method = cfg.get("severity_method", "auto")
        self.pothole_analyzer = PotholeAnalyzer(method=severity_method)
        print(f"  Stage 2a PotholeAnalyzer (method={severity_method})")

        # Stage 2b
        self.traffic_analyzer = TrafficLightAnalyzer(
            lane_x_min    = float(cfg.get("lane_x_min",          0.20)),
            lane_x_max    = float(cfg.get("lane_x_max",          0.80)),
            lane_min_size = float(cfg.get("lane_min_size_ratio",  0.001)),
            lane_min_aspect = float(cfg.get("tl_min_aspect_ratio", 1.2)),
        )
        print("  Stage 2b TrafficLightAnalyzer (classical CV)")
        print("-" * 50 + "\n")

    # ── Core inference ──────────────────────────────────

    def _select_detector(self, detector: str):
        """Pick the detector for this request. 'dashcam' uses the dashcam model
        if it has been trained, otherwise falls back to the default detector."""
        if detector == "dashcam" and self.detector_dashcam is not None:
            return self.detector_dashcam
        return self.detector

    def run(self, frame: np.ndarray, detector: str = "default") -> dict:
        """
        Process one BGR frame.

        detector : "default" (photo model) or "dashcam" (video model w/ fallback)

        Returns dict:
            annotated_frame, warnings, potholes, traffic_lights
        """
        model = self._select_detector(detector)
        if model is None:
            return {
                "annotated_frame": frame.copy(),
                "warnings":        ["Detector model not loaded."],
                "potholes":        [],
                "traffic_lights":  [],
            }

        # Stage 1 — detect
        preds = model(frame, conf=self.detector_conf, verbose=False)
        boxes = preds[0].boxes

        potholes       = []
        traffic_lights = []

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                # Guard against zero-area crops
                if x2 <= x1 or y2 <= y1:
                    continue
                crop   = frame[y1:y2, x1:x2]
                coords = [x1, y1, x2, y2]

                if cls == CLS_POTHOLE:
                    analysis = self.pothole_analyzer.analyze(crop, coords, frame.shape)
                    potholes.append({"box": coords, "conf": conf, **analysis})

                elif cls == CLS_TRAFFIC:
                    analysis = self.traffic_analyzer.analyze(crop, coords, frame.shape)
                    traffic_lights.append({"box": coords, "conf": conf, **analysis})

        annotated = draw_detections(frame, potholes, traffic_lights)
        warnings  = generate_warnings(potholes, traffic_lights)

        return {
            "annotated_frame": annotated,
            "warnings":        warnings,
            "potholes":        potholes,
            "traffic_lights":  traffic_lights,
        }

    # ── Image ───────────────────────────────────────────

    def run_image(self, image_path: str) -> dict:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        result   = self.run(frame)
        out_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(out_path, result["annotated_frame"])
        print(f"  Saved → {out_path}")
        for w in result["warnings"]:
            print(f"  {w}")
        return result

    # ── Video ───────────────────────────────────────────

    def run_video(self, video_path: str, output_path: str = None,
                 progress_callback=None, detector: str = "default") -> dict:
        """
        progress_callback(frames_done, frames_total, pct_float) called every 15 frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.run(frame, detector=detector)
            if writer:
                writer.write(result["annotated_frame"])
            frame_count += 1
            if frame_count % 15 == 0:
                pct = (frame_count / total * 100) if total else 0
                if progress_callback:
                    progress_callback(frame_count, total, pct)
                print(f"  {frame_count}/{total} frames ({pct:.0f}%)...")

        cap.release()
        if writer:
            writer.release()
        if progress_callback:
            progress_callback(frame_count, total, 100.0)
        print(f"  Done → {output_path}  ({frame_count} frames)")
        return {"frame_count": frame_count, "total_frames": total, "output": output_path}

    # ── Webcam ──────────────────────────────────────────

    def run_webcam(self, cam_index: int = 0):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam {cam_index}")
        print("  Webcam running — press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.run(frame)
            cv2.imshow("Smart Road Assistant", result["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
