# ═══════════════════════════════════════════════════════
# src/pipeline.py
# ═══════════════════════════════════════════════════════
"""
Main inference pipeline for Smart Road Assistant.

Phase 1: pothole_model.pt only  → pothole detection
Phase 2: + traffic_model.pt     → traffic light + pothole detection

Auto-detects available models. No code changes needed between phases.

Usage:
  from src.pipeline import Pipeline
  pipe   = Pipeline()
  result = pipe.run(frame)          # BGR numpy array
  result = pipe.run_image("img.jpg")
  result = pipe.run_video("in.mp4", "out.mp4")
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
MODELS_DIR         = BASE_DIR / "models"
POTHOLE_MODEL_PATH = MODELS_DIR / "pothole_model.pt"
TRAFFIC_MODEL_PATH = MODELS_DIR / "traffic_model.pt"

# ── Thresholds ─────────────────────────────────────────
TASK_CONF     = 0.40
NMS_THRESHOLD = 0.45

# ── Class maps ─────────────────────────────────────────
TL_COLORS = {0: "Green", 1: "Red", 2: "Yellow"}
TL_DRAW   = {
    "Green" : (0, 200, 0),
    "Red"   : (0, 0, 220),
    "Yellow": (0, 200, 220),
}
POTHOLE_DRAW = (0, 140, 255)  # orange

# ── Severity thresholds (pixel area at 640×640) ────────
SEV_SMALL  = 1500
SEV_MEDIUM = 5000


# ══════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════

def detect_weather(frame):
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val  = np.std(gray)
    if mean_val < 60:
        return "night"
    if std_val < 25 and mean_val > 150:
        return "fog"
    if std_val < 40:
        return "rain"
    return "clear"


def preprocess_common(frame):
    """Resize → CLAHE → blur → weather correction."""
    frame = cv2.resize(frame, (640, 640))

    # CLAHE on L channel
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    frame   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Noise reduction
    frame   = cv2.GaussianBlur(frame, (3, 3), 0)

    # Weather correction
    weather = detect_weather(frame)
    if weather == "fog":
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=-30)
    elif weather == "night":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 2.0, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif weather == "rain":
        frame = cv2.medianBlur(frame, 3)

    return frame, weather


def preprocess_traffic(frame):
    """Boost saturation to make traffic light colors more vivid."""
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(int) + 30, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def preprocess_pothole(frame):
    """Sharpen edges to improve pothole boundary detection."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ══════════════════════════════════════════════════════
# POST PROCESSING
# ══════════════════════════════════════════════════════

def get_severity(box):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area < SEV_SMALL:
        return "Low"
    elif area < SEV_MEDIUM:
        return "Medium"
    return "High"


def get_distance(box, frame_h=640):
    _, _, _, y2 = box
    ratio = y2 / frame_h
    if ratio > 0.80:
        return "Very Close"
    elif ratio > 0.60:
        return "Close"
    elif ratio > 0.40:
        return "Medium"
    return "Far"


def apply_nms(detections, iou_threshold=NMS_THRESHOLD):
    if not detections:
        return []
    boxes   = np.array([d["box"] for d in detections], dtype=np.float32)
    scores  = np.array([d["conf"] for d in detections], dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), TASK_CONF, iou_threshold
    )
    if len(indices) == 0:
        return []
    return [detections[i] for i in indices.flatten()]


# ══════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════

def draw_detections(frame, tl_dets, ph_dets, weather):
    out = frame.copy()

    # Weather overlay (top-left)
    weather_color = {
        "clear": (0, 200, 0),
        "rain" : (200, 200, 0),
        "fog"  : (180, 180, 180),
        "night": (100, 100, 255),
    }
    cv2.putText(out, f"Weather: {weather}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                weather_color.get(weather, (255, 255, 255)), 2)

    # Traffic light boxes
    for d in tl_dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = TL_DRAW.get(d["color"], (255, 255, 255))
        label = f"TL:{d['color']} {d['conf']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, y1 - 24), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Pothole boxes
    for d in ph_dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        label = f"Pothole|{d['severity']}|{d['distance']} {d['conf']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), POTHOLE_DRAW, 2)
        cv2.rectangle(out, (x1, y1 - 24), (x1 + len(label) * 9, y1), POTHOLE_DRAW, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Detection count (top-right)
    count_text = f"TL:{len(tl_dets)}  PH:{len(ph_dets)}"
    cv2.putText(out, count_text,
                (out.shape[1] - 160, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    return out


def generate_warnings(tl_dets, ph_dets):
    warnings = []
    for d in tl_dets:
        if d["color"] == "Red":
            warnings.append("🔴 STOP — Red traffic light detected!")
        elif d["color"] == "Yellow":
            warnings.append("🟡 SLOW DOWN — Yellow light ahead.")
        elif d["color"] == "Green":
            warnings.append("🟢 Green light — Safe to proceed.")
    for d in ph_dets:
        warnings.append(
            f"🟠 POTHOLE AHEAD — {d['severity']} severity, {d['distance']}!"
        )
    return warnings


# ══════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════

class Pipeline:
    """
    Smart Road Assistant inference pipeline.
    Auto-detects available models.

    Phase 1 (pothole_model.pt only): pothole detection
    Phase 2 (+ traffic_model.pt)   : full detection
    """

    def __init__(self):
        print("\nLoading Smart Road Assistant pipeline...")
        print("-" * 45)

        self.traffic_model = None
        self.pothole_model = None

        if TRAFFIC_MODEL_PATH.exists():
            self.traffic_model = YOLO(str(TRAFFIC_MODEL_PATH))
            print(f"  ✅ Traffic model  → {TRAFFIC_MODEL_PATH.name}")
        else:
            print(f"  ⬜ Traffic model  → not found (Phase 2)")

        if POTHOLE_MODEL_PATH.exists():
            self.pothole_model = YOLO(str(POTHOLE_MODEL_PATH))
            print(f"  ✅ Pothole model  → {POTHOLE_MODEL_PATH.name}")
        else:
            print(f"  ❌ Pothole model  → not found")
            print(f"     Run: python src/train.py --model pothole")

        phase = 2 if (self.traffic_model and self.pothole_model) else 1
        print(f"\n  Running in Phase {phase} mode")
        print("-" * 45 + "\n")

    # ── Core inference ──────────────────────────────────
    def run(self, frame):
        """
        Process one BGR frame.
        Returns dict: annotated_frame, warnings, traffic_lights,
                      potholes, weather
        """
        original     = frame.copy()
        h, w         = original.shape[:2]
        processed, weather = preprocess_common(frame)

        tl_dets = []
        ph_dets = []

        # ── Traffic light detection ─────────────────────
        if self.traffic_model is not None:
            tl_input = preprocess_traffic(processed)
            tl_preds = self.traffic_model(tl_input, conf=TASK_CONF, verbose=False)
            tl_boxes = tl_preds[0].boxes
            if tl_boxes is not None:
                sx = w / 640
                sy = h / 640
                for box in tl_boxes:
                    coords = box.xyxy[0].tolist()
                    cls    = int(box.cls[0])
                    conf   = float(box.conf[0])
                    color  = TL_COLORS.get(cls, "Unknown")
                    scaled = [
                        coords[0]*sx, coords[1]*sy,
                        coords[2]*sx, coords[3]*sy
                    ]
                    tl_dets.append({
                        "box"  : scaled,
                        "cls"  : cls,
                        "color": color,
                        "conf" : conf,
                    })

        # ── Pothole detection ───────────────────────────
        if self.pothole_model is not None:
            ph_input = preprocess_pothole(processed)
            ph_preds = self.pothole_model(ph_input, conf=TASK_CONF, verbose=False)
            ph_boxes = ph_preds[0].boxes
            if ph_boxes is not None:
                sx = w / 640
                sy = h / 640
                for box in ph_boxes:
                    coords   = box.xyxy[0].tolist()
                    conf     = float(box.conf[0])
                    scaled   = [
                        coords[0]*sx, coords[1]*sy,
                        coords[2]*sx, coords[3]*sy
                    ]
                    severity = get_severity(coords)
                    distance = get_distance(coords)
                    ph_dets.append({
                        "box"     : scaled,
                        "conf"    : conf,
                        "severity": severity,
                        "distance": distance,
                    })

        # ── NMS + output ────────────────────────────────
        tl_dets   = apply_nms(tl_dets)
        ph_dets   = apply_nms(ph_dets)
        annotated = draw_detections(original, tl_dets, ph_dets, weather)
        warnings  = generate_warnings(tl_dets, ph_dets)

        return {
            "annotated_frame": annotated,
            "warnings"       : warnings,
            "traffic_lights" : tl_dets,
            "potholes"       : ph_dets,
            "weather"        : weather,
        }

    def run_image(self, image_path: str):
        """Run pipeline on a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        result = self.run(frame)

        # Save annotated image next to input
        out_path = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(out_path, result["annotated_frame"])
        print(f"  Saved → {out_path}")
        for w in result["warnings"]:
            print(f"  {w}")
        return result

    def run_video(self, video_path: str, output_path: str = None):
        """Process a video file frame by frame."""
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
            result = self.run(frame)
            if writer:
                writer.write(result["annotated_frame"])
            frame_count += 1
            if frame_count % 30 == 0:
                pct = (frame_count / total * 100) if total else 0
                print(f"  Processed {frame_count}/{total} frames ({pct:.0f}%)...")

        cap.release()
        if writer:
            writer.release()
        print(f"  ✅ Done → {output_path}  ({frame_count} frames)")

    def run_webcam(self, cam_index: int = 0):
        """Live webcam inference. Press Q to quit."""
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam index {cam_index}")

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