"""
pipeline.py
Main inference pipeline.

PHASE 1: Only pothole_model.pt exists → detects potholes only.
PHASE 2: All 3 models exist          → detects traffic lights + potholes.

No code changes needed between phases.
Pipeline auto-detects which models are available in models/ folder.

Place this file at: src/pipeline.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Model paths ────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
MODELS_DIR         = BASE_DIR / "models"

POTHOLE_MODEL_PATH = MODELS_DIR / "pothole_model.pt"
TRAFFIC_MODEL_PATH = MODELS_DIR / "traffic_model.pt"
GATE_MODEL_PATH    = MODELS_DIR / "gate_model.pt"

# ── Detection thresholds ───────────────────────────────────────────────────────
GATE_CONF = 0.35
TASK_CONF = 0.40

# ── Class maps ─────────────────────────────────────────────────────────────────
GATE_TL      = 0       # gate model: traffic_light class
GATE_POTHOLE = 1       # gate model: pothole class

TL_COLORS = {0: "Green", 1: "Red", 2: "Yellow"}
TL_DRAW   = {
    "Green" : (0, 200, 0),
    "Red"   : (0, 0, 220),
    "Yellow": (0, 200, 220),
}
POTHOLE_DRAW = (0, 140, 255)   # orange

# ── Pothole severity thresholds (pixel area at 640×640) ───────────────────────
SEV_SMALL  = 1500
SEV_MEDIUM = 5000

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def detect_weather(frame):
    """Rule-based weather detection from image statistics."""
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
    """
    Common preprocessing applied to every frame before detection.
    Steps: resize → CLAHE → Gaussian blur → weather correction
    """
    # 1. Resize to YOLO input size
    frame = cv2.resize(frame, (640, 640))

    # 2. CLAHE contrast enhancement on L channel
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    frame   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Gaussian blur for noise reduction
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # 4. Weather-specific correction
    weather = detect_weather(frame)
    if weather == "fog":
        # Increase contrast to cut through haze
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=-30)
    elif weather == "night":
        # Boost brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 2.0, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif weather == "rain":
        # Median filter to reduce rain streaks
        frame = cv2.medianBlur(frame, 3)

    return frame, weather


def preprocess_traffic(frame):
    """
    Traffic light specific preprocessing.
    Converts to HSV and boosts saturation to make colors more distinct.
    """
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(int) + 30, 0, 255).astype(np.uint8)
    return cv2.merge([h, s, v])


def preprocess_pothole(frame):
    """
    Pothole specific preprocessing.
    Converts to grayscale and sharpens edges for better boundary detection.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Unsharp masking to enhance edges
    sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    # Convert back to BGR so YOLO can process it
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# POST PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def get_severity(box):
    """Calculate pothole severity from bounding box pixel area."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area < SEV_SMALL:
        return "Low"
    elif area < SEV_MEDIUM:
        return "Medium"
    return "High"


def get_distance(box, frame_h=640):
    """
    Estimate relative distance using vertical position.
    Lower in frame = closer to vehicle.
    """
    _, _, _, y2 = box
    ratio = y2 / frame_h
    if ratio > 0.80:
        return "Very Close"
    elif ratio > 0.60:
        return "Close"
    elif ratio > 0.40:
        return "Medium"
    return "Far"


def apply_nms(detections, iou_threshold=0.45):
    """Remove overlapping duplicate detections."""
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


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def draw_detections(frame, tl_dets, ph_dets):
    """Draw bounding boxes and labels on the original frame."""
    out = frame.copy()

    for d in tl_dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = TL_DRAW.get(d["color"], (255, 255, 255))
        label = f"TL:{d['color']} {d['conf']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, y1 - 22), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    for d in ph_dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        label = f"Pothole {d['severity']} {d['distance']} {d['conf']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), POTHOLE_DRAW, 2)
        cv2.rectangle(out, (x1, y1 - 22), (x1 + len(label) * 9, y1), POTHOLE_DRAW, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


def generate_warnings(tl_dets, ph_dets):
    """Generate text warning messages from detections."""
    warnings = []
    for d in tl_dets:
        if d["color"] == "Red":
            warnings.append("STOP — Red traffic light detected!")
        elif d["color"] == "Yellow":
            warnings.append("SLOW DOWN — Yellow traffic light ahead.")
        elif d["color"] == "Green":
            warnings.append("Green light — Safe to go.")
    for d in ph_dets:
        warnings.append(
            f"POTHOLE AHEAD — {d['severity']} severity, {d['distance']}!"
        )
    return warnings


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Main inference pipeline.
    Automatically uses whatever models are available in models/ folder.
    """

    def __init__(self):
        print("Loading available models...")

        # Gate model (needed only when both other models exist)
        self.gate_model = None
        if GATE_MODEL_PATH.exists():
            self.gate_model = YOLO(str(GATE_MODEL_PATH))
            print(f"  ✅ Gate model     → {GATE_MODEL_PATH.name}")
        else:
            print(f"  ⬜ Gate model     → not found (will activate all available pipelines)")

        # Traffic light model
        self.traffic_model = None
        if TRAFFIC_MODEL_PATH.exists():
            self.traffic_model = YOLO(str(TRAFFIC_MODEL_PATH))
            print(f"  ✅ Traffic model  → {TRAFFIC_MODEL_PATH.name}")
        else:
            print(f"  ⬜ Traffic model  → not found (Phase 2)")

        # Pothole model
        self.pothole_model = None
        if POTHOLE_MODEL_PATH.exists():
            self.pothole_model = YOLO(str(POTHOLE_MODEL_PATH))
            print(f"  ✅ Pothole model  → {POTHOLE_MODEL_PATH.name}")
        else:
            print(f"  ❌ Pothole model  → not found. Run: python src/train.py --model pothole")

        print()

    def run(self, frame):
        """
        Process one BGR frame through the full pipeline.
        Returns dict with annotated frame, warnings, and detection details.
        """
        original = frame.copy()
        h, w     = original.shape[:2]

        # ── Stage 2: Common preprocessing ──────────────────────────────────
        processed, weather = preprocess_common(frame)

        # ── Stage 3: Gate model or default activation ───────────────────────
        if self.gate_model is not None:
            gate_res   = self.gate_model(processed, conf=GATE_CONF, verbose=False)
            gate_boxes = gate_res[0].boxes
            activate_tl = False
            activate_ph = False
            if gate_boxes is not None and len(gate_boxes):
                for box in gate_boxes:
                    cls = int(box.cls[0])
                    if cls == GATE_TL:
                        activate_tl = True
                    elif cls == GATE_POTHOLE:
                        activate_ph = True
        else:
            # No gate model — activate whichever pipeline has a trained model
            activate_tl = self.traffic_model is not None
            activate_ph = self.pothole_model is not None

        tl_dets = []
        ph_dets = []

        # ── Stage 4A: Traffic light pipeline ────────────────────────────────
        if activate_tl and self.traffic_model is not None:
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
                    scaled = [coords[0]*sx, coords[1]*sy, coords[2]*sx, coords[3]*sy]
                    tl_dets.append({
                        "box"  : scaled,
                        "cls"  : cls,
                        "color": color,
                        "conf" : conf,
                    })

        # ── Stage 4B: Pothole pipeline ───────────────────────────────────────
        if activate_ph and self.pothole_model is not None:
            ph_input = preprocess_pothole(processed)
            ph_preds = self.pothole_model(ph_input, conf=TASK_CONF, verbose=False)
            ph_boxes = ph_preds[0].boxes
            if ph_boxes is not None:
                sx = w / 640
                sy = h / 640
                for box in ph_boxes:
                    coords   = box.xyxy[0].tolist()
                    conf     = float(box.conf[0])
                    scaled   = [coords[0]*sx, coords[1]*sy, coords[2]*sx, coords[3]*sy]
                    severity = get_severity(coords)
                    distance = get_distance(coords)
                    ph_dets.append({
                        "box"     : scaled,
                        "conf"    : conf,
                        "severity": severity,
                        "distance": distance,
                    })

        # ── Stage 5: Post processing ─────────────────────────────────────────
        tl_dets = apply_nms(tl_dets)
        ph_dets = apply_nms(ph_dets)

        # ── Stage 6 & 7: Warnings and annotation ────────────────────────────
        annotated = draw_detections(original, tl_dets, ph_dets)
        warnings  = generate_warnings(tl_dets, ph_dets)

        return {
            "annotated_frame": annotated,
            "warnings"       : warnings,
            "traffic_lights" : tl_dets,
            "potholes"       : ph_dets,
            "weather"        : weather,
        }

    def run_image(self, image_path: str):
        """Load image from path and run pipeline."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.run(frame)

    def run_video(self, video_path: str, output_path: str = None):
        """Process a video file frame by frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
                print(f"  Processed {frame_count} frames...")

        cap.release()
        if writer:
            writer.release()
        print(f"  ✅ Video saved → {output_path} ({frame_count} frames)")