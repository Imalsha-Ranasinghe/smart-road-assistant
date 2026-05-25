"""
pothole_analyzer.py

Stage 2a: Deeply analyze a cropped pothole region.

Two methods are available and compared via benchmark_severity.py:
  - "cv"  : Classical CV (edge density, depth score, texture score)
  - "cnn" : Fine-tuned YOLOv8n-cls severity classifier
  - "auto": Use CNN if model exists, otherwise fall back to CV
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

BASE_DIR           = Path(__file__).resolve().parent.parent.parent
SEVERITY_MODEL_PATH = BASE_DIR / "models" / "severity_model.pt"

# ── Instruction matrix ─────────────────────────────────
_INSTRUCTIONS = {
    "High": {
        "Very Close": "BRAKE NOW! Severe pothole directly ahead.",
        "Close":      "Reduce speed immediately. Severe pothole ahead.",
        "Medium":     "Slow down. Large pothole in your path.",
        "Far":        "Hazard ahead — prepare to slow down.",
    },
    "Medium": {
        "Very Close": "Avoid this lane if safe. Moderate pothole.",
        "Close":      "Slow down. Moderate pothole ahead.",
        "Medium":     "Caution: moderate pothole ahead.",
        "Far":        "Monitor road — pothole detected in distance.",
    },
    "Low": {
        "Very Close": "Minor pothole. Proceed carefully.",
        "Close":      "Minor pothole ahead.",
        "Medium":     "Minor road damage detected.",
        "Far":        "Minor road damage far ahead.",
    },
}

# ── CV scoring thresholds ──────────────────────────────
_CV_HIGH   = 0.18
_CV_MEDIUM = 0.10

# ── CNN class order returned by YOLOv8-cls ─────────────
# YOLOv8-cls reads classes alphabetically from folder names:
# High(0), Low(1), Medium(2)  →  remap to readable labels
_CNN_IDX_TO_LABEL = {0: "High", 1: "Low", 2: "Medium"}


class PotholeAnalyzer:
    """
    Analyze a cropped pothole image for severity, distance, and instructions.

    Parameters
    ----------
    method : str
        "cv"   — classical CV only (always available)
        "cnn"  — fine-tuned YOLOv8n-cls (requires severity_model.pt)
        "auto" — CNN if model exists, else CV  (default)
    """

    def __init__(self, method: str = "auto"):
        self.method = method
        self.model  = None

        if method in ("cnn", "auto") and _YOLO_AVAILABLE and SEVERITY_MODEL_PATH.exists():
            self.model = YOLO(str(SEVERITY_MODEL_PATH))
            print(f"  Severity model loaded ({SEVERITY_MODEL_PATH.name})")
        elif method == "cnn" and not SEVERITY_MODEL_PATH.exists():
            print("  WARNING: severity_model.pt not found — falling back to CV method.")

    # ── Public API ─────────────────────────────────────

    def analyze(self, crop: np.ndarray, box: list, frame_shape: tuple) -> dict:
        """
        Parameters
        ----------
        crop        : BGR numpy array of the pothole region
        box         : [x1, y1, x2, y2] in original frame pixel coordinates
        frame_shape : (height, width, channels) of the full frame

        Returns
        -------
        dict with keys: severity, distance, instructions, area_px, cv_scores
        """
        if crop is None or crop.size == 0:
            return _empty_result()

        cv_scores = _compute_cv_scores(crop)
        severity  = self._classify(crop, cv_scores)
        distance  = _estimate_distance(box, frame_shape)

        return {
            "severity":     severity,
            "distance":     distance,
            "instructions": _INSTRUCTIONS[severity][distance],
            "area_px":      (box[2] - box[0]) * (box[3] - box[1]),
            "cv_scores":    cv_scores,
        }

    # ── Internal ───────────────────────────────────────

    def _classify(self, crop: np.ndarray, cv_scores: dict) -> str:
        use_cnn = self.model is not None and self.method in ("cnn", "auto")
        if use_cnn:
            return _cnn_severity(self.model, crop)
        return _cv_severity(cv_scores)


# ── CV helpers ─────────────────────────────────────────

def _compute_cv_scores(crop: np.ndarray) -> dict:
    """Compute edge_density, depth_score, texture_score for a BGR crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Edge density — Canny edges / total pixels
    edges        = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)

    # Depth score — darker center vs brighter border suggests pit depth
    ch, cw   = h // 4, w // 4
    center   = gray[ch: h - ch, cw: w - cw]
    border   = np.concatenate([
        gray[:ch, :].ravel(), gray[h - ch:, :].ravel(),
        gray[:, :cw].ravel(), gray[:, w - cw:].ravel(),
    ])
    mean_c   = float(np.mean(center)) if center.size else 128.0
    mean_b   = float(np.mean(border)) if border.size else 128.0
    depth_score = max(0.0, 1.0 - (mean_c / max(mean_b, 1.0)))

    # Texture score — normalised std dev of intensities
    texture_score = float(np.std(gray)) / 255.0

    combined = (
        0.40 * edge_density
        + 0.35 * depth_score
        + 0.25 * texture_score
    )

    return {
        "edge_density":  round(edge_density,  4),
        "depth_score":   round(depth_score,   4),
        "texture_score": round(texture_score, 4),
        "combined":      round(combined,      4),
    }


def _cv_severity(cv_scores: dict) -> str:
    score = cv_scores["combined"]
    if score >= _CV_HIGH:
        return "High"
    if score >= _CV_MEDIUM:
        return "Medium"
    return "Low"


# ── CNN helper ─────────────────────────────────────────

def _cnn_severity(model, crop: np.ndarray) -> str:
    results = model(crop, imgsz=128, verbose=False)
    top1_idx = int(results[0].probs.top1)
    return _CNN_IDX_TO_LABEL.get(top1_idx, "Low")


# ── Distance ───────────────────────────────────────────

def _estimate_distance(box: list, frame_shape: tuple) -> str:
    frame_h = frame_shape[0]
    y2      = box[3]
    ratio   = y2 / max(frame_h, 1)
    if ratio > 0.80:
        return "Very Close"
    if ratio > 0.60:
        return "Close"
    if ratio > 0.40:
        return "Medium"
    return "Far"


def _empty_result() -> dict:
    return {
        "severity":     "Low",
        "distance":     "Far",
        "instructions": "Pothole detected — exercise caution.",
        "area_px":      0,
        "cv_scores":    {"edge_density": 0, "depth_score": 0,
                         "texture_score": 0, "combined": 0},
    }
