"""
traffic_analyzer.py

Stage 2b: Analyze a cropped traffic light region.
Pure classical CV — no trained model required.

Color detection: spatial thirds (position vote) + HSV thresholding (color vote)
Lane relevance:  x-position ratio + size ratio + aspect ratio heuristics
"""

import cv2
import numpy as np

# ── Lane relevance thresholds (read from pipeline_config.yaml if available) ──
_LANE_X_MIN        = 0.20   # x_center must be > 20% from left
_LANE_X_MAX        = 0.80   # x_center must be < 80% from right
_LANE_MIN_SIZE     = 0.001  # box_area / frame_area > 0.1% (not a tiny distant light)
_LANE_MIN_ASPECT   = 1.2    # height / width > 1.2 (traffic lights are tall)

# ── HSV color ranges ──────────────────────────────────
# Each entry: list of (lower_bound, upper_bound) in HSV
_HSV_RANGES = {
    "Red":    [
        (np.array([0,   80, 80]),  np.array([10,  255, 255])),
        (np.array([160, 80, 80]),  np.array([180, 255, 255])),
    ],
    "Yellow": [
        (np.array([20,  80, 80]),  np.array([35,  255, 255])),
    ],
    "Green":  [
        (np.array([40,  80, 80]),  np.array([80,  255, 255])),
    ],
}

# ── Spatial positions (which third lights which color) ────
# Standard traffic light: top=Red, middle=Yellow, bottom=Green
_SPATIAL_ORDER = ["Red", "Yellow", "Green"]   # index 0=top, 1=mid, 2=bot

# ── Instructions ─────────────────────────────────────────
_INSTRUCTIONS = {
    "Red": {
        True:  "STOP — Red light. Do not proceed.",
        False: "Red light detected in adjacent lane — stay alert.",
    },
    "Yellow": {
        True:  "SLOW DOWN — Yellow light. Prepare to stop.",
        False: "Yellow light in adjacent lane.",
    },
    "Green": {
        True:  "PROCEED — Green light. Check for crossing hazards.",
        False: "Green light in adjacent lane.",
    },
    "Unknown": {
        True:  "Caution — traffic light state unclear.",
        False: "Unrelated or unclear signal detected.",
    },
}


class TrafficLightAnalyzer:
    """
    Analyze a cropped traffic light image.

    Parameters
    ----------
    lane_x_min, lane_x_max   : x-center bounds (fraction of frame width)
    lane_min_size_ratio      : minimum box_area / frame_area
    lane_min_aspect          : minimum height / width of bounding box
    """

    def __init__(
        self,
        lane_x_min:        float = _LANE_X_MIN,
        lane_x_max:        float = _LANE_X_MAX,
        lane_min_size:     float = _LANE_MIN_SIZE,
        lane_min_aspect:   float = _LANE_MIN_ASPECT,
    ):
        self.lane_x_min      = lane_x_min
        self.lane_x_max      = lane_x_max
        self.lane_min_size   = lane_min_size
        self.lane_min_aspect = lane_min_aspect

    # ── Public API ────────────────────────────────────

    def analyze(self, crop: np.ndarray, box: list, frame_shape: tuple) -> dict:
        """
        Parameters
        ----------
        crop        : BGR numpy array of the traffic light crop
        box         : [x1, y1, x2, y2] in original frame pixels
        frame_shape : (height, width, channels) of the full frame

        Returns
        -------
        dict with keys: color, lane_relevant, color_confidence,
                        position_vote, color_vote, instructions
        """
        if crop is None or crop.size == 0:
            return _empty_result()

        color, position_vote, color_vote, confidence = _detect_color(crop)
        lane_relevant = self._check_lane_relevance(box, frame_shape)

        return {
            "color":            color,
            "lane_relevant":    lane_relevant,
            "color_confidence": round(confidence, 3),
            "position_vote":    position_vote,
            "color_vote":       color_vote,
            "instructions":     _INSTRUCTIONS[color][lane_relevant],
        }

    # ── Lane focus: pick THE light governing our lane (cross-box) ─────────────

    def select_lane_relevant(self, lights: list, frame: np.ndarray) -> float:
        """Among all detected traffic lights, mark the single one governing our lane.

        Scores each plausible light by closeness (box size) + centrality to the
        road vanishing point (so it follows curves/turns) + height + confidence,
        then flags the top one as lane_relevant and the rest as not. Updates each
        light's lane_relevant / lane_score / instructions in place.
        Returns the vanishing-point x that was used.
        """
        h, w = frame.shape[:2]
        vp = _estimate_vanishing_x(frame)
        corridor = ((self.lane_x_max - self.lane_x_min) / 2.0) * w   # half-width around VP

        for d in lights:
            d["lane_relevant"] = False
            d["lane_score"]    = 0.0

        # Candidates: real, close-enough signals inside the lane corridor (around VP)
        cands = []
        for d in lights:
            x1, y1, x2, y2 = d["box"]
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            size_ratio = (bw * bh) / max(w * h, 1)
            aspect     = bh / bw
            xc         = (x1 + x2) / 2.0
            if size_ratio < self.lane_min_size or aspect < self.lane_min_aspect:
                continue
            if abs(xc - vp) > corridor:
                continue
            cands.append((d, size_ratio, xc, (y1 + y2) / 2.0))

        if cands:
            max_size = max(c[1] for c in cands)
            for d, size_ratio, xc, yc in cands:
                centrality = 1.0 - min(abs(xc - vp) / (w / 2.0), 1.0)
                d["lane_score"] = round(
                    0.45 * (size_ratio / max_size)   # closeness — the light you're nearing
                    + 0.35 * centrality              # aligned with the road ahead (VP)
                    + 0.05 * (1.0 - yc / h)          # mounted high
                    + 0.15 * d.get("conf", 0.5),     # detector confidence
                    3)
            best = max((c[0] for c in cands), key=lambda d: d["lane_score"])
            best["lane_relevant"] = True

        # keep instructions consistent with the (possibly) updated relevance
        for d in lights:
            color = d.get("color", "Unknown")
            d["instructions"] = _INSTRUCTIONS.get(color, _INSTRUCTIONS["Unknown"])[d["lane_relevant"]]
        return vp

    # ── Lane relevance (per-box geometry gate) ────────────

    def _check_lane_relevance(self, box: list, frame_shape: tuple) -> bool:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2  = box

        x_center   = (x1 + x2) / 2.0
        x_ratio    = x_center / max(frame_w, 1)

        box_area   = max(x2 - x1, 1) * max(y2 - y1, 1)
        frame_area = frame_h * frame_w
        size_ratio = box_area / max(frame_area, 1)

        box_h  = y2 - y1
        box_w  = max(x2 - x1, 1)
        aspect = box_h / box_w

        return (
            self.lane_x_min < x_ratio  < self.lane_x_max
            and size_ratio  > self.lane_min_size
            and aspect      > self.lane_min_aspect
        )


# ── Color detection ───────────────────────────────────

def _detect_color(crop: np.ndarray):
    """
    Returns (color, position_vote, color_vote, confidence).
    confidence: 1.0 = both votes agree, 0.5 = only one vote available.
    """
    # Resize to standard tall traffic-light shape
    resized = cv2.resize(crop, (48, 144))
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # ── Position vote: brightest third ────────────────
    third_h = 48
    thirds_v = [
        hsv[0:third_h,           :, 2],   # top    → Red
        hsv[third_h:2*third_h,   :, 2],   # middle → Yellow
        hsv[2*third_h:3*third_h, :, 2],   # bottom → Green
    ]
    brightness    = [float(np.mean(t)) for t in thirds_v]
    pos_idx       = int(np.argmax(brightness))
    position_vote = _SPATIAL_ORDER[pos_idx]

    # ── Color vote: HSV pixel count in full crop ──────
    hsv_full   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_vote = _hsv_color_vote(hsv_full)

    # ── Combine ───────────────────────────────────────
    if position_vote == color_vote:
        final      = position_vote
        confidence = 1.0
    else:
        # Position vote is more reliable under glare/night
        final      = position_vote
        confidence = 0.5

    # Sanity: if confidence is low and neither is definitive, return Unknown
    max_brightness = max(brightness)
    if max_brightness < 30:
        return "Unknown", position_vote, color_vote, 0.0

    return final, position_vote, color_vote, confidence


def _hsv_color_vote(hsv: np.ndarray) -> str:
    best_color = "Unknown"
    best_count = 0

    for color, ranges in _HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        count = int(np.sum(mask > 0))
        if count > best_count:
            best_count = count
            best_color = color

    return best_color


def _empty_result() -> dict:
    return {
        "color":            "Unknown",
        "lane_relevant":    False,
        "color_confidence": 0.0,
        "position_vote":    "Unknown",
        "color_vote":       "Unknown",
        "instructions":     "Traffic light detected — exercise caution.",
    }
