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

# ── False-positive rejection (a real active light is high, tall, luminous) ──
_TL_FP_MAX_YC      = 0.62   # box center must sit in the top 62% of the frame
_TL_FP_MIN_CONF    = 0.25   # ignore very weak detections
_TL_FP_MIN_ASPECT  = 0.80   # reject very wide boxes (signs / light bars)

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

# Minimum fraction of the crop that must be a saturated R/Y/G colour for it to
# count as a lit lamp (else it's a dark box / sign / clutter → Unknown).
_MIN_COLOR_RATIO = 0.02

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

    # ── Lane focus: pick THE light nearest + relevant to our lane ─────────────

    def select_lane_relevant(self, lights: list, frame: np.ndarray,
                             vp: float = None) -> float:
        """Reject false positives, then flag the SINGLE light governing our lane.

        1. Drop implausible detections (weak / wide / no colour / mounted low).
        2. Prefer lights inside the lane corridor around the road vanishing point;
           if none qualify (e.g. a poor lane estimate), fall back to ALL plausible
           lights so the one real signal is never hidden.
        3. Score by nearness (box size), centrality to the road ahead, mounting
           height and detector confidence; flag the top one lane_relevant=True.

        Returns the vanishing-point x.
        """
        h, w = frame.shape[:2]
        if vp is None:
            vp = _estimate_vanishing_x(frame)
        corridor = ((self.lane_x_max - self.lane_x_min) / 2.0) * w   # half-width around VP

        for d in lights:
            d["lane_relevant"] = False
            d["lane_score"]    = 0.0

        # 1) Reject obvious false positives up front
        plausible = [d for d in lights if self._plausible_light(d, frame.shape)]
        if not plausible:
            return vp

        def _size(d):
            x1, y1, x2, y2 = d["box"]
            return max(x2 - x1, 1) * max(y2 - y1, 1)

        # 2) Prefer in-corridor lights; fall back to all plausible if the lane
        #    estimate excluded everything (never hide the only real signal).
        in_corridor = [d for d in plausible
                       if abs(((d["box"][0] + d["box"][2]) / 2.0) - vp) <= corridor]
        cands = in_corridor or plausible

        # 3) Score and pick the single most relevant light
        max_size = max(_size(d) for d in cands)
        for d in cands:
            x1, y1, x2, y2 = d["box"]
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            centrality = 1.0 - min(abs(xc - vp) / (w / 2.0), 1.0)
            d["lane_score"] = round(
                0.50 * (_size(d) / max_size)     # nearest = the light you're approaching
                + 0.30 * centrality              # aligned with the road ahead (VP)
                + 0.05 * (1.0 - yc / h)          # mounted high
                + 0.15 * d.get("conf", 0.5),     # detector confidence
                3)
        best = max(cands, key=lambda d: d["lane_score"])
        best["lane_relevant"] = True

        # keep instructions consistent with the updated relevance
        for d in lights:
            color = d.get("color", "Unknown")
            d["instructions"] = _INSTRUCTIONS.get(color, _INSTRUCTIONS["Unknown"])[d["lane_relevant"]]
        return vp

    # ── False-positive rejection ──────────────────────────
    def _plausible_light(self, d: dict, frame_shape: tuple) -> bool:
        """A real active traffic light is mounted HIGH, is TALLER than wide, and
        shows a luminous R/Y/G colour. This rejects the common false positives:
        tail lights / reflections (low in the frame), road signs and light bars
        (too wide), and background clutter (no colour)."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = d["box"]
        yc     = (y1 + y2) / 2.0
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        aspect = bh / bw
        if d.get("conf", 0.0) < _TL_FP_MIN_CONF:       return False   # too weak
        if yc / max(h, 1) > _TL_FP_MAX_YC:             return False   # mounted too low
        if aspect < _TL_FP_MIN_ASPECT:                 return False   # too wide → sign/bar
        if d.get("color", "Unknown") == "Unknown":     return False   # no live colour
        return True

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


# ── Road / ego-lane detection (robust classical CV) ───
# Look ONLY inside a road-ahead trapezoid (excludes buildings/sky), keep
# white/yellow markings + road-boundary edges, fit a left & right edge, then
# VALIDATE the geometry. If it isn't a sane lane (curve, junction, clutter,
# fisheye) we return None and the caller draws a clean forward corridor — so we
# never render "rubbish".

LANE_HORIZON   = 0.58          # ignore everything above this (sky/buildings/signs)
LANE_HOOD      = 0.93          # ignore the car hood below this
TRAP_TOP_HALF  = 0.13          # road-ahead trapezoid half-width at the horizon (frac W)
TRAP_BOT_HALF  = 0.46          # ... at the bottom
LANE_SLOPE_MIN = 0.35          # reject near-horizontal clutter (roofs, wires, cross-streets)
LANE_SLOPE_MAX = 2.5           # reject near-vertical edges (poles, building corners)
VP_X_BAND      = (0.32, 0.68)  # a valid vanishing point sits near frame centre ...
VP_Y_BAND      = (0.42, 0.66)  # ... and near the horizon band
LANE_W_BAND    = (0.18, 0.95)  # plausible lane width at the bottom (frac W)


def _fit_line(pts):
    """Least-squares fit x = m*y + b to a group of (x, y) points (lane edges are
    steep, so x-as-function-of-y is stable). Returns (m, b) or None."""
    if len(pts) < 4:
        return None
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    if ys.max() - ys.min() < 8:        # too little vertical spread → unstable
        return None
    m, b = np.polyfit(ys, xs, 1)
    return float(m), float(b)


def _lane_mask(frame):
    """White+yellow markings OR road-boundary edges, kept ONLY inside a road-ahead
    trapezoid. The trapezoid is what excludes buildings/sidewalks at the frame
    sides — the main source of false 'lane' edges."""
    h, w = frame.shape[:2]
    top_y, bot_y = int(h * LANE_HORIZON), int(h * LANE_HOOD)
    cx = w / 2.0
    trap = np.array([[
        (int(cx - TRAP_TOP_HALF * w), top_y), (int(cx + TRAP_TOP_HALF * w), top_y),
        (int(cx + TRAP_BOT_HALF * w), bot_y), (int(cx - TRAP_BOT_HALF * w), bot_y),
    ]], dtype=np.int32)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi, trap, 255)

    hls    = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls, np.array([0, 165, 0]),  np.array([255, 255, 70]))
    yellow = cv2.inRange(hls, np.array([15, 60, 60]), np.array([40, 255, 255]))
    edges  = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 150)

    mask = cv2.bitwise_or(cv2.bitwise_or(white, yellow), edges)
    return cv2.bitwise_and(mask, roi)


def _ego_curve(cands, side, centre, w, h):
    """Fit the ego-lane edge nearest the vehicle as a polynomial x = f(y).
    Picks the edge closest to the camera centre (the ego boundary), then fits a
    QUADRATIC when the points span enough of the road (→ traces curves), else a
    straight line. Returns an np.poly1d (x as a function of y) or None."""
    tol = 0.06 * w
    if side == "left":
        pool = [c for c in cands if c[0] < centre + tol]
        ref  = max((c[0] for c in pool), default=None)
    else:
        pool = [c for c in cands if c[0] > centre - tol]
        ref  = min((c[0] for c in pool), default=None)
    if ref is None:
        return None
    pts = []
    for xb, p1, p2 in pool:
        if abs(xb - ref) <= 0.12 * w:        # wide band so a curve's points are kept
            pts += [p1, p2]
    if len(pts) < 4:
        return None
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    if ys.max() - ys.min() < 0.06 * h:       # too little vertical spread → unstable
        return None
    deg = 2 if (len(pts) >= 8 and ys.max() - ys.min() > 0.14 * h) else 1
    return np.poly1d(np.polyfit(ys, xs, deg))


def _validated_lane(L, R, h, w):
    """Evaluate the two edge polynomials, then VALIDATE the geometry: edges
    straddle the camera, plausible lane width, the lane narrows with distance
    (perspective), and the curve stays in frame (rejects wild fits). Returns
    (lane_lines, vp_x, vp_y); lane_lines is None when it isn't a confident, sane
    lane. lane_lines carries a straight chord ('left'/'right') for geometry plus a
    polyline ('left_poly'/'right_poly') for drawing the actual curve."""
    horizon = h * LANE_HORIZON
    if L is None or R is None:
        return None, w / 2.0, horizon

    base_y = h - 1
    lx_b, rx_b = float(L(base_y)), float(R(base_y))      # bottom of frame
    lx_t, rx_t = float(L(horizon)), float(R(horizon))    # at the horizon

    if not (lx_b < w / 2.0 < rx_b):                                   return None, w / 2.0, horizon
    bottom_w = rx_b - lx_b
    if not (LANE_W_BAND[0] * w <= bottom_w <= LANE_W_BAND[1] * w):    return None, w / 2.0, horizon
    if (rx_t - lx_t) > bottom_w * 1.15:    # lane must narrow with distance, not diverge
        return None, w / 2.0, horizon

    # Vanishing point = where the two edge chords actually meet (so the drawn edges
    # CONVERGE to a single point, instead of ending apart at the horizon line).
    m_l = (lx_t - lx_b) / (horizon - base_y)
    m_r = (rx_t - rx_b) / (horizon - base_y)
    if abs(m_l - m_r) < 1e-6:
        vp_y = horizon - 0.12 * h                         # near-parallel → apex above horizon
    else:
        vp_y = base_y + (rx_b - lx_b) / (m_l - m_r)
    vp_y = max(0.30 * h, min(vp_y, horizon))
    vp_x = lx_b + m_l * (vp_y - base_y)
    if not (VP_X_BAND[0] * w <= vp_x <= VP_X_BAND[1] * w):            return None, vp_x, vp_y

    # Curved edges (near field) that CONVERGE to the vanishing point (far field).
    apex = (int(round(vp_x)), int(round(vp_y)))
    ys = np.linspace(base_y, horizon, 14)
    left_poly  = [(int(L(y)), int(y)) for y in ys] + [apex]
    right_poly = [(int(R(y)), int(y)) for y in ys] + [apex]
    if any(not (-0.15 * w <= x <= 1.15 * w) for x, _ in left_poly + right_poly):
        return None, vp_x, vp_y

    lane = {
        "left":  [int(lx_b), base_y, apex[0], apex[1]],   # straight chord → VP (geometry)
        "right": [int(rx_b), base_y, apex[0], apex[1]],
        "left_poly":  left_poly,                          # curve → VP (drawing)
        "right_poly": right_poly,
    }
    return lane, float(vp_x), float(vp_y)


def default_roi_lane(frame_shape):
    """Fallback when no lane lines exist: a fixed central ROI corridor — the
    assumed straight-ahead ego lane. Returns (vp_x, lane_lines)."""
    h, w = frame_shape[:2]
    base_y = h - 1
    top_y  = int(h * LANE_HORIZON)
    apex_x = w // 2
    return float(apex_x), {
        "left":  [int(w * 0.38), base_y, apex_x, top_y],
        "right": [int(w * 0.62), base_y, apex_x, top_y],
    }


def detect_lane(frame: np.ndarray, default_ratio: float = 0.5):
    """Robust ego-lane detection (pure CV).

    Masks white/yellow markings + road-boundary edges inside a road-ahead
    trapezoid, fits the nearest left & right edges, and VALIDATES the geometry.
    Returns (vp_x, vp_y, lane_lines, segments). lane_lines is None whenever the
    result isn't a confident, sane lane — the caller then draws a clean forward
    corridor instead of garbage.
    """
    h, w = frame.shape[:2]
    cx, horizon = w * default_ratio, h * LANE_HORIZON
    if h < 40 or w < 40:
        return cx, horizon, None, []

    mask  = _lane_mask(frame)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40,
                            minLineLength=max(int(h * 0.10), 25), maxLineGap=60)
    if lines is None:
        return cx, horizon, None, []

    base_y = h - 1
    centre = w / 2.0
    left, right, segs = [], [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (LANE_SLOPE_MIN <= abs(slope) <= LANE_SLOPE_MAX):
            continue
        segs.append((int(x1), int(y1), int(x2), int(y2)))
        x_bottom = x1 + (base_y - y1) / slope
        mid_x    = (x1 + x2) / 2.0
        if slope < 0 and mid_x < centre + 0.10 * w:
            left.append((x_bottom, (x1, y1), (x2, y2)))
        elif slope > 0 and mid_x > centre - 0.10 * w:
            right.append((x_bottom, (x1, y1), (x2, y2)))

    L = _ego_curve(left,  "left",  centre, w, h)
    R = _ego_curve(right, "right", centre, w, h)
    lane, vp_x, vp_y = _validated_lane(L, R, h, w)
    if lane is None:
        return cx, horizon, None, segs          # not confident → caller uses fallback
    return vp_x, vp_y, lane, segs


def _estimate_vanishing_x(frame: np.ndarray, default_ratio: float = 0.5) -> float:
    return detect_lane(frame, default_ratio)[0]


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

    # ── Color vote: saturated R/Y/G pixel ratio over the full crop ──
    hsv_full                = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_vote, color_ratio = _hsv_color_vote(hsv_full)

    # A live traffic light shows a SATURATED coloured lamp. If the crop has
    # almost no saturated R/Y/G pixels, it is NOT a lit light — reject it as
    # Unknown. This kills the "black/grey rectangle called Red" false positive,
    # where the brightness-based position vote alone would have guessed a colour.
    if color_ratio < _MIN_COLOR_RATIO:
        return "Unknown", position_vote, color_vote, 0.0

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


def _hsv_color_vote(hsv: np.ndarray):
    """Return (best_color, ratio), ratio = saturated pixels of that colour / total."""
    best_color = "Unknown"
    best_count = 0
    total      = max(hsv.shape[0] * hsv.shape[1], 1)

    for color, ranges in _HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        count = int(np.sum(mask > 0))
        if count > best_count:
            best_count = count
            best_color = color

    return best_color, best_count / total


def _empty_result() -> dict:
    return {
        "color":            "Unknown",
        "lane_relevant":    False,
        "color_confidence": 0.0,
        "position_vote":    "Unknown",
        "color_vote":       "Unknown",
        "instructions":     "Traffic light detected — exercise caution.",
    }
