"""
traffic_analyzer.py

Stage 2b: Analyze a cropped traffic light region.
Pure classical CV — no trained model required.

Color detection: spatial thirds (position vote) + HSV thresholding (color vote)
Lane relevance:  x-position ratio + size ratio + aspect ratio heuristics
"""

import json
from pathlib import Path

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

    def mark_all_lane_relevant(self, lights: list) -> None:
        """For still-image analysis, skip lane filtering and keep every light."""
        for d in lights:
            color = d.get("color", "Unknown")
            d["lane_relevant"] = True
            d["lane_score"] = 1.0
            d["instructions"] = _INSTRUCTIONS.get(color, _INSTRUCTIONS["Unknown"])[True]

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
VP_Y_BAND      = (0.44, 0.66)  # ... and near the horizon band
LANE_W_BAND    = (0.18, 0.95)  # plausible lane width at the bottom (frac W)

_LANE_MAX_AGE  = 5             # reuse last valid lane for at most this many frames
_lane_cache    = {"lane": None, "vp_x": None, "vp_y": None, "age": 999}


def reset_lane_history():
    """Clear lane smoothing state between independent images/videos."""
    _lane_cache.update(lane=None, vp_x=None, vp_y=None, age=999)


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


def _lane_debug_masks(frame):
    h, w = frame.shape[:2]
    top_y, bot_y = int(h * LANE_HORIZON), int(h * LANE_HOOD)
    cx = w / 2.0
    trap = np.array([[
        (int(cx - TRAP_TOP_HALF * w), top_y), (int(cx + TRAP_TOP_HALF * w), top_y),
        (int(cx + TRAP_BOT_HALF * w), bot_y), (int(cx - TRAP_BOT_HALF * w), bot_y),
    ]], dtype=np.int32)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, trap, 255)

    hls    = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls, np.array([0, 170, 0]),  np.array([255, 255, 80]))
    yellow = cv2.inRange(hls, np.array([15, 75, 80]), np.array([36, 255, 255]))
    white  = _filter_lane_color_mask(cv2.bitwise_and(white, roi_mask), w, h)
    yellow = _filter_lane_color_mask(cv2.bitwise_and(yellow, roi_mask), w, h)

    # Blur before Canny to suppress road-texture noise and gravel speckle.
    gray   = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges  = cv2.Canny(gray, 50, 130)

    mask = cv2.bitwise_or(cv2.bitwise_or(white, yellow), edges)
    return {
        "roi_mask": roi_mask,
        "roi": cv2.bitwise_and(frame, frame, mask=roi_mask),
        "white": white,
        "yellow": yellow,
        "edges": cv2.bitwise_and(edges, roi_mask),
        "mask": cv2.bitwise_and(mask, roi_mask),
    }


def _lane_mask(frame):
    """White+yellow markings OR road-boundary edges, kept ONLY inside a road-ahead
    trapezoid. The trapezoid is what excludes buildings/sidewalks at the frame
    sides — the main source of false 'lane' edges."""
    return _lane_debug_masks(frame)["mask"]


def _filter_lane_color_mask(mask, w, h):
    """Keep thin lane-marking-like color components, drop grass/sign blobs."""
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    frame_area = max(w * h, 1)
    max_area = 0.012 * frame_area
    min_area = max(12, int(0.000015 * frame_area))

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if area > max_area:
            continue
        fill = area / max(bw * bh, 1)
        longish = max(bw, bh) / max(min(bw, bh), 1)

        # Lane paint is thin or broken. Large filled regions inside the ROI are
        # usually grass, signs, or sunlit pavement patches.
        if area > 0.0025 * frame_area and fill > 0.45 and longish < 8.0:
            continue
        if bw > 0.55 * w and bh > 0.08 * h:
            continue
        clean[labels == i] = 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)


def _fit_curve_points(pts, ref, w, h):
    """Fit x=f(y) from grouped Hough endpoints.

    Start with a robust straight edge and only keep a quadratic when it is
    genuinely better. This keeps straight roads as clean VP triangles while
    still allowing real bends to curve toward the same apex.
    """
    if len(pts) < 4:
        return None
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)
    if ys.max() - ys.min() < 0.06 * h:       # too little vertical spread → unstable
        return None

    initial = np.poly1d(np.polyfit(ys, xs, 1))
    residuals = np.abs(xs - initial(ys))
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    inlier_limit = max(18.0, min(55.0, med + 3.0 * mad + 6.0))
    keep = residuals <= inlier_limit
    if int(np.count_nonzero(keep)) >= 4:
        ys, xs = ys[keep], xs[keep]

    linear = np.poly1d(np.polyfit(ys, xs, 1))
    linear_err = float(np.mean(np.abs(xs - linear(ys))))
    if len(xs) < 8 or ys.max() - ys.min() <= 0.16 * h:
        return linear

    curved = np.poly1d(np.polyfit(ys, xs, 2))
    curved_base = float(curved(h - 1))
    curved_err = float(np.mean(np.abs(xs - curved(ys))))
    curve_swing = float(abs(curved(h - 1) - curved(h * LANE_HORIZON)))

    # Quadratic fits can swing wildly when Hough segments cover only the far/mid
    # lane. Keep the curve only if it still lands near the selected bottom edge.
    if (
        abs(curved_base - ref) <= 0.16 * w
        and curved_err <= linear_err * 0.78
        and curve_swing <= 0.75 * w
    ):
        return curved
    return linear


def _candidate_refs(cands, side, centre, w):
    """Return bottom-x bands worth trying for one side of the ego lane."""
    if not cands:
        return []
    tol = 0.12 * w
    dead = 0.06 * w
    if side == "left":
        pool = [c for c in cands if c[0] < centre + tol]
        anchors = [c for c in pool if c[0] < centre - dead]
    else:
        pool = [c for c in cands if c[0] > centre - tol]
        anchors = [c for c in pool if c[0] > centre + dead]
    pool = anchors or pool
    if not pool:
        return []

    # Cluster by extrapolated bottom x. Dashed/full markings produce several
    # short Hough segments in the same band; road arrows and centre markings
    # usually land in their own band and can be rejected later by geometry.
    refs = []
    for xb, _p1, _p2 in sorted(pool, key=lambda c: c[0]):
        if not refs or abs(xb - refs[-1][-1]) > 0.07 * w:
            refs.append([xb])
        else:
            refs[-1].append(xb)

    ranked = []
    for group in refs:
        center = float(np.median(group))
        support = len(group)
        if side == "left":
            proximity = 1.0 - min(abs(center - (centre - 0.22 * w)) / (0.45 * w), 1.0)
        else:
            proximity = 1.0 - min(abs(center - (centre + 0.22 * w)) / (0.45 * w), 1.0)
        ranked.append((support + proximity, center))
    ranked.sort(reverse=True)
    return [center for _score, center in ranked[:4]]


def _edge_from_ref(cands, ref, w, h):
    """Build one edge curve from the Hough band around ref bottom-x."""
    band = 0.11 * w
    pts = []
    for xb, p1, p2 in cands:
        if abs(xb - ref) <= band:
            pts += [p1, p2]
    return _fit_curve_points(pts, ref, w, h)


def _lane_score(lane, vp_x, vp_y, L, R, h, w):
    """Higher is better for a valid candidate lane pair."""
    base_y = h - 1
    horizon = h * LANE_HORIZON
    lx_b, rx_b = float(L(base_y)), float(R(base_y))
    lx_t, rx_t = float(L(horizon)), float(R(horizon))
    bottom_w = max(rx_b - lx_b, 1.0)
    top_w = max(rx_t - lx_t, 1.0)

    center_at_base = (lx_b + rx_b) / 2.0
    width_score = 1.0 - min(abs((bottom_w / w) - 0.55) / 0.45, 1.0)
    center_score = 1.0 - min(abs(center_at_base - w / 2.0) / (w / 2.0), 1.0)
    vp_score = 1.0 - min(abs(vp_x - w / 2.0) / (w / 2.0), 1.0)
    converge_score = 1.0 - min(top_w / bottom_w, 1.0)
    height_score = 1.0 - min(abs(vp_y - horizon) / (0.35 * h), 1.0)
    left_pos_score = 1.0 - min(abs((lx_b / w) - 0.34) / 0.28, 1.0)
    right_pos_score = 1.0 - min(abs((rx_b / w) - 0.84) / 0.28, 1.0)
    return (
        0.22 * center_score
        + 0.20 * vp_score
        + 0.18 * width_score
        + 0.14 * converge_score
        + 0.08 * height_score
        + 0.09 * left_pos_score
        + 0.09 * right_pos_score
    )


def _select_lane_pair(left, right, centre, w, h):
    """Try Hough bottom-x bands and return the best valid left/right pair."""
    best = None
    for l_ref in _candidate_refs(left, "left", centre, w):
        L = _edge_from_ref(left, l_ref, w, h)
        if L is None:
            continue
        for r_ref in _candidate_refs(right, "right", centre, w):
            R = _edge_from_ref(right, r_ref, w, h)
            if R is None:
                continue
            lane, vp_x, vp_y, reason = _validated_lane(L, R, h, w)
            if lane is None:
                continue
            score = _lane_score(lane, vp_x, vp_y, L, R, h, w)
            if best is None or score > best[0]:
                best = (score, lane, vp_x, vp_y, reason)
    if best is None:
        return None, centre, h * LANE_HORIZON, "no_valid_lane_pair"
    _score, lane, vp_x, vp_y, reason = best
    return lane, vp_x, vp_y, reason


def _validated_lane(L, R, h, w):
    """Evaluate the two edge polynomials, then VALIDATE the geometry: edges
    straddle the camera, plausible lane width, the lane narrows with distance
    (perspective), and the curve stays in frame (rejects wild fits). Returns
    (lane_lines, vp_x, vp_y); lane_lines is None when it isn't a confident, sane
    lane. lane_lines carries a straight chord ('left'/'right') for geometry plus a
    polyline ('left_poly'/'right_poly') for drawing the actual curve."""
    horizon = h * LANE_HORIZON
    if L is None and R is None:
        return None, w / 2.0, horizon, "missing_left_and_right_curve"
    if L is None:
        return None, w / 2.0, horizon, "missing_left_curve"
    if R is None:
        return None, w / 2.0, horizon, "missing_right_curve"

    base_y = h - 1
    lx_b, rx_b = float(L(base_y)), float(R(base_y))      # bottom of frame
    lx_t, rx_t = float(L(horizon)), float(R(horizon))    # at the horizon

    # Lines must converge correctly — if they cross at or before the horizon the
    # fit is inverted (centre marking mistaken for a lane edge).
    if lx_t > rx_t + 0.05 * w:
        return None, w / 2.0, horizon, "lanes_cross_at_horizon"

    if not (lx_b < w / 2.0 < rx_b):
        return None, w / 2.0, horizon, "lane_does_not_straddle_center"
    bottom_w = rx_b - lx_b
    if not (LANE_W_BAND[0] * w <= bottom_w <= LANE_W_BAND[1] * w):
        return None, w / 2.0, horizon, "bottom_lane_width_out_of_range"
    if (rx_t - lx_t) > bottom_w * 1.15:    # lane must narrow with distance, not diverge
        return None, w / 2.0, horizon, "lane_diverges_toward_horizon"

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
    if not (VP_X_BAND[0] * w <= vp_x <= VP_X_BAND[1] * w):
        return None, vp_x, vp_y, "vanishing_point_x_out_of_band"
    if not (VP_Y_BAND[0] * h <= vp_y <= VP_Y_BAND[1] * h):
        return None, vp_x, vp_y, "vanishing_point_y_out_of_band"

    # Curved edges (near field) that CONVERGE to the vanishing point (far field).
    apex = (int(round(vp_x)), int(round(vp_y)))
    ys = np.linspace(base_y, vp_y, 14, endpoint=False)
    left_poly  = [(int(L(y)), int(y)) for y in ys] + [apex]
    right_poly = [(int(R(y)), int(y)) for y in ys] + [apex]
    if any(not (-0.15 * w <= x <= 1.15 * w) for x, _ in left_poly + right_poly):
        return None, vp_x, vp_y, "lane_curve_out_of_frame"

    lane = {
        "left":  [int(lx_b), base_y, apex[0], apex[1]],   # straight chord → VP (geometry)
        "right": [int(rx_b), base_y, apex[0], apex[1]],
        "left_poly":  left_poly,                          # curve → VP (drawing)
        "right_poly": right_poly,
    }
    return lane, float(vp_x), float(vp_y), "valid_lane"


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


def detect_lane(frame: np.ndarray, default_ratio: float = 0.5, explain: bool = False):
    """Robust ego-lane detection (pure CV).

    Masks white/yellow markings + road-boundary edges inside a road-ahead
    trapezoid, fits the nearest left & right edges, and VALIDATES the geometry.
    Returns (vp_x, vp_y, lane_lines, segments). With explain=True, appends a
    validation reason. lane_lines is None whenever the result isn't a confident,
    sane lane — the caller then draws a clean forward corridor instead of garbage.
    """
    h, w = frame.shape[:2]
    cx, horizon = w * default_ratio, h * LANE_HORIZON
    if h < 40 or w < 40:
        if explain:
            return cx, horizon, None, [], "frame_too_small"
        return cx, horizon, None, []

    mask  = _lane_mask(frame)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=25,
                            minLineLength=max(int(h * 0.06), 45), maxLineGap=80)
    if lines is None:
        if explain:
            return cx, horizon, None, [], "no_hough_lines"
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

    lane, vp_x, vp_y = None, cx, horizon
    reason = "no_segments_after_slope_filter"
    if segs:
        lane, vp_x, vp_y, reason = _select_lane_pair(left, right, centre, w, h)

    if lane is not None:
        _lane_cache.update(lane=lane, vp_x=vp_x, vp_y=vp_y, age=0)
        if explain:
            return vp_x, vp_y, lane, segs, reason
        return vp_x, vp_y, lane, segs

    # Detection failed — reuse last valid lane for a few frames so that
    # junctions / arrows / sparse marking frames don't drop to fallback.
    _lane_cache["age"] += 1
    if _lane_cache["lane"] is not None and _lane_cache["age"] <= _LANE_MAX_AGE:
        if explain:
            return (_lane_cache["vp_x"], _lane_cache["vp_y"],
                    _lane_cache["lane"], segs, "smoothed_from_history")
        return _lane_cache["vp_x"], _lane_cache["vp_y"], _lane_cache["lane"], segs

    if explain:
        return cx, horizon, None, segs, reason
    return cx, horizon, None, segs


def _draw_debug_lane(frame, lane, assumed=False):
    color = (0, 200, 220) if assumed else (0, 220, 0)
    thickness = 2 if assumed else 3
    for key in ("left", "right"):
        poly = lane.get(key + "_poly")
        if poly and len(poly) >= 2:
            cv2.polylines(frame, [np.array(poly, dtype=np.int32).reshape(-1, 1, 2)],
                          False, color, thickness)
        elif lane.get(key):
            x1, y1, x2, y2 = lane[key]
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def save_lane_debug(frame: np.ndarray, output_dir, prefix: str = "frame") -> dict:
    """Save lane detector debug images and return the matching JSON report."""
    frame_dir = Path(output_dir) / prefix
    frame_dir.mkdir(parents=True, exist_ok=True)

    debug = _lane_debug_masks(frame)
    vp_x, vp_y, lane, segments, validation_reason = detect_lane(frame, explain=True)
    lane_found = lane is not None
    if lane is None:
        _vp, lane = default_roi_lane(frame.shape)

    segments_img = frame.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(segments_img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    final_img = frame.copy()
    _draw_debug_lane(final_img, lane, assumed=not lane_found)
    label = "detected" if lane_found else "fallback"
    color = (0, 220, 0) if lane_found else (0, 200, 220)
    cv2.putText(final_img, f"lane: {label}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(final_img, f"vp: ({vp_x:.1f}, {vp_y:.1f})", (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite(str(frame_dir / "original.jpg"), frame)
    cv2.imwrite(str(frame_dir / "roi.jpg"), debug["roi"])
    cv2.imwrite(str(frame_dir / "white.jpg"), debug["white"])
    cv2.imwrite(str(frame_dir / "yellow.jpg"), debug["yellow"])
    cv2.imwrite(str(frame_dir / "edges.jpg"), debug["edges"])
    cv2.imwrite(str(frame_dir / "mask.jpg"), debug["mask"])
    cv2.imwrite(str(frame_dir / "segments.jpg"), segments_img)
    cv2.imwrite(str(frame_dir / "final.jpg"), final_img)

    centre = frame.shape[1] / 2.0
    left_candidates = 0
    right_candidates = 0
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2.0
        if slope < 0 and mid_x < centre + 0.10 * frame.shape[1]:
            left_candidates += 1
        elif slope > 0 and mid_x > centre - 0.10 * frame.shape[1]:
            right_candidates += 1

    report = {
        "lane_found": lane_found,
        "validation_reason": validation_reason,
        "vanishing_point": {"x": round(float(vp_x), 2), "y": round(float(vp_y), 2)},
        "white_pixels": int(np.count_nonzero(debug["white"])),
        "yellow_pixels": int(np.count_nonzero(debug["yellow"])),
        "edge_pixels": int(np.count_nonzero(debug["edges"])),
        "mask_pixels": int(np.count_nonzero(debug["mask"])),
        "segments": len(segments),
        "left_candidates": left_candidates,
        "right_candidates": right_candidates,
    }
    with open(frame_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


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
