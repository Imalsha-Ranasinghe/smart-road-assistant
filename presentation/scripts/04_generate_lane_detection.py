# Run from the project root:
#   .venv/Scripts/python.exe run_lane_detection_figure.py
# Saves lane_detection_figure.png in the project root.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from analyzers.traffic_analyzer import (
    detect_lane, default_roi_lane,
    LANE_HORIZON, LANE_HOOD, TRAP_TOP_HALF, TRAP_BOT_HALF,
    LANE_SLOPE_MIN, LANE_SLOPE_MAX,
)
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from pipeline import draw_lane

BASE     = Path(__file__).resolve().parent.parent.parent
IMG_PATH = BASE / 'data/raw/traffic_lights/traffic_light_hard_f000330_t00011.0.jpg'

# ── load ──────────────────────────────────────────────────────────────────
img = cv2.imread(str(IMG_PATH))
assert img is not None, f'Cannot load: {IMG_PATH}'
img = cv2.resize(img, (960, 540))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]


# ── reproduce each step from traffic_analyzer._lane_mask() ───────────────

def make_trapezoid_mask(h, w):
    top_y  = int(h * LANE_HORIZON)
    bot_y  = int(h * LANE_HOOD)
    cx     = w / 2.0
    trap   = np.array([[
        (int(cx - TRAP_TOP_HALF * w), top_y),
        (int(cx + TRAP_TOP_HALF * w), top_y),
        (int(cx + TRAP_BOT_HALF * w), bot_y),
        (int(cx - TRAP_BOT_HALF * w), bot_y),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, trap, 255)
    return mask

trap_mask = make_trapezoid_mask(h, w)

# HLS color thresholding inside ROI
hls    = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
white  = cv2.inRange(hls, np.array([0, 165, 0]),  np.array([255, 255, 70]))
yellow = cv2.inRange(hls, np.array([15, 60, 60]), np.array([40, 255, 255]))
color_mask = cv2.bitwise_or(white, yellow)
color_roi  = cv2.bitwise_and(color_mask, trap_mask)

# Canny edges inside ROI
gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges_full = cv2.Canny(gray, 60, 150)
edges_roi  = cv2.bitwise_and(edges_full, trap_mask)

# Combined mask (what HoughLinesP sees)
combined = cv2.bitwise_or(cv2.bitwise_or(white, yellow), edges_full)
combined = cv2.bitwise_and(combined, trap_mask)

# Hough lines
lines = cv2.HoughLinesP(combined, 1, np.pi / 180, threshold=40,
                         minLineLength=max(int(h * 0.10), 25), maxLineGap=60)

hough_vis = img.copy()
if lines is not None:
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        if LANE_SLOPE_MIN <= abs(slope) <= LANE_SLOPE_MAX:
            mid_x = (x1 + x2) / 2.0
            color  = (255, 100, 0) if mid_x < w/2 else (0, 180, 255)  # left=orange, right=blue
            cv2.line(hough_vis, (x1, y1), (x2, y2), color, 2)

# Final result: run the full pipeline
vp_x, vp_y, lane_lines, segs = detect_lane(img)
final = img.copy()
if lane_lines:
    draw_lane(final, lane_lines)
    # Draw vanishing point
    cv2.circle(final, (int(vp_x), int(vp_y)), 8,  (0, 255, 255), -1)
    cv2.circle(final, (int(vp_x), int(vp_y)), 12, (0, 255, 255),  2)
    cv2.putText(final, 'VP', (int(vp_x) + 14, int(vp_y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
else:
    # fallback ROI shown in amber
    _, roi_lane = default_roi_lane(img.shape)
    draw_lane(final, {**roi_lane, 'assumed': True})
    cv2.putText(final, 'Fallback ROI', (w//2 - 80, int(h * LANE_HORIZON) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 220), 2)

# Trap overlay on original for display
trap_overlay = rgb.copy()
cv2.polylines(trap_overlay,
              [np.array([
                  [int(w/2 - TRAP_TOP_HALF * w), int(h * LANE_HORIZON)],
                  [int(w/2 + TRAP_TOP_HALF * w), int(h * LANE_HORIZON)],
                  [int(w/2 + TRAP_BOT_HALF * w), int(h * LANE_HOOD)],
                  [int(w/2 - TRAP_BOT_HALF * w), int(h * LANE_HOOD)],
              ], dtype=np.int32)],
              True, (255, 220, 0), 2)
cv2.putText(trap_overlay, 'Road-ahead ROI', (int(w/2 - TRAP_TOP_HALF * w),
            int(h * LANE_HORIZON) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)


# ── 2 × 3 figure ─────────────────────────────────────────────────────────
panels = [
    (rgb,                                  'Original Frame',                   None),
    (trap_overlay,                         'Step 1 — Trapezoid ROI\n(road-ahead mask)', None),
    (cv2.cvtColor(color_roi,  cv2.COLOR_GRAY2RGB), 'Step 2 — HLS Color Mask\n(white + yellow markings)', None),
    (cv2.cvtColor(edges_roi,  cv2.COLOR_GRAY2RGB), 'Step 3 — Canny Edge Detector\n(road boundaries)',     None),
    (cv2.cvtColor(hough_vis,  cv2.COLOR_BGR2RGB),  'Step 4 — Hough Lines\n(orange=left  blue=right)',     None),
    (cv2.cvtColor(final,      cv2.COLOR_BGR2RGB),  'Step 5 — Lane + Vanishing Point\n(polyfit + VP)',     None),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, (panel, title, _) in zip(axes.flat, panels):
    ax.imshow(panel)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

plt.suptitle('Lane Detection Pipeline — Classical CV Techniques',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()

out = BASE / 'presentation/figures/04_lane_detection.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
status = 'DETECTED' if lane_lines else 'FALLBACK ROI (no confident lane found)'
print(f'Lane status : {status}')
if lane_lines:
    print(f'Vanishing point : ({vp_x:.0f}, {vp_y:.0f})')
