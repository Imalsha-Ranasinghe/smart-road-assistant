# Run from the project root:
#   .venv/Scripts/python.exe run_detector_io_figure.py
# Saves detector_io_figure.png in the project root.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from ultralytics import YOLO

BASE  = Path(__file__).resolve().parent.parent.parent
MODEL = BASE / 'models/detector_model.pt'

# ── Images & GT label files ───────────────────────────────────────────────
TL_IMG  = BASE / 'data/raw/traffic_lights/traffic_light_hard_f000330_t00011.0.jpg'
TL_TXT  = BASE / 'data/raw/traffic_lights/traffic_light_hard_f000330_t00011.0.txt'

PH_IMG  = BASE / 'data/raw/potholes/852_1_left000553_jpg.rf.155e6f7aff9665d433a89a574450b69f.jpg'
PH_TXT  = BASE / 'data/raw/potholes/192_4_left000253_jpg.rf.7c87f7b317a1ac8edf5ce94dbeddc956.txt'

# ── Class colours (BGR for cv2, RGB for matplotlib) ───────────────────────
CLS_COLOR_BGR = {0: (0, 100, 255),    # pothole     → orange-red
                 1: (255, 200, 0)}     # traffic_light → cyan-blue
CLS_COLOR_RGB = {0: (255, 100, 0),
                 1: (0, 200, 255)}
CLS_NAME      = {0: 'Pothole', 1: 'Traffic Light'}
GT_BGR        = (0, 220, 0)           # ground truth → green


# ── Parse YOLO label file (handles standard bbox AND polygon format) ──────
def parse_gt(txt_path, img_w, img_h):
    boxes = []
    for line in Path(txt_path).read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls  = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:                          # standard: cx cy w h
            xc, yc, bw, bh = vals
            x1 = int((xc - bw / 2) * img_w)
            y1 = int((yc - bh / 2) * img_h)
            x2 = int((xc + bw / 2) * img_w)
            y2 = int((yc + bh / 2) * img_h)
        else:                                       # polygon: x1 y1 x2 y2 …
            xs = [vals[i] * img_w for i in range(0, len(vals) - 1, 2)]
            ys = [vals[i] * img_h for i in range(1, len(vals),     2)]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
        boxes.append((cls, max(x1, 0), max(y1, 0),
                      min(x2, img_w), min(y2, img_h)))
    return boxes


# ── Draw a single box with label ──────────────────────────────────────────
def draw_box(img, x1, y1, x2, y2, color_bgr, label, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 6, th + 2)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color_bgr, -1)
    cv2.putText(img, label, (x1 + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ── Draw dashed rectangle (ground truth style) ───────────────────────────
def draw_dashed_box(img, x1, y1, x2, y2, color_bgr, dash=12, gap=6):
    pts = []
    # top edge
    for x in range(x1, x2, dash + gap):
        pts.append(((x, y1), (min(x + dash, x2), y1)))
    # bottom edge
    for x in range(x1, x2, dash + gap):
        pts.append(((x, y2), (min(x + dash, x2), y2)))
    # left edge
    for y in range(y1, y2, dash + gap):
        pts.append(((x1, y), (x1, min(y + dash, y2))))
    # right edge
    for y in range(y1, y2, dash + gap):
        pts.append(((x2, y), (x2, min(y + dash, y2))))
    for p1, p2 in pts:
        cv2.line(img, p1, p2, color_bgr, 2)


# ── Build the four panel images ───────────────────────────────────────────
model = YOLO(str(MODEL))
print('Model loaded.')

output_panels = []   # list of rgb_image

for img_path, txt_path in [
        (TL_IMG, TL_TXT),
        (PH_IMG, PH_TXT),
]:
    raw = cv2.imread(str(img_path))
    assert raw is not None, f'Cannot load {img_path}'
    h, w = raw.shape[:2]

    out = raw.copy()
    results = model(raw, conf=0.35, verbose=False)[0]
    for box in results.boxes:
        cls  = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        color = CLS_COLOR_BGR.get(cls, (200, 200, 200))
        draw_box(out, x1, y1, x2, y2, color,
                 f'{CLS_NAME.get(cls, "?")}  {conf:.0%}  cls={cls}',
                 thickness=2)
    output_panels.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ── 1 × 2 figure (detector output only) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('white')

for ax, rgb in zip(axes, output_panels):
    ax.imshow(rgb)
    ax.axis('off')

# ── Legend ────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=np.array(CLS_COLOR_RGB[1]) / 255,
                   label='Class 1 — Traffic Light'),
    mpatches.Patch(color=np.array(CLS_COLOR_RGB[0]) / 255,
                   label='Class 0 — Pothole'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2,
           fontsize=11, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.04))

plt.suptitle('YOLOv8s Object Detector Output\n'
             'Stage 1: Two-class detection  (Class 0 = Pothole  |  Class 1 = Traffic Light)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out = BASE / 'presentation/figures/01_detector_input_output.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
