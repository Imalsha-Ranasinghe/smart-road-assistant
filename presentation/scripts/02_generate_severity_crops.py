# Run from the project root:
#   .venv/Scripts/python.exe run_severity_figure.py
# Saves severity_crops.png in the project root.
import os
os.environ['YOLO_VERBOSE'] = 'False'       # suppress YOLO console spam

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

BASE  = Path(__file__).resolve().parent.parent.parent
MODEL = YOLO(str(BASE / 'models/severity_model.pt'))

IDX_TO_CLASS = {0: 'High', 1: 'Low', 2: 'Medium'}   # alphabetical folder order
SEV_COLORS   = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
PADDING      = 10

IMAGES = [
    BASE / 'data/raw/potholes/small/1067_1_img-273_jpg.rf.5cb7aacf7db557524549b408bc043e8b.jpg',
    BASE / 'data/raw/potholes/medium/779_5_img-74_jpg.rf.47c99578f55c2761e6d4f8bcf021d6f2.jpg',
    BASE / 'data/raw/potholes/large/128_5_img-541_jpg.rf.ef924ac7c487acc572f57d8daf390a49.jpg',
]
LABELS = ['Low', 'Medium', 'High']


def read_yolo_box(txt_path, img_w, img_h):
    """
    Handles both label formats:
      Standard YOLO (5 values): class cx cy w h
      Polygon/OBB  (>5 values): class x1 y1 x2 y2 x3 y3 x4 y4 ...
    Returns (xmin, ymin, xmax, ymax) in pixels for the first box.
    """
    for line in Path(txt_path).read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            # Standard YOLO: cx cy w h
            xc, yc, bw, bh = vals
            xmin = int((xc - bw / 2) * img_w)
            ymin = int((yc - bh / 2) * img_h)
            xmax = int((xc + bw / 2) * img_w)
            ymax = int((yc + bh / 2) * img_h)
        else:
            # Polygon format: x1 y1 x2 y2 x3 y3 x4 y4 ...
            xs = [vals[i] * img_w for i in range(0, len(vals) - 1, 2)]
            ys = [vals[i] * img_h for i in range(1, len(vals),     2)]
            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))
        return (
            max(xmin, 0), max(ymin, 0),
            min(xmax, img_w), min(ymax, img_h),
        )
    return None


fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                          gridspec_kw={'height_ratios': [3, 1.5]})

for col, (img_path, true_label) in enumerate(zip(IMAGES, LABELS)):
    txt_path = img_path.with_suffix('.txt')
    img = cv2.imread(str(img_path))
    assert img is not None, f'Cannot load: {img_path}'
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    box = read_yolo_box(txt_path, w, h)
    assert box is not None, f'No annotation in: {txt_path}'
    xmin, ymin, xmax, ymax = box

    x1, y1 = max(xmin - PADDING, 0), max(ymin - PADDING, 0)
    x2, y2 = min(xmax + PADDING, w), min(ymax + PADDING, h)
    crop_bgr = cv2.resize(img[y1:y2, x1:x2], (128, 128))

    print(f'Running model on {true_label} image...', end=' ', flush=True)
    result   = MODEL(crop_bgr, imgsz=128, verbose=False)[0]
    pred_cls = IDX_TO_CLASS[int(result.probs.top1)]
    conf     = float(result.probs.top1conf)
    print(f'→ {pred_cls}  ({conf:.1%})')

    color = SEV_COLORS[pred_cls]

    axes[0][col].imshow(rgb)
    axes[0][col].add_patch(patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=3, edgecolor=color, facecolor='none'))
    axes[0][col].set_title(f'Predicted: {pred_cls}  ({conf:.1%})',
                            fontsize=12, fontweight='bold', color=color)
    axes[0][col].axis('off')

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    axes[1][col].imshow(crop_rgb)
    axes[1][col].set_title('Input crop (128×128)', fontsize=9)
    axes[1][col].axis('off')
    axes[1][col].text(0.5, -0.12, f'→ {pred_cls}  ({conf:.1%})',
                      transform=axes[1][col].transAxes,
                      ha='center', fontsize=11, fontweight='bold', color=color)

plt.suptitle('Severity Classification Model — Inference Results',
             fontsize=14, fontweight='bold')
plt.tight_layout()

out = BASE / 'presentation/figures/02_pothole_severity_crops.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
print(f'\nSaved → {out}')
