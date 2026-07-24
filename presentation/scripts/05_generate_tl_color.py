# Run from the project root:
#   .venv/Scripts/python.exe run_tl_color_detection_figure.py
# Saves tl_color_detection_figure.png in the project root.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from analyzers.traffic_analyzer import _detect_color, _HSV_RANGES

BASE     = Path(__file__).resolve().parent.parent.parent
IMG_PATH = BASE / 'data/raw/traffic_lights/traffic_light_hard_f000330_t00011.0.jpg'

# ── Detected bounding boxes (from detector_model.pt) ─────────────────────
# Both are Red lights; we process the first (higher confidence)
BOX       = (891, 337, 906, 360)    # primary — conf=0.62
ALL_BOXES = [(891, 337, 906, 360),  # conf=0.62
             (588, 344, 606, 372)]  # conf=0.58

# ── Load and crop ─────────────────────────────────────────────────────────
img = cv2.imread(str(IMG_PATH))
assert img is not None, f'Cannot load: {IMG_PATH}'
x1, y1, x2, y2 = BOX
crop_bgr = img[y1:y2, x1:x2].copy()

# ── Step 1: Resize to standard 48×144 (used in production) ───────────────
STANDARD_W, STANDARD_H = 48, 144
resized_bgr = cv2.resize(crop_bgr, (STANDARD_W, STANDARD_H))
resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

# ── Step 2: Spatial thirds — brightness vote ─────────────────────────────
hsv_std    = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
third_h    = STANDARD_H // 3
thirds_v   = [hsv_std[i*third_h:(i+1)*third_h, :, 2] for i in range(3)]
brightness = [float(np.mean(t)) for t in thirds_v]
pos_idx    = int(np.argmax(brightness))
pos_labels = ['Red', 'Yellow', 'Green']
pos_colors = ['#e74c3c', '#f39c12', '#27ae60']

# Build spatial-thirds visualisation (annotated copy of resized image)
thirds_vis = resized_rgb.copy()
for i in range(3):
    y_top = i * third_h
    y_bot = (i + 1) * third_h
    border_rgb = tuple(int(c * 255) for c in plt.cm.colors.to_rgb(pos_colors[i]))
    thickness  = 4 if i == pos_idx else 2
    cv2.rectangle(thirds_vis,
                  (0, y_top), (STANDARD_W - 1, y_bot - 1),
                  border_rgb[::-1], thickness)   # note: this is still RGB array so no BGR swap needed for imshow

# ── Step 3: HSV color masks — color vote ─────────────────────────────────
hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
mask_results = {}
for color, ranges in _HSV_RANGES.items():
    mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_crop, lo, hi))
    mask_results[color] = mask

# Colored overlay: pixels colored by which mask they matched
overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).copy().astype(np.float32) * 0.35
color_map = {'Red': (220,30,30), 'Yellow': (250,180,0), 'Green': (30,200,60)}
for color, mask in mask_results.items():
    r, g, b = color_map[color]
    overlay[mask > 0, 0] = r
    overlay[mask > 0, 1] = g
    overlay[mask > 0, 2] = b
overlay = overlay.clip(0, 255).astype(np.uint8)

# Count matched pixels per color
pixel_counts = {c: int(np.sum(m > 0)) for c, m in mask_results.items()}
total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
color_vote   = max(pixel_counts, key=pixel_counts.get)

# ── Step 4: Final decision (run the actual production function) ───────────
final_color, position_vote, color_vote_out, confidence = _detect_color(crop_bgr)

# ── Full frame with both bounding boxes drawn ─────────────────────────────
tl_col_rgb = {'Red':(220,30,30),'Yellow':(240,180,0),'Green':(30,200,60),'Unknown':(180,180,180)}
frame_vis  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
sx, sy     = 1.0, 1.0   # keep native resolution (1280×720)

# Draw all boxes in neutral white — YOLO Stage 1 only detects presence, not color
for i, (bx1,by1,bx2,by2) in enumerate(ALL_BOXES):
    thick = 3 if i == 0 else 2
    cv2.rectangle(frame_vis, (bx1,by1), (bx2,by2), (255,255,255), thick)
    cv2.putText(frame_vis, 'Traffic Light', (bx1, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

# ── 2 × 3 figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.12)

axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

# ── Panel [0,0] — Full frame ──────────────────────────────────────────────
axes[0][0].imshow(frame_vis)
axes[0][0].set_title('Detected Traffic Light\n(YOLO Stage 1 bounding box)', fontsize=11, fontweight='bold')
axes[0][0].axis('off')

# ── Panel [0,1] — Raw crop ────────────────────────────────────────────────
axes[0][1].imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
axes[0][1].set_title(f'Cropped Region\n({crop_bgr.shape[1]}×{crop_bgr.shape[0]} px)', fontsize=11, fontweight='bold')
axes[0][1].axis('off')

# ── Panel [0,2] — Resized + thirds overlay ───────────────────────────────
axes[0][2].imshow(thirds_vis)
for i in range(3):
    yc = (i + 0.5) * third_h
    label_str = f'{pos_labels[i]}\n{brightness[i]:.0f}'
    weight = 'bold' if i == pos_idx else 'normal'
    axes[0][2].text(STANDARD_W + 4, yc, label_str,
                    va='center', fontsize=9, fontweight=weight,
                    color=pos_colors[i])
# horizontal dividers
for i in [1, 2]:
    axes[0][2].axhline(i * third_h - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.7)
axes[0][2].set_title(f'Resized to {STANDARD_W}×{STANDARD_H}\nVote 1: Spatial Thirds (brightness)', fontsize=11, fontweight='bold')
axes[0][2].set_xlim(0, STANDARD_W + 52)
axes[0][2].axis('off')

# ── Panel [1,0] — HSV masks overlay ──────────────────────────────────────
axes[1][0].imshow(overlay)
patches = [mpatches.Patch(color=np.array(color_map[c])/255,
                           label=f'{c}: {pixel_counts[c]} px ({pixel_counts[c]/total_pixels*100:.1f}%)')
           for c in ['Red','Yellow','Green']]
axes[1][0].legend(handles=patches, loc='lower right', fontsize=8,
                   framealpha=0.85, handlelength=1.2)
axes[1][0].set_title('Vote 2: HSV Color Masking\n(saturated R / Y / G pixels)', fontsize=11, fontweight='bold')
axes[1][0].axis('off')
cv_px  = pixel_counts.get(color_vote_out, 0)
cv_col = pos_colors[pos_labels.index(color_vote_out)] if color_vote_out in pos_labels else 'gray'
axes[1][0].text(0.5, -0.06,
                f'→ Color vote: {color_vote_out}  ({cv_px} px)',
                transform=axes[1][0].transAxes, ha='center', fontsize=9,
                fontweight='bold', color=cv_col)

# ── Panel [1,1] — Individual masks ───────────────────────────────────────
mask_rgb_combined = np.zeros((*crop_bgr.shape[:2], 3), dtype=np.uint8)
for i, (color, mask) in enumerate(mask_results.items()):
    r, g, b = color_map[color]
    mask_rgb_combined[mask > 0] = (r, g, b)
axes[1][1].imshow(mask_rgb_combined)
axes[1][1].set_title('Matched Pixels Only\n(Red | Yellow | Green)', fontsize=11, fontweight='bold')
axes[1][1].axis('off')

# ── Panel [1,2] — Final decision ─────────────────────────────────────────
result_img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).copy()
result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
col_bgr    = tuple(reversed(tl_col_rgb.get(final_color,(180,180,180))))
cv2.rectangle(result_bgr, (0,0), (crop_bgr.shape[1]-1, crop_bgr.shape[0]-1), col_bgr, 4)
axes[1][2].imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

conf_label = f'Confidence: {"High (1.0)" if confidence == 1.0 else "Low (0.5)"}'
axes[1][2].set_title(f'Final Decision: {final_color}\n{conf_label}', fontsize=12,
                      fontweight='bold',
                      color=pos_colors[pos_labels.index(final_color)] if final_color in pos_labels else 'gray')
axes[1][2].axis('off')

plt.suptitle('Traffic Light Colour Detection — Classical CV (No ML Model)\n'
             'Stage 2b: Spatial Brightness Vote  +  HSV Colour Thresholding',
             fontsize=14, fontweight='bold', y=1.02)

out = BASE / 'presentation/figures/05_traffic_light_color.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
print(f'  Position vote : {position_vote}  (brightness: {[f"{b:.0f}" for b in brightness]})')
print(f'  Color vote    : {color_vote_out}  (pixels: {pixel_counts})')
print(f'  Final result  : {final_color}  (confidence={confidence})')
