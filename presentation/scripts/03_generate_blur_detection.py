# Run from the project root:
#   .venv/Scripts/python.exe run_blur_detection_figure.py
# Saves blur_detection_figure.png in the project root.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent.parent
IMG_PATH = BASE / 'data/raw/traffic_lights/4483547-uhd_2562_1440_30fps_f000690_t00023.0.jpg'

THRESHOLD = 60.0   # same value used in src/extract_frames.py

# ── load & resize ─────────────────────────────────────────────────────────
img = cv2.imread(str(IMG_PATH))
assert img is not None, f'Cannot load: {IMG_PATH}'
img = cv2.resize(img, (640, 360))

# ── sharp frame = the real captured frame ─────────────────────────────────
sharp = img.copy()

# ── blurry frame = simulated motion blur (camera shake / fast movement) ───
# Two-step: Gaussian blur + horizontal motion kernel (mimics vehicle motion)
motion_kernel      = np.zeros((1, 25))
motion_kernel[0, :] = 1.0 / 25
blurry = cv2.filter2D(img, -1, motion_kernel)           # horizontal motion
blurry = cv2.GaussianBlur(blurry, (9, 9), 4.0)         # + out-of-focus

# ── Laplacian variance score ───────────────────────────────────────────────
def lap_score(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return gray, lap, lap.var()

sharp_gray,  sharp_lap,  sharp_var  = lap_score(sharp)
blurry_gray, blurry_lap, blurry_var = lap_score(blurry)

rows = [
    (sharp,  sharp_gray,  sharp_lap,  sharp_var,  'KEPT',     '#27ae60'),
    (blurry, blurry_gray, blurry_lap, blurry_var, 'REJECTED', '#e74c3c'),
]

# ── figure: 2 rows × 3 columns ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

col_titles = ['Captured Frame', 'Greyscale', 'Laplacian Response\n(edge energy)']

for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight='bold', pad=8)

for row_idx, (bgr, gray, lap, var, verdict, color) in enumerate(rows):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Col 0 — original frame + verdict badge
    axes[row_idx][0].imshow(rgb)
    axes[row_idx][0].text(
        0.02, 0.97, verdict,
        transform=axes[row_idx][0].transAxes,
        fontsize=14, fontweight='bold', color='white', va='top',
        bbox=dict(facecolor=color, boxstyle='round,pad=0.35', alpha=0.92)
    )
    axes[row_idx][0].set_ylabel(
        f'Variance = {var:.0f}', fontsize=11, fontweight='bold', color=color
    )

    # Col 1 — greyscale
    axes[row_idx][1].imshow(gray, cmap='gray')

    # Col 2 — Laplacian heatmap (bright = high edge response = sharp)
    axes[row_idx][2].imshow(np.abs(lap), cmap='hot')
    axes[row_idx][2].text(
        0.5, -0.06,
        f'Variance = {var:.0f}  →  {"KEEP" if var >= THRESHOLD else "DISCARD"}',
        transform=axes[row_idx][2].transAxes,
        ha='center', fontsize=10, fontweight='bold', color=color
    )

    for ax in axes[row_idx]:
        ax.axis('off')

# ── threshold rule banner ──────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    f'Decision rule:  Laplacian Variance > {THRESHOLD:.0f}  →  frame is sharp  →  KEPT for training'
    f'   |   Variance ≤ {THRESHOLD:.0f}  →  blurry  →  DISCARDED',
    ha='center', fontsize=10,
    bbox=dict(facecolor='#f4f4f4', boxstyle='round,pad=0.5', edgecolor='#cccccc')
)

plt.suptitle(
    'Blur Detection — Laplacian Operator (2nd-order Derivative)\n'
    'Applied to every frame extracted from dashcam video before training',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

out = BASE / 'presentation/figures/03_blur_detection.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
print(f'  Sharp  frame variance : {sharp_var:.1f}  → KEPT')
print(f'  Blurry frame variance : {blurry_var:.1f}  → REJECTED')
