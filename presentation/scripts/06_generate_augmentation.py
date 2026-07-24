# Run from the project root:
#   .venv/Scripts/python.exe run_augmentation_figure.py
# Saves augmentation_figure.png in the project root.

import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

random.seed(42)
torch.manual_seed(42)

BASE     = Path(__file__).resolve().parent.parent.parent
IMG_PATH = BASE / 'data/raw/traffic_lights/4483547-uhd_2562_1440_30fps_f000690_t00023.0.jpg'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── helpers ──────────────────────────────────────────────────────────────
def to_t(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2,0,1).float().div(255).to(DEVICE)

def to_bgr(t):
    rgb = t.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── stronger augmentation functions ──────────────────────────────────────

def aug_rain(bgr):
    """Heavy rain: 2500 drops, long streaks, motion blur on whole image."""
    t = to_t(bgr)
    C, H, W = t.shape

    n = 2500
    rain = torch.zeros(1, H, W, device=DEVICE)
    xs = torch.randint(0, W, (n,), device=DEVICE)
    ys = torch.randint(0, H, (n,), device=DEVICE)
    rain[0, ys, xs] = torch.FloatTensor(n).uniform_(0.7, 1.0).to(DEVICE)

    L = 35
    kernel = torch.zeros(1, 1, L, 1, device=DEVICE)
    kernel[0, 0, :, 0] = 1.0 / L
    rain = F.conv2d(rain.unsqueeze(0), kernel, padding=(L//2, 0)).squeeze(0)[:, :H, :]

    result = torch.clamp(t * 0.62 + rain.expand(C,-1,-1) * 0.9, 0, 1)

    # Gaussian blur — rain makes everything less sharp
    out = to_bgr(result)
    out = cv2.GaussianBlur(out, (7, 7), 2.0)
    return out


def aug_fog(bgr):
    """Dense fog: heavy white overlay + slight Gaussian blur."""
    t = to_t(bgr)
    intensity = 0.6                               # 0 = clear, 1 = white-out
    result = torch.clamp(t * (1.0 - intensity) + intensity, 0, 1)
    out = to_bgr(result)
    out = cv2.GaussianBlur(out, (11, 11), 3.0)    # fog softens edges
    return out


def aug_night(bgr):
    """Night: strong gamma darkening with slight blue tint."""
    t = to_t(bgr)
    brightness = random.uniform(0.18, 0.28)
    out = t * brightness
    out[2] = torch.clamp(out[2] + 0.04, 0, 1)    # blue channel boost
    return to_bgr(out)


def aug_wet_road(bgr):
    """Wet road: heavy saturation boost + horizontal light reflections on lower half."""
    h, w = bgr.shape[:2]

    # Strong saturation + slight darkening in HSV space
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.2, 0, 255)   # strong saturation
    hsv[:, :, 2] = hsv[:, :, 2] * 0.78                     # darken (wet = less reflective sky)
    result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Horizontal puddle reflections on the lower third
    lower = result[h*2//3:].astype(np.float32)
    white = np.ones_like(lower) * 230.0
    for y in range(0, lower.shape[0], random.randint(6, 12)):
        alpha = random.uniform(0.08, 0.22)
        lower[y, :] = np.clip(lower[y,:] * (1-alpha) + white[y,:] * alpha, 0, 255)
    result[h*2//3:] = lower.astype(np.uint8)

    # Slight blur on lower half — water creates softness
    result[h*2//3:] = cv2.GaussianBlur(result[h*2//3:], (5, 5), 1.5)
    return result


# ── load and resize image ─────────────────────────────────────────────────
img = cv2.imread(str(IMG_PATH))
assert img is not None, f'Cannot load: {IMG_PATH}'
img = cv2.resize(img, (640, 360))

# ── apply all effects ─────────────────────────────────────────────────────
effects = [
    ('Original',  cv2.cvtColor(img, cv2.COLOR_BGR2RGB),            '#2c3e50'),
    ('Rain',      cv2.cvtColor(aug_rain(img), cv2.COLOR_BGR2RGB),  '#3498db'),
    ('Fog',       cv2.cvtColor(aug_fog(img),  cv2.COLOR_BGR2RGB),  '#7f8c8d'),
    ('Night',     cv2.cvtColor(aug_night(img),cv2.COLOR_BGR2RGB),  '#8e44ad'),
    ('Wet Road',  cv2.cvtColor(aug_wet_road(img),cv2.COLOR_BGR2RGB),'#1abc9c'),
]

SUBTITLES = {
    'Rain':     '1D Convolution Kernel + Blur',
    'Fog':      'Alpha Blending  (α = 0.6)',
    'Night':    'Gamma Point Operation',
    'Wet Road': 'HSV Saturation + Reflections',
}

# ── 2-row layout ──────────────────────────────────────────────────────────
# Row 1: Original | Rain | Fog
# Row 2: Original | Night | Wet Road
ROW1 = [effects[0], effects[1], effects[2]]   # Original, Rain, Fog
ROW2 = [effects[0], effects[3], effects[4]]   # Original, Night, Wet Road

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

for row_idx, row in enumerate([ROW1, ROW2]):
    for col_idx, (label, rgb, color) in enumerate(row):
        ax = axes[row_idx][col_idx]
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(label, fontsize=13, fontweight='bold', color=color, pad=6)
        if label in SUBTITLES:
            ax.text(0.5, -0.04, SUBTITLES[label],
                    transform=ax.transAxes,
                    ha='center', fontsize=9, color='#555555', style='italic')

plt.suptitle('Weather Augmentation — Image Processing Techniques',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

out_path = BASE / 'presentation/figures/06_weather_augmentation.png'
plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out_path}')
