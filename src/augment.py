"""
augment.py
Applies weather augmentations to the pothole training data.

Place this file at: src/augment.py
Run: python src/augment.py

PHASE 2: Uncomment traffic/gate sections when traffic data is available.
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ── Paths (matches prepare_pothole_dataset.py BASE_DIR logic) ─────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
POTHOLE_TRAIN = BASE_DIR / "data" / "processed" / "pothole_yolo" / "images" / "train"
POTHOLE_LBLS  = BASE_DIR / "data" / "processed" / "pothole_yolo" / "labels" / "train"

# PHASE 2 — uncomment when traffic data is ready
# TRAFFIC_TRAIN = BASE_DIR / "data" / "processed" / "traffic_yolo" / "images" / "train"
# TRAFFIC_LBLS  = BASE_DIR / "data" / "processed" / "traffic_yolo" / "labels" / "train"
# GATE_TRAIN    = BASE_DIR / "data" / "processed" / "gate_yolo"    / "images" / "train"
# GATE_LBLS     = BASE_DIR / "data" / "processed" / "gate_yolo"    / "labels" / "train"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Weather effect functions ───────────────────────────────────────────────────

def add_rain(image):
    out  = image.copy()
    h, w = out.shape[:2]
    for _ in range(600):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = max(0, min(w - 1, x1 + random.randint(-3, 3)))
        y2 = max(0, min(h - 1, y1 + random.randint(8, 20)))
        cv2.line(out, (x1, y1), (x2, y2), (180, 180, 200), 1)
    return cv2.addWeighted(image, 0.85, out, 0.15, 0)


def add_fog(image, intensity=0.45):
    fog   = np.full_like(image, 255, dtype=np.float32)
    blend = cv2.addWeighted(
        image.astype(np.float32), 1 - intensity,
        fog, intensity, 0
    )
    return np.clip(blend, 0, 255).astype(np.uint8)


def add_night(image):
    dark = cv2.convertScaleAbs(image, alpha=0.25, beta=0)
    dark[:, :, 0] = np.clip(dark[:, :, 0].astype(int) + 15, 0, 255)
    return dark


def add_wet_road(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


WEATHER_FNS = [add_rain, add_fog, add_night, add_wet_road]


def pick_weather(image):
    """Apply a random weather effect."""
    return random.choice(WEATHER_FNS)(image)


# ── Core augment function ──────────────────────────────────────────────────────

def augment_split(img_dir: Path, lbl_dir: Path, copies: int = 3):
    """
    For each image in img_dir, write `copies` weather-augmented versions.
    Labels are copied as-is — bounding boxes do not change with weather effects.
    """
    if not img_dir.exists():
        print(f"  Skipping (folder not found): {img_dir}")
        return

    img_files = [
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not img_files:
        print(f"  No images found in: {img_dir}")
        return

    added = 0
    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i in range(copies):
            aug      = pick_weather(img)
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            aug_lbl  = f"{img_path.stem}_aug{i}.txt"

            cv2.imwrite(str(img_dir / aug_name), aug)
            shutil.copy2(str(lbl_path), str(lbl_dir / aug_lbl))
            added += 1

    print(f"  Added {added} augmented images → {img_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print(" PHASE 1: Augmenting pothole training data")
    print("=" * 50)

    # Pothole gets 4 copies because dataset is small (~1200 images)
    # After augmentation you will have ~6000 training images
    print(f"\nPothole train folder : {POTHOLE_TRAIN}")
    augment_split(POTHOLE_TRAIN, POTHOLE_LBLS, copies=4)

    # ── PHASE 2: Uncomment when traffic data is ready ─────────────────────────
    # print("\nAugmenting traffic light training data...")
    # augment_split(TRAFFIC_TRAIN, TRAFFIC_LBLS, copies=2)
    #
    # print("\nAugmenting gate training data...")
    # augment_split(GATE_TRAIN, GATE_LBLS, copies=2)

    print("\n✅ Augmentation complete.")

    # Quick count check
    aug_count = len(list(POTHOLE_TRAIN.glob("*_aug*")))
    total     = len(list(POTHOLE_TRAIN.glob("*")))
    print(f"   Augmented images : {aug_count}")
    print(f"   Total train imgs : {total}")