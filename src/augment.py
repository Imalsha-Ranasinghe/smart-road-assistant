"""
augment.py
Weather-augment the detector_yolo training split.

Run:
    python src/augment.py
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_TRAIN_IMG = BASE_DIR / "data" / "processed" / "detector_yolo" / "images" / "train"
DETECTOR_TRAIN_LBL = BASE_DIR / "data" / "processed" / "detector_yolo" / "labels" / "train"

AUG_TRAIN_IMG = BASE_DIR / "data" / "processed" / "detector_yolo_aug" / "images" / "train"
AUG_TRAIN_LBL = BASE_DIR / "data" / "processed" / "detector_yolo_aug" / "labels" / "train"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# WEATHER EFFECTS
# ─────────────────────────────────────────────

def add_rain(image):
    out = image.copy()
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
    blend = cv2.addWeighted(image.astype(np.float32), 1 - intensity, fog, intensity, 0)
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
    img = image.copy()
    for fn in random.sample(WEATHER_FNS, k=random.randint(1, 2)):
        img = fn(img)
    return img


# ─────────────────────────────────────────────
# AUGMENT
# ─────────────────────────────────────────────

def augment_split(img_dir, lbl_dir, out_img_dir, out_lbl_dir, copies=3):
    if not img_dir.exists():
        print(f"  Skipping (not found): {img_dir}")
        return 0

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in img_dir.iterdir()
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    if not img_files:
        print(f"  No images in: {img_dir}")
        return 0

    added = 0
    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i in range(copies):
            aug_img      = pick_weather(img)
            new_img_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            new_lbl_name = f"{img_path.stem}_aug{i}.txt"
            cv2.imwrite(str(out_img_dir / new_img_name), aug_img)
            shutil.copy2(str(lbl_path), str(out_lbl_dir / new_lbl_name))
            added += 1

    return added


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AUGMENTING DETECTOR DATASET")
    print("=" * 55)

    added = augment_split(
        DETECTOR_TRAIN_IMG, DETECTOR_TRAIN_LBL,
        AUG_TRAIN_IMG, AUG_TRAIN_LBL,
        copies=3,
    )
    print(f"  Added {added} augmented images → {AUG_TRAIN_IMG}")
    print(f"  (3 weather copies per original training image)")
    print("\n  Augmentation complete!")
