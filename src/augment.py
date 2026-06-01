"""
augment.py
Create separate augmented datasets for pothole and traffic.

Run:
    python src/augment.py
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Original datasets
POTHOLE_TRAIN = BASE_DIR / "data" / "processed" / "pothole_yolo" / "images" / "train"
POTHOLE_LBLS  = BASE_DIR / "data" / "processed" / "pothole_yolo" / "labels" / "train"

TRAFFIC_TRAIN = BASE_DIR / "data" / "processed" / "traffic_yolo" / "images" / "train"
TRAFFIC_LBLS  = BASE_DIR / "data" / "processed" / "traffic_yolo" / "labels" / "train"

# Augmented output folders (SEPARATE)
POTHOLE_AUG_IMG = BASE_DIR / "data" / "processed" / "pothole_yolo_aug" / "images" / "train"
POTHOLE_AUG_LBL = BASE_DIR / "data" / "processed" / "pothole_yolo_aug" / "labels" / "train"

TRAFFIC_AUG_IMG = BASE_DIR / "data" / "processed" / "traffic_yolo_aug" / "images" / "train"
TRAFFIC_AUG_LBL = BASE_DIR / "data" / "processed" / "traffic_yolo_aug" / "labels" / "train"

# Seed
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
    fog = np.full_like(image, 255, dtype=np.float32)
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
    """Apply 1–2 random weather effects"""
    img = image.copy()
    for fn in random.sample(WEATHER_FNS, k=random.randint(1, 2)):
        img = fn(img)
    return img


# ─────────────────────────────────────────────
# AUGMENT FUNCTION
# ─────────────────────────────────────────────

def augment_split(img_dir, lbl_dir, out_img_dir, out_lbl_dir, copies=2):
    if not img_dir.exists():
        print(f"Skipping: {img_dir}")
        return

    # Create output directories
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = [
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not img_files:
        print(f"No images found in: {img_dir}")
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
            aug_img = pick_weather(img)

            new_img_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            new_lbl_name = f"{img_path.stem}_aug{i}.txt"

            cv2.imwrite(str(out_img_dir / new_img_name), aug_img)
            shutil.copy2(str(lbl_path), str(out_lbl_dir / new_lbl_name))

            added += 1

    print(f"Added {added} augmented images → {out_img_dir}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print(" AUGMENTING DATASETS (SEPARATE OUTPUT)")
    print("=" * 50)

    # Pothole augmentation
    print("\n🔹 Pothole Augmentation")
    augment_split(
        POTHOLE_TRAIN,
        POTHOLE_LBLS,
        POTHOLE_AUG_IMG,
        POTHOLE_AUG_LBL,
        copies=4
    )

    # Traffic augmentation
    print("\n🔹 Traffic Augmentation")
    augment_split(
        TRAFFIC_TRAIN,
        TRAFFIC_LBLS,
        TRAFFIC_AUG_IMG,
        TRAFFIC_AUG_LBL,
        copies=2
    )

    print("\n✅ Augmentation complete!")

    # Quick stats
    pothole_count = len(list(POTHOLE_AUG_IMG.glob("*")))
    traffic_count = len(list(TRAFFIC_AUG_IMG.glob("*")))

    print(f"\n📊 Summary:")
    print(f"   Pothole augmented images : {pothole_count}")
    print(f"   Traffic augmented images : {traffic_count}")