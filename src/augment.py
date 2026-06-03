"""
augment.py
Weather-augment the detector_yolo training split.

GPU-accelerated via PyTorch tensor ops (already installed with ultralytics).
Parallel file I/O via multiprocessing to saturate NVMe + all i9 cores.

Run:
    python src/augment.py
"""

import cv2
import numpy as np
import random
import shutil
import multiprocessing as mp
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_TRAIN_IMG = BASE_DIR / "data" / "processed" / "detector_yolo" / "images" / "train"
DETECTOR_TRAIN_LBL = BASE_DIR / "data" / "processed" / "detector_yolo" / "labels" / "train"

AUG_TRAIN_IMG = BASE_DIR / "data" / "processed" / "detector_yolo_aug" / "images" / "train"
AUG_TRAIN_LBL = BASE_DIR / "data" / "processed" / "detector_yolo_aug" / "labels" / "train"

COPIES = 3
SEED   = 42

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# IMAGE ↔ TENSOR HELPERS
# ─────────────────────────────────────────────

def img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → float32 CHW tensor on DEVICE, values 0–1."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return t.to(DEVICE)


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """float32 CHW tensor → BGR uint8 HWC numpy array."""
    rgb = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# WEATHER EFFECTS  (all run on DEVICE)
# ─────────────────────────────────────────────

def add_rain(t: torch.Tensor) -> torch.Tensor:
    """
    Scatter bright streaks with a vertical motion-blur kernel.
    Much faster than 600 individual cv2.line() calls.
    """
    C, H, W = t.shape

    # Sparse rain dots
    n_drops = 700
    rain = torch.zeros(1, H, W, device=DEVICE)
    xs = torch.randint(0, W, (n_drops,), device=DEVICE)
    ys = torch.randint(0, H, (n_drops,), device=DEVICE)
    rain[0, ys, xs] = torch.FloatTensor(n_drops).uniform_(0.6, 1.0).to(DEVICE)

    # Vertical motion blur to turn dots into streaks
    streak_len = random.randint(10, 22)
    kernel     = torch.zeros(1, 1, streak_len, 1, device=DEVICE)
    kernel[0, 0, :, 0] = 1.0 / streak_len
    rain_blurred = F.conv2d(
        rain.unsqueeze(0), kernel,
        padding=(streak_len // 2, 0)
    ).squeeze(0)[:, :H, :]         # trim to original height

    rain_rgb = rain_blurred.expand(C, -1, -1) * 0.55
    return torch.clamp(t * 0.88 + rain_rgb, 0, 1)


def add_fog(t: torch.Tensor) -> torch.Tensor:
    intensity = random.uniform(0.35, 0.55)
    return torch.clamp(t * (1.0 - intensity) + intensity, 0, 1)


def add_night(t: torch.Tensor) -> torch.Tensor:
    brightness = random.uniform(0.18, 0.30)
    out        = t * brightness
    out[2]     = torch.clamp(out[2] + 0.04, 0, 1)   # slight blue tint
    return out


def add_wet_road(t: torch.Tensor) -> torch.Tensor:
    # torchvision funcs work on CPU tensors — move temporarily if on CUDA
    cpu_t  = t.cpu()
    cpu_t  = TF.adjust_saturation(cpu_t, random.uniform(1.2, 1.5))
    cpu_t  = TF.adjust_brightness(cpu_t, random.uniform(0.75, 0.90))
    return cpu_t.to(DEVICE)


WEATHER_FNS = [add_rain, add_fog, add_night, add_wet_road]


def pick_weather(t: torch.Tensor) -> torch.Tensor:
    """Apply 1–2 random weather effects to a tensor."""
    for fn in random.sample(WEATHER_FNS, k=random.randint(1, 2)):
        t = fn(t)
    return t


# ─────────────────────────────────────────────
# PER-IMAGE WORKER  (runs in child process)
# ─────────────────────────────────────────────

def _process_one(args):
    """
    Load one image, apply COPIES weather variants, save.
    Runs on CPU in a worker process; GPU ops run on main process batch instead.
    Returns number of files written.
    """
    img_path, lbl_path, out_img_dir, out_lbl_dir = args

    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    written = 0
    for i in range(COPIES):
        # Convert, augment (CPU fallback for worker), save
        t       = img_to_tensor(img)
        t       = pick_weather(t)
        aug_img = tensor_to_img(t)

        new_img = out_img_dir / f"{img_path.stem}_aug{i}{img_path.suffix}"
        new_lbl = out_lbl_dir / f"{img_path.stem}_aug{i}.txt"
        cv2.imwrite(str(new_img), aug_img)
        shutil.copy2(str(lbl_path), str(new_lbl))
        written += 1

    return written


# ─────────────────────────────────────────────
# MAIN AUGMENT FUNCTION
# ─────────────────────────────────────────────

def augment_split(img_dir, lbl_dir, out_img_dir, out_lbl_dir, copies=COPIES):
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

    total = len(img_files)
    print(f"  Source images  : {total}")
    print(f"  Copies each    : {copies}  →  {total * copies} augmented files")
    print(f"  Device         : {DEVICE}")

    # Build task list
    tasks = [
        (img_path, lbl_dir / f"{img_path.stem}.txt", out_img_dir, out_lbl_dir)
        for img_path in img_files
        if (lbl_dir / f"{img_path.stem}.txt").exists()
    ]

    # Number of worker processes — leave 2 cores for OS + main process
    n_workers = max(1, mp.cpu_count() - 2)
    print(f"  Workers        : {n_workers}  (of {mp.cpu_count()} logical cores)")
    print()

    added  = 0
    report = max(1, len(tasks) // 10)

    # Use multiprocessing pool for parallel I/O + CPU augmentation
    # (GPU tensor ops run per-worker on DEVICE; if CUDA, each worker
    #  creates its own context — for a dataset this size, per-worker
    #  CUDA is fastest without needing batching infrastructure)
    with mp.Pool(processes=n_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(_process_one, tasks), 1):
            added += result
            if idx % report == 0 or idx == len(tasks):
                pct = idx / len(tasks) * 100
                print(f"  [{idx:4d}/{len(tasks)}]  {pct:5.1f}%  —  {added} files written",
                      flush=True)

    return added


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 55)
    print("  AUGMENTING DETECTOR DATASET")
    print("=" * 55)

    added = augment_split(
        DETECTOR_TRAIN_IMG, DETECTOR_TRAIN_LBL,
        AUG_TRAIN_IMG, AUG_TRAIN_LBL,
        copies=COPIES,
    )

    print()
    print(f"  Added {added} augmented images")
    print(f"  Output → {AUG_TRAIN_IMG}")
    print("\n  Augmentation complete!")
