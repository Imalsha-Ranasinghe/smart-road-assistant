"""
train_dashcam.py — prepare the dashcam pothole dataset and train a SEPARATE
detector saved as models/detector_dashcam.pt. The photo model
(models/detector_model.pt) is never touched.

Data sources (all class-filtered to a single pothole class, remapped to 0):
  - road-level YOLO datasets listed in EXTERNAL_DATASETS  (dashcam viewpoint)
  - optionally your own corrected frames in data/raw/dashcam_pseudo  (--frames)
  - optionally traffic_light reuse from the existing detector data (--traffic),
    which makes the result a 2-class drop-in so the Dashboard keeps detecting lights.

Usage:
    python src/train_dashcam.py                  # pothole + traffic (drop-in), 80 epochs
    python src/train_dashcam.py --no-traffic     # pothole-only model (nc=1)
    python src/train_dashcam.py --frames         # also include your own labelled frames
    python src/train_dashcam.py --prep-only      # build the dataset, don't train
    python src/train_dashcam.py --epochs 60 --imgsz 640
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

BASE_DIR     = Path(__file__).resolve().parent.parent
RAW          = BASE_DIR / "data" / "raw"
DATASET_DIR  = BASE_DIR / "data" / "processed" / "dashcam_yolo"
MODELS_DIR   = BASE_DIR / "models"
EXISTING_DET = BASE_DIR / "data" / "processed" / "detector_yolo"   # far traffic reuse
KAGGLE_DIR   = BASE_DIR / "data" / "raw" / "traffic" / "kaggle"     # near traffic (--kaggle)
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

# Road-level / dashcam pothole datasets to include (each: data.yaml + train/valid/test).
EXTERNAL_DATASETS = [
    RAW / "pothole" / "0-No-dcs-aug.v1i.yolov8",       # dashcam view ✔ (small/distant, bbox)
    RAW / "pothole" / "dashcam.pothole",               # large/close potholes (segmentation → bbox)
    # RAW / "pothole" / "Pothole Peddal.v2i.yolov8",   # close-up only; the above covers large
]
OWN_FRAMES = RAW / "dashcam_pseudo"   # your corrected pseudo-labels (Step 2b in the notebook)


# ── helpers ─────────────────────────────────────────────────────────────────

def _pothole_index(ds: Path) -> int:
    yml = ds / "data.yaml"
    names = yaml.safe_load(yml.read_text()).get("names", []) if yml.exists() else []
    for i, n in enumerate(names):
        if "pothole" in str(n).lower():
            return i
    return 0   # single-class fallback


def _img_map(img_dir: Path) -> dict:
    if not img_dir.exists():
        return {}
    return {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}


def collect_potholes() -> list:
    """[(img_path, [yolo lines as class 0]), ...] from EXTERNAL_DATASETS."""
    items = []
    for ds in EXTERNAL_DATASETS:
        if not ds.exists():
            print(f"  !! skip missing dataset: {ds}")
            continue
        pidx = _pothole_index(ds)
        n = 0
        for split in ("train", "valid", "test"):
            lbl_dir = ds / split / "labels"
            imgs = _img_map(ds / split / "images")
            if not lbl_dir.exists():
                continue
            for lbl in lbl_dir.glob("*.txt"):
                img = imgs.get(lbl.stem)
                if img is None:
                    continue
                out = []
                for l in lbl.read_text().splitlines():
                    p = l.split()
                    if len(p) < 5 or int(p[0]) != pidx:
                        continue
                    c = list(map(float, p[1:]))
                    if len(c) == 4:                              # bounding box
                        cx, cy, bw, bh = c
                    elif len(c) >= 6 and len(c) % 2 == 0:        # polygon → bbox
                        xs, ys = c[0::2], c[1::2]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    else:
                        continue
                    if bw > 0 and bh > 0:
                        out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if out:
                    items.append((img, out))
                    n += 1
        print(f"  {ds.name}: +{n} pothole images (pothole class idx={pidx})")
    return items


def collect_own_frames() -> list:
    items = []
    imgs = _img_map(OWN_FRAMES / "images")
    lbl_dir = OWN_FRAMES / "labels"
    if lbl_dir.exists():
        for lbl in lbl_dir.glob("*.txt"):
            img = imgs.get(lbl.stem)
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            if img and lines:
                items.append((img, ["0 " + " ".join(l.split()[1:]) for l in lines]))
    print(f"  own frames: +{len(items)} images")
    return items


def collect_traffic(limit: int) -> list:
    """Reuse traffic_light (class 1) images from the existing detector dataset."""
    items = []
    for split in ("train", "val"):
        lbl_dir = EXISTING_DET / "labels" / split
        imgs = _img_map(EXISTING_DET / "images" / split)
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            tl = [l for l in lbl.read_text().splitlines() if l.startswith("1 ")]
            if not tl:
                continue
            img = imgs.get(lbl.stem)
            if img:
                items.append((img, tl))   # keep class 1 = traffic_light
    random.shuffle(items)
    if limit and len(items) > limit:
        items = items[:limit]
    print(f"  traffic reuse: +{len(items)} images")
    return items


def collect_kaggle() -> list:
    """Parse the kaggle JSON (near intersection footage) → traffic_light (class 1)."""
    import json
    from collections import defaultdict
    from PIL import Image
    root = KAGGLE_DIR / "train_dataset"
    jf = root / "train.json"
    if not jf.exists():
        print("  kaggle: train.json missing"); return []
    by_file = defaultdict(list)
    for a in json.load(jf.open()).get("annotations", []):
        b = a.get("bndbox")
        if b:
            by_file[a["filename"]].append(b)
    items = []
    for fn, boxes in by_file.items():
        img = root / fn.replace("\\", "/")
        if not img.exists():
            continue
        try:
            W, H = Image.open(img).size
        except Exception:
            continue
        out = []
        for b in boxes:
            x1, y1, x2, y2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            out.append(f"1 {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {bw:.6f} {bh:.6f}")
        if out:
            items.append((img, out))
    print(f"  kaggle: +{len(items)} traffic images (near)")
    return items


def assemble(items: list, classes: list, val_frac: float = 0.15):
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.shuffle(items)
    n_val = max(1, int(len(items) * val_frac))
    for i, (img, lines) in enumerate(items):
        split = "val" if i < n_val else "train"
        stem = f"{i:06d}_{img.stem[:40]}"
        shutil.copy2(img, DATASET_DIR / "images" / split / f"{stem}{img.suffix}")
        (DATASET_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    yaml_text = (
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    (DATASET_DIR / "dataset.yaml").write_text(yaml_text)
    print(f"\nDataset assembled: {len(items)} images "
          f"({len(items) - n_val} train / {n_val} val)")
    print(yaml_text)
    return DATASET_DIR / "dataset.yaml"


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train the dashcam pothole detector.")
    ap.add_argument("--no-traffic", action="store_true",
                    help="Pothole-only model (nc=1). Default reuses traffic_light for a drop-in.")
    ap.add_argument("--frames", action="store_true",
                    help="Also include your own labelled frames from data/raw/dashcam_pseudo.")
    ap.add_argument("--traffic-limit", type=int, default=3000,
                    help="Cap reused far traffic_light images (balance vs potholes). Default 3000.")
    ap.add_argument("--kaggle", action="store_true",
                    help="Include the kaggle NEAR traffic-light set (data/raw/traffic/kaggle).")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--prep-only", action="store_true",
                    help="Build the dataset and stop (no training).")
    args = ap.parse_args()

    use_traffic = not args.no_traffic
    classes = ["pothole", "traffic_light"] if use_traffic else ["pothole"]

    print("=" * 60)
    print("  Building dashcam dataset")
    print("=" * 60)
    items = collect_potholes()
    if args.frames:
        items += collect_own_frames()
    if use_traffic:
        items += collect_traffic(args.traffic_limit)
        if args.kaggle:
            items += collect_kaggle()

    if not items:
        print("\nNo data collected — check EXTERNAL_DATASETS paths.")
        return

    data_yaml = assemble(items, classes)

    if args.prep_only:
        print("Prep only — done. Run again without --prep-only to train.")
        return

    # ── Train ──
    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    batch  = -1 if device != "cpu" else 16
    weights = BASE_DIR / "yolov8s.pt"

    print("=" * 60)
    print(f"  Training dashcam detector  (device={device}, classes={classes})")
    print("=" * 60)

    model = YOLO(str(weights) if weights.exists() else "yolov8s.pt")
    model.train(
        data     = str(data_yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = batch,
        project  = str(BASE_DIR / "runs"),
        name     = "detector_dashcam",
        exist_ok = True,
        device   = device,
        optimizer= "AdamW",
        lr0      = 0.005,
        cos_lr   = True,
        patience = 20,
        plots    = True,
    )

    best = BASE_DIR / "runs" / "detector_dashcam" / "weights" / "best.pt"
    out  = MODELS_DIR / "detector_dashcam.pt"
    if best.exists():
        shutil.copy2(best, out)
        print(f"\n  Dashcam detector saved → {out}")
        print("  Restart the backend; the Dashboard will use it automatically.")
    else:
        print("  best.pt not found — check runs/detector_dashcam/weights/")


if __name__ == "__main__":
    main()
