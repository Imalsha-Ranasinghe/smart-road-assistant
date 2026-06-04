"""
train_dashcam.py — prepare the dashcam dataset and train a SEPARATE detector
saved as models/detector_dashcam.pt. The photo model (models/detector_model.pt)
is never touched.

Classes (matches the current pipeline — colour is decided by the HSV analyzer):
    0 pothole
    1 traffic_light

Data sources:
  - pothole YOLO datasets in EXTERNAL_DATASETS            (dashcam viewpoint)
  - traffic-light YOLO datasets in TRAFFIC_DATASETS        (your far + new near sets;
    any colour classes are collapsed into one traffic_light class)
  - optional Bosch reuse (--bosch): data/raw/traffic/bosch
  - optional reuse of the already-prepared far traffic (--reuse-existing):
    pulls traffic_light boxes from data/processed/detector_yolo
  - optional your own corrected pothole frames (--frames): data/raw/dashcam_pseudo

Usage:
    python src/train_dashcam.py                          # pothole + traffic
    python src/train_dashcam.py --reuse-existing         # also reuse existing far traffic
    python src/train_dashcam.py --bosch                  # also fold in your Bosch subset
    python src/train_dashcam.py --no-traffic             # pothole-only (nc=1)
    python src/train_dashcam.py --prep-only              # build the dataset, don't train
    python src/train_dashcam.py --epochs 100 --imgsz 640
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
EXISTING_DET = BASE_DIR / "data" / "processed" / "detector_yolo"   # for --reuse-existing
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

# ── Pothole sources (road-level / dashcam YOLO datasets) ─────────────────────
EXTERNAL_DATASETS = [
    RAW / "pothole" / "0-No-dcs-aug.v1i.yolov8",       # dashcam view ✔ (multi-class)
    # RAW / "pothole" / "Pothole Peddal.v2i.yolov8",   # skipped: close-up, not dashcam
]
OWN_FRAMES = RAW / "dashcam_pseudo"                     # your corrected pothole frames

# ── Traffic-light sources (YOLO exports) — add your FAR and NEW NEAR sets here ─
# Download from Roboflow Universe into data/raw/traffic/ and list the folders.
# Any colour classes (red/yellow/green/off) are collapsed into one traffic_light class.
TRAFFIC_DATASETS = [
    # RAW / "traffic" / "<your-far-traffic-light>.yolov8",    # <-- your existing far set
    # RAW / "traffic" / "<your-near-traffic-light>.yolov8",   # <-- the new near set
]
BOSCH_DIR  = RAW / "traffic" / "bosch"                 # optional reuse (--bosch)
KAGGLE_DIR = RAW / "traffic" / "kaggle"                # NEAR lights, JSON anno (--kaggle)


# ── helpers ─────────────────────────────────────────────────────────────────

def _img_map(img_dir: Path) -> dict:
    if not img_dir.exists():
        return {}
    return {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}


def _pothole_index(ds: Path) -> int:
    yml = ds / "data.yaml"
    names = yaml.safe_load(yml.read_text()).get("names", []) if yml.exists() else []
    for i, n in enumerate(names):
        if "pothole" in str(n).lower():
            return i
    return 0


# ── collectors: each returns [(img_path, [yolo label lines]), ...] ───────────

def collect_potholes() -> list:
    items = []
    for ds in EXTERNAL_DATASETS:
        if not ds.exists():
            print(f"  !! skip missing pothole dataset: {ds}"); continue
        pidx = _pothole_index(ds); n = 0
        for split in ("train", "valid", "test"):
            lbl_dir = ds / split / "labels"
            imgs = _img_map(ds / split / "images")
            if not lbl_dir.exists():
                continue
            for lbl in lbl_dir.glob("*.txt"):
                img = imgs.get(lbl.stem)
                if img is None:
                    continue
                out = ["0 " + " ".join(l.split()[1:])
                       for l in lbl.read_text().splitlines()
                       if len(l.split()) == 5 and int(l.split()[0]) == pidx]
                if out:
                    items.append((img, out)); n += 1
        print(f"  {ds.name}: +{n} pothole images (pothole idx={pidx})")
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
    print(f"  own frames: +{len(items)} pothole images")
    return items


def collect_traffic_yolo() -> list:
    """Import traffic-light YOLO datasets. Every box becomes traffic_light (class 1)."""
    items = []
    for ds in TRAFFIC_DATASETS:
        if not ds.exists():
            print(f"  !! skip missing traffic dataset: {ds}"); continue
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
                out = ["1 " + " ".join(l.split()[1:])
                       for l in lbl.read_text().splitlines() if len(l.split()) == 5]
                if out:
                    items.append((img, out)); n += 1
        print(f"  {ds.name}: +{n} traffic images")
    return items


def collect_existing_traffic() -> list:
    """Reuse traffic_light (class 1) boxes from the already-prepared detector dataset."""
    items = []
    for split in ("train", "val"):
        lbl_dir = EXISTING_DET / "labels" / split
        imgs = _img_map(EXISTING_DET / "images" / split)
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            tl = [l for l in lbl.read_text().splitlines() if l.startswith("1 ")]
            img = imgs.get(lbl.stem)
            if tl and img:
                items.append((img, tl))
    print(f"  existing detector_yolo: +{len(items)} traffic images")
    return items


def collect_kaggle() -> list:
    """Parse the kaggle JSON (near intersection footage) → traffic_light (class 1).
    Each annotation's housing 'bndbox' becomes a box; colour (inbox) is ignored here."""
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
        img = root / fn.replace("\\", "/")            # 'train_images\\00001.jpg'
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


def collect_bosch() -> list:
    """Parse the Bosch yaml (only images present on disk) → traffic_light (class 1)."""
    from PIL import Image
    items = []
    if not BOSCH_DIR.exists():
        print("  bosch: dir missing"); return items
    n = 0
    for yml_name in ("train.yaml", "test.yaml"):
        yml = BOSCH_DIR / yml_name
        if not yml.exists():
            continue
        for e in yaml.safe_load(yml.read_text()) or []:
            rel = str(e.get("path", "")).lstrip("./").replace("rgb/", "", 1)
            img = BOSCH_DIR / rel
            boxes = e.get("boxes") or []
            if not boxes or not img.exists():
                continue
            try:
                W, H = Image.open(img).size
            except Exception:
                continue
            out = []
            for b in boxes:
                x1, x2 = float(b["x_min"]), float(b["x_max"])
                y1, y2 = float(b["y_min"]), float(b["y_max"])
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                if bw <= 0 or bh <= 0:
                    continue
                out.append(f"1 {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {bw:.6f} {bh:.6f}")
            if out:
                items.append((img, out)); n += 1
    print(f"  bosch: +{n} traffic images (present subset)")
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
        stem = f"{i:06d}_{img.stem[:36]}"
        shutil.copy2(img, DATASET_DIR / "images" / split / f"{stem}{img.suffix}")
        (DATASET_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    yaml_text = (
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\nval: images/val\n\n"
        f"nc: {len(classes)}\nnames: {classes}\n"
    )
    (DATASET_DIR / "dataset.yaml").write_text(yaml_text)
    print(f"\nDataset assembled: {len(items)} images "
          f"({len(items) - n_val} train / {n_val} val)")
    print(yaml_text)
    return DATASET_DIR / "dataset.yaml"


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train the dashcam detector (pothole + traffic_light).")
    ap.add_argument("--no-traffic", action="store_true", help="Pothole-only model (nc=1).")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="Also reuse traffic from data/processed/detector_yolo (your far set).")
    ap.add_argument("--kaggle", action="store_true",
                    help="Include the kaggle NEAR traffic-light set (data/raw/traffic/kaggle).")
    ap.add_argument("--bosch", action="store_true",
                    help="Also include the Bosch subset in data/raw/traffic/bosch.")
    ap.add_argument("--frames", action="store_true",
                    help="Also include your own labelled frames from data/raw/dashcam_pseudo.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--prep-only", action="store_true", help="Build the dataset, don't train.")
    args = ap.parse_args()

    use_traffic = not args.no_traffic
    classes = ["pothole", "traffic_light"] if use_traffic else ["pothole"]

    print("=" * 60)
    print(f"  Building dashcam dataset  (classes={classes})")
    print("=" * 60)
    items = collect_potholes()
    if args.frames:
        items += collect_own_frames()
    if use_traffic:
        items += collect_traffic_yolo()
        if args.kaggle:
            items += collect_kaggle()
        if args.reuse_existing:
            items += collect_existing_traffic()
        if args.bosch:
            items += collect_bosch()

    if not items:
        print("\nNo data collected — check EXTERNAL_DATASETS / TRAFFIC_DATASETS paths.")
        return

    # quick label histogram
    hist = {}
    for _, lines in items:
        for l in lines:
            hist[int(l.split()[0])] = hist.get(int(l.split()[0]), 0) + 1
    print("  box counts:", {classes[k]: v for k, v in sorted(hist.items()) if k < len(classes)})

    data_yaml = assemble(items, classes)

    if args.prep_only:
        print("Prep only — done. Re-run without --prep-only to train.")
        return

    # ── Train ──
    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    batch  = -1 if device != "cpu" else 16
    weights = BASE_DIR / "yolov8s.pt"

    print("=" * 60)
    print(f"  Training dashcam detector  (device={device})")
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
