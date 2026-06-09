"""
merge_traffic_video.py

Merge your CORRECTED video frames (data/raw/traffic/video_labeled) into the
existing detector dataset (data/processed/detector_yolo) WITHOUT removing any of
the current 26k+ images. New frames are added to train/val with a 'vid_' prefix
so the merge is idempotent (re-running cleans the previous 'vid_' files first).

Because a few hundred video frames would otherwise be drowned by 26k existing
images, each new TRAIN frame is duplicated `--oversample` times so the model
actually adapts to your camera/lights. Validation frames are added once (no
oversampling — we want honest val metrics).

Run:
    python src/merge_traffic_video.py
    python src/merge_traffic_video.py --oversample 8 --val-frac 0.15
"""

import argparse
import random
import shutil
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
LABELED_DIR = BASE_DIR / "data" / "raw" / "traffic" / "video_labeled"
DATASET_DIR = BASE_DIR / "data" / "processed" / "detector_yolo"
PREFIX      = "vid_"
random.seed(42)


def _clean_previous():
    """Remove 'vid_' files from a prior merge so re-running is idempotent."""
    removed = 0
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = DATASET_DIR / sub / split
            if not d.exists():
                continue
            for p in d.glob(f"{PREFIX}*"):
                p.unlink()
                removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oversample", type=int, default=8,
                    help="Copies of each new TRAIN frame (default 8)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()

    img_dir = LABELED_DIR / "images"
    lbl_dir = LABELED_DIR / "labels"
    if not img_dir.exists():
        print(f"  No corrected frames at {LABELED_DIR}")
        print("  Run src/prepare_traffic_video.py and correct the labels first.")
        return

    pairs = [(p, lbl_dir / f"{p.stem}.txt")
             for p in sorted(img_dir.iterdir())
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    pairs = [(im, lb) for im, lb in pairs if lb.exists()]
    if not pairs:
        print("  No image/label pairs found — nothing to merge.")
        return

    removed = _clean_previous()
    if removed:
        print(f"  Cleaned {removed} files from a previous merge.")

    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * a.val_frac))
    val_set = set(range(n_val))

    added_train = added_val = 0
    for i, (im, lb) in enumerate(pairs):
        if i in val_set:                       # val: add once
            dst_im = DATASET_DIR / "images" / "val" / f"{PREFIX}{im.name}"
            dst_lb = DATASET_DIR / "labels" / "val" / f"{PREFIX}{im.stem}.txt"
            shutil.copy2(im, dst_im); shutil.copy2(lb, dst_lb)
            added_val += 1
        else:                                  # train: oversample
            for k in range(a.oversample):
                stem = f"{PREFIX}{im.stem}_o{k}"
                shutil.copy2(im, DATASET_DIR / "images" / "train" / f"{stem}{im.suffix}")
                shutil.copy2(lb, DATASET_DIR / "labels" / "train" / f"{stem}.txt")
                added_train += 1

    print(f"\n  Merged {len(pairs)} corrected frames into detector_yolo:")
    print(f"    +{added_train} train (x{a.oversample} oversample)   +{added_val} val")
    print(f"  Existing images untouched. Dataset yaml: {DATASET_DIR / 'dataset.yaml'}")
    print("\n  FINE-TUNE (continues from your current model, keeps pothole + LISA):")
    print("    yolo detect train \\")
    print(f"      model=models/detector_model.pt \\")
    print(f"      data={DATASET_DIR / 'dataset.yaml'} \\")
    print("      epochs=30 imgsz=1280 lr0=0.001 freeze=10 \\")
    print("      project=runs name=detector_traffic_ft exist_ok=True")
    print("\n  Then:  cp runs/detector_traffic_ft/weights/best.pt models/detector_model.pt")


if __name__ == "__main__":
    main()
