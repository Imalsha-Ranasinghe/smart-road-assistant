"""
prepare_traffic_video.py

Tier-2 fine-tuning data prep: turn your traffic dashcam VIDEOS into labelled
training frames WITHOUT touching the existing dataset.

Pipeline:
  1. Extract sharp, de-duplicated frames from data/raw/traffic/video/*.mp4
  2. Auto-pseudo-label traffic lights with the COCO-pretrained YOLOv8s
     (COCO class 9 = 'traffic light'), remapped to our class 1.
  3. Stage them in data/raw/traffic/video_labeled/{images,labels} for you to
     CORRECT in Roboflow / LabelImg / CVAT before merging + fine-tuning.

The COCO model only PROPOSES boxes — correcting them (delete the black-rectangle
false positives, add missed lights, tighten boxes) is what makes the fine-tune
actually fix your video. Frames you blank out teach the model those shapes are
NOT lights (free hard negatives).

Run:
    python src/prepare_traffic_video.py
    python src/prepare_traffic_video.py --every 1.0 --max-per-video 400 --conf 0.25
    python src/prepare_traffic_video.py --source data/raw/traffic/video/video_traffic.mp4
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_DIR  = BASE_DIR / "data" / "raw" / "traffic" / "video"
FRAMES_DIR = BASE_DIR / "data" / "raw" / "traffic" / "video_frames"
OUT_DIR    = BASE_DIR / "data" / "raw" / "traffic" / "video_labeled"
COCO_MODEL = BASE_DIR / "notebooks" / "yolov8s.pt"   # COCO-pretrained: class 9 = traffic light

COCO_TL_CLASS = 9   # 'traffic light' in COCO
OUR_TL_CLASS  = 1   # our detector: pothole=0, traffic_light=1


def extract(source: Path, every: float, max_per_video, min_blur: float):
    cmd = [sys.executable, str(BASE_DIR / "src" / "extract_frames.py"),
           str(source), "-o", str(FRAMES_DIR),
           "--every", str(every), "--min-blur", str(min_blur)]
    if max_per_video:
        cmd += ["--max-per-video", str(max_per_video)]
    subprocess.run(cmd, check=True)


DETECTOR_MODEL = BASE_DIR / "models" / "detector_model.pt"   # your LISA-trained model


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xb - xa) * max(0, yb - ya)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _union_dedup(boxes, iou_thr=0.5):
    """Greedy NMS-style union: keep highest-conf box, drop overlaps."""
    boxes = sorted(boxes, key=lambda d: -d[4])   # by conf desc
    keep  = []
    for b in boxes:
        if all(_iou(b, k) <= iou_thr for k in keep):
            keep.append(b)
    return keep


def pseudolabel(conf: float, imgsz: int):
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    coco = YOLO(str(COCO_MODEL) if COCO_MODEL.exists() else "yolov8s.pt")
    ours = YOLO(str(DETECTOR_MODEL)) if DETECTOR_MODEL.exists() else None

    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    kept = lights = 0
    for f in frames:
        cands = []   # [x1, y1, x2, y2, conf]

        # COCO traffic lights (class 9)
        rc = coco(str(f), classes=[COCO_TL_CLASS], conf=conf, imgsz=imgsz, verbose=False)[0]
        for b in (rc.boxes or []):
            cands.append([*b.xyxy[0].tolist(), float(b.conf[0])])

        # Your model's traffic lights (class 1) — complementary recall
        if ours is not None:
            ro = ours(str(f), conf=conf, imgsz=imgsz, verbose=False)[0]
            for b in (ro.boxes or []):
                if int(b.cls[0]) == OUR_TL_CLASS:
                    cands.append([*b.xyxy[0].tolist(), float(b.conf[0])])

        merged = _union_dedup(cands)
        if not merged:
            continue

        H, W = rc.orig_shape
        lines = []
        for x1, y1, x2, y2, _c in merged:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{OUR_TL_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        shutil.copy2(f, OUT_DIR / "images" / f.name)
        (OUT_DIR / "labels" / f"{f.stem}.txt").write_text("\n".join(lines))
        kept   += 1
        lights += len(lines)

    # Class-name files for free labelling tools — order matters: 0=pothole, 1=traffic_light.
    names = "pothole\ntraffic_light\n"
    (OUT_DIR / "classes.txt").write_text(names)             # LabelImg / reference
    (OUT_DIR / "images" / "classes.txt").write_text(names)
    # makesense.ai's YOLO import requires a file literally named labels.txt sitting
    # in the labels folder next to the annotation .txt files (and NO other non-
    # annotation file there, or it tries to parse it as boxes).
    (OUT_DIR / "labels" / "labels.txt").write_text(names)
    return kept, lights, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(VIDEO_DIR),
                    help="Video file or folder (default: data/raw/traffic/video)")
    ap.add_argument("--every", type=float, default=1.0,
                    help="Seconds between sampled frames (default 1.0)")
    ap.add_argument("--max-per-video", type=int, default=400)
    ap.add_argument("--min-blur", type=float, default=60.0)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="COCO confidence for pseudo-labels (recall-first)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--skip-extract", action="store_true",
                    help="Re-label existing frames without re-extracting")
    a = ap.parse_args()

    if not a.skip_extract:
        print("1/2  Extracting frames...")
        extract(Path(a.source), a.every, a.max_per_video, a.min_blur)

    print("2/2  Pseudo-labelling traffic lights with COCO YOLOv8s...")
    kept, lights, total = pseudolabel(a.conf, a.imgsz)

    print(f"\n  {lights} traffic-light boxes proposed on {kept}/{total} frames")
    print(f"  Staged -> {OUT_DIR}")
    print("\n  NEXT:")
    print("   1. Open data/raw/traffic/video_labeled in Roboflow / LabelImg / CVAT")
    print("      and CORRECT the boxes (delete black-rectangle FPs, add missed lights).")
    print("   2. Merge into the existing dataset:  python src/merge_traffic_video.py")
    print("   3. Fine-tune:                         (see merge script output)")


if __name__ == "__main__":
    main()
