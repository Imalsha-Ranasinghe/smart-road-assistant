"""
extract_frames.py — turn dashcam videos into a set of diverse still images
ready for labelling / training a dashcam-perspective pothole detector.

It samples frames at a fixed time interval, then drops:
  - near-duplicate frames (perceptual-hash distance below a threshold), and
  - blurry frames (low Laplacian variance — common during bumps/motion).

Usage:
    # one video
    python src/extract_frames.py "clip.mp4" -o data/raw/dashcam_frames

    # a whole folder of videos, 1 frame every 2 seconds, max 400 per video
    python src/extract_frames.py ~/Downloads -o data/raw/dashcam_frames \
        --every 2.0 --max-per-video 400

The output folder can be uploaded straight to a labelling tool
(Roboflow / CVAT / LabelImg) — see notebooks/06-dashcam-detector.ipynb.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}


def _phash(gray: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash (8x8 DCT) as a boolean array."""
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    return dct > np.median(dct)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    every_sec: float = 1.0,
    max_per_video: int | None = None,
    dedup_dist: int = 6,
    min_blur: float = 60.0,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  !! cannot open {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(fps * every_sec)), 1)

    stem = video_path.stem.replace(" ", "_")[:40]
    kept, skipped_blur, skipped_dup = 0, 0, 0
    last_hashes: list[np.ndarray] = []
    fno = 0

    while True:
        if not cap.grab():
            break
        if fno % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if _blur_score(gray) < min_blur:
                    skipped_blur += 1
                else:
                    h = _phash(gray)
                    if any(_hamming(h, prev) <= dedup_dist for prev in last_hashes):
                        skipped_dup += 1
                    else:
                        last_hashes.append(h)
                        last_hashes = last_hashes[-30:]  # only compare against recent frames
                        t = fno / fps
                        name = f"{stem}_f{fno:06d}_t{t:07.1f}.jpg"
                        cv2.imwrite(str(out_dir / name), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                        kept += 1
                        if max_per_video and kept >= max_per_video:
                            break
        fno += 1

    cap.release()
    print(f"  {video_path.name}: kept {kept}  "
          f"(skipped {skipped_blur} blurry, {skipped_dup} duplicate, "
          f"of ~{total // step} sampled)")
    return kept


def main():
    ap = argparse.ArgumentParser(description="Extract diverse frames from dashcam videos.")
    ap.add_argument("source", help="A video file or a folder containing videos.")
    ap.add_argument("-o", "--out", default="data/raw/dashcam_frames",
                    help="Output folder (default: data/raw/dashcam_frames).")
    ap.add_argument("--every", type=float, default=1.0,
                    help="Seconds between sampled frames (default: 1.0).")
    ap.add_argument("--max-per-video", type=int, default=None,
                    help="Cap frames kept per video (default: no cap).")
    ap.add_argument("--dedup-dist", type=int, default=6,
                    help="pHash Hamming distance below which frames are 'duplicates' (default: 6).")
    ap.add_argument("--min-blur", type=float, default=60.0,
                    help="Min Laplacian variance to keep a frame (default: 60).")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    elif src.suffix.lower() in VIDEO_EXTS:
        videos = [src]
    else:
        print(f"No video(s) found at {src}")
        return

    if not videos:
        print(f"No videos with extensions {sorted(VIDEO_EXTS)} in {src}")
        return

    print(f"Extracting from {len(videos)} video(s) → {out_dir}/")
    print("-" * 60)
    total = 0
    for v in videos:
        total += extract_from_video(
            v, out_dir,
            every_sec=args.every,
            max_per_video=args.max_per_video,
            dedup_dist=args.dedup_dist,
            min_blur=args.min_blur,
        )
    print("-" * 60)
    print(f"Done. {total} frames written to {out_dir}/")
    print("Next: label them (see notebooks/06-dashcam-detector.ipynb).")


if __name__ == "__main__":
    main()
