"""
train.py
Train Stage 1 (detector) and/or Stage 2a (severity classifier).

Run:
    python src/train.py --stage detector          # Stage 1 only
    python src/train.py --stage severity          # Stage 2a only
    python src/train.py --stage all               # both
    python src/train.py --stage all --aug         # use augmented detector data
    python src/train.py --stage detector --eval   # train + evaluate
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / "data" / "processed"


# ── Stage 1: Detector ──────────────────────────────────

def get_detector_config(use_aug: bool):
    if use_aug:
        aug_yaml = DATA_DIR / "detector_yolo_aug" / "dataset.yaml"
        ensure_aug_yaml(aug_yaml)
        return aug_yaml
    return DATA_DIR / "detector_yolo" / "dataset.yaml"


def ensure_aug_yaml(yaml_path: Path):
    """
    The aug folder only has a train split.
    Create a dataset.yaml that borrows val/test from the original dataset.
    """
    if yaml_path.exists():
        return
    orig_dir = DATA_DIR / "detector_yolo"
    aug_dir  = yaml_path.parent
    aug_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {aug_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: {(orig_dir / 'images' / 'val').resolve()}\n"
        f"test: {(orig_dir / 'images' / 'test').resolve()}\n\n"
        f"nc: 2\n"
        f"names: ['pothole', 'traffic_light']\n"
    )
    yaml_path.write_text(content)
    print(f"  Generated aug yaml → {yaml_path}")


def train_detector(use_aug: bool = False, run_eval: bool = False):
    config_path = get_detector_config(use_aug)
    if not config_path.exists():
        print(f"  Config not found: {config_path}")
        print("  Run: python src/prepare_data.py")
        if use_aug:
            print("  Then: python src/augment.py")
        return

    weights_path = BASE_DIR / "yolov8s.pt"
    weights_in   = str(weights_path) if weights_path.exists() else "yolov8s.pt"

    suffix = " (augmented)" if use_aug else ""
    print(f"\n{'='*55}")
    print(f"  Training Stage 1: Detector{suffix}")
    print(f"  Config  : {config_path}")
    print(f"  Weights : {weights_in}")
    print(f"  Epochs  : 100  |  Batch: 16  |  ImgSz: 640")
    print(f"{'='*55}\n")

    model   = YOLO(weights_in)
    results = model.train(
        data     = str(config_path),
        epochs   = 100,
        imgsz    = 640,
        batch    = 16,
        project  = str(BASE_DIR / "runs"),
        name     = "detector",
        patience = 20,
        save     = True,
        plots    = True,
        exist_ok = True,
    )

    best = BASE_DIR / "runs" / "detector" / "weights" / "best.pt"
    save_as = MODELS_DIR / "detector_model.pt"
    if best.exists():
        shutil.copy2(str(best), str(save_as))
        print(f"\n  Detector saved → {save_as}")
    else:
        print("  best.pt not found — check runs/detector/weights/")

    if run_eval:
        evaluate_detector()

    return results


def evaluate_detector():
    model_path = MODELS_DIR / "detector_model.pt"
    if not model_path.exists():
        print("  Detector model not found — train first.")
        return
    config_path = DATA_DIR / "detector_yolo" / "dataset.yaml"
    print(f"\n{'='*55}")
    print(f"  Evaluating Stage 1 Detector")
    print(f"{'='*55}")
    model   = YOLO(str(model_path))
    metrics = model.val(data=str(config_path))
    print(f"\n  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")


# ── Stage 2a: Severity Classifier ──────────────────────

def train_severity(run_eval: bool = False):
    crops_dir = DATA_DIR / "severity_crops"
    if not crops_dir.exists():
        print("  Severity crops not found.")
        print("  Run: python src/prepare_data.py")
        return

    weights_path = BASE_DIR / "yolov8n-cls.pt"
    weights_in   = str(weights_path) if weights_path.exists() else "yolov8n-cls.pt"

    print(f"\n{'='*55}")
    print(f"  Training Stage 2a: Severity Classifier")
    print(f"  Data    : {crops_dir}")
    print(f"  Weights : {weights_in}")
    print(f"  Epochs  : 50  |  Batch: 32  |  ImgSz: 128")
    print(f"  Classes : Low / Medium / High")
    print(f"{'='*55}\n")

    model   = YOLO(weights_in)
    results = model.train(
        data     = str(crops_dir),
        epochs   = 50,
        imgsz    = 128,
        batch    = 32,
        project  = str(BASE_DIR / "runs"),
        name     = "severity",
        patience = 15,
        save     = True,
        plots    = True,
        exist_ok = True,
    )

    best = BASE_DIR / "runs" / "severity" / "weights" / "best.pt"
    save_as = MODELS_DIR / "severity_model.pt"
    if best.exists():
        shutil.copy2(str(best), str(save_as))
        print(f"\n  Severity classifier saved → {save_as}")
    else:
        print("  best.pt not found — check runs/severity/weights/")

    if run_eval:
        evaluate_severity()

    return results


def evaluate_severity():
    model_path = MODELS_DIR / "severity_model.pt"
    if not model_path.exists():
        print("  Severity model not found — train first.")
        return
    crops_dir = DATA_DIR / "severity_crops"
    print(f"\n{'='*55}")
    print(f"  Evaluating Stage 2a Severity Classifier")
    print(f"{'='*55}")
    model   = YOLO(str(model_path))
    metrics = model.val(data=str(crops_dir))
    print(f"\n  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")


# ── Main ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Road Assistant models")
    parser.add_argument(
        "--stage",
        choices=["detector", "severity", "all"],
        default="detector",
        help="Which stage to train (default: detector)",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Use augmented dataset for detector training",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after training",
    )
    args = parser.parse_args()

    stages = ["detector", "severity"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "detector":
            train_detector(use_aug=args.aug, run_eval=args.eval)
        elif stage == "severity":
            train_severity(run_eval=args.eval)

    print(f"\n{'='*55}")
    print("  Training Summary")
    print(f"{'='*55}")
    for stage in stages:
        if stage == "detector":
            p      = MODELS_DIR / "detector_model.pt"
            status = "saved" if p.exists() else "NOT FOUND"
            print(f"  detector  → {status}  ({p.name})")
        elif stage == "severity":
            p      = MODELS_DIR / "severity_model.pt"
            status = "saved" if p.exists() else "NOT FOUND"
            print(f"  severity  → {status}  ({p.name})")
    print(f"{'='*55}")
    if "severity" in stages:
        print("\n  Next step: python src/benchmark_severity.py")
