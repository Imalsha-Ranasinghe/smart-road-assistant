# ═══════════════════════════════════════════════════════
# src/train.py
# ═══════════════════════════════════════════════════════
"""
Train YOLO models for pothole and traffic light detection.

Run:
  python src/train.py                        # trains pothole (default)
  python src/train.py --model pothole        # pothole only
  python src/train.py --model traffic        # traffic only
  python src/train.py --model all            # both
  python src/train.py --model pothole --eval # train + evaluate
  python src/train.py --model all --aug      # use augmented datasets
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / "data" / "processed"

# ── Model configs ──────────────────────────────────────
def get_models(use_aug=False):
    pothole_data = (
        DATA_DIR / "pothole_yolo_aug" / "dataset.yaml"
        if use_aug else
        DATA_DIR / "pothole_yolo" / "dataset.yaml"
    )
    traffic_data = (
        DATA_DIR / "traffic_yolo_aug" / "dataset.yaml"
        if use_aug else
        DATA_DIR / "traffic_yolo" / "dataset.yaml"
    )

    return {
        "pothole": {
            "weights" : BASE_DIR / "yolov8s.pt",
            "config"  : pothole_data,
            "epochs"  : 100,
            "imgsz"   : 640,
            "batch"   : 16,
            "name"    : "pothole_model",
            "save_as" : MODELS_DIR / "pothole_model.pt",
        },
        "traffic": {
            "weights" : BASE_DIR / "yolov8s.pt",
            "config"  : traffic_data,
            "epochs"  : 100,
            "imgsz"   : 640,
            "batch"   : 16,
            "name"    : "traffic_model",
            "save_as" : MODELS_DIR / "traffic_model.pt",
        },
        
        "gate": {
        "weights" : BASE_DIR / "yolov8n.pt",    # nano — lightweight
        "config"  : DATA_DIR / "gate_yolo" / "dataset.yaml",
        "epochs"  : 50,
        "imgsz"   : 640,
        "batch"   : 16,
        "name"    : "gate_model",
        "save_as" : MODELS_DIR / "gate_model.pt",
        },
    }


# ── Augmented dataset.yaml generator ──────────────────
def ensure_aug_yaml(model_key: str):
    """
    The augmented folder only has train/. This creates a dataset.yaml
    for it that points val/test back to the original processed folder.
    """
    if model_key == "pothole":
        aug_dir  = DATA_DIR / "pothole_yolo_aug"
        orig_dir = DATA_DIR / "pothole_yolo"
        nc       = 1
        names    = ["pothole"]
    else:
        aug_dir  = DATA_DIR / "traffic_yolo_aug"
        orig_dir = DATA_DIR / "traffic_yolo"
        nc       = 3
        names    = ["Green", "Red", "Yellow"]

    yaml_path = aug_dir / "dataset.yaml"
    if yaml_path.exists():
        return  # already created

    aug_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {aug_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: {(orig_dir / 'images' / 'val').resolve()}\n"
        f"test: {(orig_dir / 'images' / 'test').resolve()}\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(content)
    print(f"  Generated aug yaml → {yaml_path}")


# ── Train ──────────────────────────────────────────────
def train(key: str, use_aug: bool = False):
    models = get_models(use_aug)
    cfg    = models.get(key)

    if cfg is None:
        print(f"  Unknown model: {key}")
        return

    # Generate aug yaml if needed
    if use_aug:
        ensure_aug_yaml(key)

    config_path = Path(cfg["config"])
    if not config_path.exists():
        print(f"\n  ❌ Config not found: {cfg['config']}")
        print(f"     Run prepare_data.py first.")
        if use_aug:
            print(f"     Then run augment.py.")
        return

    weights_path = Path(cfg["weights"])
    if not weights_path.exists():
        print(f"  ⚠️  {weights_path.name} not found in project root.")
        print(f"     Falling back to downloading yolov8s.pt from ultralytics...")
        weights_input = "yolov8s.pt"  # ultralytics will auto-download
    else:
        weights_input = str(weights_path)

    suffix = " (augmented)" if use_aug else ""
    print(f"\n{'='*55}")
    print(f"  Training  : {key}{suffix}")
    print(f"  Weights   : {cfg['weights']}")
    print(f"  Config    : {cfg['config']}")
    print(f"  Epochs    : {cfg['epochs']}  |  Batch: {cfg['batch']}")
    print(f"  Image size: {cfg['imgsz']}")
    print(f"{'='*55}\n")

    model   = YOLO(weights_input)
    results = model.train(
        data     = str(cfg["config"]),
        epochs   = cfg["epochs"],
        imgsz    = cfg["imgsz"],
        batch    = cfg["batch"],
        project  = str(BASE_DIR / "runs"),
        name     = cfg["name"],
        patience = 20,
        save     = True,
        plots    = True,
        exist_ok = True,   # overwrite previous run folder
    )

    # Copy best weights to models/
    best = BASE_DIR / "runs" / cfg["name"] / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(str(best), str(cfg["save_as"]))
        print(f"\n  ✅ Best weights saved → {cfg['save_as']}")
    else:
        print("  ⚠️  best.pt not found — check runs/ folder manually.")

    return results


# ── Evaluate ───────────────────────────────────────────
def evaluate(key: str, use_aug: bool = False):
    models     = get_models(use_aug)
    cfg        = models.get(key)
    model_path = Path(cfg["save_as"])

    if not model_path.exists():
        print(f"  ❌ Trained model not found: {model_path}")
        print(f"     Run training first: python src/train.py --model {key}")
        return

    # Always evaluate on original (non-augmented) test set
    orig_config = (
        DATA_DIR / "pothole_yolo" / "dataset.yaml"
        if key == "pothole" else
        DATA_DIR / "traffic_yolo" / "dataset.yaml"
    )

    print(f"\n{'='*55}")
    print(f"  Evaluating : {key}")
    print(f"  Model      : {model_path}")
    print(f"  Data       : {orig_config}")
    print(f"{'='*55}")

    model   = YOLO(str(model_path))
    metrics = model.val(data=str(orig_config))

    print(f"\n  ── Results ──────────────────────────")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print(f"  ─────────────────────────────────────")


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Road Assistant models")
    parser.add_argument(
        "--model",
        choices=["pothole", "traffic","gate", "all"],
        default="pothole",
        help="Which model to train (default: pothole)"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Use augmented dataset for training"
    )
    args = parser.parse_args()

    targets = ["pothole", "traffic"] if args.model == "all" else [args.model]

    for key in targets:
        train(key, use_aug=args.aug)
        if args.eval:
            evaluate(key, use_aug=args.aug)

    print(f"\n{'='*55}")
    print("  Training Summary")
    print(f"{'='*55}")
    models = get_models(args.aug)
    for key in targets:
        path   = Path(models[key]["save_as"])
        status = "✅ saved" if path.exists() else "❌ not found"
        print(f"  {key:10} → {status}  ({path.name})")
    print(f"{'='*55}")