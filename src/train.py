"""
train.py
Trains YOLO models for pothole detection (Phase 1)
and traffic light + gate models (Phase 2).

Place this file at: src/train.py

Run:
  python src/train.py                      # trains pothole (default)
  python src/train.py --model pothole      # pothole only
  python src/train.py --model pothole --eval  # train + evaluate
  
PHASE 2 (when traffic data ready):
  python src/train.py --model traffic
  python src/train.py --model gate
  python src/train.py --model all
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Model configs ──────────────────────────────────────────────────────────────
MODELS = {

    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    "pothole": {
        "weights" : "yolov8s.pt",
        "config"  : str(BASE_DIR / "data" / "processed" / "pothole_yolo" / "dataset.yaml"),
        "epochs"  : 100,
        "imgsz"   : 640,
        "batch"   : 16,
        "name"    : "pothole_model",
        "save_as" : str(MODELS_DIR / "pothole_model.pt"),
    },

    # ── PHASE 2: Uncomment when traffic data is ready ─────────────────────────
    "traffic": {
        "weights" : "yolov8s.pt",
        "config"  : str(BASE_DIR / "data" / "processed" / "traffic_yolo" / "dataset.yaml"),
        "epochs"  : 100,
        "imgsz"   : 640,
        "batch"   : 16,
        "name"    : "traffic_model",
        "save_as" : str(MODELS_DIR / "traffic_model.pt"),
    },
    "gate": {
        "weights" : "yolov8n.pt",    # nano — lightweight gatekeeper
        "config"  : str(BASE_DIR / "data" / "processed" / "gate_yolo" / "dataset.yaml"),
        "epochs"  : 50,
        "imgsz"   : 640,
        "batch"   : 16,
        "name"    : "gate_model",
        "save_as" : str(MODELS_DIR / "gate_model.pt"),
    },
}

# ── Train ──────────────────────────────────────────────────────────────────────

def train(key: str):
    cfg = MODELS.get(key)
    if cfg is None:
        print(f"  Unknown model: {key}")
        return

    config_path = Path(cfg["config"])
    if not config_path.exists():
        print(f"\n  ❌ Config not found: {cfg['config']}")
        print(f"     Make sure you have run the data preparation script first.")
        return

    print(f"\n{'='*55}")
    print(f"  Training  : {key}")
    print(f"  Weights   : {cfg['weights']}")
    print(f"  Config    : {cfg['config']}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  Image size: {cfg['imgsz']}")
    print(f"{'='*55}\n")

    model   = YOLO(cfg["weights"])
    results = model.train(
        data      = cfg["config"],
        epochs    = cfg["epochs"],
        imgsz     = cfg["imgsz"],
        batch     = cfg["batch"],
        project   = str(BASE_DIR / "runs"),
        name      = cfg["name"],
        patience  = 20,       # early stop if no improvement for 20 epochs
        save      = True,
        plots     = True,
    )

    # Copy best weights to models/ folder
    best = BASE_DIR / "runs" / cfg["name"] / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(str(best), cfg["save_as"])
        print(f"\n  ✅ Best weights saved → {cfg['save_as']}")
    else:
        print("  ⚠️  best.pt not found. Check runs/ folder.")

    return results


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(key: str):
    cfg        = MODELS.get(key)
    model_path = Path(cfg["save_as"])

    if not model_path.exists():
        print(f"  ❌ Trained model not found: {model_path}")
        print(f"     Run training first.")
        return

    print(f"\n{'='*55}")
    print(f"  Evaluating: {key}")
    print(f"{'='*55}")

    model   = YOLO(str(model_path))
    metrics = model.val(data=cfg["config"])

    print(f"\n  Results:")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Road Assistant models")
    parser.add_argument(
        "--model",
        choices=["pothole", "traffic", "gate", "all"],
        default="pothole",
        help="Which model to train (default: pothole)"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after training"
    )
    args = parser.parse_args()

    targets = ["pothole", "traffic", "gate"] if args.model == "all" else [args.model]

    for key in targets:
        train(key)
        if args.eval:
            evaluate(key)

    # Summary
    print(f"\n{'='*55}")
    print("  Training Summary")
    print(f"{'='*55}")
    for key in targets:
        path   = Path(MODELS.get(key, {}).get("save_as", ""))
        status = "✅ saved" if path.exists() else "❌ not found"
        print(f"  {key:10} → {status}  ({path.name})")