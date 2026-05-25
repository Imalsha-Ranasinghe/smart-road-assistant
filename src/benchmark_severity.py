"""
benchmark_severity.py

Compare Classical CV vs Fine-tuned CNN for pothole severity classification
on the held-out test set (severity_crops/test/).

Writes the winning method to configs/pipeline_config.yaml.

Run:
    python src/benchmark_severity.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR      = Path(__file__).resolve().parent.parent
SEVERITY_TEST = BASE_DIR / "data" / "processed" / "severity_crops" / "test"
MODEL_PATH    = BASE_DIR / "models" / "severity_model.pt"
CONFIG_PATH   = BASE_DIR / "configs" / "pipeline_config.yaml"
RESULTS_PATH  = BASE_DIR / "benchmark_results.json"

CLASSES = ["Low", "Medium", "High"]   # folder names (alphabetical for CNN)

# CNN returns classes alphabetically: High(0), Low(1), Medium(2)
CNN_IDX_TO_LABEL = {0: "High", 1: "Low", 2: "Medium"}


# ── Load test images ──────────────────────────────────

def load_test_set():
    """Return list of (image_bgr, true_label)."""
    items = []
    if not SEVERITY_TEST.exists():
        print(f"  Test set not found: {SEVERITY_TEST}")
        print("  Run: python src/prepare_data.py")
        sys.exit(1)
    for cls_dir in sorted(SEVERITY_TEST.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        if label not in CLASSES:
            continue
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is not None:
                items.append((img, label))
    return items


# ── Classical CV ──────────────────────────────────────

def cv_predict(crop: np.ndarray) -> str:
    from analyzers.pothole_analyzer import _compute_cv_scores, _cv_severity
    scores = _compute_cv_scores(crop)
    return _cv_severity(scores)


# ── CNN ───────────────────────────────────────────────

def load_cnn():
    if not MODEL_PATH.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"  Could not load CNN model: {e}")
        return None


def cnn_predict(model, crop: np.ndarray) -> str:
    results  = model(crop, imgsz=128, verbose=False)
    top1_idx = int(results[0].probs.top1)
    return CNN_IDX_TO_LABEL.get(top1_idx, "Low")


# ── Metrics ───────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    per_class = {}
    for cls in CLASSES:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }
    return {"accuracy": round(accuracy, 4), "per_class": per_class}


def print_metrics(name: str, metrics: dict):
    print(f"\n  ── {name} ──────────────────────────")
    print(f"  Overall Accuracy : {metrics['accuracy']:.2%}")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<10} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")


# ── Update pipeline config ────────────────────────────

def update_pipeline_config(winning_method: str):
    CONFIG_PATH.parent.mkdir(exist_ok=True)

    # Read existing config or start fresh
    if CONFIG_PATH.exists():
        lines = CONFIG_PATH.read_text().splitlines()
        new_lines = []
        replaced = False
        for line in lines:
            if line.startswith("severity_method:"):
                new_lines.append(f"severity_method: {winning_method}")
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(f"severity_method: {winning_method}")
        CONFIG_PATH.write_text("\n".join(new_lines) + "\n")
    else:
        CONFIG_PATH.write_text(
            f"severity_method: {winning_method}\n"
            f"detector_conf: 0.40\n"
            f"nms_iou: 0.45\n"
            f"lane_x_min: 0.20\n"
            f"lane_x_max: 0.80\n"
            f"lane_min_size_ratio: 0.001\n"
            f"tl_min_aspect_ratio: 1.2\n"
        )

    print(f"\n  Written to {CONFIG_PATH}:")
    print(f"    severity_method: {winning_method}")


# ── Main ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SEVERITY BENCHMARK: CV vs CNN")
    print("=" * 55)

    test_set = load_test_set()
    print(f"  Test set: {len(test_set)} crops  "
          f"({sum(1 for _,l in test_set if l=='Low')} Low / "
          f"{sum(1 for _,l in test_set if l=='Medium')} Medium / "
          f"{sum(1 for _,l in test_set if l=='High')} High)")

    y_true = [label for _, label in test_set]

    # ── Run Classical CV ──────────────────────────────
    print("\n  Running Classical CV...")
    cv_preds = [cv_predict(img) for img, _ in test_set]
    cv_metrics = compute_metrics(y_true, cv_preds)
    print_metrics("Classical CV", cv_metrics)

    # ── Run CNN ───────────────────────────────────────
    cnn_model   = load_cnn()
    cnn_metrics = None
    cnn_preds   = None
    if cnn_model is not None:
        print("\n  Running CNN...")
        cnn_preds   = [cnn_predict(cnn_model, img) for img, _ in test_set]
        cnn_metrics = compute_metrics(y_true, cnn_preds)
        print_metrics("Fine-tuned CNN", cnn_metrics)
    else:
        print("\n  CNN model not found — skipping CNN evaluation.")
        print(f"  (expected at {MODEL_PATH})")

    # ── Pick winner ───────────────────────────────────
    print("\n" + "=" * 55)
    if cnn_metrics is not None:
        if cnn_metrics["accuracy"] >= cv_metrics["accuracy"]:
            winner = "cnn"
            print(f"  WINNER: CNN  ({cnn_metrics['accuracy']:.2%} vs {cv_metrics['accuracy']:.2%})")
        else:
            winner = "cv"
            print(f"  WINNER: CV   ({cv_metrics['accuracy']:.2%} vs {cnn_metrics['accuracy']:.2%})")
    else:
        winner = "cv"
        print(f"  Using: CV (CNN model not available)")

    # ── Save results ──────────────────────────────────
    results = {
        "test_size": len(test_set),
        "cv":        cv_metrics,
        "cnn":       cnn_metrics,
        "winner":    winner,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"  Results saved → {RESULTS_PATH}")

    # ── Update config ─────────────────────────────────
    update_pipeline_config(winner)
    print("=" * 55)
    print(f"\n  Pipeline will use: {winner.upper()} method for severity analysis.")
    print("  Next: python backend/app.py")


if __name__ == "__main__":
    # Add src/ to path so analyzers can be imported
    sys.path.insert(0, str(BASE_DIR / "src"))
    main()
