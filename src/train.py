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
import multiprocessing
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / "data" / "processed"


# ── HPC auto-config ────────────────────────────────────

def _gpu_info():
    """Return (device_str, n_gpus, vram_gb)."""
    if not torch.cuda.is_available():
        return "cpu", 0, 0.0
    n   = torch.cuda.device_count()
    gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    dev = ",".join(str(i) for i in range(n)) if n > 1 else "0"
    return dev, n, gb


def _auto_batch(vram_gb: float, imgsz: int) -> int:
    """
    Pick a safe batch size from VRAM.

    VRAM budget rule: keep training footprint ≤ 75% of VRAM.
    YOLOv8s 640×640 AMP footprint per batch unit (empirical, includes
    activations + gradients + AdamW states + prefetch buffers):
      ~200 MB per 16-image increment at 640px

    Thresholds are intentionally conservative — OOM is a hard crash,
    slightly smaller batch is only a minor throughput loss.
    """
    table = {
        #       VRAM (GB) : batch
        640:  {8: 16, 16: 32, 24: 64, 36: 96, 48: 128},
        832:  {8:  8, 16: 16, 24: 32, 36: 48, 48:  64},
        1280: {8:  4, 16:  8, 24: 16, 36: 24, 48:  32},
    }
    limits = table.get(imgsz, table[640])
    for threshold in sorted(limits.keys()):
        if vram_gb <= threshold:
            return limits[threshold]
    return limits[max(limits)]


def _auto_workers(cached: bool = False) -> int:
    """
    Return the optimal number of dataloader worker threads.

    When NOT caching (large datasets), workers are the main I/O speed lever.
    We use more workers so the GPU never waits for the next batch.
    When caching to RAM, fewer workers are fine since data is already in memory.
    """
    cores = multiprocessing.cpu_count()
    if cached:
        return min(cores, 4)    # cache is in memory — few workers are enough
    return min(cores, 8)        # no cache: 8 is the sweet spot; beyond this prefetch
                                # queues consume too much RAM without speed gain


def _count_images(yaml_path: Path) -> int:
    """Count images in the train split of a YOLO dataset YAML."""
    try:
        import yaml as _yaml
        cfg      = _yaml.safe_load(yaml_path.read_text())
        base     = Path(cfg.get("path", yaml_path.parent))
        train    = cfg.get("train", "images/train")
        img_dir  = base / train if not Path(train).is_absolute() else Path(train)
        return sum(1 for _ in img_dir.iterdir() if _.suffix.lower() in (".jpg", ".jpeg", ".png"))
    except Exception:
        return 0


def _safe_cache(yaml_path: Path) -> str | bool:
    """
    Decide cache strategy from dataset size and available RAM.

    Rule:
      - decoded image size ≈ 1.2 MB per 640×640 image
      - only use RAM cache if dataset fits in ≤ 40% of available RAM
      - otherwise: no cache (parallel workers compensate)
    """
    n_images = _count_images(yaml_path)
    if n_images == 0:
        return False

    needed_gb = n_images * 1.2 / 1024          # GB of decoded images

    try:
        import psutil
        available_gb = psutil.virtual_memory().available / 1e9
        threshold_gb = available_gb * 0.40
    except ImportError:
        threshold_gb = 8.0                      # conservative default if psutil missing

    if needed_gb <= threshold_gb:
        print(f"  Cache: ram  ({n_images} images, ~{needed_gb:.1f} GB decoded, "
              f"{threshold_gb:.1f} GB available)")
        return "ram"
    else:
        print(f"  Cache: off  ({n_images} images would need ~{needed_gb:.1f} GB — "
              f"too large for RAM cache, using parallel workers instead)")
        return False


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


def _auto_fraction(n_images: int, target_steps: int = 220) -> float:
    """
    When a dataset is very large, sample a fraction each epoch so the
    per-epoch time stays manageable (~1.5 min at typical throughput).

    target_steps: desired steps per epoch at the auto-detected batch size.
    We use ~220 steps (~1.5 min on RTX 5090 with batch=128).

    The model still sees EVERY image — just spread across more epochs
    (random sampling without replacement per epoch).
    Returns a value in [0.25, 1.0].
    """
    n_images_per_epoch = target_steps * 128          # assume batch=128 for estimate
    fraction = n_images_per_epoch / max(n_images, 1)
    return round(max(0.25, min(1.0, fraction)), 2)


def train_detector(use_aug: bool = False, run_eval: bool = False,
                   imgsz: int = 640, fraction: float = None,
                   epochs_override: int = None, batch_override: int = None,
                   lr: float = 0.005):
    """
    fraction : float or None
        Fraction of dataset to sample per epoch (0.0–1.0).
        None → auto-choose: 1.0 for small datasets, ~0.35 for large ones.
        Useful when GPU is already at 96%+ — reduces epoch time without
        reducing the image diversity the model sees over full training.
    """
    config_path = get_detector_config(use_aug)
    if not config_path.exists():
        print(f"  Config not found: {config_path}")
        print("  Run: python src/prepare_data.py")
        if use_aug:
            print("  Then: python src/augment.py")
        return

    weights_path = BASE_DIR / "yolov8s.pt"
    weights_in   = str(weights_path) if weights_path.exists() else "yolov8s.pt"

    device, n_gpus, vram_gb = _gpu_info()
    batch   = batch_override if batch_override is not None else _auto_batch(vram_gb, imgsz)
    cache   = _safe_cache(config_path)
    workers = _auto_workers(cached=(cache == "ram"))

    # Auto-fraction: keep per-epoch time ~1.5 min on any dataset size
    n_images = _count_images(config_path)
    if fraction is None:
        fraction = _auto_fraction(n_images)

    # With fraction < 1, more epochs are needed to see the full dataset the
    # same number of times — scale up patience and epochs proportionally
    epoch_scale = max(1.0, 1.0 / fraction)
    epochs      = min(300, int(100 * epoch_scale))
    patience    = min(50,  int(20  * epoch_scale))

    # Manual override (e.g. --epochs 25 for a quick test run)
    if epochs_override is not None:
        epochs  = epochs_override
        patience = min(patience, epochs // 2)   # patience can't exceed half of epochs

    effective_per_epoch = int(n_images * fraction)
    est_mins = (effective_per_epoch / batch * 0.338) / 60   # 338ms/step observed

    suffix = " (augmented)" if use_aug else ""
    print(f"\n{'='*60}")
    print(f"  Training Stage 1: Detector{suffix}")
    print(f"  Config       : {config_path}")
    print(f"  Weights      : {weights_in}")
    print(f"  Device       : {device}  ({n_gpus} GPU(s), {vram_gb:.1f} GB VRAM)")
    print(f"  Dataset      : {n_images:,} images")
    print(f"  Fraction     : {fraction}  →  {effective_per_epoch:,} images/epoch")
    print(f"  Batch        : {batch}  |  ImgSz: {imgsz}  |  AMP: True")
    print(f"  Epochs       : {epochs}  |  Patience: {patience}")
    print(f"  Workers      : {workers}  |  Cache: {cache}")
    print(f"  Optimizer    : AdamW  |  lr0={lr}  |  LR schedule: cosine")
    print(f"  Est. per epoch: ~{est_mins:.1f} min")
    print(f"  Est. total    : ~{est_mins * min(epochs, 60):.0f} min "
          f"(assuming early stop ~ep{min(epochs, 60)})")
    print(f"{'='*60}\n")

    model   = YOLO(weights_in)
    results = model.train(
        data          = str(config_path),
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        fraction      = fraction,
        project       = str(BASE_DIR / "runs"),
        name          = "detector",
        patience      = patience,
        save          = True,
        plots         = True,
        exist_ok      = True,
        device        = device,
        cache         = cache,
        workers       = workers,
        amp           = True,
        optimizer     = "AdamW",
        cos_lr        = True,
        close_mosaic  = 10,
        warmup_epochs = 3,
        lr0           = lr,     # lower lr prevents EMA NaN/Inf with AMP + large batch
        lrf           = 0.01,   # final lr = lr0 * lrf
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
    print(f"  Epochs  : 50  |  Batch: 128  |  ImgSz: 128")
    print(f"  Classes : Low / Medium / High")

    device, n_gpus, vram_gb = _gpu_info()
    # Severity crops are tiny (~2.4k × 128px) — always safe to cache in RAM
    cache   = "ram"
    workers = _auto_workers(cached=True)

    print(f"  Device   : {device} ({n_gpus} GPU(s), {vram_gb:.1f} GB VRAM)")
    print(f"  Batch    : 128  |  Workers: {workers}  |  Cache: {cache}")
    print(f"{'='*55}\n")

    model   = YOLO(weights_in)
    results = model.train(
        data          = str(crops_dir),
        epochs        = 50,
        imgsz         = 128,
        batch         = 128,          # crops are tiny — large batch saturates GPU efficiently
        project       = str(BASE_DIR / "runs"),
        name          = "severity",
        patience      = 15,
        save          = True,
        plots         = True,
        exist_ok      = True,
        device        = device,
        cache         = cache,
        workers       = workers,
        amp           = True,
        optimizer     = "AdamW",
        cos_lr        = True,
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (e.g. 25 for a quick test run).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help=(
            "Initial learning rate (default: 0.005). "
            "Lower values prevent EMA NaN/Inf with AMP + large batch. "
            "Use 0.01 only if training is very slow to start."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=(
            "Override batch size (e.g. --batch 64 if auto-detected batch causes OOM). "
            "Default: auto from VRAM."
        ),
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help=(
            "Fraction of dataset to sample per epoch (0.0–1.0). "
            "Default: auto (keeps ~1.5 min/epoch). "
            "Use 1.0 to disable sampling and train on full dataset every epoch."
        ),
    )
    args = parser.parse_args()

    stages = ["detector", "severity"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "detector":
            train_detector(use_aug=args.aug, run_eval=args.eval,
                           fraction=args.fraction, epochs_override=args.epochs,
                           batch_override=args.batch, lr=args.lr)
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
