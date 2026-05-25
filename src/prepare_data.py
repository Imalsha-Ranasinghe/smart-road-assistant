"""
prepare_data.py

Prepares two datasets:
  1. detector_yolo/  — 2-class YOLO dataset (pothole + traffic_light) for Stage 1
  2. severity_crops/ — ImageFolder dataset (Low/Medium/High) for Stage 2a severity classifier

Run:
    python src/prepare_data.py
"""

import csv
import glob
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw inputs
POTHOLE_RAW_DIR = BASE_DIR / "data" / "raw" / "pothole"
TRAFFIC_RAW_DIR = BASE_DIR / "data" / "raw" / "traffic"

DS1_DIR   = POTHOLE_RAW_DIR / "annotated-pothole-images-chitholian"
DS2_DIR   = POTHOLE_RAW_DIR / "pothole-kaggle-dataset"
LISA_DIR  = TRAFFIC_RAW_DIR / "lisa"
BOSCH_DIR = TRAFFIC_RAW_DIR / "bosch"

# Processed outputs
DETECTOR_OUT_DIR = BASE_DIR / "data" / "processed" / "detector_yolo"
SEVERITY_OUT_DIR = BASE_DIR / "data" / "processed" / "severity_crops"

# ─────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────
DETECTOR_CLASSES     = {"pothole": 0, "traffic_light": 1}
DETECTOR_CLASS_NAMES = ["pothole", "traffic_light"]
DETECTOR_COLOR_MAP   = {0: (0, 0, 255), 1: (0, 200, 0)}   # Red=pothole, Green=TL

SEVERITY_CLASSES = ["Low", "Medium", "High"]

# Severity auto-label thresholds (box_area / image_area)
SEVERITY_LOW_MAX    = 0.03
SEVERITY_MEDIUM_MAX = 0.12
# >= 0.12 → High

# Pothole crop padding (pixels)
CROP_PADDING = 10

# ─────────────────────────────────────────────
# SPLIT RATIOS
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

KNOWN_LISA_SETS = {
    "sample-dayClip6", "sample-nightClip1",
    "dayTrain", "nightTrain",
    "daySequence1", "daySequence2",
    "nightSequence1", "nightSequence2",
}


# ═════════════════════════════════════════════
# SHARED UTILITIES
# ═════════════════════════════════════════════

def convert_bbox_to_yolo(size, box):
    """size=(w,h), box=(xmin, xmax, ymin, ymax) → (xc, yc, w, h) normalised."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def get_image_size(image_path: Path):
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to load image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def prepare_directories(out_dir: Path, splits=("train", "val", "test")):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def prepare_imagefolder(out_dir: Path, classes, splits=("train", "val", "test")):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in splits:
        for cls in classes:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)


def split_dataset(all_data):
    random.seed(RANDOM_SEED)
    random.shuffle(all_data)
    total     = len(all_data)
    train_end = int(total * TRAIN_RATIO)
    val_end   = train_end + int(total * VAL_RATIO)
    return all_data[:train_end], all_data[train_end:val_end], all_data[val_end:]


def build_image_index(root: Path):
    index = defaultdict(list)
    for ext in IMAGE_EXTENSIONS:
        for img_path in root.rglob(f"*{ext}"):
            index[img_path.name.lower()].append(img_path)
    return index


def draw_and_save_preview(img_path: Path, labels_raw, out_dir: Path,
                          class_names: list, color_map: dict):
    import cv2
    image = cv2.imread(str(img_path))
    if image is None:
        return
    h, w = image.shape[:2]
    for cls_id, xc, yc, bw, bh in labels_raw:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = color_map.get(cls_id, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(image, label_text, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), image)


# ═════════════════════════════════════════════
# POTHOLE PARSING (Pascal VOC XML)
# ═════════════════════════════════════════════

def parse_pothole_xml(xml_file):
    """Returns (img_filename, [(xmin, xmax, ymin, ymax), ...], img_w, img_h)."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename_attr = root.find("filename")
    img_filename  = filename_attr.text if filename_attr is not None else None

    size  = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.iter("object"):
        cls_name = obj.find("name").text.lower().strip()
        if cls_name != "pothole":
            continue
        xmlbox = obj.find("bndbox")
        boxes.append((
            float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text),
        ))
    return img_filename, boxes, img_w, img_h


def gather_pothole_chitholian():
    data = []
    for xml_path in map(Path, glob.glob(str(DS1_DIR / "*.xml"))):
        _, boxes, img_w, img_h = parse_pothole_xml(xml_path)
        img_path = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            print(f"  Warning: Image missing for {xml_path.name}")
            continue
        data.append({
            "img_path": img_path, "boxes": boxes,
            "img_w": img_w, "img_h": img_h, "prefix": "chitholian_",
        })
    return data


def gather_pothole_kaggle():
    data = []
    annotations_dir = DS2_DIR / "annotations"
    images_dir      = DS2_DIR / "images"
    for xml_path in map(Path, glob.glob(str(annotations_dir / "*.xml"))):
        img_filename, boxes, img_w, img_h = parse_pothole_xml(xml_path)
        img_path = images_dir / img_filename
        if not img_path.exists():
            print(f"  Warning: Image missing for {xml_path.name}")
            continue
        data.append({
            "img_path": img_path, "boxes": boxes,
            "img_w": img_w, "img_h": img_h, "prefix": "kaggle_",
        })
    return data


# ═════════════════════════════════════════════
# TRAFFIC PARSING (LISA CSV + Bosch YAML)
# ═════════════════════════════════════════════

def normalize_traffic_label(raw_label: str):
    label = raw_label.strip().lower()
    if label.startswith("go") or label.startswith("green"):
        return "traffic_light"
    if label.startswith("stop") or label.startswith("red"):
        return "traffic_light"
    if label.startswith("warning") or label.startswith("yellow"):
        return "traffic_light"
    return None


def identify_lisa_set(csv_path: Path):
    for part in csv_path.parts:
        if part in KNOWN_LISA_SETS:
            return part
    return None


def resolve_lisa_image_path(filename: str, csv_path: Path, image_index):
    base_name  = Path(filename).name.lower()
    candidates = image_index.get(base_name, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    lisa_set = identify_lisa_set(csv_path)
    if lisa_set:
        filtered = [p for p in candidates if lisa_set in p.parts]
        if len(filtered) == 1:
            return filtered[0]
    return sorted(candidates)[0]


def parse_lisa_annotations():
    image_index = build_image_index(LISA_DIR)
    records     = {}
    missing     = 0

    for csv_path in LISA_DIR.rglob("frameAnnotationsBOX.csv"):
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                raw_name  = row.get("Filename") or row.get("filename")
                raw_label = row.get("Annotation tag") or row.get("Annotation tag ")
                if not raw_name or not raw_label:
                    continue
                label_name = normalize_traffic_label(raw_label)
                if label_name is None:
                    continue
                try:
                    x_min = float(row["Upper left corner X"])
                    y_min = float(row["Upper left corner Y"])
                    x_max = float(row["Lower right corner X"])
                    y_max = float(row["Lower right corner Y"])
                except Exception:
                    continue
                img_path = resolve_lisa_image_path(raw_name, csv_path, image_index)
                if img_path is None or not img_path.exists():
                    missing += 1
                    continue
                key = str(img_path.resolve())
                if key not in records:
                    records[key] = {"img_path": img_path, "boxes_raw": [], "prefix": "lisa_"}
                records[key]["boxes_raw"].append((x_min, y_min, x_max, y_max))

    if missing:
        print(f"  LISA: {missing} annotations skipped (image not found)")
    data = list(records.values())
    print(f"  LISA dataset: {len(data)} images found.")
    return data


def parse_bosch_yaml_file(yaml_path: Path):
    records = []
    current = {"path": None, "boxes": []}

    def flush():
        if current["path"] is not None:
            records.append({"path": current["path"], "boxes": list(current["boxes"])})

    for line in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("- boxes:"):
            flush()
            current = {"path": None, "boxes": []}
        elif stripped.startswith("path:"):
            current["path"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- {"):
            fields    = dict(re.findall(r"(\w+):\s*([^,}]+)", stripped))
            label_raw = fields.get("label")
            if not label_raw:
                continue
            try:
                x_min = float(fields.get("x_min", "0"))
                y_min = float(fields.get("y_min", "0"))
                x_max = float(fields.get("x_max", "0"))
                y_max = float(fields.get("y_max", "0"))
            except ValueError:
                continue
            if normalize_traffic_label(label_raw):
                current["boxes"].append((x_min, y_min, x_max, y_max))
    flush()
    return records


def resolve_bosch_image_path(path_text: str, image_index):
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = BOSCH_DIR / path_text.lstrip("./")
        if candidate.exists():
            return candidate
    basename = Path(path_text).name.lower()
    matches  = image_index.get(basename, [])
    return matches[0] if matches else None


def parse_bosch_annotations():
    image_index = build_image_index(BOSCH_DIR)
    dataset     = []
    missing     = 0

    for yaml_path in [BOSCH_DIR / "train.yaml", BOSCH_DIR / "test.yaml"]:
        if not yaml_path.exists():
            continue
        for entry in parse_bosch_yaml_file(yaml_path):
            if not entry["boxes"]:
                continue
            img_path = resolve_bosch_image_path(entry["path"], image_index)
            if img_path is None or not img_path.exists():
                if not entry["path"].startswith("/net/"):
                    print(f"  Warning: Missing Bosch image for {entry['path']}")
                missing += 1
                continue
            dataset.append({
                "img_path": img_path,
                "boxes_raw": entry["boxes"],
                "prefix": "bosch_",
            })

    if missing:
        print(f"  Bosch: {missing} entries skipped")
    print(f"  Bosch dataset: {len(dataset)} images found.")
    return dataset


# ═════════════════════════════════════════════
# WRITE YOLO FILES  (detector dataset)
# ═════════════════════════════════════════════

def write_detector_pothole_files(dataset, split, preview_count=5):
    """Write pothole items as class-0 in detector_yolo/."""
    images_dir  = DETECTOR_OUT_DIR / "images" / split
    labels_dir  = DETECTOR_OUT_DIR / "labels" / split
    preview_dir = DETECTOR_OUT_DIR / "previews" / split
    previews_saved = 0

    for item in dataset:
        img_path     = item["img_path"]
        img_w, img_h = item["img_w"], item["img_h"]
        boxes        = item["boxes"]
        new_basename = f"{item['prefix']}{img_path.stem}"
        dest_img     = images_dir / f"{new_basename}{img_path.suffix}"
        dest_txt     = labels_dir / f"{new_basename}.txt"

        shutil.copy2(img_path, dest_img)

        yolo_labels = []
        with open(dest_txt, "w") as f:
            for box in boxes:
                xc, yc, bw, bh = convert_bbox_to_yolo((img_w, img_h), box)
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                yolo_labels.append((0, xc, yc, bw, bh))

        if previews_saved < preview_count:
            draw_and_save_preview(dest_img, yolo_labels, preview_dir,
                                  DETECTOR_CLASS_NAMES, DETECTOR_COLOR_MAP)
            previews_saved += 1


def write_detector_traffic_files(dataset, split, preview_count=5):
    """Write traffic light items as class-1 in detector_yolo/."""
    import cv2
    images_dir  = DETECTOR_OUT_DIR / "images" / split
    labels_dir  = DETECTOR_OUT_DIR / "labels" / split
    preview_dir = DETECTOR_OUT_DIR / "previews" / split
    previews_saved = 0

    for item in dataset:
        img_path     = item["img_path"]
        new_basename = f"{item['prefix']}{img_path.stem}"
        dest_img     = images_dir / f"{new_basename}{img_path.suffix}"
        dest_txt     = labels_dir / f"{new_basename}.txt"

        shutil.copy2(img_path, dest_img)

        img = cv2.imread(str(dest_img))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        yolo_labels = []
        with open(dest_txt, "w", encoding="utf-8") as f:
            for (x_min, y_min, x_max, y_max) in item["boxes_raw"]:
                xc, yc, bw, bh = convert_bbox_to_yolo(
                    (img_w, img_h), (x_min, x_max, y_min, y_max)
                )
                f.write(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                yolo_labels.append((1, xc, yc, bw, bh))

        if previews_saved < preview_count:
            draw_and_save_preview(dest_img, yolo_labels, preview_dir,
                                  DETECTOR_CLASS_NAMES, DETECTOR_COLOR_MAP)
            previews_saved += 1


def generate_detector_yaml():
    content = (
        f"path: {DETECTOR_OUT_DIR.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: 2\nnames: {DETECTOR_CLASS_NAMES}\n"
    )
    yaml_path = DETECTOR_OUT_DIR / "dataset.yaml"
    yaml_path.write_text(content)
    print(f"  Generated {yaml_path}")


# ═════════════════════════════════════════════
# SEVERITY CROPS  (ImageFolder for YOLOv8-cls)
# ═════════════════════════════════════════════

def severity_label(box, img_w, img_h):
    xmin, xmax, ymin, ymax = box
    box_area   = max(xmax - xmin, 1) * max(ymax - ymin, 1)
    image_area = img_w * img_h
    ratio      = box_area / image_area
    if ratio < SEVERITY_LOW_MAX:
        return "Low"
    if ratio < SEVERITY_MEDIUM_MAX:
        return "Medium"
    return "High"


def crop_pothole(img_path: Path, box, img_w, img_h, padding=CROP_PADDING):
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    xmin, xmax, ymin, ymax = box
    x1 = max(int(xmin) - padding, 0)
    y1 = max(int(ymin) - padding, 0)
    x2 = min(int(xmax) + padding, img_w)
    y2 = min(int(ymax) + padding, img_h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def write_severity_crops(all_pothole_data):
    """
    Crops each pothole bounding box, auto-labels by area ratio,
    and writes to severity_crops/{train,val,test}/{Low,Medium,High}/.
    """
    import cv2

    # Build flat list of (img_path, box, img_w, img_h) per individual pothole
    items = []
    for item in all_pothole_data:
        for box in item["boxes"]:
            items.append({
                "img_path": item["img_path"],
                "box":      box,
                "img_w":    item["img_w"],
                "img_h":    item["img_h"],
                "prefix":   item["prefix"],
            })

    train_set, val_set, test_set = split_dataset(items)
    splits = {"train": train_set, "val": val_set, "test": test_set}

    counts = {cls: 0 for cls in SEVERITY_CLASSES}

    for split_name, split_data in splits.items():
        for idx, item in enumerate(split_data):
            label  = severity_label(item["box"], item["img_w"], item["img_h"])
            crop   = crop_pothole(item["img_path"], item["box"], item["img_w"], item["img_h"])
            if crop is None:
                continue

            out_dir  = SEVERITY_OUT_DIR / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)

            stem     = item["img_path"].stem
            out_name = f"{item['prefix']}{stem}_{idx}.jpg"
            cv2.imwrite(str(out_dir / out_name), crop)
            counts[label] += 1

    print(f"  Severity crop distribution: {counts}")
    return counts


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def prepare_detector():
    print("\n" + "="*55)
    print("  STAGE 1 — DETECTOR DATASET (pothole + traffic_light)")
    print("="*55)
    prepare_directories(DETECTOR_OUT_DIR)

    # ── Pothole ────────────────────────────────────────
    ds1 = gather_pothole_chitholian()
    print(f"  Chitholian pothole: {len(ds1)} samples")
    ds2 = gather_pothole_kaggle()
    print(f"  Kaggle pothole:     {len(ds2)} samples")
    all_pothole = ds1 + ds2
    print(f"  Pothole total:      {len(all_pothole)} samples")

    ph_train, ph_val, ph_test = split_dataset(all_pothole)
    print(f"  Pothole split  → Train:{len(ph_train)} Val:{len(ph_val)} Test:{len(ph_test)}")
    write_detector_pothole_files(ph_train, "train")
    write_detector_pothole_files(ph_val,   "val")
    write_detector_pothole_files(ph_test,  "test", preview_count=0)

    # ── Traffic ────────────────────────────────────────
    lisa_data  = parse_lisa_annotations()
    bosch_data = parse_bosch_annotations()
    all_traffic = lisa_data + bosch_data
    print(f"  Traffic total:      {len(all_traffic)} samples")

    tl_train, tl_val, tl_test = split_dataset(all_traffic)
    print(f"  Traffic split  → Train:{len(tl_train)} Val:{len(tl_val)} Test:{len(tl_test)}")
    write_detector_traffic_files(tl_train, "train")
    write_detector_traffic_files(tl_val,   "val")
    write_detector_traffic_files(tl_test,  "test", preview_count=0)

    generate_detector_yaml()
    print(f"  Previews → {DETECTOR_OUT_DIR / 'previews'}")
    print("  Detector dataset ready!")

    return all_pothole


def prepare_severity(all_pothole_data):
    print("\n" + "="*55)
    print("  STAGE 2a — SEVERITY CROPS (Low / Medium / High)")
    print("="*55)
    if SEVERITY_OUT_DIR.exists():
        shutil.rmtree(SEVERITY_OUT_DIR)
    counts = write_severity_crops(all_pothole_data)
    total  = sum(counts.values())
    print(f"  Total severity crops: {total}")
    print("  Severity crops ready!")


def main():
    all_pothole = prepare_detector()
    prepare_severity(all_pothole)

    print("\n" + "="*55)
    print("  All datasets prepared!")
    print("="*55)
    print(f"  Detector dataset : {DETECTOR_OUT_DIR}")
    print(f"  Severity crops   : {SEVERITY_OUT_DIR}")
    print("\n  Next steps:")
    print("    python src/augment.py")
    print("    python src/train.py --stage all --aug")
    print("    python src/benchmark_severity.py")


if __name__ == "__main__":
    main()
