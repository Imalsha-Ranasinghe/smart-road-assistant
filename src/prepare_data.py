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

# Pothole
POTHOLE_RAW_DIR = BASE_DIR / "data" / "raw" / "pothole"
POTHOLE_OUT_DIR = BASE_DIR / "data" / "processed" / "pothole_yolo"
DS1_DIR         = POTHOLE_RAW_DIR / "annotated-pothole-images-chitholian"
DS2_DIR         = POTHOLE_RAW_DIR / "pothole-kaggle-dataset"

# Traffic
TRAFFIC_RAW_DIR = BASE_DIR / "data" / "raw" / "traffic"
TRAFFIC_OUT_DIR = BASE_DIR / "data" / "processed" / "traffic_yolo"
LISA_DIR        = TRAFFIC_RAW_DIR / "lisa"
BOSCH_DIR       = TRAFFIC_RAW_DIR / "bosch"

# ─────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────
POTHOLE_CLASSES     = {"pothole": 0}
TRAFFIC_CLASSES     = {"Green": 0, "Red": 1, "Yellow": 2}
TRAFFIC_CLASS_NAMES = list(TRAFFIC_CLASSES.keys())

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
    """size=(w,h), box=(xmin, xmax, ymin, ymax)"""
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


def prepare_directories(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


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


def draw_and_save_preview(img_path: Path, labels_raw, out_dir: Path, class_names: list, color_map: dict):
    """Draws bounding boxes on image and saves to previews folder."""
    import cv2
    image = cv2.imread(str(img_path))
    if image is None:
        return
    h, w = image.shape[:2]

    for label in labels_raw:
        # label format: (class_id, x_center, y_center, bw, bh) — normalized
        cls_id, xc, yc, bw, bh = label
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
# POTHOLE
# ═════════════════════════════════════════════

POTHOLE_CLASS_NAMES = list(POTHOLE_CLASSES.keys())
POTHOLE_COLOR_MAP   = {0: (0, 0, 255)}   # Red for pothole


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename_attr = root.find("filename")
    img_filename  = filename_attr.text if filename_attr is not None else None

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    labels = []
    for obj in root.iter("object"):
        cls_name = obj.find("name").text.lower().strip()
        if cls_name not in POTHOLE_CLASSES:
            continue
        cls_id = POTHOLE_CLASSES[cls_name]
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text),
        )
        labels.append((cls_id, *convert_bbox_to_yolo((w, h), b)))

    return img_filename, labels


def gather_pothole_chitholian():
    data = []
    for xml_path in map(Path, glob.glob(str(DS1_DIR / "*.xml"))):
        _, labels = parse_xml(xml_path)
        img_path  = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            print(f"  Warning: Image missing for {xml_path.name}")
            continue
        data.append({"img_path": img_path, "labels": labels, "prefix": "chitholian_"})
    return data


def gather_pothole_kaggle():
    data = []
    annotations_dir = DS2_DIR / "annotations"
    images_dir      = DS2_DIR / "images"
    for xml_path in map(Path, glob.glob(str(annotations_dir / "*.xml"))):
        img_filename, labels = parse_xml(xml_path)
        img_path = images_dir / img_filename
        if not img_path.exists():
            print(f"  Warning: Image missing for {xml_path.name}")
            continue
        data.append({"img_path": img_path, "labels": labels, "prefix": "kaggle_"})
    return data


def write_pothole_yolo_files(dataset, split, preview_count=5):
    images_dir  = POTHOLE_OUT_DIR / "images" / split
    labels_dir  = POTHOLE_OUT_DIR / "labels" / split
    preview_dir = POTHOLE_OUT_DIR / "previews" / split

    previews_saved = 0

    for item in dataset:
        img_path     = item["img_path"]
        labels       = item["labels"]
        new_basename = f"{item['prefix']}{img_path.stem}"
        dest_img     = images_dir / f"{new_basename}{img_path.suffix}"
        dest_txt     = labels_dir / f"{new_basename}.txt"

        shutil.copy2(img_path, dest_img)

        with open(dest_txt, "w") as f:
            for cls_id, x, y, w, h in labels:
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Save preview images with bboxes drawn
        if previews_saved < preview_count:
            draw_and_save_preview(dest_img, labels, preview_dir,
                                  POTHOLE_CLASS_NAMES, POTHOLE_COLOR_MAP)
            previews_saved += 1


def generate_pothole_yaml():
    content = (
        f"path: {POTHOLE_OUT_DIR.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(POTHOLE_CLASSES)}\nnames: {list(POTHOLE_CLASSES.keys())}\n"
    )
    yaml_path = POTHOLE_OUT_DIR / "dataset.yaml"
    yaml_path.write_text(content)
    print(f"  Generated {yaml_path}")


# ═════════════════════════════════════════════
# TRAFFIC
# ═════════════════════════════════════════════

TRAFFIC_COLOR_MAP = {
    0: (0, 255, 0),    # Green
    1: (0, 0, 255),    # Red
    2: (0, 255, 255),  # Yellow
}


def normalize_traffic_label(raw_label: str):
    label = raw_label.strip().lower()
    if label.startswith("go") or label.startswith("green"):
        return "Green"
    if label.startswith("stop") or label.startswith("red"):
        return "Red"
    if label.startswith("warning") or label.startswith("yellow"):
        return "Yellow"
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
                    records[key] = {"img_path": img_path, "labels": [], "prefix": "lisa_"}
                records[key]["labels"].append((label_name, x_min, y_min, x_max, y_max))

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
            label_name = normalize_traffic_label(label_raw)
            if label_name:
                current["boxes"].append((label_name, x_min, y_min, x_max, y_max))

    flush()
    return records


def resolve_bosch_image_path(path_text: str, image_index):
    candidate = Path(path_text)
    # Try local relative path first
    if not candidate.is_absolute():
        candidate = BOSCH_DIR / path_text.lstrip("./")
        if candidate.exists():
            return candidate
    # Fall back to filename-only match (handles remote server paths)
    basename = Path(path_text).name.lower()
    matches  = image_index.get(basename, [])
    return matches[0] if matches else None


def parse_bosch_annotations():
    # Search entire bosch dir to catch all subfolders
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
                # Only warn for non-server paths (suppress remote path flood)
                if not entry["path"].startswith("/net/"):
                    print(f"  Warning: Missing Bosch image for {entry['path']}")
                missing += 1
                continue
            dataset.append({"img_path": img_path, "labels": entry["boxes"], "prefix": "bosch_"})

    if missing:
        print(f"  Bosch: {missing} entries skipped (remote server paths or missing files)")
    print(f"  Bosch dataset: {len(dataset)} images found.")
    return dataset


def write_traffic_yolo_files(dataset, split, preview_count=5):
    images_dir  = TRAFFIC_OUT_DIR / "images" / split
    labels_dir  = TRAFFIC_OUT_DIR / "labels" / split
    preview_dir = TRAFFIC_OUT_DIR / "previews" / split

    previews_saved = 0

    for item in dataset:
        img_path     = item["img_path"]
        new_basename = f"{item['prefix']}{img_path.stem}"
        dest_img     = images_dir / f"{new_basename}{img_path.suffix}"
        dest_txt     = labels_dir / f"{new_basename}.txt"

        shutil.copy2(img_path, dest_img)

        # Convert raw labels to YOLO format while writing
        yolo_labels = []
        with open(dest_txt, "w", encoding="utf-8") as f:
            width, height = get_image_size(dest_img)
            for label_name, x_min, y_min, x_max, y_max in item["labels"]:
                class_id    = TRAFFIC_CLASSES[label_name]
                x, y, w, h = convert_bbox_to_yolo((width, height), (x_min, x_max, y_min, y_max))
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                yolo_labels.append((class_id, x, y, w, h))

        # Save preview images with bboxes drawn
        if previews_saved < preview_count:
            draw_and_save_preview(dest_img, yolo_labels, preview_dir,
                                  TRAFFIC_CLASS_NAMES, TRAFFIC_COLOR_MAP)
            previews_saved += 1


def generate_traffic_yaml():
    content = (
        f"path: {TRAFFIC_OUT_DIR.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(TRAFFIC_CLASSES)}\nnames: {TRAFFIC_CLASS_NAMES}\n"
    )
    yaml_path = TRAFFIC_OUT_DIR / "dataset.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  Generated {yaml_path}")


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def prepare_pothole():
    print("\n" + "="*50)
    print("  POTHOLE DATASET")
    print("="*50)
    prepare_directories(POTHOLE_OUT_DIR)

    ds1 = gather_pothole_chitholian()
    print(f"  Chitholian: {len(ds1)} samples")
    ds2 = gather_pothole_kaggle()
    print(f"  Kaggle:     {len(ds2)} samples")

    all_data = ds1 + ds2
    print(f"  Total:      {len(all_data)} samples")

    train_set, val_set, test_set = split_dataset(all_data)
    print(f"  Split -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    print("  Writing files...")
    write_pothole_yolo_files(train_set, "train")
    write_pothole_yolo_files(val_set,   "val")
    write_pothole_yolo_files(test_set,  "test", preview_count=0)
    generate_pothole_yaml()
    print(f"  Preview images saved to: {POTHOLE_OUT_DIR / 'previews'}")
    print("  ✅ Pothole preparation complete!")


def prepare_traffic():
    print("\n" + "="*50)
    print("  TRAFFIC DATASET")
    print("="*50)
    prepare_directories(TRAFFIC_OUT_DIR)

    lisa_data  = parse_lisa_annotations()
    bosch_data = parse_bosch_annotations()

    all_data = lisa_data + bosch_data
    print(f"  Total: {len(all_data)} samples")

    train_set, val_set, test_set = split_dataset(all_data)
    print(f"  Split -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    print("  Writing files...")
    write_traffic_yolo_files(train_set, "train")
    write_traffic_yolo_files(val_set,   "val")
    write_traffic_yolo_files(test_set,  "test", preview_count=0)
    generate_traffic_yaml()
    print(f"  Preview images saved to: {TRAFFIC_OUT_DIR / 'previews'}")
    print("  ✅ Traffic preparation complete!")


def main():
    prepare_pothole()
    prepare_traffic()
    print("\n" + "="*50)
    print("✅ All datasets prepared successfully!")
    print("="*50)
    print(f"\nPothole output : {POTHOLE_OUT_DIR}")
    print(f"Traffic output : {TRAFFIC_OUT_DIR}")
    print("\nCheck 'previews/' folders to verify bounding boxes look correct!")


if __name__ == "__main__":
    main()