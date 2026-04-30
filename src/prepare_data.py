import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths Setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "pothole"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "pothole_yolo"

DS1_DIR = RAW_DIR / "annotated-pothole-images-chitholian"
DS2_DIR = RAW_DIR / "pothole-kaggle-dataset"

# Classes Setup (YOLO requires integer IDs)
CLASSES = {"pothole": 0}

# Splitting ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42

def convert_bbox_to_yolo(size, box):
    """
    Converts Pascal VOC bounding box to YOLO format.
    size = (width, height)
    box = (xmin, xmax, ymin, ymax)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    # Calculate center x and y
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    
    # Calculate width and height
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # Normalize coordinates
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def parse_xml(xml_file):
    """
    Parses a Pascal VOC XML file and extracts bounding boxes.
    Returns: image_filename, List of (class_id, x, y, w, h)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename_attr = root.find('filename')
    img_filename = filename_attr.text if filename_attr is not None else None
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    labels = []
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        difficult_val = int(difficult.text) if difficult is not None else 0
            
        cls_name = obj.find('name').text.lower().strip()
        
        if cls_name not in CLASSES:
            continue
            
        cls_id = CLASSES[cls_name]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
             
        bb = convert_bbox_to_yolo((w, h), b)
        labels.append((cls_id, *bb))
        
    return img_filename, labels

def gather_dataset1():
    """Gathers data from Chitholian dataset."""
    data = []
    xml_files = glob.glob(str(DS1_DIR / "*.xml"))
    for xml_path in xml_files:
        xml_path = Path(xml_path)
        img_filename, labels = parse_xml(xml_path)
        
        # Determine image file path (it's in the same folder, .jpg extension)
        img_path = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            print(f"Warning: Image missing for {xml_path.name}")
            continue
            
        data.append({
            "img_path": img_path,
            "labels": labels,
            "prefix": "chitholian_"
        })
    return data

def gather_dataset2():
    """Gathers data from Kaggle dataset."""
    data = []
    annotations_dir = DS2_DIR / "annotations"
    images_dir = DS2_DIR / "images"
    
    xml_files = glob.glob(str(annotations_dir / "*.xml"))
    for xml_path in xml_files:
        xml_path = Path(xml_path)
        img_filename, labels = parse_xml(xml_path)
        
        # Verify image file
        img_path = images_dir / img_filename
        if not img_path.exists():
            print(f"Warning: Image missing for {xml_path.name} -> expected {img_path.name}")
            continue
            
        data.append({
            "img_path": img_path,
            "labels": labels,
            "prefix": "kaggle_"
        })
    return data

def prepare_directories():
    """Creates the YOLO directory structure."""
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
        
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
def write_yolo_files(dataset, split):
    """Copies images and writes YOLO txt files to the specific split folder."""
    images_dir = PROCESSED_DIR / "images" / split
    labels_dir = PROCESSED_DIR / "labels" / split
    
    for item in dataset:
        img_path = item["img_path"]
        labels = item["labels"]
        # Prefix the filename to avoid any collision between datasets
        new_basename = f"{item['prefix']}{img_path.stem}"
        new_img_name = f"{new_basename}{img_path.suffix}"
        new_txt_name = f"{new_basename}.txt"
        
        # destination paths
        dest_img_path = images_dir / new_img_name
        dest_txt_path = labels_dir / new_txt_name
        
        # 1. Copy image
        shutil.copy2(img_path, dest_img_path)
        
        # 2. Write labels file
        with open(dest_txt_path, "w") as f:
            for label in labels:
                cls_id, x, y, w, h = label
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def generate_yaml():
    """Generates the dataset.yaml file required for YOLO training."""
    yaml_content = f"""path: {PROCESSED_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {list(CLASSES.keys())}
"""
    yaml_path = PROCESSED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Generated {yaml_path.name}")

def main():
    print("Preparing YOLO directories...")
    prepare_directories()
    
    print("Gathering Dataset 1 (Chitholian)...")
    ds1_data = gather_dataset1()
    print(f"Found {len(ds1_data)} valid samples.")
    
    print("Gathering Dataset 2 (Kaggle)...")
    ds2_data = gather_dataset2()
    print(f"Found {len(ds2_data)} valid samples.")
    
    # Combine everything
    all_data = ds1_data + ds2_data
    print(f"Total samples combined: {len(all_data)}")
    
    # Shuffle for randomness in splits
    random.seed(RANDOM_SEED)
    random.shuffle(all_data)
    
    # Calculate indices
    total = len(all_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_set = all_data[:train_end]
    val_set = all_data[train_end:val_end]
    test_set = all_data[val_end:]
    
    print(f"Split sizes -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # Write to disk
    print("Writing train set...")
    write_yolo_files(train_set, "train")
    print("Writing val set...")
    write_yolo_files(val_set, "val")
    print("Writing test set...")
    write_yolo_files(test_set, "test")
    
    generate_yaml()
    print("Dataset preparation completed successfully! It is available at data/processed/pothole_yolo")

if __name__ == "__main__":
    main()
