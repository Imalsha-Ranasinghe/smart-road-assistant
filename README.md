# Data Setup

Data files are not committed to this repo. Download manually and place in the correct folders.

## Traffic Light Data
- LISA: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset
  → Extract to: data/raw/traffic/lisa/

- Bosch: https://www.kaggle.com/datasets/isaienkov/bosch-small-traffic-lights-dataset
  → Extract to: data/raw/traffic/bosch/

## Pothole Data
- Kaggle pothole: https://www.kaggle.com/datasets/sachinkumar413/pothole-images-dataset
  → Extract to: data/raw/pothole/kaggle/

- Chitholian: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset
  → Extract to: data/raw/pothole/chitholian/

## After downloading
Run: python src/preprocessing/prepare_datasets.py
This will convert annotations and split into train/val/test automatically.