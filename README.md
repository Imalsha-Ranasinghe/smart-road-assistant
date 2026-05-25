# Smart Road Assistant — Group 35

> Dashboard-camera computer vision system that detects **potholes** and **traffic lights** in real time using a two-stage cascade pipeline built on YOLOv8.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DASHBOARD CAMERA                             │
│                      (video / image input)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │  BGR frame
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1 — DETECTOR                               │
│                                                                     │
│   Model : YOLOv8s   (models/detector_model.pt)                      │
│   Input : 640 × 640 frame                                           │
│   Output: bounding boxes + class label + confidence                 │
│                                                                     │
│   Class 0 → pothole        Class 1 → traffic_light                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │ crop(y1:y2, x1:x2)      │ crop(y1:y2, x1:x2)
              ▼                         ▼
┌─────────────────────┐   ┌─────────────────────────────────────────┐
│   STAGE 2a          │   │   STAGE 2b                              │
│   POTHOLE ANALYZER  │   │   TRAFFIC LIGHT ANALYZER                │
│                     │   │                                         │
│  Method A (CNN):    │   │   Step 1 — Color detection (pure CV):   │
│    YOLOv8n-cls      │   │     • Resize crop → 48 × 144            │
│    severity_model   │   │     • Split into 3 vertical thirds      │
│    Low/Medium/High  │   │       top=Red mid=Yellow bot=Green      │
│                     │   │     • Brightness vote → position winner  │
│  Method B (CV):     │   │     • HSV pixel count → colour winner   │
│    Canny edges       │   │     • Combine both → final colour       │
│    depth score      │   │                                         │
│    texture score    │   │   Step 2 — Lane relevance (heuristic):  │
│    weighted sum     │   │     • x_center 20–80% of frame width    │
│                     │   │     • box area > 0.1% of frame area     │
│  benchmark picks    │   │     • aspect ratio (H/W) > 1.2          │
│  best method        │   │                                         │
└──────────┬──────────┘   └──────────────────────┬──────────────────┘
           │                                      │
           ▼                                      ▼
  severity: Low/Med/High             color: Red/Yellow/Green/Unknown
  distance: Very Close→Far           lane_relevant: True/False
  instructions: text                 instructions: text
  cv_scores: {edge,depth,texture}
           │                                      │
           └──────────────────┬───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                      │
│                                                                     │
│  • Annotated frame with coloured bounding boxes                     │
│  • Warning messages (lane-relevant lights only, all potholes)       │
│  • JSON: potholes[], traffic_lights[], warnings[]                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌───────────────────┐        ┌───────────────────────┐
   │  Flask REST API   │        │  Python script /       │
   │  backend/app.py   │        │  Jupyter notebook      │
   │  port 5000        │        │  direct Pipeline use   │
   └────────┬──────────┘        └───────────────────────┘
            │
            ▼
   ┌──────────────────┐
   │  React Frontend  │
   │  frontend/       │
   └──────────────────┘
```

---

## Two-Stage Cascade — Why This Design?

| Approach | Problem |
|---|---|
| One big multi-class model | Less specialised per object type, harder to add deep analysis |
| Two separate full models (old design) | Doubles inference time, no crop-level analysis |
| **Two-stage cascade (this design)** | Stage 1 is fast and light; Stage 2 runs only on small crops, enabling deep per-object analysis without cost |

Stage 2b (traffic lights) needs **zero training** — pure classical CV is reliable for color detection on cropped, isolated traffic light regions. Stage 2a benchmarks CV against a CNN and picks the winner automatically.

---

## File Structure

```
smart-road-assistant/
│
├── src/                            ← all Python source code
│   ├── prepare_data.py             # converts raw datasets → YOLO format
│   │                               #   output: data/processed/detector_yolo/
│   │                               #   output: data/processed/severity_crops/
│   │
│   ├── augment.py                  # weather augmentation (rain/fog/night/wet)
│   │                               #   output: data/processed/detector_yolo_aug/
│   │
│   ├── train.py                    # fine-tunes YOLOv8 models
│   │                               #   --stage detector  → models/detector_model.pt
│   │                               #   --stage severity  → models/severity_model.pt
│   │                               #   --stage all       → both
│   │                               #   --aug             → use augmented data
│   │                               #   --eval            → run validation after training
│   │
│   ├── benchmark_severity.py       # evaluates CV vs CNN on test set
│   │                               #   output: benchmark_results.json
│   │                               #   output: configs/pipeline_config.yaml (winner)
│   │
│   ├── pipeline.py                 # main inference class
│   │                               #   Pipeline.run(frame)        → dict
│   │                               #   Pipeline.run_image(path)   → saves annotated jpg
│   │                               #   Pipeline.run_video(in,out) → annotated mp4
│   │                               #   Pipeline.run_webcam()      → live display
│   │
│   └── analyzers/
│       ├── __init__.py
│       ├── pothole_analyzer.py     # Stage 2a: severity + distance + instructions
│       │                           #   method="cnn"  → YOLOv8n-cls classifier
│       │                           #   method="cv"   → edge/depth/texture scoring
│       │                           #   method="auto" → CNN if available, else CV
│       └── traffic_analyzer.py     # Stage 2b: color + lane relevance (pure CV)
│
├── configs/
│   ├── detector.yaml               # reference config for Stage 1 (nc=2)
│   └── pipeline_config.yaml        # runtime thresholds, written by benchmark_severity.py
│                                   #   severity_method: cv | cnn | auto
│                                   #   detector_conf, lane_x_min/max, etc.
│
├── notebooks/                      ← development and presentation notebooks
│   ├── 01-explore-data.ipynb       # dataset stats, class distribution, sample images
│   ├── 02-train-detector.ipynb     # Stage 1 training + curves + predictions
│   ├── 03-train-severity.ipynb     # Stage 2a training + confusion matrix
│   ├── 04-benchmark.ipynb          # CV vs CNN side-by-side comparison
│   └── 05-pipeline-demo.ipynb      # end-to-end demo with radar charts + internals
│
├── backend/
│   └── app.py                      # Flask API — async video jobs, per-request upload dirs
│
├── frontend/
│   ├── src/App.jsx                 # React UI
│   ├── index.html
│   └── package.json
│
├── models/                         ← git-ignored (download or train locally)
│   ├── detector_model.pt           # Stage 1 — YOLOv8s, 2 classes
│   └── severity_model.pt           # Stage 2a — YOLOv8n-cls, 3 classes
│
├── data/                           ← git-ignored
│   ├── raw/
│   │   ├── pothole/
│   │   │   ├── annotated-pothole-images-chitholian/   ← Pascal VOC XML + JPG
│   │   │   └── pothole-kaggle-dataset/                ← annotations/ + images/
│   │   └── traffic/
│   │       ├── lisa/                                  ← frameAnnotationsBOX.csv
│   │       └── bosch/                                 ← train.yaml + images/
│   └── processed/                  ← generated by prepare_data.py
│       ├── detector_yolo/          # YOLO format, nc=2
│       ├── detector_yolo_aug/      # weather-augmented train split
│       └── severity_crops/         # ImageFolder: train/val/test × Low/Med/High
│
├── runs/                           ← git-ignored — YOLO training logs + weights
├── benchmark_results.json          ← git-ignored — generated after benchmark
├── requirements.txt
└── .gitignore
```

---

## Data Setup

Raw data is **not committed** to this repo. Download and place in the correct folders.

| Dataset | Download | Extract to |
|---|---|---|
| Chitholian potholes | [Kaggle — Chitholian](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) | `data/raw/pothole/annotated-pothole-images-chitholian/` |
| Kaggle potholes | [Kaggle — Sachinkumar413](https://www.kaggle.com/datasets/sachinkumar413/pothole-images-dataset) | `data/raw/pothole/pothole-kaggle-dataset/` |
| LISA traffic lights | [Kaggle — mbornoe](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) | `data/raw/traffic/lisa/` |
| Bosch traffic lights | [Kaggle — isaienkov](https://www.kaggle.com/datasets/isaienkov/bosch-small-traffic-lights-dataset) | `data/raw/traffic/bosch/` |

---

## Quick Start

### 1 — Environment

```bash
git clone <repo-url>
cd smart-road-assistant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # installs ultralytics, torch+CUDA, opencv, flask
```

### 2 — Prepare Data

```bash
python src/prepare_data.py
```

Creates:
- `data/processed/detector_yolo/` — combined 2-class YOLO dataset with `dataset.yaml`
- `data/processed/severity_crops/` — cropped pothole patches in `Low/Medium/High` folders

Verify with notebook **01-explore-data.ipynb** before training.

### 3 — Augment

```bash
python src/augment.py
```

Adds 3 weather-augmented copies of each training image (rain, fog, night, wet road) into `detector_yolo_aug/`.

### 4 — Train

```bash
# Train both models in one command
python src/train.py --stage all --aug --eval

# Or separately:
python src/train.py --stage detector --aug     # YOLOv8s, 100 epochs
python src/train.py --stage severity           # YOLOv8n-cls, 50 epochs
```

Output: `models/detector_model.pt` and `models/severity_model.pt`

Training progress is logged to `runs/detector/` and `runs/severity/`.

### 5 — Benchmark Severity Methods

```bash
python src/benchmark_severity.py
```

Evaluates Classical CV vs Fine-tuned CNN on the held-out test set.
Prints accuracy table and writes the winning method to `configs/pipeline_config.yaml`.

### 6 — Run Inference

**Python:**
```python
from src.pipeline import Pipeline

pipe = Pipeline()

# Single image — saves road_detected.jpg next to input
pipe.run_image("road.jpg")

# Video file — saves annotated mp4
pipe.run_video("dashcam.mp4", "output.mp4")

# Live webcam — press Q to quit
pipe.run_webcam()
```

**Backend API:**
```bash
python backend/app.py    # http://localhost:5000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev              # http://localhost:5173
```

---

## Notebook Workflow

Run notebooks in order after the data and models are ready:

| # | Notebook | When to run | What you get |
|---|---|---|---|
| 01 | `01-explore-data.ipynb` | After `prepare_data.py` | Dataset stats, sample images, severity distribution |
| 02 | `02-train-detector.ipynb` | After training Stage 1 | Loss/mAP curves, val predictions, confusion matrix |
| 03 | `03-train-severity.ipynb` | After training Stage 2a | Accuracy curves, per-class F1, test confusion matrix |
| 04 | `04-benchmark.ipynb` | After both trained | CV vs CNN comparison charts, winner selection |
| 05 | `05-pipeline-demo.ipynb` | After benchmark | Full end-to-end demo, radar charts, TL internals |

```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

---

## Backend API Reference

```bash
python backend/app.py    # starts at http://localhost:5000
```

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/health` | GET | — | `{detector_model: bool, severity_model: bool}` |
| `/detect/image` | POST | `image` file (multipart) | annotated image (base64) + potholes[] + traffic_lights[] + warnings[] |
| `/detect/video` | POST | `video` file (multipart) | `{job_id, status: "processing"}` |
| `/video/status/<job_id>` | GET | — | `{status: "processing" \| "done" \| "error"}` |
| `/video/output/<job_id>` | GET | — | annotated `.mp4` file |

Video processing is **asynchronous** — submit and poll `/video/status/<job_id>` until `"done"`, then fetch `/video/output/<job_id>`.

---

## Models

| Model | File | Architecture | Input | Classes | Trained on |
|---|---|---|---|---|---|
| Stage 1 Detector | `detector_model.pt` | YOLOv8s | 640×640 | pothole, traffic_light | Chitholian + Kaggle + LISA + Bosch |
| Stage 2a Severity | `severity_model.pt` | YOLOv8n-cls | 128×128 crop | Low, Medium, High | Auto-labelled pothole crops |

Stage 2b (traffic light analyzer) uses **no model file** — pure classical CV only.

---

## Severity Auto-Labeling Rule

Pothole crops are labeled automatically from ground-truth bounding box size:

```
box_area / image_area < 0.03   →  Low     (small, distant pothole)
box_area / image_area < 0.12   →  Medium
box_area / image_area ≥ 0.12   →  High    (large, close pothole)
```

After training, `benchmark_severity.py` validates this on the test set and picks the best method (CV or CNN). The result is written to `configs/pipeline_config.yaml`.

---

## Traffic Light Lane Relevance

A detected traffic light is considered relevant to your lane if **all three** conditions hold:

```
x_center / frame_width  in  [0.20, 0.80]   — not at far edge of frame
box_area / frame_area   >   0.001           — large enough (not distant)
box_height / box_width  >   1.2             — taller than wide (traffic light shape)
```

Thresholds are tunable in `configs/pipeline_config.yaml`.
