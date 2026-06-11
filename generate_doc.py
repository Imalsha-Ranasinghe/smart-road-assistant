"""
Generate Smart Road Assistant project explanation document.
Run: python generate_doc.py
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

doc = Document()

# ── Page margins ─────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)

# ── Style helpers ─────────────────────────────────────────
TITLE_COLOR   = RGBColor(0x1A, 0x53, 0x76)   # dark blue
H1_COLOR      = RGBColor(0x1A, 0x53, 0x76)
H2_COLOR      = RGBColor(0x2E, 0x75, 0xB6)
H3_COLOR      = RGBColor(0x2E, 0x75, 0xB6)
TABLE_HEADER  = RGBColor(0x2E, 0x75, 0xB6)
CODE_COLOR    = RGBColor(0x16, 0x3A, 0x2B)

def set_font(run, size=11, bold=False, italic=False, color=None, font_name="Calibri"):
    run.font.name      = font_name
    run.font.size      = Pt(size)
    run.font.bold      = bold
    run.font.italic    = italic
    if color:
        run.font.color.rgb = color

def heading1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    set_font(run, size=15, bold=True, color=H1_COLOR)
    # bottom border
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'),  '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '2E75B6')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p

def heading2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(text)
    set_font(run, size=13, bold=True, color=H2_COLOR)
    return p

def heading3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(text)
    set_font(run, size=11, bold=True, color=H3_COLOR)
    return p

def body(text, indent=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(3)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    run = p.add_run(text)
    set_font(run, size=11)
    return p

def bullet(text, level=0, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent  = Cm(0.5 + level * 0.5)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    if bold_prefix:
        r1 = p.add_run(bold_prefix + ": ")
        set_font(r1, size=11, bold=True)
        r2 = p.add_run(text)
        set_font(r2, size=11)
    else:
        r = p.add_run(text)
        set_font(r, size=11)
    return p

def code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.8)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    set_font(run, size=10, font_name="Courier New", color=CODE_COLOR)
    return p

def table_header_row(table, headers, widths=None):
    row = table.rows[0]
    for i, (cell, header) in enumerate(zip(row.cells, headers)):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.clear()
        run = p.add_run(header)
        set_font(run, size=10, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
        # shading
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement('w:shd')
        shd.set(qn('w:val'),   'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'),  '2E75B6')
        tcPr.append(shd)

def table_row(table, row_idx, values, shaded=False):
    row = table.rows[row_idx]
    for cell, val in zip(row.cells, values):
        p = cell.paragraphs[0]
        p.clear()
        run = p.add_run(val)
        set_font(run, size=10)
        if shaded:
            tc   = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd  = OxmlElement('w:shd')
            shd.set(qn('w:val'),   'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'),  'DEEAF1')
            tcPr.append(shd)

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = 'Table Grid'
    table_header_row(t, headers)
    for i, row in enumerate(rows):
        table_row(t, i+1, row, shaded=(i%2==0))
    if col_widths:
        for i, row in enumerate(t.rows):
            for j, (cell, w) in enumerate(zip(row.cells, col_widths)):
                cell.width = Cm(w)
    doc.add_paragraph()
    return t

def spacer():
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)

# ════════════════════════════════════════════════════════
# TITLE PAGE
# ════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(40)
run = p.add_run("Smart Road Assistant")
set_font(run, size=26, bold=True, color=TITLE_COLOR)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Group 35  ·  Computer Vision Mini Project")
set_font(run, size=13, color=RGBColor(0x40,0x40,0x40))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Complete Project Explanation Note")
set_font(run, size=12, italic=True, color=RGBColor(0x60,0x60,0x60))

doc.add_page_break()

# ════════════════════════════════════════════════════════
# 1. PROJECT OVERVIEW
# ════════════════════════════════════════════════════════
heading1("1. Project Overview")
body("Smart Road Assistant is a dashboard camera computer vision system that processes video frames in real time to detect two types of road hazards: potholes and traffic lights. For each detected object, the system performs deep analysis — classifying pothole severity and distance, or identifying traffic light color and lane relevance — then generates driver warnings and annotates the output frame.")
spacer()
body("The system is deployed as a full-stack application: a Flask REST API backend processes images and videos, and a React frontend dashboard allows users to upload footage and view annotated results.")

# ════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════
heading1("2. System Architecture — Two-Stage Cascade")
body("The core design is a two-stage cascade pipeline:")

heading2("Stage 1 — Detector")
body("A single YOLOv8s model scans the full frame at 640×640 pixels (or 1280×1280 for video) and outputs bounding boxes with class labels and confidence scores for two classes:")
bullet("Class 0 — pothole")
bullet("Class 1 — traffic_light")

heading2("Stage 2 — Specialized Analyzers")
body("For each bounding box, a small crop of that region is extracted and passed to a dedicated analyzer:")
bullet("Pothole crops → Stage 2a: PotholeAnalyzer — classifies severity (Low / Medium / High), estimates distance, generates instructions")
bullet("Traffic light crops → Stage 2b: TrafficLightAnalyzer — detects color (Red / Yellow / Green / Unknown), checks lane relevance, generates instructions")

heading2("Why This Design?")
add_table(
    ["Approach", "Problem"],
    [
        ["One large multi-class model", "Less specialised per object type; harder to add deep analysis"],
        ["Two separate full-frame models", "Doubles inference time; no crop-level analysis"],
        ["Two-stage cascade (this project)", "Stage 1 is fast and light; Stage 2 runs only on small crops — deep per-object analysis at low cost"],
    ],
    col_widths=[7, 9]
)

# ════════════════════════════════════════════════════════
# 3. DATASETS
# ════════════════════════════════════════════════════════
heading1("3. Datasets")
body("Four public datasets were combined into one unified YOLO-format dataset with 2 classes (pothole = 0, traffic_light = 1):")

add_table(
    ["Dataset", "Class", "Annotation Format", "Source"],
    [
        ["Chitholian Annotated Potholes", "pothole", "Pascal VOC XML", "Kaggle"],
        ["Kaggle Pothole Dataset (Sachinkumar)", "pothole", "Pascal VOC XML", "Kaggle"],
        ["LISA Traffic Light Dataset", "traffic_light", "CSV (frameAnnotationsBOX)", "Kaggle"],
        ["Bosch Small Traffic Lights", "traffic_light", "YAML (train.yaml)", "Kaggle"],
    ],
    col_widths=[5.5, 3, 4, 3]
)

body("All datasets are parsed and converted to YOLO format by src/prepare_data.py. The combined dataset is split 70% train / 20% val / 10% test using a fixed random seed of 42.")

# ════════════════════════════════════════════════════════
# 4. DATA AUGMENTATION
# ════════════════════════════════════════════════════════
heading1("4. Data Augmentation")
body("src/augment.py expands the training set using GPU-accelerated PyTorch tensor operations. Each original training image gets 3 augmented copies (COPIES = 3), resulting in approximately 4× the original training data size. Each copy applies 1–2 randomly chosen weather effects:")

bullet("Rain — 700 sparse bright streaks with a vertical motion-blur kernel to simulate raindrops")
bullet("Fog — brightness overlay blending (intensity 0.35–0.55)")
bullet("Night — brightness reduced to 18–30% with a slight blue channel tint")
bullet("Wet road — increased saturation (1.2–1.5×) and reduced brightness (0.75–0.90×)")

spacer()
body("Only the training split is augmented. Val and test sets remain clean to give unbiased evaluation. Augmentation improves robustness to real-world weather conditions that would otherwise degrade detection.")

# ════════════════════════════════════════════════════════
# 5. SEVERITY AUTO-LABELLING
# ════════════════════════════════════════════════════════
heading1("5. Severity Auto-Labelling")
body("Pothole severity labels are assigned automatically — no manual annotation is needed. The rule uses the ratio of the bounding box area to the total image area:")

code_block("box_area / image_area  <  0.03   →  Low      (small / distant pothole)")
code_block("box_area / image_area  <  0.12   →  Medium")
code_block("box_area / image_area  ≥  0.12   →  High     (large / close pothole)")

body("This works because a physically closer or larger pothole occupies more of the frame. Each pothole bounding box is cropped with 10px padding and saved as an individual image into severity_crops/{train,val,test}/{Low,Medium,High}/. These crops are then used to train the Stage 2a severity classifier.")

# ════════════════════════════════════════════════════════
# 6. MODELS
# ════════════════════════════════════════════════════════
heading1("6. Models — What Was Trained and How")

# Model 1
heading2("6.1  Stage 1 Detector (Original)  —  detector_model.pt")
add_table(
    ["Parameter", "Value"],
    [
        ["Base weights",    "yolov8s.pt  (YOLOv8 Small, COCO pretrained)"],
        ["Architecture",    "YOLOv8s"],
        ["Task",            "Detect potholes and traffic lights in full frames"],
        ["Classes",         "2  —  pothole (0), traffic_light (1)"],
        ["Input size",      "640 × 640 pixels"],
        ["Epochs",          "Auto-scaled: base 100, scaled up by 1/fraction (~285 with augmented data)"],
        ["Batch size",      "Auto from available VRAM"],
        ["Optimizer",       "AdamW, lr₀ = 0.005, cosine LR decay (lrf = 0.01)"],
        ["Warmup",          "3 epochs"],
        ["Mixed precision", "AMP enabled"],
        ["Dataset",         "Original (~26k images) or augmented (~79k images)"],
        ["Output file",     "models/detector_model.pt"],
    ],
    col_widths=[5, 11]
)

# Model 2
heading2("6.2  Dashcam Detector (Production)  —  detector_dashcam.pt")
body("This is a separate model trained specifically for dashcam-perspective footage. The original detector was trained mostly on close-up pothole photos and produced only ~2 detections per 659 dashcam frames due to a domain gap (small, distant potholes at shallow angles look very different).")
spacer()
add_table(
    ["Parameter", "Value"],
    [
        ["Base weights",    "yolov8s.pt  (YOLOv8 Small, COCO pretrained — fresh start, NOT from detector_model.pt)"],
        ["Architecture",    "YOLOv8s"],
        ["Task",            "Detect potholes in dashcam/driving-perspective footage"],
        ["Input size",      "640 × 640 pixels"],
        ["Epochs",          "80 (default)"],
        ["Batch size",      "Auto (-1) on GPU, 16 on CPU"],
        ["Optimizer",       "AdamW, lr₀ = 0.005, cosine LR, patience 20"],
        ["Data sources",    "Road-level YOLO datasets (Roboflow) + optional own frames + reused traffic_light images"],
        ["Train/Val split", "85% / 15%"],
        ["Output file",     "models/detector_dashcam.pt  (falls back to detector_model.pt if not trained)"],
    ],
    col_widths=[5, 11]
)

# Model 3
heading2("6.3  Traffic Light Fine-Tune  —  updates detector_model.pt")
body("This fine-tunes the existing Stage 1 detector on local dashcam footage to improve traffic light detection. The original LISA-trained model had low confidence (0.29–0.60) and flickery detections on local footage.")
spacer()
add_table(
    ["Parameter", "Value"],
    [
        ["Base weights",       "detector_model.pt  (the trained Stage 1 — NOT yolov8s.pt)"],
        ["Task",               "Adapt traffic light detection to local dashcam footage"],
        ["Input size",         "1280 × 1280 (traffic lights are small objects)"],
        ["Epochs",             "30"],
        ["Learning rate",      "lr₀ = 0.001  (low — preserve existing knowledge)"],
        ["Frozen layers",      "First 10 backbone layers frozen"],
        ["Optimizer",          "AdamW, cosine LR, patience 15"],
        ["New frame oversampling", "8× copies per new frame (prevents being drowned by 26k base images)"],
        ["Output",             "Overwrites models/detector_model.pt  (backup → detector_model_prefinetune.pt)"],
    ],
    col_widths=[5, 11]
)

# Model 4
heading2("6.4  Severity Classifier (CNN)  —  severity_model.pt")
add_table(
    ["Parameter", "Value"],
    [
        ["Base weights",    "yolov8n-cls.pt  (YOLOv8 Nano Classifier, ImageNet pretrained)"],
        ["Architecture",    "YOLOv8n-cls (lightest classification variant)"],
        ["Task",            "Classify pothole crop as Low / Medium / High severity"],
        ["Input size",      "128 × 128 pixels (matches crop size exactly)"],
        ["Classes",         "3  —  High (0), Low (1), Medium (2)  [alphabetical — remapped in code]"],
        ["Epochs",          "50"],
        ["Batch size",      "128"],
        ["Optimizer",       "AdamW, cosine LR, patience 15"],
        ["Data cache",      "RAM (crops are tiny — always fits)"],
        ["Mixed precision", "AMP enabled"],
        ["Output file",     "models/severity_model.pt"],
    ],
    col_widths=[5, 11]
)

# ════════════════════════════════════════════════════════
# 7. STAGE 2a — POTHOLE ANALYZER
# ════════════════════════════════════════════════════════
heading1("7. Stage 2a — Pothole Analyzer")
body("The PotholeAnalyzer has two severity classification methods. benchmark_severity.py evaluates both on the held-out test set and writes the winner to configs/pipeline_config.yaml, which is used at runtime.")

heading2("Method A — Classical CV")
body("Computes three scores from the cropped image and combines them as a weighted sum:")

add_table(
    ["Score", "How Computed", "Weight"],
    [
        ["Edge density",   "Canny edge pixels / total pixels  →  more edges = more surface damage", "0.40"],
        ["Depth score",    "Ratio of center darkness to border brightness  →  dark pit center = deeper pothole", "0.35"],
        ["Texture score",  "Normalized standard deviation of grayscale intensities", "0.25"],
    ],
    col_widths=[3.5, 10, 2]
)

body("Severity thresholds on the combined score:")
code_block("Combined score  ≥  0.18   →  High")
code_block("Combined score  ≥  0.10   →  Medium")
code_block("Combined score  <  0.10   →  Low")

heading2("Method B — Fine-Tuned CNN")
body("Passes the 128×128 crop to severity_model.pt (YOLOv8n-cls) and takes the top-1 classification result.")

heading2("Distance Estimation (both methods)")
body("Based on how low the bottom edge of the bounding box sits in the frame:")
add_table(
    ["Condition", "Distance Label"],
    [
        ["Box bottom > 80% of frame height", "Very Close"],
        ["Box bottom > 60% of frame height", "Close"],
        ["Box bottom > 40% of frame height", "Medium"],
        ["Box bottom ≤ 40% of frame height", "Far"],
    ],
    col_widths=[9, 7]
)

heading2("Warning Instructions")
body("A fixed matrix combines severity × distance to generate specific text warnings. For example:")
bullet("High + Very Close → \"BRAKE NOW! Severe pothole directly ahead.\"")
bullet("High + Far → \"Hazard ahead — prepare to slow down.\"")
bullet("Low + Very Close → \"Minor pothole. Proceed carefully.\"")

# ════════════════════════════════════════════════════════
# 8. STAGE 2b — TRAFFIC LIGHT ANALYZER
# ════════════════════════════════════════════════════════
heading1("8. Stage 2b — Traffic Light Analyzer")
body("No model is trained for this stage. It uses only classical OpenCV operations on the cropped traffic light region. This is reliable because the Stage 1 detector has already isolated the traffic light — color detection on a clean crop is a solved problem with HSV thresholding.")

heading2("Color Detection — Two Votes Combined")

heading3("Position Vote")
body("Resize the crop to 48×144 pixels. Split into 3 equal vertical thirds (top = Red, middle = Yellow, bottom = Green). The third with the highest average brightness in the HSV Value channel wins the position vote.")

heading3("Color Vote (HSV Thresholding)")
body("Run HSV range thresholding over the full crop to count saturated pixels of each color:")
add_table(
    ["Color", "HSV Range"],
    [
        ["Red",    "[0–10, 80–255, 80–255]  +  [160–180, 80–255, 80–255]  (wraps around 180°)"],
        ["Yellow", "[20–35, 80–255, 80–255]"],
        ["Green",  "[40–80, 80–255, 80–255]"],
    ],
    col_widths=[2.5, 13]
)
body("The color with the most saturated pixels wins the color vote.")

heading3("Combining the Two Votes")
bullet("Both votes agree → final color = that color, confidence = 1.0")
bullet("Votes disagree → position vote wins, confidence = 0.5")
bullet("Crop has < 2% saturated R/Y/G pixels → Unknown (rejects dark rectangles, signs)")
bullet("Maximum brightness < 30 in any third → Unknown (no lit lamp)")

heading2("False Positive Rejection")
body("Before lane selection, obviously wrong detections are removed:")
bullet("Confidence < 0.25 — too weak")
bullet("Box center in bottom 38% of frame — mounted too low (tail lights, reflections)")
bullet("Height/Width ratio < 0.8 — too wide (signs, light bars)")
bullet("Color = Unknown — no lit color detected")

heading2("Lane Selection — Scoring the Best Light")
body("From all plausible detections, the single light governing the driver's lane is selected by scoring each candidate:")
add_table(
    ["Factor", "Weight", "Reasoning"],
    [
        ["Box size (area)",             "0.50", "Nearest light = the one you are approaching"],
        ["Centrality to vanishing point", "0.30", "Aligned with the road ahead"],
        ["Detector confidence",          "0.15", "Model's certainty"],
        ["Mounting height",              "0.05", "Traffic lights are mounted high"],
    ],
    col_widths=[6, 2.5, 7.5]
)
body("The highest-scoring light is flagged lane_relevant = True and triggers driver warnings. All others are shown faintly as [adj] (adjacent lane).")

# ════════════════════════════════════════════════════════
# 9. LANE DETECTION
# ════════════════════════════════════════════════════════
heading1("9. Lane Detection (Classical CV)")
body("The pipeline detects the ego lane using pure OpenCV — no model. The result is used to filter out potholes that are not in the driver's path and to help identify which traffic light governs the current lane.")

heading2("Detection Steps")
bullet("Mask white and yellow lane markings (HLS thresholding) + Canny edges inside a road-ahead trapezoid (excludes sky and buildings)")
bullet("Run HoughLinesP to find line segments. Filter by slope (0.35–2.5) to reject horizontal clutter and vertical poles")
bullet("Fit polynomial curves (quadratic if enough points) to the left and right lane edges")
bullet("Validate geometry: edges must straddle the frame center, lane width must be 18–95% of frame width, lane must narrow toward the horizon (perspective), vanishing point must be within 32–68% horizontally and 42–66% vertically")

heading2("Fallback Strategy")
bullet("If validation fails → use a fixed central corridor ROI (assumed lane drawn in amber)")
bullet("On video → last detected valid lane is reused for up to 1 second between frames")

heading2("Usage in the Pipeline")
bullet("Potholes whose bottom-center falls outside the lane corridor are filtered out")
bullet("The vanishing point (where lane lines meet) defines the center of the road ahead — used as the reference for traffic light centrality scoring")

# ════════════════════════════════════════════════════════
# 10. FULL PIPELINE FLOW
# ════════════════════════════════════════════════════════
heading1("10. Full Pipeline Flow (Per Frame)")
body("When a frame arrives at Pipeline.run():")

bullet("1. Stage 1 detector runs on the full frame → list of (box, class, confidence)", level=0)
bullet("2. Each pothole box: confidence checked against threshold (0.40), crop extracted, passed to PotholeAnalyzer → severity + distance + instruction text", level=0)
bullet("3. Each traffic light box: crop extracted, passed to TrafficLightAnalyzer → color + raw lane relevance", level=0)
bullet("4. Lane detection runs on the frame → left and right edge lines + vanishing point", level=0)
bullet("5. Potholes whose bottom-center lies outside the detected lane corridor are filtered out", level=0)
bullet("6. Traffic lights: false positives rejected, then single lane-governing light selected by scoring", level=0)
bullet("7. On video only: ByteTrack tracking used across frames. A tracked light must appear for 3+ consecutive frames OR have confidence ≥ 0.50 before it is shown (reduces 1–2 frame false positives)", level=0)
bullet("8. Frame is annotated: colored bounding boxes, labels, lane lines, warning text overlay", level=0)
bullet("9. Returns: annotated_frame, warnings[], potholes[], traffic_lights[], lane", level=0)

spacer()
heading2("Model Selection by Page")
add_table(
    ["Page / Mode", "Detector Used", "Input Size", "Tracking"],
    [
        ["Photo / image upload",    "detector_model.pt",    "640 × 640",   "No"],
        ["Video / dashboard page",  "detector_dashcam.pt (or fallback)", "1280 × 1280", "ByteTrack"],
        ["Webcam",                  "detector_model.pt",    "640 × 640",   "No"],
    ],
    col_widths=[5, 6, 3.5, 2]
)

# ════════════════════════════════════════════════════════
# 11. BACKEND AND FRONTEND
# ════════════════════════════════════════════════════════
heading1("11. Backend and Frontend")

heading2("Flask REST API  (backend/app.py)")
add_table(
    ["Endpoint", "Method", "Description"],
    [
        ["GET  /health",                "GET",  "Returns which models are loaded"],
        ["POST /detect/image",          "POST", "Upload image → returns annotated image (base64) + JSON detections + warnings"],
        ["POST /detect/video",          "POST", "Upload video → returns job_id immediately (async processing)"],
        ["GET  /video/status/<job_id>", "GET",  "Poll status: processing | done | error"],
        ["GET  /video/output/<job_id>", "GET",  "Download the annotated MP4 once done"],
    ],
    col_widths=[5.5, 2, 8.5]
)
body("Video processing is asynchronous — the client submits the video and polls /video/status until done, then fetches the output file. This prevents HTTP timeouts on long videos.")

heading2("React Frontend  (frontend/)")
bullet("Built with Vite + React")
bullet("Image upload page: sends to /detect/image, displays annotated image + warnings + detection JSON")
bullet("Video/dashboard page: sends to /detect/video, polls status, downloads and plays annotated output")

# ════════════════════════════════════════════════════════
# 12. CONFIGURATION
# ════════════════════════════════════════════════════════
heading1("12. Runtime Configuration  (configs/pipeline_config.yaml)")
body("Key thresholds written by benchmark_severity.py and read at pipeline startup:")

add_table(
    ["Key", "Default", "Meaning"],
    [
        ["severity_method",     "auto",  "cv | cnn | auto — which severity method to use"],
        ["detector_conf",       "0.40",  "Minimum confidence for any detection"],
        ["pothole_conf",        "0.40",  "Stricter threshold applied to pothole detections only"],
        ["detector_imgsz",      "640",   "Input size for photo page"],
        ["video_imgsz",         "1280",  "Input size for video/dashcam page"],
        ["lane_x_min",          "0.20",  "Traffic light x_center left bound (fraction of width)"],
        ["lane_x_max",          "0.80",  "Traffic light x_center right bound"],
        ["lane_min_size_ratio", "0.001", "Minimum box_area / frame_area for traffic lights"],
        ["tl_min_aspect_ratio", "1.2",   "Minimum height/width ratio for traffic lights"],
    ],
    col_widths=[5, 2.5, 8.5]
)

# ════════════════════════════════════════════════════════
# 13. MODELS SUMMARY
# ════════════════════════════════════════════════════════
heading1("13. Models Summary")
add_table(
    ["Model File", "Architecture", "Base Weights", "Input", "Classes", "Purpose"],
    [
        ["detector_model.pt",        "YOLOv8s",     "COCO (yolov8s.pt)",       "640px",  "2",        "Detect potholes + traffic lights (photo)"],
        ["detector_dashcam.pt",      "YOLOv8s",     "COCO (yolov8s.pt)",       "640px",  "2",        "Detect potholes + traffic lights (dashcam video)"],
        ["severity_model.pt",        "YOLOv8n-cls", "ImageNet (yolov8n-cls.pt)","128px",  "3",        "Classify pothole severity"],
        ["Traffic light Stage 2b",   "None",        "None (pure CV)",           "48×144", "4 colors", "Detect traffic light color"],
    ],
    col_widths=[4, 3, 4, 2, 2, 5]
)

# ════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════
out_path = r"d:\7 Sem\Computer vision\Mini Project\smart-road-assistant\Smart_Road_Assistant_Explanation.docx"
doc.save(out_path)
print(f"Document saved to: {out_path}")
