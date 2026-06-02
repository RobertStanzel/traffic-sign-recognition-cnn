# Traffic Sign Recognition — CNN + YOLOv8

A full end-to-end traffic sign detection and classification system with a live web interface.  
Detects and classifies all **43 GTSRB classes** in real-time from images and videos.

---

## Architecture

```
Input frame
    │
    ▼
┌─────────────────────┐
│  YOLOv8n Detector   │  ← trained on GTSDB (900 real road images)
│  (WHERE is the sign)│    falls back to Hough + Contour if unavailable
└─────────────────────┘
    │  bounding boxes
    ▼
┌─────────────────────┐
│  ResNet-18 CNN      │  ← trained on GTSRB (39,000 cropped sign images)
│  (WHAT is the sign) │    96×96 input, 43-class output
└─────────────────────┘
    │
    ▼
Label + Confidence (e.g. "Speed limit (70km/h) — 94.2%")
```

**Why two models?**  
YOLO handles real-world detection (small, distant, occluded signs). The CNN does fine-grained 43-class classification using the much larger GTSRB dataset. Together they outperform either model alone.

---

## Features

- **Live web interface** — drop an image or video, get annotated results instantly
- **Real progress tracking** — upload %, frame counter, ETA for long videos
- **Auto frame-skip** — scales with video length (3–8 frames) to keep processing fast
- **Temporal tracker** — IoU-based tracker prevents label flickering between frames
- **Vertical ROI band** — classical detector ignores sky and road surface (5%–80% of frame height)
- **43-class GTSRB** support with human-readable sign names
- **ResNet-18 backbone** — pretrained ImageNet weights, fine-tuned with progressive unfreezing
- **Real-world augmentation** — perspective warp, motion blur, fog simulation, strong colour jitter, random erasing

---

## Project Structure

```
traffic_sign_cnn/
├── app.py                  ← Flask web server (run this to launch the UI)
├── roi_interface.html      ← web UI (served by app.py)
├── train.py                ← CNN classifier training loop
├── train_yolo.py           ← YOLOv8n detector training
├── detect_predict.py       ← detection + classification pipeline
├── prepare_gtsdb.py        ← download & convert GTSDB to YOLO format
├── evaluate.py             ← evaluation + confusion matrix
├── predict.py              ← standalone CLI inference
├── config.py               ← all hyperparameters and paths
├── models/
│   ├── cnn_model.py        ← ResNet-18 + custom CNN architectures
│   ├── best_model.pth      ← trained CNN weights (not in git — train locally)
│   └── yolo_detector.pt    ← trained YOLO weights (not in git — train locally)
├── utils/
│   ├── dataset_loader.py   ← Dataset + real-world augmentation pipeline
│   ├── visualize.py        ← training curves, confusion matrix
│   └── extract_frames.py   ← video → JPEG frames
└── data/
    ├── dataset/            ← GTSRB training data (not in git)
    ├── gtsdb/              ← GTSDB detection data (not in git)
    └── gtsdb_yolo/         ← converted YOLO format (auto-generated)
```

---

## Setup

```bash
pip install -r requirements.txt
pip install ultralytics flask
```

---

## Training

### Step 1 — CNN Classifier (GTSRB)

Place GTSRB class subfolders inside `data/dataset/` then:

```bash
cd traffic_sign_cnn
python train.py
```

Key features: ResNet-18 pretrained backbone, progressive unfreezing, OneCycleLR, mixed precision, label smoothing, real-world augmentation.

Output: `models/best_model.pth`

---

### Step 2 — Prepare GTSDB for YOLO

Download `FullIJCNN2013.zip` from https://benchmark.ini.rub.de/gtsdb_dataset.html  
Extract to `data/gtsdb/` then:

```bash
python prepare_gtsdb.py
```

---

### Step 3 — YOLO Detector (GTSDB)

```bash
python train_yolo.py --device 0      # GPU
python train_yolo.py --device cpu    # CPU
```

GTSDB validation results after training:
```
mAP@0.5:       97.1%
mAP@0.5:0.95:  77.1%
Precision:     97.8%
Recall:        89.3%
```

Output: `models/yolo_detector.pt`

---

## Web Interface

```bash
cd traffic_sign_cnn
python app.py
```

Open **http://localhost:5000**

- Drop an image → annotated result in ~1 second
- Drop a video → live progress bar with frame counter and ETA
- Confidence slider to tune detection sensitivity
- Download annotated result with the ↓ button

The server auto-uses YOLO if `yolo_detector.pt` exists, otherwise falls back to classical Hough + contour detection.

---

## CLI

```bash
python detect_predict.py --input image.jpg
python detect_predict.py --input video.mp4 --conf 60
```

---

## Detection Pipeline

```
detect_sign_regions(frame)
    ├── YOLO available → YOLOv8n.predict() → bounding boxes
    └── fallback → ROI band (5%–80%) → HoughCircles + colour contours → NMS

→ classify_crop(crop) → ("Speed limit (70km/h)", 94.2%)
→ SignTracker (IoU) → stable labels across frames
```

---

## Key Config Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 96 | CNN input resolution (upgraded from 64) |
| `USE_PRETRAINED_RESNET` | True | ResNet-18 backbone |
| `LEARNING_RATE` | 3e-4 | AdamW LR |
| `LABEL_SMOOTHING` | 0.1 | Cross-entropy smoothing |
| `UNFREEZE_EPOCH` | 5 | When backbone unfreezes |
| `ROI_Y_MIN / ROI_Y_MAX` | 0.05 / 0.80 | Vertical ROI band |
| `YOLO_CONF_THRESHOLD` | 0.35 | Min YOLO confidence |
