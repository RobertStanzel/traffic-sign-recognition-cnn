"""Central configuration — all hyperparameters and paths in one place."""

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = "data/dataset"
RAW_DIR         = "data/raw"
FRAMES_DIR      = "data/frames"
MODEL_SAVE_PATH = "models/best_model.pth"
OUTPUT_DIR      = "output"
YOLO_MODEL_PATH = "models/yolo_detector.pt"   # trained by train_yolo.py
GTSDB_DIR       = "data/gtsdb"                # raw GTSDB images + annotations

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMG_SIZE = 96                              # upgraded from 64 — more detail in crops

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE               = 32
NUM_EPOCHS               = 60
LEARNING_RATE            = 3e-4            # lower LR suits pretrained backbone
WEIGHT_DECAY             = 1e-4
EARLY_STOPPING_PATIENCE  = 10
LABEL_SMOOTHING          = 0.1            # reduces overconfidence on GTSRB

# Freeze ResNet backbone for this many epochs before unfreezing all layers.
# Lets the new head stabilise before fine-tuning the backbone weights.
UNFREEZE_EPOCH           = 5

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
VAL_SPLIT                    = 0.2
FRAME_EXTRACT_INTERVAL_SEC   = 1.0
NUM_WORKERS                  = 0          # keep 0 on Windows

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
USE_PRETRAINED_RESNET = True              # upgraded from False

# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------
YOLO_CONF_THRESHOLD = 0.35               # minimum YOLO detection confidence
YOLO_IMG_SIZE       = 640                # inference resolution for YOLO

# Vertical ROI band for the classical Hough fallback.
# Signs never appear in the top 5 % (sky) or bottom 20 % (road surface).
ROI_Y_MIN = 0.05
ROI_Y_MAX = 0.80

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# GTSRB 43-class label mapping (index → English name)
# ---------------------------------------------------------------------------
GTSRB_CLASSES = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}
