"""Prepare GTSDB (German Traffic Sign Detection Benchmark) for YOLOv8 training.

GTSDB is a real-world dataset of full road images with bounding box annotations.
Unlike GTSRB (cropped signs), GTSDB images look exactly like what a dashcam records.
Training YOLOv8n on GTSDB teaches the detector to find small, distant, real signs.

─── HOW TO GET THE DATASET ────────────────────────────────────────────────────
Option A — Official (free, ~600 MB):
    1. Go to: https://benchmark.ini.rub.de/gtsdb_dataset.html
    2. Download "FullIJCNN2013.zip"
    3. Extract to:  traffic_sign_cnn/data/gtsdb/
       Expected layout after extraction:
           data/gtsdb/
               00000.ppm          ← 900 full road images
               00001.ppm
               ...
               00899.ppm
               gt.txt             ← annotations: filename;x1;y1;x2;y2;class_id

Option B — Kaggle (requires kaggle CLI, same dataset):
    kaggle datasets download -d safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb
    unzip -q german-traffic-sign-detection-benchmark-gtsdb.zip -d data/gtsdb/

─── AFTER DOWNLOADING ─────────────────────────────────────────────────────────
Run this script from traffic_sign_cnn/:
    python prepare_gtsdb.py

It will:
  1. Parse gt.txt (GTSDB annotation file)
  2. Group the 43 GTSRB fine-grained classes into 4 super-classes for detection:
       0 prohibitory  (speed limits, no-entry, no-passing …)
       1 danger       (warning triangles …)
       2 mandatory    (blue circles …)
       3 other
  3. Convert bounding boxes to YOLO format (normalised cx cy w h)
  4. Copy/convert .ppm images to .jpg
  5. Create 80/20 train/val split
  6. Write data/gtsdb_yolo/dataset.yaml  ← used by train_yolo.py
"""

import csv
import os
import random
import shutil
import sys
from pathlib import Path

import cv2

# ── Class grouping: GTSDB fine-grained id → 4 super-class id ─────────────────
# Prohibitory: 0-8, 9-10, 15-16
# Danger:      11 (right-of-way), 18-31
# Mandatory:   33-43
# Other:       12-14, 17, 32

def gtsdb_to_superclass(class_id: int) -> int:
    if class_id in range(0, 9) or class_id in (9, 10, 15, 16):
        return 0  # prohibitory
    if class_id in (11,) or class_id in range(18, 32):
        return 1  # danger
    if class_id in range(33, 44):
        return 2  # mandatory
    return 3      # other (stop, yield, priority, no-vehicles, …)


SUPER_CLASSES = ["prohibitory", "danger", "mandatory", "other"]

BASE_DIR  = Path(__file__).parent
_gtsdb_base = BASE_DIR / "data" / "gtsdb"
# Auto-detect: official zip extracts into a FullIJCNN2013 subfolder
GTSDB_DIR = (
    _gtsdb_base / "FullIJCNN2013"
    if (_gtsdb_base / "FullIJCNN2013" / "gt.txt").exists()
    else _gtsdb_base
)
OUT_DIR   = BASE_DIR / "data" / "gtsdb_yolo"


def parse_annotations(gt_path: Path) -> dict:
    """Parse gt.txt → {filename: [(x1,y1,x2,y2,class_id), ...]}"""
    annotations = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 6:
                continue
            fname, x1, y1, x2, y2, cls = parts[0], *map(int, parts[1:6])
            annotations.setdefault(fname, []).append((x1, y1, x2, y2, cls))
    return annotations


def convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h) -> tuple:
    """Convert (x1,y1,x2,y2) pixel box → YOLO normalised (cx,cy,w,h)."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def main():
    # ── Validate dataset presence ────────────────────────────────────────
    gt_path = GTSDB_DIR / "gt.txt"
    if not gt_path.exists():
        print(
            "[ERROR] gt.txt not found.\n"
            "Please download GTSDB and extract to data/gtsdb/\n"
            "See the docstring at the top of this file for instructions."
        )
        sys.exit(1)

    ppm_files = sorted(GTSDB_DIR.glob("*.ppm"))
    if not ppm_files:
        print("[ERROR] No .ppm images found in data/gtsdb/")
        sys.exit(1)

    print(f"[INFO] Found {len(ppm_files)} images and gt.txt in {GTSDB_DIR}")

    # ── Parse annotations ────────────────────────────────────────────────
    annotations = parse_annotations(gt_path)
    print(f"[INFO] {sum(len(v) for v in annotations.values())} annotated boxes "
          f"across {len(annotations)} images")

    # ── Create output directory structure ────────────────────────────────
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Shuffle and split 80/20 ──────────────────────────────────────────
    random.seed(42)
    all_ppm = list(ppm_files)
    random.shuffle(all_ppm)
    n_val    = max(1, int(len(all_ppm) * 0.20))
    val_set  = set(p.name for p in all_ppm[:n_val])

    converted = skipped = 0

    for ppm_path in all_ppm:
        split    = "val" if ppm_path.name in val_set else "train"
        jpg_name = ppm_path.stem + ".jpg"
        img_out  = OUT_DIR / "images" / split / jpg_name
        lbl_out  = OUT_DIR / "labels" / split / (ppm_path.stem + ".txt")

        # Convert .ppm → .jpg
        img = cv2.imread(str(ppm_path))
        if img is None:
            skipped += 1
            continue
        cv2.imwrite(str(img_out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        img_h, img_w = img.shape[:2]

        # Write YOLO label file (empty if image has no annotations)
        boxes = annotations.get(ppm_path.name, [])
        with open(lbl_out, "w") as f:
            for x1, y1, x2, y2, cls_id in boxes:
                super_cls       = gtsdb_to_superclass(cls_id)
                cx, cy, bw, bh  = convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h)
                f.write(f"{super_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1

    print(f"[INFO] Converted {converted} images ({skipped} skipped)")
    print(f"       Train: {converted - n_val}  |  Val: {n_val}")

    # ── Write dataset.yaml ───────────────────────────────────────────────
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(SUPER_CLASSES)}\n"
        f"names: {SUPER_CLASSES}\n"
    )
    print(f"\n[DONE] Dataset ready at {OUT_DIR}")
    print(f"       YAML → {yaml_path}")
    print(f"\nNext step: python train_yolo.py")


if __name__ == "__main__":
    main()
