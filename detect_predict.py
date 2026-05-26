"""Detection + Classification pipeline.

Detection strategy:
  1. Hough Circle Transform  — finds actual geometric circles (speed limits,
     no entry, mandatory signs). Gives precise center+radius for tight crops.
  2. Color+contour triangles — finds red triangular warning signs.

Both candidates are verified by color ratio before being passed to the CNN.
A temporal tracker prevents flickering between frames.

Usage:
    python detect_predict.py --input path/to/image.jpg
    python detect_predict.py --input path/to/video.mp4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from models.cnn_model import get_model


COLORS = [
    (0, 220, 60),
    (0, 160, 255),
    (255, 120, 0),
    (200, 0, 255),
    (180, 255, 0),
    (0, 255, 210),
]


# ---------------------------------------------------------------------------
# Classifier loader
# ---------------------------------------------------------------------------

def load_classifier(checkpoint_path: str = config.MODEL_SAVE_PATH):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'. Run train.py first."
        )
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    model = get_model(ckpt["num_classes"], ckpt.get("use_pretrained_resnet", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
    return model, ckpt["idx_to_class"], transform


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _color_ratios(roi_bgr) -> Tuple[float, float]:
    """Return (red_ratio, blue_ratio) pixel fractions in an HSV ROI."""
    if roi_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   110, 80]),  np.array([12,  255, 255])),
        cv2.inRange(hsv, np.array([163, 110, 80]),  np.array([180, 255, 255])),
    )
    blue = cv2.inRange(hsv, np.array([100, 110, 70]), np.array([132, 255, 255]))
    return np.count_nonzero(red) / total, np.count_nonzero(blue) / total


# ---------------------------------------------------------------------------
# Hough Circle detector  (circular signs)
# ---------------------------------------------------------------------------

def _check_sign_ring(frame_bgr, cx: int, cy: int, r: int) -> bool:
    """Verify a detected circle matches one of two real traffic sign patterns:

    Pattern A — Red-ringed sign (prohibition/warning):
        vivid red outer ring  +  bright/white interior (numbers, symbols)
        e.g. Speed limit, No entry, No passing

    Pattern B — Blue filled sign (mandatory):
        mostly vivid blue throughout  +  some white pixels (arrow symbols)
        e.g. Keep right, Go straight, Roundabout

    Rejects: truck logos, wheels, blue truck panels (uniform color, no
    white symbols or wrong ring structure).
    """
    fh, fw = frame_bgr.shape[:2]
    inner_r = max(1, int(r * 0.55))

    y_min = max(0, cy - r)
    y_max = min(fh, cy + r)
    x_min = max(0, cx - r)
    x_max = min(fw, cx + r)
    if y_max <= y_min or x_max <= x_min:
        return False

    roi = frame_bgr[y_min:y_max, x_min:x_max]
    roi_h, roi_w = roi.shape[:2]
    lx, ly = cx - x_min, cy - y_min

    Y, X = np.ogrid[:roi_h, :roi_w]
    dist = np.sqrt((X - lx) ** 2 + (Y - ly) ** 2)
    whole_mask = dist <= r
    outer_mask = (dist <= r) & (dist > inner_r)
    inner_mask = dist <= inner_r

    if outer_mask.sum() == 0 or inner_mask.sum() == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    vivid_red  = (((h <= 12) | (h >= 163)) & (s > 100) & (v > 80))
    vivid_blue = ((h >= 100) & (h <= 132)  & (s > 100) & (v > 70))
    bright     = v > 155  # white/light pixels (sign symbols)

    # --- Pattern A: red ring + white interior ---
    red_outer_ratio   = (vivid_red  & outer_mask).sum() / outer_mask.sum()
    inner_bright_ratio = (bright    & inner_mask).sum() / inner_mask.sum()
    inner_red_ratio    = (vivid_red & inner_mask).sum() / inner_mask.sum()
    pattern_a = (
        red_outer_ratio   > 0.15   # ring has red
        and inner_bright_ratio > 0.18   # center is mostly white/light
        and inner_red_ratio    < 0.40   # center not uniformly red
    )

    # --- Pattern B: blue filled circle + white arrow symbols ---
    blue_whole_ratio  = (vivid_blue & whole_mask).sum() / whole_mask.sum()
    bright_whole_ratio = (bright    & whole_mask).sum() / whole_mask.sum()
    pattern_b = (
        blue_whole_ratio   > 0.28   # circle is mostly vivid blue
        and bright_whole_ratio > 0.10   # has visible white symbols (arrow)
        and bright_whole_ratio < 0.60   # not a mostly-white object
    )

    return pattern_a or pattern_b


def _detect_circles(frame_bgr) -> List[Tuple[int, int, int, int]]:
    """Detect circular traffic signs using Hough Circle Transform + ring check.

    Returns:
        List of tight (x1, y1, x2, y2) bounding boxes around each circle.
    """
    fh, fw = frame_bgr.shape[:2]
    min_r = max(14, int(min(fh, fw) * 0.020))
    max_r = int(min(fh, fw) * 0.16)   # cap size — huge circles are not signs

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,
        param1=60,
        param2=25,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    boxes = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        if cy - r < fh * 0.10 or cy + r > fh * 0.94:
            continue
        # Ring pattern check — rejects truck logos, wheels, etc.
        if not _check_sign_ring(frame_bgr, cx, cy, r):
            continue
        pad = max(4, int(r * 0.10))
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(fw, cx + r + pad)
        y2 = min(fh, cy + r + pad)
        boxes.append((x1, y1, x2, y2))

    return boxes


# ---------------------------------------------------------------------------
# Contour-based triangle detector  (warning signs)
# ---------------------------------------------------------------------------

def _detect_triangles(frame_bgr) -> List[Tuple[int, int, int, int]]:
    """Detect red triangular warning signs using contour approximation.

    Returns:
        List of (x1, y1, x2, y2) bounding boxes.
    """
    fh, fw = frame_bgr.shape[:2]
    frame_area = fh * fw
    min_area = int(frame_area * 0.0008)
    max_area = int(frame_area * 0.10)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   130, 100]), np.array([12,  255, 255])),
        cv2.inRange(hsv, np.array([163, 130, 100]), np.array([180, 255, 255])),
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or area / hull_area < 0.65:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
        if len(approx) != 3:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if not (0.5 < aspect < 2.0):
            continue
        if y < fh * 0.10 or (y + bh) > fh * 0.94:
            continue

        pad = 8
        boxes.append((max(0, x - pad), max(0, y - pad),
                       min(fw, x + bw + pad), min(fh, y + bh + pad)))
    return boxes


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, iou_threshold=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if _iou(cur, b) < iou_threshold]
    return keep


def detect_sign_regions(frame_bgr) -> List[Tuple[int, int, int, int]]:
    """Run both circle and triangle detectors and merge results."""
    boxes = _detect_circles(frame_bgr) + _detect_triangles(frame_bgr)
    return _nms(boxes)


# ---------------------------------------------------------------------------
# CNN classifier
# ---------------------------------------------------------------------------

def classify_crop(crop_bgr, model, idx_to_class, transform) -> Tuple[str, float]:
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    t = transform(pil).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0]
    prob, idx = probs.max(0)
    folder = idx_to_class.get(idx.item(), f"Class {idx.item()}")
    try:
        name = config.GTSRB_CLASSES.get(int(folder), folder)
    except ValueError:
        name = folder
    return name, prob.item() * 100.0


# ---------------------------------------------------------------------------
# Temporal tracker
# ---------------------------------------------------------------------------

class SignTracker:
    """IoU-based tracker — keeps labels stable across frames."""

    def __init__(self, max_missing: int = 12, iou_threshold: float = 0.25):
        self.tracks: List[Dict] = []
        self.max_missing = max_missing
        self.iou_thr = iou_threshold
        self._nid = 0

    def update(self, detections: List[Tuple]) -> List[Dict]:
        matched_t, matched_d = set(), set()
        for di, (x1, y1, x2, y2, label, conf) in enumerate(detections):
            dbox = (x1, y1, x2, y2)
            best_iou, best_ti = self.iou_thr, -1
            for ti, tr in enumerate(self.tracks):
                if ti in matched_t:
                    continue
                iou = _iou(tr["box"], dbox)
                if iou > best_iou:
                    best_iou, best_ti = iou, ti
            if best_ti >= 0:
                self.tracks[best_ti].update(box=dbox, label=label, conf=conf, missing=0)
                matched_t.add(best_ti)
                matched_d.add(di)

        for di, (x1, y1, x2, y2, label, conf) in enumerate(detections):
            if di not in matched_d:
                self.tracks.append(dict(box=(x1,y1,x2,y2), label=label,
                                        conf=conf, missing=0, id=self._nid))
                self._nid += 1

        for ti in range(len(self.tracks)):
            if ti not in matched_t:
                self.tracks[ti]["missing"] += 1

        self.tracks = [t for t in self.tracks if t["missing"] <= self.max_missing]
        return self.tracks


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detection(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
    ly = y1 - 6 if y1 - th - 6 >= 0 else y2 + th + 6
    cv2.rectangle(frame, (x1, ly-th-bl), (x1+tw+4, ly+bl), color, -1)
    cv2.putText(frame, label, (x1+2, ly), font, scale, (0,0,0), thick, cv2.LINE_AA)


def process_frame(frame, tracker, classifier, idx_to_class, transform, cls_threshold):
    boxes = detect_sign_regions(frame)
    detections = []
    for x1, y1, x2, y2 in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, conf = classify_crop(crop, classifier, idx_to_class, transform)
        if conf >= cls_threshold:
            detections.append((x1, y1, x2, y2, label, conf))

    for i, tr in enumerate(tracker.update(detections)):
        x1, y1, x2, y2 = tr["box"]
        draw_detection(frame, x1, y1, x2, y2,
                       f"{tr['label']}  {tr['conf']:.1f}%",
                       COLORS[tr["id"] % len(COLORS)])


# ---------------------------------------------------------------------------
# Image / video runners
# ---------------------------------------------------------------------------

def predict_image(path, classifier, idx_to_class, transform, thr):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read: '{path}'"); return
    process_frame(frame, SignTracker(), classifier, idx_to_class, transform, thr)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, f"{Path(path).stem}_detected.jpg")
    cv2.imwrite(out, frame)
    print(f"[DONE] Saved to '{out}'")


def predict_video(path, classifier, idx_to_class, transform, thr, detect_every: int = 3):
    """Process video with detection every N frames; tracker fills the gaps.

    Args:
        detect_every: Run the detector only on every Nth frame. The tracker
                      keeps labels visible on skipped frames, so output is
                      smooth while processing is ~3× faster.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: '{path}'"); return
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out  = os.path.join(config.OUTPUT_DIR, f"{Path(path).stem}_detected.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"[INFO] Processing '{path}' — {total} frames (detecting every {detect_every} frames)\n")

    # Detection runs on a downscaled copy to speed up HoughCircles
    detect_w = min(width, 640)
    detect_h  = int(height * detect_w / width)
    scale_x = width  / detect_w
    scale_y = height / detect_h

    tracker = SignTracker(max_missing=15)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if count % detect_every == 0:
            # Downscale for fast detection
            small = cv2.resize(frame, (detect_w, detect_h))
            boxes = detect_sign_regions(small)

            # Scale boxes back to original resolution
            scaled_boxes = []
            for x1, y1, x2, y2 in boxes:
                scaled_boxes.append((
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y),
                ))

            # Classify crops from the full-res frame
            detections = []
            for x1, y1, x2, y2 in scaled_boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                label, conf = classify_crop(crop, classifier, idx_to_class, transform)
                if conf >= thr:
                    detections.append((x1, y1, x2, y2, label, conf))
            tracker.update(detections)

        # Always draw current tracker state (smooth, no flicker)
        for i, tr in enumerate(tracker.tracks):
            if tr["missing"] <= tracker.max_missing:
                x1, y1, x2, y2 = tr["box"]
                draw_detection(frame, x1, y1, x2, y2,
                               f"{tr['label']}  {tr['conf']:.1f}%",
                               COLORS[tr["id"] % len(COLORS)])

        writer.write(frame)
        count += 1
        if count % 200 == 0:
            print(f"  [{100*count/total:5.1f}%]  {count}/{total}")

    cap.release(); writer.release()
    print(f"\n[DONE] Saved to '{out}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Detector + Classifier")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--checkpoint", default=config.MODEL_SAVE_PATH)
    parser.add_argument("--conf",       type=float, default=80.0,
                        help="Min CNN confidence %% (default: 80)")
    args = parser.parse_args()

    print("[INFO] Loading CNN classifier...")
    try:
        classifier, idx_to_class, transform = load_classifier(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}"); sys.exit(1)
    print(f"[INFO] Ready — {len(idx_to_class)} classes | {config.DEVICE}\n")

    if not os.path.exists(args.input):
        print(f"[ERROR] Not found: '{args.input}'"); sys.exit(1)

    ext = Path(args.input).suffix.lower()
    if ext in {".jpg",".jpeg",".png",".bmp",".webp",".ppm"}:
        predict_image(args.input, classifier, idx_to_class, transform, args.conf)
    elif ext in {".mp4",".avi",".mov",".mkv",".wmv",".flv"}:
        predict_video(args.input, classifier, idx_to_class, transform, args.conf)
    else:
        print(f"[ERROR] Unsupported type: '{ext}'"); sys.exit(1)


if __name__ == "__main__":
    main()
