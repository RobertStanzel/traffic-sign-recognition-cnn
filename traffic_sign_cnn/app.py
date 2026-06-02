"""
Flask web server for TrafficSense.
Drop an image or video → get back annotated result + detected sign list.

Run from inside traffic_sign_cnn/:
    pip install flask
    python app.py

Then open http://localhost:5000
"""

import os
import sys
import uuid
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory

# ── make sure project modules are importable ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from models.cnn_model import get_model
from detect_predict import detect_sign_regions, classify_crop, draw_detection, SignTracker, COLORS

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB upload limit

UPLOAD_DIR = os.path.join(BASE_DIR, '_uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, config.OUTPUT_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model once at startup ────────────────────────────────────────────────
print("[TrafficSense] Loading CNN classifier...")
_ckpt         = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
_model        = get_model(_ckpt["num_classes"], _ckpt.get("use_pretrained_resnet", False))
_model.load_state_dict(_ckpt["model_state_dict"])
_model.to(config.DEVICE).eval()
_idx_to_class = _ckpt["idx_to_class"]
_transform    = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])
print(f"[TrafficSense] Ready — {len(_idx_to_class)} classes on {config.DEVICE}")

# ── Async job store (for videos) ──────────────────────────────────────────────
_jobs      = {}
_jobs_lock = threading.Lock()


# ── Image processing ──────────────────────────────────────────────────────────

def _process_image(in_path: str, out_path: str, threshold: float = 80.0):
    frame = cv2.imread(in_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {in_path}")

    boxes = detect_sign_regions(frame)
    detections = []
    for x1, y1, x2, y2 in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, conf = classify_crop(crop, _model, _idx_to_class, _transform)
        if conf >= threshold:
            color = COLORS[len(detections) % len(COLORS)]
            draw_detection(frame, x1, y1, x2, y2, f"{label}  {conf:.1f}%", color)
            detections.append({"label": label, "conf": round(conf, 1), "box": [x1, y1, x2, y2]})

    cv2.imwrite(out_path, frame)
    return detections


# ── Video processing (runs in background thread) ──────────────────────────────

def _auto_detect_every(fps: float, total_frames: int) -> int:
    """Scale frame-skip based on video length to keep processing time reasonable.
    Short (<2 min): every 3 frames  — smooth tracking
    Medium (2-10 min): every 5 frames
    Long (>10 min): every 8 frames  — still catches every sign, just less often
    """
    duration_min = total_frames / max(fps, 1) / 60
    if duration_min < 2:
        return 3
    if duration_min < 10:
        return 5
    return 8


def _process_video_thread(job_id: str, in_path: str, out_path: str,
                           threshold: float = 80.0):
    try:
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {in_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detect_every = _auto_detect_every(fps, total)

        # Expose video metadata to the frontend immediately
        with _jobs_lock:
            _jobs[job_id].update({
                "total_frames": total,
                "fps":          round(fps, 2),
                "duration_sec": round(total / fps, 1) if fps else 0,
                "detect_every": detect_every,
            })

        # mp4v is the most reliable codec across OpenCV builds on Windows
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open VideoWriter")

        detect_w = min(width, 640)
        detect_h = int(height * detect_w / width)
        scale_x  = width  / detect_w
        scale_y  = height / detect_h
        tracker  = SignTracker(max_missing=15)
        seen: dict[str, float] = {}   # label → best conf across all frames
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % detect_every == 0:
                small = cv2.resize(frame, (detect_w, detect_h))
                boxes = detect_sign_regions(small)
                scaled = [
                    (int(x1 * scale_x), int(y1 * scale_y),
                     int(x2 * scale_x), int(y2 * scale_y))
                    for x1, y1, x2, y2 in boxes
                ]
                detections = []
                for x1, y1, x2, y2 in scaled:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    label, conf = classify_crop(crop, _model, _idx_to_class, _transform)
                    if conf >= threshold:
                        detections.append((x1, y1, x2, y2, label, conf))
                        seen[label] = max(seen.get(label, 0), conf)
                tracker.update(detections)

            for tr in tracker.tracks:
                if tr["missing"] <= tracker.max_missing:
                    x1, y1, x2, y2 = tr["box"]
                    draw_detection(frame, x1, y1, x2, y2,
                                   f"{tr['label']}  {tr['conf']:.1f}%",
                                   COLORS[tr["id"] % len(COLORS)])

            writer.write(frame)
            count += 1

            if total > 0:
                with _jobs_lock:
                    _jobs[job_id]["progress"]      = int(100 * count / total)
                    _jobs[job_id]["current_frame"] = count

        cap.release()
        writer.release()

        det_list = sorted(
            [{"label": k, "conf": round(v, 1)} for k, v in seen.items()],
            key=lambda d: -d["conf"],
        )

        with _jobs_lock:
            _jobs[job_id].update({
                "status":     "done",
                "progress":   100,
                "result_url": f"/result/{Path(out_path).name}",
                "detections": det_list,
            })

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})


# ── Routes ────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "roi_interface.html"))


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file attached"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext      = Path(file.filename).suffix.lower()
    uid      = str(uuid.uuid4())[:8]
    conf_thr = float(request.form.get("conf", 80.0))

    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    in_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    file.save(in_path)

    if ext in IMAGE_EXTS:
        out_name = f"{uid}_detected.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        try:
            detections = _process_image(in_path, out_path, conf_thr)
            return jsonify({
                "type":       "image",
                "url":        f"/result/{out_name}",
                "detections": detections,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # video — async
    out_name = f"{uid}_detected.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    job_id   = uid

    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "progress": 0}

    t = threading.Thread(
        target=_process_video_thread,
        args=(job_id, in_path, out_path, conf_thr),
        daemon=True,
    )
    t.start()

    return jsonify({"type": "video", "job_id": job_id})


@app.route("/job/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/result/<filename>")
def result(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TrafficSense] Server starting at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
