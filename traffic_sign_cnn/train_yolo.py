"""Train YOLOv8n traffic sign detector on GTSDB.

Prerequisites:
    pip install ultralytics
    python prepare_gtsdb.py   ← run this first to download + convert the dataset

What this trains:
    YOLOv8n — the smallest/fastest YOLO variant (3.2 M params).
    Detects 4 super-classes: prohibitory / danger / mandatory / other.
    After training the CNN classifier handles fine-grained 43-class recognition.

Output:
    models/yolo_detector.pt   ← copied here automatically after training
    detect_predict.py loads this at startup if present.

Run from traffic_sign_cnn/:
    python train_yolo.py
    python train_yolo.py --epochs 150 --imgsz 640 --device 0   (GPU)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

BASE_DIR  = Path(__file__).parent
DATA_YAML = BASE_DIR / "data" / "gtsdb_yolo" / "dataset.yaml"
MODEL_OUT = BASE_DIR / "models" / "yolo_detector.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=120,
                        help="Training epochs (default 120)")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="Input resolution (default 640; use 416 for speed)")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size (default 16; lower if OOM)")
    parser.add_argument("--device",  type=str,   default="cpu",
                        help="'cpu', '0' for GPU 0, '0,1' for multi-GPU")
    parser.add_argument("--weights", type=str,   default="yolov8n.pt",
                        help="Starting weights (default: ImageNet pretrained yolov8n.pt)")
    args = parser.parse_args()

    # ── Checks ────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(
            f"[ERROR] Dataset YAML not found: {DATA_YAML}\n"
            "Run  python prepare_gtsdb.py  first."
        )
        sys.exit(1)

    os.makedirs(BASE_DIR / "models", exist_ok=True)

    print("=" * 60)
    print("  YOLOv8n Traffic Sign Detector — Training")
    print("=" * 60)
    print(f"  Dataset   : {DATA_YAML}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Img size  : {args.imgsz}")
    print(f"  Batch     : {args.batch}")
    print(f"  Device    : {args.device}")
    print("=" * 60)

    model = YOLO(args.weights)

    results = model.train(
        data    = str(DATA_YAML),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = str(BASE_DIR / "runs" / "detect"),
        name    = "gtsdb_yolov8n",

        # ── Augmentation tuned for traffic signs ──────────────────────────
        # Signs have strict colour rules → keep hue jitter small
        hsv_h     = 0.012,
        hsv_s     = 0.65,
        hsv_v     = 0.40,
        # Signs are often viewed at oblique angles
        degrees   = 12,
        perspective = 0.0003,
        # Scale: signs appear at 2 %–25 % of frame area
        scale     = 0.6,
        translate = 0.1,
        # Mosaic + mixup dramatically help small-object detection
        mosaic    = 1.0,
        mixup     = 0.05,
        # No horizontal flip — arrows and text are direction-sensitive
        fliplr    = 0.0,
        flipud    = 0.0,

        # ── Training settings ─────────────────────────────────────────────
        patience  = 20,          # early stopping
        save      = True,
        plots     = True,
        verbose   = True,
    )

    # ── Copy best weights to models/ ─────────────────────────────────────
    run_dir   = Path(results.save_dir)
    best_pt   = run_dir / "weights" / "best.pt"

    if best_pt.exists():
        shutil.copy(best_pt, MODEL_OUT)
        print(f"\n[DONE] Best weights copied to {MODEL_OUT}")
        print(f"       detect_predict.py will automatically use YOLO on next run.")
    else:
        print(f"\n[WARN] Could not find best.pt in {run_dir}")
        print(f"       Copy it manually to {MODEL_OUT}")

    # ── Quick validation report ───────────────────────────────────────────
    print("\n[INFO] Running final validation...")
    val_model = YOLO(str(MODEL_OUT) if MODEL_OUT.exists() else str(best_pt))
    metrics   = val_model.val(data=str(DATA_YAML), imgsz=args.imgsz, device=args.device, workers=0)
    print(f"\n  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
