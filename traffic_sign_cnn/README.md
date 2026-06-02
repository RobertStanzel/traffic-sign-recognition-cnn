# German Traffic Sign Recognition — CNN Classifier

A PyTorch-based CNN that trains on German traffic sign images and classifies
them at inference time, outputting human-readable sign names (e.g. "Stop",
"Speed limit (50km/h)", "No entry").

Supports the full **43-class GTSRB** label set, plus any custom classes you
organise yourself.

---

## Project Structure

```
traffic_sign_cnn/
├── data/
│   ├── raw/          ← put raw images / videos here
│   ├── frames/       ← extracted video frames land here
│   └── dataset/      ← training data, organised by class
├── models/
│   └── cnn_model.py  ← CNN architecture (custom + ResNet18 option)
├── utils/
│   ├── extract_frames.py   ← video → JPEG frames
│   ├── dataset_loader.py   ← PyTorch Dataset + transforms
│   └── visualize.py        ← training curves, confusion matrix
├── train.py          ← full training loop
├── evaluate.py       ← evaluation + confusion matrix
├── predict.py        ← image / video inference CLI
├── config.py         ← all hyperparameters and paths
└── requirements.txt
```

---

## 1 — Setup

### Requirements

- Python 3.10+
- (Recommended) a virtual environment

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Windows note:** If you encounter multiprocessing errors with the DataLoader,
> set `NUM_WORKERS = 0` in `config.py`.

---

## 2 — Organise Your Dataset

Place images inside `data/dataset/` with **one sub-folder per class**.
The folder name becomes the class label.

```
data/dataset/
    Stop/
        img001.jpg
        img002.jpg
    SpeedLimit50/
        img001.jpg
        img002.jpg
    NoEntry/
        img001.jpg
```

**Using the GTSRB benchmark?**  
Download from https://benchmark.ini.rub.de/ and rename the numbered folders
(e.g. `00014` → `Stop`) to match the names in `GTSRB_CLASSES` in `config.py`,
or leave the numeric names — the model works either way and maps them to
English labels automatically during inference.

---

## 3 — Extract Frames from Videos (optional)

Place `.mp4 / .avi / .mov / .mkv` files in `data/raw/`, then run:

```bash
python utils/extract_frames.py
```

Frames are saved to `data/frames/<video_name>/frame_XXXXXX.jpg`.
Copy the frames you want into the appropriate class folder in `data/dataset/`.

Adjust the extraction interval in `config.py`:
```python
FRAME_EXTRACT_INTERVAL_SEC = 1.0   # 1 frame per second
```

---

## 4 — Configure (optional)

All settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `IMG_SIZE` | 64 | Input image size (pixels) |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `EARLY_STOPPING_PATIENCE` | 7 | Epochs without improvement before stopping |
| `VAL_SPLIT` | 0.2 | Fraction of data used for validation |
| `USE_PRETRAINED_RESNET` | False | Set `True` to use ResNet18 transfer learning |
| `NUM_WORKERS` | 0 | DataLoader workers (set 0 on Windows if errors) |
| `DEVICE` | auto | `"cuda"` if GPU available, else `"cpu"` |

---

## 5 — Train

```bash
python train.py
```

- Saves the best checkpoint to `models/best_model.pth`
- Saves training curves to `output/training_curves.png`
- Early stopping kicks in if validation loss doesn't improve for
  `EARLY_STOPPING_PATIENCE` epochs

**Switch to ResNet18** (better accuracy, slower):
```python
# config.py
USE_PRETRAINED_RESNET = True
```

---

## 6 — Evaluate

```bash
python evaluate.py
```

Outputs:
- Overall accuracy
- Per-class accuracy table
- Confusion matrix saved to `output/confusion_matrix.png`

---

## 7 — Inference

### Single image
```bash
python predict.py --input path/to/sign.jpg
```
Prints top-3 predictions with confidence scores.

### Video
```bash
python predict.py --input path/to/drive.mp4
```
Produces `output/drive_annotated.mp4` with the predicted sign label overlaid
on each frame.

### Custom checkpoint
```bash
python predict.py --input sign.jpg --checkpoint models/best_model.pth
```

---

## Architecture

### Custom CNN (default)

| Block | Layers | Output Shape |
|---|---|---|
| 1 | Conv2d(3→32) + BN + ReLU + MaxPool | 32×32×32 |
| 2 | Conv2d(32→64) + BN + ReLU + MaxPool | 64×16×16 |
| 3 | Conv2d(64→128) + BN + ReLU + MaxPool | 128×8×8 |
| 4 | Conv2d(128→256) + BN + ReLU + MaxPool | 256×4×4 |
| FC | Dropout(0.5) → Linear(4096→512) → ReLU → Dropout(0.3) → Linear(512→N) | N logits |

### ResNet18 (transfer learning)

Pretrained ImageNet backbone with the final FC layer replaced by
`Linear(512, N)`. Set `USE_PRETRAINED_RESNET = True` in `config.py`.

---

## GTSRB Classes

The 43 standard classes are defined in `config.GTSRB_CLASSES`. Examples:

| Index | Name |
|---|---|
| 0 | Speed limit (20km/h) |
| 14 | Stop |
| 17 | No entry |
| 25 | Road work |
| 38 | Keep right |

---

## Tips

- **Not enough data?** Use the ResNet18 option — it generalises better with
  fewer images due to pretrained features.
- **Overfitting?** Reduce `LEARNING_RATE`, increase `WEIGHT_DECAY`, or add
  more images via frame extraction.
- **Slow training on CPU?** Reduce `BATCH_SIZE` and `IMG_SIZE`, or use a
  machine with a CUDA GPU.

---

## Future Improvements

### Real-time Video Detection
The current video pipeline (`predict.py`, `detect_predict.py`) overlays
predictions on every frame but uses heuristic color/shape detection to locate
signs, which produces false positives on real-world footage (flags, vehicles,
reflections). The planned improvement is:

- **Train a YOLOv8 detector** on annotated street-level footage so the system
  can find exact bounding boxes for each sign before passing the crop to the
  CNN classifier. This two-stage pipeline (detect → classify) will be far more
  reliable than the current color/shape heuristics.
- **Fine-tune the CNN on real video crops** to reduce domain mismatch between
  the tightly-cropped GTSRB training images and real dashcam footage.
- **Perspective correction** on each detected crop before classification to
  handle signs viewed at an angle.

### Additional Planned Improvements
- **Larger input resolution** — increase `IMG_SIZE` from 64 to 128 for better
  feature detail, especially on small or distant signs.
- **Test-time augmentation (TTA)** — average predictions over multiple
  augmented versions of each crop for higher confidence on edge cases.
- **ONNX / TensorRT export** — export the trained model for faster CPU/edge
  inference without PyTorch as a dependency.
- **Web UI** — a simple Flask or Gradio interface for drag-and-drop image
  prediction without using the command line.


