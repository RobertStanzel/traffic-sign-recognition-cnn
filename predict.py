"""Inference script: classify a single image or annotate a full video.

Usage:
    python predict.py --input path/to/image.jpg
    python predict.py --input path/to/video.mp4
    python predict.py --input path/to/image.jpg --checkpoint models/best_model.pth
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from models.cnn_model import get_model


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str = config.MODEL_SAVE_PATH):
    """Load a trained model and its class mapping from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved .pth checkpoint.

    Returns:
        Tuple of (model, idx_to_class, inference_transform).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'\n"
            "Run train.py first to produce a trained model."
        )

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    idx_to_class: dict = checkpoint["idx_to_class"]
    num_classes: int = checkpoint["num_classes"]
    use_resnet: bool = checkpoint.get("use_pretrained_resnet", False)

    model = get_model(num_classes, use_resnet).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Deterministic, no-augmentation transform for inference
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return model, idx_to_class, transform


# ---------------------------------------------------------------------------
# Image inference
# ---------------------------------------------------------------------------

def _resolve_class_name(idx: int, idx_to_class: dict) -> str:
    """Return a human-readable sign name, preferring the GTSRB global map.

    Folders are sorted alphabetically so internal index != folder number.
    We must convert the folder name to int to get the correct GTSRB label.
    """
    folder_name = idx_to_class.get(idx, f"Class {idx}")
    try:
        return config.GTSRB_CLASSES.get(int(folder_name), folder_name)
    except ValueError:
        return folder_name


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    idx_to_class: dict,
    transform: transforms.Compose,
    top_k: int = 3,
) -> None:
    """Run inference on a single image and print the top-k predictions.

    Args:
        image_path:   Path to the input image file.
        model:        Loaded, eval-mode PyTorch model.
        idx_to_class: Mapping {index → folder_name}.
        transform:    Preprocessing pipeline.
        top_k:        Number of top predictions to display.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top = probs.topk(min(top_k, len(idx_to_class)))

    print(f"\n[RESULT] Image: {image_path}")
    print("-" * 58)
    for rank, (prob, idx) in enumerate(zip(top.values.cpu(), top.indices.cpu()), 1):
        name = _resolve_class_name(idx.item(), idx_to_class)
        print(f"  #{rank}  {name:<45}  {prob.item() * 100:>6.2f}%")
    print()


# ---------------------------------------------------------------------------
# Video inference
# ---------------------------------------------------------------------------

def predict_video(
    video_path: str,
    model: torch.nn.Module,
    idx_to_class: dict,
    transform: transforms.Compose,
) -> None:
    """Process a video frame-by-frame, overlay predictions, and save output.

    The annotated video is saved to config.OUTPUT_DIR with an '_annotated'
    suffix.

    Args:
        video_path:   Path to the input video file.
        model:        Loaded, eval-mode PyTorch model.
        idx_to_class: Mapping {index → folder_name}.
        transform:    Preprocessing pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    stem = Path(video_path).stem
    out_path = os.path.join(config.OUTPUT_DIR, f"{stem}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[INFO] Processing video: '{video_path}'")
    print(f"       Frames: {total_frames}  |  FPS: {fps:.1f}  |  Resolution: {width}×{height}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR frame → PIL RGB for the transform pipeline
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_img).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_prob, top_idx = probs.max(0)
        name = _resolve_class_name(top_idx.item(), idx_to_class)
        label_text = f"{name}  {top_prob.item() * 100:.1f}%"

        # Black banner at top, green text overlay
        cv2.rectangle(frame, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(
            frame, label_text,
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 60),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_count += 1

        if frame_count % 200 == 0:
            pct = 100.0 * frame_count / total_frames if total_frames > 0 else 0
            print(f"  [{pct:5.1f}%]  {frame_count}/{total_frames} frames processed...")

    cap.release()
    writer.release()
    print(f"[DONE] Annotated video saved to '{out_path}'")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and dispatch to image or video inference."""
    parser = argparse.ArgumentParser(
        description="German Traffic Sign Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py --input data/raw/sign.jpg\n"
            "  python predict.py --input data/raw/drive.mp4\n"
        ),
    )
    parser.add_argument("--input", help="Path to an image or video file.")
    parser.add_argument(
        "--checkpoint",
        default=config.MODEL_SAVE_PATH,
        help=f"Path to model checkpoint (default: {config.MODEL_SAVE_PATH})",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of top predictions to show for images (default: 3)",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="Print all 43 recognizable sign classes and exit.",
    )
    args = parser.parse_args()

    # --list-classes: no model needed, just print and exit
    if args.list_classes:
        print("\n  All 43 GTSRB sign classes this model can recognize:\n")
        for idx, name in sorted(config.GTSRB_CLASSES.items()):
            print(f"  {idx:>2}  {name}")
        print()
        sys.exit(0)

    if not args.input:
        parser.error("--input is required unless --list-classes is used.")

    # Load model
    try:
        model, idx_to_class, transform = load_model(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"[INFO] Model loaded — Classes: {len(idx_to_class)}  |  Device: {config.DEVICE}")

    # Validate input path
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: '{args.input}'")
        sys.exit(1)

    ext = Path(args.input).suffix.lower()
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}

    if ext in image_exts:
        predict_image(args.input, model, idx_to_class, transform, top_k=args.top_k)
    elif ext in video_exts:
        predict_video(args.input, model, idx_to_class, transform)
    else:
        print(f"[ERROR] Unsupported file extension: '{ext}'")
        print(f"        Images : {', '.join(sorted(image_exts))}")
        print(f"        Videos : {', '.join(sorted(video_exts))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
