"""Evaluate the best saved model on the validation set.

Run from the traffic_sign_cnn/ directory:
    python evaluate.py
"""

import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.cnn_model import get_model
from utils.dataset_loader import build_dataset
from utils.visualize import plot_confusion_matrix


def evaluate() -> None:
    """Load the best checkpoint and report per-class and overall accuracy."""

    # --- Validate checkpoint exists ------------------------------------
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(
            f"[ERROR] No checkpoint found at '{config.MODEL_SAVE_PATH}'.\n"
            "        Run train.py first to produce a trained model."
        )
        sys.exit(1)

    # --- Load checkpoint -----------------------------------------------
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    idx_to_class: dict = checkpoint["idx_to_class"]
    num_classes: int = checkpoint["num_classes"]
    use_resnet: bool = checkpoint.get("use_pretrained_resnet", False)

    model = get_model(num_classes, use_resnet).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 60)
    print("  German Traffic Sign Recognition — Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: epoch {checkpoint['epoch']}  "
          f"val_loss={checkpoint['val_loss']:.4f}  "
          f"val_acc={checkpoint['val_acc']:.2f}%")
    print(f"  Classes: {num_classes}  |  Device: {config.DEVICE}")
    print("=" * 60)

    # --- Rebuild validation split (same seed = same split as training) -
    try:
        _, val_ds, _ = build_dataset(config.DATA_DIR, config.VAL_SPLIT)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # --- Inference pass ------------------------------------------------
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # --- Overall accuracy ----------------------------------------------
    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    overall_acc = 100.0 * correct / len(all_labels)
    print(f"\n[RESULT] Overall Accuracy: {overall_acc:.2f}%  "
          f"({correct}/{len(all_labels)} correct)\n")

    # --- Per-class accuracy --------------------------------------------
    class_correct: dict = defaultdict(int)
    class_total: dict = defaultdict(int)
    for true, pred in zip(all_labels, all_preds):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    header = f"{'Class':<45}  {'Samples':>7}  {'Acc':>6}"
    print(header)
    print("-" * len(header))

    for idx in sorted(idx_to_class):
        name = idx_to_class[idx]
        total = class_total[idx]
        acc = 100.0 * class_correct[idx] / total if total > 0 else 0.0
        print(f"{name:<45}  {total:>7}  {acc:>5.1f}%")

    # --- Confusion matrix ----------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, idx_to_class, save_path=cm_path)

    print(f"\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    evaluate()
