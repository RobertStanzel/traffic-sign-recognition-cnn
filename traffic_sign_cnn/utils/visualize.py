"""Plotting utilities: training curves and confusion matrix."""

import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "output/training_curves.png",
) -> None:
    """Save a two-panel figure showing loss and accuracy curves over epochs.

    Args:
        train_losses: Per-epoch mean training loss values.
        val_losses:   Per-epoch mean validation loss values.
        train_accs:   Per-epoch training accuracy (%).
        val_accs:     Per-epoch validation accuracy (%).
        save_path:    File path for the saved PNG image.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, val_losses, label="Val Loss", marker="o", markersize=3)
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(epochs, train_accs, label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, val_accs, label="Val Acc", marker="o", markersize=3)
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.suptitle("Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to '{save_path}'")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    idx_to_class: Dict[int, str],
    save_path: str = "output/confusion_matrix.png",
) -> None:
    """Generate and save a labelled confusion matrix as a PNG.

    Args:
        y_true:       Ground-truth class indices.
        y_pred:       Predicted class indices.
        idx_to_class: Mapping {index → class_name} for axis labels.
        save_path:    File path for the saved PNG image.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)

    num_classes = len(idx_to_class)
    labels = [idx_to_class[i] for i in range(num_classes) if i in idx_to_class]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Scale figure so labels don't overlap for large class counts
    fig_size = max(10, num_classes // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Confusion matrix saved to '{save_path}'")
