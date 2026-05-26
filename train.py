"""Full training loop for German Traffic Sign CNN.

Run from the traffic_sign_cnn/ directory:
    python train.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.cnn_model import get_model
from utils.dataset_loader import build_dataset
from utils.visualize import plot_training_curves


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Run a single training epoch.

    Args:
        model:     The neural network.
        loader:    DataLoader for the training split.
        criterion: Loss function.
        optimizer: Gradient-descent optimiser.
        device:    'cuda' or 'cpu'.

    Returns:
        (mean_loss, accuracy_percent)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Run a full validation pass without gradient computation.

    Args:
        model:     The neural network (set to eval mode internally).
        loader:    DataLoader for the validation split.
        criterion: Loss function.
        device:    'cuda' or 'cpu'.

    Returns:
        (mean_loss, accuracy_percent)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: full training loop with early stopping and checkpointing."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Path(config.MODEL_SAVE_PATH).parent, exist_ok=True)

    print("=" * 65)
    print("  German Traffic Sign Recognition — Training")
    print("=" * 65)
    print(f"  Device  : {config.DEVICE}")
    print(f"  Model   : {'ResNet18 (pretrained)' if config.USE_PRETRAINED_RESNET else 'Custom CNN'}")
    print(f"  Epochs  : {config.NUM_EPOCHS}  |  Batch: {config.BATCH_SIZE}  |  LR: {config.LEARNING_RATE}")
    print("=" * 65)

    # --- Data -----------------------------------------------------------
    try:
        train_ds, val_ds, idx_to_class = build_dataset(config.DATA_DIR, config.VAL_SPLIT)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    num_classes = len(idx_to_class)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    # --- Model ----------------------------------------------------------
    model = get_model(num_classes, config.USE_PRETRAINED_RESNET).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # --- Training state -------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    print(f"\n[INFO] Starting training — up to {config.NUM_EPOCHS} epochs "
          f"(early stopping patience: {config.EARLY_STOPPING_PATIENCE})\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{config.NUM_EPOCHS}]  "
            f"Train → loss={train_loss:.4f}  acc={train_acc:6.2f}%  |  "
            f"Val → loss={val_loss:.4f}  acc={val_acc:6.2f}%  |  "
            f"lr={lr_now:.2e}  ({elapsed:.1f}s)"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "idx_to_class": idx_to_class,
                    "num_classes": num_classes,
                    "use_pretrained_resnet": config.USE_PRETRAINED_RESNET,
                },
                config.MODEL_SAVE_PATH,
            )
            print(f"           ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"           · No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[INFO] Early stopping after {epoch} epochs — "
                      f"no val_loss improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
                break

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(config.OUTPUT_DIR, "training_curves.png"),
    )
    print(f"\n[DONE] Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"       Checkpoint: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
