"""Full training loop for German Traffic Sign CNN.

Key upgrades over the baseline:
  - Label smoothing  (CrossEntropyLoss label_smoothing=0.1)
  - OneCycleLR scheduler  (better convergence than ReduceLROnPlateau for pretrained nets)
  - Mixed-precision training  (torch.amp — 2× faster on CUDA, same results)
  - Progressive unfreezing  (freeze ResNet backbone for UNFREEZE_EPOCH epochs, then thaw)
  - Gradient clipping  (prevents exploding gradients during backbone fine-tuning)

Run from traffic_sign_cnn/:
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
from models.cnn_model import get_model, ResNet18Classifier
from utils.dataset_loader import build_dataset
from utils.visualize import plot_training_curves


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    clip_grad: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.split(":")[0], enabled=(device == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=(device == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Path(config.MODEL_SAVE_PATH).parent, exist_ok=True)

    print("=" * 65)
    print("  Traffic Sign Recognition — Training")
    print("=" * 65)
    print(f"  Device       : {config.DEVICE}")
    print(f"  Model        : {'ResNet18 (pretrained)' if config.USE_PRETRAINED_RESNET else 'Custom CNN'}")
    print(f"  Image size   : {config.IMG_SIZE}×{config.IMG_SIZE}")
    print(f"  Epochs       : {config.NUM_EPOCHS}  |  Batch: {config.BATCH_SIZE}  |  LR: {config.LEARNING_RATE}")
    print(f"  Label smooth : {config.LABEL_SMOOTHING}")
    if config.USE_PRETRAINED_RESNET:
        print(f"  Unfreeze at  : epoch {config.UNFREEZE_EPOCH}")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    try:
        train_ds, val_ds, idx_to_class = build_dataset(config.DATA_DIR, config.VAL_SPLIT)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    num_classes  = len(idx_to_class)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(config.DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(config.DEVICE == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(num_classes, config.USE_PRETRAINED_RESNET, config.IMG_SIZE)
    model = model.to(config.DEVICE)

    # Progressive unfreezing: start with frozen backbone so only the head
    # trains first. This prevents destroying pretrained features with a
    # large random gradient from the randomly-initialised head.
    if config.USE_PRETRAINED_RESNET and isinstance(model, ResNet18Classifier):
        model.freeze_backbone()
        print(f"[INFO] Backbone frozen — will unfreeze at epoch {config.UNFREEZE_EPOCH}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # OneCycleLR: warmup + cosine annealing in one pass — works well for
    # fine-tuning pretrained nets and converges faster than ReduceLROnPlateau.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.NUM_EPOCHS,
        pct_start=0.1,       # 10 % of training is warmup
        anneal_strategy="cos",
    )

    # AMP scaler — does nothing on CPU (enabled=False), speeds up CUDA.
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == "cuda"))

    # ── Training state ────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    backbone_unfrozen = False

    print(f"\n[INFO] Starting — up to {config.NUM_EPOCHS} epochs "
          f"(early-stop patience {config.EARLY_STOPPING_PATIENCE})\n")

    t_start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):

        # ── Progressive unfreeze ─────────────────────────────────────────
        if (config.USE_PRETRAINED_RESNET
                and isinstance(model, ResNet18Classifier)
                and not backbone_unfrozen
                and epoch >= config.UNFREEZE_EPOCH):
            model.unfreeze_backbone()
            backbone_unfrozen = True
            # Re-build optimiser so backbone params are included
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE * 0.1,   # lower LR for backbone layers
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.LEARNING_RATE * 0.1,
                steps_per_epoch=len(train_loader),
                epochs=config.NUM_EPOCHS - epoch + 1,
                pct_start=0.05,
                anneal_strategy="cos",
            )
            print(f"[INFO] Epoch {epoch}: backbone unfrozen — full fine-tuning begins (LR×0.1)")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config.DEVICE
        )
        scheduler.step()
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);   val_accs.append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{config.NUM_EPOCHS}]  "
            f"Train loss={train_loss:.4f} acc={train_acc:6.2f}%  |  "
            f"Val loss={val_loss:.4f} acc={val_acc:6.2f}%  |  "
            f"lr={lr_now:.2e}  ({time.time()-t0:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch":                  epoch,
                    "model_state_dict":       model.state_dict(),
                    "optimizer_state_dict":   optimizer.state_dict(),
                    "val_loss":               val_loss,
                    "val_acc":                val_acc,
                    "idx_to_class":           idx_to_class,
                    "num_classes":            num_classes,
                    "use_pretrained_resnet":  config.USE_PRETRAINED_RESNET,
                    "img_size":               config.IMG_SIZE,
                },
                config.MODEL_SAVE_PATH,
            )
            print(f"           ✓ Best saved (val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"           · No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch}.")
                break

    elapsed = time.time() - t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)

    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(config.OUTPUT_DIR, "training_curves.png"),
    )
    print(f"\n[DONE] {h:02d}h {m:02d}m {s:02d}s — best val_loss={best_val_loss:.4f}")
    print(f"       Checkpoint → {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
