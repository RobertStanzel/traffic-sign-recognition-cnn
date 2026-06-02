"""CNN architecture for German Traffic Sign Recognition.

Two options controlled by config.USE_PRETRAINED_RESNET:
  1. TrafficSignCNN — custom 4-block CNN (fast, no internet needed)
  2. ResNet18       — pretrained ImageNet backbone with upgraded head (recommended)

ResNet18 upgrade over the baseline:
  - Two-layer classification head instead of a single FC
  - Dropout before each FC layer
  - expose freeze_backbone() / unfreeze_backbone() for progressive fine-tuning
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Custom CNN (kept as fallback)
# ---------------------------------------------------------------------------

class TrafficSignCNN(nn.Module):
    """4-block custom CNN. Input: (B, 3, IMG_SIZE, IMG_SIZE). Output: logits."""

    def __init__(self, num_classes: int, img_size: int = 96):
        super().__init__()
        # Each MaxPool halves spatial dims: 96 → 48 → 24 → 12 → 6
        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
        )
        flat = 256 * (img_size // 16) ** 2   # 256 * 6 * 6 = 9216 for img_size=96
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat, 512), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


# ---------------------------------------------------------------------------
# ResNet-18 with upgraded head
# ---------------------------------------------------------------------------

class ResNet18Classifier(nn.Module):
    """Fine-tuned ResNet-18 with a two-layer dropout head.

    Progressive fine-tuning API:
        model.freeze_backbone()    # freeze everything except the head
        model.unfreeze_backbone()  # unfreeze all layers for full fine-tuning
    """

    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Keep everything except the original FC
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # output: (B, 512, 1, 1)

        # Two-layer head: more capacity + regularisation vs a single linear layer
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self) -> None:
        """Freeze backbone weights — only the head will be trained."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all weights for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(
    num_classes: int,
    use_pretrained_resnet: bool = False,
    img_size: int = 96,
) -> nn.Module:
    """Return the configured model.

    Args:
        num_classes:          Number of output sign categories.
        use_pretrained_resnet: True → ResNet18Classifier, False → TrafficSignCNN.
        img_size:             Input spatial resolution (used only by custom CNN).
    """
    if use_pretrained_resnet:
        print("[MODEL] ResNet18 (pretrained ImageNet) + 2-layer dropout head")
        return ResNet18Classifier(num_classes)

    print(f"[MODEL] Custom TrafficSignCNN (4-block, input {img_size}×{img_size})")
    return TrafficSignCNN(num_classes, img_size=img_size)
