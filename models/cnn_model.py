"""CNN architecture for German Traffic Sign Recognition.

Two options are available:
  1. TrafficSignCNN  — custom 4-block CNN (default)
  2. ResNet18        — pretrained ImageNet backbone, fine-tuned head
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TrafficSignCNN(nn.Module):
    """Custom CNN with 4 convolutional blocks for traffic sign classification.

    Architecture per block: Conv2d → BatchNorm2d → ReLU → MaxPool2d
    Input:  (B, 3, 64, 64)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of output sign categories.
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32, spatial 64 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 → 64, spatial 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 → 128, spatial 16 → 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 128 → 256, spatial 8 → 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Flattened size: 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 64, 64).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _get_resnet18(num_classes: int) -> nn.Module:
    """Return a fine-tuned ResNet18 with replaced classification head.

    Args:
        num_classes: Number of output categories.

    Returns:
        ResNet18 model with custom final FC layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(num_classes: int, use_pretrained_resnet: bool = False) -> nn.Module:
    """Factory function: return the configured model architecture.

    Args:
        num_classes: Number of output sign categories.
        use_pretrained_resnet: If True, return fine-tuned ResNet18;
                               otherwise return the custom TrafficSignCNN.

    Returns:
        Initialized, untrained (or pretrained-backbone) PyTorch model.
    """
    if use_pretrained_resnet:
        print("[MODEL] Using ResNet18 (pretrained ImageNet backbone)")
        return _get_resnet18(num_classes)

    print("[MODEL] Using custom TrafficSignCNN (4-block CNN)")
    return TrafficSignCNN(num_classes)
