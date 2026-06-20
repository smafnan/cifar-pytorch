"""Two models to compare: a from-scratch CNN and a transfer-learning ResNet.

The project's "Done when" is to show the transfer-learning model beats the
from-scratch one *and know why*. The why: the ResNet backbone arrives already
knowing low-level visual features (edges, textures, shapes) learned from
ImageNet's ~1.2M images, so it needs far less CIFAR data to reach high accuracy.
The small CNN must learn everything from 50k images alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """A compact VGG-style CNN for 32x32 CIFAR images.

    Three conv blocks (conv -> BN -> ReLU -> conv -> BN -> ReLU -> maxpool)
    progressively halve the spatial size (32 -> 16 -> 8 -> 4) while growing the
    channel count, then a small classifier head maps to 10 logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 64),    # 32 -> 16
            block(64, 128),  # 16 -> 8
            block(128, 256), # 8  -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_transfer_model(
    num_classes: int = 10, freeze_backbone: bool = True, pretrained: bool = True
) -> nn.Module:
    """A ResNet-18 with its final layer replaced for CIFAR-10.

    Parameters
    ----------
    freeze_backbone:
        If True, train only the new classification head (fast "feature
        extraction"). If False, fine-tune the whole network (slower, usually a
        bit more accurate).
    pretrained:
        Load ImageNet weights. Set False to construct the architecture offline
        without downloading weights (e.g. in CI).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False  # frozen feature extractor

    # Replace the 1000-class ImageNet head with a fresh 10-class head. Its
    # parameters have requires_grad=True by default, so they will train.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """Factory dispatching on a model name."""
    if name == "cnn":
        return SmallCNN(num_classes)
    if name == "transfer":
        return build_transfer_model(num_classes, **kwargs)
    raise ValueError(f"Unknown model '{name}' (expected 'cnn' or 'transfer').")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
