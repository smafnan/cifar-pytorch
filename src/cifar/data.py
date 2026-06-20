"""CIFAR-10 data loading with augmentation.

Two transform pipelines:

  * **train**: data augmentation (random crop with padding + horizontal flip) so
    the model sees slightly different images every epoch and overfits less,
    followed by normalisation.
  * **eval**: normalisation only — never augment the validation/test set, or the
    metrics become noisy and non-comparable.

``image_size`` lets the *same* CIFAR data feed two very different models:
  * 32 px for the small from-scratch CNN (native CIFAR resolution),
  * 224 px for a pretrained ImageNet backbone, which expects that input size.

We use the standard CIFAR-10 channel mean/std for normalisation.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel statistics of the CIFAR-10 training set (a well-known constant).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transforms(image_size: int = 32, train: bool = True):
    """Return a torchvision transform pipeline."""
    steps = []
    if image_size != 32:
        # Upscale for ImageNet-pretrained backbones.
        steps.append(transforms.Resize(image_size))
    if train:
        # Augmentation: pad-and-crop jitters position; flip adds mirror symmetry.
        steps.append(transforms.RandomCrop(image_size, padding=4))
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
    return transforms.Compose(steps)


def get_dataloaders(
    root: str = "./data",
    batch_size: int = 128,
    image_size: int = 32,
    download: bool = True,
    num_workers: int = 0,
    subset: int | None = None,
):
    """Return ``(train_loader, test_loader)`` for CIFAR-10.

    ``subset`` (if set) limits each split to N samples — handy for a fast smoke
    test or CPU-only runs.
    """
    train_set = datasets.CIFAR10(
        root=root, train=True, download=download,
        transform=build_transforms(image_size, train=True),
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=download,
        transform=build_transforms(image_size, train=False),
    )

    if subset is not None:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        test_set = Subset(test_set, range(min(subset, len(test_set))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader
