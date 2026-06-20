"""cifar - PyTorch CNN + transfer learning on CIFAR-10."""

from .data import build_transforms, get_dataloaders, CLASSES
from .models import SmallCNN, build_model, build_transfer_model, count_trainable_params
from .engine import train_one_epoch, evaluate, get_device

__all__ = [
    "build_transforms", "get_dataloaders", "CLASSES",
    "SmallCNN", "build_model", "build_transfer_model", "count_trainable_params",
    "train_one_epoch", "evaluate", "get_device",
]
__version__ = "1.0.0"
