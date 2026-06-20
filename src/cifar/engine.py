"""Training / evaluation loops — framework plumbing kept separate from models.

These functions are deliberately model-agnostic: they take any ``nn.Module`` and
a dataloader, so the same engine trains the from-scratch CNN and the transfer
model without change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """One pass over the training data. Returns mean loss and accuracy."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()           # clear gradients from the last step
        logits = model(images)          # forward
        loss = criterion(logits, targets)
        loss.backward()                 # backprop (autograd)
        optimizer.step()                # update parameters

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Evaluate on a loader with gradients disabled (eval mode + no_grad)."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. 'auto' uses CUDA when available, else CPU."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)
