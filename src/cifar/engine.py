"""Training / evaluation loops — framework plumbing kept separate from models.

These functions are deliberately model-agnostic: they take any ``nn.Module`` and
a dataloader, so the same engine trains the from-scratch CNN and the transfer
model without change.
"""

from __future__ import annotations

import copy
import time

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


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    criterion: nn.Module | None = None,
    scheduler=None,
    on_epoch_end=None,
) -> dict[str, object]:
    """Run the full train/eval loop for ``epochs`` epochs.

    Tracks the best test accuracy seen across epochs and keeps a deep copy of
    the model's ``state_dict`` from *that* epoch — not merely the final one —
    so callers checkpoint the model that actually earned the reported best
    accuracy, instead of whatever the last epoch happened to produce.

    Returns a dict with:
      * ``history``: per-epoch train/test loss & accuracy lists.
      * ``best_acc``: the best test accuracy seen.
      * ``best_epoch``: the 1-indexed epoch that achieved it.
      * ``best_state``: a deep-copied ``state_dict`` from that epoch.

    ``on_epoch_end(epoch, epochs, train_metrics, test_metrics, elapsed)`` is
    called after each epoch, if given (e.g. for progress printing).
    """
    criterion = criterion or nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, criterion)
        te = evaluate(model, test_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()
        for k, v in {"train_loss": tr["loss"], "train_acc": tr["acc"],
                     "test_loss": te["loss"], "test_acc": te["acc"]}.items():
            history[k].append(v)
        if te["acc"] > best_acc:
            best_acc = te["acc"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if on_epoch_end is not None:
            on_epoch_end(epoch, epochs, tr, te, time.time() - t0)
    return {
        "history": history,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_state": best_state,
    }


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. 'auto' uses CUDA when available, else CPU."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)
