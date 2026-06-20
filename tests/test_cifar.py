"""Tests for the CIFAR pipeline.

These avoid downloading CIFAR-10: they use random tensors and torchvision's
FakeData so they run fast and offline, while still exercising the real models,
transforms, and training loop.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from cifar import (
    SmallCNN,
    build_model,
    build_transforms,
    count_trainable_params,
    evaluate,
    get_device,
    train_one_epoch,
)
from cifar.models import build_transfer_model


# --- models ---------------------------------------------------------------- #

def test_smallcnn_output_shape():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    assert model(x).shape == (4, 10)


def test_transfer_model_frozen_backbone_trains_only_head():
    # Build offline (no pretrained download); check freezing logic.
    model = build_transfer_model(num_classes=10, freeze_backbone=True, pretrained=False)
    trainable = count_trainable_params(model)
    # Only the new fc layer (in_features*10 weights + 10 biases) should train.
    expected_head = model.fc.in_features * 10 + 10
    assert trainable == expected_head


def test_transfer_model_finetune_trains_everything():
    frozen = build_transfer_model(num_classes=10, freeze_backbone=True, pretrained=False)
    full = build_transfer_model(num_classes=10, freeze_backbone=False, pretrained=False)
    assert count_trainable_params(full) > count_trainable_params(frozen)


def test_build_model_factory():
    assert isinstance(build_model("cnn"), SmallCNN)


# --- transforms ------------------------------------------------------------ #

def test_train_transform_outputs_normalised_tensor():
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))
    t = build_transforms(image_size=32, train=True)(img)
    assert t.shape == (3, 32, 32)
    # Normalised data should have values outside [0,1] (mean subtracted).
    assert t.min() < 0


def test_resize_transform_changes_size():
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))
    t = build_transforms(image_size=224, train=False)(img)
    assert t.shape == (3, 224, 224)


# --- training loop --------------------------------------------------------- #

def _fake_loader(n=64, size=32):
    import torchvision.transforms as T
    ds = FakeData(size=n, image_size=(3, size, size), num_classes=10,
                  transform=T.ToTensor())
    return DataLoader(ds, batch_size=16)


def test_training_step_runs_and_updates_params():
    device = get_device("cpu")
    model = SmallCNN().to(device)
    before = [p.clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = train_one_epoch(model, _fake_loader(), opt, device)
    assert set(metrics) == {"loss", "acc"}
    # At least one parameter must have changed after a training epoch.
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_runs_in_no_grad():
    device = get_device("cpu")
    model = SmallCNN().to(device)
    metrics = evaluate(model, _fake_loader(), device)
    assert 0.0 <= metrics["acc"] <= 1.0
