"""Tests for the `train.py` CLI entry point.

These never touch the network or download real CIFAR-10 / ImageNet weights:
``get_dataloaders`` is monkeypatched with a small offline ``FakeData`` stand-in,
and any real weight download is intercepted before it happens.
"""

from __future__ import annotations

import json

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

import train


def _fake_dataloaders(*, batch_size=8, image_size=32, download=True, subset=None, **_):
    ds = FakeData(size=16, image_size=(3, image_size, image_size), num_classes=10,
                  transform=T.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader


def test_train_cnn_smoke(tmp_path, monkeypatch):
    """The CLI runs end-to-end for the from-scratch CNN and writes its outputs."""
    monkeypatch.setattr(train, "get_dataloaders", _fake_dataloaders)

    rc = train.main(["--model", "cnn", "--epochs", "1", "--output-dir", str(tmp_path)])

    assert rc == 0
    assert (tmp_path / "cnn.pt").exists()
    assert (tmp_path / "cnn_curves.png").exists()
    metrics = json.loads((tmp_path / "cnn_metrics.json").read_text())
    assert metrics["model"] == "cnn"
    assert metrics["best_epoch"] == 1


def test_no_download_does_not_disable_pretrained(tmp_path, monkeypatch):
    """Regression test: `--no-download` must only skip the dataset download, not
    ImageNet-pretrained weight loading for the transfer model (they used to be
    conflated behind the same flag)."""
    monkeypatch.setattr(train, "get_dataloaders", _fake_dataloaders)
    captured: dict = {}
    real_build_model = train.build_model

    def spy_build_model(name, num_classes=10, **kwargs):
        captured.update(kwargs)
        # Force the actual construction offline regardless of what the CLI
        # requested, so this test never downloads real ImageNet weights.
        return real_build_model(name, num_classes=num_classes, **{**kwargs, "pretrained": False})

    monkeypatch.setattr(train, "build_model", spy_build_model)

    rc = train.main([
        "--model", "transfer", "--epochs", "1", "--no-download",
        "--output-dir", str(tmp_path),
    ])

    assert rc == 0
    assert captured["pretrained"] is True  # --no-download alone must not disable it


def test_no_pretrained_flag_disables_pretrained(tmp_path, monkeypatch):
    """`--no-pretrained` is the dedicated flag for skipping pretrained weights."""
    monkeypatch.setattr(train, "get_dataloaders", _fake_dataloaders)
    captured: dict = {}
    real_build_model = train.build_model

    def spy_build_model(name, num_classes=10, **kwargs):
        captured.update(kwargs)
        return real_build_model(name, num_classes=num_classes, **kwargs)

    monkeypatch.setattr(train, "build_model", spy_build_model)

    rc = train.main([
        "--model", "transfer", "--epochs", "1", "--no-pretrained",
        "--output-dir", str(tmp_path),
    ])

    assert rc == 0
    assert captured["pretrained"] is False
