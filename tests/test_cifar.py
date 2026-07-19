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
    fit,
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


# --- checkpoint selection (regression test for the final-vs-best-epoch bug) - #

def test_fit_reports_best_epoch_consistent_with_history():
    """`best_epoch`/`best_acc` from `fit()` must agree with `history["test_acc"]`.

    Guards the bug where the checkpoint corresponded to the final epoch while
    the reported accuracy was the best-across-epochs value: whatever epoch
    `fit()` names as best must actually be the (first) epoch that hit the max
    recorded test accuracy.
    """
    device = get_device("cpu")
    torch.manual_seed(0)
    model = SmallCNN(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    train_loader = _fake_loader(n=32, size=32)
    test_loader = _fake_loader(n=32, size=32)

    result = fit(model, train_loader, test_loader, optimizer, device, epochs=4)

    history = result["history"]
    assert set(result) == {"history", "best_acc", "best_epoch", "best_state"}
    assert 1 <= result["best_epoch"] <= 4
    assert result["best_acc"] == max(history["test_acc"])
    # best_epoch is 1-indexed; the history entry at that position must match.
    assert history["test_acc"][result["best_epoch"] - 1] == result["best_acc"]
    assert result["best_state"] is not None


def test_fit_checkpoint_matches_best_epoch_not_final_epoch():
    """The saved state_dict must be the weights *as of the best epoch*.

    Deterministically replays training up to `best_epoch` with a freshly
    seeded model on the same (unshuffled) fake data and checks the resulting
    weights match `best_state` exactly. If `fit()` regressed to saving the
    final-epoch weights instead, this would fail whenever the best epoch
    isn't the last one.
    """
    device = get_device("cpu")
    train_loader = _fake_loader(n=32, size=32)
    test_loader = _fake_loader(n=32, size=32)

    torch.manual_seed(0)
    model = SmallCNN(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    result = fit(model, train_loader, test_loader, optimizer, device, epochs=4)
    best_epoch = result["best_epoch"]
    best_state = result["best_state"]

    # Replay training deterministically (same seed, same unshuffled loaders)
    # up to (and only up to) the best epoch.
    torch.manual_seed(0)
    replay_model = SmallCNN(num_classes=10).to(device)
    replay_optimizer = torch.optim.SGD(replay_model.parameters(), lr=0.5)
    for _ in range(best_epoch):
        train_one_epoch(replay_model, train_loader, replay_optimizer, device)
        evaluate(replay_model, test_loader, device)

    replayed_state = replay_model.state_dict()
    for key in best_state:
        assert torch.allclose(best_state[key], replayed_state[key], atol=1e-6)

    # If the best epoch wasn't the final one, the checkpoint must differ from
    # the final-epoch weights -- this is the actual bug the fix addresses.
    if best_epoch < 4:
        final_state = model.state_dict()
        assert any(
            not torch.allclose(best_state[k], final_state[k], atol=1e-6)
            for k in best_state
        )
