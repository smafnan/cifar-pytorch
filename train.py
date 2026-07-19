"""Train a CIFAR-10 classifier: from-scratch CNN or transfer-learning ResNet.

Examples
--------
    # Small CNN from scratch (32x32, native resolution):
    python train.py --model cnn --epochs 30

    # Transfer learning from ImageNet ResNet-18 (images upscaled to 224):
    python train.py --model transfer --image-size 224 --epochs 10

    # Fast CPU smoke test (tiny subset, 1 epoch) — proves the loop runs:
    python train.py --model cnn --epochs 1 --subset 256 --no-download

The README's headline comparison (transfer beats from-scratch) comes from running
both with their recommended settings on a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.cifar import (
    build_model,
    count_trainable_params,
    fit,
    get_dataloaders,
    get_device,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="Default depends on model.")
    p.add_argument("--image-size", type=int, default=None,
                   help="32 for cnn, 224 for transfer (set automatically if omitted).")
    p.add_argument("--subset", type=int, default=None, help="Limit samples per split.")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--finetune", action="store_true",
                   help="For transfer: unfreeze the whole backbone.")
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Sensible per-model defaults.
    image_size = args.image_size or (224 if args.model == "transfer" else 32)
    lr = args.lr or (0.001 if args.model == "transfer" else 0.01)
    device = get_device(args.device)
    print(f"Device: {device}  |  model={args.model}  image_size={image_size}  lr={lr}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, image_size=image_size,
        download=not args.no_download, subset=args.subset,
    )

    model_kwargs = {}
    if args.model == "transfer":
        model_kwargs = {"freeze_backbone": not args.finetune,
                        "pretrained": not args.no_download}
    model = build_model(args.model, num_classes=10, **model_kwargs).to(device)
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    # Only optimise parameters that require gradients (matters for a frozen backbone).
    params = [pm for pm in model.parameters() if pm.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def _log_epoch(epoch, epochs, tr, te, elapsed):
        print(f"epoch {epoch + 1:3d}/{epochs}  "
              f"train_loss={tr['loss']:.3f} train_acc={tr['acc']:.3f}  "
              f"test_acc={te['acc']:.3f}  ({elapsed:.1f}s)")

    # `fit` checkpoints the model state from whichever epoch achieves the best
    # test accuracy (not just the final epoch), so reports/<model>.pt always
    # matches the best_test_acc reported below.
    result = fit(
        model, train_loader, test_loader, optimizer, device, args.epochs,
        criterion=criterion, scheduler=scheduler, on_epoch_end=_log_epoch,
    )
    history = result["history"]
    best_acc = result["best_acc"]
    best_epoch = result["best_epoch"]
    best_state = result["best_state"]

    print(f"\nBest test accuracy: {best_acc:.4f} (epoch {best_epoch}/{args.epochs})")
    _plot_history(history, args.output_dir / f"{args.model}_curves.png")
    # Save the best-epoch state, not the final one, so the checkpoint always
    # matches the reported best_test_acc / best_epoch in the metrics JSON.
    torch.save(best_state, args.output_dir / f"{args.model}.pt")
    (args.output_dir / f"{args.model}_metrics.json").write_text(json.dumps({
        "model": args.model, "epochs": args.epochs, "lr": lr,
        "image_size": image_size, "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "trainable_params": count_trainable_params(model),
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"Saved best-epoch model (epoch {best_epoch}) + metrics to {args.output_dir}/")
    return 0


def _plot_history(history: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
