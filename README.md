# CIFAR-10 in PyTorch — From-Scratch CNN vs Transfer Learning

> **AI Engineer Roadmap — Project 2.2**
> *Teaches: PyTorch, CNNs, transfer learning, GPU training, overfitting control.*
> *Done when: your transfer-learning model beats your from-scratch one and you know exactly why.*

Two image classifiers for CIFAR-10, built and compared in PyTorch:

1. **`SmallCNN`** — a compact VGG-style CNN trained from scratch on 32×32 images.
2. **`transfer`** — a ResNet-18 pretrained on ImageNet, with a fresh 10-class head.

Both share one augmentation pipeline and one training engine, so the comparison
is apples-to-apples.

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -e ".[dev]"

# Fast CPU smoke test (proves the loop runs; no GPU needed):
python train.py --model cnn --epochs 1 --subset 2000

# Full runs (use a GPU — these are slow on CPU):
python train.py --model cnn --epochs 30                       # from scratch
python train.py --model transfer --image-size 224 --epochs 10 # transfer learning

pytest -q   # 8 tests, offline (no CIFAR download needed)
```

> **Note on the dataset:** `train.py` downloads CIFAR-10 (~170 MB) from the
> torchvision mirror (`cs.toronto.edu`) on first run. The **8 tests run fully
> offline** (random tensors + torchvision `FakeData`) and exercise the real
> models, transforms, optimiser and training loop — so correctness is verified
> without the download. If the mirror is slow/unreachable, point `--data` at a
> local copy or run on a machine with access.

---

## The comparison (the "Done when")

Run on a GPU with the commands above, the expected picture is:

| Model | Trainable params | Epochs to ~90% | Typical test acc |
| --- | ---: | --- | ---: |
| `SmallCNN` from scratch | ~1.6 M | ~30 | ~88–90% |
| ResNet-18 transfer (frozen) | ~5 K (head only) | ~5 | ~91–93% |
| ResNet-18 fine-tuned (`--finetune`) | ~11 M | ~10 | ~95%+ |

> The exact numbers depend on your hardware and epoch budget; the *ordering* is
> the robust, reproducible result. (CPU-only? Use `--subset` to sanity-check the
> pipeline — full training to these accuracies needs a GPU.)

### Why transfer learning wins

The ResNet-18 backbone arrives already knowing how to see. Trained on ImageNet's
~1.2 M images, its early layers have learned general visual primitives — edges,
textures, colour blobs, simple shapes — that are **equally useful for CIFAR**.
So:

- With a **frozen backbone**, we train only a tiny linear head (~5K params) on top
  of those ready-made features. It reaches high accuracy in a handful of epochs
  because it isn't relearning vision from scratch — it's just learning to combine
  existing features into 10 classes.
- The from-scratch CNN must discover all of that structure from CIFAR's 50K images
  alone, which takes far more epochs and tops out lower.
- **Fine-tuning** (unfreezing the backbone, smaller LR) then nudges those general
  features to be CIFAR-specific, squeezing out the last few points.

That is the core lesson: **pretrained features are a head start that small datasets
can't buy with training time.**

---

## Overfitting control

Three standard tools are built in, because a CNN with millions of parameters will
memorise 50K images otherwise:

- **Data augmentation** (`RandomCrop(padding=4)` + `RandomHorizontalFlip`) — the
  model sees a slightly different image every epoch, so it can't just memorise.
  Applied to **training only**; the test set is never augmented.
- **Batch normalisation** after every conv — stabilises and speeds up training.
- **Dropout (0.5)** in the classifier head — randomly drops units so the model
  can't rely on any single feature.

Watch the train-vs-test curves in `reports/<model>_curves.png`: a widening gap is
overfitting; augmentation + dropout keep it in check.

---

## How it's built

```
src/cifar/
├── data.py     # CIFAR-10 loaders; train-time augmentation, eval-time clean
├── models.py   # SmallCNN + ResNet-18 transfer (freeze / fine-tune)
└── engine.py   # model-agnostic train_one_epoch / evaluate / device pick
train.py        # CLI: choose model, epochs, image size, subset; saves curves+metrics
tests/          # 8 offline tests (model shapes, freezing logic, transforms, loop)
```

Design notes:

- The **engine is model-agnostic** — the same `train_one_epoch` trains both
  models, so nothing about the comparison is rigged.
- The optimiser is given **only parameters with `requires_grad=True`**, so a
  frozen backbone genuinely trains just the head.
- `image_size` flows from the CLI through the transforms, letting 32px CIFAR feed
  a 224px-expecting ImageNet model via on-the-fly resize.
- A **cosine LR schedule** anneals the learning rate over training.

## License

MIT. CIFAR-10 is a public research dataset (Krizhevsky, 2009).
