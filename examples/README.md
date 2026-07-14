# Examples

## MNIST with a ternary ResNet-18

The MNIST example uses the library's `ResNetModels.R18` builder with a one-channel
stem and narrower stages (`16, 32, 64, 128`) for fast local training. Its default
run uses 12,000 training images and 2,000 test images.

```bash
uv run python examples/train_mnist_resnet18.py
```

Run a no-download end-to-end check with generated images:

```bash
uv run python examples/train_mnist_resnet18.py --smoke-test
```

Use the complete MNIST dataset and save the trained weights:

```bash
uv run python examples/train_mnist_resnet18.py \
  --train-samples 0 --test-samples 0 --epochs 5 --save mnist-resnet18.pt
```

Pass `--device cpu`, `--device cuda`, or `--device mps` to override automatic
device selection. Use `--help` for all training options.
