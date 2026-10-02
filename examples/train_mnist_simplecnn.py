"""Train a tiny ternary SimpleCNN on MNIST and run W1.58A8 integer inference.

Architecture:
    1x28x28
      -> Bit.Conv2d(1, 8, 3, padding=1)
      -> ReLU
      -> MaxPool2d(2)
      -> Flatten
      -> Bit.Linear(8*14*14, 10)

Flow:
    1. Train with Bit.Conv2d / Bit.Linear fake-quantized ternary weights.
    2. Convert the trained model to frozen ternary inference layers.
    3. Verify frozen weights are int8 values in {-1, 0, +1}.
    4. Calibrate scalar INT8 activation scales from representative data.
    5. Prepare per-output-channel fixed-point M,n requantization metadata.
    6. Evaluate:
         - train-time/fake-quant model (floating activations)
         - frozen ternary model (floating reconstruction)
         - W1.58A8 integer reference path

The integer path is a correctness/reference implementation, not a speed benchmark.
It is evaluated on CPU because generic PyTorch int32 convolution support varies by backend.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from ternarylayers.bit import (
        Bit,
        convert_to_ternary,
        dequantize_symmetric_int8,
        quantize_symmetric_int8,
    )

class SimpleCNN(nn.Module):
    """Small MNIST CNN using ternary-weight Bit layers."""

    def __init__(self, channels: int = 8, num_classes: int = 10) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.channels = channels
        self.conv = Bit.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=True,
            scale_op="median",
        )
        self.fc = Bit.Linear(
            channels * 14 * 14,
            num_classes,
            bias=True,
            scale_op="median",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SimpleCNNIntegerRunner:
    """Integer runtime wrapper for a converted SimpleCNN.

    The wrapped model must already contain Bit.Conv2dInfer and Bit.LinearInfer,
    and prepare_integer(...) must have been called for both layers.
    """

    def __init__(
        self,
        model: nn.Module,
        input_scale: float,
        hidden_scale: float,
        output_scale: float,
    ) -> None:
        self.model = model
        self.input_scale = float(input_scale)
        self.hidden_scale = float(hidden_scale)
        self.output_scale = float(output_scale)

    @torch.no_grad()
    def forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise ValueError("integer reference runner expects CPU input")

        # Floating input -> A8. For real deployment, uint8 camera/image input
        # could often be converted directly into the model's input integer domain.
        x_q = quantize_symmetric_int8(x, self.input_scale)

        # W1.58A8 -> Acc32 -> M,n -> A8
        x_q = self.model.conv.forward_integer(x_q)

        # ReLU is exact in the same integer scale domain.
        x_q = torch.clamp(x_q, min=0)

        # MaxPool keeps the same activation scale.
        # PyTorch max_pool2d support for int8 can vary, so widen to int32 for
        # this reference operation and cast back; numerically this is exact.
        x_q = F.max_pool2d(x_q.to(torch.int32), kernel_size=2, stride=2).to(torch.int8)

        # Reshape does not alter quantization scale.
        x_q = torch.flatten(x_q, 1)

        # W1.58A8 -> Acc32 -> M,n -> A8 logits
        return self.model.fc.forward_integer(x_q)

    @torch.no_grad()
    def forward_float(self, x: torch.Tensor) -> torch.Tensor:
        """Integer path followed by output dequantization, useful for comparisons."""
        return dequantize_symmetric_int8(self.forward_int8(x), self.output_scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "ternarycnn",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Use 0 for the full MNIST training set.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="Use 0 for the full MNIST test set.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=32,
        help="Number of training batches used for INT8 activation calibration; 0 means all.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(2, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("simplecnn_mnist.pt"),
        help="Output training checkpoint path.",
    )
    parser.add_argument(
        "--save-ternary",
        type=Path,
        default=Path("simplecnn_mnist_ternary_int8.pt"),
        help="Output frozen ternary/integer checkpoint path.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save checkpoints after training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny generated dataset and one epoch.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def limit_dataset(dataset: Dataset, size: int, seed: int) -> Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def create_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    # Keep pixels in [0, 1]. This makes the activation quantization domain easy
    # to inspect and avoids hiding input normalization inside the demo.
    transform = transforms.ToTensor()

    if args.smoke_test:
        train_data: Dataset = datasets.FakeData(
            size=256,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_data: Dataset = datasets.FakeData(
            size=128,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
    else:
        train_data = datasets.MNIST(
            args.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_data = datasets.MNIST(
            args.data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        train_data = limit_dataset(train_data, args.train_samples, args.seed)
        test_data = limit_dataset(test_data, args.test_samples, args.seed)

    options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(train_data, shuffle=True, **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    return train_loader, test_loader


def make_cpu_loader(loader: DataLoader) -> DataLoader:
    """Create a simple non-worker CPU loader over the same dataset."""
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
    )


def int8_scale_from_absmax(max_abs: float) -> float:
    if max_abs <= 0.0:
        return 1.0
    return max_abs / 127.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size

    return loss_sum / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size

    return loss_sum / seen, correct / seen


@torch.no_grad()
def inspect_ternary_weights(model: nn.Module) -> None:
    """Show that converted weights are real int8 ternary tensors."""
    print("\n[frozen ternary weights]")
    for name in ("conv", "fc"):
        layer = getattr(model, name)
        values = torch.unique(layer.weight.detach().cpu()).tolist()
        print(
            f"{name:>4}: dtype={layer.weight.dtype} "
            f"shape={tuple(layer.weight.shape)} values={values}"
        )
        if layer.weight.dtype != torch.int8:
            raise TypeError(f"{name}.weight should be int8 after conversion")
        if not set(values).issubset({-1, 0, 1}):
            raise ValueError(f"{name}.weight contains non-ternary values: {values}")


@torch.no_grad()
def calibrate_integer_scales(
    ternary_model: nn.Module,
    loader: DataLoader,
    max_batches: int,
) -> tuple[float, float, float]:
    """Calibrate scalar activation scales for SimpleCNN.

    We collect maxima for exactly the tensors that cross quantized layer
    boundaries:

        input -> Conv -> ReLU -> MaxPool -> Linear -> logits
          sx              sh                sy

    ReLU and MaxPool preserve the same integer scale, and Flatten is only a
    reshape, so one hidden scale is enough between Conv and Linear.
    """
    ternary_model.eval()

    input_absmax = 0.0
    hidden_absmax = 0.0
    output_absmax = 0.0

    for batch_index, (images, _) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break

        images = images.cpu()
        input_absmax = max(input_absmax, float(images.abs().max().item()))

        x = ternary_model.conv(images)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        hidden_absmax = max(hidden_absmax, float(x.abs().max().item()))

        logits = ternary_model.fc(torch.flatten(x, 1))
        output_absmax = max(output_absmax, float(logits.abs().max().item()))

    input_scale = int8_scale_from_absmax(input_absmax)
    hidden_scale = int8_scale_from_absmax(hidden_absmax)
    output_scale = int8_scale_from_absmax(output_absmax)

    return input_scale, hidden_scale, output_scale


@torch.no_grad()
def prepare_integer_model(
    ternary_model: nn.Module,
    input_scale: float,
    hidden_scale: float,
    output_scale: float,
) -> SimpleCNNIntegerRunner:
    ternary_model.conv.prepare_integer(
        input_scale=input_scale,
        output_scale=hidden_scale,
    )
    ternary_model.fc.prepare_integer(
        input_scale=hidden_scale,
        output_scale=output_scale,
    )

    return SimpleCNNIntegerRunner(
        ternary_model,
        input_scale=input_scale,
        hidden_scale=hidden_scale,
        output_scale=output_scale,
    )


@torch.no_grad()
def evaluate_integer(
    runner: SimpleCNNIntegerRunner,
    loader: DataLoader,
) -> tuple[float, int, int]:
    """Return accuracy plus agreement with frozen floating ternary inference."""
    correct = 0
    agree = 0
    seen = 0

    runner.model.eval()

    for images, targets in loader:
        images = images.cpu()
        targets = targets.cpu()

        logits_q = runner.forward_int8(images)
        pred_int = logits_q.argmax(dim=1)

        logits_float = runner.model(images)
        pred_float = logits_float.argmax(dim=1)

        correct += (pred_int == targets).sum().item()
        agree += (pred_int == pred_float).sum().item()
        seen += targets.numel()

    return correct / seen, agree, seen


def print_fixed_point_metadata(model: nn.Module) -> None:
    print("\n[fixed-point requantization metadata]")
    for name in ("conv", "fc"):
        layer = getattr(model, name)
        print(f"{name}:")
        print(f"  input_scale  = {layer.input_scale:.10g}")
        print(f"  output_scale = {layer.output_scale:.10g}")
        print(f"  M            = {layer.requant_M.detach().cpu().tolist()}")
        print(f"  n            = {layer.requant_n.detach().cpu().tolist()}")


def save_training_checkpoint(
    path: Path,
    model: SimpleCNN,
    args: argparse.Namespace,
    test_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "channels": model.channels,
            "test_accuracy": test_accuracy,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        path,
    )
    print(f"saved training checkpoint: {path}")


def save_ternary_checkpoint(
    path: Path,
    model: nn.Module,
    channels: int,
    input_scale: float,
    hidden_scale: float,
    output_scale: float,
    integer_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "channels": channels,
            "quantization": {
                "scheme": "W1.58A8-Acc32-fixedpoint",
                "input_scale": input_scale,
                "hidden_scale": hidden_scale,
                "output_scale": output_scale,
            },
            "integer_test_accuracy": integer_accuracy,
        },
        path,
    )
    print(f"saved ternary/integer checkpoint: {path}")


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.num_workers = 0
        args.calibration_batches = min(args.calibration_batches, 2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    train_loader, test_loader = create_dataloaders(args, device)

    model = SimpleCNN(channels=args.channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    parameter_count = sum(p.numel() for p in model.parameters())
    print(model)
    print(
        f"device={device} "
        f"parameters={parameter_count:,} "
        f"train={len(train_loader.dataset):,} "
        f"test={len(test_loader.dataset):,}"
    )

    # ------------------------------------------------------------------
    # 1. QAT-style ternary-weight training
    # ------------------------------------------------------------------
    started = time.perf_counter()
    final_test_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        scheduler.step()
        final_test_accuracy = test_accuracy

        print(
            f"epoch={epoch:02d}/{args.epochs:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_accuracy:.2%}"
        )

    print(f"training completed in {time.perf_counter() - started:.1f}s")

    if not args.no_save:
        save_training_checkpoint(args.save, model, args, final_test_accuracy)

    # ------------------------------------------------------------------
    # 2. Convert trained layers to frozen ternary inference layers
    # ------------------------------------------------------------------
    # Keep integer reference inference on CPU.
    ternary_model = convert_to_ternary(deepcopy(model).cpu())
    ternary_model.eval()

    inspect_ternary_weights(ternary_model)

    cpu_train_loader = make_cpu_loader(train_loader)
    cpu_test_loader = make_cpu_loader(test_loader)
    cpu_criterion = nn.CrossEntropyLoss()

    # Floating compatibility path of the frozen ternary model.
    ternary_loss, ternary_accuracy = evaluate(
        ternary_model,
        cpu_test_loader,
        cpu_criterion,
        torch.device("cpu"),
    )
    print(
        "\n[frozen ternary floating reconstruction] "
        f"loss={ternary_loss:.4f} accuracy={ternary_accuracy:.2%}"
    )

    # ------------------------------------------------------------------
    # 3. Activation calibration
    # ------------------------------------------------------------------
    input_scale, hidden_scale, output_scale = calibrate_integer_scales(
        ternary_model,
        cpu_train_loader,
        max_batches=args.calibration_batches,
    )

    print("\n[activation calibration]")
    print(f"input_scale  = {input_scale:.10g}")
    print(f"hidden_scale = {hidden_scale:.10g}")
    print(f"output_scale = {output_scale:.10g}")

    # ------------------------------------------------------------------
    # 4. Convert float scale ratios into integer M,n metadata
    # ------------------------------------------------------------------
    runner = prepare_integer_model(
        ternary_model,
        input_scale=input_scale,
        hidden_scale=hidden_scale,
        output_scale=output_scale,
    )
    print_fixed_point_metadata(ternary_model)

    # ------------------------------------------------------------------
    # 5. W1.58A8 integer inference
    # ------------------------------------------------------------------
    integer_accuracy, agreement_count, seen = evaluate_integer(
        runner,
        cpu_test_loader,
    )
    agreement = agreement_count / seen

    print("\n[W1.58A8 integer reference inference]")
    print(f"accuracy                    = {integer_accuracy:.2%}")
    print(f"agreement with float ternary = {agreement:.2%} ({agreement_count}/{seen})")

    if not args.no_save:
        save_ternary_checkpoint(
            args.save_ternary,
            ternary_model,
            channels=args.channels,
            input_scale=input_scale,
            hidden_scale=hidden_scale,
            output_scale=output_scale,
            integer_accuracy=integer_accuracy,
        )


if __name__ == "__main__":
    main()
