"""Train a compact ternary ResNet-18 on MNIST."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from ternarylayers import ResNetModels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/ternarycnn")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=16, help="First-stage channel count")
    parser.add_argument("--train-samples", type=int, default=12_000, help="Use 0 for all")
    parser.add_argument("--test-samples", type=int, default=2_000, help="Use 0 for all")
    parser.add_argument("--num-workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path, help="Optional output path for the state dict")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny generated dataset instead of downloading MNIST",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(width: int = 16) -> nn.Module:
    if width <= 0:
        raise ValueError("width must be positive")
    return ResNetModels.R18(
        num_classes=10,
        in_ch=1,
        inplanes=width,
        small_stem=True,
        scale_op="mean",
    ).build()


def limit_dataset(dataset: Dataset, size: int, seed: int) -> Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def create_dataloaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if args.smoke_test:
        train_data: Dataset = datasets.FakeData(
            size=128, image_size=(1, 28, 28), num_classes=10, transform=transform
        )
        test_data: Dataset = datasets.FakeData(
            size=64, image_size=(1, 28, 28), num_classes=10, transform=transform
        )
    else:
        train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        train_data = limit_dataset(train_data, args.train_samples, args.seed)
        test_data = limit_dataset(test_data, args.test_samples, args.seed)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, test_loader


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
        batch_size = targets.size(0)
        loss_sum += criterion(logits, targets).item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
    return loss_sum / seen, correct / seen


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.width = min(args.width, 8)
        args.num_workers = 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    train_loader, test_loader = create_dataloaders(args, device)
    model = build_model(args.width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters={parameter_count:,} train={len(train_loader.dataset):,}")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(
            f"epoch={epoch:02d}/{args.epochs:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.2%}"
        )
    print(f"completed in {time.perf_counter() - started:.1f}s")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
