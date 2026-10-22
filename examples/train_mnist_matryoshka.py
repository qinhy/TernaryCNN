"""Sandwich-train a configurable ternary Matryoshka ViT on MNIST.

Example:
    python train_mnist_matryoshka.py --epochs 20 --device auto

Use --arch small, base or large to select the shared network.
Saves one shared student plus compact *learned-weight* exports for its splits.
Exported weights have NOT been frozen into ternary integers;
perform backend-specific quantization/export AFTER extraction.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ternarylayers.dinov3.models.vision_transformer_matryoshka import (
    MatryoshkaClassifier, vit_matryoshka_small,
    vit_matryoshka_base, vit_matryoshka_large,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/mnist'))
    p.add_argument('--save-dir', type=Path, default=Path('checkpoints/mnist-matryoshka'))
    p.add_argument('--arch', choices=('small', 'base', 'large'), default='small')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--limit-train', type=int)
    p.add_argument('--limit-test', type=int)
    p.add_argument('--resume', type=Path)
    return p.parse_args()


def select_device(arg):
    if arg != 'auto':
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def dataset(root, train, limit):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,)*3, (0.3081,)*3),
    ])
    ds = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError('Dataset limit must be > 0')
        ds = Subset(ds, range(min(limit, len(ds))))
    return ds


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    count = 0
    total_loss = {name: 0.0 for name in model.splits}
    correct = {name: 0 for name in model.splits}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = []
            # The sandwich rule uses max, min and intermediate widths at
            # each update. Every branch accumulates shared-parameter grads.
            for name in model.splits:
                logits = model(images, name=name)
                loss = F.cross_entropy(logits, labels)
                losses.append(loss)
                total_loss[name] += float(loss.detach()) * labels.size(0)
                correct[name] += int((logits.argmax(1) == labels).sum())
            if training:
                (sum(losses) / len(losses)).backward()
                optimizer.step()
            count += labels.size(0)
    return {name: {'loss': total_loss[name]/count, 'accuracy': correct[name]/count}
            for name in model.splits}


def main():
    args = args_parser()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type=='cuda')
    train_loader = DataLoader(dataset(args.data_dir, True, args.limit_train),
                              shuffle=True, **loader_args)
    test_loader = DataLoader(dataset(args.data_dir, False, args.limit_test),
                             shuffle=False, **loader_args)

    factories = dict(small=vit_matryoshka_small,
                     base=vit_matryoshka_base, large=vit_matryoshka_large)
    backbone = factories[args.arch](img_size=32, patch_size=8, no_conv=True)
    largest_name = max(backbone.splits, key=lambda name: backbone.splits[name])
    backbone.init_weights()   # parent DINOv3 explicitly exposes init_weights()
    model = MatryoshkaClassifier(backbone=backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_accuracy = 0, -1.0
    best_path = args.save_dir/'best_shared.pt'
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        start_epoch = int(payload['epoch'])
        best_accuracy = float(payload.get('best_accuracy', -1.0))

    for epoch in range(start_epoch + 1, args.epochs+1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, test_loader, device)
        scheduler.step()
        print(f'Epoch {epoch}/{args.epochs}')
        for name in model.splits:
            t, v = train_metrics[name], val_metrics[name]
            print(f'  {name:5s}: train_acc={t["accuracy"]:.4f} val_acc={v["accuracy"]:.4f} '
                  f'val_loss={v["loss"]:.4f}')
        shared = {'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'epoch': epoch,
                  'best_accuracy': max(best_accuracy, val_metrics[largest_name]['accuracy']),
                  'arch': args.arch,
                  'splits': backbone.splits,
                  'metrics': val_metrics}
        torch.save(shared, args.save_dir/'last_shared.pt')
        if val_metrics[largest_name]['accuracy'] >= best_accuracy:
            best_accuracy = val_metrics[largest_name]['accuracy']
            torch.save(shared, best_path)

    if best_path.exists():
        saved = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    for name, (width, depth) in backbone.splits.items():
        compact = model.export_subnetwork(name).eval()
        torch.save({
            'state_dict': compact.state_dict(),
            'config': {'name': name, 'img_size': 32, 'patch_size': 8,
                       'in_chans': 3, 'no_conv': True, 'embed_dim': width,
                       'num_heads': width//backbone.head_dim, 'depth': depth,
                       'ffn_ratio': compact.backbone.ffn_ratio, 'num_classes': 10},
        }, args.save_dir/f'{name}_extracted.pt')
        print(f'Extracted {name}: width={width} depth={depth} heads={width//backbone.head_dim}')


if __name__ == '__main__':
    main()
