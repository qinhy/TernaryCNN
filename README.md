# TernaryCNN

TernaryCNN is a reusable PyTorch library for training with fake-quantized ternary
weights and converting trained layers into frozen ternary inference modules. It also
contains composable, Pydantic-configured convolution, linear, normalization, pooling,
attention, and vision-model building blocks.

> The project is in an early alpha stage. The low-level `Bit` API is the most stable
> surface; higher-level model configurations may evolve before 1.0.

## Installation with uv

For development from a checkout:

```bash
uv sync --dev
```

To use the library from another uv project before it is published:

```bash
uv add "ternarycnn @ git+https://github.com/qinhy/TernaryCNN.git"
```

Optional features are installed explicitly:

```bash
uv sync --extra augment
uv sync --extra dinov3 --extra dinov3-eval
```

PyTorch is intentionally not pinned to a CUDA build. This keeps the package portable
across CPU, CUDA, and macOS environments. Configure the appropriate PyTorch index in
the consuming application when a specific accelerator build is required.

## Quick start

```python
from copy import deepcopy

import torch

from ternarylayers import Bit, convert_to_ternary

layer = Bit.Linear(16, 4)
x = torch.randn(2, 16)

# Fake-quantized weights with a straight-through estimator during training.
y = layer(x)
y.sum().backward()

# Freeze ternary weights and their per-output scale for inference.
inference_layer = convert_to_ternary(deepcopy(layer)).eval()
prediction = inference_layer(x)
```

Higher-level layers use validated configuration objects:

```python
from ternarylayers import ActModels, LinearModels

config = LinearModels.LinearAct(
    in_features=16,
    out_features=4,
    act=ActModels.GELU(),
)
layer = config.build()
```

## Development

```bash
uv run pytest
uv run ruff check ternarylayers tests
uv build --no-sources
```

`uv.lock` records the reproducible development environment. Library consumers still
resolve compatible versions from the ranges declared in `pyproject.toml`.

## Package layout

- `ternarylayers.bit`: fake-quantized and frozen ternary linear/convolution layers
- `ternarylayers.{convs,linear,norms,pool,acts}`: validated, composable layer factories
- `ternarylayers.aug`: optional image augmentations (`augment` extra)
- `ternarylayers.dinov3`: bundled DINOv3-derived models (`dinov3` extras)

The DINOv3-derived subtree is distributed under its own terms in
[`ternarylayers/dinov3/LICENSE.md`](ternarylayers/dinov3/LICENSE.md).

