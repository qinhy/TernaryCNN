from __future__ import annotations

import torch

from ternarylayers.bit import Bit


class Linear(Bit.Linear):
    """Ternary linear layer with an ``nn.Linear``-compatible constructor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        bit: bool = True,
        scale_op: str = "median",
    ) -> None:
        if not bit:
            raise ValueError("The DINOv3 BitLinear adapter only supports bit=True")
        super().__init__(in_features, out_features, bias=bias, scale_op=scale_op)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)


class Conv2d(Bit.Conv2d):
    """Ternary convolution layer with optional device and dtype arguments."""

    def __init__(self, *args, device=None, dtype=None, bit: bool = True, **kwargs) -> None:
        if not bit:
            raise ValueError("The DINOv3 BitConv2d adapter only supports bit=True")
        super().__init__(*args, **kwargs)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
