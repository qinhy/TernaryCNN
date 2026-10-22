# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Optional

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MatryoshkaRMSNorm(nn.RMSNorm):
    """Optional width-elastic RMSNorm with standard RMSNorm state_dict keys."""

    def __init__(
        self,
        normalized_shape: int,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if not isinstance(normalized_shape, int) or normalized_shape <= 0:
            raise ValueError("MatryoshkaRMSNorm expects a positive channel dimension")
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        width = x.shape[-1]
        max_width = self.normalized_shape[0]
        if not (1 <= width <= max_width):
            raise ValueError(f"Active width must be in [1, {max_width}], got {width}")
        if width == max_width:
            return super().forward(x)
        weight = self.weight[:width] if self.weight is not None else None
        return torch.nn.functional.rms_norm(x, (width,), weight=weight, eps=self.eps)

