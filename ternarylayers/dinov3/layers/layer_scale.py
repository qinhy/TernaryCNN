# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim).to(device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MatryoshkaLayerScale(LayerScale):
    """LayerScale with nested channel widths."""

    def forward(self, x: Tensor) -> Tensor:
        width = x.shape[-1]

        if not 1 <= width <= self.gamma.numel():
            raise ValueError(f"Invalid width: {width}")

        if width == self.gamma.numel():
            return super().forward(x)

        gamma = self.gamma[:width]

        return x.mul_(gamma) if self.inplace else x * gamma
