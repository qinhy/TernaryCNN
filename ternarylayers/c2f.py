from __future__ import annotations

from typing import List

import torch
from pydantic import Field, model_validator
from torch import nn

from .acts import ActModels
from .base import CommonModel, CommonModule
from .convs import Conv2dModels
from .helpers import convert_padding, make_divisible
from .norms import NormModels


class C2fBottleneck(CommonModel):
    """Lightweight YOLO-style bottleneck used inside the C2f block."""

    in_channels: int
    out_channels: int
    shortcut: bool = True
    kernel_size: int = 3
    groups: int = 1
    bias: bool = False

    conv_reduce_layer: Conv2dModels.Conv2dNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )
    conv_expand_layer: Conv2dModels.Conv2dNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )

    def build(self):
        return C2fBottleneckModule(self)

    @model_validator(mode="after")
    def valid_model(self):
        self.conv_reduce_layer.in_channels = self.in_channels
        self.conv_reduce_layer.out_channels = self.out_channels
        self.conv_reduce_layer.kernel_size = 1
        self.conv_reduce_layer.stride = 1
        self.conv_reduce_layer.padding = 0
        self.conv_reduce_layer.groups = 1
        self.conv_reduce_layer.bias = self.bias

        self.conv_expand_layer.in_channels = self.out_channels
        self.conv_expand_layer.out_channels = self.out_channels
        self.conv_expand_layer.kernel_size = self.kernel_size
        self.conv_expand_layer.stride = 1
        self.conv_expand_layer.padding = convert_padding("same")
        self.conv_expand_layer.groups = self.groups
        self.conv_expand_layer.bias = self.bias
        return self


class C2f(CommonModel):
    """YOLOv8 C2f block with configurable inner bottlenecks."""

    in_channels: int
    out_channels: int
    num_blocks: int = 1
    expansion: float = 0.5
    shortcut: bool = False
    kernel_size: int = 3
    groups: int = 1
    bias: bool = False

    conv_expand_layer: Conv2dModels.Conv2dNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )
    conv_project_layer: Conv2dModels.Conv2dNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )
    bottleneck_layer: C2fBottleneck = Field(
        default_factory=lambda: C2fBottleneck(in_channels=-1, out_channels=-1)
    )

    hidden_channels: int = Field(default=0, exclude=True)

    def build(self):
        return C2fModule(self)

    @model_validator(mode="after")
    def valid_model(self):
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1 for C2f.")
        if self.expansion <= 0:
            raise ValueError("expansion must be > 0 for C2f.")

        self.hidden_channels = make_divisible(self.out_channels * self.expansion)

        self.conv_expand_layer.in_channels = self.in_channels
        self.conv_expand_layer.out_channels = 2 * self.hidden_channels
        self.conv_expand_layer.kernel_size = 1
        self.conv_expand_layer.stride = 1
        self.conv_expand_layer.padding = 0
        self.conv_expand_layer.bias = self.bias

        concat_channels = (2 + self.num_blocks) * self.hidden_channels
        self.conv_project_layer.in_channels = concat_channels
        self.conv_project_layer.out_channels = self.out_channels
        self.conv_project_layer.kernel_size = 1
        self.conv_project_layer.stride = 1
        self.conv_project_layer.padding = 0
        self.conv_project_layer.bias = self.bias

        self.bottleneck_layer.in_channels = self.hidden_channels
        self.bottleneck_layer.out_channels = self.hidden_channels
        self.bottleneck_layer.shortcut = self.shortcut
        self.bottleneck_layer.kernel_size = self.kernel_size
        self.bottleneck_layer.groups = self.groups
        self.bottleneck_layer.bias = self.bias
        return self


class C2fBottleneckModule(CommonModule):
    def __init__(self, para):
        super().__init__(para, para_cls=C2fBottleneck)
        self.para: C2fBottleneck = self.para
        self.conv_reduce = self.para.conv_reduce_layer.build()
        self.conv_expand = self.para.conv_expand_layer.build()
        self.has_skip = self.para.shortcut and self.para.in_channels == self.para.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_reduce(x)
        x = self.conv_expand(x)
        if self.has_skip:
            x = x + shortcut
        return x

    def to_ternary(self, mods: List[str] = ["conv_reduce", "conv_expand"]):
        self.convert_to_ternary(self, mods)
        return self


class C2fModule(CommonModule):
    def __init__(self, para):
        super().__init__(para, para_cls=C2f)
        self.para: C2f = self.para
        self.conv_expand = self.para.conv_expand_layer.build()
        self.conv_project = self.para.conv_project_layer.build()
        self.blocks = nn.ModuleList(
            [
                C2fBottleneckModule(self.para.bottleneck_layer.model_copy(deep=True))
                for _ in range(self.para.num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1, y2 = torch.chunk(self.conv_expand(x), 2, dim=1)
        feats = [y1, y2]
        for block in self.blocks:
            feats.append(block(feats[-1]))
        x = torch.cat(feats, dim=1)
        x = self.conv_project(x)
        return x

    def to_ternary(self, mods: List[str] = ["conv_expand", "blocks", "conv_project"]):
        self.convert_to_ternary(self, mods)
        return self

