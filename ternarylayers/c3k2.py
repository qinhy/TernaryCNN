from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from pydantic import Field, model_validator
from torch import nn

from .acts import ActModels
from .base import CommonModel, CommonModule
from .convs import Conv2dModels
from .helpers import make_divisible, to_2tuple
from .norms import NormModels


class C3k2Bottleneck(CommonModel):
    """K2 bottleneck used inside YOLOv11 C3k2 blocks."""

    in_channels: int
    out_channels: int
    shortcut: bool = True
    kernel_sizes: Sequence[int] = (3, 5)
    groups: int = 1
    bias: bool = False

    conv_reduce_layer: Conv2dModels.Conv2dNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )
    conv_k1_layer: Conv2dModels.Conv2dDepthwiseNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dDepthwiseNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )
    conv_k2_layer: Conv2dModels.Conv2dDepthwiseNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dDepthwiseNormAct(
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

    kernel_pair: Tuple[int, int] = Field(default=(3, 5), exclude=True)

    def build(self):
        return C3k2BottleneckModule(self)

    @model_validator(mode="after")
    def valid_model(self):
        ks = to_2tuple(self.kernel_sizes)
        self.kernel_pair = ks

        self.conv_reduce_layer.in_channels = self.in_channels
        self.conv_reduce_layer.out_channels = self.out_channels
        self.conv_reduce_layer.kernel_size = 1
        self.conv_reduce_layer.stride = 1
        self.conv_reduce_layer.padding = 0
        self.conv_reduce_layer.bias = self.bias

        for conv_layer, kernel_size in zip((self.conv_k1_layer, self.conv_k2_layer), ks):
            conv_layer.in_channels = self.out_channels
            conv_layer.out_channels = self.out_channels
            conv_layer.kernel_size = kernel_size
            conv_layer.stride = 1
            conv_layer.padding = "same"
            conv_layer.group_size = self.groups
            conv_layer.bias = self.bias

        self.conv_expand_layer.in_channels = self.out_channels
        self.conv_expand_layer.out_channels = self.out_channels
        self.conv_expand_layer.kernel_size = 1
        self.conv_expand_layer.stride = 1
        self.conv_expand_layer.padding = 0
        self.conv_expand_layer.bias = self.bias
        return self


class C3k2(CommonModel):
    """YOLOv11 C3k2 block."""

    in_channels: int
    out_channels: int
    num_blocks: int = 1
    expansion: float = 0.5
    kernel_sizes: Sequence[int] = (3, 5)
    groups: int = 1
    shortcut: bool = True
    bias: bool = False

    conv_expand1_layer: Conv2dModels.Conv2dNormAct = Field(
        default_factory=lambda: Conv2dModels.Conv2dNormAct(
            in_channels=-1,
            norm=NormModels.BatchNorm2d(num_features=-1),
            act=ActModels.SiLU(),
        )
    )
    conv_expand2_layer: Conv2dModels.Conv2dNormAct = Field(
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
    bottleneck_layer: C3k2Bottleneck = Field(
        default_factory=lambda: C3k2Bottleneck(in_channels=-1, out_channels=-1)
    )

    hidden_channels: int = Field(default=0, exclude=True)

    def build(self):
        return C3k2Module(self)

    @model_validator(mode="after")
    def valid_model(self):
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1 for C3k2.")
        if self.expansion <= 0:
            raise ValueError("expansion must be > 0 for C3k2.")

        self.hidden_channels = make_divisible(self.out_channels * self.expansion)

        self.conv_expand1_layer.in_channels = self.in_channels
        self.conv_expand1_layer.out_channels = self.hidden_channels
        self.conv_expand1_layer.kernel_size = 1
        self.conv_expand1_layer.padding = 0
        self.conv_expand1_layer.bias = self.bias

        self.conv_expand2_layer.in_channels = self.in_channels
        self.conv_expand2_layer.out_channels = self.hidden_channels
        self.conv_expand2_layer.kernel_size = 1
        self.conv_expand2_layer.padding = 0
        self.conv_expand2_layer.bias = self.bias

        self.conv_project_layer.in_channels = 2 * self.hidden_channels
        self.conv_project_layer.out_channels = self.out_channels
        self.conv_project_layer.kernel_size = 1
        self.conv_project_layer.padding = 0
        self.conv_project_layer.bias = self.bias

        self.bottleneck_layer.in_channels = self.hidden_channels
        self.bottleneck_layer.out_channels = self.hidden_channels
        self.bottleneck_layer.kernel_sizes = self.kernel_sizes
        self.bottleneck_layer.groups = self.groups
        self.bottleneck_layer.shortcut = self.shortcut
        self.bottleneck_layer.bias = self.bias
        return self


class C3k2BottleneckModule(CommonModule):
    def __init__(self, para):
        super().__init__(para, para_cls=C3k2Bottleneck)
        self.para: C3k2Bottleneck = self.para
        self.conv_reduce = self.para.conv_reduce_layer.build()
        self.conv_k1 = self.para.conv_k1_layer.build()
        self.conv_k2 = self.para.conv_k2_layer.build()
        self.conv_expand = self.para.conv_expand_layer.build()
        self.has_skip = self.para.shortcut and self.para.in_channels == self.para.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_reduce(x)
        x = self.conv_k1(x)
        x = self.conv_k2(x)
        x = self.conv_expand(x)
        if self.has_skip:
            x = x + shortcut
        return x

    def to_ternary(self, mods: List[str] = ["conv_reduce", "conv_k1", "conv_k2", "conv_expand"]):
        self.convert_to_ternary(self, mods)
        return self


class C3k2Module(CommonModule):
    def __init__(self, para):
        super().__init__(para, para_cls=C3k2)
        self.para: C3k2 = self.para
        self.conv_expand1 = self.para.conv_expand1_layer.build()
        self.conv_expand2 = self.para.conv_expand2_layer.build()
        self.conv_project = self.para.conv_project_layer.build()
        self.blocks = nn.ModuleList(
            [
                C3k2BottleneckModule(self.para.bottleneck_layer.model_copy(deep=True))
                for _ in range(self.para.num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv_expand1(x)
        y2 = self.conv_expand2(x)
        for block in self.blocks:
            y1 = block(y1)
        x = torch.cat((y1, y2), dim=1)
        x = self.conv_project(x)
        return x

    def to_ternary(self, mods: List[str] = ["conv_expand1", "conv_expand2", "blocks", "conv_project"]):
        self.convert_to_ternary(self, mods)
        return self
