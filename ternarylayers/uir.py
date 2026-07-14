from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from pydantic import Field, model_validator
from torch import nn

from .convs import Conv2dModels
from .base import CommonModel, CommonModule
from .helpers import convert_padding, make_divisible
from .pool import PoolModels
from .norms import NormModels
from .drop import DropPath
from .acts import ActModels
from .linear import LinearModels, LinearModules

IntOrPair = Union[int, Tuple[int, int]]
KernelSizeArg = Union[int, Tuple[int, int], Sequence[int]]
PadArg = Union[str, int, Tuple[int, int]]

class UniversalInvertedResidual(CommonModel):
        """Universal Inverted Bottleneck with configurable depthwise stages."""

        in_channels: int
        out_channels: int

        dw_kernel_size_start: int = 0
        dw_kernel_size_mid: int = 3
        dw_kernel_size_end: int = 0

        stride: int = 1
        dilation: int = 1
        group_size: int = 1
        padding: PadArg = 'same'
        bias: bool = False

        noskip: bool = False
        exp_ratio: float = 1.0

        aa_layer: Optional[PoolModels.type] = None
        se_layer: Optional[Conv2dModels.SqueezeExcite] = None

        conv_dw_start_layer: Optional[Union[Conv2dModels.Conv2dDepthwiseNorm]] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )
        conv_pw_exp_layer: Union[Conv2dModels.Conv2dPointwiseNormAct] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv_dw_mid_layer: Optional[Union[Conv2dModels.Conv2dDepthwiseNormAct]] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv_pw_proj_layer: Union[Conv2dModels.Conv2dPointwiseNorm] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )
        
        conv_dw_end_layer: Optional[Union[Conv2dModels.Conv2dNorm]] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )

        drop_path_rate: float = 0.0
        layer_scale_init_value: Optional[float] = 1e-5

        def build(self):
            return UniversalInvertedResidualModule(self)
        
        @model_validator(mode='after')
        def valid_model(self):
            if self.stride > 1 and not (
                self.dw_kernel_size_start or self.dw_kernel_size_mid or self.dw_kernel_size_end
            ):
                raise ValueError("UniversalInvertedResidual needs a depthwise kernel when stride > 1.")

            self.padding = convert_padding(self.padding)

            def num_groups(group_size: Optional[int], channels: int) -> int:
                if not group_size:
                    return 1
                if channels % group_size != 0:
                    raise ValueError("channels must be divisible by group_size.")
                return channels // group_size

            mid_chs = make_divisible(self.in_channels * self.exp_ratio)

            if self.dw_kernel_size_start:
                dw_start_stride = self.stride if not self.dw_kernel_size_mid else 1
                use_aa_start = self.aa_layer is not None and dw_start_stride > 1
                self.conv_dw_start_layer.in_channels = self.in_channels
                self.conv_dw_start_layer.out_channels = self.in_channels
                self.conv_dw_start_layer.kernel_size = self.dw_kernel_size_start
                self.conv_dw_start_layer.stride = 1 if use_aa_start else dw_start_stride
                self.conv_dw_start_layer.padding = self.padding
                self.conv_dw_start_layer.group_size = self.group_size
                self.conv_dw_start_layer.bias = self.bias
                self.conv_dw_start_layer.dilation = self.dilation
            else:
                self.conv_dw_start_layer = None

            self.conv_pw_exp_layer.in_channels = self.in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.kernel_size = 1
            self.conv_pw_exp_layer.stride = 1
            self.conv_pw_exp_layer.padding = self.padding
            self.conv_pw_exp_layer.bias = self.bias

            if self.dw_kernel_size_mid:
                use_aa_mid = self.aa_layer is not None and self.stride > 1
                self.conv_dw_mid_layer.in_channels = mid_chs
                self.conv_dw_mid_layer.out_channels = mid_chs
                self.conv_dw_mid_layer.kernel_size = self.dw_kernel_size_mid
                self.conv_dw_mid_layer.stride = 1 if use_aa_mid else self.stride
                self.conv_dw_mid_layer.padding = self.padding
                self.conv_dw_mid_layer.group_size = self.group_size
                self.conv_dw_mid_layer.bias = self.bias
                self.conv_dw_mid_layer.dilation = self.dilation
            else:
                self.conv_dw_mid_layer = None

            if self.se_layer is not None:
                self.se_layer.in_channels = mid_chs

            self.conv_pw_proj_layer.in_channels = mid_chs
            self.conv_pw_proj_layer.out_channels = self.out_channels
            self.conv_pw_proj_layer.kernel_size = 1
            self.conv_pw_proj_layer.stride = 1
            self.conv_pw_proj_layer.padding = self.padding
            self.conv_pw_proj_layer.bias = self.bias

            if self.dw_kernel_size_end:
                dw_end_stride = self.stride if not (self.dw_kernel_size_start or self.dw_kernel_size_mid) else 1
                if self.aa_layer is not None and dw_end_stride > 1:
                    raise ValueError("Anti-aliasing on the ending depthwise stage with stride > 1 is not supported.")
                self.conv_dw_end_layer.in_channels = self.out_channels
                self.conv_dw_end_layer.out_channels = self.out_channels
                self.conv_dw_end_layer.kernel_size = self.dw_kernel_size_end
                self.conv_dw_end_layer.stride = dw_end_stride
                self.conv_dw_end_layer.padding = self.padding
                self.conv_dw_end_layer.groups = num_groups(self.group_size, self.out_channels)
                self.conv_dw_end_layer.bias = self.bias
                self.conv_dw_end_layer.dilation = self.dilation
            else:
                self.conv_dw_end_layer = None

            if self.aa_layer is not None and hasattr(self.aa_layer, "in_channels"):
                if self.dw_kernel_size_mid and self.stride > 1:
                    self.aa_layer.in_channels = mid_chs
                elif self.dw_kernel_size_start and self.stride > 1 and not self.dw_kernel_size_mid:
                    self.aa_layer.in_channels = self.in_channels

            return self

class UniversalInvertedResidualModule(CommonModule):
        def __init__(self, para):
            super().__init__(para,para_cls=UniversalInvertedResidual)
            self.para: UniversalInvertedResidual = self.para

            self.dw_start = self.para.conv_dw_start_layer.build() if self.para.conv_dw_start_layer else nn.Identity()
            self.pw_exp = self.para.conv_pw_exp_layer.build()
            self.dw_mid = self.para.conv_dw_mid_layer.build() if self.para.conv_dw_mid_layer else nn.Identity()
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.pw_proj = self.para.conv_pw_proj_layer.build()
            self.dw_end = self.para.conv_dw_end_layer.build() if self.para.conv_dw_end_layer else nn.Identity()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()
            self._aa_after = self._resolve_aa_location()
            self.aa_start = self.aa if self._aa_after == "start" else nn.Identity()
            self.aa_mid = self.aa if self._aa_after == "mid" else nn.Identity()

            if self.para.layer_scale_init_value is not None:
                self.layer_scale = LinearModules.LayerScale2d(
                    LinearModels.LayerScale2d(
                        dim=self.para.out_channels,
                        init_values=self.para.layer_scale_init_value,
                    )
                )
            else:
                self.layer_scale = nn.Identity()

            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate else nn.Identity()
            self.has_skip = (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (
                not self.para.noskip
            )

        def _resolve_aa_location(self):
            if self.para.aa_layer is None:
                return None
            if self.para.dw_kernel_size_mid and self.para.stride > 1:
                return "mid"
            if self.para.dw_kernel_size_start and not self.para.dw_kernel_size_mid and self.para.stride > 1:
                return "start"
            return None

        def forward(self, x):
            shortcut = x
            x = self.dw_start(x)
            x = self.aa_start(x)

            x = self.pw_exp(x)
            x = self.dw_mid(x)
            x = self.aa_mid(x)
            x = self.se(x)
            x = self.pw_proj(x)

            x = self.dw_end(x)
            x = self.layer_scale(x)
            x = self.drop_path(x)
            if self.has_skip:
                x = x + shortcut
            return x

        def to_ternary(self, mods=['dw_start', 'pw_exp', 'dw_mid', 'se', 'aa', 'pw_proj', 'dw_end']):
            self.convert_to_ternary(self,mods)
            self.drop_path = nn.Identity()
            return self



