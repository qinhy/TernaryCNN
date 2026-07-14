from __future__ import annotations

from typing import Iterable, List, Tuple

from pydantic import Field
from torch import nn
from torch.nn.init import trunc_normal_
import torch

from .linear import LinearModels, LinearModules
from .convs import Conv2dModels
from .norms import NormModels
from .bit import Bit
from .base import CommonModel, CommonModule

class ConvNeXtV2Models:
    class BasicModel(CommonModel):
        depths: Tuple[int,int,int,int]
        dims: Tuple[int,int,int,int]

        num_classes: int
        in_ch: int = 3

        drop_path_rate: float = 0.0
        head_init_scale: float = 1.0

        kernel_size: int = 7
        mlp_ratio: int = 4
        
        norm_layers: List[NormModels.LayerNorm] = Field(default_factory=list)
        downsample_layers: List[Conv2dModels.Conv2d] = Field(default_factory=list)

        stages: Tuple[List[Conv2dModels.ConvNeXtV2Block],
                      List[Conv2dModels.ConvNeXtV2Block],
                      List[Conv2dModels.ConvNeXtV2Block],
                      List[Conv2dModels.ConvNeXtV2Block]] = Field(default_factory=lambda:([],[],[],[]))

        head_norm: NormModels.LayerNorm = Field(default_factory=lambda: NormModels.LayerNorm(
                                    num_features=-1, eps=1e-6, data_format="channels_last")
        )
        head: LinearModels.Linear = Field(default_factory=lambda: LinearModels.Linear(
                                    in_features=-1, out_features=-1, bias=True)
        )



        def model_post_init(self, context):
            super().model_post_init(context)

            depths = list(self.depths)
            dims = list(self.dims)

            if len(depths) != 4:
                raise ValueError(f"ConvNeXtV2 expects 4 depths, got {depths}")
            if len(dims) != 4:
                raise ValueError(f"ConvNeXtV2 expects 4 dims, got {dims}")

            self.norm_layers.append(
                NormModels.LayerNorm(num_features=dims[0], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(Conv2dModels.Conv2d(
                in_channels=self.in_ch,
                out_channels=dims[0],
                kernel_size=4,
                stride=4,
                scale_op=self.scale_op,
            ))
            for i in range(3):
                self.norm_layers.append(
                    NormModels.LayerNorm(num_features=dims[i], eps=1e-6, data_format="channels_first")
                )
                self.downsample_layers.append(
                    Conv2dModels.Conv2d(
                        in_channels=dims[i],
                        out_channels=dims[i + 1],
                        kernel_size=2,
                        stride=2,
                        scale_op=self.scale_op,
                    )
                )

            dp_rates = [
                x.item()
                for x in torch.linspace(0, self.drop_path_rate, sum(depths))
            ]

            cur = 0
            for stage_idx in range(4):
                for block_idx in range(depths[stage_idx]):
                    self.stages[stage_idx].append(
                        Conv2dModels.ConvNeXtV2Block(
                            dim=dims[stage_idx],
                            kernel_size=self.kernel_size,
                            mlp_ratio=self.mlp_ratio,
                            drop_path=dp_rates[cur + block_idx],
                            scale_op=self.scale_op,
                        )
                    )
                cur += depths[stage_idx]

            self.head_norm.num_features = dims[-1]
            self.head.in_features = dims[-1]
            self.head.out_features = self.num_classes
            self.head.bias=True
            self.head.scale_op=self.scale_op

        def build(self):
            return self._build(self, ConvNeXtV2Modules)

    class Atto(BasicModel):
        depths: Tuple[int,int,int,int] = [2, 2, 6, 2]
        dims: Tuple[int,int,int,int] = [40, 80, 160, 320]

    class Femto(BasicModel):
        depths: Tuple[int,int,int,int] = [2, 2, 6, 2]
        dims: Tuple[int,int,int,int] = [48, 96, 192, 384]

    class Pico(BasicModel):
        depths: Tuple[int,int,int,int] = [2, 2, 6, 2]
        dims: Tuple[int,int,int,int] = [64, 128, 256, 512]

    class Nano(BasicModel):
        depths: Tuple[int,int,int,int] = [2, 2, 8, 2]
        dims: Tuple[int,int,int,int] = [80, 160, 320, 640]

    class Tiny(BasicModel):
        depths: Tuple[int,int,int,int] = [3, 3, 9, 3]
        dims: Tuple[int,int,int,int] = [96, 192, 384, 768]

    class Base(BasicModel):
        depths: Tuple[int,int,int,int] = [3, 3, 27, 3]
        dims: Tuple[int,int,int,int] = [128, 256, 512, 1024]

    class Large(BasicModel):
        depths: Tuple[int,int,int,int] = [3, 3, 27, 3]
        dims: Tuple[int,int,int,int] = [192, 384, 768, 1536]

    class Huge(BasicModel):
        depths: Tuple[int,int,int,int] = [3, 3, 27, 3]
        dims: Tuple[int,int,int,int] = [352, 704, 1408, 2816]


class ConvNeXtV2Modules:
    class Module(CommonModule):
        def __init__(self, para: ConvNeXtV2Models.BasicModel, para_cls=None):
            super().__init__(para, ConvNeXtV2Models, para_cls)
            self.para: ConvNeXtV2Models.BasicModel = self.para
            self.norm_layers = nn.Sequential(*[i.build() for i in para.norm_layers])
            self.downsample_layers = nn.Sequential(*[i.build() for i in para.downsample_layers])
            self.stages = nn.Sequential(*[nn.Sequential(*[i.build() for i in stage]) for stage in para.stages])
            self.head_norm = para.head_norm.build()
            self.head:LinearModules.Linear = para.head.build()

            self.apply(self._init_weights)
            with torch.no_grad():
                self.head.linear.weight.data.mul_(para.head_init_scale)
                if getattr(self.head, "bias", None) is not None:
                    self.head.linear.bias.data.mul_(para.head_init_scale)

        def _init_weights(self, m: nn.Module):
            if isinstance(m, (Bit.Conv2d, Bit.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

        def forward_features(self, x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x = self.norm_layers[0](self.downsample_layers[0](x))
            c2 = self.stages[0](x)

            x = self.downsample_layers[1](self.norm_layers[1](c2))
            c3 = self.stages[1](x)

            x = self.downsample_layers[2](self.norm_layers[2](c3))
            c4 = self.stages[2](x)

            x = self.downsample_layers[3](self.norm_layers[3](c4))
            c5 = self.stages[3](x)

            return c2, c3, c4, c5

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, _, _, c5 = self.forward_features(x)
            x = c5.mean([-2, -1])
            return self.head(self.head_norm(x))

        def clone(self):
            return self.__class__(self.para.model_copy(deep=True))

    class Atto(Module):pass
    class Femto(Module):pass
    class Pico(Module):pass
    class Nano(Module):pass
    class Tiny(Module):pass
    class Base(Module):pass
    class Large(Module):pass
    class Huge(Module):pass