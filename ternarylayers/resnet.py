from __future__ import annotations

from typing import List, Tuple, Type, Union

from pydantic import Field
from torch import nn
import torch

from ternarylayers.linear import LinearModels

from .pool import PoolModels
from .convs import Conv2dModels
from .norms import NormModels
from .acts import ActModels
from .base import CommonModel, CommonModule

class ResNetModels:
    class BasicModel(CommonModel):
        act: ActModels.type = ActModels.SiLU(inplace=True)
        norm: NormModels.type = NormModels.BatchNorm2d(num_features=-1)
        blocks: List[int]
        num_classes: int
        expansion: int
        in_ch: int = 3
        small_stem: bool = True
        inplanes:int = 64

        layers:Tuple[List,List,List,List] = Field(default_factory=lambda: ([],[],[],[]))
        stem:Conv2dModels.Conv2dNormAct=None
        stem_pool:Union[PoolModels.MaxPool2d, NormModels.Identity]=None
        head_pool:PoolModels.AdaptiveAvgPool2d=None
        head:LinearModels.Linear=None
        
        def get_block(self,in_channels,out_channels,stride):
            return Conv2dModels.ResNetBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=1,
                    act_layer=self.act.model_copy(),
                    conv_reduce_layer=Conv2dModels.Conv2dNormAct(
                                            in_channels=-1,
                                            norm=self.norm.model_copy(),
                                            act=self.act.model_copy(),
                                            scale_op=self.scale_op,
                                        ),
                    conv_transform_layer=Conv2dModels.Conv2dNormAct(
                                            in_channels=-1,
                                            norm=self.norm.model_copy(),
                                            act=self.act.model_copy(),
                                            scale_op=self.scale_op,
                                        ),
                    conv_expand_layer=Conv2dModels.Conv2dNorm(
                                            in_channels=-1,
                                            norm=self.norm.model_copy(),
                                            scale_op=self.scale_op,
                                        ),
                    scale_op=self.scale_op,
                    bit=True,
                )
        
        def model_post_init(self, context):        
            in_channels = [self.inplanes * (2**stage) for stage in range(4)]
            strides = [1, 2, 2, 2]    

            if self.small_stem:
                # CIFAR / Tiny stem: 3x3 stride 1, no maxpool
                self.stem = Conv2dModels.Conv2dNormAct(
                    in_channels=self.in_ch,
                    out_channels=in_channels[0],
                    kernel_size=3,stride=1,padding=1,
                    scale_op=self.scale_op,
                    bias=False,
                    norm=self.norm.model_copy(),
                    act=self.act.model_copy(),
                )
                self.stem_pool = NormModels.Identity()
            else:
                # ImageNet stem: 7x7 stride 2 + maxpool
                self.stem = Conv2dModels.Conv2dNormAct(
                        in_channels=self.in_ch,
                        out_channels=in_channels[0],
                        kernel_size=7,stride=2,padding=3,
                        scale_op=self.scale_op,
                        bias=False,
                        norm=self.norm.model_copy(),
                        act=self.act.model_copy(),
                    )
                self.stem_pool = PoolModels.MaxPool2d(kernel_size=3, stride=2, padding=1)

            current_in_ch = in_channels[0]
            for i, (num_blocks, stage_ch, stage_stride) in enumerate(zip(self.blocks, in_channels, strides)):
                out_ch = stage_ch * self.expansion

                for block_idx in range(num_blocks):
                    block_stride = stage_stride if block_idx == 0 else 1
                    self.layers[i].append(
                        self.get_block(current_in_ch, out_ch, block_stride)
                    )
                    current_in_ch = out_ch
                    
            self.head_pool = PoolModels.AdaptiveAvgPool2d(output_size=1)
            self.head = LinearModels.Linear(in_features=in_channels[-1] * self.expansion,
                                            out_features=self.num_classes,
                                            bias=True, scale_op=self.scale_op)

            return super().model_post_init(context)

        def build(self): return self._build(self,ResNetModules)
    
    class R18(BasicModel):
        blocks: List[int] = [2, 2, 2, 2]
        num_classes: int
        expansion: int = 1
        in_ch: int = 3
        small_stem: bool = True

        def get_block(self,in_channels,out_channels,stride):
            return Conv2dModels.ResNetBasicBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            stride=stride,
                            padding=1,
                            act_layer=self.act.model_copy(),
                            conv1_layer=Conv2dModels.Conv2dNormAct(
                                            in_channels=-1,
                                            norm=self.norm.model_copy(),
                                            act=self.act.model_copy(),
                                            scale_op=self.scale_op,
                                        ),
                            conv2_layer=Conv2dModels.Conv2dNorm(
                                            in_channels=-1,
                                            norm=self.norm.model_copy(),
                                            scale_op=self.scale_op,
                                        ),
                            scale_op=self.scale_op,
                            bit=True,
                        )

    class R50(BasicModel):
        blocks: List[int] = [3, 4, 6, 3]
        num_classes: int
        expansion: int = 4
        in_ch: int = 3
        small_stem: bool = True
    
    class R101(BasicModel):
        blocks: List[int] = [3, 4, 23, 3]
        num_classes: int
        expansion: int = 4
        in_ch: int = 3
        small_stem: bool = True

class ResNetModules:
    class Module(CommonModule):
        def __init__(self, para:ResNetModels.BasicModel, para_cls=None):
            super().__init__(para, ResNetModels, para_cls)
            self.para:ResNetModels.BasicModel = self.para
            
            self.stem = self.para.stem.build()
            self.stem_pool = self.para.stem_pool.build()

            self.layers = nn.Sequential(
                *[nn.Sequential(*[i.build() for i in blocks]) for blocks in self.para.layers]
            )
            # self.layer1 = nn.Sequential(*[i.build() for i in self.para.layers[0]])
            # self.layer2 = nn.Sequential(*[i.build() for i in self.para.layers[1]])
            # self.layer3 = nn.Sequential(*[i.build() for i in self.para.layers[2]])
            # self.layer4 = nn.Sequential(*[i.build() for i in self.para.layers[3]])

            self.head_pool = self.para.head_pool.build()
            self.head = self.para.head.build()

        def forward_features(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x = self.stem_pool(self.stem(x))
            c2 = self.layers[1-1](x)
            c3 = self.layers[2-1](c2)
            c4 = self.layers[3-1](c3)
            c5 = self.layers[4-1](c4)
            return c2, c3, c4, c5
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem_pool(self.stem(x))
            x = self.layers(x)
            x = self.head_pool(x)
            x = torch.flatten(x, 1)
            return self.head(x)

        def clone(self):
            return self.__class__(self.para.model_copy(deep=True))
        
    class R18(Module):
        pass
    class R50(Module):
        pass
    class R101(Module):
        pass
