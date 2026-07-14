from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, Field, model_validator
from torch import nn
import torch


from .helpers import make_divisible, to_2tuple, convert_padding
from .pool import PoolModels
from .norms import NormModels
from .drop import DropPath
from .acts import ActModels
from .bit import Bit
from .base import CommonModel, CommonModule
from .linear import LinearModels

IntOrPair = Union[int, Tuple[int, int]]
KernelSizeArg = Union[int, Tuple[int, int], Sequence[int]]
PadArg = Union[str, int, Tuple[int, int]]

def shortcut_prefix_fields(json_schema:Dict,specifics:Dict[str,str]={},le=1,alow=['integer','boolean']):
    ts = json_schema.items()#cls.model_json_schema()['properties'].items()
    ts = [([v.get('type')] or [i['type'] for i in v['anyOf']]) for k,v in ts]
    ts = [set(i).intersection(set(alow)) for i in ts]

    ms = [i for i,j in zip(json_schema.keys(),ts) if len(i)>0]
    ks = {i:i[:le] for i in ms if i not in specifics}
    ks.update(specifics)
    ks = {v:k for k,v in ks.items() if v is not None}
    if len(set(ks.keys()))!=len(set(ks.values())):
        raise ValueError(f'short keys is duplicat, {list(ks.keys())} <=> {list(ks.values())}')
    return ks

def parse_shortcut_kwargs(cmd: str, prefix: str, json_schema:Dict,
                            specifics:Dict[str,str], le=1, alow=['integer','boolean']):
    if not cmd.startswith(prefix): return None
    cmds = cmd.split('_')[1:]
    sf = shortcut_prefix_fields(json_schema,specifics,le,alow)
    ks = list(sf.keys())
    ks = sorted(ks, key=len, reverse=True)
    args_tmp = {}
    for k in ks:
        for i,c in enumerate(cmds):
            if c.startswith(k):
                args_tmp[sf[k]] = c[len(k):]
                cmds.pop(i)
                break
            
    args = {}
    for k,v in args_tmp.items():
        try:
            v = int(v)
        except: 
            v = True
        args[k] = v
    return args

class Conv2dModels:
    """Lightweight Pydantic wrappers for key convolutional primitives."""
    class BasicModel(CommonModel):
        def build(self): return self._build(self,Conv2dModules)

        
    class Conv2d(BasicModel):
        in_channels: int
        out_channels: int = -1 # just a place holder not valid
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 'same'
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        padding_mode: str = 'zeros'
        
        bit: bool = True
        scale_op: str = "median"
        
        @classmethod
        def specifics(cls):
            return {
                'bias': 'bs',
                'bit': 'bit',
                'padding_mode': None,
                'scale_op': None,
            }

        @classmethod
        def shortcut(cls,cmd: str = "conv_i3_o32_k3_s1_p1_d1_bs_bit",
                     prefix: str = None, specifics:Dict[str,str]=None,
                     json_schema: Dict=None,
                     le=1, alow=['integer','boolean']):
            if prefix is None: prefix = "conv"
            if specifics is None: specifics = cls.specifics()
            if json_schema is None: json_schema = cls.model_json_schema()['properties']
            kwargs = parse_shortcut_kwargs(
                            cmd=cmd, prefix=prefix, specifics=specifics,le=le, alow=alow,
                            json_schema=json_schema)
            if not kwargs:return None
            return cls(**kwargs)

    class ConvTranspose2d(BasicModel):
        in_channels: int
        out_channels: int = -1 # just a place holder not valid
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 0
        output_padding: IntOrPair = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        padding_mode: str = 'zeros'

        bit: bool = True
        scale_op: str = "median"

    class Conv2dDepthwise(Conv2d):
        group_size: int = Field(default=1, gt=0, exclude=True)
        # If group_size = 1:
        # You get true depthwise convolution (each channel independent).

        # If group_size = in_channels:
        # You effectively get a regular convolution (no grouping).

        # If 1 < group_size < in_channels:
        # You get something in between: grouped depthwise-ish conv, where each group has group_size channels.

        @model_validator(mode='after')
        def valid_model(self):
            if self.in_channels % self.group_size != 0:
                raise ValueError("in_channels must be a multiple of group_size.")
            
            self.groups = self.in_channels // self.group_size

            if self.out_channels % self.groups != 0:
                raise ValueError("out_channels must be a multiple of groups.")
            return self

    class Conv2dPointwise(Conv2d):
        @model_validator(mode='after')
        def valid_model(self):
            self.kernel_size = 1
            self.padding = 0
            self.groups = 1
            return self

    class Conv2dNorm(Conv2d):
        norm: NormModels.type

    class Conv2dAct(Conv2d):
        act: ActModels.type

    class Conv2dNormAct(Conv2d):
        norm: NormModels.type
        act: ActModels.type

    class Conv2dDepthwiseNorm(Conv2dDepthwise,Conv2dNorm):pass
    class Conv2dDepthwiseAct(Conv2dDepthwise,Conv2dAct):pass
    class Conv2dDepthwiseNormAct(Conv2dDepthwise,Conv2dNormAct):pass

    class Conv2dPointwiseNorm(Conv2dPointwise,Conv2dNorm):pass
    class Conv2dPointwiseAct(Conv2dPointwise,Conv2dAct):pass
    class Conv2dPointwiseNormAct(Conv2dPointwise,Conv2dNormAct):pass


    class SqueezeExcite(BasicModel):
        in_channels: int

        conv_reduce_layer: Union['Conv2dModels.Conv2dAct'] = Field(
                default_factory=lambda:Conv2dModels.Conv2dAct(
                                            in_channels=-1, act=ActModels.ReLU(),))
        
        conv_expand_layer: Union['Conv2dModels.Conv2dAct'] = Field(
                default_factory=lambda:Conv2dModels.Conv2dAct(
                                            in_channels=-1, act=ActModels.Sigmoid(),))
        rd_ratio: float = 0.25
        rd_channels: Optional[int] = None
        
        @model_validator(mode='after')
        def valid_model(self):
            if self.rd_channels is None or self.rd_channels<1:
                self.rd_channels = round(self.in_channels * self.rd_ratio)
            
            self.conv_reduce_layer.update(
                    in_channels  = self.in_channels,
                    out_channels = self.rd_channels
            )

            self.conv_expand_layer.update(
                    in_channels  = self.rd_channels,
                    out_channels = self.in_channels
            )
            return self
        
    class DepthwiseSeparableConv(Conv2d):
        in_channels: int
        out_channels: int

        kernel_size: Optional[IntOrPair] = None
        dw_kernel_size: int = 3
        pw_kernel_size: int = 1

        group_size: int = 1
        padding: PadArg = 'same'

        stride: IntOrPair = 1
        bias: bool = False

        noskip: bool = False
        drop_path_rate: float = 0.0

        conv_s2d_layer: Optional[Union['Conv2dModels.Conv2dNormAct']] = Field(default=None)

        conv_dw_layer: Union['Conv2dModels.Conv2dDepthwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )

        se_layer: Optional['Conv2dModels.SqueezeExcite'] = None
        aa_layer: Optional[PoolModels.type] = None

        conv_pw_layer: Union['Conv2dModels.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )

        @model_validator(mode='after')
        def valid_model(self):
            self.kernel_size = None
            self.padding = convert_padding(self.padding)
            dw_kernel_local = self.dw_kernel_size
            dw_pad_type = self.padding

            if self.conv_s2d_layer is not None:
                self.conv_s2d_layer.in_channels = self.in_channels
                self.conv_s2d_layer.out_channels = int(self.in_channels * 4)
                self.conv_s2d_layer.kernel_size = 2
                self.conv_s2d_layer.stride = 2
                self.conv_s2d_layer.padding = 'same'
                self.conv_s2d_layer.bias = self.bias
                if dw_kernel_local in (3,4):
                    dw_pad_type = 'same'
                # we already downsampled
                self.aa_layer = None

            use_aa = (self.aa_layer is None)

            in_channels = self.conv_s2d_layer.in_channels if self.conv_s2d_layer else self.in_channels

            # Depthwise conv
            self.conv_dw_layer.in_channels = in_channels
            self.conv_dw_layer.out_channels = in_channels
            self.conv_dw_layer.kernel_size = dw_kernel_local
            self.conv_dw_layer.stride = 1 if use_aa else self.stride
            self.conv_dw_layer.padding = 1 if dw_kernel_local==3 else dw_pad_type
            self.conv_dw_layer.group_size = self.group_size
            self.conv_dw_layer.bias = self.bias

            # Pointwise conv
            self.conv_pw_layer.in_channels = in_channels
            self.conv_pw_layer.out_channels = self.out_channels
            self.conv_pw_layer.kernel_size = self.pw_kernel_size # always 1
            self.conv_pw_layer.stride = 1
            self.conv_pw_layer.padding = self.padding
            self.conv_pw_layer.groups = 1
            self.conv_pw_layer.bias = self.bias
            return self
        
    class InvertedResidual(DepthwiseSeparableConv):
        """InvertedResidual (MBConv) block using string layer names."""
        exp_ratio: float = 1.0
        exp_kernel_size: int = 1

        conv_pw_exp_layer: Union['Conv2dModels.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )

        conv_pw_layer: Union['Conv2dModels.Conv2dPointwiseNorm'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )

        @model_validator(mode='after')
        def valid_model(self):
            super().valid_model()

            in_channels = self.conv_s2d_layer.in_channels if self.conv_s2d_layer else self.in_channels
            mid_chs = make_divisible(in_channels * self.exp_ratio)

            # Point-wise expansion
            self.conv_pw_exp_layer.in_channels = in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.kernel_size = self.exp_kernel_size
            self.conv_pw_exp_layer.stride = 1 if self.aa_layer else self.stride
            self.conv_pw_exp_layer.padding = self.padding
            self.conv_pw_exp_layer.dilation = self.dilation
            self.conv_pw_exp_layer.bias = self.bias

            # Depth-wise convolution
            self.conv_dw_layer.in_channels = mid_chs
            self.conv_dw_layer.out_channels = mid_chs
            
            # Point-wise linear projection
            self.conv_pw_layer.in_channels = mid_chs
            self.conv_pw_layer.padding = self.padding
            self.conv_pw_layer.bias = self.bias
            return self

    class CondConvResidual(InvertedResidual):
        num_experts:int
        routing_layer: LinearModels.Linear =  Field(
            default_factory=lambda: LinearModels.Linear(in_features=-1)
        )
        
        @model_validator(mode='after')
        def valid_model(self):
            self.routing_layer.in_features = self.in_channels
            self.routing_layer.out_features = self.num_experts
        
    class EdgeResidual(InvertedResidual):
        """Fused MBConv / EdgeResidual block configured via Pydantic model."""
        force_in_channels: int = 0
        bias: bool = False


        conv_pw_layer: Union['Conv2dModels.Conv2dPointwiseNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dPointwiseNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        
        @staticmethod
        def _num_groups(group_size: int, channels: int) -> int:
            if not group_size:
                return 1
            if channels % group_size != 0:
                raise ValueError("channels must be divisible by group_size.")
            return channels // group_size
        
        @model_validator(mode='after')
        def valid_model(self):
            super().valid_model()

            mid_base = self.force_in_channels if self.force_in_channels > 0 else self.in_channels
            mid_chs = make_divisible(mid_base * self.exp_ratio)
            groups = self._num_groups(self.group_size, mid_chs)

            self.conv_pw_exp_layer.in_channels = self.in_channels
            self.conv_pw_exp_layer.out_channels = mid_chs
            self.conv_pw_exp_layer.groups = groups

            self.conv_pw_layer.in_channels = mid_chs
            self.conv_pw_layer.out_channels = self.out_channels
            return self
    
    class ResNetBasicBlock(BasicModel):
        in_channels: int
        out_channels: int
        stride: int = 1
        dilation: int = 1
        padding: PadArg = 'same'
        drop_path_rate: float = 0.0
        act_layer: ActModels.type = Field(default_factory=ActModels.ReLU)

        conv1_layer: Union['Conv2dModels.Conv2dNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv2_layer: Union['Conv2dModels.Conv2dNorm'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )
        shortcut_layer: Union['Conv2dModels.Conv2dNorm',NormModels.Identity] = Field(default_factory=NormModels.Identity)

        @model_validator(mode='after')
        def valid_model(self):
            self.padding = convert_padding(self.padding)
            conv_pad = self.dilation if self.padding == 'same' else self.padding

            self.conv1_layer.in_channels = self.in_channels
            self.conv1_layer.out_channels = self.out_channels
            self.conv1_layer.kernel_size = 3
            self.conv1_layer.stride = self.stride
            self.conv1_layer.padding = conv_pad
            self.conv1_layer.dilation = self.dilation
            self.conv1_layer.bias = False

            self.conv2_layer.in_channels = self.out_channels
            self.conv2_layer.out_channels = self.out_channels
            self.conv2_layer.kernel_size = 3
            self.conv2_layer.stride = 1
            self.conv2_layer.padding = conv_pad
            self.conv2_layer.dilation = self.dilation
            self.conv2_layer.bias = False

            need_downsample = self.stride != 1 or self.in_channels != self.out_channels
            if need_downsample:
                self.shortcut_layer = Conv2dModels.Conv2dNorm(
                    in_channels=-1,
                    norm=NormModels.BatchNorm2d(num_features=-1),
                )
                self.shortcut_layer.in_channels = self.in_channels
                self.shortcut_layer.out_channels = self.out_channels
                self.shortcut_layer.kernel_size = 1
                self.shortcut_layer.stride = self.stride
                self.shortcut_layer.padding = 0
                self.shortcut_layer.bias = False
            else:
                self.shortcut_layer = NormModels.Identity()

            return self

    class ResNetBottleneck(BasicModel):
        in_channels: int
        out_channels: int
        stride: int = 1
        dilation: int = 1
        padding: PadArg = 'same'
        drop_path_rate: float = 0.0
        bottleneck_ratio: int = 4
        mid_channels: Optional[int] = None
        act_layer: ActModels.type = Field(default_factory=ActModels.ReLU)

        conv_reduce_layer: Union['Conv2dModels.Conv2dNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv_transform_layer: Union['Conv2dModels.Conv2dNormAct'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dNormAct(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
                act=ActModels.ReLU(),
            )
        )
        conv_expand_layer: Union['Conv2dModels.Conv2dNorm'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dNorm(
                in_channels=-1,
                norm=NormModels.BatchNorm2d(num_features=-1),
            )
        )
        shortcut_layer: Union['Conv2dModels.Conv2dNorm',NormModels.Identity] = Field(default_factory=NormModels.Identity)

        @model_validator(mode='after')
        def valid_model(self):
            self.padding = convert_padding(self.padding)
            conv_pad = self.dilation if self.padding == 'same' else self.padding

            if self.mid_channels is None:
                if self.out_channels % self.bottleneck_ratio != 0:
                    raise ValueError("out_channels must be divisible by bottleneck_ratio when mid_channels not set.")
                self.mid_channels = self.out_channels // self.bottleneck_ratio

            self.conv_reduce_layer.in_channels = self.in_channels
            self.conv_reduce_layer.out_channels = self.mid_channels
            self.conv_reduce_layer.kernel_size = 1
            self.conv_reduce_layer.stride = 1
            self.conv_reduce_layer.padding = 0
            self.conv_reduce_layer.bias = False

            self.conv_transform_layer.in_channels = self.mid_channels
            self.conv_transform_layer.out_channels = self.mid_channels
            self.conv_transform_layer.kernel_size = 3
            self.conv_transform_layer.stride = self.stride
            self.conv_transform_layer.padding = conv_pad
            self.conv_transform_layer.dilation = self.dilation
            self.conv_transform_layer.bias = False

            self.conv_expand_layer.in_channels = self.mid_channels
            self.conv_expand_layer.out_channels = self.out_channels
            self.conv_expand_layer.kernel_size = 1
            self.conv_expand_layer.stride = 1
            self.conv_expand_layer.padding = 0
            self.conv_expand_layer.bias = False

            need_downsample = self.stride != 1 or self.in_channels != self.out_channels
            if need_downsample:
                self.shortcut_layer = Conv2dModels.Conv2dNorm(
                    in_channels=-1,
                    norm=NormModels.BatchNorm2d(num_features=-1),
                )
                self.shortcut_layer.in_channels = self.in_channels
                self.shortcut_layer.out_channels = self.out_channels
                self.shortcut_layer.kernel_size = 1
                self.shortcut_layer.stride = self.stride
                self.shortcut_layer.padding = 0
                self.shortcut_layer.bias = False
            else:
                self.shortcut_layer = NormModels.Identity()

            return self
    

    class ConvNeXtV2Block(BasicModel):
        dim: int = Field(..., gt=0)
        kernel_size: IntOrPair = 7
        mlp_ratio: int = Field(default=4, gt=0)
        drop_path: float = Field(default=0.0, ge=0.0)
        scale_op: str = "median"

        conv_dw_layer: Union['Conv2dModels.Conv2dDepthwise'] = Field(
            default_factory=lambda: Conv2dModels.Conv2dDepthwise(
                in_channels=-1,
                out_channels=-1,
                kernel_size=7,
                padding=3,
                group_size=1,
            )
        )
        
        norm: NormModels.LayerNorm = Field(
            default_factory=lambda: NormModels.LayerNorm(num_features=-1)
        )

        pwconv1_layer: LinearModels.Linear = Field(
            default_factory=lambda: LinearModels.Linear(in_features=-1)
        )
        
        act: ActModels.type = Field(
            default_factory=lambda: ActModels.GELU()
        )
        grn: NormModels.type = Field(
            default_factory=lambda: NormModels.GlobalResponseNorm(num_features=-1)
        )

        pwconv2_layer: LinearModels.Linear = Field(
            default_factory=lambda: LinearModels.Linear(in_features=-1)
        )

        @model_validator(mode='after')
        def valid_model(self):
            kernel_size = to_2tuple(self.kernel_size)
            if kernel_size[0] != kernel_size[1]:
                raise ValueError("ConvNeXtV2Block expects a square depthwise kernel.")
            self.kernel_size = kernel_size

            hidden_dim = self.mlp_ratio * self.dim

            self.conv_dw_layer.update(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                group_size=1,
                scale_op=self.scale_op,
            )
            self.norm.num_features = self.conv_dw_layer.out_channels

            self.pwconv1_layer.update(
                in_features=self.dim,
                out_features=hidden_dim,
                scale_op=self.scale_op,
            )

            self.grn.num_features = hidden_dim

            self.pwconv2_layer.update(
                in_features=hidden_dim,
                out_features=self.dim,
                scale_op=self.scale_op,
            )
            return self

class Conv2dModules:
    class Module(CommonModule):
        def __init__(self, para, para_cls=None):
            super().__init__(para, Conv2dModels, para_cls)

    class Conv2d(Module):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2d = self.para
            self.bit = self.para.bit

            if self.para.bit:
                self.conv_para = self.para.model_dump(exclude=['bit','norm','act'])
                self.conv = Bit.Conv2d(**self.conv_para)
            else:
                self.conv_para = self.para.model_dump(exclude=['bit','norm','act','scale_op'])
                self.conv = nn.Conv2d(**self.conv_para)
            self.weight_used = True

        def forward_weight(self,x, weight:Optional[torch.Tensor]=None):
            self.weight_used = False
            if self.bit:
                return Bit.functional.conv2d(x,weight=weight,**self.conv_para)
            return torch.nn.functional.conv2d(x,weight=weight,**self.conv_para)

        def forward(self,x):
            return self.conv(x)

        @torch.no_grad()
        def to_ternary(self):
            if not self.weight_used:
                self.conv = nn.Identity()
            if not self.bit:
                print('to_ternary is off here!')
            if self.weight_used and self.bit:
                self.conv = self.conv.to_ternary()                
            return self

    class ConvTranspose2d(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.ConvTranspose2d = self.para
            self.bit = self.para.bit

            if self.para.bit:
                self.conv_para = self.para.model_dump(exclude=['bit','norm','act'])
                self.conv = Bit.ConvTranspose2d(**self.conv_para)
            else:
                self.conv_para = self.para.model_dump(exclude=['bit','scale_op','padding_mode','norm','act'])
                self.conv = nn.ConvTranspose2d(**self.conv_para)
            self.weight_used = True

        def forward_weight(self, x, weight:Optional[torch.Tensor]=None):
            self.weight_used = False
            if weight is None:
                return self.conv(x)
            if self.bit:
                return self.conv(x, weight=weight)
            conv_para = dict(self.conv_para)
            conv_para.pop('bias', None)
            return torch.nn.functional.conv_transpose2d(
                x,
                weight=weight,
                bias=self.conv.bias,
                **conv_para,
            )

        def forward(self, x):
            return self.conv(x)

        @torch.no_grad()
        def to_ternary(self):
            if not self.bit:
                print('to_ternary is off here!')
            if self.bit:
                self.conv = self.conv.to_ternary()
            return self
                
    class Conv2dDepthwise(Conv2d):pass
    class Conv2dPointwise(Conv2d):pass

    class Conv2dNorm(Conv2d):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2dNorm = self.para
            self.para.norm.num_features = self.para.out_channels
            self.norm = self.para.norm.build()

        def forward_weight(self, x, weight = None):
            return self.norm(super().forward_weight(x, weight))

        def forward(self, x):
            return self.norm(super().forward(x))
        
    class Conv2dDepthwiseNorm(Conv2dNorm):pass
    class Conv2dPointwiseNorm(Conv2dNorm):pass

    class Conv2dNormAct(Conv2dNorm):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2dNormAct = self.para
            self.act = self.para.act.build()

        def forward_weight(self, x, weight = None):
            return self.act(super().forward_weight(x, weight))

        def forward(self, x):
            return self.act(super().forward(x))
        
    class Conv2dDepthwiseNormAct(Conv2dNormAct):pass
    class Conv2dPointwiseNormAct(Conv2dNormAct):pass

    class Conv2dAct(Conv2d):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.Conv2dAct = self.para
            self.act = self.para.act.build()

        def forward_weight(self, x, weight = None):
            return self.act(super().forward_weight(x, weight))

        def forward(self, x):
            return self.act(super().forward(x))
                
    class Conv2dDepthwiseAct(Conv2dAct):pass
    class Conv2dPointwiseAct(Conv2dAct):pass
    
    class SqueezeExcite(Module):
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.SqueezeExcite=self.para
            self.conv_reduce:Conv2dModules.Conv2dAct = self.para.conv_reduce_layer.build()
            self.conv_expand:Conv2dModules.Conv2dAct = self.para.conv_expand_layer.build()
            
        def forward(self, x:torch.Tensor):
            x_se = x.mean((2, 3), keepdim=True)
            x_se = self.conv_reduce(x_se)
            x_se = self.conv_expand(x_se)
            return x * x_se
        
        def to_ternary(self,mods=['conv_reduce','conv_reduce']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class DepthwiseSeparableConv(Module):
        """Depthwise-separable block with Pydantic config and string layer names."""
        def __init__(self,para):
            super().__init__(para)
            self.para:Conv2dModels.DepthwiseSeparableConv=self.para
            self.out_channels = self.para.out_channels
            self.conv_s2d = self.para.conv_s2d_layer.build() if self.para.conv_s2d_layer else nn.Identity()      
            self.conv_dw = self.para.conv_dw_layer.build()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()
            if self.para.se_layer:
                if self.para.aa_layer:
                    # aa_layer is a pooling layer in_channels is out_channels
                    se_in_channels = self.para.aa_layer.in_channels
                else:
                    se_in_channels = self.para.conv_dw_layer.out_channels
                self.para.se_layer = self.para.se_layer.__class__(in_channels=se_in_channels,
                                                                  rd_ratio=self.para.se_layer.rd_ratio,
                                                                  rd_channels=self.para.se_layer.rd_channels,
                )
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.conv_pw = self.para.conv_pw_layer.build()
            # ---- DropPath / stochastic depth ----
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate > 0 else nn.Identity()
            self.has_skip = (self.para.in_channels == self.para.out_channels and self.para.stride == 1) and (
                not self.para.noskip
            )

        def pre_conv(self,x):
            x = self.conv_s2d(x) # with norm and act OR Identity
            return x
        
        def mid_conv(self,x):
            x = self.aa(x)
            x = self.se(x)
            return x

        def end_conv(self,x,shortcut):
            x = self.drop_path(x)           
            if self.has_skip:
                x = x + shortcut
            return x

        def forward(self, x):
            shortcut = x
            x = self.pre_conv(x)
            x = self.conv_dw(x) # with norm and act
            x = self.mid_conv(x)           
            x = self.conv_pw(x) # with norm and act
            x = self.end_conv(x,shortcut)
            return x
        
        def to_ternary(self,mods=['conv_s2d','conv_dw','se','aa','conv_pw']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class InvertedResidual(DepthwiseSeparableConv):
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.InvertedResidual=self.para
            self.conv_pw_exp = self.para.conv_pw_exp_layer.build()
        
        def pre_conv(self,x):
            if self.para.conv_s2d_layer is not None:
                x = self.conv_s2d(x) # with norm and act
            x = self.conv_pw_exp(x) # with norm and act
            return x
        
        def to_ternary(self,mods=['conv_s2d','conv_pw','conv_dw','se','aa','conv_pw_exp']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class EdgeResidual(InvertedResidual):
        """Fused MBConv / EdgeResidual block using Pydantic config."""
        def __init__(self, para):
            super().__init__(para)
            self.para: Conv2dModels.EdgeResidual = self.para

            self.conv_s2d = nn.Identity()
            self.conv_dw = nn.Identity()
            self.conv_exp = self.para.conv_pw_exp_layer.build()
            self.aa = self.para.aa_layer.build() if self.para.aa_layer else nn.Identity()
            self.se = self.para.se_layer.build() if self.para.se_layer else nn.Identity()
            self.conv_pw = self.para.conv_pw_layer.build()
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate else nn.Identity()
            self.has_skip = (
                self.para.in_channels == self.para.out_channels and self.para.stride == 1 and not self.para.noskip
            )

        # def forward(self, x):
        #     shortcut = x
        #     x = self.conv_exp(x)
        #     # no conv_dw
        #     x = self.aa(x)
        #     x = self.se(x)
        #     x = self.conv_pw(x)
        #     if self.has_skip:
        #         x = self.drop_path(x) + shortcut
        #     return x

        def to_ternary(self, mods=['conv_exp', 'aa', 'se', 'conv_pw']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self, mods)
            return self

    class CondConvResidual(InvertedResidual):
        """ Inverted residual block w/ CondConv routing"""
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.CondConvResidual=self.para
            self.routing_fn = self.para.routing_layer.build()

        def forward(self, x):
            shortcut = x
            # CondConv routing
            pooled_inputs = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
            x = self.conv_pw_exp(x, routing_weights)
            x = self.conv_dw(x, routing_weights)
            x = self.se(x)
            x = self.conv_pw(x, routing_weights)
            if self.has_skip:
                x = self.drop_path(x) + shortcut
            return x
        
        def to_ternary(self, mods=['conv_pw','conv_dw','se','conv_pwl','routing_fn']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class ResNetBasicBlock(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.ResNetBasicBlock = self.para
            self.conv1 = self.para.conv1_layer.build()
            self.conv2 = self.para.conv2_layer.build()
            self.shortcut_layer = self.para.shortcut_layer.build()
            self.act = self.para.act_layer.build()
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate > 0 else nn.Identity()

        def forward(self, x):
            shortcut = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.drop_path(x)
            x = x + self.shortcut_layer(shortcut)
            x = self.act(x)
            return x

        def to_ternary(self, mods=['conv1','conv2','shortcut']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self

    class ResNetBottleneck(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.ResNetBottleneck = self.para
            self.conv_reduce = self.para.conv_reduce_layer.build()
            self.conv_transform = self.para.conv_transform_layer.build()
            self.conv_expand = self.para.conv_expand_layer.build()
            self.shortcut_layer = self.para.shortcut_layer.build()
            self.act = self.para.act_layer.build()
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate > 0 else nn.Identity()

        def forward(self, x):
            shortcut = x
            x = self.conv_reduce(x)
            x = self.conv_transform(x)
            x = self.conv_expand(x)
            x = self.drop_path(x)
            x = x + self.shortcut_layer(shortcut)
            x = self.act(x)
            return x

        def to_ternary(self, mods=['conv_reduce','conv_transform','conv_expand','shortcut']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self,mods)
            return self
        
    class ConvNeXtV2Block(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para:Conv2dModels.ConvNeXtV2Block=self.para
            self.conv_dw_layer = self.para.conv_dw_layer.build()
            self.norm = self.para.norm.build()
            self.pwconv1_layer = self.para.pwconv1_layer.build()
            self.act = self.para.act.build()
            self.grn = self.para.grn.build()
            self.pwconv2_layer = self.para.pwconv2_layer.build()
            self.drop_path = DropPath(self.para.drop_path) if self.para.drop_path > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shortcut = x

            x = self.conv_dw_layer(x)
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x = self.norm(x)
            x = self.pwconv1_layer(x)
            x = self.act(x)
            x = self.grn(x)
            x = self.pwconv2_layer(x)
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

            return shortcut + self.drop_path(x)