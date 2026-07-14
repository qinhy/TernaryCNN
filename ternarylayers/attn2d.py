from __future__ import annotations

from typing import  Optional, Sequence, Tuple, Union

from pydantic import Field, model_validator
from torch import nn
import torch
import torch.nn.functional as F

from .convs import Conv2dModels
from .helpers import convert_padding, to_2tuple
from .pool import PoolModels
from .norms import NormModels
from .drop import DropPath
from .base import CommonModel, CommonModule
from .linear import LinearModels

IntOrPair = Union[int, Tuple[int, int]]
KernelSizeArg = Union[int, Tuple[int, int], Sequence[int]]
PadArg = Union[str, int, Tuple[int, int]]

class Attention2dModels:
    class BasicModel(CommonModel):
        def build(self): return self._build(self,Attention2dModules)

    class Attention2d(BasicModel):
        """Multi-head attention for 2D NCHW tensors."""

        in_channels: int
        out_channels: Optional[int] = None
        num_heads: int = 32
        bias: bool = True
        expand_first: bool = False
        head_first: bool = False
        attn_drop: float = 0.0
        proj_drop: float = 0.0

        qkv_layer: Union[Conv2dModels.Conv2d] = Field(
            default_factory=lambda: Conv2dModels.Conv2d(in_channels=-1))
        proj_layer: Union[Conv2dModels.Conv2d] = Field(
            default_factory=lambda: Conv2dModels.Conv2d(in_channels=-1))

        # derived, not part of the public schema
        dim_attn: Optional[int] = Field(default=None, exclude=True)
        dim_head: Optional[int] = Field(default=None, exclude=True)

        fused_attn:bool = False # TODO


        @model_validator(mode='after')
        def valid_model(self):
            self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
            self.dim_attn = self.out_channels if self.expand_first else self.in_channels
            if self.dim_attn % self.num_heads != 0:
                raise ValueError("dim_attn must be divisible by num_heads.")
            self.dim_head = self.dim_attn // self.num_heads

            self.qkv_layer.in_channels  = self.in_channels
            self.qkv_layer.out_channels = self.dim_attn * 3
            self.qkv_layer.kernel_size  = 1
            self.qkv_layer.bias  = self.bias

            self.proj_layer.in_channels  = self.dim_attn
            self.proj_layer.out_channels = self.out_channels
            self.proj_layer.kernel_size  = 1
            self.proj_layer.bias  = self.bias
            return self

    class MultiQueryAttention2d(BasicModel):
        """Multi Query Attention with optional spatial downsampling."""
        in_channels: int
        out_channels: Optional[int] = None
        num_heads: int = 8
        key_dim: Optional[int] = None
        value_dim: Optional[int] = None
        query_strides: IntOrPair = 1
        kv_stride: int = 1
        dw_kernel_size: int = 3
        dilation: int = 1
        padding: PadArg = ''
        attn_drop: float = 0.0
        proj_drop: float = 0.0
        bias: bool = False
        
        norm_layer: NormModels.BatchNorm2d = Field(
            default_factory=lambda:NormModels.BatchNorm2d(num_features=-1))
        conv_layer: Conv2dModels.Conv2d = Field(
            default_factory=lambda:Conv2dModels.Conv2d(in_channels=-1))
        
        einsum: bool = False

        has_query_strides: bool = Field(default=False)
        fused_attn: bool = False # TODO


        @model_validator(mode='after')
        def valid_model(self):
            self.out_channels = self.in_channels if self.out_channels is None else self.out_channels
            self.key_dim = self.in_channels // self.num_heads if self.key_dim is None else self.key_dim
            self.value_dim = self.in_channels // self.num_heads if self.value_dim is None else self.value_dim
            self.query_strides = to_2tuple(self.query_strides)
            self.has_query_strides = any(s > 1 for s in self.query_strides)
            self.norm_layer.num_features = self.in_channels
            return self

    class MobileAttention(BasicModel):
        """Mobile attention configuration in line with the other Pydantic models."""
        in_channels: int
        out_channels: int
        stride: int = 1
        dw_kernel_size: int = 3
        dilation: int = 1
        group_size: int = 1
        padding: PadArg = ''
        num_heads: Optional[int] = 8
        key_dim: Optional[int] = None
        value_dim: Optional[int] = None
        use_multi_query: bool = False
        query_strides: IntOrPair = (1, 1)
        kv_stride: int = 1
        cpe_dw_kernel_size: int = 3
        noskip: bool = False
        drop_path_rate: float = 0.0
        attn_drop: float = 0.0
        proj_drop: float = 0.0
        layer_scale_init_value: Optional[float] = 1e-5
        bias: bool = Field(default=False, validation_alias='use_bias')
        use_cpe: bool = False
        fused_attn: bool = Field(default_factory=lambda: False) #use_fused_attn()) TODO

        norm_layer: Optional[NormModels.type] = Field(
            default_factory=lambda: NormModels.BatchNorm2d(num_features=-1)
        )
        attn_layer: Optional[Union[Attention2dModels.MultiQueryAttention2d, Attention2dModels.Attention2d]] = Field(
            default_factory=lambda:Attention2dModels.Attention2d(in_channels=0)
        )
        conv_cpe_layer: Optional[Conv2dModels.Conv2dDepthwise] = Field(
                        default_factory=lambda:Conv2dModels.Conv2dDepthwise(in_channels=-1))        
        layer_scale_layer: Optional[LinearModels.LayerScale2d] = None

        @model_validator(mode='after')
        def valid_model(self):
            self.padding = convert_padding(self.padding)
            self.query_strides = to_2tuple(self.query_strides)

            if self.num_heads is None and self.key_dim is None:
                raise ValueError("Either num_heads or key_dim must be set.")
            if self.num_heads is None:
                if self.in_channels % self.key_dim != 0:
                    raise ValueError("in_channels must be divisible by key_dim when num_heads is None.")
                self.num_heads = self.in_channels // self.key_dim
            key_dim = self.key_dim if self.key_dim is not None else self.in_channels // self.num_heads
            value_dim = self.value_dim if self.value_dim is not None else self.in_channels // self.num_heads
            if key_dim <= 0 or value_dim <= 0:
                raise ValueError("key_dim and value_dim must be positive.")

            if self.use_cpe:
                self.conv_cpe_layer.in_channels = self.in_channels
                self.conv_cpe_layer.out_channels = self.in_channels
                self.conv_cpe_layer.kernel_size = self.cpe_dw_kernel_size
                self.conv_cpe_layer.stride = 1
                self.conv_cpe_layer.dilation = self.dilation
                self.conv_cpe_layer.padding = 'same'
                self.conv_cpe_layer.bias = True
            else:
                self.conv_cpe_layer = None

            if self.norm_layer is not None:
                self.norm_layer.num_features = self.in_channels

            fused_attn = bool(self.fused_attn)
            if self.use_multi_query:
                self.attn_layer=Attention2dModels.MultiQueryAttention2d(in_channels=-1)
                if isinstance(self.attn_layer, Attention2dModels.MultiQueryAttention2d):
                    attn = self.attn_layer
                    attn.in_channels = self.in_channels
                    attn.out_channels = self.out_channels
                    attn.num_heads = self.num_heads
                    attn.key_dim = key_dim
                    attn.value_dim = value_dim
                    attn.query_strides = self.query_strides
                    attn.kv_stride = self.kv_stride
                    attn.dw_kernel_size = self.dw_kernel_size
                    attn.dilation = self.dilation
                    attn.padding = self.padding
                    attn.attn_drop = self.attn_drop
                    attn.proj_drop = self.proj_drop
                    attn.bias = self.bias
                    attn.norm_layer = self.norm_layer.model_copy()
                    attn.fused_attn = fused_attn
                    self.attn_layer = attn
                else:
                    raise ValueError("attn_layer must be MultiQueryAttention2d when use_multi_query is True.")
            
            elif isinstance(self.attn_layer, Attention2dModels.Attention2d):
                    attn = self.attn_layer
                    attn.in_channels = self.in_channels
                    attn.out_channels = self.out_channels
                    attn.num_heads = self.num_heads
                    attn.attn_drop = self.attn_drop
                    attn.proj_drop = self.proj_drop
                    attn.bias = self.bias
                    attn.fused_attn = fused_attn
                    self.attn_layer = attn
            else:
                raise ValueError("attn_layer must be Attention2d when use_multi_query is False.")

            if self.layer_scale_init_value is not None:
                self.layer_scale_layer = LinearModels.LayerScale2d(
                    dim=self.out_channels,
                    init_values=self.layer_scale_init_value,
                )
            else:
                self.layer_scale_layer = None

            return self


class Attention2dModules:
    class Module(CommonModule):
        def __init__(self, para, para_cls=None):
            super().__init__(para, Attention2dModels, para_cls)

    class MultiQueryAttention2d(Module):
        fused_attn: torch.jit.Final[bool]

        def __init__(self, para):
            super().__init__(para)
            self.para: Attention2dModels.MultiQueryAttention2d = self.para

            self.num_heads = self.para.num_heads
            self.key_dim = self.para.key_dim
            self.value_dim = self.para.value_dim
            self.query_strides = self.para.query_strides
            self.kv_stride = self.para.kv_stride
            self.has_query_strides = self.para.has_query_strides
            self.scale = self.key_dim ** -0.5
            self.fused_attn = self.para.fused_attn
            self.einsum = self.para.einsum

            def norm():
                if self.para.norm_layer is None:
                    return nn.Identity()
                return self.para.norm_layer.model_copy().build()
            
            def create_conv2d(in_channels: int, out_channels: int = -1, kernel_size: IntOrPair = 3,
                              stride: IntOrPair = 1, padding: PadArg = 0, dilation: IntOrPair = 1,
                              groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
                              bit: bool = True, scale_op: str = "median",
                              depthwise=False):
                kwargs = locals()
                # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
                if kwargs.pop('depthwise', False): kwargs['groups'] = kwargs['in_channels']
                args = {**self.para.conv_layer.model_dump(),**kwargs}
                return self.para.conv_layer.__class__(**args).build()
            
            self.query = nn.Sequential()
            if self.has_query_strides:
                self.query.add_module('down_pool', PoolModels.AvgPool2d(
                                                    kernel_size=self.query_strides,
                                                    padding=self.para.padding).build())
                self.query.add_module('norm', norm())

            self.query.add_module('proj', create_conv2d(
                self.para.in_channels,
                self.num_heads * self.key_dim,
                kernel_size=1,
                bias=self.para.bias,
            ))

            self.key = nn.Sequential()
            if self.kv_stride > 1:
                self.key.add_module('down_conv', create_conv2d(
                    self.para.in_channels,
                    self.para.in_channels,
                    kernel_size=self.para.dw_kernel_size,
                    stride=self.kv_stride,
                    dilation=self.para.dilation,
                    padding=self.para.padding,
                    depthwise=True,
                ))
                self.key.add_module('norm', norm())

            self.key.add_module('proj', create_conv2d(
                self.para.in_channels,
                self.key_dim,
                kernel_size=1,
                padding=self.para.padding,
                bias=self.para.bias,
            ))

            self.value = nn.Sequential()
            if self.kv_stride > 1:
                self.value.add_module('down_conv', create_conv2d(
                    self.para.in_channels,
                    self.para.in_channels,
                    kernel_size=self.para.dw_kernel_size,
                    stride=self.kv_stride,
                    dilation=self.para.dilation,
                    padding=self.para.padding,
                    depthwise=True,
                ))
                self.value.add_module('norm', norm())

            self.value.add_module('proj', create_conv2d(
                self.para.in_channels,
                self.value_dim,
                kernel_size=1,
                bias=self.para.bias,
            ))

            self.attn_drop = nn.Dropout(self.para.attn_drop)

            self.output = nn.Sequential()
            if self.has_query_strides:
                self.output.add_module('upsample',
                    nn.Upsample(scale_factor=self.query_strides, mode='bilinear', align_corners=False),
                )

            self.output.add_module('proj', create_conv2d(
                self.value_dim * self.num_heads,
                self.para.out_channels,
                kernel_size=1,
                bias=self.para.bias,
            ))
            self.output.add_module('drop', nn.Dropout(self.para.proj_drop))

        def init_weights(self):
            nn.init.xavier_uniform_(self.query.proj.weight)
            nn.init.xavier_uniform_(self.key.proj.weight)
            nn.init.xavier_uniform_(self.value.proj.weight)
            if self.kv_stride > 1:
                nn.init.xavier_uniform_(self.key.down_conv.weight)
                nn.init.xavier_uniform_(self.value.down_conv.weight)
            nn.init.xavier_uniform_(self.output.proj.weight)

        def _reshape_input(self, t: torch.Tensor):
            s = t.shape
            t = t.reshape(s[0], s[1], -1).transpose(1, 2)
            if self.einsum:
                return t
            return t.unsqueeze(1).contiguous()

        def _reshape_projected_query(self, t: torch.Tensor, num_heads: int, key_dim: int):
            s = t.shape
            t = t.reshape(s[0], num_heads, key_dim, -1)
            if self.einsum:
                return t.permute(0, 3, 1, 2).contiguous()
            return t.transpose(-1, -2).contiguous()

        def _reshape_output(self, t: torch.Tensor, num_heads: int, h_px: int, w_px: int):
            s = t.shape
            feat_dim = s[-1] * num_heads
            if not self.einsum:
                t = t.transpose(1, 2)
            return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()

        def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
            B, C, H, W = x.shape

            q = self.query(x)
            q = self._reshape_projected_query(q, self.num_heads, self.key_dim)

            k = self.key(x)
            k = self._reshape_input(k)

            v = self.value(x)
            v = self._reshape_input(v)

            if self.einsum:
                attn = torch.einsum('blhk,bpk->blhp', q, k) * self.scale
                if attn_mask is not None:
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                o = torch.einsum('blhp,bpk->blhk', attn, v)
            else:
                if self.fused_attn:
                    o = F.scaled_dot_product_attention(
                        q,k,v, attn_mask=attn_mask,
                        dropout_p=self.attn_drop.p if self.training else 0.,
                    )
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-1, -2)
                    if attn_mask is not None:
                        attn = attn + attn_mask
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    o = attn @ v

            o = self._reshape_output(o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1])
            x = self.output(o)
            return x

        def to_ternary(self):
            self.attn_drop = nn.Identity()
            self.convert_to_ternary(self)
            return self
        
    class Attention2d(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para: Attention2dModels.Attention2d = self.para

            self.num_heads = self.para.num_heads
            self.dim_head = self.para.dim_head
            self.head_first = self.para.head_first
            self.fused_attn = self.para.fused_attn

            self.qkv = self.para.qkv_layer.build()
            self.attn_drop = nn.Dropout(self.para.attn_drop)
            self.proj = self.para.proj_layer.build()
            self.proj_drop = nn.Dropout(self.para.proj_drop)

        def forward(self, x:torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            B, C, H, W = x.shape
            res:torch.Tensor = self.qkv(x)

            if self.head_first:
                q, k, v = res.view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
            else:
                q, k, v = res.reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)

            if self.fused_attn:
                x = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(-1, -2).contiguous(),
                    k.transpose(-1, -2).contiguous(),
                    v.transpose(-1, -2).contiguous(),
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                ).transpose(-1, -2).reshape(B, -1, H, W)
            else:
                q = q.transpose(-1, -2)
                v = v.transpose(-1, -2)
                attn:torch.Tensor = q @ k * q.size(-1) ** -0.5
                if attn_mask is not None:
                    # NOTE: assumes mask is float and in correct shape
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(-1, -2).reshape(B, -1, H, W)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        def to_ternary(self):
            self.attn_drop = nn.Identity()
            self.convert_to_ternary(self,mods=['qkv', 'proj'])
            return self

    class MobileAttention(Module):
        def __init__(self, para):
            super().__init__(para)
            self.para: Attention2dModels.MobileAttention = self.para

            self.conv_cpe = self.para.conv_cpe_layer.build() if self.para.conv_cpe_layer else None
            self.norm = self.para.norm_layer.build() if self.para.norm_layer is not None else nn.Identity()
            self.attn = self.para.attn_layer.build()
            self.layer_scale = (
                self.para.layer_scale_layer.build() if self.para.layer_scale_layer is not None else nn.Identity()
            )
            self.drop_path = DropPath(self.para.drop_path_rate) if self.para.drop_path_rate else nn.Identity()
            self.has_skip = (
                self.para.stride == 1 and self.para.in_channels == self.para.out_channels and not self.para.noskip
            )

        def forward(self, x):
            if self.conv_cpe is not None:
                x = x + self.conv_cpe(x)

            shortcut = x
            x = self.norm(x)
            x = self.attn(x)
            x = self.layer_scale(x)
            if self.has_skip:
                x = self.drop_path(x) + shortcut
            return x

        def to_ternary(self, mods=['conv_cpe', 'attn']):
            self.drop_path = nn.Identity()
            self.convert_to_ternary(self, mods)
            return self









