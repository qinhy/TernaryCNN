from __future__ import annotations

import json
import numbers
from typing import List, Optional, Tuple, Type, Union

from pydantic import BaseModel
from torch import nn
import torch
from torchvision.ops.misc import FrozenBatchNorm2d

class NormModules:
    class RmsNorm2d(nn.Module):
        """ RmsNorm2D for NCHW tensors, w/ fast apex or cast norm if available

        NOTE: It's currently (2025-05-10) faster to use an eager 2d kernel that does reduction
        on dim=1 than to permute and use internal PyTorch F.rms_norm, this may change if something
        like https://github.com/pytorch/pytorch/pull/150576 lands.
        """
        __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', '_fast_norm']
        normalized_shape: Tuple[int, ...]
        eps: float
        elementwise_affine: bool
        _fast_norm: bool

        def __init__(
                self,
                num_features: int,
                eps: float = 1e-6,
                affine: bool = True,
                device=None,
                dtype=None,
        ) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            normalized_shape = channels = num_features
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.elementwise_affine = affine
            self._fast_norm = False # is_fast_norm()  # can't script unless we have these flags here (no globals)

            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('weight', None)

            self.reset_parameters()

        def reset_parameters(self) -> None:
            if self.elementwise_affine:
                nn.init.ones_(self.weight)

        @staticmethod
        def rms_norm2d(
            x: torch.Tensor,
            normalized_shape: List[int],
            weight: Optional[torch.Tensor] = None,
            eps: float = 1e-5,
        ):
            assert len(normalized_shape) == 1
            v = x.pow(2)
            v = torch.mean(v, dim=1, keepdim=True)
            x = x * torch.rsqrt(v + eps)
            if weight is not None:
                x = x * weight.reshape(1, -1, 1, 1)
            return x
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
            # Since there is no built-in PyTorch impl, always use APEX RmsNorm if is installed.
            if self._fast_norm:
                pass
                # x = fast_rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
            else:
                x = self.rms_norm2d(x, self.normalized_shape, self.weight, self.eps)
            return x

class NormModels:
    class _BatchNormBase(BaseModel):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True

    class BatchNorm1d(_BatchNormBase):
        def build(self) -> nn.Module:
            return nn.BatchNorm1d(**self.model_dump())

    class BatchNorm2d(_BatchNormBase):
        def build(self) -> nn.Module:
            return nn.BatchNorm2d(**self.model_dump())

    class BatchNorm3d(_BatchNormBase):
        def build(self) -> nn.Module:
            return nn.BatchNorm3d(**self.model_dump())

    class SyncBatchNorm(_BatchNormBase):
        process_group: Optional[object] = None

        def build(self) -> nn.Module:
            return nn.SyncBatchNorm(**self.model_dump())

    class _InstanceNormBase(BaseModel):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = False
        track_running_stats: bool = False

    class InstanceNorm1d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return nn.InstanceNorm1d(**self.model_dump())

    class InstanceNorm2d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return nn.InstanceNorm2d(**self.model_dump())

    class InstanceNorm3d(_InstanceNormBase):
        def build(self) -> nn.Module:
            return nn.InstanceNorm3d(**self.model_dump())

    class GroupNorm(BaseModel):
        num_features: int
        num_groups: int = 32
        eps: float = 1e-5
        affine: bool = True

        def build(self) -> nn.Module:
            return nn.GroupNorm(**self.model_dump())

    # class GroupNorm1(BaseModel):
    #     num_features: int
    #     eps: float = 1e-5
    #     affine: bool = True

    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.GroupNorm1)

    class LayerNorm(BaseModel):
        num_features: int  # as same as normalized_shape
        eps: float = 1e-5
        elementwise_affine: bool = True
        bias: bool = True
        data_format:str = "channels_last"

        def build(self) -> nn.Module:
            return NormModels._LayerNormModule(**self.model_dump(exclude=["elementwise_affine"]))

    class _LayerNormModule(nn.Module):
        """ LayerNorm that supports two data formats: channels_last (default) or channels_first. """
        def __init__(self, num_features, eps=1e-6, data_format="channels_last", bias = True):
            super().__init__()
            normalized_shape = num_features
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
            self.eps = eps
            self.data_format = data_format
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError
            self.normalized_shape = (normalized_shape, )

        def forward(self, x:torch.Tensor):
            if self.data_format == "channels_last":
                return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x
                if self.bias is not None:
                    x = x + self.bias[:, None, None]
                return x

    class GlobalResponseNorm(BaseModel):
        num_features: int
        
        def build(self) -> nn.Module:
            return NormModels._GlobalResponseNormModule(**self.model_dump())

    class _GlobalResponseNormModule(nn.Module):
        """ GRN (Global Response Normalization) layer """
        def __init__(self, num_features):
            super().__init__()
            dim = num_features
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

        def forward(self, x):
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            return self.gamma * (x * Nx) + self.beta + x
        
    # class LayerNormFp32(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNormFp32)

    class LayerNorm2d(BaseModel):
        num_features: int

        def build(self) -> nn.Module:
            return NormModels._LayerNorm2D(**self.model_dump())
    
    class _LayerNorm2D(nn.Module):
        def __init__(self, num_features, norm_layer=nn.LayerNorm):
            super().__init__()
            self.ln = norm_layer(num_features) if norm_layer is not None else nn.Identity()

        def forward(self, x: torch.Tensor):
            """
            x: N C H W
            """
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)
            return x

    # class LayerNorm2dFp32(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNorm2dFp32)

    # class LayerNormExp2d(LayerNorm):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.LayerNormExp2d)

    class _RmsNormBase(BaseModel):
        num_features: int
        eps: float = 1e-6
        affine: bool = True

    class RmsNorm(_RmsNormBase):
        def build(self) -> nn.Module:
            return nn.RMSNorm(**self.model_dump())

    # class RmsNormFp32(_RmsNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.RMSNormFp32)

    class RmsNorm2d(_RmsNormBase):
        def build(self) -> nn.Module:
            return NormModules.RmsNorm2d(**self.model_dump())

    # class RmsNorm2dFp32(_RmsNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.RMSNorm2dFp32)

    # class _SimpleNormBase(BaseModel):
    #     num_features: int
    #     eps: float = 1e-6
    #     affine: bool = True

    # class SimpleNorm(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNorm)

    # class SimpleNormFp32(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNormFp32)

    # class SimpleNorm2d(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNorm2d)

    # class SimpleNorm2dFp32(_SimpleNormBase):
    #     def build(self) -> nn.Module:
    #         return NormdModules.Norm(self, type(self), nn.SimpleNorm2dFp32)

    class FrozenBatchNorm2d(BaseModel):
        num_features: int
        eps: float = 1e-5

        def build(self) -> nn.Module:
            return FrozenBatchNorm2d(**self.model_dump())

    class Identity(BaseModel):
        def build(self) -> nn.Module:
            return nn.Identity(**self.model_dump())

    type = Union[BatchNorm1d,BatchNorm2d,BatchNorm3d,SyncBatchNorm,
                 InstanceNorm1d,InstanceNorm2d,InstanceNorm3d,GroupNorm,LayerNorm,
                 RmsNorm,RmsNorm2d,FrozenBatchNorm2d,Identity]
