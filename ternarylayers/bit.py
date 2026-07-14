"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""
import collections
from itertools import repeat
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ternarylayers.padding import PadSame, get_padding_value
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
to_2tuple = _pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

# Constants
EPS = 1e-12

# ----------------------------
# Quantization Utilities
# ----------------------------
def _reduce_abs(x: torch.Tensor, keep_dim: int, op: str = "mean") -> torch.Tensor:
    """
    Reduce absolute values along all dimensions except keep_dim.
    Supports mean and median operations.
    Returns a tensor broadcastable to x with singleton dims in reduced axes.
    """
    if keep_dim < 0 or keep_dim >= x.dim():
        raise ValueError(f"keep_dim={keep_dim} out of range for tensor of dim {x.dim()}")
    dims = [d for d in range(x.dim()) if d != keep_dim]
    a = x.abs()
    if op == "mean":
        s = a.mean(dim=dims, keepdim=True)
    elif op == "median":
        # median over flattened other dims
        perm = (keep_dim,) + tuple(d for d in range(x.dim()) if d != keep_dim)
        flat = a.permute(perm).contiguous().view(a.size(keep_dim), -1)
        s = flat.median(dim=1).values.view([a.size(keep_dim)] + [1] * (x.dim() - 1))
        inv = [0] * x.dim()
        for i, p in enumerate(perm):
            inv[p] = i
        s = s.permute(*inv).contiguous()
    else:
        raise ValueError("op must be 'mean' or 'median'")
    return s.clamp_min(EPS)


# ----------------------------
# Model-wide conversion helpers
# ----------------------------
@torch.no_grad()
def convert_to_ternary(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with their *Infer counterparts, in-place.

    Usage:
        from copy import deepcopy
        ternary_model = convert_to_ternary(deepcopy(model))
    """
    for name, child in list(module.named_children()):
        if hasattr(child, "to_ternary"):
            setattr(module, name, child.to_ternary())
        else:
            convert_to_ternary(child)
    return module


# ----------------------------
# Bit Quantization Classes
# ----------------------------
class Bit:
    """
    Collection of classes for bit-level quantization of neural networks.
    Includes fake-quant building blocks, inference modules, and training modules.
    """

    class functional:
        @staticmethod
        def bit1p58_weight(
            weight: torch.Tensor,
            dim: int = 0,
            scale_op: str = "median",
        ) -> torch.Tensor:
            """
            Fake-quant ternary weights (~1.58 bits) with STE,
            using per-channel scale from |w| reduction.
            """
            s = _reduce_abs(weight, keep_dim=dim, op=scale_op)
            w_bar = (weight / s).detach()
            w_q = torch.round(w_bar)
            w_q = w_q.clamp(-1, 1)
            # STE: pass-through gradient
            return weight + (w_q * s - weight).detach()

        @staticmethod
        def conv2d(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias=None,
            stride=1,
            padding=0,
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            dim: int = 0,
            scale_op: str = "median",
        ) -> torch.Tensor:
            """
            Conv2d with fake-quant ternary weights (STE). Supports padding_mode
            similar to nn.Conv2d when padding_mode != "zeros".
            """
            weight = Bit.functional.bit1p58_weight(weight, dim, scale_op)

            # Emulate nn.Conv2d behavior for padding_mode != 'zeros'
            if padding_mode != "zeros" and padding != 0:
                if isinstance(padding, int):
                    pad = (padding, padding, padding, padding)  # left, right, top, bottom
                elif isinstance(padding, tuple) and len(padding) == 2:
                    pad_h, pad_w = padding
                    pad = (pad_w, pad_w, pad_h, pad_h)
                else:
                    raise ValueError(f"Unsupported padding={padding} for padding_mode='{padding_mode}'")
                input = F.pad(input, pad, mode=padding_mode)
                padding_eff = 0
            else:
                padding_eff = padding

            return F.conv2d(
                input,
                weight,
                bias,
                stride=stride,
                padding=padding_eff,
                dilation=dilation,
                groups=groups,
            )

        @staticmethod
        def linear(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias=None,
            dim: int = 0,
            scale_op: str = "median",
        ) -> torch.Tensor:
            weight = Bit.functional.bit1p58_weight(weight, dim, scale_op)
            return F.linear(input, weight, bias)

    # ------------------------------------------------------------------
    # CommonConv2d: shared conv implementation with SAME padding
    # ------------------------------------------------------------------
    class CommonConv2d(nn.Module):
        """
        Shared conv2d implementation that supports:
        - 'same' / 'valid' / int / tuple paddings via ternarylayers.padding
        - Optional dynamic SAME padding (PadSame)
        Subclasses must implement:
        - init_weights(...)
        - get_weights(dtype, device) -> (weight, scale or None)

        IMPORTANT SEMANTICS (to match your old working code):
        - Training (scale is None):  y = Conv(x, Wq, bias)
        - Inference (scale is not None):
              y = Conv(x, Wq, bias=None)
              y = y * scale
              if bias is not None: y = y + bias
          i.e., **bias is NOT scaled**.
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,          # can be int, tuple, 'same', 'valid'
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            bias: bool = True,
            scale_op: str = "median",
        ):
            super().__init__()
            kh, kw = to_2tuple(kernel_size)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = (kh, kw)
            self.stride = to_2tuple(stride)
            self.dilation = to_2tuple(dilation)
            self.padding = padding           # original user argument
            self.padding_mode = padding_mode
            self.scale_op = scale_op
            self.groups = groups

            # Resolve static vs dynamic SAME padding using your ternarylayers.padding utilities
            self.padding_value, self.dynamic_pad = get_padding_value(
                padding,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )

            if self.dynamic_pad:
                self.pad_layer = PadSame(self.kernel_size, self.stride, self.dilation)
            else:
                self.pad_layer = None

            # Placeholder; subclasses will create real Parameters
            self.bias = None

        # --- abstract-ish API for subclasses ---
        def get_weights(self, dtype: torch.dtype, device: torch.device):
            """
            Return (weight, scale_or_None).
            - For training Conv2d: (w_q, None)
            - For inference Conv2dInfer: (weight_float, scale)
            """
            raise NotImplementedError

        def init_weights(self, *args, **kwargs):
            raise NotImplementedError

        # --- shared forward ---
        def forward(self, x: torch.Tensor,
                    weight: Optional[torch.Tensor]=None,
                    bias: Optional[torch.Tensor]=None):
            scale = None
            if weight is None:
                weight, scale = self.get_weights(x.dtype, x.device)

            if self.dynamic_pad:
                x = self.pad_layer(x)
                padding_value = 0
            else:
                padding_value = self.padding_value

            bias = self.bias if scale is None else None

            y = F.conv2d(
                x,
                weight,
                bias=bias,
                stride=self.stride,
                padding=padding_value,
                dilation=self.dilation,
                groups=self.groups,
            )

            if scale is not None:
                y = y * scale
                if self.bias is not None:
                    y = y + self.bias.view(1, -1, 1, 1)

            return y

    # ------------------------------------------------------------------
    # Fake-quant building block (QAT) — weight only
    # ------------------------------------------------------------------
    class Bit1p58Weight(nn.Module):
        """1.58-bit (ternary) weight quantizer with per-out-channel scaling."""

        def __init__(self, dim: int = 0, scale_op: str = "median"):
            super().__init__()
            self.dim = dim
            self.scale_op = scale_op

        def forward(self, w: torch.Tensor) -> torch.Tensor:
            s = _reduce_abs(w, keep_dim=self.dim, op=self.scale_op)
            w_bar = (w / s).detach()
            w_q = torch.round(w_bar)
            w_q = w_q.clamp(-1, 1)
            return w + (w_q * s - w).detach()

    # ------------------------------------------------------------------
    # Train-time Conv2d (fake-quant weights) with SAME support
    # ------------------------------------------------------------------
    class Conv2d(CommonConv2d):
        """
        Conv2d with ternary weights (fake-quant for training).
        This keeps the old Bit.Conv2d API, but adds SAME padding support.

        Old API compatibility:
            Bit.Conv2d(in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale_op="median")
        New extras:
            - padding can now also be 'same', 'valid', etc. as supported by ternarylayers.padding.
            - padding_mode (for non-zero padding) is available but optional.
        """
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,          # can be int, tuple, 'same', 'valid'
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            bias: bool = True,
            scale_op: str = "median",
        ):
            super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                bias=bias,
                scale_op=scale_op,
            )
            self.init_weights(bias)

        def init_weights(self, bias: bool):
            kh, kw = self.kernel_size
            self.weight = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels // self.groups, kh, kw)
            )
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=self.scale_op)

        def get_weights(self, dtype: torch.dtype, device: torch.device):
            # Fake-quant weights for training, no separate scale factor
            wq = self.w_q(self.weight).to(dtype=dtype, device=device)
            return wq, None

        @torch.no_grad()
        def to_ternary(self, dtype=torch.int8):
            """
            Convert this layer into a frozen Bit.Conv2dInfer, carrying over:
            - per-out-channel weight scale `s` and Wq in {-1,0,+1},
            - SAME padding behavior (via CommonConv2d) preserved.
            """
            w = self.weight.data
            s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            s = s_vec.view(-1, 1, 1)                                        # [out,1,1] for conv broadcast
            w_bar = w / s_vec.view(-1, 1, 1, 1)
            w_q = torch.round(w_bar).to(w.dtype)
            w_q = w_q.clamp(-1, 1)

            return Bit.Conv2dInfer(
                weight=w_q.to(dtype=torch.int8) if dtype else w_q,
                scale=s,
                bias=(None if self.bias is None else self.bias.data.clone()),
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                padding_mode=self.padding_mode,
                dilation=self.dilation,
                groups=self.groups,
                scale_op=self.scale_op,
            ).to(device=self.weight.device,dtype=self.weight.dtype)

    # ------------------------------------------------------------------
    # Inference Conv2d (frozen ternary)
    # ------------------------------------------------------------------
    class Conv2dInfer(CommonConv2d):
        """
        Frozen ternary conv:
            y = Conv(x, Wq) * s_per_out + b
        where:
            - Wq in {-1,0,+1} stored as int8,
            - s_per_out is float per output channel (shape [out,1,1]),
            - bias is *not* scaled by s.
        """

        def __init__(
            self,
            weight: torch.Tensor,
            scale: torch.Tensor,
            bias: torch.Tensor,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            scale_op: str = "median",
        ):
            super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                bias=True if bias is not None else False,
                scale_op=scale_op,
            )
            self.save_dtype = torch.int8
            self.init_weights(bias, weight, scale)

        # ---- custom save / load hooks ----
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if self.save_dtype==torch.int8 and (
                (self.weight.data>127).sum() + (self.weight.data<-128).sum()>0):
                raise ValueError("weight.data is not in (-128, 127)")
            self.weight.data = self.weight.data.to(self.save_dtype)
            # let nn.Module save everything as usual
            super()._save_to_state_dict(destination, prefix, keep_vars)

        def init_weights(self, bias, weight: torch.Tensor, scale: torch.Tensor):
            # Make them Parameters so param counters include them (but keep frozen)
            self.weight = nn.Parameter(weight, requires_grad=False) # [out,in,kh,kw]
            self.scale  = nn.Parameter(scale, requires_grad=False)  # [out,1,1]
            self.bias = bias if bias is None else nn.Parameter(bias, requires_grad=False) # [out]

        def get_weights(self, dtype: torch.dtype, device: torch.device):
            return self.weight.to(dtype=dtype), self.scale

    # ------------------------------------------------------------------
    # Train-time ConvTranspose2d (fake-quant weights)
    # ------------------------------------------------------------------
    class ConvTranspose2d(CommonConv2d):
        """
        ConvTranspose2d with ternary weights (fake-quant for training).
        Signature follows nn.ConvTranspose2d, with SAME padding support
        via CommonConv2d padding utilities.
        """
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            bias: bool = True,
            scale_op: str = "median",
        ):
            super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                bias=bias,
                scale_op=scale_op,
            )
            self.output_padding = to_2tuple(output_padding)
            self.init_weights(bias)

        def init_weights(self, bias: bool):
            kh, kw = self.kernel_size
            self.weight = nn.Parameter(
                torch.empty(self.in_channels, self.out_channels // self.groups, kh, kw)
            )
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None

        def _reshape_weight_to_out(self, weight: torch.Tensor) -> torch.Tensor:
            # Reorder to (out_channels, in_channels_per_group, kh, kw) for per-out quantization
            g = self.groups
            in_per = self.in_channels // g
            out_per = self.out_channels // g
            kh, kw = self.kernel_size
            w_view = weight.view(g, in_per, out_per, kh, kw)
            w_view = w_view.permute(0, 2, 1, 3, 4).contiguous()
            return w_view.view(self.out_channels, in_per, kh, kw)

        def _reshape_weight_from_out(self, weight: torch.Tensor) -> torch.Tensor:
            # Inverse of _reshape_weight_to_out
            g = self.groups
            in_per = self.in_channels // g
            out_per = self.out_channels // g
            kh, kw = self.kernel_size
            w_view = weight.view(g, out_per, in_per, kh, kw)
            w_view = w_view.permute(0, 2, 1, 3, 4).contiguous()
            return w_view.view(self.in_channels, out_per, kh, kw)

        def get_weights(self, dtype: torch.dtype, device: torch.device):
            w = self._reshape_weight_to_out(self.weight)
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op)
            w_bar = (w / s).detach()
            w_q = torch.round(w_bar).clamp(-1, 1)
            w_q = w + (w_q * s - w).detach()
            w_q = self._reshape_weight_from_out(w_q)
            return w_q.to(dtype=dtype, device=device), None

        def forward(
            self,
            x: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
        ):
            scale = None
            if weight is None:
                weight, scale = self.get_weights(x.dtype, x.device)

            if self.dynamic_pad:
                x = self.pad_layer(x)
                padding_value = 0
            else:
                padding_value = self.padding_value

            bias = self.bias if scale is None else None

            y = F.conv_transpose2d(
                x,
                weight,
                bias=bias,
                stride=self.stride,
                padding=padding_value,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            if scale is not None:
                y = y * scale
                if self.bias is not None:
                    y = y + self.bias.view(1, -1, 1, 1)

            return y

        @torch.no_grad()
        def to_ternary(self, dtype=torch.int8):
            w = self.weight.data
            w_view = self._reshape_weight_to_out(w)
            s_vec = _reduce_abs(w_view, keep_dim=0, op=self.scale_op).squeeze()
            s = s_vec.view(-1, 1, 1)
            w_bar = w_view / s_vec.view(-1, 1, 1, 1)
            w_q = torch.round(w_bar).clamp(-1, 1).to(w.dtype)
            w_q = self._reshape_weight_from_out(w_q)
            return Bit.ConvTranspose2dInfer(
                weight=w_q.to(dtype=torch.int8) if dtype else w_q,
                scale=s,
                bias=(None if self.bias is None else self.bias.data.clone()),
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                padding_mode=self.padding_mode,
                dilation=self.dilation,
                groups=self.groups,
                scale_op=self.scale_op,
            ).to(device=self.weight.device, dtype=self.weight.dtype)

    # ------------------------------------------------------------------
    # Inference ConvTranspose2d (frozen ternary)
    # ------------------------------------------------------------------
    class ConvTranspose2dInfer(CommonConv2d):
        """
        Frozen ternary conv transpose:
            y = ConvTranspose(x, Wq) * s_per_out + b
        """

        def __init__(
            self,
            weight: torch.Tensor,
            scale: torch.Tensor,
            bias: torch.Tensor,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            scale_op: str = "median",
        ):
            super().__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
                groups=groups,
                bias=True if bias is not None else False,
                scale_op=scale_op,
            )
            self.output_padding = to_2tuple(output_padding)
            self.save_dtype = torch.int8
            self.init_weights(bias, weight, scale)

        # ---- custom save / load hooks ----
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if self.save_dtype == torch.int8 and (
                (self.weight.data > 127).sum() + (self.weight.data < -128).sum() > 0
            ):
                raise ValueError("weight.data is not in (-128, 127)")
            self.weight.data = self.weight.data.to(self.save_dtype)
            super()._save_to_state_dict(destination, prefix, keep_vars)

        def init_weights(self, bias, weight: torch.Tensor, scale: torch.Tensor):
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.scale = nn.Parameter(scale, requires_grad=False)
            self.bias = bias if bias is None else nn.Parameter(bias, requires_grad=False)

        def get_weights(self, dtype: torch.dtype, device: torch.device):
            return self.weight.to(dtype=dtype), self.scale

        def forward(
            self,
            x: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
        ):
            scale = None
            if weight is None:
                weight, scale = self.get_weights(x.dtype, x.device)

            if self.dynamic_pad:
                x = self.pad_layer(x)
                padding_value = 0
            else:
                padding_value = self.padding_value

            bias = self.bias if scale is None else None

            y = F.conv_transpose2d(
                x,
                weight,
                bias=bias,
                stride=self.stride,
                padding=padding_value,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

            if scale is not None:
                y = y * scale
                if self.bias is not None:
                    y = y + self.bias.view(1, -1, 1, 1)

            return y

    # ------------------------------------------------------------------
    # Train-time Linear & Inference Linear (unchanged from old working code)
    # ------------------------------------------------------------------
    class Linear(nn.Module):
        def __init__(self, in_f: int, out_f: int, bias: bool = True, scale_op: str = "median"):
            super().__init__()
            self.in_features = in_f
            self.out_features = out_f
            self.weight = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
            self.scale_op = scale_op

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            wq = self.w_q(self.weight)
            return F.linear(x, wq, self.bias)

        @torch.no_grad()
        def to_ternary(self, dtype=torch.int8):
            w = self.weight.data
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            w_q = torch.round(w / s.view(-1, 1)).clamp(-1, 1).to(w.dtype)
            bias = None if self.bias is None else self.bias.data.clone()
            w_q = w_q.to(dtype=torch.int8) if dtype else w_q
            return Bit.LinearInfer(in_f=self.in_features, out_f=self.out_features, 
                                   weight=w_q, scale=s, bias=bias,
                                ).to(device=self.weight.device,dtype=self.weight.dtype)

    class LinearInfer(nn.Module):
        """Frozen ternary linear: y = (x @ Wq^T) * s + b"""
        def __init__(self, in_f: int, out_f: int, weight: torch.Tensor, scale: torch.Tensor, bias):
            super().__init__()
            self.in_features = in_f
            self.out_features = out_f
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.scale = nn.Parameter(scale, requires_grad=False)
            self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
            self.save_dtype = torch.int8

        # ---- custom save / load hooks ----
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if self.save_dtype==torch.int8 and (
                (self.weight.data>127).sum() + (self.weight.data<-128).sum()>0):
                raise ValueError("weight.data is not in (-128, 127)")
            self.weight.data = self.weight.data.to(self.save_dtype)
            # let nn.Module save everything as usual
            super()._save_to_state_dict(destination, prefix, keep_vars)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.weight.to(dtype=x.dtype)
            y = F.linear(x, w, bias=None) * self.scale
            if self.bias is not None:
                y = y + self.bias
            return y

    # For debugging you can switch back to full-precision:
    # class Conv2d(nn.Conv2d): pass
    # class Linear(nn.Linear): pass
