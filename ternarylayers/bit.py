"""Compact BitNetCNN ternary layers with optional int8 reference inference."""

import collections
import math
from itertools import repeat
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ternarylayers.padding import PadSame, get_padding_value

EPS = 1e-12


def _ntuple(n, name="parse"):
    def parse(x):
        return tuple(x) if isinstance(x, collections.abc.Iterable) else tuple(repeat(x, n))
    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
to_2tuple = _pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


# -----------------------------------------------------------------------------
# Quantization helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def scale_to_mn(scale: torch.Tensor, multiplier_bits: int = 31):
    """Approximate scale as M * 2**(-n), with int32 M/n."""
    if not 1 <= multiplier_bits <= 31:
        raise ValueError("multiplier_bits must be in [1, 31]")

    s = torch.as_tensor(scale).detach().to(torch.float64)
    if torch.any(s < 0):
        raise ValueError("scale must be non-negative")

    mantissa, exponent = torch.frexp(s)
    M = torch.round(mantissa * (1 << multiplier_bits)).to(torch.int64)
    overflow = M == (1 << multiplier_bits)
    M = torch.where(overflow, M >> 1, M)
    exponent = torch.where(overflow, exponent + 1, exponent)
    n = multiplier_bits - exponent

    zero = s == 0
    M = torch.where(zero, 0, M)
    n = torch.where(zero, 0, n)
    return M.to(torch.int32), n.to(torch.int32)


def requantize_int32(acc: torch.Tensor, M: torch.Tensor, n: torch.Tensor):
    """Integer reference: round(acc * M / 2**n)."""
    if acc.dtype != torch.int32:
        raise TypeError(f"acc must be torch.int32, got {acc.dtype}")

    prod = acc.to(torch.int64) * torch.as_tensor(M, device=acc.device, dtype=torch.int64)
    n = torch.as_tensor(n, device=acc.device, dtype=torch.int64)
    if torch.any(n < 0):
        raise ValueError("negative shifts are not supported by requantize_int32")

    normal = n < 63
    safe_n = torch.where(normal, n, 0)
    rounding = torch.where(
        normal & (safe_n > 0),
        torch.bitwise_left_shift(torch.ones_like(safe_n), (safe_n - 1).clamp_min(0)),
        0,
    )
    shifted = torch.bitwise_right_shift(prod.abs() + rounding, safe_n)
    shifted = torch.where(normal, shifted, 0)
    return torch.where(prod < 0, -shifted, shifted).to(torch.int32)


def quantize_symmetric_int8(x: torch.Tensor, scale: float | torch.Tensor):
    s = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    if torch.any(s <= 0):
        raise ValueError("scale must be > 0")
    return torch.round(x / s).clamp(-128, 127).to(torch.int8)


def dequantize_symmetric_int8(x_q: torch.Tensor, scale: float | torch.Tensor):
    s = torch.as_tensor(scale, dtype=torch.float32, device=x_q.device)
    return x_q.float() * s


def _reduce_abs(x: torch.Tensor, keep_dim: int, op: str = "mean"):
    """Reduce |x| over every dimension except keep_dim, preserving broadcast shape."""
    if not 0 <= keep_dim < x.dim():
        raise ValueError(f"keep_dim={keep_dim} out of range for tensor of dim {x.dim()}")

    a = x.abs()
    dims = [d for d in range(x.dim()) if d != keep_dim]
    if op == "mean":
        return a.mean(dim=dims, keepdim=True).clamp_min(EPS)
    if op != "median":
        raise ValueError("op must be 'mean' or 'median'")

    perm = (keep_dim, *dims)
    s = a.permute(perm).contiguous().view(a.size(keep_dim), -1).median(1).values
    s = s.view(a.size(keep_dim), *([1] * (x.dim() - 1)))
    inv = [perm.index(i) for i in range(x.dim())]
    return s.permute(*inv).contiguous().clamp_min(EPS)


def _ternary_ste(weight: torch.Tensor, dim=0, scale_op="median"):
    s = _reduce_abs(weight, keep_dim=dim, op=scale_op)
    q = torch.round((weight / s).detach()).clamp(-1, 1)
    return weight + (q * s - weight).detach()


def _freeze_ternary(weight: torch.Tensor, scale_op="median"):
    """Quantize along output dim 0; return ternary q and flat per-output scale."""
    scale = _reduce_abs(weight, keep_dim=0, op=scale_op).reshape(weight.shape[0])
    shape = (weight.shape[0],) + (1,) * (weight.dim() - 1)
    q = torch.round(weight / scale.view(shape)).clamp(-1, 1).to(weight.dtype)
    return q, scale


# -----------------------------------------------------------------------------
# Integer reference kernels
# -----------------------------------------------------------------------------
def _conv2d_int32_reference(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if x.dtype != torch.int32 or weight.dtype != torch.int32:
        raise TypeError("x and weight must both be torch.int32")

    dh, dw = to_2tuple(dilation)
    if (dh, dw) != (1, 1):
        kh, kw = weight.shape[-2:]
        expanded = torch.zeros(
            (*weight.shape[:-2], (kh - 1) * dh + 1, (kw - 1) * dw + 1),
            dtype=torch.int32,
            device=weight.device,
        )
        expanded[..., ::dh, ::dw] = weight
        weight = expanded

    return F.conv2d(x, weight, bias, stride, padding, 1, groups)


def _conv_transpose2d_int32_reference(
    x, weight, stride=1, padding=0, output_padding=0, dilation=1, groups=1
):
    """ConvTranspose2d via zero insertion + int32 conv2d (reference only)."""
    if x.dtype != torch.int32 or weight.dtype != torch.int32:
        raise TypeError("x and weight must both be torch.int32")

    sh, sw = to_2tuple(stride)
    ph, pw = to_2tuple(padding)
    oph, opw = to_2tuple(output_padding)
    dh, dw = to_2tuple(dilation)
    kh, kw = weight.shape[-2:]
    if oph >= sh or opw >= sw:
        raise ValueError("output_padding must be smaller than stride")

    n, cin, h, w = x.shape
    up = torch.zeros((n, cin, (h - 1) * sh + 1, (w - 1) * sw + 1), dtype=torch.int32, device=x.device)
    up[:, :, ::sh, ::sw] = x

    pt, pl = dh * (kh - 1) - ph, dw * (kw - 1) - pw
    pb, pr = pt + oph, pl + opw
    ct, cb, cl, cr = max(-pt, 0), max(-pb, 0), max(-pl, 0), max(-pr, 0)
    if ct or cb or cl or cr:
        he = up.shape[-2] - cb if cb else up.shape[-2]
        we = up.shape[-1] - cr if cr else up.shape[-1]
        up = up[:, :, ct:he, cl:we]

    pt, pb, pl, pr = max(pt, 0), max(pb, 0), max(pl, 0), max(pr, 0)
    if pt or pb or pl or pr:
        up = F.pad(up, (pl, pr, pt, pb))

    cin_g, cout_g = cin // groups, weight.shape[1]
    w_conv = (
        weight.view(groups, cin_g, cout_g, kh, kw)
        .permute(0, 2, 1, 3, 4).flip(-1, -2).contiguous()
        .view(groups * cout_g, cin_g, kh, kw)
    )
    return _conv2d_int32_reference(up, w_conv, dilation=dilation, groups=groups)


# -----------------------------------------------------------------------------
# Shared inference behavior
# -----------------------------------------------------------------------------
class _IntegerInferMixin:
    save_dtype = torch.int8

    def _init_frozen(self, weight, scale, bias):
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        for name in ("requant_M", "requant_n", "bias_int32"):
            self.register_buffer(name, None)
        self.input_scale = self.output_scale = None

    @property
    def _out_count(self):
        return getattr(self, "out_channels", getattr(self, "out_features", None))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.save_dtype == torch.int8 and torch.any((self.weight.data > 127) | (self.weight.data < -128)):
            raise ValueError("weight.data is not in (-128, 127)")
        self.weight.data = self.weight.data.to(self.save_dtype)
        super()._save_to_state_dict(destination, prefix, keep_vars)

    @torch.no_grad()
    def prepare_integer(self, input_scale, output_scale, multiplier_bits: int = 31):
        device = self.scale.device
        sx = torch.as_tensor(input_scale, dtype=torch.float64, device=device)
        sy = torch.as_tensor(output_scale, dtype=torch.float64, device=device)
        if sx.numel() != 1 or sy.numel() != 1:
            raise ValueError("input_scale and output_scale must be scalar")
        if sx.item() <= 0 or sy.item() <= 0:
            raise ValueError("input_scale and output_scale must be > 0")

        sw = self.scale.detach().reshape(-1).to(torch.float64)
        if sw.numel() != self._out_count:
            raise ValueError(f"expected {self._out_count} weight scales, got {sw.numel()}")

        self.requant_M, self.requant_n = [t.to(device) for t in scale_to_mn(sx * sw / sy, multiplier_bits)]
        if self.bias is None:
            self.bias_int32 = None
        else:
            bias_q = torch.round(self.bias.detach().double() / (sx * sw)).to(torch.int64)
            i32 = torch.iinfo(torch.int32)
            if torch.any((bias_q < i32.min) | (bias_q > i32.max)):
                raise OverflowError("quantized bias does not fit int32")
            self.bias_int32 = bias_q.to(device=device, dtype=torch.int32)

        self.input_scale, self.output_scale = float(sx.item()), float(sy.item())
        return self

    def _check_integer_input(self, x):
        if x.dtype != torch.int8:
            raise TypeError(f"forward_integer expects torch.int8 input, got {x.dtype}")
        if self.requant_M is None or self.requant_n is None:
            raise RuntimeError("integer metadata is not prepared; call prepare_integer(input_scale, output_scale) first")

    def _finish_integer(self, acc, x, channel_shape):
        if acc.dtype != torch.int32:
            acc = acc.to(torch.int32)
        if self.bias_int32 is not None:
            acc = acc + self.bias_int32.to(x.device).view(channel_shape)
        M = self.requant_M.to(x.device).view(channel_shape)
        n = self.requant_n.to(x.device).view(channel_shape)
        return requantize_int32(acc, M, n).clamp(-128, 127).to(torch.int8)


def convert_to_ternary(module: nn.Module):
    """Recursively replace layers exposing to_ternary(), in-place."""
    if hasattr(module, "to_ternary"):
        return module.to_ternary()
    for name, child in list(module.named_children()):
        setattr(module, name, child.to_ternary()) if hasattr(child, "to_ternary") else convert_to_ternary(child)
    return module


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
class Bit:
    class functional:
        bit1p58_weight = staticmethod(_ternary_ste)

        @staticmethod
        def conv2d(input, weight, bias=None, stride=1, padding=0, padding_mode="zeros",
                   dilation=1, groups=1, dim=0, scale_op="median"):
            weight = _ternary_ste(weight, dim, scale_op)
            if padding_mode != "zeros" and padding != 0:
                if isinstance(padding, int):
                    pad = (padding,) * 4
                elif isinstance(padding, tuple) and len(padding) == 2:
                    ph, pw = padding
                    pad = (pw, pw, ph, ph)
                else:
                    raise ValueError(f"Unsupported padding={padding} for padding_mode='{padding_mode}'")
                input, padding = F.pad(input, pad, mode=padding_mode), 0
            return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        @staticmethod
        def linear(input, weight, bias=None, dim=0, scale_op="median"):
            return F.linear(input, _ternary_ste(weight, dim, scale_op), bias)

    class CommonConv2d(nn.Module):
        """Shared padding, float forward, and inference scale/bias semantics."""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     padding_mode="zeros", dilation=1, groups=1, bias=True, scale_op="median"):
            super().__init__()
            self.in_channels, self.out_channels = in_channels, out_channels
            self.kernel_size = to_2tuple(kernel_size)
            self.stride, self.dilation = to_2tuple(stride), to_2tuple(dilation)
            self.padding, self.padding_mode = padding, padding_mode
            self.groups, self.scale_op, self.bias = groups, scale_op, None
            self.padding_value, self.dynamic_pad = get_padding_value(
                padding, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation
            )
            self.pad_layer = PadSame(self.kernel_size, self.stride, self.dilation) if self.dynamic_pad else None

        def get_weights(self, dtype, device):
            raise NotImplementedError

        def _pad(self, x):
            return (self.pad_layer(x), 0) if self.dynamic_pad else (x, self.padding_value)

        def _op(self, x, weight, bias, padding):
            return F.conv2d(x, weight, bias, self.stride, padding, self.dilation, self.groups)

        def forward(self, x, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
            scale = None
            if weight is None:
                weight, scale = self.get_weights(x.dtype, x.device)
            x, padding = self._pad(x)
            y = self._op(x, weight, self.bias if scale is None else None, padding)
            if scale is not None:
                y = y * scale
                if self.bias is not None:
                    y = y + self.bias.view(1, -1, 1, 1)
            return y

    class CommonConvTranspose2d(CommonConv2d):
        def __init__(self, *args, output_padding=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_padding = to_2tuple(output_padding)

        def _op(self, x, weight, bias, padding):
            return F.conv_transpose2d(
                x, weight, bias, self.stride, padding, self.output_padding, self.groups, self.dilation
            )

    class Bit1p58Weight(nn.Module):
        def __init__(self, dim=0, scale_op="median"):
            super().__init__()
            self.dim, self.scale_op = dim, scale_op

        def forward(self, w):
            return _ternary_ste(w, self.dim, self.scale_op)

    class Conv2d(CommonConv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     padding_mode="zeros", dilation=1, groups=1, bias=True, scale_op="median"):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                             padding_mode, dilation, groups, bias, scale_op)
            kh, kw = self.kernel_size
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
            self.w_q = Bit.Bit1p58Weight(0, scale_op)

        def get_weights(self, dtype, device):
            return self.w_q(self.weight).to(dtype=dtype, device=device), None

        @torch.no_grad()
        def to_ternary(self, dtype=torch.int8):
            q, s = _freeze_ternary(self.weight.data, self.scale_op)
            return Bit.Conv2dInfer(
                q.to(torch.int8) if dtype else q, s.view(-1, 1, 1),
                None if self.bias is None else self.bias.data.clone(),
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.padding, self.padding_mode, self.dilation, self.groups, self.scale_op,
            ).to(self.weight.device)

    class Conv2dInfer(_IntegerInferMixin, CommonConv2d):
        def __init__(self, weight, scale, bias, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, padding_mode="zeros", dilation=1, groups=1, scale_op="median"):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                             padding_mode, dilation, groups, bias is not None, scale_op)
            self._init_frozen(weight, scale, bias)

        def get_weights(self, dtype, device):
            return self.weight.to(device=device, dtype=dtype), self.scale.to(device=device)

        def forward_integer(self, x):
            self._check_integer_input(x)
            x, padding = self._pad(x)
            acc = _conv2d_int32_reference(
                x.int(), self.weight.to(device=x.device, dtype=torch.int32),
                stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups,
            )
            return self._finish_integer(acc, x, (1, -1, 1, 1))

        forward_int8 = forward_integer

    class ConvTranspose2d(CommonConvTranspose2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     output_padding=0, padding_mode="zeros", dilation=1, groups=1,
                     bias=True, scale_op="median"):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                             padding_mode, dilation, groups, bias, scale_op,
                             output_padding=output_padding)
            kh, kw = self.kernel_size
            self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kh, kw))
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        def _weight_to_out(self, w):
            g, kh, kw = self.groups, *self.kernel_size
            ip, op = self.in_channels // g, self.out_channels // g
            return w.view(g, ip, op, kh, kw).permute(0, 2, 1, 3, 4).contiguous().view(self.out_channels, ip, kh, kw)

        def _weight_from_out(self, w):
            g, kh, kw = self.groups, *self.kernel_size
            ip, op = self.in_channels // g, self.out_channels // g
            return w.view(g, op, ip, kh, kw).permute(0, 2, 1, 3, 4).contiguous().view(self.in_channels, op, kh, kw)

        def get_weights(self, dtype, device):
            w = self._weight_to_out(self.weight)
            return self._weight_from_out(_ternary_ste(w, 0, self.scale_op)).to(dtype=dtype, device=device), None

        @torch.no_grad()
        def to_ternary(self, dtype=torch.int8):
            q, s = _freeze_ternary(self._weight_to_out(self.weight.data), self.scale_op)
            q = self._weight_from_out(q)
            return Bit.ConvTranspose2dInfer(
                q.to(torch.int8) if dtype else q, s.view(-1, 1, 1),
                None if self.bias is None else self.bias.data.clone(),
                self.in_channels, self.out_channels, self.kernel_size,
                self.stride, self.padding, self.output_padding, self.padding_mode,
                self.dilation, self.groups, self.scale_op,
            ).to(self.weight.device)

    class ConvTranspose2dInfer(_IntegerInferMixin, CommonConvTranspose2d):
        def __init__(self, weight, scale, bias, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, output_padding=0, padding_mode="zeros",
                     dilation=1, groups=1, scale_op="median"):
            super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                             padding_mode, dilation, groups, bias is not None, scale_op,
                             output_padding=output_padding)
            self._init_frozen(weight, scale, bias)

        def get_weights(self, dtype, device):
            return self.weight.to(device=device, dtype=dtype), self.scale.to(device=device)

        def forward_integer(self, x):
            self._check_integer_input(x)
            x, padding = self._pad(x)
            acc = _conv_transpose2d_int32_reference(
                x.int(), self.weight.to(device=x.device, dtype=torch.int32),
                self.stride, padding, self.output_padding, self.dilation, self.groups,
            )
            return self._finish_integer(acc, x, (1, -1, 1, 1))

        forward_int8 = forward_integer

    class Linear(nn.Module):
        def __init__(self, in_f, out_f, bias=True, scale_op="median"):
            super().__init__()
            self.in_features, self.out_features, self.scale_op = in_f, out_f, scale_op
            self.weight = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
            self.w_q = Bit.Bit1p58Weight(0, scale_op)

        def forward(self, x):
            return F.linear(x, self.w_q(self.weight), self.bias)

        @torch.no_grad()
        def to_ternary(self, dtype=torch.int8):
            q, s = _freeze_ternary(self.weight.data, self.scale_op)
            return Bit.LinearInfer(
                self.in_features, self.out_features,
                q.to(torch.int8) if dtype else q, s,
                None if self.bias is None else self.bias.data.clone(),
            ).to(self.weight.device)

    class LinearInfer(_IntegerInferMixin, nn.Module):
        def __init__(self, in_f, out_f, weight, scale, bias):
            super().__init__()
            self.in_features, self.out_features = in_f, out_f
            self._init_frozen(weight, scale, bias)

        def forward(self, x):
            y = F.linear(x, self.weight.to(device=x.device, dtype=x.dtype)) * self.scale.to(x.device)
            return y if self.bias is None else y + self.bias.to(x.device)

        def forward_integer(self, x):
            self._check_integer_input(x)
            acc = F.linear(x.int(), self.weight.to(device=x.device, dtype=torch.int32))
            shape = (1,) * (acc.dim() - 1) + (-1,)
            return self._finish_integer(acc, x, shape)

        forward_int8 = forward_integer

    # For debugging you can switch back to full-precision:
    # class Conv2d(nn.Conv2d): pass
    # class Linear(nn.Linear): pass
