# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from ternarylayers.dinov3.layers.bitlayers import Linear as BitLinear
from ternarylayers.dinov3.utils import cat_keep_shapes, uncat_with_shapes
from torch import Tensor, nn


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(BitLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        if hasattr(self, "_inner") and hasattr(self._inner.linear, "w_q"):
            wq = self._inner.linear.w_q(self._inner.linear.weight)
            return F.linear(input, wq, masked_bias)
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
        qkv_quant_dim=0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else BitLinear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, dim=qkv_quant_dim).to(device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = BitLinear(dim, dim, bias=proj_bias).to(device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class SelfMatryoshkaAttention(SelfAttention):
    """Self-attention with nested channel widths and shared BitLinear weights.

    The active width is x.shape[-1]. We keep the original head dimension and
    reduce the number of heads. All reduced linear layers quantize AFTER slicing.
    Width	Active heads	Head dimension	QKV channels
    1024	16	64	3072
    768	    12	64	2304
    512	    8	64	1536
    256	    4	64	768
    128	    2	64	384
    64	    1	64	192
    """

    def _check_width(self, x: Tensor, width: int | None = None) -> int:
        full = self.qkv.in_features
        head_dim = full // self.num_heads
        width = x.shape[-1] if width is None else width
        if x.shape[-1] != width:
            raise ValueError("width must equal x.shape[-1]")
        if not (head_dim <= width <= full) or width % head_dim:
            raise ValueError(
                f"width must be a multiple of {head_dim} between {head_dim} and {full}"
            )
        return width

    def linear_qkv(self, x: Tensor, width: int | None = None) -> Tensor:
        width = self._check_width(x, width)
        full = self.qkv.in_features
        if width == full:
            return self.qkv(x)  # Preserve the original full-width implementation.

        # Packed rows are Q(full), K(full), V(full): slice EACH group.
        weight = self.qkv.weight.reshape(3, full, full)[:, :width, :width]
        weight = weight.reshape(3 * width, width)

        bias = self.qkv.bias
        if bias is not None:
            bias = bias.reshape(3, full)[:, :width].reshape(-1)
            mask = getattr(self.qkv, "bias_mask", None)
            if mask is not None:
                mask = mask.reshape(3, full)[:, :width].reshape(-1)
                bias = bias * mask.to(dtype=bias.dtype)

        return F.linear(x, self.qkv.w_q(weight), bias)

    def linear_proj(self, x: Tensor) -> Tensor:
        width = self._check_width(x)
        if width == self.proj.in_features:
            return self.proj(x)
        weight = self.proj.weight[:width, :width]
        bias = self.proj.bias[:width] if self.proj.bias is not None else None
        return F.linear(x, self.proj.w_q(weight), bias)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        if attn_bias is not None:
            raise ValueError("attn_bias is not supported")
        batch, tokens, packed = qkv.shape
        if packed % 3:
            raise ValueError("QKV must contain three equally sized groups")
        width = packed // 3
        full = self.qkv.in_features
        head_dim = full // self.num_heads
        if not (head_dim <= width <= full) or width % head_dim:
            raise ValueError("QKV width must contain complete attention heads")
        heads = width // head_dim

        q, k, v = qkv.reshape(batch, tokens, 3, heads, head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if rope is not None:
            sin, cos = rope
            # RoPE may have a head axis: [H, patches, D] or [B, H, patches, D].
            # A singleton head axis broadcasts and does not need slicing.
            if sin.ndim >= 3 and sin.shape[-3] == self.num_heads:
                sin = sin[..., :heads, :, :]
                cos = cos[..., :heads, :, :]
            q, k = self.apply_rope(q, k, (sin, cos))

        y = F.scaled_dot_product_attention(q, k, v)
        return y.transpose(1, 2).reshape(batch, tokens, width)

    def forward(self, x: Tensor, attn_bias=None, rope=None, width=None) -> Tensor:
        width = self._check_width(x, width)
        if width == self.qkv.in_features:
            return super().forward(x, attn_bias=attn_bias, rope=rope)
        qkv = self.linear_qkv(x, width)
        y = self.compute_attention(qkv, attn_bias=attn_bias, rope=rope)
        return self.proj_drop(self.linear_proj(y))

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        if rope_list is None:
            rope_list = [None] * len(x_list)
        if len(x_list) != len(rope_list):
            raise ValueError("x_list and rope_list must have the same length")
        if all(x.shape[-1] == self.qkv.in_features for x in x_list):
            return super().forward_list(
                x_list, attn_bias=attn_bias, rope_list=rope_list
            )
        # Handles small and mixed widths; no full-width projections are computed.
        return [
            self.forward(x, attn_bias=attn_bias, rope=rope)
            for x, rope in zip(x_list, rope_list)
        ]

    
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = BitLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = BitLinear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
