# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Literal, Tuple

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
        return F.linear(input, self.w_q(self.weight), masked_bias)


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

class DotPlusAttentionMixin:
    """Mixin to compose BEFORE the user's SelfAttention in the MRO.

    All existing SelfAttention constructor arguments and method signatures stay
    valid. Extra keyword-only arguments control the experimental attention core.

    score_op: 'dot_plus' (negative L1) or 'dot' (ordinary dot product).
    value_op: 'dot_plus' (negative L1 reduction) or 'matmul'.
    value_reduction: 'sum' is the literal dot-plus; 'mean' normalizes by the
        number of included keys, stabilizing magnitudes across sequence lengths.
    temperature_init: Positive scalar per-head initial temperature, or None
        for the variance-based initialization.
    is_causal: Whether each token may only read itself and preceding tokens.
    query_chunk_size: Limits peak memory of masked dot-plus aggregation.

    Supported attn_bias: None, bool (True=allowed, as in SDPA), or additive
    floating tensor (use -inf to exclude). Shapes [N,N], [B,N,N], [B,H,N,N]
    and broadcast-compatible 4D versions. No xFormers AttentionBias objects.
    """

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
        *,
        score_op: Literal["dot_plus", "dot"] = "dot_plus",
        value_op: Literal["dot_plus", "matmul"] = "dot_plus",
        value_reduction: Literal["sum", "mean"] = "sum",
        learnable_temperature: bool = True,
        temperature_init: float | None = None,
        is_causal: bool = False,
        query_chunk_size: int = 16,
    ) -> None:
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be > 0 and divisible by num_heads")
        if score_op not in ("dot_plus", "dot"):
            raise ValueError("score_op must be 'dot_plus' or 'dot'")
        if value_op not in ("dot_plus", "matmul"):
            raise ValueError("value_op must be 'dot_plus' or 'matmul'")
        if value_reduction not in ("sum", "mean"):
            raise ValueError("value_reduction must be 'sum' or 'mean'")
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be >= 1")
        if not 0 <= attn_drop < 1:
            raise ValueError("attn_drop must be in [0, 1)")

        # Calls the user's original SelfAttention.__init__.
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mask_k_bias=mask_k_bias,
            device=device,
            qkv_quant_dim=qkv_quant_dim,
        )

        d = dim // num_heads
        initial = (math.sqrt(2 * d * (1 - 2 / math.pi))
                   if score_op == "dot_plus" else math.sqrt(d))
        if temperature_init is not None:
            initial = float(temperature_init)
        if not math.isfinite(initial) or initial <= 0:
            raise ValueError("temperature_init must be finite and positive")

        log_tau = torch.full((num_heads,), math.log(initial),
                             dtype=torch.float32, device=device)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_temperature", log_tau)

        self.score_op = score_op
        self.value_op = value_op
        self.value_reduction = value_reduction
        self.is_causal = is_causal
        self.query_chunk_size = query_chunk_size

    def temperatures(self) -> Tensor:
        """Positive temperature of each head, [num_heads]."""
        return self.log_temperature.exp()

    @staticmethod
    def _parse_attn_bias(attn_bias, B: int, H: int, N: int, device):
        """Return (float additive bias or None, bool allowed mask or None)."""
        if attn_bias is None:
            return None, None
        if not isinstance(attn_bias, Tensor):
            raise TypeError("attn_bias must be a Tensor (not an xFormers object)")

        m = attn_bias.to(device=device)
        if m.ndim == 2:
            m = m[None, None, :, :]
        elif m.ndim == 3:
            m = m[:, None, :, :]
        elif m.ndim != 4:
            raise ValueError("attn_bias must be 2D, 3D or 4D")
        try:
            m = torch.broadcast_to(m, (B, H, N, N))
        except RuntimeError as exc:
            raise ValueError("attn_bias is not broadcastable to [B,H,N,N]") from exc

        if m.dtype == torch.bool:
            return None, m  # True=allowed (torch SDPA convention)
        if not m.is_floating_point():
            raise TypeError("attn_bias must be bool or floating point")
        if not bool((torch.isfinite(m) | torch.isneginf(m)).all()):
            raise ValueError("attn_bias must contain finite values or -inf")
        valid = ~torch.isneginf(m)
        return m, (None if bool(valid.all()) else valid)

    def _dot_plus_merge(self, weights: Tensor, v: Tensor,
                        allowed: Tensor | None) -> Tensor:
        """-sum_k |weights[...,i,k]-v[...,k,j]| over INCLUDED keys only."""
        if allowed is None:
            out = -torch.cdist(weights, v.transpose(-2, -1), p=1)
            if self.value_reduction == "mean":
                out = out / weights.shape[-1]
            return out

        B, H, N, L = weights.shape
        # Query chunks avoid allocating the full [B,H,N,L,head_dim] tensor.
        chunks = []
        for start in range(0, N, self.query_chunk_size):
            stop = min(N, start + self.query_chunk_size)
            w = weights[:, :, start:stop, :, None]  # [B,H,chunk,L,1]
            diff = (w - v[:, :, None, :, :]).abs()
            valid = allowed[:, :, start:stop, :, None]
            dist = diff.masked_fill(~valid, 0.0).sum(dim=-2)
            if self.value_reduction == "mean":
                dist = dist / valid.sum(dim=-2).clamp_min(1)
            chunks.append(-dist)
        return torch.cat(chunks, dim=-2)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        """Match the original signature: [B,N,3*dim] -> [B,N,dim]."""
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        if qkv.shape[-1] != 3 * C:
            raise ValueError("qkv must be [B,N,3*dim]")
        H = self.num_heads
        d = C // H
        qkv = qkv.reshape(B, N, 3, H, d)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)  # Preserve the original RoPE.

        # Compute the experimental operation in float32; cast back before the
        # existing BitLinear output projection (safer for fp16/bfloat16).
        q, k, v32 = q.float(), k.float(), v.float()
        if self.score_op == "dot_plus":
            scores = -torch.cdist(q, k, p=1)
        else:
            scores = q @ k.transpose(-2, -1)
        scores = scores / self.temperatures().to(scores.dtype)[None, :, None, None]

        bias, allowed = self._parse_attn_bias(attn_bias, B, H, N, scores.device)
        if bias is not None:
            scores = scores + bias.to(scores.dtype)
        if self.is_causal:
            causal = torch.ones((N, N), dtype=torch.bool, device=scores.device).tril()
            causal = causal[None, None, :, :]
            allowed = causal if allowed is None else (allowed & causal)
        if allowed is not None:
            allowed = torch.broadcast_to(allowed, scores.shape)
            if not bool(allowed.any(dim=-1).all()):
                raise ValueError("Each query must have at least one allowed key")
            scores = scores.masked_fill(~allowed, -torch.inf)

        weights = scores.softmax(dim=-1)
        p = self.attn_drop.p
        if self.training and p > 0:
            keep = torch.rand_like(weights) >= p
            weights = weights * keep / (1 - p)
            if self.value_op == "dot_plus":
                # Dropped weights must exclude their corresponding V too.
                allowed = keep if allowed is None else (allowed & keep)

        if self.value_op == "dot_plus":
            out = self._dot_plus_merge(weights, v32, allowed)
        else:
            out = weights @ v32
        return out.to(dtype=v.dtype).transpose(1, 2).reshape(B, N, C)

class DotPlusSelfAttention(DotPlusAttentionMixin, SelfAttention):
    pass