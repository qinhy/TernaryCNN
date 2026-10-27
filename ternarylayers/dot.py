"""Experimental multi-head DotPlusAttention.

Scores:     S_ij = -sum_r |Q_ir - K_jr| / tau_h
Weights:    A = softmax(mask(S))
Value merge: O_ij = -reduce_{k in allowed(i)} |A_ik - V_kj|

Supports self/cross-attention, boolean attention and padding masks, causal
self-attention, learnable per-head temperature, dropout, and ablations.

Not an optimized FlashAttention replacement.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


ScoreOp = Literal["dot_plus", "dot"]
ValueOp = Literal["dot_plus", "matmul"]
Reduction = Literal["sum", "mean"]


def dot_plus(A: Tensor, B: Tensor) -> Tensor:
    """Generalized negative-L1 matrix product.

    A: [..., M, N], B: [..., N, P] -> [..., M, P]
    (A ⊕ B)[i,j] = -sum_k |A[i,k] - B[k,j]|.
    """
    if A.ndim < 2 or B.ndim < 2 or A.shape[-1] != B.shape[-2]:
        raise ValueError("Expected A[...,M,N], B[...,N,P] with same N")
    return -torch.cdist(A, B.transpose(-2, -1), p=1)


class DotPlusAttention(nn.Module):
    """Multi-head experimental attention with two independently selectable operations.

    Args:
        embed_dim: Model embedding dimension.
        num_heads: Head count (embed_dim must be divisible).
        score_op: 'dot_plus' (negative L1) or 'dot' (baseline).
        value_op: 'dot_plus' (negative L1) or 'matmul' (baseline).
        value_reduction: 'sum' gives literal dot-plus; 'mean' divides by the
            number of included keys, making output scale steadier across lengths.
        learnable_temperature: Train one positive softmax temperature per head.
        attn_dropout: Drop attention links during training.
        query_chunk_size: Limits intermediate memory for masked dot-plus merge.

    Inputs:
        x: [batch, query_length, embed_dim]
        context: [batch, key_length, embed_dim] or None for self-attention
        attn_mask: BOOL True=allowed, shape [Lq,Lk], [B,Lq,Lk], or
                   [B,H,Lq,Lk] (broadcastable batch/head dims allowed).
        key_padding_mask: BOOL True=padding/excluded, shape [B,Lk].
        is_causal: Only supported for self-attention, with Lq=Lk.
        return_weights: Return [B,H,Lq,Lk] post-dropout attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        score_op: ScoreOp = "dot_plus",
        value_op: ValueOp = "dot_plus",
        value_reduction: Reduction = "sum",
        learnable_temperature: bool = True,
        attn_dropout: float = 0.0,
        query_chunk_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if score_op not in ("dot_plus", "dot"):
            raise ValueError("score_op must be 'dot_plus' or 'dot'")
        if value_op not in ("dot_plus", "matmul"):
            raise ValueError("value_op must be 'dot_plus' or 'matmul'")
        if value_reduction not in ("sum", "mean"):
            raise ValueError("value_reduction must be 'sum' or 'mean'")
        if not 0 <= attn_dropout < 1:
            raise ValueError("attn_dropout must be in [0, 1)")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be > 0")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.score_op = score_op
        self.value_op = value_op
        self.value_reduction = value_reduction
        self.attn_dropout = float(attn_dropout)
        self.query_chunk_size = query_chunk_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        d = self.head_dim
        initial_tau = (math.sqrt(2 * d * (1 - 2 / math.pi))
                       if score_op == "dot_plus" else math.sqrt(d))
        t = torch.full((num_heads,), math.log(initial_tau))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(t)
        else:
            self.register_buffer("log_temperature", t)

    def temperatures(self) -> Tensor:
        """Current positive temperature for each head: [H]."""
        return self.log_temperature.exp()

    def _reshape_heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _allowed_mask(
        b: int, h: int, lq: int, lk: int, device: torch.device,
        *,
        is_causal: bool,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        allowed = torch.ones((b, h, lq, lk), dtype=torch.bool, device=device)
        if is_causal:
            if lq != lk:
                raise ValueError("Causal mode currently requires equal Q/K lengths")
            allowed &= torch.ones((lq, lk), device=device, dtype=torch.bool).tril()

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be boolean; True means allowed")
            if attn_mask.ndim == 2:
                mask = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                mask = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                mask = attn_mask
            else:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
            allowed &= mask.to(device=device)

        if key_padding_mask is not None:
            if (key_padding_mask.dtype != torch.bool or
                    tuple(key_padding_mask.shape) != (b, lk)):
                raise ValueError("key_padding_mask must be bool [B,Lk]")
            allowed &= ~key_padding_mask.to(device=device)[:, None, None, :]

        # An entirely invalid row makes softmax undefined. Reject it explicitly.
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("At least one query has no allowed key")
        return allowed

    def _merge_values(self, weights: Tensor, v: Tensor, allowed: Tensor) -> Tensor:
        """Computes -sum/mean of |weight[i,k] - v[k,j]| over allowed k.

        Never allows masked values to affect the output: masked attention
        weights equal zero, but |0-V| is NOT zero, so cdist(weights,V.T)
        alone would leak information from masked tokens.
        """
        if self.value_op == "matmul":
            return weights @ v

        if bool(allowed.all()):
            result = dot_plus(weights, v)
            if self.value_reduction == "mean":
                result = result / weights.size(-1)
            return result

        parts = []
        _, _, lq, _ = weights.shape
        # Chunk queries to bound temporary [B,H,chunk,Lk,Dv] memory.
        for begin in range(0, lq, self.query_chunk_size):
            end = min(begin + self.query_chunk_size, lq)
            w = weights[:, :, begin:end, :, None]  # [B,H,chunk,Lk,1]
            diff = (w - v[:, :, None, :, :]).abs()
            valid = allowed[:, :, begin:end, :, None]
            dist = diff.masked_fill(~valid, 0).sum(dim=-2)  # [B,H,chunk,Dv]
            if self.value_reduction == "mean":
                dist = dist / valid.sum(dim=-2).clamp_min(1)
            parts.append(-dist)
        return torch.cat(parts, dim=-2)

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        *,
        is_causal: bool = False,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError("x must be [batch, query_length, embed_dim]")
        if context is not None and is_causal:
            raise ValueError("is_causal is supported for self-attention only")
        kv = x if context is None else context
        if (kv.ndim != 3 or kv.shape[0] != x.shape[0] or
                kv.shape[-1] != self.embed_dim):
            raise ValueError("context must be [batch, key_length, embed_dim]")

        b, lq, _ = x.shape
        lk = kv.shape[1]
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(kv))
        v = self._reshape_heads(self.v_proj(kv))

        allowed = self._allowed_mask(
            b, self.num_heads, lq, lk, x.device,
            is_causal=is_causal, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        if self.score_op == "dot_plus":
            scores = -torch.cdist(q, k, p=1)
        else:
            scores = q @ k.transpose(-2, -1)
        scores = scores / self.temperatures().view(1, -1, 1, 1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        weights = torch.softmax(scores, dim=-1)

        if self.training and self.attn_dropout > 0:
            keep = torch.rand_like(weights) >= self.attn_dropout
            weights = weights * keep / (1 - self.attn_dropout)
            # For dot-plus, dropped positions must also be removed from L1 sum.
            if self.value_op == "dot_plus":
                allowed = allowed & keep

        z = self._merge_values(weights, v, allowed)
        z = z.transpose(1, 2).contiguous().reshape(b, lq, self.embed_dim)
        z = self.out_proj(z)
        return (z, weights) if return_weights else z


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DotPlusAttention(embed_dim=128, num_heads=4, attn_dropout=0.0)
    x = torch.randn(2, 12, 128, requires_grad=True)
    out, weights = model(x, is_causal=True, return_weights=True)
    loss = out.square().mean()
    loss.backward()
    print("Output:", tuple(out.shape))
    print("Weights:", tuple(weights.shape))
    print("Temperatures:", model.temperatures().detach().tolist())
    print("Gradient OK:", model.log_temperature.grad is not None)
