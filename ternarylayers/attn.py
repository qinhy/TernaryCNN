from __future__ import annotations

from typing import  Optional, Sequence, Tuple, Union

from pydantic import Field, model_validator
from torch import nn
import torch
import torch.nn.functional as F

from ternarylayers.bit import Bit

from .convs import Conv2dModels
from .helpers import convert_padding, to_2tuple
from .pool import PoolModels
from .norms import NormModels
from .drop import DropPath
from .base import CommonModel, CommonModule
from .linear import LinearModels

class AttentionModels:
    class BasicModel(CommonModel):
        def build(self): return self._build(self,AttentionModules)

    class Attention(BasicModel):
        pass

class AttentionModules:
    class Module(CommonModule):
        def __init__(self, para, para_cls=None):
            super().__init__(para, AttentionModels, para_cls)

    class Attention(Module):        
        pass

    class MultiheadAttention(nn.Module):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: int | None = None,
            vdim: int | None = None,
            batch_first: bool = False,
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}

            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.dropout = float(dropout)
            self.batch_first = batch_first
            self.add_zero_attn = add_zero_attn

            self.kdim = embed_dim if kdim is None else kdim
            self.vdim = embed_dim if vdim is None else vdim

            if embed_dim % num_heads != 0:
                raise ValueError("embed_dim must be divisible by num_heads")
            self.head_dim = embed_dim // num_heads
            self.scaling:float = float(self.head_dim) ** -0.5

            self.linear_Q = Bit.Linear(embed_dim, embed_dim, bias=bias)
            self.linear_K = Bit.Linear(self.kdim, embed_dim, bias=bias)
            self.linear_V = Bit.Linear(self.vdim, embed_dim, bias=bias)
            self.out_proj = Bit.Linear(embed_dim, embed_dim, bias=bias)

            if add_bias_kv:
                self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
                self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
            else:
                self.bias_k = None
                self.bias_v = None

            self._reset_parameters()

        def _reset_parameters(self) -> None:
            nn.init.xavier_uniform_(self.linear_Q.weight)
            nn.init.xavier_uniform_(self.linear_K.weight)
            nn.init.xavier_uniform_(self.linear_V.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

            for lin in (self.linear_Q, self.linear_K, self.linear_V, self.out_proj):
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0.0)

            if self.bias_k is not None:
                nn.init.xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                nn.init.xavier_normal_(self.bias_v)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            need_weights: bool = True,
            attn_mask: torch.Tensor | None = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if attn_mask is not None and is_causal:
                raise AssertionError("Only allow causal mask or attn_mask, not both")

            if self.batch_first:
                # (N, L, E) -> (L, N, E)
                query, key, value = (x.transpose(0, 1) for x in (query, key, value))

            tgt_len, bsz, embed_dim = query.shape
            if embed_dim != self.embed_dim:
                raise ValueError(f"Expected query embed_dim={self.embed_dim}, got {embed_dim}")
            if key.shape[1] != bsz or value.shape[1] != bsz:
                raise ValueError("key/value batch size must match query batch size")
            if key.shape[0] != value.shape[0]:
                raise ValueError("key/value sequence length must match")

            src_len = key.shape[0]

            # causal mask (bool) where True means "masked out"
            if is_causal:
                if src_len != tgt_len:
                    raise ValueError("is_causal=True requires src_len == tgt_len")
                attn_mask = torch.ones(tgt_len, src_len, dtype=torch.bool, device=query.device).triu(1)

            # projections
            q = self.linear_Q(query) * self.scaling  # (L, N, E)
            k = self.linear_K(key)
            v = self.linear_V(value)

            # attn_mask checks/reshape
            if attn_mask is not None:
                if attn_mask.dtype == torch.uint8:
                    print(
                        "Byte tensor for `attn_mask` is deprecated. Use bool instead.",
                        stacklevel=2,
                    )
                    attn_mask = attn_mask.to(torch.bool)

                if not (attn_mask.is_floating_point() or attn_mask.dtype == torch.bool):
                    raise TypeError(f"attn_mask must be float or bool, got {attn_mask.dtype}")

                if attn_mask.dim() == 2:
                    if list(attn_mask.shape) != [tgt_len, src_len]:
                        raise RuntimeError("The size of the 2D attn_mask is not correct.")
                    attn_mask = attn_mask.unsqueeze(0)  # (1, L, S)
                elif attn_mask.dim() == 3:
                    if list(attn_mask.shape) != [bsz * self.num_heads, tgt_len, src_len]:
                        raise RuntimeError("The size of the 3D attn_mask is not correct.")
                else:
                    raise RuntimeError(f"attn_mask dim {attn_mask.dim()} is not supported")

            # key_padding_mask checks
            if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
                print(
                    "Byte tensor for `key_padding_mask` is deprecated. Use bool instead.",
                    stacklevel=2,
                )
                key_padding_mask = key_padding_mask.to(torch.bool)
            if key_padding_mask is not None:
                if key_padding_mask.shape != (bsz, src_len):
                    raise RuntimeError("key_padding_mask shape must be (N, S)")

            # add_bias_kv: append 1 token to key/value
            if self.bias_k is not None and self.bias_v is not None:
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)], dim=0)
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)], dim=0)
                src_len = src_len + 1
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))

            # add_zero_attn: append 1 token of zeros
            if self.add_zero_attn:
                k = torch.cat([k, torch.zeros((1, bsz, self.embed_dim), device=k.device, dtype=k.dtype)], dim=0)
                v = torch.cat([v, torch.zeros((1, bsz, self.embed_dim), device=v.device, dtype=v.dtype)], dim=0)
                src_len = src_len + 1
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))

            # reshape to (N*H, L/S, D)
            q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            # attention logits: (N*H, L, S)
            attn_logits = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_logits.masked_fill_(attn_mask, float("-inf"))
                else:
                    attn_logits = attn_logits + attn_mask

            if key_padding_mask is not None:
                attn_logits = attn_logits.view(bsz, self.num_heads, tgt_len, src_len)
                attn_logits = attn_logits.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
                attn_logits = attn_logits.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_weights, v)  # (N*H, L, D)
            attn_output = (
                attn_output.transpose(0, 1)
                .contiguous()
                .view(tgt_len, bsz, self.embed_dim)
            )

            if self.batch_first:
                attn_output = attn_output.transpose(0, 1)  # (N, L, E)

            attn_output = self.out_proj(attn_output)

            if not need_weights:
                return attn_output, None

            # return weights as (N, L, S) or (N, H, L, S)
            attn_weights_ = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_weights_ = attn_weights_.mean(dim=1)  # (N, L, S)
            return attn_output, attn_weights_
        