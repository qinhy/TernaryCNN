# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import contextlib
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ternarylayers.dinov3.layers.bitlayers import Linear
from ternarylayers.dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale  # , DropPath

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values).to(device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values).to(device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError

class SelfAttentionTRMStage(nn.Module):
    """
    Transformer block stack + optional iterative refinement (TRM-style).

    Core:
        forward(x, rope) -> apply stacked SelfAttentionBlocks (rope is threaded through).

    Refinement (unrolled):
        latent   <- core( LN(x + solution + latent) ) repeated num_latent_steps times
        solution <- core( LN(solution + latent) ) once
        Both updates use a damped / relaxed update:
            state <- state + damping * (state_next - state)

    Notes:
      - Core forward supports Tensor or List[Tensor] (because SelfAttentionBlock does).
      - Refinement supports Tensor or singleton List[Tensor] only.
      - rope_or_rope_list can be:
          * None
          * a single rope object (shared across all blocks)
          * a per-block list of rope objects (len == depth)
    """

    def __init__(
        self,
        depth: int = 0,
        dim: int = 0,
        num_heads: int = 0,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Optional[Callable[..., nn.Module]] = None,
        ffn_layer: Optional[Callable[..., nn.Module]] = None,
        mask_k_bias: bool = False,
        init_std: float = 0.02,
        device=None,
        blocks_list=None,
    ):
        super().__init__()

        if attn_class is None:
            attn_class = SelfAttention
        if ffn_layer is None:
            ffn_layer = Mlp

        if blocks_list is None:
            blocks_list = [
                SelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attn_class=attn_class,
                    ffn_layer=ffn_layer,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(depth)
            ]
        else:
            if len(blocks_list) == 0:
                raise ValueError("blocks_list must not be empty")
            dim = blocks_list[0].dim
            depth = len(blocks_list)

        self.blocks = nn.ModuleList(blocks_list)
        self.d_model = dim
        self.depth = depth

        # Learned initial states: [1, 1, D]
        self.init_solution_state = nn.Parameter(torch.empty(1, 1, dim, device=device))
        self.init_latent_state = nn.Parameter(torch.empty(1, 1, dim, device=device))
        nn.init.normal_(self.init_solution_state, std=init_std)
        nn.init.normal_(self.init_latent_state, std=init_std)

    # -----------------------------
    # Small utilities
    # -----------------------------
    @staticmethod
    def _ln_no_affine(t: Tensor, eps: float = 1e-5) -> Tensor:
        """Parameter-free LayerNorm (no affine weight/bias)."""
        return torch.nn.functional.layer_norm(t, (t.shape[-1],), weight=None, bias=None, eps=eps)

    @staticmethod
    def _maybe_no_grad(enabled: bool):
        """If enabled=True -> no_grad(); else -> nullcontext()."""
        return torch.no_grad() if enabled else contextlib.nullcontext()

    def _normalize_rope_for_block(
        self,
        x: Union[Tensor, List[Tensor]],
        rope_or_rope_list: Any,
        block_idx: int,
    ):
        if rope_or_rope_list is None:
            return [None for _ in x] if isinstance(x, list) else None

        # If rope_or_rope_list matches depth, treat it as per-block rope.
        rope_for_block = (
            rope_or_rope_list[block_idx]
            if isinstance(rope_or_rope_list, list) and len(rope_or_rope_list) == len(self.blocks)
            else rope_or_rope_list
        )

        # For list inputs, block expects a rope list aligned with items.
        if isinstance(x, list):
            if rope_for_block is None:
                return [None for _ in x]
            if isinstance(rope_for_block, list):
                if len(rope_for_block) != len(x):
                    raise ValueError(
                        f"rope list length mismatch for list input: got {len(rope_for_block)}, expected {len(x)}"
                    )
                return rope_for_block
            return [rope_for_block for _ in x]

        return rope_for_block

    def _unwrap_singleton_for_refinement(self, x, rope):
        """
        Refinement uses tensor states [B, L, D], so only Tensor or singleton List[Tensor] is supported.
        Returns:
            x_tensor, rope_for_tensor_path, wrap_solution_back_as_list
        """
        if not isinstance(x, list):
            return x, rope, False

        if len(x) != 1:
            raise ValueError(
                "Refinement currently supports only Tensor or singleton List[Tensor]. "
                "Got a list with length > 1."
            )

        x_tensor = x[0]
        wrap_back = True

        # Common list-API pattern: rope passed as singleton list for singleton x-list.
        if isinstance(rope, list) and len(rope) == 1:
            rope = rope[0]

        return x_tensor, rope, wrap_back

    # -----------------------------
    # Init refinement states
    # -----------------------------
    def init_refinement_states(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        solution = self.init_solution_state.expand(batch_size, seq_len, -1)
        latent = self.init_latent_state.expand(batch_size, seq_len, -1)

        if dtype is None:
            dtype = solution.dtype

        return (
            solution.to(device=device, dtype=dtype),
            latent.to(device=device, dtype=dtype),
        )

    # -----------------------------
    # Core forward (no refinement)
    # -----------------------------
    def forward(
        self,
        x_or_x_list: Union[Tensor, List[Tensor]],
        rope_or_rope_list=None,
    ) -> Union[Tensor, List[Tensor]]:
        x = x_or_x_list
        for i, block in enumerate(self.blocks):
            block_rope = self._normalize_rope_for_block(x, rope_or_rope_list, i)
            x = block(x, block_rope)
        return x

    # -----------------------------
    # Refinement step
    # -----------------------------
    def refine_states(
        self,
        x: Tensor,
        solution: Tensor,
        latent: Tensor,
        num_latent_steps: int,
        damping: float,
        rope_list=None,
        track_latent_grads: bool = False,
        track_solution_grads: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        if num_latent_steps < 0:
            raise ValueError(f"num_latent_steps must be >= 0, got {num_latent_steps}")

        # ---- Latent refinement (optional) ----
        with self._maybe_no_grad(enabled=not track_latent_grads):
            for _ in range(num_latent_steps):
                fused_latent = self._ln_no_affine(x + solution + latent)
                latent_next = self.forward(fused_latent, rope_list)
                latent = latent + damping * (latent_next - latent)

        # ---- Solution refinement (always) ----
        with self._maybe_no_grad(enabled=not track_solution_grads):
            # If we skipped latent refinement, latent has no x-dependent info.
            # Inject x once so the solution update isn't "blind" to the input.
            if num_latent_steps == 0:
                fused_solution = x + solution + latent
            else:
                fused_solution = solution + latent

            fused_solution = self._ln_no_affine(fused_solution)
            solution_next = self.forward(fused_solution, rope_list)
            solution = solution + damping * (solution_next - solution)

        return solution, latent

    # -----------------------------
    # Full TRM-style forward
    # -----------------------------
    def forward_with_refinement(
        self,
        x: Union[Tensor, List[Tensor]],
        rope_list=None,
        num_latent_steps: int = 4,
        T: int = 1,
        damping: float = 0.5,
        solution: Optional[Tensor] = None,
        latent: Optional[Tensor] = None,
        *,
        return_latent: bool = False,
        track_latent_grads: bool = False,
    ):
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")

        x_tensor, rope_tensorish, wrap_solution_back = self._unwrap_singleton_for_refinement(x, rope_list)

        bsz, seq_len, dim = x_tensor.shape
        if dim != self.d_model:
            raise ValueError(f"Input dim {dim} does not match stage dim {self.d_model}")

        if solution is None or latent is None:
            solution, latent = self.init_refinement_states(
                batch_size=bsz,
                seq_len=seq_len,
                device=x_tensor.device,
                dtype=x_tensor.dtype,
            )

        # First T-1 passes: no grads on solution/latent
        for _ in range(T - 1):
            solution, latent = self.refine_states(
                x=x_tensor,
                solution=solution,
                latent=latent,
                num_latent_steps=num_latent_steps,
                damping=damping,
                rope_list=rope_tensorish,
                track_latent_grads=False,
                track_solution_grads=False,
            )

        # Final pass: solution grads on; latent grads optional
        solution, latent = self.refine_states(
            x=x_tensor,
            solution=solution,
            latent=latent,
            num_latent_steps=num_latent_steps,
            damping=damping,
            rope_list=rope_tensorish,
            track_latent_grads=track_latent_grads,
            track_solution_grads=True,
        )

        solution_out: Union[Tensor, List[Tensor]] = [solution] if wrap_solution_back else solution
        return (solution_out, latent) if return_latent else solution_out   

class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn









