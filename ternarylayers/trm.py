import contextlib
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attn import AttentionModules
from .bit import Bit


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: (LN -> SelfAttn -> residual) + (LN -> FFN -> residual)."""

    def __init__(self, d_model: int, num_heads: int, ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = AttentionModules.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=dropout
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.pre_ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = ffn_ratio * d_model
        self.ffn = nn.Sequential(
            Bit.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Bit.Linear(hidden_dim, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_inp = self.pre_attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_inp,
            attn_inp,
            attn_inp,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        ffn_inp = self.pre_ffn_norm(x)
        x = x + self.ffn_dropout(self.ffn(ffn_inp))
        return x


class TransformerStage(nn.Module):
    """A plain transformer stage: a stack of TransformerBlock modules."""

    def __init__(
        self,
        depth: int,
        d_model: int,
        num_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x
    

class TransformerTRMStage(TransformerStage):
    """
    Transformer block stack + optional iterative refinement (AnytimeTRM-like).

    Core (no refinement): forward(x) == applying the TransformerBlock stack to x.

    Refinement process (if enabled):
      latent <- core( combine(x, solution, latent) + role_latent ) repeated num_latent_steps
      solution <- core( combine(solution, latent) + role_solution ) once
    """

    def __init__(
        self,
        depth: int,
        d_model: int,
        num_heads: int,
        ffn_ratio: int = 4,
        enable_refinement: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__(depth=depth, d_model=d_model, num_heads=num_heads, ffn_ratio=ffn_ratio)
        self.d_model = d_model
        self.enable_refinement = enable_refinement
        
        # --- refinement components (optional) ---
        if enable_refinement:
            self.combine_x_solution_latent = Bit.Linear(3 * d_model, d_model)
            self.combine_solution_latent = Bit.Linear(2 * d_model, d_model)

            # role embeddings: [2,1,1,D] => 0: update latent, 1: update solution
            self.role_embeddings = nn.Parameter(torch.zeros(2, 1, 1, d_model))
            nn.init.normal_(self.role_embeddings, std=init_std)

            # learned initial states (broadcast over seq_len): [1,1,D]
            self.init_solution_state = nn.Parameter(torch.zeros(1, 1, d_model))
            self.init_latent_state = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.init_solution_state, std=init_std)
            nn.init.normal_(self.init_latent_state, std=init_std)

    # -------- refinement helpers --------
    def init_refinement_states(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_refinement:
            raise RuntimeError("enable_refinement=False: init_refinement_states is unavailable.")

        solution = self.init_solution_state.expand(batch_size, seq_len, -1).to(device=device)
        latent = self.init_latent_state.expand(batch_size, seq_len, -1).to(device=device)

        if dtype is not None:
            solution = solution.to(dtype=dtype)
            latent = latent.to(dtype=dtype)

        return solution, latent
    
    # -------- core forward (no refinement) --------
    def forward_core(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_core(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

    def _refine_latent(
        self,
        x: torch.Tensor,
        solution: torch.Tensor,
        latent: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enable_refinement:
            raise RuntimeError("enable_refinement=False: _refine_latent is unavailable.")

        fused = self.combine_x_solution_latent(torch.cat([x, solution, latent], dim=-1))
        fused = fused + self.role_embeddings[0]  # latent update role
        return self.forward_core(fused, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

    def _refine_solution(
        self,
        solution: torch.Tensor,
        latent: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enable_refinement:
            raise RuntimeError("enable_refinement=False: _refine_solution is unavailable.")

        fused = self.combine_solution_latent(torch.cat([solution, latent], dim=-1))
        fused = fused + self.role_embeddings[1]  # solution update role
        return self.forward_core(fused, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

    def refine_states(
        self,
        x: torch.Tensor,
        solution: torch.Tensor,
        latent: torch.Tensor,
        num_latent_steps: int,
        damping: float,
        *,
        track_latent_grads: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x/solution/latent: [B, L, D]
        num_latent_steps: number of latent refinement steps
        damping: update damping (0..1)
        track_latent_grads: if False, latent steps run under no_grad (good for TTC/inference).
        """
        if not self.enable_refinement:
            raise RuntimeError("enable_refinement=False: refine_states is unavailable.")

        ctx = contextlib.nullcontext() if track_latent_grads else torch.no_grad()
        with ctx:
            for _ in range(num_latent_steps):
                latent_next = self._refine_latent(
                    x, solution, latent, attn_mask=attn_mask, key_padding_mask=key_padding_mask
                )
                latent = latent + damping * (latent_next - latent)

        solution_next = self._refine_solution(
            solution, latent, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        solution = solution + damping * (solution_next - solution)
        return solution, latent

    def forward_with_refinement(
        self,
        x: torch.Tensor,
        num_latent_steps: int = 4,
        damping: float = 0.5,
        solution: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        *,
        return_latent: bool = True,
        track_latent_grads: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        If solution/latent not provided, uses learned init_solution_state/init_latent_state.

        If return_latent=True, returns (solution, latent), else returns solution only.
        """
        bsz, seq_len, dim = x.shape
        if dim != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {dim}")

        if solution is None or latent is None:
            solution, latent = self.init_refinement_states(bsz, seq_len, x.device, dtype=x.dtype)

        solution, latent = self.refine_states(
            x,
            solution,
            latent,
            num_latent_steps=num_latent_steps,
            damping=damping,
            track_latent_grads=track_latent_grads,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return (solution, latent) if return_latent else solution