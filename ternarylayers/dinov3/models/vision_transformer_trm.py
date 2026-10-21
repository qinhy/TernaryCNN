# Copyright (c) HUANGYI QIN
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor

from ternarylayers.dinov3.layers.block import SelfAttentionTRMStage
from ternarylayers.dinov3.models.vision_transformer import DinoVisionTransformer

logger = logging.getLogger("dinov3_trm")


class DinoVisionTransformerTRM(DinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_trm()

    def init_trm(self):
        # Wrap the existing transformer blocks as a TRM stage
        self.blocks = SelfAttentionTRMStage(blocks_list=[*self.blocks])

    def _normalize_state_arg(self, state, n_items: int, name: str):
        """
        state can be:
          - None
          - Tensor (only valid when n_items == 1)
          - List[Tensor] of length n_items
        """
        if state is None:
            return [None] * n_items
        if isinstance(state, torch.Tensor):
            if n_items != 1:
                raise ValueError(
                    f"{name} was a Tensor but x_list has {n_items} items. "
                    f"Pass {name} as a list of length {n_items} for multi-crop input."
                )
            return [state]
        if isinstance(state, list):
            if len(state) != n_items:
                raise ValueError(f"{name} list length mismatch: got {len(state)}, expected {n_items}")
            return state
        raise TypeError(f"{name} must be None, Tensor, or List[Tensor]")

    def forward_features(
        self,
        x: Tensor | List[Tensor],
        masks: Optional[Tensor] = None,
        solution: Tensor | List[Tensor] | None = None,
        latent: Tensor | List[Tensor] | None = None,
        n: int = 4,
        T: int = 3,
        track_latent_grads: bool = True,  # paper-faithful default (more memory)
    ):
        args = dict(solution=solution,latent=latent,n=n,T=T,
                    track_latent_grads=track_latent_grads,)
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x],[masks],**args)[0]
        else:
            if masks is None:
                masks = [None] * len(x)
            return self.forward_features_list(x,masks,**args)

    def forward_features_list(
        self,
        x_list: List[Tensor],
        masks_list: List[Tensor | None],
        solution: Tensor | List[Tensor] | None = None,
        latent: Tensor | List[Tensor] | None = None,
        n: int = 4,
        T: int = 3,
        track_latent_grads: bool = True,
    ) -> List[Dict[str, Tensor]]:
        if len(x_list) != len(masks_list):
            raise ValueError(f"x_list and masks_list length mismatch: {len(x_list)} vs {len(masks_list)}")

        solution_list = self._normalize_state_arg(solution, len(x_list), "solution")
        latent_list = self._normalize_state_arg(latent, len(x_list), "latent")

        outputs = []

        # Process each crop-group independently (safe for variable H/W and token lengths)
        # This avoids the singleton-list refinement limitation.
        for idx, (t_x, t_masks, t_solution, t_latent) in enumerate(zip(x_list, masks_list, solution_list, latent_list)):
            t2_x, (H, W) = self.prepare_tokens_with_masks(t_x, t_masks)

            # Per-block rope for this crop-group (shared across TRM recursion passes)
            if self.rope_embed is not None:
                rope_per_block = [self.rope_embed(H=H, W=W) for _ in self.blocks.blocks]
            else:
                rope_per_block = [None for _ in self.blocks.blocks]

            # TRM refinement on a single tensor
            x_refined, z_latent = self.blocks.forward_with_refinement(
                t2_x,
                rope_per_block,
                solution=t_solution,
                latent=t_latent,
                num_latent_steps=n,
                T=T,
                return_latent=True,
                track_latent_grads=track_latent_grads,
            )

            # Normalization path (same logic as base class)
            x = x_refined
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry corresponds to local crops (same as base behavior)
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

            outputs.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,         # solution state (y-like hidden)
                    "z_latent": z_latent,   # latent state (z)
                    "masks": t_masks,
                }
            )

        return outputs

    # Optional: make unsupported behavior explicit (instead of silently breaking)
    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1):
        raise NotImplementedError(
            "get_intermediate_layers is not implemented for TRM-wrapped blocks. "
            "Use the base DinoVisionTransformer or add a TRM-aware version."
        )

