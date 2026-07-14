# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from ternarylayers.dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from ternarylayers.dinov3.layers.bitlayers import Linear as BitLinear
from ternarylayers.dinov3.layers.block import SelfAttentionTRMStage
from ternarylayers.dinov3.layers.patch_embed import PatchEmbedNoConv
from ternarylayers.dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, (nn.Linear, BitLinear)):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        no_conv: bool = False,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.pos_embed_rope_base = pos_embed_rope_base
        self.pos_embed_rope_min_period = pos_embed_rope_min_period
        self.pos_embed_rope_max_period = pos_embed_rope_max_period
        self.pos_embed_rope_normalize_coords = pos_embed_rope_normalize_coords
        self.pos_embed_rope_shift_coords = pos_embed_rope_shift_coords
        self.pos_embed_rope_jitter_coords = pos_embed_rope_jitter_coords
        self.pos_embed_rope_rescale_coords = pos_embed_rope_rescale_coords
        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.layerscale_init = layerscale_init
        self.norm_layer = norm_layer
        self.ffn_layer = ffn_layer
        self.ffn_bias = ffn_bias
        self.proj_bias = proj_bias
        self.n_storage_tokens = n_storage_tokens
        self.mask_k_bias = mask_k_bias
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.device = device
        self.no_conv = no_conv
        
        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = (PatchEmbed if not self.no_conv else PatchEmbedNoConv)(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim).to(device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim).to(device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim).to(device=device))

    def clone(self):
        return  self.__class__(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            pos_embed_rope_base=self.pos_embed_rope_base,
            pos_embed_rope_min_period=self.pos_embed_rope_min_period,
            pos_embed_rope_max_period=self.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=self.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=self.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=self.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=self.pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=self.pos_embed_rope_dtype,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            ffn_ratio=self.ffn_ratio,
            qkv_bias=self.qkv_bias,
            drop_path_rate=self.drop_path_rate,
            layerscale_init=self.layerscale_init,
            norm_layer=self.norm_layer,
            ffn_layer=self.ffn_layer,
            ffn_bias=self.ffn_bias,
            proj_bias=self.proj_bias,
            n_storage_tokens=self.n_storage_tokens,
            mask_k_bias=self.mask_k_bias,
            untie_cls_and_patch_norms=self.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=self.untie_global_and_local_cls_norm,
            device=self.device,
            no_conv=self.no_conv,
        )
    
    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0.0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
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
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

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

def vit_femto(patch_size=16, cls=DinoVisionTransformer, **kwargs):
    # ultra tiny, minimal capacity
    model = cls(
        patch_size=patch_size,
        embed_dim=64,
        depth=3,
        num_heads=1,     # 64 % 1 == 0
        ffn_ratio=2,
        **kwargs,
    )
    return model

def vit_pico(patch_size=16, cls=DinoVisionTransformer, **kwargs):
    # smaller than micro, good for very fast experiments
    model = cls(
        patch_size=patch_size,
        embed_dim=96,
        depth=4,
        num_heads=3,     # 96 % 3 == 0
        ffn_ratio=2,
        **kwargs,
    )
    return model

def vit_micro(patch_size=8, cls=DinoVisionTransformer, **kwargs):
    # very small + very fast
    model = cls(
        patch_size=patch_size,
        embed_dim=128,
        depth=4,
        num_heads=2,     # 128 % 2 == 0
        ffn_ratio=2,
        **kwargs,
    )
    return model

def vit_nano(patch_size=8, cls=DinoVisionTransformer, **kwargs):
    # small, still fast, a bit more capacity
    model = cls(
        patch_size=patch_size,
        embed_dim=192,
        depth=6,
        num_heads=3,     # 192 % 3 == 0
        ffn_ratio=2,
        **kwargs,
    )
    return model

def vit_tiny(patch_size=8, cls=DinoVisionTransformer, **kwargs):
    # common "tiny-ish" setting for quick experiments
    model = cls(
        patch_size=patch_size,
        embed_dim=256,
        depth=6,
        num_heads=4,     # 256 % 4 == 0
        ffn_ratio=4,
        **kwargs,
    )
    return model

def vit_small(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    model = cls(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    model = cls(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    model = cls(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    model = cls(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    model = cls(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = cls(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16,  cls=DinoVisionTransformer, **kwargs):
    model = cls(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
