# Matryoshka-BitViT: shared ternary ViT-Micro -> Pico -> Femto.
# Install alongside vision_transformer.py in ternarylayers/dinov3/models/.
# This is an extension of the user-provided DINOv3-derived source; observe
# its existing DINOv3 license terms.

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternarylayers.dinov3.layers.rms_norm import MatryoshkaRMSNorm

from .vision_transformer import DinoVisionTransformer
from ternarylayers.dinov3.layers.block import (
    MatryoshkaLayerNorm,
    SelfMatryoshkaAttentionBlock,
)


# One head dimension is necessary for prefix-compatible QKV and RoPE.
# These are *elastic equivalents*, not identical attention configurations to
# the original presets (Micro 2x64 and Femto 1x64).
SPLITS = {
    "micro": (128, 4),
    "pico": (96, 4),
    "femto": (64, 3),
}
HEAD_DIM = 32


class DinoVisionMatryoshkaMicro(DinoVisionTransformer):
    """Train once at width 128; run/export Micro, Pico and Femto prefixes.

    Width/depth are per-forward selections; parameters remain shared.
    A recognised no-conv linear patch projector is width-sliced before
    quantization. Other patch embeddings fall back to full projection; the
    exported standalone submodel always has a smaller patch projection.
    The base DINOv3 forward_features dict keys are preserved.
    """

    def __init__(self, *, patch_size: int = 8, **kwargs):
        for reserved in ("embed_dim", "depth", "num_heads", "ffn_ratio"):
            if reserved in kwargs:
                raise TypeError(f"{reserved} is fixed for Matryoshka Micro; use the preset")
        super().__init__(
            patch_size=patch_size,
            embed_dim=128,
            depth=4,
            num_heads=4,
            ffn_ratio=2.0,
            **kwargs,
        )
        if self.ffn_layer != "mlp":
            raise ValueError("This implementation currently supports ffn_layer='mlp' only")
        if self.norm_layer not in ("layernorm", "layernormbf16", "rmsnorm"):
            raise ValueError(f"Unsupported elastic norm {self.norm_layer!r}")

        # The parent builds the ordinary, unmodified model. Replace its blocks
        # without changing the names in their state_dict (blocks.i.*).
        norm_factory = (
            MatryoshkaRMSNorm if self.norm_layer == "rmsnorm"
            else partial(MatryoshkaLayerNorm, eps=(1e-5 if self.norm_layer == "layernormbf16" else 1e-6))
        )
        def per_block(v, i):
            return v if isinstance(v, int) else v[i]

        qdim = kwargs.get("sattn_qkv_quant_dim", 0)
        fc1dim = kwargs.get("sattn_fc1_quant_dim", 0)
        fc2dim = kwargs.get("sattn_fc2_quant_dim", 1)
        self.blocks = nn.ModuleList([
            SelfMatryoshkaAttentionBlock(
                dim=128,
                num_heads=4,
                ffn_ratio=2.0,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                drop_path=self.drop_path_rate,
                norm_layer=norm_factory,
                init_values=self.layerscale_init,
                mask_k_bias=self.mask_k_bias,
                device=self.device,
                qkv_quant_dim=per_block(qdim, i),
                fc1_quant_dim=per_block(fc1dim, i),
                fc2_quant_dim=per_block(fc2dim, i),
            )
            for i in range(self.depth)
        ])
        self.norm = norm_factory(128).to(device=self.device)
        if self.cls_norm is not None:
            self.cls_norm = norm_factory(128).to(device=self.device)
        if self.local_cls_norm is not None:
            self.local_cls_norm = norm_factory(128).to(device=self.device)
        # Parameter initialisation follows the parent contract: call
        # model.init_weights() just as for the standard DinoVisionTransformer.

    @staticmethod
    def _resolve(width: int, depth: Optional[int], out_dim: Optional[int]):
        allowed = {w: d for w, d in SPLITS.values()}
        if width not in allowed:
            raise ValueError(f"width must be one of {tuple(allowed)}, got {width}")
        if depth is None:
            depth = allowed[width]
        if not isinstance(depth, int) or not (1 <= depth <= 4):
            raise ValueError("depth must be an integer from 1 to 4")
        if out_dim is None:
            out_dim = width
        if not isinstance(out_dim, int) or not (1 <= out_dim <= width):
            raise ValueError("out_dim must be an integer in [1, width]")
        return width, depth, out_dim

    def _embed_patches(self, images: Tensor, width: int) -> Tensor:
        """Prefer a narrower no-conv patch GEMM before weight quantization.

        Recognises simple direct linear patch projectors. For unfamiliar
        PatchEmbedNoConv implementations (and convolutional embedding), use the
        original full embedding followed by a prefix. This fallback is correct
        for dimensions but may not reproduce post-split quantization scales.
        """
        if self.no_conv and width != self.embed_dim:
            patch_in = self.in_chans * self.patch_size**2
            for attr in ("proj", "projection", "linear"):
                proj = getattr(self.patch_embed, attr, None)
                weight = getattr(proj, "weight", None)
                if (
                    isinstance(weight, Tensor) and weight.ndim == 2
                    and tuple(weight.shape) == (self.embed_dim, patch_in)
                ):
                    b, c, h, w = images.shape
                    if c != self.in_chans or h % self.patch_size or w % self.patch_size:
                        raise ValueError("Input channels/size are incompatible with patch_size")
                    patch_tokens = F.unfold(
                        images, kernel_size=self.patch_size, stride=self.patch_size
                    ).transpose(1, 2)
                    active_weight = weight[:width]
                    quantizer = getattr(proj, "w_q", None)
                    if quantizer is not None:
                        active_weight = quantizer(active_weight)
                    bias = getattr(proj, "bias", None)
                    active_bias = bias[:width] if bias is not None else None
                    return F.linear(patch_tokens, active_weight, active_bias).reshape(
                        b, h // self.patch_size, w // self.patch_size, width
                    )
        return self.patch_embed(images)[..., :width]

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Optional[Tensor] = None, *, width: int = 128
    ):
        self._resolve(width, None, None)
        # PatchEmbed / PatchEmbedNoConv returns [B,H,W,width].
        patches = self._embed_patches(x, width)
        bsz, h, w, _ = patches.shape
        patches = patches.flatten(1, 2)
        cls = self.cls_token[..., :width]
        mask_token = self.mask_token[..., :width]
        if masks is not None:
            if masks.shape != (bsz, h * w):
                raise ValueError(f"masks must be [B,{h*w}], got {tuple(masks.shape)}")
            patches = torch.where(masks.unsqueeze(-1), mask_token.to(patches.dtype), patches)
        else:
            # Keep the unused mask parameter in the graph, as the parent does.
            cls = cls + 0.0 * mask_token
        storage = (
            self.storage_tokens[..., :width].expand(bsz, -1, -1)
            if self.n_storage_tokens else patches.new_empty(bsz, 0, width)
        )
        return torch.cat((cls.expand(bsz, -1, -1), storage, patches), dim=1), (h, w)

    def forward_features_list(
        self,
        x_list: List[Tensor],
        masks_list: Optional[List[Optional[Tensor]]] = None,
        *,
        width: int = 128,
        depth: Optional[int] = None,
        out_dim: Optional[int] = None,
    ) -> List[Dict[str, Tensor]]:
        width, depth, out_dim = self._resolve(width, depth, out_dim)
        if masks_list is None:
            masks_list = [None] * len(x_list)
        if len(x_list) != len(masks_list):
            raise ValueError("x_list and masks_list length mismatch")
        if not x_list:
            return []

        tokens, shapes = [], []
        for image, mask in zip(x_list, masks_list):
            token, hw = self.prepare_tokens_with_masks(image, mask, width=width)
            tokens.append(token)
            shapes.append(hw)
        rope = [self.rope_embed(H=h, W=w) if self.rope_embed is not None else None
                for h, w in shapes]
        for block in self.blocks[:depth]:
            tokens = block(tokens, rope)

        result = []
        prefix = 1 + self.n_storage_tokens
        for idx, (t, mask) in enumerate(zip(tokens, masks_list)):
            if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                cls_reg = self.local_cls_norm(t[:, :prefix])
            elif self.untie_cls_and_patch_norms:
                cls_reg = self.cls_norm(t[:, :prefix])
            else:
                cls_reg = self.norm(t[:, :prefix])
            patches = self.norm(t[:, prefix:])
            result.append({
                "x_norm_clstoken": cls_reg[:, 0, :out_dim],
                "x_storage_tokens": cls_reg[:, 1:, :out_dim],
                "x_norm_patchtokens": patches[..., :out_dim],
                "x_prenorm": t[..., :out_dim],
                "masks": mask,
            })
        return result

    def forward_features(
        self,
        x: Union[Tensor, List[Tensor]],
        masks: Optional[Union[Tensor, List[Optional[Tensor]]]] = None,
        *,
        width: int = 128,
        depth: Optional[int] = None,
        out_dim: Optional[int] = None,
    ):
        if isinstance(x, Tensor):
            return self.forward_features_list(
                [x], [masks], width=width, depth=depth, out_dim=out_dim
            )[0]
        return self.forward_features_list(
            x, masks, width=width, depth=depth, out_dim=out_dim
        )

    def forward(
        self,
        x: Union[Tensor, List[Tensor]],
        masks=None,
        *,
        width: int = 128,
        depth: Optional[int] = None,
        out_dim: Optional[int] = None,
        is_training: bool = False,
    ):
        ret = self.forward_features(x, masks, width=width, depth=depth, out_dim=out_dim)
        if is_training:
            return ret
        if isinstance(ret, list):
            return [self.head(r["x_norm_clstoken"]) for r in ret]
        return self.head(ret["x_norm_clstoken"])

    def clone(self):
        clone = self.__class__(
            patch_size=self.patch_size,
            img_size=self.img_size,
            in_chans=self.in_chans,
            pos_embed_rope_base=self.pos_embed_rope_base,
            pos_embed_rope_min_period=self.pos_embed_rope_min_period,
            pos_embed_rope_max_period=self.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=self.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=self.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=self.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=self.pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=self.pos_embed_rope_dtype,
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
            no_conv=self.no_conv,
        ).to(device=self.cls_token.device, dtype=self.cls_token.dtype)
        clone.load_state_dict(self.state_dict(), strict=True)
        return clone

    def export_subnetwork(self, name: str) -> DinoVisionTransformer:
        """Return a compact, *ordinary* DINOv3 model with prefix-copied weights.

        Export first materialises floating-point/learned weights. The real
        BitLinear ternary freeze/export must be applied separately, with
        the width-specific quantization scale recomputed AFTER slicing.
        """
        if name not in SPLITS:
            raise ValueError(f"name must be one of {tuple(SPLITS)}, got {name!r}")
        width, depth = SPLITS[name]
        target = DinoVisionTransformer(
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
            embed_dim=width,
            depth=depth,
            num_heads=width // HEAD_DIM,
            ffn_ratio=2.0,
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
            no_conv=self.no_conv,
        ).to(device=self.cls_token.device, dtype=self.cls_token.dtype)

        source_sd = self.state_dict()
        small_sd = target.state_dict()
        copied = {}
        for key, dst in small_sd.items():
            if key not in source_sd:
                raise KeyError(f"Missing exported state key: {key}")
            src = source_sd[key]
            if src.shape == dst.shape:
                copied[key] = src.clone()
                continue
            if src.ndim != dst.ndim or any(s < d for s, d in zip(src.shape, dst.shape)):
                raise ValueError(f"Cannot prefix-export {key}: {tuple(src.shape)} -> {tuple(dst.shape)}")
            # Q, K and V are packed in three *separate* full-size row groups.
            # Only attention QKV's packed leading output dimension gets
            # group-aware slicing. Regular matrices use top-left prefixes.
            if ".attn.qkv." in key and src.ndim >= 1 and src.shape[0] == 3 * self.embed_dim and dst.shape[0] == 3 * width:
                chunks = src.reshape(3, self.embed_dim, *src.shape[1:])
                sections = [slice(None), slice(0, width)] + [slice(0, n) for n in dst.shape[1:]]
                copied[key] = chunks[tuple(sections)].reshape(dst.shape).clone()
            else:
                copied[key] = src[tuple(slice(0, n) for n in dst.shape)].clone()
        target.load_state_dict(copied, strict=True)
        target.train(self.training)
        return target


def vit_matryoshka_micro(patch_size: int = 8, **kwargs):
    """Micro width=128/depth=4, with Pico/Femto width/depth prefixes."""
    return DinoVisionMatryoshkaMicro(patch_size=patch_size, **kwargs)


class MatryoshkaMNISTClassifier(nn.Module):
    """Optional shared 10-class head for sandwich training on MNIST.

    Head rows represent classes; head columns are prefix-sliced so each
    width gets a genuine w->num_classes matrix multiplication.
    """

    def __init__(self, num_classes: int = 10, *, backbone: Optional[DinoVisionMatryoshkaMicro] = None, **kwargs):
        super().__init__()
        self.backbone = backbone if backbone is not None else vit_matryoshka_micro(**kwargs)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: Tensor, *, name: str = "micro") -> Tensor:
        if name not in SPLITS:
            raise ValueError(f"Unknown subnet {name!r}")
        width, depth = SPLITS[name]
        cls = self.backbone(images, width=width, depth=depth)
        return F.linear(cls, self.classifier.weight[:, :width], self.classifier.bias)

    def export_subnetwork(self, name: str) -> nn.Module:
        """Return a standalone classifier with the ordinary reduced ViT."""
        width, _ = SPLITS[name]
        model = _StandaloneClassifier(self.backbone.export_subnetwork(name), width, self.classifier.out_features)
        with torch.no_grad():
            model.classifier.weight.copy_(self.classifier.weight[:, :width])
            model.classifier.bias.copy_(self.classifier.bias)
        model.train(self.training)
        return model


class _StandaloneClassifier(nn.Module):
    def __init__(self, backbone: DinoVisionTransformer, width: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(width, num_classes).to(
            device=backbone.cls_token.device, dtype=backbone.cls_token.dtype
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.backbone(images))


def build_standalone_classifier(
    name: str, *, num_classes: int = 10, patch_size: int = 8, **backbone_kwargs
) -> nn.Module:
    """Rebuild an exported split without allocating the full 128-wide model.

    Example:
        small = build_standalone_classifier('femto', img_size=32, no_conv=True)
        small.load_state_dict(torch.load('femto_extracted.pt',
                                         weights_only=True)['state_dict'])
    """
    if name not in SPLITS:
        raise ValueError(f"name must be one of {tuple(SPLITS)}, got {name!r}")
    width, depth = SPLITS[name]
    for reserved in ('embed_dim', 'depth', 'num_heads', 'ffn_ratio'):
        if reserved in backbone_kwargs:
            raise TypeError(f"{reserved} is determined by the split name")
    backbone = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=width,
        depth=depth,
        num_heads=width // HEAD_DIM,
        ffn_ratio=2.0,
        **backbone_kwargs,
    )
    return _StandaloneClassifier(backbone, width, num_classes)
