# Matryoshka-BitViT: shared ternary ViTs with configurable width/depth prefixes.
# Install alongside vision_transformer.py in ternarylayers/dinov3/models/.
# This is an extension of the user-provided DINOv3-derived source; observe
# its existing DINOv3 license terms.

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternarylayers.dinov3.layers.rms_norm import MatryoshkaRMSNorm

from .vision_transformer import DinoVisionTransformer
from ternarylayers.dinov3.layers.ffn_layers import Mlp
from ternarylayers.dinov3.layers.block import (
    MatryoshkaLayerNorm,
    SelfMatryoshkaAttentionBlock,
)


class DinoVisionMatryoshka(DinoVisionTransformer):
    """A configurable shared ViT with width/depth prefix subnetworks.

    ``embed_dim``, ``depth``, ``num_heads`` and ``ffn_ratio`` describe the largest
    network. ``splits`` maps names to (width, depth) pairs for training/export.
    Any width divisible by the fixed head dimension can run, even if unnamed.
    Omitted width/depth select the full network (or a named width's depth).
    Call ``init_weights()`` before training, as for DinoVisionTransformer.
    Currently supports MLP FFNs and LayerNorm/RMSNorm only.
    """

    def __init__(
        self, *, patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
        num_heads: int = 12, ffn_ratio: float = 4.0,
        splits: Optional[Mapping[str, tuple[int, int]]] = None, **kwargs,
    ):
        for name, value in (("embed_dim", embed_dim), ("depth", depth), ("num_heads", num_heads)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if embed_dim % num_heads or (embed_dim // num_heads) % 4:
            raise ValueError("embed_dim / num_heads must be an integer multiple of 4 for RoPE")
        if not isinstance(ffn_ratio, (int, float)) or not 0 < ffn_ratio < float("inf") or int(embed_dim * ffn_ratio) < 1:
            raise ValueError("ffn_ratio must be finite and produce a positive hidden dimension")
        if kwargs.get("ffn_layer", "mlp") != "mlp":
            raise ValueError("This implementation currently supports ffn_layer='mlp' only")
        if kwargs.get("norm_layer", "layernorm") not in ("layernorm", "layernormbf16", "rmsnorm"):
            raise ValueError(f"Unsupported elastic norm {kwargs['norm_layer']!r}")
        self.head_dim = embed_dim // num_heads
        self.splits = dict(splits) if splits is not None else {"full": (embed_dim, depth)}
        if not self.splits:
            raise ValueError("splits must contain at least one named subnetwork")
        # Validate before allocating a potentially large model.
        for name, split in self.splits.items():
            if not isinstance(name, str) or not name or not isinstance(split, (tuple, list)) or len(split) != 2:
                raise ValueError("splits must map nonempty names to (width, depth) pairs")
            width, sub_depth = split
            if type(width) is not int or not self.head_dim <= width <= embed_dim or width % self.head_dim:
                raise ValueError(f"split {name!r}: width must be a multiple of {self.head_dim} up to {embed_dim}")
            if type(sub_depth) is not int or not 1 <= sub_depth <= depth:
                raise ValueError(f"split {name!r}: depth must be an integer from 1 to {depth}")
        self.splits = {name: tuple(split) for name, split in self.splits.items()}
        self._quant_dims = {}
        for name, default in (("sattn_qkv_quant_dim", 0), ("sattn_fc1_quant_dim", 0), ("sattn_fc2_quant_dim", 1)):
            value = kwargs.get(name, default)
            if not isinstance(value, int) and len(value) != depth:
                raise ValueError(f"{name} must have one entry per block ({depth})")
            self._quant_dims[name] = deepcopy(value)
        super().__init__(
            patch_size=patch_size, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, ffn_ratio=ffn_ratio, **kwargs,
        )
        # The parent builds the ordinary, unmodified model. Replace its blocks
        # without changing the names in their state_dict (blocks.i.*).
        norm_factory = (
            partial(MatryoshkaRMSNorm, eps=1e-5) if self.norm_layer == "rmsnorm"
            else partial(MatryoshkaLayerNorm, eps=(1e-5 if self.norm_layer == "layernormbf16" else 1e-6))
        )
        def per_block(v, i):
            return v if isinstance(v, int) else v[i]

        qdim = kwargs.get("sattn_qkv_quant_dim", 0)
        fc1dim = kwargs.get("sattn_fc1_quant_dim", 0)
        fc2dim = kwargs.get("sattn_fc2_quant_dim", 1)
        # Release the original stack before allocating its elastic replacement.
        del self.blocks
        self.blocks = nn.ModuleList([
            SelfMatryoshkaAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
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
        self.norm = norm_factory(embed_dim).to(device=self.device)
        if self.cls_norm is not None:
            self.cls_norm = norm_factory(embed_dim).to(device=self.device)
        if self.local_cls_norm is not None:
            self.local_cls_norm = norm_factory(embed_dim).to(device=self.device)
        # Parameter initialisation follows the parent contract: call
        # model.init_weights() just as for the standard DinoVisionTransformer.

    def _resolve(self, width: Optional[int], depth: Optional[int], out_dim: Optional[int]):
        width = self.embed_dim if width is None else width
        if type(width) is not int or not self.head_dim <= width <= self.embed_dim or width % self.head_dim:
            raise ValueError(f"width must be a multiple of {self.head_dim} between {self.head_dim} and {self.embed_dim}")
        if depth is None:
            depths = {d for w, d in self.splits.values() if w == width}
            if len(depths) > 1 and width != self.embed_dim:
                raise ValueError("Multiple splits use this width; specify depth explicitly")
            depth = self.depth if width == self.embed_dim or not depths else next(iter(depths))
        if type(depth) is not int or not 1 <= depth <= self.depth:
            raise ValueError(f"depth must be an integer from 1 to {self.depth}")
        out_dim = width if out_dim is None else out_dim
        if type(out_dim) is not int or not 1 <= out_dim <= width:
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
        self, x: Tensor, masks: Optional[Tensor] = None, *, width: Optional[int] = None
    ):
        width, _, _ = self._resolve(width, self.depth, None)
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
        width: Optional[int] = None,
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
        width: Optional[int] = None,
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
        width: Optional[int] = None,
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
        """Copy weights, architecture, quantizer settings and train/eval mode."""
        return deepcopy(self)

    def export_subnetwork(self, name: str) -> DinoVisionTransformer:
        """Return a compact, *ordinary* DINOv3 model with prefix-copied weights.

        Export first materialises floating-point/learned weights. The real
        BitLinear ternary freeze/export must be applied separately, with
        the width-specific quantization scale recomputed AFTER slicing.
        """
        if name not in self.splits:
            raise ValueError(f"name must be one of {tuple(self.splits)}, got {name!r}")
        width, depth = self.splits[name]
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
            num_heads=width // self.head_dim,
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
            no_conv=self.no_conv,
            **{key: value if isinstance(value, int) else value[:depth]
               for key, value in self._quant_dims.items()},
        ).to(device=self.cls_token.device, dtype=self.cls_token.dtype)

        # The elastic MLP scales the already rounded full hidden dimension.
        # For fractional ratios this can differ from int(width * ffn_ratio).
        def per_block_quant(value, index):
            return value if isinstance(value, int) else value[index]

        hidden_dim = max(1, self.blocks[0].mlp.fc1.out_features * width // self.embed_dim)
        target.ffn_ratio = hidden_dim / width
        for i, block in enumerate(target.blocks):
            if block.mlp.fc1.out_features != hidden_dim:
                block.mlp = Mlp(
                    in_features=width, hidden_features=hidden_dim, bias=self.ffn_bias,
                    fc1_quant_dim=per_block_quant(self._quant_dims["sattn_fc1_quant_dim"], i),
                    fc2_quant_dim=per_block_quant(self._quant_dims["sattn_fc2_quant_dim"], i),
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


# Width/depth follow the ordinary ViT presets. Prefixes share their parent's
# FFN ratio and head dimension. Pico also shares the ordinary FFN ratio.
_DEFAULT_SPLITS = {
    "large": (1024, 24),
    "base": (768, 12),
    "small": (384, 12),
    "tiny": (256, 6),
    "nano": (192, 6),
    "micro": (128, 4),
    "pico": (64, 6),
    "femto": (64, 3),
}


def _downstream_splits(width: int, depth: int):
    return {name: (w, d) for name, (w, d) in _DEFAULT_SPLITS.items()
            if w <= width and d <= depth}

def vit_matryoshka_femto(patch_size: int = 16, **kwargs):
    """64-wide, 3-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(64, 3))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=64, depth=3,
                                num_heads=1, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_pico(patch_size: int = 16, **kwargs):
    """64-wide, 6-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(64, 6))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=64, depth=6,
                                num_heads=1, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_micro(patch_size: int = 8, **kwargs):
    """128-wide, 4-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(128, 4))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=128, depth=4,
                                num_heads=2, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_nano(patch_size: int = 8, **kwargs):
    """192-wide, 6-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(192, 6))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=192, depth=6,
                                num_heads=3, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_tiny(patch_size: int = 8, **kwargs):
    """256-wide, 6-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(256, 6))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=256, depth=6,
                                num_heads=4, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_small(patch_size: int = 16, **kwargs):
    """384-wide, 12-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(384, 12))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=384, depth=12,
                                num_heads=6, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_base(patch_size: int = 16, **kwargs):
    """768-wide, 12-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(768, 12))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=768, depth=12,
                                num_heads=12, ffn_ratio=4.0, **kwargs)

def vit_matryoshka_large(patch_size: int = 16, **kwargs):
    """1024-wide, 24-block network with all compatible smaller presets."""
    kwargs.setdefault("splits", _downstream_splits(1024, 24))
    return DinoVisionMatryoshka(patch_size=patch_size, embed_dim=1024, depth=24,
                                num_heads=16, ffn_ratio=4.0, **kwargs)


class MatryoshkaClassifier(nn.Module):
    """Shared classification head for sandwich training at any backbone size.

    Head rows represent classes; head columns are prefix-sliced so each
    width gets a genuine w->num_classes matrix multiplication.
    """

    def __init__(self, num_classes: int = 10, *, backbone: Optional[DinoVisionMatryoshka] = None, **kwargs):
        super().__init__()
        self.backbone = backbone if backbone is not None else DinoVisionMatryoshka(**kwargs)
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes).to(
            device=self.backbone.cls_token.device, dtype=self.backbone.cls_token.dtype
        )

    @property
    def splits(self):
        return self.backbone.splits

    def forward(self, images: Tensor, *, name: Optional[str] = None) -> Tensor:
        if name is None:
            width, depth = self.backbone.embed_dim, self.backbone.depth
        elif name not in self.splits:
            raise ValueError(f"Unknown subnet {name!r}")
        else:
            width, depth = self.splits[name]
        cls = self.backbone(images, width=width, depth=depth)
        return F.linear(cls, self.classifier.weight[:, :width], self.classifier.bias)

    def export_subnetwork(self, name: str) -> nn.Module:
        """Return a standalone classifier with the ordinary reduced ViT."""
        if name not in self.splits:
            raise ValueError(f"Unknown subnet {name!r}")
        width, _ = self.splits[name]
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
    name: str, *, splits: Mapping[str, tuple[int, int]], head_dim: int,
    num_classes: int = 10, patch_size: int = 16, ffn_ratio: float = 4.0, **backbone_kwargs
) -> nn.Module:
    """Rebuild an exported split without allocating the shared model.

    Pass the source's splits and head_dim, the exported FFN ratio, and the
    original patch/image settings before loading the exported state_dict.
    """
    if name not in splits:
        raise ValueError(f"name must be one of {tuple(splits)}, got {name!r}")
    width, depth = splits[name]
    for reserved in ('embed_dim', 'depth', 'num_heads', 'ffn_ratio'):
        if reserved in backbone_kwargs:
            raise TypeError(f"{reserved} is determined by the split name")
    backbone = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=width,
        depth=depth,
        num_heads=width // head_dim,
        ffn_ratio=ffn_ratio,
        **backbone_kwargs,
    )
    return _StandaloneClassifier(backbone, width, num_classes)
