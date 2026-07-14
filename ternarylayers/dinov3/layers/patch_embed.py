# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, Tuple, Union

from ternarylayers.dinov3.layers.ternarylayers import Conv2d as BitConv2d, Linear
from torch import Tensor, nn


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = BitConv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C Hp Wp
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HpxWp C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B Hp Wp C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))

class PatchEmbedNoConv(nn.Module):
    """
    2D image to patch embedding without Conv2d: (B,C,H,W) -> (B,N,D)
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
        bias=True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        patch_dim = in_chans * patch_HW[0] * patch_HW[1]

        # If you have a quantized/bit linear layer, replace nn.Linear with BitLinear
        self.proj = Linear(patch_dim, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        pH, pW = self.patch_size

        assert C == self.in_chans, f"Expected {self.in_chans} channels, got {C}"
        assert H % pH == 0, f"Input height {H} is not divisible by patch height {pH}"
        assert W % pW == 0, f"Input width {W} is not divisible by patch width {pW}"

        Hp, Wp = H // pH, W // pW

        # Patchify without conv:
        # (B, C, H, W)
        # -> (B, C, Hp, pH, Wp, pW)
        # -> (B, Hp, Wp, C, pH, pW)
        # -> (B, Hp*Wp, C*pH*pW)
        x = x.reshape(B, C, Hp, pH, Wp, pW)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, Hp * Wp, C * pH * pW)

        # Linear projection per patch
        x = self.proj(x)  # (B, N, D)
        x = self.norm(x)

        if not self.flatten_embedding:
            x = x.reshape(B, Hp, Wp, self.embed_dim)  # (B, Hp, Wp, D)

        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        patch_area = self.patch_size[0] * self.patch_size[1]

        # Linear: N * (Cin * patch_area) * D
        flops = Ho * Wo * self.embed_dim * self.in_chans * patch_area

        # Only count norm if it's real norm, not Identity
        if not isinstance(self.norm, nn.Identity):
            flops += Ho * Wo * self.embed_dim

        return flops

    def reset_parameters(self):
        # Fan-in for linear = in_chans * patch_area
        fan_in = self.in_chans * self.patch_size[0] * self.patch_size[1]
        k = 1 / fan_in
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))