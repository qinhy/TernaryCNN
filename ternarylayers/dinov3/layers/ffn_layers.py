# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional

import torch.nn.functional as F
from ternarylayers.dinov3.layers.bitlayers import Linear as BitLinear
from torch import Tensor, nn

from ternarylayers.dinov3.utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
        fc1_quant_dim=0,
        fc2_quant_dim=1,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = BitLinear(in_features, hidden_features, bias=bias, dim=fc1_quant_dim).to(device=device)
        self.act = act_layer()
        self.fc2 = BitLinear(hidden_features, out_features, bias=bias, dim=fc2_quant_dim).to(device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = BitLinear(in_features, swiglu_hidden_features, bias=bias).to(device=device)
        self.w2 = BitLinear(in_features, swiglu_hidden_features, bias=bias).to(device=device)
        self.w3 = BitLinear(swiglu_hidden_features, out_features, bias=bias).to(device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class MatryoshkaMlp(Mlp):
    """MLP with shared weights and an input-dependent active width.

    The input's last dimension selects the active residual width. Both the
    expansion layer and the contraction layer use weight prefixes. Quantization
    happens *after* slicing, using each BitLinear's own w_q quantizer.

    Example with a full 1024 -> 4096 -> 1024 MLP:
        x.shape[-1] == 256  ->  256 -> 1024 -> 256
    """

    @staticmethod
    def _linear_prefix(layer, x: Tensor, out_features: int) -> Tensor:
        """Evaluate only the requested top-left submatrix of a BitLinear."""
        if x.shape[-1] == layer.in_features and out_features == layer.out_features:
            return layer(x)  # Identical to the original BitLinear call.

        weight = layer.weight[:out_features, : x.shape[-1]]
        bias = layer.bias[:out_features] if layer.bias is not None else None
        return F.linear(x, layer.w_q(weight), bias)

    def forward(self, x: Tensor) -> Tensor:
        width = x.shape[-1]
        full_width = self.fc1.in_features
        if not 1 <= width <= full_width:
            raise ValueError(f"Input width must be in [1, {full_width}], got {width}")

        if width == full_width:
            return super().forward(x)  # Keep full-width Mlp behavior unchanged.

        # Preserve the original FFN expansion/output ratios.
        hidden = max(1, self.fc1.out_features * width // full_width)
        out_width = max(1, self.fc2.out_features * width // full_width)

        x = self._linear_prefix(self.fc1, x, hidden)
        x = self.act(x)
        x = self.drop(x)
        x = self._linear_prefix(self.fc2, x, out_width)
        x = self.drop(x)
        return x

    def forward_list(self, x_list: list[Tensor]) -> list[Tensor]:
        """Handle multiple images/crops, including different active widths."""
        return [self.forward(x) for x in x_list]
