import torch

from ternarylayers.dinov3.layers.bitlayers import Conv2d, Linear
from ternarylayers.dinov3.models.vision_transformer import vit_femto


def test_bitlinear_accepts_torch_constructor_keywords() -> None:
    layer = Linear(4, 2, device="cpu", dtype=torch.float64)

    output = layer(torch.ones(3, 4, dtype=torch.float64))

    assert output.shape == (3, 2)
    assert output.dtype == torch.float64


def test_bitconv2d_accepts_torch_constructor_keywords() -> None:
    layer = Conv2d(3, 4, kernel_size=3, device="cpu", dtype=torch.float64)

    output = layer(torch.ones(2, 3, 5, 5, dtype=torch.float64))

    assert output.shape == (2, 4, 3, 3)
    assert output.dtype == torch.float64


def test_small_dinov3_model_runs() -> None:
    model = vit_femto(img_size=32, patch_size=8).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))

    assert output.shape == (1, 64)
