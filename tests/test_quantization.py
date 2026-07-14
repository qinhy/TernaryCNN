import pytest
import torch

from ternarylayers import Bit


def test_bit1p58_weight_uses_per_output_scale() -> None:
    weight = torch.tensor([[0.2, 1.0], [-0.5, 0.1]], requires_grad=True)

    quantized = Bit.functional.bit1p58_weight(weight, dim=0, scale_op="mean")

    expected = torch.tensor([[0.0, 0.6], [-0.3, 0.0]])
    torch.testing.assert_close(quantized, expected)

    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_bit1p58_weight_rejects_unknown_scale_operation() -> None:
    with pytest.raises(ValueError, match="op must be"):
        Bit.functional.bit1p58_weight(torch.ones(2, 2), scale_op="maximum")


def test_dynamic_same_padding_preserves_ceil_output_shape() -> None:
    layer = Bit.Conv2d(3, 5, kernel_size=3, stride=2, padding="same")

    output = layer(torch.randn(2, 3, 7, 9))

    assert output.shape == (2, 5, 4, 5)
