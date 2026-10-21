import pytest
import torch


from ternarylayers.dinov3.layers.attention import (
    SelfAttention,
    SelfMatryoshkaAttention,
)
from ternarylayers.dinov3.layers.bitlayers import Conv2d, Linear
from ternarylayers.dinov3.layers.ffn_layers import MatryoshkaMlp, Mlp
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


def test_matryoshka_attention_supports_multiple_widths() -> None:
    attention = SelfMatryoshkaAttention(dim=128, num_heads=2, qkv_bias=True).eval()

    with torch.no_grad():
        for width in (64, 128):  # Keep head_dim=64; use 1 or 2 heads.
            x = torch.randn(2, 5, width)
            y = attention(x)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()


def test_matryoshka_full_width_matches_original() -> None:
    attention = SelfMatryoshkaAttention(dim=128, num_heads=2, qkv_bias=True).eval()
    original = SelfAttention(dim=128, num_heads=2, qkv_bias=True).eval()
    original.load_state_dict(attention.state_dict())
    x = torch.randn(2, 5, 128)

    with torch.no_grad():
        torch.testing.assert_close(attention(x), original(x))


def test_matryoshka_qkv_slices_q_k_v_separately() -> None:
    attention = SelfMatryoshkaAttention(dim=128, num_heads=2, qkv_bias=True)
    x = torch.randn(2, 5, 64)

    actual = attention.linear_qkv(x)
    weight = attention.qkv.weight.reshape(3, 128, 128)[:, :64, :64]
    weight = weight.reshape(3 * 64, 64)
    bias = attention.qkv.bias.reshape(3, 128)[:, :64].reshape(-1)
    expected = torch.nn.functional.linear(x, attention.qkv.w_q(weight), bias)

    assert actual.shape == (2, 5, 192)
    torch.testing.assert_close(actual, expected)


def test_matryoshka_attention_forward_list_and_rope() -> None:
    attention = SelfMatryoshkaAttention(dim=128, num_heads=2).eval()
    x1 = torch.randn(2, 5, 64)
    x2 = torch.randn(1, 5, 64)
    # Four patch positions, plus one CLS token; head axis is full-size (2).
    sin = torch.zeros(1, 2, 4, 64)
    cos = torch.ones(1, 2, 4, 64)

    with torch.no_grad():
        outputs = attention.forward_list([x1, x2], rope_list=[(sin, cos)] * 2)
    assert [tuple(o.shape) for o in outputs] == [(2, 5, 64), (1, 5, 64)]


def test_matryoshka_attention_backward_and_invalid_width() -> None:
    attention = SelfMatryoshkaAttention(dim=128, num_heads=2, qkv_bias=True)
    x = torch.randn(2, 5, 64, requires_grad=True)

    attention(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert attention.qkv.weight.grad is not None
    assert attention.proj.weight.grad is not None
    assert attention.proj.weight.grad[:64, :64].abs().sum() > 0
    assert torch.count_nonzero(attention.proj.weight.grad[64:, :]) == 0
    assert torch.count_nonzero(attention.proj.weight.grad[:64, 64:]) == 0

    with pytest.raises(ValueError, match="multiple"):
        attention(torch.randn(2, 5, 96))


def test_matryoshka_mlp_widths() -> None:
    mlp = MatryoshkaMlp(in_features=128, hidden_features=512).eval()
    with torch.no_grad():
        for width in (32, 64, 128):
            x = torch.randn(2, 5, width)
            y = mlp(x)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()


def test_matryoshka_mlp_full_width_equals_mlp() -> None:
    nested = MatryoshkaMlp(in_features=128, hidden_features=512).eval()
    original = Mlp(in_features=128, hidden_features=512).eval()
    original.load_state_dict(nested.state_dict())
    with torch.no_grad():
        x = torch.randn(2, 5, 128)
        torch.testing.assert_close(nested(x), original(x))


def test_matryoshka_mlp_really_slices_and_quantizes() -> None:
    mlp = MatryoshkaMlp(in_features=128, hidden_features=512).eval()
    x = torch.randn(2, 5, 64)

    hidden = torch.nn.functional.linear(x, mlp.fc1.w_q(mlp.fc1.weight[:256, :64]),
                      mlp.fc1.bias[:256] if mlp.fc1.bias is not None else None)
    hidden = mlp.act(hidden)
    expected = torch.nn.functional.linear(hidden, mlp.fc2.w_q(mlp.fc2.weight[:64, :256]),
                        mlp.fc2.bias[:64] if mlp.fc2.bias is not None else None)
    torch.testing.assert_close(mlp(x), expected)


def test_matryoshka_mlp_backward_only_active_weights() -> None:
    mlp = MatryoshkaMlp(in_features=128, hidden_features=512)
    x = torch.randn(2, 5, 64, requires_grad=True)
    mlp(x).square().mean().backward()

    assert x.grad is not None
    assert mlp.fc1.weight.grad is not None
    assert mlp.fc2.weight.grad is not None
    assert mlp.fc1.weight.grad[:256, :64].abs().sum() > 0
    assert mlp.fc2.weight.grad[:64, :256].abs().sum() > 0
    assert torch.count_nonzero(mlp.fc1.weight.grad[256:, :]) == 0
    assert torch.count_nonzero(mlp.fc1.weight.grad[:, 64:]) == 0
    assert torch.count_nonzero(mlp.fc2.weight.grad[64:, :]) == 0
    assert torch.count_nonzero(mlp.fc2.weight.grad[:, 256:]) == 0


def test_matryoshka_mlp_list_and_invalid_widths() -> None:
    mlp = MatryoshkaMlp(in_features=128, hidden_features=512).eval()
    with torch.no_grad():
        ys = mlp.forward_list([torch.randn(2, 5, 32), torch.randn(1, 5, 64)])
    assert [tuple(y.shape) for y in ys] == [(2, 5, 32), (1, 5, 64)]

    with pytest.raises(ValueError, match="Input width"):
        mlp(torch.randn(1, 5, 129))
