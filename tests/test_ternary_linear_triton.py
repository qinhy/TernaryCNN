"""Pytest suite for :mod:`ternary_linear_triton`.

Run the CPU/reference suite:

    pytest -q test_ternary_linear_triton.py

Run only CUDA/Triton tests:

    pytest -q test_ternary_linear_triton.py -k cuda

The CUDA tests skip automatically when CUDA, Triton, or BF16 support is absent.
"""

from __future__ import annotations

import io
from typing import Optional

import pytest
import torch

import ternarylayers.ternary_linear_triton as ternary


CPU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CUDA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _make_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
    leading_shape: tuple[int, ...] = (3,),
    out_features: int = 17,
    in_features: int = 29,
    scale_kind: str = "vector",
    bias_kind: str = "vector",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(20260721)

    x = torch.randn(
        (*leading_shape, in_features),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randint(
        -1,
        2,
        (out_features, in_features),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )

    def make_parameter(kind: str, *, bias: bool) -> Optional[torch.Tensor]:
        if kind == "none":
            return None
        if kind == "scalar":
            value = -0.015 if bias else 0.02
            return torch.tensor(value, device=device, dtype=torch.float32)
        if kind == "vector":
            if bias:
                return torch.randn(
                    out_features,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                ) * 0.01
            return (
                torch.rand(
                    out_features,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                * 0.02
                + 0.005
            )
        raise AssertionError(f"unknown parameter kind: {kind}")

    return (
        x,
        weight,
        make_parameter(scale_kind, bias=False),
        make_parameter(bias_kind, bias=True),
    )


def _cuda_ready() -> bool:
    return ternary.triton_is_available()


def _cuda_dtype_supported(dtype: torch.dtype) -> bool:
    if not _cuda_ready():
        return False
    if dtype == torch.bfloat16:
        return torch.cuda.is_bf16_supported()
    return True


def _tolerances(dtype: torch.dtype, kernel: str) -> tuple[float, float]:
    if dtype == torch.float16:
        return 3e-2, 3e-2
    if dtype == torch.bfloat16:
        return 8e-2, 8e-2
    if kernel == "dot":
        return 2e-3, 2e-3
    return 5e-4, 5e-4


# ---------------------------------------------------------------------------
# Pure validation and dispatch tests
# ---------------------------------------------------------------------------


def test_validate_ternary_weight_accepts_valid_weight() -> None:
    weight = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.int8)
    ternary.validate_ternary_weight(weight)


@pytest.mark.parametrize(
    ("weight", "exception"),
    [
        (torch.zeros(2, 3, 4, dtype=torch.int8), ValueError),
        (torch.zeros(2, 3, dtype=torch.float32), TypeError),
        (torch.tensor([[0, 2]], dtype=torch.int8), ValueError),
        (torch.tensor([[-2, 0]], dtype=torch.int8), ValueError),
    ],
)
def test_validate_ternary_weight_rejects_invalid_weight(
    weight: torch.Tensor,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ternary.validate_ternary_weight(weight)


def test_validate_ternary_weight_can_skip_value_scan() -> None:
    weight = torch.tensor([[127, -128]], dtype=torch.int8)
    ternary.validate_ternary_weight(weight, check_values=False)


@pytest.mark.parametrize(
    ("requested", "rows", "threshold", "expected"),
    [
        ("select", 100, 4, "select"),
        ("dot", 1, 4, "dot"),
        ("auto", 1, 4, "select"),
        ("auto", 4, 4, "select"),
        ("auto", 5, 4, "dot"),
    ],
)
def test_choose_kernel(
    requested: str,
    rows: int,
    threshold: int,
    expected: str,
) -> None:
    assert (
        ternary._choose_kernel(  # noqa: SLF001 - intentional unit test
            requested=requested,
            flattened_rows=rows,
            select_max_rows=threshold,
        )
        == expected
    )


@pytest.mark.parametrize("name", ["f16", "fp16", "bf16", "f32", "fp32"])
def test_dtype_from_name_accepts_aliases(name: str) -> None:
    assert ternary._dtype_from_name(name) in CPU_DTYPES  # noqa: SLF001


def test_dtype_from_name_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        ternary._dtype_from_name("int8")  # noqa: SLF001


# ---------------------------------------------------------------------------
# CPU/reference tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", CPU_DTYPES)
@pytest.mark.parametrize(
    ("scale_kind", "bias_kind"),
    [
        ("none", "none"),
        ("scalar", "none"),
        ("vector", "scalar"),
        ("vector", "vector"),
    ],
)
def test_reference_matches_explicit_float32_formula(
    dtype: torch.dtype,
    scale_kind: str,
    bias_kind: str,
) -> None:
    x, weight, scale, bias = _make_case(
        device=torch.device("cpu"),
        dtype=dtype,
        leading_shape=(2, 3),
        scale_kind=scale_kind,
        bias_kind=bias_kind,
    )

    actual = ternary.ternary_linear_reference(x, weight, scale, bias)
    expected = torch.nn.functional.linear(x.float(), weight.float())
    if scale is not None:
        expected = expected * scale.float()
    if bias is not None:
        expected = expected + bias.float()
    expected = expected.to(dtype)

    assert actual.shape == (*x.shape[:-1], weight.shape[0])
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_auto_backend_on_cpu_matches_reference_exactly(dtype: torch.dtype) -> None:
    x, weight, scale, bias = _make_case(
        device=torch.device("cpu"),
        dtype=dtype,
        leading_shape=(2, 2),
    )
    expected = ternary.ternary_linear_reference(x, weight, scale, bias)
    actual = ternary.ternary_linear(
        x,
        weight,
        scale,
        bias,
        backend="auto",
        kernel="auto",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_noncontiguous_cpu_input() -> None:
    rows, out_features, in_features = 4, 13, 19
    base = torch.randn(rows, in_features * 2, dtype=torch.float32)
    x = base[:, ::2]
    assert not x.is_contiguous()
    weight = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)

    actual = ternary.ternary_linear(x, weight, backend="auto")
    expected = ternary.ternary_linear_reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 11),
        (2, 0, 11),
    ],
)
def test_empty_leading_dimensions(shape: tuple[int, ...]) -> None:
    x = torch.empty(shape, dtype=torch.float32)
    weight = torch.randint(-1, 2, (7, 11), dtype=torch.int8)
    actual = ternary.ternary_linear(x, weight, backend="auto")
    assert actual.shape == (*shape[:-1], 7)
    assert actual.numel() == 0


def test_zero_in_features_reference_path() -> None:
    x = torch.empty((2, 0), dtype=torch.float32)
    weight = torch.empty((5, 0), dtype=torch.int8)
    bias = torch.arange(5, dtype=torch.float32)
    actual = ternary.ternary_linear(x, weight, bias=bias, backend="auto")
    expected = bias.expand(2, 5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_autograd_fallback_propagates_gradients() -> None:
    x = torch.randn(3, 7, dtype=torch.float32, requires_grad=True)
    weight = torch.randint(-1, 2, (5, 7), dtype=torch.int8)
    scale = torch.randn(5, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(5, dtype=torch.float32, requires_grad=True)

    output = ternary.ternary_linear(
        x,
        weight,
        scale,
        bias,
        backend="auto",
        autograd_fallback=True,
    )
    output.square().mean().backward()

    assert x.grad is not None
    assert scale.grad is not None
    assert bias.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(scale.grad).all()
    assert torch.isfinite(bias.grad).all()


def test_disabling_autograd_fallback_raises() -> None:
    x = torch.randn(2, 7, dtype=torch.float32, requires_grad=True)
    weight = torch.randint(-1, 2, (5, 7), dtype=torch.int8)
    with pytest.raises(RuntimeError, match="inference-only"):
        ternary.ternary_linear(
            x,
            weight,
            backend="auto",
            autograd_fallback=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"backend": "invalid"}, ValueError),
        ({"kernel": "invalid"}, ValueError),
        ({"fp32_precision": "invalid"}, ValueError),
        ({"select_max_rows": 0}, ValueError),
    ],
)
def test_public_argument_validation(
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    x = torch.randn(2, 7)
    weight = torch.randint(-1, 2, (5, 7), dtype=torch.int8)
    with pytest.raises(exception):
        ternary.ternary_linear(x, weight, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("x", "weight", "scale", "bias", "exception"),
    [
        (
            torch.tensor(1.0),
            torch.zeros(5, 7, dtype=torch.int8),
            None,
            None,
            ValueError,
        ),
        (
            torch.zeros(2, 6),
            torch.zeros(5, 7, dtype=torch.int8),
            None,
            None,
            ValueError,
        ),
        (
            torch.zeros(2, 7, dtype=torch.float64),
            torch.zeros(5, 7, dtype=torch.int8),
            None,
            None,
            TypeError,
        ),
        (
            torch.zeros(2, 7),
            torch.zeros(5, 7, dtype=torch.int8),
            torch.zeros(2),
            None,
            ValueError,
        ),
        (
            torch.zeros(2, 7),
            torch.zeros(5, 7, dtype=torch.int8),
            torch.zeros(5, dtype=torch.float64),
            None,
            TypeError,
        ),
        (
            torch.zeros(2, 7),
            torch.zeros(5, 7, dtype=torch.int8),
            None,
            torch.zeros(2),
            ValueError,
        ),
    ],
)
def test_tensor_validation(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ternary.ternary_linear(x, weight, scale, bias)


def test_explicit_triton_backend_rejects_cpu_tensors() -> None:
    x = torch.randn(2, 7)
    weight = torch.randint(-1, 2, (5, 7), dtype=torch.int8)
    with pytest.raises(RuntimeError):
        ternary.ternary_linear(x, weight, backend="triton")


# ---------------------------------------------------------------------------
# nn.Module and serialization tests
# ---------------------------------------------------------------------------


def test_module_forward_repr_and_state_dict() -> None:
    x, weight, scale, bias = _make_case(
        device=torch.device("cpu"),
        dtype=torch.float32,
        leading_shape=(2, 3),
    )
    layer = ternary.TernaryLinear(
        weight.shape[1],
        weight.shape[0],
        weight,
        scale,
        bias,
        backend="torch",
        kernel="select",
    )

    actual = layer(x)
    expected = ternary.ternary_linear_reference(x, weight, scale, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    state = layer.state_dict()
    assert state["weight"].dtype == torch.int8
    assert state["scale"].dtype == torch.float32
    assert state["bias"].dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in layer.parameters())

    representation = repr(layer)
    assert "in_features=29" in representation
    assert "out_features=17" in representation
    assert "backend='torch'" in representation


def test_module_to_dtype_preserves_int8_weight() -> None:
    _, weight, scale, bias = _make_case(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    layer = ternary.TernaryLinear(29, 17, weight, scale, bias, backend="torch")
    layer = layer.to(dtype=torch.float16)

    assert layer.weight.dtype == torch.int8
    assert layer.scale is not None and layer.scale.dtype == torch.float16
    assert layer.bias is not None and layer.bias.dtype == torch.float16


def test_module_state_dict_round_trip() -> None:
    x, weight, scale, bias = _make_case(
        device=torch.device("cpu"),
        dtype=torch.float32,
        leading_shape=(4,),
    )
    first = ternary.TernaryLinear(29, 17, weight, scale, bias, backend="torch")
    second = ternary.TernaryLinear(
        29,
        17,
        torch.zeros_like(weight),
        torch.ones_like(scale),
        torch.zeros_like(bias),
        backend="torch",
        validate_values=False,
    )

    # Exercise a real torch.save/torch.load round trip without touching disk.
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    try:
        restored_state = torch.load(buffer, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        restored_state = torch.load(buffer)
    second.load_state_dict(restored_state)

    assert second.weight.dtype == torch.int8
    torch.testing.assert_close(first(x), second(x), rtol=0, atol=0)


def test_module_copies_noncontiguous_weight_to_contiguous_buffer() -> None:
    base = torch.randint(-1, 2, (17, 58), dtype=torch.int8)
    weight = base[:, ::2]
    assert not weight.is_contiguous()
    layer = ternary.TernaryLinear(
        29,
        17,
        weight,
        backend="torch",
    )
    assert layer.weight.is_contiguous()
    torch.testing.assert_close(layer.weight, weight)


# ---------------------------------------------------------------------------
# CUDA/Triton integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", CUDA_DTYPES)
@pytest.mark.parametrize("kernel", ("select", "dot"))
def test_cuda_kernels_match_reference(dtype: torch.dtype, kernel: str) -> None:
    if not _cuda_ready():
        pytest.skip("CUDA and Triton are required")
    if not _cuda_dtype_supported(dtype):
        pytest.skip(f"CUDA device does not support {dtype}")

    rows = 2 if kernel == "select" else 9
    x, weight, scale, bias = _make_case(
        device=torch.device("cuda"),
        dtype=dtype,
        leading_shape=(rows,),
        out_features=37,
        in_features=65,
    )
    weight_pointer = weight.data_ptr()

    expected = ternary.ternary_linear_reference(x, weight, scale, bias)
    actual = ternary.ternary_linear(
        x,
        weight,
        scale,
        bias,
        backend="triton",
        kernel=kernel,  # type: ignore[arg-type]
        fp32_precision="ieee",
    )
    torch.cuda.synchronize()

    rtol, atol = _tolerances(dtype, kernel)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert actual.dtype == dtype
    assert weight.dtype == torch.int8
    assert weight.data_ptr() == weight_pointer


@pytest.mark.parametrize(
    ("scale_kind", "bias_kind"),
    [
        ("none", "none"),
        ("scalar", "none"),
        ("none", "scalar"),
        ("scalar", "scalar"),
        ("vector", "vector"),
    ],
)
def test_cuda_epilogue_variants(scale_kind: str, bias_kind: str) -> None:
    if not _cuda_ready():
        pytest.skip("CUDA and Triton are required")

    x, weight, scale, bias = _make_case(
        device=torch.device("cuda"),
        dtype=torch.float16,
        leading_shape=(3,),
        out_features=31,
        in_features=47,
        scale_kind=scale_kind,
        bias_kind=bias_kind,
    )
    expected = ternary.ternary_linear_reference(x, weight, scale, bias)
    actual = ternary.ternary_linear(
        x,
        weight,
        scale,
        bias,
        backend="triton",
        kernel="select",
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_cuda_arbitrary_leading_dimensions_and_auto_dispatch() -> None:
    if not _cuda_ready():
        pytest.skip("CUDA and Triton are required")

    x, weight, scale, bias = _make_case(
        device=torch.device("cuda"),
        dtype=torch.float16,
        leading_shape=(2, 3),
        out_features=35,
        in_features=67,
    )
    expected = ternary.ternary_linear_reference(x, weight, scale, bias)
    actual = ternary.ternary_linear(
        x,
        weight,
        scale,
        bias,
        backend="auto",
        kernel="auto",
        select_max_rows=4,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert actual.shape == (2, 3, 35)


def test_cuda_noncontiguous_input() -> None:
    if not _cuda_ready():
        pytest.skip("CUDA and Triton are required")

    rows, out_features, in_features = 3, 23, 41
    base = torch.randn(rows, in_features * 2, device="cuda", dtype=torch.float16)
    x = base[:, ::2]
    assert not x.is_contiguous()
    weight = torch.randint(
        -1,
        2,
        (out_features, in_features),
        device="cuda",
        dtype=torch.int8,
    )

    expected = ternary.ternary_linear_reference(x, weight)
    actual = ternary.ternary_linear(
        x,
        weight,
        backend="triton",
        kernel="select",
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_cuda_module_keeps_int8_weight_after_forward() -> None:
    if not _cuda_ready():
        pytest.skip("CUDA and Triton are required")

    x, weight, scale, bias = _make_case(
        device=torch.device("cuda"),
        dtype=torch.float16,
        leading_shape=(1,),
        out_features=19,
        in_features=33,
    )
    layer = ternary.TernaryLinear(
        33,
        19,
        weight,
        scale,
        bias,
        backend="triton",
        kernel="select",
    )
    pointer_before = layer.weight.data_ptr()

    with torch.inference_mode():
        actual = layer(x)
    expected = ternary.ternary_linear_reference(x, weight, scale, bias)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    assert layer.weight.dtype == torch.int8
    assert layer.weight.data_ptr() == pointer_before
    assert layer.state_dict()["weight"].dtype == torch.int8


def test_cuda_fp32_tf32_mode_executes() -> None:
    if not _cuda_ready():
        pytest.skip("CUDA and Triton are required")

    x, weight, scale, bias = _make_case(
        device=torch.device("cuda"),
        dtype=torch.float32,
        leading_shape=(8,),
        out_features=32,
        in_features=64,
    )
    output = ternary.ternary_linear(
        x,
        weight,
        scale,
        bias,
        backend="triton",
        kernel="dot",
        fp32_precision="tf32",
    )
    assert output.shape == (8, 32)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
