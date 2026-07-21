#!/usr/bin/env python3
"""Production-oriented ternary linear inference kernels in one file.

Computes

    y = (x @ weight.T) * scale + bias

where:
  * x is float16, bfloat16, or float32
  * weight is int8 with values restricted to {-1, 0, 1}
  * scale is optional and may be scalar or shape [out_features]
  * bias is optional and may be scalar or shape [out_features]

Design
------
* The persistent model weight always remains INT8.
* ``select`` kernel: uses x / -x / 0 selection and never casts a complete
  weight tensor. It is intended for GEMV and small-token batches.
* ``dot`` kernel: loads INT8 tiles and casts only those tiles inside the
  Triton program before ``tl.dot``. It is intended for larger flattened
  batches (prefill/training-like matrix shapes).
* Accumulation and scale/bias epilogues use float32; the output dtype matches x.
* CPU, missing-Triton, and autograd-required calls can use a PyTorch reference
  fallback.

This is an inference-oriented implementation. Benchmark and tune the dispatch
threshold/configurations on every target GPU family before deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as exc:  # Keep the reference backend importable on CPU.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False
    _TRITON_IMPORT_ERROR = exc


Backend = Literal["auto", "triton", "torch"]
KernelKind = Literal["auto", "select", "dot"]
FP32Precision = Literal["ieee", "tf32", "tf32x3"]

_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_PARAM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_VALID_BACKENDS = {"auto", "triton", "torch"}
_VALID_KERNELS = {"auto", "select", "dot"}
_VALID_FP32_PRECISIONS = {"ieee", "tf32", "tf32x3"}


def triton_is_available() -> bool:
    """Return True when Triton imported and a CUDA device is available."""

    return _TRITON_AVAILABLE and torch.cuda.is_available()


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
        raise RuntimeError(f"Triton is not available{detail}")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton backend requires a CUDA device")


def _shape_is_scalar_or_vector(tensor: torch.Tensor, length: int) -> bool:
    return tensor.ndim == 0 or tuple(tensor.shape) == (length,)


def _validate_scale_or_bias(
    tensor: Optional[torch.Tensor],
    *,
    name: str,
    out_features: int,
    device: torch.device,
) -> None:
    if tensor is None:
        return
    if tensor.device != device:
        raise ValueError(
            f"{name}.device must match weight.device; "
            f"got {tensor.device} and {device}"
        )
    if tensor.dtype not in _SUPPORTED_PARAM_DTYPES:
        raise TypeError(
            f"{name} must use float16, bfloat16, or float32; "
            f"got {tensor.dtype}"
        )
    if not _shape_is_scalar_or_vector(tensor, out_features):
        raise ValueError(
            f"{name} must be scalar or shape [{out_features}]; "
            f"got {tuple(tensor.shape)}"
        )


def validate_ternary_weight(
    weight: torch.Tensor,
    *,
    check_values: bool = True,
) -> None:
    """Validate shape, dtype, and optionally values of a ternary weight.

    ``check_values=True`` performs a device reduction and synchronizes when its
    result is read on the host. Do this once when loading a model, not in every
    forward call.
    """

    if weight.ndim != 2:
        raise ValueError(
            f"weight must have shape [out_features, in_features]; "
            f"got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.int8:
        raise TypeError(f"weight must be torch.int8; got {weight.dtype}")
    if check_values and weight.numel() > 0:
        invalid = torch.any((weight < -1) | (weight > 1))
        if bool(invalid.item()):
            raise ValueError("weight contains values outside {-1, 0, 1}")


def ternary_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Numerically stable PyTorch reference with float32 accumulation.

    This function materializes a float32 weight and is therefore a correctness,
    CPU, and autograd fallback—not the optimized inference path.
    """

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    validate_ternary_weight(weight, check_values=False)
    out_features, in_features = weight.shape
    if x.shape[-1] != in_features:
        raise ValueError(
            f"x.shape[-1] must equal {in_features}; got {x.shape[-1]}"
        )
    if x.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(
            f"x must use float16, bfloat16, or float32; got {x.dtype}"
        )
    if x.device != weight.device:
        raise ValueError(
            f"x.device must match weight.device; got {x.device} and {weight.device}"
        )
    _validate_scale_or_bias(
        scale,
        name="scale",
        out_features=out_features,
        device=weight.device,
    )
    _validate_scale_or_bias(
        bias,
        name="bias",
        out_features=out_features,
        device=weight.device,
    )

    y = F.linear(x.float(), weight.float(), bias=None)
    if scale is not None:
        y = y * scale.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype=x.dtype)


if _TRITON_AVAILABLE:
    _SELECT_CONFIGS = [
        triton.Config(
            {"BLOCK_N": 1, "BLOCK_K": 128},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 2, "BLOCK_K": 256},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 4, "BLOCK_K": 256},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 4, "BLOCK_K": 512},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 8, "BLOCK_K": 256},
            num_warps=8,
            num_stages=2,
        ),
    ]

    @triton.autotune(
        configs=_SELECT_CONFIGS,
        key=["N", "K"],
        cache_results=True,
    )
    @triton.jit
    def _ternary_select_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        HAS_SCALE: tl.constexpr,
        SCALE_IS_SCALAR: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BIAS_IS_SCALAR: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One flattened input row and BLOCK_N output rows per program."""

        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        mask_n = offsets_n < N

        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + offsets_k
            mask_k = k < K

            x = tl.load(
                x_ptr + pid_m * stride_xm + k * stride_xk,
                mask=mask_k,
                other=0.0,
            )
            weights = tl.load(
                weight_ptr
                + offsets_n[:, None] * stride_wn
                + k[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )

            # Keep the global-memory representation INT8. For each ternary
            # value, select x, -x, or zero without materializing a float weight.
            x_rows = x[None, :]
            selected = tl.where(
                weights == 0,
                0.0,
                tl.where(weights < 0, -x_rows, x_rows),
            )
            accumulator += tl.sum(selected, axis=1, dtype=tl.float32)

        if HAS_SCALE:
            if SCALE_IS_SCALAR:
                scale = tl.load(scale_ptr).to(tl.float32)
            else:
                scale = tl.load(
                    scale_ptr + offsets_n,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
            accumulator *= scale

        if HAS_BIAS:
            if BIAS_IS_SCALAR:
                bias = tl.load(bias_ptr).to(tl.float32)
            else:
                bias = tl.load(
                    bias_ptr + offsets_n,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
            accumulator += bias

        tl.store(
            output_ptr
            + pid_m * stride_ym
            + offsets_n * stride_yn,
            accumulator,
            mask=mask_n,
        )

    # A deliberately broad, but bounded, search space.  The previous list
    # stopped at BLOCK_M=64/BLOCK_N=128, which under-explored prefill shapes
    # such as M=128, N=2560.  Avoid enormous accumulator tiles such as
    # 128x256: they tend to spill registers and make autotuning expensive.
    _DOT_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        # BLOCK_K=64 can reduce loop overhead on some architectures/shapes.
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
    ]

    @triton.autotune(
        configs=_DOT_CONFIGS,
        key=["M", "N", "K"],
        cache_results=True,
    )
    @triton.jit
    def _ternary_dot_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        HAS_SCALE: tl.constexpr,
        SCALE_IS_SCALAR: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BIAS_IS_SCALAR: tl.constexpr,
        X_IS_FP16: tl.constexpr,
        X_IS_BF16: tl.constexpr,
        FP32_INPUT_PRECISION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Tiled matrix path; INT8 weights are cast only after tile loads."""

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + offsets_k
            x = tl.load(
                x_ptr
                + offsets_m[:, None] * stride_xm
                + k[None, :] * stride_xk,
                mask=(offsets_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            weights_i8 = tl.load(
                weight_ptr
                + offsets_n[None, :] * stride_wn
                + k[:, None] * stride_wk,
                mask=(offsets_n[None, :] < N) & (k[:, None] < K),
                other=0,
            )

            # This is a tile-local register/shared-memory conversion, not a
            # full weight-tensor conversion. Global weight traffic stays INT8.
            if X_IS_FP16:
                weights = weights_i8.to(tl.float16)
            elif X_IS_BF16:
                weights = weights_i8.to(tl.bfloat16)
            else:
                weights = weights_i8.to(tl.float32)

            accumulator = tl.dot(
                x,
                weights,
                accumulator,
                input_precision=FP32_INPUT_PRECISION,
                out_dtype=tl.float32,
            )

        if HAS_SCALE:
            if SCALE_IS_SCALAR:
                scale = tl.load(scale_ptr).to(tl.float32)
            else:
                scale = tl.load(
                    scale_ptr + offsets_n,
                    mask=offsets_n < N,
                    other=0.0,
                ).to(tl.float32)
            accumulator *= scale[None, :] if not SCALE_IS_SCALAR else scale

        if HAS_BIAS:
            if BIAS_IS_SCALAR:
                bias = tl.load(bias_ptr).to(tl.float32)
            else:
                bias = tl.load(
                    bias_ptr + offsets_n,
                    mask=offsets_n < N,
                    other=0.0,
                ).to(tl.float32)
            accumulator += bias[None, :] if not BIAS_IS_SCALAR else bias

        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_ym
            + offsets_n[None, :] * stride_yn,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


def _normalize_optional_parameter(
    tensor: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if tensor is None or tensor.ndim == 0 or tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _choose_kernel(
    *,
    requested: KernelKind,
    flattened_rows: int,
    select_max_rows: int,
) -> Literal["select", "dot"]:
    if requested == "select":
        return "select"
    if requested == "dot":
        return "dot"
    return "select" if flattened_rows <= select_max_rows else "dot"


def _launch_triton(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    *,
    kernel: KernelKind,
    select_max_rows: int,
    fp32_precision: FP32Precision,
) -> torch.Tensor:
    _require_triton()

    m, k = x_2d.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=x_2d.device, dtype=x_2d.dtype)
    if m == 0 or n == 0:
        return output

    selected_kernel = _choose_kernel(
        requested=kernel,
        flattened_rows=m,
        select_max_rows=select_max_rows,
    )

    # Triton kernel arguments cannot portably use None as a pointer on all
    # supported versions. Pass a valid dummy pointer and specialize it away.
    scale_arg = scale if scale is not None else output
    bias_arg = bias if bias is not None else output
    has_scale = scale is not None
    has_bias = bias is not None
    scale_is_scalar = bool(scale is not None and scale.ndim == 0)
    bias_is_scalar = bool(bias is not None and bias.ndim == 0)

    if selected_kernel == "select":
        grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
        _ternary_select_kernel[grid](
            x_2d,
            weight,
            scale_arg,
            bias_arg,
            output,
            M=m,
            N=n,
            K=k,
            stride_xm=x_2d.stride(0),
            stride_xk=x_2d.stride(1),
            stride_wn=weight.stride(0),
            stride_wk=weight.stride(1),
            stride_ym=output.stride(0),
            stride_yn=output.stride(1),
            HAS_SCALE=has_scale,
            SCALE_IS_SCALAR=scale_is_scalar,
            HAS_BIAS=has_bias,
            BIAS_IS_SCALAR=bias_is_scalar,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"])
            * triton.cdiv(n, meta["BLOCK_N"]),
        )
        _ternary_dot_kernel[grid](
            x_2d,
            weight,
            scale_arg,
            bias_arg,
            output,
            M=m,
            N=n,
            K=k,
            stride_xm=x_2d.stride(0),
            stride_xk=x_2d.stride(1),
            stride_wn=weight.stride(0),
            stride_wk=weight.stride(1),
            stride_ym=output.stride(0),
            stride_yn=output.stride(1),
            HAS_SCALE=has_scale,
            SCALE_IS_SCALAR=scale_is_scalar,
            HAS_BIAS=has_bias,
            BIAS_IS_SCALAR=bias_is_scalar,
            X_IS_FP16=x_2d.dtype == torch.float16,
            X_IS_BF16=x_2d.dtype == torch.bfloat16,
            FP32_INPUT_PRECISION=fp32_precision,
        )

    return output


def ternary_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    backend: Backend = "auto",
    kernel: KernelKind = "auto",
    select_max_rows: int = 4,
    fp32_precision: FP32Precision = "ieee",
    validate_values: bool = False,
    autograd_fallback: bool = True,
) -> torch.Tensor:
    """Apply a frozen ternary linear layer.

    Args:
        x: Tensor with shape ``[..., in_features]`` and dtype FP16/BF16/FP32.
        weight: INT8 tensor ``[out_features, in_features]`` containing -1/0/1.
        scale: Optional scalar or vector ``[out_features]``.
        bias: Optional scalar or vector ``[out_features]``.
        backend: ``auto``, ``triton``, or ``torch``.
        kernel: ``auto``, ``select``, or ``dot`` for the Triton backend.
        select_max_rows: In ``auto`` mode, use the select kernel when the
            flattened leading dimension is at most this value.
        fp32_precision: Triton ``tl.dot`` precision for FP32 matrix inputs.
            ``ieee`` is the accurate default; ``tf32`` is faster on supported
            NVIDIA GPUs; ``tf32x3`` is an intermediate option.
        validate_values: Check every weight value. This synchronizes once and
            should normally be done at model construction/load time instead.
        autograd_fallback: Use the PyTorch reference when gradients are needed.
            When False, raise instead of silently losing autograd connectivity.
    """

    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}; got {backend!r}")
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"kernel must be one of {_VALID_KERNELS}; got {kernel!r}")
    if fp32_precision not in _VALID_FP32_PRECISIONS:
        raise ValueError(
            "fp32_precision must be 'ieee', 'tf32', or 'tf32x3'; "
            f"got {fp32_precision!r}"
        )
    if select_max_rows < 1:
        raise ValueError("select_max_rows must be at least 1")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if x.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(
            f"x must use float16, bfloat16, or float32; got {x.dtype}"
        )

    validate_ternary_weight(weight, check_values=validate_values)
    out_features, in_features = weight.shape
    if x.shape[-1] != in_features:
        raise ValueError(
            f"x.shape[-1] must equal {in_features}; got {x.shape[-1]}"
        )
    if x.device != weight.device:
        raise ValueError(
            f"x.device must match weight.device; got {x.device} and {weight.device}"
        )
    _validate_scale_or_bias(
        scale,
        name="scale",
        out_features=out_features,
        device=weight.device,
    )
    _validate_scale_or_bias(
        bias,
        name="bias",
        out_features=out_features,
        device=weight.device,
    )

    needs_autograd = torch.is_grad_enabled() and (
        x.requires_grad
        or (scale is not None and scale.requires_grad)
        or (bias is not None and bias.requires_grad)
    )
    if needs_autograd:
        if autograd_fallback:
            return ternary_linear_reference(x, weight, scale, bias)
        raise RuntimeError(
            "Triton ternary kernels are inference-only. Disable gradients or "
            "set autograd_fallback=True."
        )

    use_triton = backend == "triton" or (
        backend == "auto"
        and x.is_cuda
        and weight.is_cuda
        and _TRITON_AVAILABLE
    )
    if not use_triton:
        if backend == "triton":
            _require_triton()
        return ternary_linear_reference(x, weight, scale, bias)

    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("Triton backend requires CUDA tensors")

    # Empty inner dimensions are uncommon for real linear layers and are
    # handled by the reference path to avoid compiling a zero-trip kernel.
    if in_features == 0:
        return ternary_linear_reference(x, weight, scale, bias)

    # Flatten arbitrary leading dimensions. reshape() preserves a view when
    # possible and creates the necessary contiguous copy otherwise.
    leading_shape = tuple(x.shape[:-1])
    x_2d = x.reshape(-1, in_features)
    if x_2d.stride(1) != 1:
        x_2d = x_2d.contiguous()

    scale_kernel = _normalize_optional_parameter(scale)
    bias_kernel = _normalize_optional_parameter(bias)
    output_2d = _launch_triton(
        x_2d,
        weight,
        scale_kernel,
        bias_kernel,
        kernel=kernel,
        select_max_rows=select_max_rows,
        fp32_precision=fp32_precision,
    )
    return output_2d.reshape(*leading_shape, out_features)


class TernaryLinear(nn.Module):
    """Frozen ternary ``nn.Module`` with an INT8 state-dict weight."""

    __constants__ = [
        "in_features",
        "out_features",
        "backend",
        "kernel",
        "select_max_rows",
        "fp32_precision",
        "autograd_fallback",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        *,
        backend: Backend = "auto",
        kernel: KernelKind = "auto",
        select_max_rows: int = 4,
        fp32_precision: FP32Precision = "ieee",
        validate_values: bool = True,
        autograd_fallback: bool = True,
    ) -> None:
        super().__init__()

        if in_features < 0 or out_features < 0:
            raise ValueError("in_features and out_features must be non-negative")
        validate_ternary_weight(weight, check_values=validate_values)
        if tuple(weight.shape) != (out_features, in_features):
            raise ValueError(
                "weight shape must equal "
                f"[{out_features}, {in_features}]; got {tuple(weight.shape)}"
            )
        _validate_scale_or_bias(
            scale,
            name="scale",
            out_features=out_features,
            device=weight.device,
        )
        _validate_scale_or_bias(
            bias,
            name="bias",
            out_features=out_features,
            device=weight.device,
        )
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"invalid backend: {backend!r}")
        if kernel not in _VALID_KERNELS:
            raise ValueError(f"invalid kernel: {kernel!r}")
        if fp32_precision not in _VALID_FP32_PRECISIONS:
            raise ValueError(f"invalid fp32_precision: {fp32_precision!r}")
        if select_max_rows < 1:
            raise ValueError("select_max_rows must be at least 1")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.backend = backend
        self.kernel = kernel
        self.select_max_rows = int(select_max_rows)
        self.fp32_precision = fp32_precision
        self.autograd_fallback = bool(autograd_fallback)

        # Buffers are the correct representation for frozen inference data.
        # Module.to(dtype=...) leaves the INT8 weight unchanged while casting
        # floating-point scale and bias.
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer(
            "scale",
            None if scale is None else scale.contiguous(),
        )
        self.register_buffer(
            "bias",
            None if bias is None else bias.contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ternary_linear(
            x,
            self.weight,
            self.scale,
            self.bias,
            backend=self.backend,
            kernel=self.kernel,
            select_max_rows=self.select_max_rows,
            fp32_precision=self.fp32_precision,
            validate_values=False,
            autograd_fallback=self.autograd_fallback,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"scale={self.scale is not None}, "
            f"backend={self.backend!r}, "
            f"kernel={self.kernel!r}, "
            f"select_max_rows={self.select_max_rows}, "
            f"fp32_precision={self.fp32_precision!r}"
        )


@dataclass(frozen=True)
class BenchmarkResult:
    dtype: str
    rows: int
    out_features: int
    in_features: int
    kernel: str
    triton_us: float
    torch_us: float
    speedup: float
    triton_tflops: float
    torch_tflops: float
    int8_footprint_gbps: float
    dense_footprint_gbps: float

    @property
    def int8_weight_gbps(self) -> float:
        """Backward-compatible alias for the old, misleading field name."""

        return self.int8_footprint_gbps


def _benchmark_metrics(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    dense_element_size: int,
    triton_us: float,
    torch_us: float,
) -> tuple[float, float, float, float]:
    """Return throughput and footprint/time metrics.

    ``int8_footprint_gbps`` is intentionally *not* called memory bandwidth.
    The tiled dot kernel may fetch a weight tile more than once, while caches
    may serve some reads.  Without hardware counters, footprint divided by
    latency is the honest metric.
    """

    operations = 2.0 * rows * out_features * in_features
    triton_seconds = triton_us * 1e-6
    torch_seconds = torch_us * 1e-6
    int8_footprint_bytes = out_features * in_features
    dense_footprint_bytes = int8_footprint_bytes * dense_element_size
    return (
        operations / triton_seconds / 1e12,
        operations / torch_seconds / 1e12,
        int8_footprint_bytes / triton_seconds / 1e9,
        dense_footprint_bytes / torch_seconds / 1e9,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "f16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype name: {name!r}") from exc


def _tolerances(dtype: torch.dtype, *, dot_path: bool) -> tuple[float, float]:
    if dtype == torch.float16:
        return (3e-2, 3e-2)
    if dtype == torch.bfloat16:
        return (8e-2, 8e-2)
    return ((2e-3, 2e-3) if dot_path else (5e-4, 5e-4))


def run_correctness_tests(
    *,
    device: Optional[str] = None,
    quick: bool = False,
) -> None:
    """Run reference tests and, when available, CUDA Triton tests."""

    requested_device = device or ("cuda" if triton_is_available() else "cpu")
    dev = torch.device(requested_device)
    torch.manual_seed(1234)

    dtypes: Sequence[torch.dtype]
    if dev.type == "cuda":
        dtypes = [torch.float16, torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes = [torch.float16, torch.bfloat16, torch.float32]
    else:
        dtypes = [torch.float32]

    shapes = [(1, 73, 129), (3, 257, 513)]
    if not quick:
        shapes += [(9, 255, 511), (33, 320, 768)]

    for dtype in dtypes:
        for rows, n, k in shapes:
            x = torch.randn((rows, k), device=dev, dtype=dtype)
            weight = torch.randint(-1, 2, (n, k), device=dev, dtype=torch.int8)
            scale = torch.rand((n,), device=dev, dtype=torch.float32) * 0.02 + 0.005
            bias = torch.randn((n,), device=dev, dtype=torch.float32) * 0.01
            expected = ternary_linear_reference(x, weight, scale, bias)

            if dev.type == "cuda" and _TRITON_AVAILABLE:
                kernels = ["select"]
                if rows > 1:
                    kernels.append("dot")
                for kernel_name in kernels:
                    actual = ternary_linear(
                        x,
                        weight,
                        scale,
                        bias,
                        backend="triton",
                        kernel=kernel_name,  # type: ignore[arg-type]
                        fp32_precision="ieee",
                    )
                    rtol, atol = _tolerances(
                        dtype,
                        dot_path=kernel_name == "dot",
                    )
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=rtol,
                        atol=atol,
                    )
            else:
                actual = ternary_linear(
                    x,
                    weight,
                    scale,
                    bias,
                    backend="auto",
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Scalar scale, missing bias, arbitrary leading dimensions, and state dict.
    dtype = dtypes[0]
    x = torch.randn((2, 3, 65), device=dev, dtype=dtype)
    weight = torch.randint(-1, 2, (17, 65), device=dev, dtype=torch.int8)
    scale = torch.tensor(0.01, device=dev, dtype=torch.float32)
    layer = TernaryLinear(
        65,
        17,
        weight,
        scale,
        bias=None,
        backend="auto",
        validate_values=True,
    )
    actual = layer(x)
    expected = ternary_linear_reference(x, weight, scale, None)
    rtol, atol = _tolerances(dtype, dot_path=True)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert layer.state_dict()["weight"].dtype == torch.int8

    print(f"Correctness tests passed on {dev} for {[str(d) for d in dtypes]}")


def benchmark(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    kernel: KernelKind = "auto",
    fp32_precision: FP32Precision = "ieee",
    select_max_rows: int = 4,
) -> BenchmarkResult:
    _require_triton()
    if rows < 1 or out_features < 1 or in_features < 1:
        raise ValueError("rows, out_features, and in_features must be positive")
    if select_max_rows < 1:
        raise ValueError("select_max_rows must be at least 1")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16")

    device = torch.device("cuda")
    x = torch.randn((rows, in_features), device=device, dtype=dtype)
    weight = torch.randint(
        -1,
        2,
        (out_features, in_features),
        device=device,
        dtype=torch.int8,
    )
    scale = torch.rand((out_features,), device=device, dtype=torch.float32) * 0.02 + 0.005
    bias = torch.randn((out_features,), device=device, dtype=torch.float32) * 0.01

    # Pre-materialized dense baseline: conversion is intentionally outside the
    # timed region, representing a model that permanently stores dense weights.
    dense_weight = (weight.to(dtype) * scale.to(dtype)[:, None]).contiguous()
    dense_bias = bias.to(dtype)

    def run_triton() -> torch.Tensor:
        return ternary_linear(
            x,
            weight,
            scale,
            bias,
            backend="triton",
            kernel=kernel,
            select_max_rows=select_max_rows,
            fp32_precision=fp32_precision,
        )

    def run_torch() -> torch.Tensor:
        return F.linear(x, dense_weight, dense_bias)

    expected = run_torch()
    actual = run_triton()
    torch.cuda.synchronize()
    chosen = _choose_kernel(
        requested=kernel,
        flattened_rows=rows,
        select_max_rows=select_max_rows,
    )
    rtol, atol = _tolerances(dtype, dot_path=chosen == "dot")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    triton_ms = triton.testing.do_bench(run_triton, warmup=100, rep=500)
    torch_ms = triton.testing.do_bench(run_torch, warmup=100, rep=500)
    triton_us = float(triton_ms * 1000.0)
    torch_us = float(torch_ms * 1000.0)
    (
        triton_tflops,
        torch_tflops,
        int8_footprint_gbps,
        dense_footprint_gbps,
    ) = _benchmark_metrics(
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        dense_element_size=dense_weight.element_size(),
        triton_us=triton_us,
        torch_us=torch_us,
    )

    return BenchmarkResult(
        dtype=str(dtype).replace("torch.", ""),
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        kernel=chosen,
        triton_us=triton_us,
        torch_us=torch_us,
        speedup=torch_us / triton_us,
        triton_tflops=triton_tflops,
        torch_tflops=torch_tflops,
        int8_footprint_gbps=int8_footprint_gbps,
        dense_footprint_gbps=dense_footprint_gbps,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="run correctness tests")
    test_parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    test_parser.add_argument("--quick", action="store_true")

    bench_parser = subparsers.add_parser("bench", help="benchmark one shape")
    bench_parser.add_argument("--rows", type=int, default=1)
    bench_parser.add_argument("--out-features", type=int, default=2560)
    bench_parser.add_argument("--in-features", type=int, default=2560)
    bench_parser.add_argument(
        "--dtype",
        choices=["f16", "bf16", "f32"],
        default="f16",
    )
    bench_parser.add_argument(
        "--kernel",
        choices=["auto", "select", "dot"],
        default="auto",
    )
    bench_parser.add_argument(
        "--fp32-precision",
        choices=["ieee", "tf32", "tf32x3"],
        default="ieee",
    )
    bench_parser.add_argument(
        "--select-max-rows",
        type=int,
        default=4,
        help="auto-dispatch threshold between select and dot kernels",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "test":
        run_correctness_tests(device=args.device, quick=args.quick)
        return 0

    result = benchmark(
        rows=args.rows,
        out_features=args.out_features,
        in_features=args.in_features,
        dtype=_dtype_from_name(args.dtype),
        kernel=args.kernel,
        fp32_precision=args.fp32_precision,
        select_max_rows=args.select_max_rows,
    )
    print(
        f"dtype={result.dtype} rows={result.rows} "
        f"N={result.out_features} K={result.in_features} "
        f"kernel={result.kernel}"
    )
    print(
        f"Triton ternary: {result.triton_us:.2f} us "
        f"({result.triton_tflops:.2f} TFLOP/s)"
    )
    print(
        f"PyTorch dense:  {result.torch_us:.2f} us "
        f"({result.torch_tflops:.2f} TFLOP/s)"
    )
    print(f"Speedup:        {result.speedup:.3f}x")
    print(
        "INT8 footprint/time: "
        f"{result.int8_footprint_gbps:.1f} GB/s"
    )
    print(
        "Dense footprint/time: "
        f"{result.dense_footprint_gbps:.1f} GB/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python ternary_linear_triton.py bench --rows 4  --out-features 2560 --in-features 2560 --dtype f16 --kernel select
# python ternary_linear_triton.py bench --rows 4  --out-features 2560 --in-features 2560 --dtype f16 --kernel dot

# python ternary_linear_triton.py bench --rows 8  --out-features 2560 --in-features 2560 --dtype f16 --kernel select
# python ternary_linear_triton.py bench --rows 8  --out-features 2560 --in-features 2560 --dtype f16 --kernel dot