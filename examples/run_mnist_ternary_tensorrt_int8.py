"""Run a trained ternary SimpleCNN through NVIDIA TensorRT INT8.

This script consumes the checkpoint produced by train_mnist_simplecnn.py:

    simplecnn_mnist_ternary_int8.pt

The checkpoint stores frozen ternary weights as int8 values in {-1, 0, +1}
plus one floating weight scale per output channel.  This script reconstructs
an ordinary PyTorch model with the exact floating weights

    W_real[c] = W_ternary[c] * weight_scale[c]

and then uses NVIDIA ModelOpt + Torch-TensorRT to build an explicit-Q/DQ
INT8 TensorRT engine.

Why reconstruct first?
----------------------
TensorRT sees a standard quantized Conv/Linear graph and can use its highly
optimized INT8 kernels/Tensor Cores.  ModelOpt's default INT8 configuration
uses per-tensor activation quantization and per-channel weight quantization.
For our ternary weights, each reconstructed channel contains only
{-s_w[c], 0, +s_w[c]}, so INT8 per-channel quantization naturally preserves
the three-value structure (typically mapping it close to {-127, 0, +127}).

The checkpoint's custom fixed-point M,n metadata is intentionally NOT used in
this CUDA backend.  M,n remains useful for the portable/reference integer
runtime; TensorRT owns requantization and kernel fusion on NVIDIA GPUs.

Typical use on Windows:

    uv run examples/run_mnist_ternary_tensorrt_int8.py \
        --checkpoint simplecnn_mnist_ternary_int8.pt \
        --batch-size 128

Required packages in addition to your normal PyTorch environment:

    nvidia-modelopt
    torch-tensorrt
    tensorrt

Use Torch/Torch-TensorRT/TensorRT builds compatible with your installed CUDA.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TRTCompatibleSimpleCNN(nn.Module):
    """Ordinary PyTorch version of the trained ternary SimpleCNN.

    The training network was:

        Bit.Conv2d(1, channels, 3, padding=1)
        ReLU
        MaxPool2d(2)
        Flatten
        Bit.Linear(channels * 14 * 14, 10)

    Here we reconstruct its frozen ternary floating semantics using regular
    nn.Conv2d / nn.Linear so ModelOpt and TensorRT can quantize it normally.
    """

    def __init__(self, channels: int, conv_bias: bool = True, fc_bias: bool = True) -> None:
        super().__init__()
        self.channels = int(channels)
        self.conv = nn.Conv2d(
            1,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.fc = nn.Linear(
            self.channels * 14 * 14,
            10,
            bias=fc_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("simplecnn_mnist_ternary_int8.pt"),
        help="Checkpoint produced by train_mnist_simplecnn.py.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "ternarycnn",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Static TensorRT batch size and benchmark batch size.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=32,
        help="MNIST training batches used by ModelOpt calibration; 0 means all.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="CUDA benchmark warmup iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="CUDA benchmark timed iterations.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=4,
        choices=(0, 1, 2, 3, 4, 5),
        help="TensorRT builder optimization level.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow PyTorch fallback subgraphs instead of requiring full TensorRT compilation.",
    )
    parser.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Torch-TensorRT CUDA graphs for lower static-shape launch overhead.",
    )
    parser.add_argument(
        "--trt-fp16",
        action="store_true",
        help="Also compile a TensorRT FP16 engine for a fair TensorRT-vs-TensorRT baseline.",
    )
    parser.add_argument(
        "--save-trt",
        type=Path,
        default=Path("simplecnn_mnist_int8_trt.ep"),
        help="Where to save the compiled INT8 Torch-TensorRT ExportedProgram.",
    )
    parser.add_argument(
        "--no-save-trt",
        action="store_true",
        help="Do not serialize the compiled TensorRT program.",
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip MNIST accuracy/agreement evaluation and only build/benchmark.",
    )
    return parser.parse_args()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this TensorRT script")


def import_nvidia_stack():
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.utils import export_torch_mode
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA ModelOpt is missing or incomplete. Install a compatible 'nvidia-modelopt' package."
        ) from exc

    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError(
            "Torch-TensorRT is missing. Install a torch-tensorrt build matching your "
            "PyTorch/CUDA environment, plus TensorRT."
        ) from exc

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python package is missing. Install a TensorRT package matching your CUDA major version."
        ) from exc

    return mtq, export_torch_mode, torch_tensorrt, trt


def _unwrap_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"], checkpoint

    # Also accept a raw state_dict for convenience.
    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint, {}

    raise ValueError("checkpoint does not contain a usable 'model' state_dict")


def _require_tensor(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"required checkpoint tensor is missing: {key!r}")
    value = state[key]
    if not torch.is_tensor(value):
        raise TypeError(f"checkpoint entry {key!r} is not a tensor")
    return value


@torch.no_grad()
def load_ternary_checkpoint_as_float_model(path: Path) -> tuple[TRTCompatibleSimpleCNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state, metadata = _unwrap_state_dict(checkpoint)

    conv_wq = _require_tensor(state, "conv.weight").detach().cpu()
    conv_sw = _require_tensor(state, "conv.scale").detach().cpu().float()
    fc_wq = _require_tensor(state, "fc.weight").detach().cpu()
    fc_sw = _require_tensor(state, "fc.scale").detach().cpu().float()

    if conv_wq.ndim != 4:
        raise ValueError(f"expected conv.weight to be 4-D, got {tuple(conv_wq.shape)}")
    if fc_wq.ndim != 2:
        raise ValueError(f"expected fc.weight to be 2-D, got {tuple(fc_wq.shape)}")

    channels = int(metadata.get("channels", conv_wq.shape[0]))
    if channels != conv_wq.shape[0]:
        raise ValueError(
            f"checkpoint channels={channels} but conv.weight has {conv_wq.shape[0]} outputs"
        )

    # Verify the algorithm-level weights really are ternary.
    conv_values = set(torch.unique(conv_wq).tolist())
    fc_values = set(torch.unique(fc_wq).tolist())
    allowed = {-1, 0, 1}
    if not conv_values.issubset(allowed):
        raise ValueError(f"conv.weight is not ternary: {sorted(conv_values)}")
    if not fc_values.issubset(allowed):
        raise ValueError(f"fc.weight is not ternary: {sorted(fc_values)}")

    conv_bias_tensor = state.get("conv.bias")
    fc_bias_tensor = state.get("fc.bias")
    model = TRTCompatibleSimpleCNN(
        channels=channels,
        conv_bias=conv_bias_tensor is not None,
        fc_bias=fc_bias_tensor is not None,
    )

    # Conv2dInfer stores scale as [out, 1, 1].
    conv_scale = conv_sw.reshape(channels, 1, 1, 1)
    conv_real = conv_wq.float() * conv_scale
    model.conv.weight.copy_(conv_real)
    if conv_bias_tensor is not None:
        model.conv.bias.copy_(conv_bias_tensor.detach().cpu().float())

    # LinearInfer stores scale as [out].
    fc_scale = fc_sw.reshape(fc_wq.shape[0], 1)
    fc_real = fc_wq.float() * fc_scale
    model.fc.weight.copy_(fc_real)
    if fc_bias_tensor is not None:
        model.fc.bias.copy_(fc_bias_tensor.detach().cpu().float())

    model.eval()

    print("[checkpoint]")
    print(f"path                  = {path}")
    print(f"channels              = {channels}")
    print(f"conv stored dtype     = {conv_wq.dtype}")
    print(f"conv ternary values   = {sorted(conv_values)}")
    print(f"fc stored dtype       = {fc_wq.dtype}")
    print(f"fc ternary values     = {sorted(fc_values)}")

    quant_meta = metadata.get("quantization") if isinstance(metadata, dict) else None
    if isinstance(quant_meta, dict):
        print(f"saved quant scheme    = {quant_meta.get('scheme', 'unknown')}")
        if "input_scale" in quant_meta:
            print(f"saved input_scale     = {quant_meta['input_scale']}")
        if "hidden_scale" in quant_meta:
            print(f"saved hidden_scale    = {quant_meta['hidden_scale']}")
        if "output_scale" in quant_meta:
            print(f"saved output_scale    = {quant_meta['output_scale']}")
        print("saved M,n backend     = portable/reference only; TensorRT will use Q/DQ")

    return model, metadata


def create_mnist_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    # Calibration does not need shuffle for this demonstration; reproducibility
    # is more useful here.
    train_loader = DataLoader(train_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, test_loader


@torch.no_grad()
def evaluate_fixed_batch_cuda(
    model: nn.Module,
    loader: DataLoader,
    static_batch_size: int,
    input_dtype: torch.dtype = torch.float32,
    reference_model: nn.Module | None = None,
) -> tuple[float, float | None]:
    """Evaluate a model compiled for one static batch size.

    The final MNIST batch is padded to static_batch_size and sliced back to the
    real item count.  This keeps one fixed TensorRT engine shape and also works
    for ordinary PyTorch models.
    """
    correct = 0
    agree = 0
    seen = 0

    for images, targets in loader:
        n = targets.numel()
        if n > static_batch_size:
            raise ValueError("loader batch is larger than static TensorRT batch")

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if n < static_batch_size:
            pad = torch.zeros(
                static_batch_size - n,
                *images.shape[1:],
                dtype=images.dtype,
                device=images.device,
            )
            images = torch.cat((images, pad), dim=0)

        model_input = images.to(dtype=input_dtype)
        logits = model(model_input)[:n]
        pred = logits.argmax(dim=1)

        correct += (pred == targets).sum().item()

        if reference_model is not None:
            ref_logits = reference_model(images.float())[:n]
            ref_pred = ref_logits.argmax(dim=1)
            agree += (pred == ref_pred).sum().item()

        seen += n

    accuracy = correct / seen
    agreement = (agree / seen) if reference_model is not None else None
    return accuracy, agreement


def make_modelopt_calibration_loop(loader: DataLoader, max_batches: int):
    @torch.no_grad()
    def calibration_loop(model: nn.Module) -> None:
        model.eval()
        for batch_index, (images, _) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            images = images.cuda(non_blocking=True)
            model(images)

    return calibration_loop


def enable_trt_cuda_graphs_if_available(torch_tensorrt: Any, enabled: bool) -> bool:
    if not enabled:
        return False

    try:
        runtime = torch_tensorrt.runtime
        setter = getattr(runtime, "set_cudagraphs_mode", None)
        if setter is None:
            print("[cuda graphs] API not available in this Torch-TensorRT build; continuing without it")
            return False
        setter(True)
        print("[cuda graphs] enabled")
        return True
    except Exception as exc:
        print(f"[cuda graphs] could not enable ({exc}); continuing without it")
        return False


@torch.no_grad()
def benchmark_cuda(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")

    for _ in range(warmup):
        model(x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        model(x)
    end.record()

    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    ms = total_ms / iterations
    throughput = x.shape[0] * 1000.0 / ms

    print(
        f"{name:<24} {ms:>10.4f} ms/batch   "
        f"{throughput:>12,.0f} images/s"
    )
    return ms


def print_versions(torch_tensorrt: Any, trt: Any) -> None:
    print("\n[environment]")
    print(f"torch                 = {torch.__version__}")
    print(f"torch CUDA runtime    = {torch.version.cuda}")
    print(f"GPU                   = {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"compute capability    = {torch.cuda.get_device_capability()}")
    print(f"torch_tensorrt        = {getattr(torch_tensorrt, '__version__', 'unknown')}")
    print(f"TensorRT              = {getattr(trt, '__version__', 'unknown')}")


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    require_cuda()
    mtq, export_torch_mode, torch_tensorrt, trt = import_nvidia_stack()

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    print_versions(torch_tensorrt, trt)

    # ------------------------------------------------------------------
    # 1. Reconstruct the exact frozen ternary floating model from checkpoint.
    # ------------------------------------------------------------------
    source_cpu, _metadata = load_ternary_checkpoint_as_float_model(args.checkpoint)
    source_fp32 = source_cpu.cuda().eval()

    train_loader, test_loader = create_mnist_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if not args.skip_accuracy:
        fp32_acc, _ = evaluate_fixed_batch_cuda(
            source_fp32,
            test_loader,
            static_batch_size=args.batch_size,
            input_dtype=torch.float32,
        )
        print(f"\n[reconstructed frozen ternary] accuracy = {fp32_acc:.2%}")

    # ------------------------------------------------------------------
    # 2. ModelOpt INT8 PTQ.
    # ------------------------------------------------------------------
    quant_model = deepcopy(source_cpu).cuda().eval()
    calibration_loop = make_modelopt_calibration_loop(
        train_loader,
        max_batches=args.calibration_batches,
    )

    print("\n[ModelOpt INT8 calibration]")
    print(f"calibration batches   = {args.calibration_batches if args.calibration_batches > 0 else 'all'}")
    started = time.perf_counter()
    mtq.quantize(
        quant_model,
        mtq.INT8_DEFAULT_CFG,
        forward_loop=calibration_loop,
    )
    torch.cuda.synchronize()
    print(f"completed in          = {time.perf_counter() - started:.2f}s")

    if not args.skip_accuracy:
        modelopt_acc, modelopt_agreement = evaluate_fixed_batch_cuda(
            quant_model,
            test_loader,
            static_batch_size=args.batch_size,
            input_dtype=torch.float32,
            reference_model=source_fp32,
        )
        print(f"ModelOpt Q/DQ accuracy = {modelopt_acc:.2%}")
        print(f"agreement with source  = {modelopt_agreement:.2%}")

    # ------------------------------------------------------------------
    # 3. Compile the quantized graph to TensorRT INT8.
    # ------------------------------------------------------------------
    example = torch.zeros(
        args.batch_size,
        1,
        28,
        28,
        dtype=torch.float32,
        device="cuda",
    )

    print("\n[TensorRT INT8 compile]")
    print(f"static input shape     = {tuple(example.shape)}")
    print(f"optimization level     = {args.optimization_level}")
    print(f"require full compile   = {not args.allow_fallback}")

    started = time.perf_counter()
    # ModelOpt inserts custom fake-quant/QDQ modules.  During torch.export,
    # export_torch_mode() rewrites those modules into export-friendly Q/DQ ops.
    # Without this context, torch.export may lift a FakeTensor such as
    # conv.input_quantizer.lifted_tensor_0 into the constants table and fail.
    with export_torch_mode():
        trt_int8 = torch_tensorrt.compile(
            quant_model,
            ir="dynamo",
            arg_inputs=[example],
            use_explicit_typing=True,
            min_block_size=1,
            require_full_compilation=not args.allow_fallback,
            optimization_level=args.optimization_level,
        )
    torch.cuda.synchronize()
    print(f"compile time           = {time.perf_counter() - started:.2f}s")

    cuda_graphs_enabled = enable_trt_cuda_graphs_if_available(
        torch_tensorrt,
        args.cuda_graphs,
    )

    # First call also validates that the compiled graph can execute.
    _ = trt_int8(example)
    torch.cuda.synchronize()

    if not args.skip_accuracy:
        trt_acc, trt_agreement = evaluate_fixed_batch_cuda(
            trt_int8,
            test_loader,
            static_batch_size=args.batch_size,
            input_dtype=torch.float32,
            reference_model=source_fp32,
        )
        print("\n[TensorRT INT8 accuracy]")
        print(f"accuracy               = {trt_acc:.2%}")
        print(f"agreement with source  = {trt_agreement:.2%}")

    # ------------------------------------------------------------------
    # 4. Optional TensorRT FP16 baseline.
    # ------------------------------------------------------------------
    trt_fp16 = None
    example_half = example.half()
    if args.trt_fp16:
        print("\n[TensorRT FP16 compile]")
        fp16_source = deepcopy(source_cpu).half().cuda().eval()
        started = time.perf_counter()
        trt_fp16 = torch_tensorrt.compile(
            fp16_source,
            ir="dynamo",
            arg_inputs=[example_half],
            use_explicit_typing=True,
            min_block_size=1,
            require_full_compilation=not args.allow_fallback,
            optimization_level=args.optimization_level,
        )
        torch.cuda.synchronize()
        print(f"compile time           = {time.perf_counter() - started:.2f}s")
        _ = trt_fp16(example_half)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 5. Serialize the compiled INT8 program.
    # ------------------------------------------------------------------
    if not args.no_save_trt:
        args.save_trt.parent.mkdir(parents=True, exist_ok=True)
        # The engine is static-shape, so the same example input is sufficient
        # for serialization/retrace metadata.
        torch_tensorrt.save(
            trt_int8,
            str(args.save_trt),
            arg_inputs=[example],
        )
        print(f"\n[saved] {args.save_trt}")

    # ------------------------------------------------------------------
    # 6. CUDA benchmark.
    # ------------------------------------------------------------------
    # Use non-trivial random data so the benchmark resembles real execution.
    bench_x = torch.rand_like(example)
    fp16_model = deepcopy(source_cpu).half().cuda().eval()

    print("\n[CUDA benchmark]")
    print(f"batch size             = {args.batch_size}")
    print(f"warmup                 = {args.warmup}")
    print(f"iterations             = {args.iterations}")
    print(f"TRT CUDA graphs        = {cuda_graphs_enabled}")
    print("-" * 72)

    fp32_ms = benchmark_cuda(
        "PyTorch FP32",
        source_fp32,
        bench_x,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fp16_ms = benchmark_cuda(
        "PyTorch FP16",
        fp16_model,
        bench_x.half(),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    int8_ms = benchmark_cuda(
        "TensorRT INT8",
        trt_int8,
        bench_x,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    trt_fp16_ms = None
    if trt_fp16 is not None:
        trt_fp16_ms = benchmark_cuda(
            "TensorRT FP16",
            trt_fp16,
            bench_x.half(),
            warmup=args.warmup,
            iterations=args.iterations,
        )

    print("-" * 72)
    print(f"INT8 speedup vs PyTorch FP32 = {fp32_ms / int8_ms:.3f}x")
    print(f"INT8 speedup vs PyTorch FP16 = {fp16_ms / int8_ms:.3f}x")
    if trt_fp16_ms is not None:
        print(f"INT8 speedup vs TensorRT FP16 = {trt_fp16_ms / int8_ms:.3f}x")

    # Restore default runtime behavior for callers embedding this script.
    if cuda_graphs_enabled:
        try:
            torch_tensorrt.runtime.set_cudagraphs_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
