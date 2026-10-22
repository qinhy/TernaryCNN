# Examples

## MNIST with a ternary ResNet-18

The MNIST example uses the library's `ResNetModels.R18` builder with a one-channel
stem and narrower stages (`16, 32, 64, 128`) for fast local training. Its default
run uses 12,000 training images and 2,000 test images.

```bash
uv run python examples/train_mnist_resnet18.py
```

Run a no-download end-to-end check with generated images:

```bash
uv run python examples/train_mnist_resnet18.py --smoke-test
```

Use the complete MNIST dataset and save the trained weights:

```bash
uv run python examples/train_mnist_resnet18.py \
  --train-samples 0 --test-samples 0 --epochs 5 --save mnist-resnet18.pt
```

Pass `--device cpu`, `--device cuda`, or `--device mps` to override automatic
device selection. Use `--help` for all training options.

## Configurable Matryoshka ViTs

`DinoVisionMatryoshka` accepts `embed_dim`, `depth`, `num_heads`, `ffn_ratio`,
and a `splits` mapping from names to `(width, depth)`.
`MatryoshkaClassifier` adds a shared classification head to any configuration.

```python
from ternarylayers.dinov3.models.vision_transformer_matryoshka import (
    MatryoshkaClassifier, vit_matryoshka_large,
)

backbone = vit_matryoshka_large(img_size=224, patch_size=16, no_conv=True)
backbone.init_weights()
model = MatryoshkaClassifier(num_classes=1000, backbone=backbone)

# Train each selected split against the labels and accumulate their losses.
logits = model(images, name="large")
small_logits = model(images, name="small")
compact = model.export_subnetwork("small")
```

| Factory | Largest width / depth / heads | Default prefixes |
| --- | --- | --- |
| `vit_matryoshka_femto` | 64 / 3 / 1 | None |
| `vit_matryoshka_pico` | 64 / 6 / 1 | Femto 64 / 3 |
| `vit_matryoshka_micro` | 128 / 4 / 2 | Femto 64 / 3 |
| `vit_matryoshka_nano` | 192 / 6 / 3 | Micro 128 / 4, Pico 64 / 6, Femto 64 / 3 |
| `vit_matryoshka_tiny` | 256 / 6 / 4 | Nano, Micro, Pico, Femto |
| `vit_matryoshka_small` | 384 / 12 / 6 | Tiny 256 / 6, Nano 192 / 6, Micro 128 / 4, Pico 64 / 6, Femto 64 / 3 |
| `vit_matryoshka_base` | 768 / 12 / 12 | Small, Tiny, Nano, Micro, Pico, Femto |
| `vit_matryoshka_large` | 1024 / 24 / 16 | Base, Small, Tiny, Nano, Micro, Pico, Femto |

For a custom hierarchy, configure the largest network directly:

```python
from ternarylayers.dinov3.models.vision_transformer_matryoshka import DinoVisionMatryoshka

backbone = DinoVisionMatryoshka(
    embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0,
    splits={"large": (1024, 24), "medium": (512, 12), "micro": (128, 4)},
    no_conv=True,
)
backbone.init_weights()
features = backbone(images, width=512, depth=12)  # [batch, 512]
```

The attention head dimension stays fixed: `embed_dim // num_heads`. Every split
width must be a multiple of it, and each split depth must fit the largest network.
The factories use head dimension 64. Pico uses width 64, depth 6, one head,
and FFN ratio 4.0 in both `vit_pico` and the default Matryoshka hierarchy.
With patch size 16 and three input channels, its backbone has **349,376 parameters**
(excluding a classification head). Patch size 8 gives 312,512 parameters at
the same depth; a custom depth of 7 gives 362,496 parameters at patch size 8.
The former standalone Pico configuration (96 / 4 / 3 / 2) requires an explicit
`DinoVisionTransformer` configuration; its checkpoints do not fit the new preset. Tiny and Nano use the width/depth pairs from
the ordinary repo presets. Micro, Nano, and Femto prefixes share the parent's FFN ratio
of 4 rather than their ordinary presets' ratio of 2.
All Matryoshka factories use FFN ratio 4.0 and head dimension 64, and support
MLP FFNs with LayerNorm or RMSNorm. Tiny/Nano/Micro default to patch size 8;
Pico/Femto/Small/Base/Large default to 16. Override `patch_size` explicitly when
sharing checkpoints across factories. A prefix must fit both the parent width
and depth: Micro excludes the six-block Pico; Femto contains only itself.
An omitted width/depth runs the full network. A reduced width uses its configured
depth when unambiguous. Pico and Femto both use width 64, so select them through
`model(images, name="pico")` or `model(images, name="femto")` on the classifier or pass `width=64, depth=6`
for Pico and `width=64, depth=3` for Femto to the backbone. `out_dim` only truncates output features after the blocks.

`no_conv=True` slices the patch projector before quantization. Convolutional
patch embeddings use a full projection followed by a channel prefix; their
standalone exports can differ because quantization scales are recomputed.
Floating-point rounding, especially with BF16 RoPE, can also cause small
shared-versus-exported output differences. Use `pos_embed_rope_dtype="fp32"`
when checking close numerical agreement.

Exported submodels contain learned weights; apply ternary freezing after extraction.
To rebuild a custom export, pass the source's `splits`, `head_dim`, `ffn_ratio`,
and patch/image options to `build_standalone_classifier` before loading its state.
For fractional FFN ratios, use the exported backbone's `ffn_ratio`, which accounts
for the hidden dimension rounded when selecting a prefix.
Matching ordinary ternary `DinoVisionTransformer` checkpoints can be loaded with
`load_state_dict`; pretrained weights are not downloaded by these factories.

The MNIST example defaults to Small and accepts all eight presets through `--arch`
(`femto`, `pico`, `micro`, `nano`, `tiny`, `small`, `base`, `large`):

```bash
uv run python examples/train_mnist_matryoshka.py --arch large --batch-size 8 \
  --save-dir checkpoints/mnist-matryoshka-large
```

It trains all configured splits and selects the best checkpoint using the largest
split's validation accuracy. Larger capacity alone does not establish better
accuracy or generalization; assess that on the target dataset.
