# Matryoshka-BitViT — Project Handoff and Paper Draft (v0.1)

**Date:** 2026-10-21  
**Status:** Architecture prototypes exist; full integration, real-backend testing, distillation experiments, and quantitative results remain to be done.  
**Working title:** *Matryoshka-BitViT: Elastic Width and Depth in Ternary Vision Transformers via DINOv3 Distillation*

## 1. One-paragraph restart summary

We are developing a **decoder-free Matryoshka DINOv3 Vision Transformer** with a single set of shared **BitNet 1.58 / ternary** weights. One trained student should run at several widths (channel and attention-head prefixes), depths (initial Transformer-block prefixes), and returned embedding dimensions. We will distill this student from a **frozen, normal DINOv3 teacher**, supervising its **CLS and patch-token representations without PCA**. Narrow subnetworks must perform genuinely narrower quantized matrix multiplications—not full-width multiplications followed by output truncation. The design uses ordinary end-to-end gradients, without `.detach()` at layer boundaries or added decoder heads. The research question is whether such a shared ternary model can retain useful visual representations across a range of measured inference costs.

## 2. Proposed abstract (prospective; no experimental claims)

Vision Transformers offer strong visual representations but generally operate at a fixed computation budget. We investigate **Matryoshka-BitViT**, a ternary Vision Transformer designed to support nested execution widths and depths within a single shared-parameter model. The method applies prefix slicing to BitLinear projections before ternary quantization, including grouped query/key/value projections, attention output projections, and feed-forward expansion/contraction layers. The attention head dimension is held fixed while the number of active heads varies with model width; LayerScale and normalization must also support active channel prefixes. We propose training with a frozen DINOv3 teacher using multi-configuration distillation of global CLS and dense patch similarities, with optional direct feature alignment at matching dimensions. We will evaluate representation quality, latency, memory, and storage across elastic configurations and compare against floating-point and independently trained baselines. **All empirical claims await experiments.**

## 3. Key design decisions

- **Teacher:** Frozen ordinary DINOv3 ViT-L/16, e.g. 1024 channels, 24 Transformer blocks, 16 attention heads, and 64 dimensions/head.
- **Student:** `DinoVisionMatryoshkaTransformer(DinoVisionTransformer)` with a maximum configuration comparable to ViT-L/16. Shared weights, no extra decoders or projection heads by default.
- **Width elasticity:** Apply active channel prefixes to **both** input and output dimensions of internal projections; reduce head count while preserving the original head dimension.
- **Depth elasticity:** Execute the first `d` blocks, where `d` is selected at training or inference time. Smaller-depth outputs use the existing feature normalization path.
- **Output elasticity:** Return the first `m` embedding channels (`m <= width`) for CLS and patch tokens. Earlier output dimensions are *encouraged* to be useful by multi-configuration training; their PCA-like importance is **not mathematically guaranteed**.
- **Quantization:** In reduced-width BitLinear calls, slice the relevant weight matrix **before** applying `w_q`; validate width-specific quantization scales and export behavior.
- **Training:** Teacher frozen; student trained end-to-end with shared gradients; **no PCA**, no auxiliary decoders, no layer-boundary `.detach()`.
- **Image views:** Teacher and student process matching image geometry. Offline patch features require exactly matching preprocessing and patch indexing; online teacher is safer when using geometric augmentation.

## 4. Matryoshka modules discussed and drafted

| Component | Intended behavior | Prototype file |
|---|---|---|
| `Linear` / nested BitLinear | Prefix input/output submatrix; grouped QKV or gate/value slicing; quantize active submatrix | `linear_matryoshka_dropin.py` |
| `SelfMatryoshkaAttention(SelfAttention)` | Keep head_dim fixed, use active `heads = width // head_dim`; grouped Q/K/V prefixes; narrowed output projection; support `forward_list()` | `self_matryoshka_attention.py`, `attention_with_matryoshka.py` |
| `MatryoshkaMlp(Mlp)` | Narrow `fc1` and `fc2`, preserving expansion ratio and ternary quantization | `matryoshka_mlp.py` |
| `MatryoshkaLayerScale(LayerScale)` | Multiply by `gamma[:width]`; preserve weight compatibility | `matryoshka_layer_scale.py` |
| `DinoVisionMatryoshkaTransformer` | Choose `(width, depth, out_dim)`, share weights, return CLS/patch features | `matryoshka_class.py`, `vision_transformer_matryoshka.py` |
| **Still needed:** Matryoshka normalization | `LayerNorm` / `RMSNorm` with active width; preserve original parameter prefixes | Not yet integrated as a reusable module |
| **Still needed:** integrated Matryoshka block | Wire Matryoshka attention, MLP, LayerScale, normalization into `SelfAttentionBlock` | Not yet fully consolidated |

**Source files provided by the user:** `vision_transformer.py`, `attention.py`, `block.py`; original BitLinear `Linear` and `LayerScale` snippets; original `Mlp` snippet. An earlier `coco_dinov3_pca.py` exists, but PCA **is not part of this proposed training method**. It may only be useful as reference for teacher feature extraction and image preprocessing.

**Integration caution:** The earlier `DinoVisionMatryoshkaTransformer` prototype contains manual prefix implementations of attention, MLP, normalization, and LayerScale. The newer individual Matryoshka modules are **not yet a guaranteed single integrated code path**. Avoid duplicating two separate prefix implementations: consolidate and test one clean path.

## 5. Tensor and attention rules

For a maximum-width ViT-L with `W_max = 1024`, `H_max = 16`, and `head_dim = 64`:

- Allowed attention widths include `128, 256, 384, 512, 768, 1024` (multiples of 64).
- Active head count: `H(w) = w / 64`.
- Packed QKV weight layout is `[Q full rows; K full rows; V full rows]`.
- For each group, take rows `[:w]` and input columns `[:w]`. **Never** replace this with `qkv.weight[:3*w, :w]`, which mixes up group boundaries.
- Reshape selected QKV as `[B, N, 3, H(w), 64]`, then transpose to `[B, H(w), N, 64]` before attention.
- RoPE must broadcast to or be sliced to the **active head count** as appropriate for its actual tensor shape.
- After attention, `out_proj` uses an active `[w, w]` weight.
- For an FFN with ratio 4, `fc1: w -> 4w` and `fc2: 4w -> w`, using shared submatrices of the maximum-size weights.

## 6. Proposed frozen-teacher distillation plan

**Illustrative configuration sets, to be refined:**

- Widths: `W = {128, 256, 384, 512, 768, 1024}`.
- Depths: `D = {6, 12, 18, 24}`.
- Output dimensions: `M = {64, 96, 128, 192, 256, 384, 768, 1024}`, subject to `m <= w`.

At each batch:

1. Compute the teacher's CLS `[B,1024]` and patch `[B,P,1024]` outputs once with gradients disabled.
2. Sample student configurations using a **sandwich rule**: the maximum model, minimum model, and one or more random valid configurations.
3. For each configuration `(w,d,m)`, compute student CLS `[B,m]` and patch `[B,P,m]` outputs.
4. Compute **dimension-agnostic relational losses** using row-normalized Gram / similarity matrices: cross-image CLS similarities and within-image patch similarities (sample patches if needed).
5. At the full matching dimension, optionally add direct CLS and patch cosine alignment so the largest student preserves teacher feature coordinates.
6. Average configuration losses and backpropagate through the shared student parameters. No gradient stops between blocks.

An illustrative objective is

`L(w,d,m) = L_cls_gram + 0.5 * L_patch_gram + 1[m=1024] * 0.5 * L_direct_alignment`

The coefficients are **initial experimental choices**, not validated results. The CLS Gram loss needs batches of at least two images; the patch Gram loss should compare matching patch locations. Pure relational matching preserves similarities but does **not** uniquely align coordinates with the teacher.

**Suggested experimental phases:** (i) verify the full-size path; (ii) train elastic widths and output prefixes at full depth; (iii) introduce elastic depths; (iv) train joint sandwich configurations. Do not claim a particular schedule is optimal without ablations.

## 7. Required correctness checks before training

- Full-width, full-depth outputs match the expected original DINOv3 model path **when weights and quantization behavior are identical**. Do not assume full-precision pretrained weights and ternary inference are numerically identical.
- Shapes and finite outputs across allowed widths; robust invalid-width errors.
- QKV Q/K/V **group-wise** slicing and correct active-head reshape.
- RoPE handling for full-width and narrowed head axes, masks/storage/CLS tokens, and list/multi-crop forward paths.
- Full-width equivalence of `SelfMatryoshkaAttention` to `SelfAttention`, and `MatryoshkaMlp` to `Mlp`, with identical weights.
- Backpropagation gradients reach active weight submatrices but not inactive ones when testing an isolated subnetwork.
- `LayerScale`, `LayerNorm`, `RMSNorm` and, if used, SwiGLU behave correctly at every width.
- Ensure the **real installed** `ternarylayers` / `Bit1p58Weight` backend works, including masked-K bias and `to_ternary()` export per width.
- Profile real computational savings. Current patch embedding may still compute its **full width** before slicing; narrow matrices alone do not prove whole-model speedups.

Existing prototype tests include `test_self_matryoshka_attention_simple.py`, `test_matryoshka_mlp_simple.py`, `test_matryoshka_layer_scale_simple.py`, and other exploratory tests. **Some previous tests used mock backends.** Syntax compilation was checked; real-package end-to-end results must not be inferred from mock tests.

## 8. Experimental evaluation and paper-worthy claims

**Baselines:** frozen full-precision DINOv3; independently distilled smaller ViTs; elastic floating-point ViT; independently trained ternary small ViTs; proposed shared ternary elastic ViT.

**Metrics:** k-NN / retrieval, linear probes, patch/dense-feature performance; latency with batch size and device specified, FLOPs, peak memory, checkpoint size / exported subnetwork size; sensitivity to width/depth and quantization scale.

**Ablations:** width only, depth only, combined; output-prefix supervision on/off; CLS vs patch loss; slice-before-quantization vs alternative scale handling; full-precision vs ternary; single configuration vs sandwich training.

**Positioning:** Prior work already includes Matryoshka Representation Learning, nested/elastic Transformers and ViTs (such as MatFormer, HydraViT, EA-ViT), and slimmable/once-for-all networks. The architectural idea alone is **not established as novel**. A publishable claim must rest on reproducible evidence for *joint ternary quantization + elastic width/depth + DINOv3 feature distillation*, not on claiming first-ever Matryoshka attention.

## 9. Current known limitations / next engineering priority

1. **Integrate width-aware normalization** (`MatryoshkaLayerNorm` and optional `MatryoshkaRMSNorm`) and a single clear Matryoshka block, reusing the attention/MLP/LayerScale components rather than parallel manual implementations.
2. Verify the integrated model using the **actual** `ternarylayers` quantization backend and both full/small ViT configurations.
3. Check preprocessing, RoPE, storage tokens, patch embedding cost, list path, and cloned models.
4. Build a minimal frozen-teacher distillation training script with online CLS and sampled patch Gram loss; use no PCA.
5. Train the first reproducible experiment and collect a width/depth accuracy–latency grid.
6. Only then refine the paper's claims and results section.

## 10. Paste this to restart in a new chat

> We are continuing **Matryoshka-BitViT**, a decoder-free, ternary-weight DINOv3 ViT with one shared set of parameters and elastic attention width, MLP width, depth, and output feature prefixes. We have draft classes `SelfMatryoshkaAttention`, `MatryoshkaMlp`, `MatryoshkaLayerScale`, an extended BitLinear, and `DinoVisionMatryoshkaTransformer`. Train by distilling CLS and patch similarities from a frozen normal DINOv3 teacher, **without PCA, auxiliary decoders, or `.detach()` between blocks**. Our next task is to consolidate these modules into a clean integrated Transformer, starting with Matryoshka LayerNorm / RMSNorm and a Matryoshka SelfAttentionBlock, then test on the actual ternary backend. Please read the attached project handoff and code files before editing; preserve the original full-width code path, weight compatibility, and group-wise QKV slicing. Treat performance and novelty claims as unverified until benchmarks exist.
