import pytest
import torch

from ternarylayers.dinov3.models.vision_transformer import DinoVisionTransformer
from ternarylayers.dinov3.models.vision_transformer_matryoshka import (
    DinoVisionMatryoshka,
    MatryoshkaClassifier,
    build_standalone_classifier,
    vit_matryoshka_large,
    vit_matryoshka_base,
    vit_matryoshka_small,
)


def make_backbone(**kwargs):
    config = dict(img_size=16, patch_size=8, embed_dim=256, depth=3,
                  num_heads=8, ffn_ratio=3.0, no_conv=True, pos_embed_rope_dtype='fp32',
                  splits={'full': (256, 3), 'half': (128, 2), 'tiny': (64, 1)})
    config.update(kwargs)
    model = DinoVisionMatryoshka(**config)
    model.init_weights()
    return model


@pytest.mark.parametrize('norm', ['layernorm', 'layernormbf16', 'rmsnorm'])
def test_configurable_exports_match_shared_prefixes(norm):
    model = MatryoshkaClassifier(num_classes=7, backbone=make_backbone(
        norm_layer=norm, n_storage_tokens=2, mask_k_bias=True,
        untie_cls_and_patch_norms=True, layerscale_init=0.5,
        sattn_qkv_quant_dim=[0, 1, 0], sattn_fc1_quant_dim=[1, 0, 1],
        sattn_fc2_quant_dim=[0, 1, 0],
    )).eval()
    images = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        for name, (width, depth) in model.splits.items():
            exported = model.export_subnetwork(name)
            assert exported.backbone.embed_dim == width
            assert len(exported.backbone.blocks) == depth
            assert exported.backbone.num_heads == width // 32
            torch.testing.assert_close(model(images, name=name), exported(images))
        cloned = model.backbone.clone()
        assert not cloned.training
        assert cloned.splits == model.splits
        assert cloned.blocks[1].attn.qkv.w_q.dim == 1
        torch.testing.assert_close(cloned(images), model.backbone(images))


def test_arbitrary_width_depth_masks_and_gradients():
    model = make_backbone(n_storage_tokens=1)
    images = torch.randn(2, 3, 16, 16)
    masks = torch.tensor([[True, False, False, True]] * 2)
    # 96 is valid although it is not a named split.
    result = model(images, masks, width=96, depth=1, out_dim=48, is_training=True)
    assert result['x_norm_clstoken'].shape == (2, 48)
    assert result['x_norm_patchtokens'].shape == (2, 4, 48)
    assert result['x_storage_tokens'].shape == (2, 1, 48)
    result['x_norm_clstoken'].square().sum().backward()
    grad = model.blocks[0].attn.qkv.weight.grad.reshape(3, 256, 256)
    assert grad[:, :96, :96].abs().sum() > 0
    assert torch.count_nonzero(grad[:, 96:, :]) == 0
    assert torch.count_nonzero(grad[:, :, 96:]) == 0
    assert model.blocks[1].attn.qkv.weight.grad is None
    results = model.forward_features([images, images[:1]], width=128)
    assert [r['x_norm_clstoken'].shape for r in results] == [(2, 128), (1, 128)]


def test_full_width_accepts_ordinary_checkpoint():
    elastic = make_backbone().eval()
    ordinary = DinoVisionTransformer(img_size=16, patch_size=8, embed_dim=256,
                                    depth=3, num_heads=8, ffn_ratio=3.0, no_conv=True,
                                    pos_embed_rope_dtype='fp32').eval()
    ordinary.load_state_dict(elastic.state_dict(), strict=True)
    images = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(elastic(images), ordinary(images))


def test_rebuild_generic_export():
    images = torch.randn(1, 3, 16, 16)
    model = MatryoshkaClassifier(backbone=make_backbone()).eval()
    compact = model.export_subnetwork('half')
    rebuilt = build_standalone_classifier('half', splits=model.splits, head_dim=32,
                                          ffn_ratio=3.0, img_size=16, patch_size=8, no_conv=True,
                                          pos_embed_rope_dtype='fp32').eval()
    rebuilt.load_state_dict(compact.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(images), model(images, name='half'))


@pytest.mark.parametrize('factory,width,depth,heads', [
    (vit_matryoshka_small, 384, 12, 6),
    (vit_matryoshka_base, 768, 12, 12),
    (vit_matryoshka_large, 1024, 24, 16),
])
def test_larger_presets_on_meta(factory, width, depth, heads):
    model = factory(device='meta')
    assert (model.embed_dim, model.depth, model.num_heads) == (width, depth, heads)
    assert model.head_dim == 64
    assert all(block.attn.qkv.weight.is_meta for block in model.blocks)
    assert len(model.blocks) == depth
    expected = {'large': (1024, 24), 'base': (768, 12), 'small': (384, 12),
                'tiny': (256, 6), 'nano': (192, 6), 'micro': (128, 4), 'femto': (64, 3)}
    assert model.splits == {name: pair for name, pair in expected.items() if pair[0] <= width}


@pytest.mark.parametrize('kwargs', [
    {'embed_dim': 255}, {'num_heads': 0}, {'depth': 0},
    {'splits': {}}, {'splits': {'bad': (95, 1)}},
    {'splits': {'bad': (128, 4)}}, {'ffn_layer': 'swiglu'},
    {'sattn_qkv_quant_dim': [0]},
])
def test_invalid_architecture(kwargs):
    with pytest.raises(ValueError):
        make_backbone(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'width': 95}, {'width': 288}, {'depth': 4}, {'out_dim': 257},
])
def test_invalid_forward_selection(kwargs):
    with pytest.raises(ValueError):
        make_backbone()(torch.randn(1, 3, 16, 16), **kwargs)


def test_mnist_epoch_supports_generic_splits():
    from examples.train_mnist_matryoshka import run_epoch
    from torch.utils.data import DataLoader, TensorDataset

    model = MatryoshkaClassifier(backbone=make_backbone())
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 16, 16), torch.tensor([2, 5])),
                        batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.classifier.weight.detach().clone()
    metrics = run_epoch(model, loader, torch.device('cpu'), optimizer)
    assert set(metrics) == set(model.splits)
    assert all(torch.isfinite(torch.tensor(m['loss'])) for m in metrics.values())
    assert not torch.equal(before, model.classifier.weight)
    assert set(run_epoch(model, loader, torch.device('cpu'))) == set(model.splits)


def test_fractional_ffn_ratio_and_same_width_different_depths():
    model = make_backbone(ffn_ratio=2.1,
                          splits={'shallow': (160, 1), 'deep': (160, 2)}).eval()
    images = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        for name, (width, depth) in model.splits.items():
            compact = model.export_subnetwork(name)
            assert compact.blocks[0].mlp.fc1.out_features == 335
            rebuilt = DinoVisionTransformer(img_size=16, patch_size=8, embed_dim=160,
                depth=depth, num_heads=5, ffn_ratio=compact.ffn_ratio, no_conv=True,
                pos_embed_rope_dtype='fp32').eval()
            rebuilt.load_state_dict(compact.state_dict(), strict=True)
            torch.testing.assert_close(model(images, width=width, depth=depth), compact(images))
    with pytest.raises(ValueError, match='specify depth'):
        model(images, width=160)


def test_masked_qkv_quantizes_weights():
    from ternarylayers.dinov3.layers.attention import LinearKMaskedBias

    layer = LinearKMaskedBias(8, 24, bias=True)
    with torch.no_grad():
        layer.bias.fill_(0.5)
        layer.bias_mask.fill_(1)
        layer.bias_mask[8:16].zero_()
    x = torch.randn(2, 8)
    expected = torch.nn.functional.linear(x, layer.w_q(layer.weight),
                                          layer.bias * layer.bias_mask)
    torch.testing.assert_close(layer(x), expected)
