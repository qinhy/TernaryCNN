"""Scaled PRepBN for 2D feature maps: four training/inference modes.

Shape: x = [B, C, H, W], with all learned scale/bias parameters [C].
Equivalent broadcasting shape for parameters: [1, C, 1, 1].

This is a pedagogical adaptation of the PRepBN idea, not the original implementation.
  RepBN(z)  = BN2d(z) + eta * z
  PRepBN(z) = lam * ChannelLayerNorm(z) + (1-lam) * RepBN(z)
  output    = pos_scale * PRepBN(pre_scale * x)

Non-causal training: BN2d statistics across (B,H,W) *of the current input*.
Causal training: uses running statistics from PREVIOUS training steps for the
                  current forward, then updates running stats AFTER backward.
Inference for either training mode (when lam=0): y = A*x + C_bias.

The word "causal" means that changing future spatial positions cannot affect
an earlier position *through this normalization layer*. For an autoregressive
image model, surrounding convolutions/attention must also be causal/masked.

Usage:
  python scaled_prepbn_2d_four_modes.py
  python scaled_prepbn_2d_four_modes.py --full-size
  python scaled_prepbn_2d_four_modes.py --steps 8
"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


class ScaledPRepBN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.2):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.channels = num_features

        # Feature-wise (channel-wise) learnable parameters.
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eta = nn.Parameter(torch.full((num_features,), 0.1))

        # Moving stats used in inference, and in causal training forwards.
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.register_buffer("pre_scale", torch.ones(num_features))
        self.register_buffer("pos_scale", torch.ones(num_features))

    @staticmethod
    def broadcast(v):
        # [C] -> [1,C,1,1]
        return v.reshape(1, -1, 1, 1)

    @staticmethod
    def batch_stats(z):
        """BN2d statistics: for each channel, aggregate over B,H,W."""
        mu = z.mean(dim=(0, 2, 3))
        var = (z - mu[None, :, None, None]).square().mean(dim=(0, 2, 3))
        return mu, var

    def channel_layer_norm(self, z):
        """LN across channels at EACH pixel (not across spatial positions)."""
        # [B,C,H,W] -> [B,H,W,C] -> LN(C) -> [B,C,H,W]
        z_nhwc = z.permute(0, 2, 3, 1)
        return F.layer_norm(z_nhwc, (self.channels,), eps=self.eps).permute(0, 3, 1, 2)

    def repbn(self, z, mean, var):
        mean = self.broadcast(mean)
        var = self.broadcast(var)
        norm = (z - mean) * torch.rsqrt(var + self.eps)
        bn = self.broadcast(self.gamma) * norm + self.broadcast(self.beta)
        return bn + self.broadcast(self.eta) * z

    def mix(self, z, repbn, lam):
        if not 0.0 <= lam <= 1.0:
            raise ValueError('lam must be between 0 and 1')
        if lam == 0.0:
            mixed = repbn
        elif lam == 1.0:
            mixed = self.channel_layer_norm(z)
        else:
            mixed = lam * self.channel_layer_norm(z) + (1.0 - lam) * repbn
        return self.broadcast(self.pos_scale) * mixed

    @torch.no_grad()
    def update_running(self, observed_stats):
        """Must be called only AFTER backward, never inside causal forward."""
        mu, var = observed_stats
        self.running_mean.lerp_(mu.detach(), self.momentum)
        self.running_var.lerp_(var.detach(), self.momentum)

    # -------- PART 1: NON-CAUSAL, TRAIN --------
    def forward_train_noncausal(self, x, lam):
        z = x * self.broadcast(self.pre_scale)
        mean, var = self.batch_stats(z)  # Includes all spatial positions.
        y = self.mix(z, self.repbn(z, mean, var), lam)
        return y, (mean.detach(), var.detach())

    # -------- PART 2: NON-CAUSALLY TRAINED MODEL, INFERENCE --------
    def forward_inference_noncausal(self, x, A, C_bias):
        return x * self.broadcast(A) + self.broadcast(C_bias)

    # -------- PART 3: CAUSAL, TRAIN --------
    def forward_train_causal(self, x, lam):
        z = x * self.broadcast(self.pre_scale)
        # Fixed within this forward; stats are from PREVIOUS training steps.
        y = self.mix(z, self.repbn(z, self.running_mean, self.running_var), lam)
        observed = self.batch_stats(z.detach())  # For LATER forward passes.
        return y, observed

    # -------- PART 4: CAUSALLY TRAINED MODEL, INFERENCE --------
    def forward_inference_causal(self, x, A, C_bias):
        return x * self.broadcast(A) + self.broadcast(C_bias)

    @torch.no_grad()
    def fused_affine(self):
        """Exact inference affine map when lam=0, after stats are frozen."""
        inv_std = torch.rsqrt(self.running_var + self.eps)
        A = self.pos_scale * self.pre_scale * (self.gamma * inv_std + self.eta)
        C_bias = self.pos_scale * (self.beta - self.gamma * self.running_mean * inv_std)
        return A, C_bias

    @torch.no_grad()
    def inference_unfused(self, x):
        z = x * self.broadcast(self.pre_scale)
        return self.broadcast(self.pos_scale) * self.repbn(
            z, self.running_mean, self.running_var
        )


def train_demo(model, causal, shape, steps=6):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(steps):
        x = torch.randn(shape, device=model.pre_scale.device)
        lam = 1.0 - step / (steps - 1) if steps > 1 else 0.0
        optimizer.zero_grad(set_to_none=True)
        if causal:
            y, stats = model.forward_train_causal(x, lam)
        else:
            y, stats = model.forward_train_noncausal(x, lam)
        loss = (y - 0.3 * x).square().mean()  # Toy loss, NOT generative modeling
        loss.backward()
        optimizer.step()
        # Important: use current observed stats only in future forwards.
        model.update_running(stats)
        print(f'  step={step:02d} lam={lam:.2f} loss={loss.item():.5f}')


class ChannelAffine2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.register_buffer("scale", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))

    def add_pre_scale_bias(self, scale, bias=None):
        if bias:
            self.bias += self.scale*bias
        self.scale *= scale

    def add_pos_scale_bias(self, scale, bias=None):
        if bias:
            self.bias = scale*self.bias + bias
        self.scale *= scale

    def forward(self, x):
        return (
            x * self.scale.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )

    @classmethod
    @torch.no_grad()
    def from_ScaledPRepBN2d(cls, trained_ScaledPRepBN2d:ScaledPRepBN2d):
        A, C_bias = trained_ScaledPRepBN2d.fused_affine()

        layer = cls(A.numel()).to(device=A.device, dtype=A.dtype)
        layer.scale.copy_(A)
        layer.bias.copy_(C_bias)

        return layer

    
def verify(model, causal, shape):
    model.eval()
    x = torch.randn(shape, device=model.pre_scale.device)
    x_changed = x.clone()
    # Raster scan ordering: flatten H,W; perturb only the *later* pixels.
    cut = shape[-2] * shape[-1] // 2
    x_changed.flatten(start_dim=2)[:, :, cut:] += 5.0

    with torch.no_grad():
        train_fn = model.forward_train_causal if causal else model.forward_train_noncausal
        a, _ = train_fn(x, lam=0.35)
        b, _ = train_fn(x_changed, lam=0.35)
        # Compare ONLY earlier pixels.
        train_diff = (a.flatten(2)[:, :, :cut] - b.flatten(2)[:, :, :cut]).abs().max().item()

        # Verify exact inference fusion (at lam=0).
        A, C_bias = model.fused_affine()
        inference_fn = model.forward_inference_causal if causal else model.forward_inference_noncausal
        fused = inference_fn(x, A, C_bias)
        unfused = model.inference_unfused(x)
        torch.testing.assert_close(fused, unfused, rtol=1e-4, atol=1e-5)
        changed_fused = inference_fn(x_changed, A, C_bias)
        infer_diff = (
            fused.flatten(2)[:, :, :cut] - changed_fused.flatten(2)[:, :, :cut]
        ).abs().max().item()

    print(f'  earlier-pixel difference during training (lam=.35): {train_diff:.7f}')
    print(f'  earlier-pixel difference during inference:         {infer_diff:.7f}')
    print('  fused inference == unfused inference: PASS')
    if causal:
        assert train_diff < 1e-6, 'Causal forward leaked future spatial information!'
    else:
        assert train_diff > 1e-4, 'Expected leakage from non-causal batch statistics.'
    assert infer_diff < 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-size', action='store_true')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('--steps must be at least 2 to demonstrate the transition')

    torch.manual_seed(7)
    shape = (8, 64, 32, 32) if args.full_size else (2, 4, 4, 4)
    print('x.shape [B,C,H,W]:', shape)

    print('\n[1] NON-CAUSAL scaled PRepBN2d — TRAIN')
    noncausal = ScaledPRepBN2d(shape[1]).to(args.device)
    train_demo(noncausal, causal=False, shape=shape, steps=args.steps)
    print('\n[2] NON-CAUSAL model — INFERENCE')
    verify(noncausal, causal=False, shape=shape)

    print('\n[3] CAUSAL scaled PRepBN2d — TRAIN')
    causal = ScaledPRepBN2d(shape[1]).to(args.device)
    train_demo(causal, causal=True, shape=shape, steps=args.steps)
    print('\n[4] CAUSAL model — INFERENCE')
    verify(causal, causal=True, shape=shape)


if __name__ == '__main__':
    main()
