"""Four-mode, from-scratch scaled PRepBN demo (PyTorch only).

Based on the RepBN/PRepBN *idea* in Guo et al., ICML 2024:
    RepBN(z) = BN(z) + eta*z
    PRepBN(z) = lam*LN(z) + (1-lam)*RepBN(z)

Our extensions are feature-wise pre_scale and pos_scale, plus an experimental
causal training variant that uses statistics from *previous* training steps.
It is NOT the authors' official implementation.

Axes: x is [B, T, D]; pre_scale, pos_scale, gamma, beta, eta are all [D].
Run: python scaled_prepbn_four_modes.py
     python scaled_prepbn_four_modes.py --full-size  # B=32,T=128,D=4096 (heavy)
"""

import argparse
import torch
from torch import nn
import torch.nn.functional as F


class ScaledPRepBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.2):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        
        # Learnable [D] vectors: fixed across batch and token positions.
        self.gamma = nn.Parameter(torch.ones(num_features))   # BN affine scale
        self.beta = nn.Parameter(torch.zeros(num_features))    # BN affine shift
        self.eta = nn.Parameter(torch.full((num_features,), 0.1))  # RepBN residual

        # Buffers, not learned by gradient descent.
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.register_buffer("pre_scale", torch.ones(num_features))
        self.register_buffer("pos_scale", torch.ones(num_features))

    @staticmethod
    def batch_stats(z):
        # For noncausal BN, reductions mix the entire sequence.
        mean = z.mean(dim=(0, 1))
        var = ((z - mean) ** 2).mean(dim=(0, 1))
        return mean, var

    def mix(self, z, repbn, lam):
        # LN has no learned affine here; pre/post scales are separate.
        if lam == 0.0:
            mixed = repbn
        elif lam == 1.0:
            mixed = F.layer_norm(z, (z.shape[-1],), eps=self.eps)
        else:
            ln = F.layer_norm(z, (z.shape[-1],), eps=self.eps)
            mixed = lam * ln + (1.0 - lam) * repbn
        return self.pos_scale * mixed

    def repbn(self, z, mean, var):
        bn = self.gamma * (z - mean) * torch.rsqrt(var + self.eps) + self.beta
        return bn + self.eta * z

    @torch.no_grad()
    def update_running(self, observed_stats):
        """Call AFTER backward: current data affect only later forward passes."""
        mean, var = observed_stats
        self.running_mean.lerp_(mean.detach(), self.momentum)
        self.running_var.lerp_(var.detach(), self.momentum)

    # --------- PART 1: NON-CAUSAL TRAINING ---------
    def forward_train_noncausal(self, x, lam):
        z = x * self.pre_scale
        mean, var = self.batch_stats(z)     # Includes FUTURE token positions
        y = self.mix(z, self.repbn(z, mean, var), lam)
        return y, (mean.detach(), var.detach())

    # --------- PART 2: NON-CAUSAL MODEL, INFERENCE ---------
    def forward_inference_noncausal(self, x, A, C):
        # Running stats frozen -> affine transformation, no future leakage.
        return x * A + C

    # --------- PART 3: CAUSAL TRAINING ---------
    def forward_train_causal(self, x, lam):
        z = x * self.pre_scale
        # ONLY statistics accumulated from previous training steps are used.
        y = self.mix(
            z, self.repbn(z, self.running_mean, self.running_var), lam
        )
        # Statistics collected now are returned for use AFTER backward.
        # They do not influence any output in this forward pass.
        observed = self.batch_stats(z.detach())
        return y, observed

    # --------- PART 4: CAUSAL MODEL, INFERENCE ---------
    def forward_inference_causal(self, x, A, C):
        # Same affine formula, but parameters were learned causally.
        return x * A + C

    @torch.no_grad()
    def fused_affine(self):
        """Valid once lam=0 and running statistics/parameters are frozen."""
        inv_std = torch.rsqrt(self.running_var + self.eps)
        A = self.pos_scale * self.pre_scale * (self.gamma * inv_std + self.eta)
        C = self.pos_scale * (self.beta - self.gamma * self.running_mean * inv_std)
        return A, C

    @torch.no_grad()
    def inference_unfused(self, x):
        """Reference RepBN inference for verifying A*x+C."""
        z = x * self.pre_scale
        return self.pos_scale * self.repbn(z, self.running_mean, self.running_var)


def train_demo(model, causal, shape, steps=6):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(steps):
        x = torch.randn(shape)
        lam = 1.0 - step / (steps - 1)  # Gradual LN -> RepBN; ends at 0

        optimizer.zero_grad()
        if causal:
            y, observed = model.forward_train_causal(x, lam)
        else:
            y, observed = model.forward_train_noncausal(x, lam)

        loss = (y - 0.3 * x).square().mean()  # Toy loss, not GPT loss
        loss.backward()
        model.update_running(observed)       # AFTER backward; never mid-forward
        optimizer.step()

        print(f"  step={step} lam={lam:.2f} loss={loss.item():.5f}")


def verify(model, causal, shape):
    model.eval()
    B, T, D = shape
    x = torch.randn(shape)
    x_changed = x.clone()
    x_changed[:, T // 2:] += 5.0  # Alter only FUTURE token positions

    # First test TRAINING formula, holding saved statistics unchanged.
    # In causal mode, observed new statistics aren't used for this output.
    with torch.no_grad():
        train_fn = (model.forward_train_causal if causal
                    else model.forward_train_noncausal)
        a, _ = train_fn(x, lam=0.0)
        b, _ = train_fn(x_changed, lam=0.0)
        train_diff = (a[:, :T // 2] - b[:, :T // 2]).abs().max().item()

        # After training, both modes use the exact same type of affine formula.
        A, C = model.fused_affine()
        infer_fn = (model.forward_inference_causal if causal
                    else model.forward_inference_noncausal)
        fused = infer_fn(x, A, C)
        unfused = model.inference_unfused(x)
        torch.testing.assert_close(fused, unfused, rtol=1e-4, atol=1e-5)

        fused_changed = infer_fn(x_changed, A, C)
        infer_diff = (
            fused[:, :T // 2] - fused_changed[:, :T // 2]
        ).abs().max().item()

    print(f"  earlier-token change (training formula):  {train_diff:.7f}")
    print(f"  earlier-token change (fused inference): {infer_diff:.7f}")
    print("  fused inference equals unfused inference: PASS")
    if causal:
        assert train_diff < 1e-6, "Causal forward leaked future tokens!"
    else:
        assert train_diff > 1e-4, "Expected conventional BN future leakage"
    assert infer_diff < 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-size", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    shape = (32, 128, 4096) if args.full_size else (2, 8, 16)
    print("x.shape:", shape)

    print("\n[1] Non-causal scaled PRepBN — TRAIN")
    noncausal = ScaledPRepBN(shape[-1])
    train_demo(noncausal, causal=False, shape=shape)
    print("\n[2] Non-causally trained PRepBN — INFERENCE / checks")
    verify(noncausal, causal=False, shape=shape)

    print("\n[3] Causal scaled PRepBN — TRAIN")
    causal = ScaledPRepBN(shape[-1])
    train_demo(causal, causal=True, shape=shape)
    print("\n[4] Causally trained PRepBN — INFERENCE / checks")
    verify(causal, causal=True, shape=shape)


if __name__ == "__main__":
    main()
