import torch
import torch.nn as nn

# -------------------------
# EMA (optional but helps stability)
# -------------------------
class EMA:
    decay = 0.999
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not hasattr(self, "backup"):
            return
        for name, p in model.named_parameters():
            if name in self.backup:
                p.copy_(self.backup[name])
        del self.backup