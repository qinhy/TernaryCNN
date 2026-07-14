import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torchvision.transforms import v2
import torchvision.transforms.functional as vF

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
import albumentations as A
import albumentations.augmentations.crops.functional as fcrops
import albumentations.augmentations.geometric.functional as fgeometric

from typing import Any, Callable, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator

class ToTensor(BaseModel):
    dtype:str = "float32"
    scale:bool=True
    _torch_dtype:Any = None

    @model_validator(mode='after')
    def valid_model(self):
        self._torch_dtype = {
            "float32":torch.float32,
            "float16":torch.float16,
        }[self.dtype]
        return self

    def build(self):
        return v2.Compose([v2.ToImage(),
                           v2.ToDtype(self._torch_dtype, scale=self.scale)])


class Cutout(nn.Module):
    """
    Cutout for tensors [C, H, W], where the hole size is computed from image size.

    ratio:
        - float        -> square hole: (H*ratio, W*ratio)
        - (rh, rw)     -> rectangular hole: (H*rh, W*rw)

    ratio_range:
        - (lo, hi)     -> sample r ~ Uniform(lo, hi) each call (applies to both H and W)
                          (overrides ratio if provided)

    Example:
        For 32x32 and want 8x8 cutout => ratio = 8/32 = 0.25
    """

    def __init__(
        self,
        ratio: Union[float, Tuple[float, float]] = 0.25,
        ratio_range: Optional[Tuple[float, float]] = None,
        fill: float = 0.0,
        min_size: int = 1,
    ):
        super().__init__()
        self.ratio = ratio
        self.ratio_range = ratio_range
        self.fill = float(fill)
        self.min_size = int(min_size)

    def _get_ratios(self, device) -> Tuple[float, float]:
        if self.ratio_range is not None:
            lo, hi = self.ratio_range
            r = float(torch.empty(1, device=device).uniform_(lo, hi).item())
            return r, r

        if isinstance(self.ratio, tuple):
            return float(self.ratio[0]), float(self.ratio[1])

        r = float(self.ratio)
        return r, r

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img):
            return img

        if img.ndim != 3:
            raise ValueError(f"Cutout expects [C,H,W] tensor, got shape: {tuple(img.shape)}")

        _, h, w = img.shape
        if h <= 0 or w <= 0:
            return img

        rh, rw = self._get_ratios(device=img.device)
        rh = max(rh, 0.0)
        rw = max(rw, 0.0)

        mask_h = int(round(h * rh))
        mask_w = int(round(w * rw))

        mask_h = max(self.min_size, min(mask_h, h))
        mask_w = max(self.min_size, min(mask_w, w))

        cy = int(torch.randint(0, h, (1,), device=img.device).item())
        cx = int(torch.randint(0, w, (1,), device=img.device).item())

        y1 = max(0, cy - mask_h // 2)
        y2 = min(h, y1 + mask_h)
        x1 = max(0, cx - mask_w // 2)
        x2 = min(w, x1 + mask_w)

        out = img.clone()
        out[:, y1:y2, x1:x2] = self.fill
        return out

class AutoRandomCrop(nn.Module):
    """
    Output size is always the original (H, W).
    Padding is computed dynamically as round(H*ratio_h) and round(W*ratio_w).
    """

    def __init__(
        self,
        pad_ratio: Union[float, Tuple[float, float]] = 0.125,
        pad_ratio_range: Optional[Tuple[float, float]] = None,  # e.g. (0.05, 0.15)
        fill: int = 0,
    ) -> None:
        super().__init__()
        self.pad_ratio = pad_ratio
        self.pad_ratio_range = pad_ratio_range
        self.fill = fill

    def _get_ratios(self) -> Tuple[float, float]:
        if self.pad_ratio_range is not None:
            lo, hi = self.pad_ratio_range
            r = float(torch.empty(1).uniform_(lo, hi).item())
            return r, r

        if isinstance(self.pad_ratio, tuple):
            return float(self.pad_ratio[0]), float(self.pad_ratio[1])

        r = float(self.pad_ratio)
        return r, r

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = vF.get_dimensions(img)  # (C, H, W)

        rh, rw = self._get_ratios()
        rh = max(rh, 0.0)
        rw = max(rw, 0.0)

        pad_h = int(round(h * rh))
        pad_w = int(round(w * rw))

        if pad_h > 0 or pad_w > 0:
            # padding format: [left, top, right, bottom]
            img = vF.pad(img, padding=[pad_w, pad_h, pad_w, pad_h], fill=self.fill)

            _, H, W = vF.get_dimensions(img)
            max_i = H - h
            max_j = W - w

            i = int(torch.randint(0, max_i + 1, (1,)).item()) if max_i > 0 else 0
            j = int(torch.randint(0, max_j + 1, (1,)).item()) if max_j > 0 else 0

            img = vF.crop(img, i, j, h, w)

        return img

class AutoRandomResizedCrop(A.RandomResizedCrop):
    def __init__(self, scale = ..., ratio = ..., interpolation = cv2.INTER_LINEAR, mask_interpolation = cv2.INTER_NEAREST, area_for_downscale = None, p = 1):
        super().__init__((1,1), scale, ratio, interpolation, mask_interpolation, area_for_downscale, p)

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        self.size = (h, w)
        crop = fcrops.crop(img, *crop_coords)
        interpolation = self._get_interpolation_for_resize(crop.shape[:2], "image")
        return fgeometric.resize(crop, self.size, interpolation)
        

class SimpleImageTrainAugment(BaseModel):
    mean: Sequence[float]
    std: Sequence[float]
    p: float = Field(default=0.325, ge=0.0, le=1.0)
    

    # advance
    flip: bool = True
    noise_std_range: Tuple[float, float] = (0.0125, 0.025)
    cout_ratio_range: Tuple[float, float] = (0.1, 0.2)

    def build(self):
        rmin, rmax = self.cout_ratio_range
        return  A.Compose([
            A.Rotate(limit=(-10,10), p=self.p),
            AutoRandomResizedCrop(scale=(0.8, 1.0), ratio=(0.75, 1.33),p=self.p),
            A.HorizontalFlip(p=self.p) if self.flip else A.NoOp(),
            A.OneOf([
                A.ColorJitter(p=1.0,brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                A.ToGray(p=1.0),
            ], p=self.p),
            A.OneOf([
                A.GaussianBlur(p=1.0,blur_limit=(1, 3)),
                A.GaussNoise(p=1.0,std_range=self.noise_std_range),
            ], p=self.p),
             A.CoarseDropout(num_holes_range=(1,4), 
                             hole_height_range=(rmin, rmax), 
                             hole_width_range=(rmin, rmax), p=self.p),
            A.Normalize(mean=self.mean, std=self.std),
            A.ToTensorV2(),
        ])

class SimpleImageValAugment(BaseModel):
    mean: Sequence[float]
    std: Sequence[float]

    dtype: str= "float32"

    def build(self) -> v2.Compose:
        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            A.ToTensorV2(),
        ])

class MixupCutmix(BaseModel):
    num_classes: int = Field(ge=2)

    enable_mixup: bool = False
    enable_cutmix: bool = False

    # Shared alpha (you can split later if you want)
    beta_alpha: float = Field(default=1.0, gt=0.0)
    p: float = Field(default=0.325, ge=0.0, le=1.0)


    def build(self) -> Optional[v2.Transform]:
        candidates = []
        if self.enable_mixup:
            candidates.append(v2.MixUp(num_classes=self.num_classes, alpha=self.beta_alpha))
        if self.enable_cutmix:
            candidates.append(v2.CutMix(num_classes=self.num_classes, alpha=self.beta_alpha))

        if not candidates:
            return v2.Identity()

        chosen = candidates[0] if len(candidates) == 1 else v2.RandomChoice(candidates)
        return v2.RandomApply([chosen], p=self.p)

class CIFAR100Augment(BaseModel):
    # --- normalization ---
    mean: Tuple[float, ...] = (0.5071, 0.4867, 0.4408)
    std: Tuple[float, ...] = (0.2675, 0.2565, 0.2761)
    # --- crop ---
    crop_output_size: int = 32
    crop_padding_px: int = 4
    def build(self):
        return (
            SimpleImageTrainAugment.model_validate(self.model_dump()).build(),
            SimpleImageValAugment.model_validate(self.model_dump()).build(),
            MixupCutmix(num_classes=100,enable_cutmix=True,enable_mixup=True).build(),
        )


# Helpers: 1/f fractal + blur + rotate
# -----------------------------
def generate_fractal_1_over_f(batch, c, h, w, beta=2.0, device=None, dtype=torch.float32):
    """
    Returns [B,C,H,W] in [0,1] (float32 by default).
    Uses rfft/irfft so the result is properly real-valued.
    """
    device = device or "cpu"
    fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device, dtype=torch.float32).view(1, -1)
    f = torch.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = 1.0 / (f ** (beta / 2.0))  # [H, W//2+1]

    real = torch.randn(batch, c, h, w // 2 + 1, device=device, dtype=torch.float32)
    imag = torch.randn(batch, c, h, w // 2 + 1, device=device, dtype=torch.float32)
    spec = torch.complex(real, imag) * amp  # broadcast

    img = torch.fft.irfft2(spec, s=(h, w))  # [B,C,H,W], real
    img = img - img.amin(dim=(2, 3), keepdim=True)
    img = img / (img.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return img.to(dtype=dtype).clamp(0, 1)


def rotate_tensor(x, degrees):
    """
    Rotate a tensor [N,C,H,W] by degrees around center using grid_sample.
    """
    n, c, h, w = x.shape
    device, dtype = x.device, x.dtype
    angle = torch.tensor(degrees, device=device, dtype=dtype) * (math.pi / 180.0)
    ca, sa = torch.cos(angle), torch.sin(angle)

    theta = torch.zeros((n, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = ca
    theta[:, 0, 1] = -sa
    theta[:, 1, 0] = sa
    theta[:, 1, 1] = ca

    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def normalize_map(m):
    m = m - m.amin(dim=(2, 3), keepdim=True)
    m = m / (m.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return m


# -----------------------------
# Integrated S²-FracMix module
# -----------------------------
class S2FracMix(BaseModel):
    scales: Tuple[float, ...] = Field(default=(0.2, 0.35, 0.5), min_length=1)
    theta_max: float = Field(default=30.0, ge=0.0)
    saliency_thresh: float = Field(default=0.5, ge=0.0, le=1.0)

    fractal_lambda: float = Field(default=0.2, ge=0.0, le=1.0)
    fractal_beta: float = Field(default=2.0, gt=0.0)

    blur_kernel: int = Field(default=7, ge=1)
    blur_sigma: Tuple[float, float] = Field(default=(0.1, 2.0))

    s2_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    apply_in_eval: bool = Field(default=False)
    return_saliency: bool = Field(default=False)

    model_input_fn: Optional[Callable] = Field(default=None, exclude=True)

    def build(self):return S2FracMixModule(self)    

class S2FracMixModule(nn.Module):
    """
    s2 = S2FracMix(
        s2_prob=0.5,
        model_input_fn=None,      # or your normalize() wrapper
        return_saliency=False
    ).cuda()

    model.train()
    images_aug, y_soft, used, _ = s2(model, images, labels)  # labels can be [B] ints

    logits = model(images_aug)
    loss = -(F.log_softmax(logits, dim=1) * y_soft).sum(dim=1).mean()

    Integrated augmenter:
      forward(model, images, labels_or_onehot) -> (images_aug, labels_onehot, used, saliency)

    images: [B,C,H,W] assumed in [0,1] (or at least bounded)
    labels_or_onehot: [B] int OR [B,K] float onehot/soft
    """
    def __init__(self,config:S2FracMix):
        super().__init__()
        self.scales = tuple(config.scales)
        self.theta_max = float(config.theta_max)
        self.t = float(config.saliency_thresh)
        self.lam_f = float(config.fractal_lambda)
        self.beta = float(config.fractal_beta)
        self.blur = GaussianBlur(kernel_size=config.blur_kernel, sigma=config.blur_sigma)
        self.s2_prob = float(config.s2_prob)
        self.apply_in_eval = bool(config.apply_in_eval)
        self.model_input_fn = config.model_input_fn
        self.return_saliency = bool(config.return_saliency)

    def _sample_salient_topleft(self, mask, ph, pw):
        # mask: [H,W] bool
        H, W = mask.shape
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            y = torch.randint(0, max(1, H - ph + 1), (1,), device=mask.device).item()
            x = torch.randint(0, max(1, W - pw + 1), (1,), device=mask.device).item()
            return y, x

        idx = torch.randint(0, ys.numel(), (1,), device=mask.device).item()
        cy, cx = ys[idx].item(), xs[idx].item()
        y = int(max(0, min(H - ph, cy - ph // 2)))
        x = int(max(0, min(W - pw, cx - pw // 2)))
        return y, x

    def _labels_to_onehot(self, labels_or_onehot, K, device, dtype):
        if labels_or_onehot.dim() == 1:
            return F.one_hot(labels_or_onehot.to(device=device), num_classes=K).to(dtype=dtype)
        if labels_or_onehot.dim() == 2:
            return labels_or_onehot.to(device=device, dtype=dtype)
        raise ValueError("labels_or_onehot must be [B] or [B,K]")

    def _compute_saliency(self, model, images, labels_onehot):
        """
        One backward wrt input only. Runs model in eval mode and disables param grads temporarily.
        """
        was_training = model.training
        req = [p.requires_grad for p in model.parameters()]
        try:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                x = images.detach().to(dtype=torch.float32).clone().requires_grad_(True)

                x_in = self.model_input_fn(x) if self.model_input_fn is not None else x
                logits = model(x_in)  # [B,K]

                # make labels match logits dtype
                y = labels_onehot.to(device=logits.device, dtype=logits.dtype)
                score = (logits * y).sum()

                grads = torch.autograd.grad(score, x, create_graph=False, retain_graph=False)[0]
                sal = grads.abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
                sal = normalize_map(sal)
                return sal.detach(), logits.detach()
        finally:
            for p, r in zip(model.parameters(), req):
                p.requires_grad_(r)
            model.train(was_training)

    @torch.no_grad()
    def _augment_with_saliency(self, images, saliency):
        B, C, H, W = images.shape
        out = []
        n_scales = len(self.scales)

        for i in range(B):
            I = images[i:i+1]     # [1,C,H,W]
            S = saliency[i:i+1]   # [1,1,H,W]
            mask = (S[0, 0] >= self.t)

            Pm = torch.zeros_like(I)

            for s in self.scales:
                ph = max(1, int(math.floor(s * H)))
                pw = max(1, int(math.floor(s * W)))
                y0, x0 = self._sample_salient_topleft(mask, ph, pw)

                Pk = I[:, :, y0:y0+ph, x0:x0+pw]  # [1,C,ph,pw]
                Sk = S[:, :, y0:y0+ph, x0:x0+pw]  # [1,1,ph,pw]

                # FracMix inside patch
                Fk = generate_fractal_1_over_f(1, C, ph, pw, beta=self.beta,
                                               device=Pk.device, dtype=Pk.dtype)
                Pk_mix = self.lam_f * Fk + (1.0 - self.lam_f) * Pk

                # Transform: rotate + blur gated by saliency
                theta = (torch.rand((), device=Pk.device).item() * 2.0 - 1.0) * self.theta_max
                Pk_rot = rotate_tensor(Pk_mix, theta)
                Pk_blr = self.blur(Pk_mix)

                Tk = Pk_rot * (1.0 - Sk) + Pk_blr * Sk

                # Resize to full and accumulate
                Rk = F.interpolate(Tk, size=(H, W), mode="bilinear", align_corners=False)
                Pm = Pm + (1.0 / n_scales) * Rk

            alpha = torch.rand((), device=I.device).item()
            I_tilde = alpha * I + (1.0 - alpha) * Pm
            out.append(I_tilde)

        return torch.cat(out, dim=0).clamp(0, 1)

    def forward(self, model, images, labels_or_onehot):
        """
        Returns:
          images_out: [B,C,H,W]
          labels_onehot: [B,K]
          used: bool
          saliency (optional): [B,1,H,W] or None
        """
        if (not self.training) and (not self.apply_in_eval):
            # still convert labels to onehot for downstream
            with torch.no_grad():
                # infer K by a cheap forward if needed
                logits = model(self.model_input_fn(images) if self.model_input_fn else images)
                K = logits.shape[1]
                y = self._labels_to_onehot(labels_or_onehot, K, images.device, logits.dtype)
            return images, y, False, (None if not self.return_saliency else None)

        # decide apply
        if torch.rand((), device=images.device) >= self.s2_prob:
            with torch.no_grad():
                logits = model(self.model_input_fn(images) if self.model_input_fn else images)
                K = logits.shape[1]
                y = self._labels_to_onehot(labels_or_onehot, K, images.device, logits.dtype)
            return images, y, False, (None if not self.return_saliency else None)

        # run saliency and augment
        # infer K from model output during saliency pass
        with torch.no_grad():
            # (only to get K/dtype if labels are ints; keeps behavior consistent)
            tmp_logits = model(self.model_input_fn(images) if self.model_input_fn else images)
            K = tmp_logits.shape[1]
            logits_dtype = tmp_logits.dtype

        labels_onehot = self._labels_to_onehot(labels_or_onehot, K, images.device, logits_dtype)
        saliency, _ = self._compute_saliency(model, images, labels_onehot)
        images_aug = self._augment_with_saliency(images, saliency)

        return images_aug, labels_onehot, True, (saliency if self.return_saliency else None)
