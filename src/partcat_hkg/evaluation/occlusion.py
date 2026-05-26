from __future__ import annotations

import torch


def apply_patch_occlusion(images: torch.Tensor, frac: float = 0.25, value: float = 0.0) -> torch.Tensor:
    out = images.clone()
    _, _, h, w = out.shape
    ph, pw = max(1, int(h * frac)), max(1, int(w * frac))
    y0 = max(0, (h - ph) // 2)
    x0 = max(0, (w - pw) // 2)
    out[:, :, y0:y0 + ph, x0:x0 + pw] = value
    return out
