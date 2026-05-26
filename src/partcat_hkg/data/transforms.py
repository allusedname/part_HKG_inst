from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance


_FLIP_LEFT_RIGHT = getattr(getattr(Image, "Transpose", Image), "FLIP_LEFT_RIGHT", Image.FLIP_LEFT_RIGHT)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _normalize(img_raw: torch.Tensor) -> torch.Tensor:
    return (img_raw - _IMAGENET_MEAN) / _IMAGENET_STD


def _jitter_factor(amount: float) -> float:
    return 1.0 + random.uniform(-float(amount), float(amount))


def _apply_color_jitter(img: Image.Image) -> Image.Image:
    # Lightweight PIL-only replacement for torchvision ColorJitter. Hue jitter is
    # intentionally omitted to keep the dependency-free path simple and stable.
    img = ImageEnhance.Brightness(img).enhance(_jitter_factor(0.20))
    img = ImageEnhance.Contrast(img).enhance(_jitter_factor(0.20))
    img = ImageEnhance.Color(img).enhance(_jitter_factor(0.15))
    return img


class JointImageMaskTransform:
    def __init__(self, img_size: int = 384, train: bool = True):
        self.img_size = int(img_size)
        self.train = bool(train)

    def __call__(self, img: Image.Image, masks: dict[str, np.ndarray]):
        if self.train and random.random() < 0.5:
            img = img.transpose(_FLIP_LEFT_RIGHT)
            masks = {k: np.ascontiguousarray(v[:, ::-1]) for k, v in masks.items()}
        if self.train:
            img = _apply_color_jitter(img)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_raw = _to_tensor(img)
        img_norm = _normalize(img_raw)
        masks_t = {}
        for key, value in masks.items():
            m = Image.fromarray((value > 0).astype(np.uint8) * 255).resize(
                (self.img_size, self.img_size), Image.NEAREST
            )
            masks_t[key] = torch.from_numpy((np.array(m) > 0).astype(np.uint8)).unsqueeze(0)
        return img_norm, img_raw, masks_t


class ImageOnlyTransform:
    def __init__(self, img_size: int = 384, train: bool = True):
        self.img_size = int(img_size)
        self.train = bool(train)

    def __call__(self, img: Image.Image):
        if self.train and random.random() < 0.5:
            img = img.transpose(_FLIP_LEFT_RIGHT)
        if self.train:
            img = _apply_color_jitter(img)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_raw = _to_tensor(img)
        return _normalize(img_raw), img_raw
