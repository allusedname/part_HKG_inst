from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, s: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(_valid_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )


class _FallbackFeatureBackbone(nn.Module):
    def __init__(self, name: str = "resnet18"):
        super().__init__()
        if name == "resnet50":
            self.skip_ch, self.low_ch, self.high_ch = 256, 512, 1024
        else:
            self.skip_ch, self.low_ch, self.high_ch = 64, 128, 256
        self.name = f"fallback_{name}"
        self.stem = nn.Sequential(
            ConvGNAct(3, self.skip_ch, k=7, p=3, s=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ConvGNAct(self.skip_ch, self.skip_ch), ConvGNAct(self.skip_ch, self.skip_ch))
        self.layer2 = nn.Sequential(ConvGNAct(self.skip_ch, self.low_ch, s=2), ConvGNAct(self.low_ch, self.low_ch))
        self.layer3 = nn.Sequential(ConvGNAct(self.low_ch, self.high_ch, s=2), ConvGNAct(self.high_ch, self.high_ch))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        skip = self.layer1(x)
        low = self.layer2(skip)
        high = self.layer3(low)
        return {"skip": skip, "low": low, "high": high}


class ResNetFeatureBackbone(nn.Module):
    """ResNet feature backbone returning skip/low/high maps.

    If torchvision is unavailable or mismatched with the local PyTorch build, a
    small convolutional fallback with the same channel contract is used. This
    keeps tests and CPU smoke runs working while preserving real ResNet behavior
    in normal environments. ``backbone_name="tiny"`` intentionally selects the
    lightweight fallback for fast synthetic smoke tests.
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = False, freeze: bool = False):
        super().__init__()
        name = str(name).lower()
        if name == "tiny":
            fallback = _FallbackFeatureBackbone("resnet18")
            self.using_fallback = True
            self.name = "tiny_fallback"
            self.skip_ch, self.low_ch, self.high_ch = fallback.skip_ch, fallback.low_ch, fallback.high_ch
            self.stem = fallback.stem
            self.layer1 = fallback.layer1
            self.layer2 = fallback.layer2
            self.layer3 = fallback.layer3
        else:
            self.using_fallback = False
            try:
                from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

                if name == "resnet50":
                    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                    net = resnet50(weights=weights)
                    self.skip_ch, self.low_ch, self.high_ch = 256, 512, 1024
                elif name == "resnet18":
                    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                    net = resnet18(weights=weights)
                    self.skip_ch, self.low_ch, self.high_ch = 64, 128, 256
                else:
                    raise ValueError(f"Unsupported backbone_name={name!r}; expected 'resnet18', 'resnet50', or 'tiny'")
                self.name = name
                self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
                self.layer1 = net.layer1
                self.layer2 = net.layer2
                self.layer3 = net.layer3
            except Exception as exc:  # pragma: no cover - depends on local torchvision build
                if name not in {"resnet18", "resnet50", "tiny"}:
                    raise ValueError(f"Unsupported backbone_name={name!r}; expected 'resnet18', 'resnet50', or 'tiny'") from exc
                fallback = _FallbackFeatureBackbone(name)
                self.using_fallback = True
                self.name = fallback.name
                self.skip_ch, self.low_ch, self.high_ch = fallback.skip_ch, fallback.low_ch, fallback.high_ch
                self.stem = fallback.stem
                self.layer1 = fallback.layer1
                self.layer2 = fallback.layer2
                self.layer3 = fallback.layer3
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        skip = self.layer1(x)
        low = self.layer2(skip)
        high = self.layer3(low)
        return {"skip": skip, "low": low, "high": high}


def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
    """Convert ViT token output to a square feature map."""
    if tokens.ndim == 4:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Cannot convert DINO output with shape {tuple(tokens.shape)} to feature map")
    bsz, n_tokens, channels = tokens.shape
    side = int(round(math.sqrt(n_tokens)))
    if side * side == n_tokens:
        return tokens.transpose(1, 2).reshape(bsz, channels, side, side)
    side = int(round(math.sqrt(n_tokens - 1)))
    if side * side == n_tokens - 1:
        return tokens[:, 1:, :].transpose(1, 2).reshape(bsz, channels, side, side)
    raise ValueError(f"Cannot infer square feature map from token shape {tuple(tokens.shape)}")


class OptionalDINOFeatureMap(nn.Module):
    """Frozen DINO/timm feature extractor with a zero fallback.

    The wrapper mirrors the previous notebook's robust path: resize only the DINO
    branch to its expected input size, read ``forward_features`` patch tokens, and
    interpolate the resulting map back to the ResNet feature resolution. If timm
    or weights are unavailable, it returns zeros so the rest of Stage 1 remains
    runnable for smoke tests and CPU-only debugging.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "vit_small_patch16_224.dino",
        weights_path: str = "",
        input_size: int = 224,
        freeze: bool = True,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.input_size = int(input_size or 224)
        self.out_ch = 1
        self.model: nn.Module | None = None
        self._status = "disabled"
        if not self.enabled:
            return
        try:
            import timm

            self.model = timm.create_model(model_name, pretrained=not bool(weights_path), num_classes=0)
            if weights_path:
                weights = Path(weights_path)
                if not weights.exists():
                    raise FileNotFoundError(str(weights))
                payload = torch.load(weights, map_location="cpu")
                state = payload.get("model", payload.get("state_dict", payload))
                self.model.load_state_dict(state, strict=False)
            self.model.eval()
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad_(False)
            with torch.no_grad():
                probe = torch.zeros(1, 3, self.input_size, self.input_size)
                fmap = self.forward(probe, target_hw=(max(1, self.input_size // 16), max(1, self.input_size // 16)))
                self.out_ch = int(fmap.shape[1])
            self._status = f"timm:{model_name}; input_resized_to={self.input_size}"
        except Exception as exc:  # pragma: no cover - depends on optional packages/weights
            self.model = None
            self.out_ch = 1
            self._status = f"zero_fallback:{exc}"

    def _resize_for_dino(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size and (x.shape[-2] != self.input_size or x.shape[-1] != self.input_size):
            return F.interpolate(x.float(), size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        return x.float()

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        if target_hw is None:
            target_hw = (max(1, x.shape[-2] // 16), max(1, x.shape[-1] // 16))
        if self.model is None:
            return x.new_zeros((x.shape[0], 1, target_hw[0], target_hw[1]))
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward_features(self._resize_for_dino(x))
        if isinstance(out, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state", "x_prenorm"):
                if key in out:
                    out = out[key]
                    break
        if isinstance(out, (list, tuple)):
            out = out[0]
        fmap = _tokens_to_feature_map(out)
        if tuple(fmap.shape[-2:]) != tuple(target_hw):
            fmap = F.interpolate(fmap.float(), size=target_hw, mode="bilinear", align_corners=False)
        return fmap.to(device=x.device, dtype=x.dtype)
