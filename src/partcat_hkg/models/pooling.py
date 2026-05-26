from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_presence(prob: torch.Tensor, k: int = 64) -> torch.Tensor:
    """Top-k mean presence from a probability mask tensor [B,C,H,W]."""

    if prob.ndim != 4:
        raise ValueError(f"topk_presence expects [B,C,H,W], got {tuple(prob.shape)}")
    bsz, channels, _, _ = prob.shape
    flat = torch.nan_to_num(prob.float(), nan=0.0, posinf=1.0, neginf=0.0).flatten(2)
    kk = max(1, min(int(k), flat.shape[-1]))
    return flat.topk(kk, dim=-1).values.mean(dim=-1).view(bsz, channels)


def topmean_presence(prob: torch.Tensor, q: float = 0.01) -> torch.Tensor:
    """Top-q mean presence used by the proposal; q is a fraction in (0,1]."""

    flat = prob.flatten(2)
    k = max(1, int(round(float(q) * flat.shape[-1])))
    return topk_presence(prob, k=k)


def sharpen_prob_mask(prob: torch.Tensor, *, temperature: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """Sharpen a probability mask without changing its support.

    The proposal uses \tilde{M}_k = sigmoid(L_k/T)^alpha before token pooling.
    When probabilities are already available, we map back through logits in a
    numerically safe way so the same temperature behavior is retained.
    """

    p = torch.nan_to_num(prob.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
    temperature = max(float(temperature), 1e-6)
    if abs(temperature - 1.0) > 1e-6:
        p = torch.sigmoid(torch.logit(p) / temperature)
    alpha = max(float(alpha), 1e-6)
    if abs(alpha - 1.0) > 1e-6:
        p = p.pow(alpha)
    return p.clamp(0.0, 1.0)


def masked_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pool feature maps through masks.

    Args:
        feat: [B, C, Hf, Wf]
        mask: [B, K, Hm, Wm]
    Returns:
        [B, K, C]
    """

    if feat.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode="bilinear", align_corners=False)
    w = torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    num = torch.einsum("bchw,bkhw->bkc", feat.float(), w)
    den = w.flatten(2).sum(-1).clamp_min(eps).unsqueeze(-1)
    return num / den


def mask_sharpened_pool(
    feat: torch.Tensor,
    prob_mask: torch.Tensor,
    *,
    presence: torch.Tensor | None = None,
    temperature: float = 1.0,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mask-sharpened part-token pooling with optional presence gating."""

    weights = sharpen_prob_mask(prob_mask, temperature=temperature, alpha=alpha)
    tokens = masked_pool(feat, weights, eps=eps)
    if presence is not None:
        tokens = tokens * presence.to(tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return tokens


def gated_masked_pool(
    feat: torch.Tensor,
    prob_mask: torch.Tensor,
    presence: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    power: float = 1.0,
    gate: bool = True,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Proposal-style mask-sharpened pooling plus optional presence gating.

    Args:
        feat: Feature map ``[B,C,H,W]``.
        prob_mask: Probability masks ``[B,K,H,W]``.
        presence: Presence scores ``[B,K]``.
        temperature: Pooling temperature ``T``.
        power: Sharpening exponent ``alpha``.
        gate: If true, multiply tokens by presence.
        normalize: If true, L2-normalize token vectors.
    """
    # Normalize appearance first, then apply presence gating.  This preserves
    # the proposal's behavior z_k <- p_k z_k; normalizing after the gate would
    # erase the magnitude of nonzero presence scores.
    tokens = mask_sharpened_pool(
        feat,
        prob_mask,
        presence=None,
        temperature=temperature,
        alpha=power,
        eps=eps,
    )
    if normalize:
        tokens = F.normalize(tokens.float(), dim=-1).to(feat.dtype)
    if gate and presence is not None:
        tokens = tokens * presence.to(tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return tokens
