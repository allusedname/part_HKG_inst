from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


GEOM_FEATURE_NAMES = ["cx", "cy", "bbox_w", "bbox_h", "area", "score"]
GEOM_DIM = len(GEOM_FEATURE_NAMES)

# Keep the same order as partcat_hkg.kg.relations.RELATION_FEATURE_NAMES.
RELATION_DIM = 14


@dataclass
class GPUComponentConfig:
    """Configuration for fully GPU component/terminal extraction.

    The extractor deliberately avoids CPU connected-components.  It creates a
    small fixed budget of soft component terminals by local-maximum NMS and
    gaussian support masks.  This is much faster for Stage-2 training and, with a
    frozen Stage-1 model, can be cached once and reused for all Stage-2 epochs.
    """

    mask_size: int = 64
    threshold: float = 0.20
    local_max_kernel: int = 7
    gaussian_sigma: float = 0.075
    max_components_per_part: int = 2
    max_total_components: int = 32
    min_presence: float = 0.05
    keep_component_masks: bool = False


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def config_from_any(cfg: Any) -> GPUComponentConfig:
    return GPUComponentConfig(
        mask_size=int(_cfg(cfg, "gpu_component_mask_size", _cfg(cfg, "component_mask_size", 64))),
        threshold=float(_cfg(cfg, "component_threshold", 0.20)),
        local_max_kernel=int(_cfg(cfg, "gpu_component_local_max_kernel", 7)),
        gaussian_sigma=float(_cfg(cfg, "gpu_component_gaussian_sigma", 0.075)),
        max_components_per_part=int(_cfg(cfg, "max_components_per_part", 2)),
        max_total_components=int(_cfg(cfg, "max_total_components", 32)),
        min_presence=float(_cfg(cfg, "component_min_presence", 0.05)),
        keep_component_masks=bool(_cfg(cfg, "keep_component_masks", False)),
    )


def _resize_prob(prob: torch.Tensor, mask_size: int) -> torch.Tensor:
    if prob.shape[-2:] == (int(mask_size), int(mask_size)):
        return prob.float().clamp(0, 1)
    return F.interpolate(prob.float(), size=(int(mask_size), int(mask_size)), mode="bilinear", align_corners=False).clamp(0, 1)


def _make_xy_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    yy = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, 1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, 1, width)
    return xx, yy


def _gather_last(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather x[..., N, D?] along the second-to-last component axis."""
    if x.ndim == idx.ndim:
        return torch.gather(x, -1, idx)
    expand_shape = list(idx.shape) + list(x.shape[idx.ndim:])
    return torch.gather(x, idx.ndim - 1, idx.view(*idx.shape, *([1] * (x.ndim - idx.ndim))).expand(*expand_shape))


def _topk_local_maxima(prob_low: torch.Tensor, presence: torch.Tensor | None, cfg: GPUComponentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return peak score/x/y for each part and soft validity mask.

    Inputs
    ------
    prob_low: [B,K,H,W]
    presence: [B,K] or None

    Returns
    -------
    peak_val, peak_x, peak_y, peak_valid with shape [B,K,M].
    """
    bsz, k_num, height, width = prob_low.shape
    kernel = max(1, int(cfg.local_max_kernel))
    if kernel % 2 == 0:
        kernel += 1
    pooled = F.max_pool2d(prob_low, kernel_size=kernel, stride=1, padding=kernel // 2)
    maxima = (prob_low >= pooled - 1e-6).to(prob_low.dtype)
    scores = prob_low * maxima
    # Fallback: if a part has no strict local maximum above threshold, topk still
    # returns the strongest pixels from the zeroed score map. Validity below will
    # remove weak detections.
    flat = scores.flatten(-2)
    m = max(1, int(cfg.max_components_per_part))
    peak_val, peak_idx = torch.topk(flat, k=min(m, flat.shape[-1]), dim=-1)
    if peak_val.shape[-1] < m:
        pad = m - peak_val.shape[-1]
        peak_val = F.pad(peak_val, (0, pad))
        peak_idx = F.pad(peak_idx, (0, pad))
    peak_y = (peak_idx // width).to(prob_low.dtype) / float(max(height - 1, 1))
    peak_x = (peak_idx % width).to(prob_low.dtype) / float(max(width - 1, 1))
    valid = peak_val >= float(cfg.threshold)
    if presence is not None:
        valid = valid & (presence.float().clamp(0, 1).unsqueeze(-1) >= float(cfg.min_presence))
    return peak_val, peak_x, peak_y, valid


def _soft_component_masks(prob_low: torch.Tensor, peak_x: torch.Tensor, peak_y: torch.Tensor, cfg: GPUComponentConfig) -> torch.Tensor:
    """Create gaussian-windowed soft masks [B,K,M,H,W]."""
    bsz, k_num, height, width = prob_low.shape
    dtype = prob_low.dtype
    xx, yy = _make_xy_grid(height, width, prob_low.device, dtype)
    px = peak_x.to(dtype).unsqueeze(-1).unsqueeze(-1)
    py = peak_y.to(dtype).unsqueeze(-1).unsqueeze(-1)
    sigma = max(float(cfg.gaussian_sigma), 1e-4)
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    gate = torch.exp(-0.5 * dist2 / (sigma * sigma))
    masks = prob_low.unsqueeze(2) * gate
    return torch.nan_to_num(masks, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _geometry_from_soft_masks(masks: torch.Tensor, peak_val: torch.Tensor) -> torch.Tensor:
    """Compute centroid/size/area/score from soft masks.

    masks: [B,K,M,H,W]
    peak_val: [B,K,M]
    returns geom [B,K,M,6]
    """
    bsz, k_num, m, height, width = masks.shape
    dtype, device = masks.dtype, masks.device
    yy = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, 1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, 1, width)
    mass = masks.sum(dim=(-2, -1)).clamp_min(1e-6)
    cx = (masks * xx).sum(dim=(-2, -1)) / mass
    cy = (masks * yy).sum(dim=(-2, -1)) / mass
    vx = (masks * (xx - cx.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(-2, -1)) / mass
    vy = (masks * (yy - cy.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(dim=(-2, -1)) / mass
    # Gaussian-window components are soft, so use variance-derived sizes rather
    # than hard min/max boxes.  Four standard deviations approximates a compact
    # bbox for the component support.
    bbox_w = (4.0 * torch.sqrt(vx.clamp_min(1e-8))).clamp(1.0 / max(width, 1), 1.0)
    bbox_h = (4.0 * torch.sqrt(vy.clamp_min(1e-8))).clamp(1.0 / max(height, 1), 1.0)
    area = (mass / float(max(height * width, 1))).clamp(0, 1)
    geom = torch.stack([cx, cy, bbox_w, bbox_h, area, peak_val.to(dtype).clamp(0, 1)], dim=-1)
    return torch.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)


def _pool_tokens_from_masks(
    masks: torch.Tensor,
    token_map: torch.Tensor | None,
    part_tokens: torch.Tensor,
) -> torch.Tensor:
    """Pool component tokens from Stage-1 token maps or fallback part tokens.

    masks: [B,K,M,H,W]
    token_map: [B,D,h,w] or None
    part_tokens: [B,K,D]
    returns [B,K,M,D]
    """
    bsz, k_num, m, height, width = masks.shape
    if token_map is None:
        return part_tokens.unsqueeze(2).expand(bsz, k_num, m, part_tokens.shape[-1]).float()
    if token_map.ndim != 4:
        raise ValueError(f"token_map must be [B,D,h,w], got {tuple(token_map.shape)}")
    _, dim, th, tw = token_map.shape
    flat_masks = masks.reshape(bsz * k_num * m, 1, height, width)
    weights = F.interpolate(flat_masks.float(), size=(th, tw), mode="bilinear", align_corners=False).reshape(bsz, k_num, m, th, tw)
    denom = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
    tok = torch.einsum("bdhw,bkmhw->bkmd", token_map.float(), weights) / denom.unsqueeze(-1)
    fallback = part_tokens.unsqueeze(2).expand_as(tok).float()
    tok = torch.where(torch.isfinite(tok).all(dim=-1, keepdim=True), tok, fallback)
    return torch.nan_to_num(tok, nan=0.0, posinf=0.0, neginf=0.0)


def extract_gpu_instance_components(
    part_prob: torch.Tensor,
    part_tokens: torch.Tensor,
    *,
    part_presence: torch.Tensor | None = None,
    token_map: torch.Tensor | None = None,
    cfg: GPUComponentConfig | None = None,
    return_masks: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Fully GPU, padded component extraction from Stage-1 part outputs.

    Parameters
    ----------
    part_prob:
        ``[B,K,H,W]`` part probability maps.
    part_tokens:
        ``[B,K,D]`` functional part tokens.
    part_presence:
        Optional ``[B,K]`` presence scores. If absent, a top-pixel estimate is
        used for validity gating.
    token_map:
        Optional ``[B,D,h,w]`` Stage-1 token map. If present, component-specific
        tokens are pooled from it; otherwise part tokens are repeated.

    Returns
    -------
    A padded dict with N = ``cfg.max_total_components``:
    ``component_valid [B,N]``, ``component_part [B,N]``,
    ``component_presence [B,N]``, ``component_geom [B,N,6]``,
    ``component_token [B,N,D]`` and optionally ``component_mask [B,N,M,M]``.
    """
    if part_prob.ndim != 4:
        raise ValueError(f"part_prob must be [B,K,H,W], got {tuple(part_prob.shape)}")
    if part_tokens.ndim != 3:
        raise ValueError(f"part_tokens must be [B,K,D], got {tuple(part_tokens.shape)}")
    cfg = cfg or GPUComponentConfig()
    bsz, k_num, _, _ = part_prob.shape
    dim = int(part_tokens.shape[-1])
    prob_low = _resize_prob(part_prob, int(cfg.mask_size))
    if part_presence is None:
        presence = prob_low.flatten(-2).topk(k=min(32, prob_low.shape[-1] * prob_low.shape[-2]), dim=-1).values.mean(-1)
    else:
        presence = part_presence.float().clamp(0, 1)
    peak_val, peak_x, peak_y, peak_valid = _topk_local_maxima(prob_low, presence, cfg)
    masks = _soft_component_masks(prob_low, peak_x, peak_y, cfg)
    geom = _geometry_from_soft_masks(masks, peak_val)
    tokens = _pool_tokens_from_masks(masks, token_map, part_tokens)
    m = int(cfg.max_components_per_part)
    total = k_num * m
    part_ids = torch.arange(k_num, device=part_prob.device, dtype=torch.long).view(1, k_num, 1).expand(bsz, k_num, m)
    comp_score = (peak_val * peak_valid.to(peak_val.dtype)).reshape(bsz, total)
    comp_part = part_ids.reshape(bsz, total)
    comp_valid = peak_valid.reshape(bsz, total)
    comp_presence = peak_val.reshape(bsz, total).clamp(0, 1)
    comp_geom = geom.reshape(bsz, total, GEOM_DIM)
    comp_token = tokens.reshape(bsz, total, dim)
    comp_mask = masks.reshape(bsz, total, int(cfg.mask_size), int(cfg.mask_size))

    n = min(max(1, int(cfg.max_total_components)), total)
    top_score, top_idx = torch.topk(comp_score, k=n, dim=-1)
    comp_valid = torch.gather(comp_valid, 1, top_idx) & (top_score >= float(cfg.threshold))
    comp_part = torch.gather(comp_part, 1, top_idx)
    comp_presence = torch.gather(comp_presence, 1, top_idx) * comp_valid.to(comp_presence.dtype)
    comp_geom = _gather_last(comp_geom, top_idx)
    comp_token = _gather_last(comp_token, top_idx)
    comp_mask = _gather_last(comp_mask, top_idx) if (return_masks if return_masks is not None else cfg.keep_component_masks) else None

    # Pad if max_total_components exceeds K * max_components_per_part.
    target_n = max(1, int(cfg.max_total_components))
    if n < target_n:
        pad_n = target_n - n
        comp_valid = F.pad(comp_valid, (0, pad_n), value=False)
        comp_part = F.pad(comp_part, (0, pad_n), value=0)
        comp_presence = F.pad(comp_presence, (0, pad_n), value=0.0)
        comp_geom = F.pad(comp_geom, (0, 0, 0, pad_n), value=0.0)
        comp_token = F.pad(comp_token, (0, 0, 0, pad_n), value=0.0)
        if comp_mask is not None:
            comp_mask = F.pad(comp_mask, (0, 0, 0, 0, 0, pad_n), value=0.0)
    out = {
        "component_valid": comp_valid.bool(),
        "component_part": comp_part.long(),
        "component_presence": comp_presence.float(),
        "component_geom": comp_geom.float(),
        "component_token": comp_token.float(),
    }
    if comp_mask is not None:
        out["component_mask"] = comp_mask.float()
    return out


def average_stage1_token_maps(out: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Average available Stage-1 token maps on GPU."""
    maps: list[torch.Tensor] = []
    for key in ("token_res_map", "token_dino_map"):
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim == 4:
            maps.append(value.float())
    if not maps:
        return None
    if len(maps) == 1:
        return maps[0]
    target = maps[0].shape[-2:]
    aligned = []
    for m in maps:
        aligned.append(m if m.shape[-2:] == target else F.interpolate(m, size=target, mode="bilinear", align_corners=False))
    return torch.stack(aligned, dim=0).mean(0)


def bbox_from_geom(geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate boxes from [cx,cy,w,h,...] geometry."""
    cx, cy, bw, bh = geom[..., 0], geom[..., 1], geom[..., 2].clamp_min(1e-4), geom[..., 3].clamp_min(1e-4)
    x0 = (cx - 0.5 * bw).clamp(0, 1)
    y0 = (cy - 0.5 * bh).clamp(0, 1)
    x1 = (cx + 0.5 * bw).clamp(0, 1)
    y1 = (cy + 0.5 * bh).clamp(0, 1)
    return x0, y0, x1, y1


def relation_features_from_geometry(geom_i: torch.Tensor, geom_j: torch.Tensor) -> torch.Tensor:
    """GPU relation vector from component geometry.

    The output follows RELATION_FEATURE_NAMES used by the existing HKG code:
    dx, dy, dist, area_i, area_j, log_area_ratio, bbox sizes, IoU, contact,
    contain_i_in_j, contain_j_in_i.
    """
    eps = 1e-6
    xi0, yi0, xi1, yi1 = bbox_from_geom(geom_i)
    xj0, yj0, xj1, yj1 = bbox_from_geom(geom_j)
    ub_w = (torch.maximum(xi1, xj1) - torch.minimum(xi0, xj0)).clamp_min(eps)
    ub_h = (torch.maximum(yi1, yj1) - torch.minimum(yi0, yj0)).clamp_min(eps)
    dx = (geom_j[..., 0] - geom_i[..., 0]) / ub_w
    dy = (geom_j[..., 1] - geom_i[..., 1]) / ub_h
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    area_i = geom_i[..., 4].clamp_min(eps)
    area_j = geom_j[..., 4].clamp_min(eps)
    inter_w = (torch.minimum(xi1, xj1) - torch.maximum(xi0, xj0)).clamp_min(0.0)
    inter_h = (torch.minimum(yi1, yj1) - torch.maximum(yi0, yj0)).clamp_min(0.0)
    inter = inter_w * inter_h
    box_i = ((xi1 - xi0).clamp_min(eps) * (yi1 - yi0).clamp_min(eps)).clamp_min(eps)
    box_j = ((xj1 - xj0).clamp_min(eps) * (yj1 - yj0).clamp_min(eps)).clamp_min(eps)
    union = (box_i + box_j - inter).clamp_min(eps)
    iou = (inter / union).clamp(0, 1)
    # Smooth box contact: high when boxes overlap or their normalized gap is tiny.
    gap_x = torch.maximum(torch.maximum(xj0 - xi1, xi0 - xj1), torch.zeros_like(dx)) / ub_w
    gap_y = torch.maximum(torch.maximum(yj0 - yi1, yi0 - yj1), torch.zeros_like(dy)) / ub_h
    contact = torch.exp(-8.0 * torch.sqrt(gap_x * gap_x + gap_y * gap_y + eps)).clamp(0, 1)
    contain_i_in_j = (inter / box_i).clamp(0, 1)
    contain_j_in_i = (inter / box_j).clamp(0, 1)
    out = torch.stack([
        dx,
        dy,
        dist,
        area_i,
        area_j,
        torch.log((area_i + eps) / (area_j + eps)),
        geom_i[..., 2].clamp(0, 1),
        geom_i[..., 3].clamp(0, 1),
        geom_j[..., 2].clamp(0, 1),
        geom_j[..., 3].clamp(0, 1),
        iou,
        contact,
        contain_i_in_j,
        contain_j_in_i,
    ], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def pairwise_relation_features_from_geometry(geom: torch.Tensor) -> torch.Tensor:
    """Return all pairwise relation features [B,N,N,14] from [B,N,6]."""
    gi = geom.unsqueeze(2)
    gj = geom.unsqueeze(1)
    return relation_features_from_geometry(gi, gj)


def slotize_components_by_part_order(
    component_valid: torch.Tensor,
    component_part: torch.Tensor,
    component_geom: torch.Tensor,
    component_token: torch.Tensor,
    *,
    num_parts: int,
    slots_per_part: int,
) -> dict[str, torch.Tensor]:
    """Deterministically convert unordered components to part/rank slots on GPU.

    The rank is left-to-right within each part type.  The grammar still performs
    latent slot-component matching at training/inference time; this function is
    mainly used to build fixed-length layout statistics efficiently.
    """
    bsz, ncomp = component_valid.shape
    device = component_valid.device
    k = int(num_parts)
    m = int(slots_per_part)
    part_ids = torch.arange(k, device=device).view(1, k, 1)
    valid_k = component_valid.bool().unsqueeze(1) & (component_part.long().unsqueeze(1) == part_ids)
    # Sort left-to-right; invalid components go to the end.
    x = component_geom[..., 0].unsqueeze(1).expand(bsz, k, ncomp)
    key = torch.where(valid_k, x, torch.full_like(x, 1.0e6))
    order = torch.argsort(key, dim=-1)[..., :m]
    valid_slots = torch.gather(valid_k, 2, order)
    slot_part = part_ids.expand(bsz, k, m)
    flat_order = order.reshape(bsz, k * m)
    slot_geom = _gather_last(component_geom, flat_order).reshape(bsz, k, m, GEOM_DIM)
    slot_token = _gather_last(component_token, flat_order).reshape(bsz, k, m, component_token.shape[-1])
    return {
        "slot_valid": valid_slots.reshape(bsz, k * m),
        "slot_part": slot_part.reshape(bsz, k * m),
        "slot_geom": slot_geom.reshape(bsz, k * m, GEOM_DIM),
        "slot_token": slot_token.reshape(bsz, k * m, component_token.shape[-1]),
    }


def save_component_cache_shard(path: str | Path, payload: dict[str, torch.Tensor | Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Component extraction runs on GPU, but the cache is stored on CPU to make
    # Stage-2 data loading cheap and device-independent.
    cpu_payload: dict[str, Any] = {}
    for k, v in payload.items():
        cpu_payload[k] = v.detach().cpu() if torch.is_tensor(v) else v
    torch.save(cpu_payload, path)
