from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


GEOM_FEATURE_NAMES = ["cx", "cy", "bbox_w", "bbox_h", "area", "score"]
GEOM_DIM = len(GEOM_FEATURE_NAMES)


@dataclass
class ComponentExtractionConfig:
    """Runtime knobs for converting part masks into unordered part instances."""

    threshold: float = 0.40
    min_area_frac: float = 1.0e-4
    max_components_per_part: int = 4
    max_total_components: int = 32
    min_presence: float = 0.05


def _empty_components(device: torch.device, token_dim: int, height: int, width: int) -> dict[str, torch.Tensor]:
    return {
        "part_type": torch.zeros(0, dtype=torch.long, device=device),
        "presence": torch.zeros(0, dtype=torch.float32, device=device),
        "geom": torch.zeros(0, GEOM_DIM, dtype=torch.float32, device=device),
        "bbox": torch.zeros(0, 4, dtype=torch.float32, device=device),
        "mask": torch.zeros(0, height, width, dtype=torch.float32, device=device),
        "token": torch.zeros(0, token_dim, dtype=torch.float32, device=device),
    }


def _connected_components(binary: torch.Tensor, *, min_pixels: int = 1) -> list[torch.Tensor]:
    """Return 4-connected component masks for a CPU boolean image.

    The implementation is deliberately dependency-free.  It is intended for the
    IS-AOG parsing/builder path where component extraction is part of discrete
    parse inference rather than a differentiable segmentation layer.
    """
    b = binary.detach().cpu().bool()
    if b.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {tuple(b.shape)}")
    height, width = b.shape
    visited = torch.zeros_like(b, dtype=torch.bool)
    comps: list[torch.Tensor] = []
    coords = torch.nonzero(b, as_tuple=False)
    for yy, xx in coords.tolist():
        if visited[yy, xx] or not bool(b[yy, xx]):
            continue
        q: deque[tuple[int, int]] = deque([(int(yy), int(xx))])
        visited[yy, xx] = True
        pix: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not bool(b[ny, nx]):
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))
        if len(pix) >= int(min_pixels):
            cm = torch.zeros_like(b, dtype=torch.bool)
            ys = torch.tensor([p[0] for p in pix], dtype=torch.long)
            xs = torch.tensor([p[1] for p in pix], dtype=torch.long)
            cm[ys, xs] = True
            comps.append(cm)
    comps.sort(key=lambda m: int(m.sum().item()), reverse=True)
    return comps


def _mask_geometry(mask: torch.Tensor, score_map: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized geometry, bbox, and scalar confidence for a component."""
    m = mask.float()
    device = m.device
    height, width = m.shape
    eps = torch.tensor(1e-6, device=device)
    area_pix = m.sum().clamp_min(eps)
    yy = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    cx_pix = (m * xx).sum() / area_pix
    cy_pix = (m * yy).sum() / area_pix
    cols = m.amax(dim=0) > 0
    rows = m.amax(dim=1) > 0
    xgrid = torch.arange(width, device=device, dtype=torch.float32)
    ygrid = torch.arange(height, device=device, dtype=torch.float32)
    minx = torch.where(cols, xgrid, torch.full_like(xgrid, float(width))).min()
    maxx = torch.where(cols, xgrid, torch.zeros_like(xgrid)).max()
    miny = torch.where(rows, ygrid, torch.full_like(ygrid, float(height))).min()
    maxy = torch.where(rows, ygrid, torch.zeros_like(ygrid)).max()
    norm_w = float(max(width - 1, 1))
    norm_h = float(max(height - 1, 1))
    bbox_w = (maxx - minx + 1.0).clamp_min(1.0) / float(max(width, 1))
    bbox_h = (maxy - miny + 1.0).clamp_min(1.0) / float(max(height, 1))
    area = area_pix / float(max(height * width, 1))
    if score_map is None:
        score = torch.ones((), dtype=torch.float32, device=device)
    else:
        score = (score_map.float().clamp(0, 1) * m).sum() / area_pix
    geom = torch.stack([
        cx_pix / norm_w,
        cy_pix / norm_h,
        bbox_w.clamp(0, 1),
        bbox_h.clamp(0, 1),
        area.clamp(0, 1),
        score.clamp(0, 1),
    ])
    bbox = torch.stack([
        minx / norm_w,
        miny / norm_h,
        maxx / norm_w,
        maxy / norm_h,
    ]).clamp(0, 1)
    return torch.nan_to_num(geom), torch.nan_to_num(bbox), score.clamp(0, 1)


def _pool_component_token(
    token_map: torch.Tensor | None,
    component_mask: torch.Tensor,
    fallback_token: torch.Tensor | None,
    token_dim: int,
) -> torch.Tensor:
    """Pool a token for a component; fall back to the part token when needed."""
    if token_map is None:
        if fallback_token is None:
            return component_mask.new_zeros(token_dim)
        return fallback_token.float()
    if token_map.ndim != 3:
        raise ValueError(f"Expected token_map [D,h,w], got shape {tuple(token_map.shape)}")
    dim, h, w = token_map.shape
    weights = F.interpolate(
        component_mask.float().view(1, 1, *component_mask.shape),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    denom = weights.sum().clamp_min(1e-6)
    token = (token_map.float() * weights.unsqueeze(0)).flatten(1).sum(-1) / denom
    if not torch.isfinite(token).all() and fallback_token is not None:
        token = fallback_token.float()
    if dim != token_dim:
        if dim > token_dim:
            token = token[:token_dim]
        else:
            token = F.pad(token, (0, token_dim - dim))
    return torch.nan_to_num(token)


def extract_instance_components(
    part_prob: torch.Tensor,
    *,
    token_map: torch.Tensor | None = None,
    part_tokens: torch.Tensor | None = None,
    part_presence: torch.Tensor | None = None,
    threshold: float = 0.40,
    min_area_frac: float = 1.0e-4,
    max_components_per_part: int = 4,
    max_total_components: int = 32,
    min_presence: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Split functional part masks into unordered component terminals.

    Parameters
    ----------
    part_prob:
        Tensor with shape ``[K,H,W]``.  It may be a probability map or a binary
        mask.  Each connected component above ``threshold`` becomes a terminal.
    token_map:
        Optional feature map ``[D,h,w]``.  If supplied, component tokens are
        pooled from the exact component support.  Otherwise the functional part
        token is reused as a fallback.
    part_tokens:
        Optional tensor ``[K,D]`` used as a fallback token source.
    part_presence:
        Optional tensor ``[K]`` used to gate extremely weak part channels.

    Returns
    -------
    A dict containing padded-free tensors over N detected components:
    ``part_type [N]``, ``presence [N]``, ``geom [N,6]``, ``bbox [N,4]``,
    ``mask [N,H,W]`` and ``token [N,D]``.
    """
    if part_prob.ndim != 3:
        raise ValueError(f"Expected part_prob [K,H,W], got shape {tuple(part_prob.shape)}")
    device = part_prob.device
    k_num, height, width = part_prob.shape
    token_dim = 0
    if token_map is not None:
        token_dim = int(token_map.shape[0])
    elif part_tokens is not None:
        token_dim = int(part_tokens.shape[-1])
    if token_dim <= 0:
        token_dim = 1
    min_pixels = max(1, int(round(float(min_area_frac) * float(height * width))))
    rows: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prob_cpu = torch.nan_to_num(part_prob.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    presence_cpu = None if part_presence is None else torch.nan_to_num(part_presence.detach().float().cpu(), nan=0.0).clamp(0, 1)
    for k in range(k_num):
        if presence_cpu is not None and float(presence_cpu[k].item()) < float(min_presence):
            continue
        binary = prob_cpu[k] > float(threshold)
        comps = _connected_components(binary, min_pixels=min_pixels)
        for comp_cpu in comps[: max(1, int(max_components_per_part))]:
            comp = comp_cpu.to(device=device)
            score_map = part_prob[k].detach().float().clamp(0, 1)
            geom, bbox, score = _mask_geometry(comp, score_map=score_map)
            fallback = None if part_tokens is None else part_tokens[k]
            token = _pool_component_token(token_map, comp, fallback, token_dim)
            rank = float(score.item()) * float(comp.float().mean().item())
            rows.append((rank, k, score, geom, bbox, comp.float(), token.float()))
    if not rows:
        return _empty_components(device, token_dim, height, width)
    rows.sort(key=lambda z: z[0], reverse=True)
    rows = rows[: max(1, int(max_total_components))]
    part_type = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    presence = torch.stack([r[2].to(device=device) for r in rows]).float()
    geom = torch.stack([r[3].to(device=device) for r in rows]).float()
    bbox = torch.stack([r[4].to(device=device) for r in rows]).float()
    mask = torch.stack([r[5].to(device=device) for r in rows]).float()
    token = torch.stack([r[6].to(device=device) for r in rows]).float()
    return {
        "part_type": part_type,
        "presence": presence,
        "geom": geom,
        "bbox": bbox,
        "mask": mask,
        "token": token,
    }


def average_stage1_token_maps(out: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor | None:
    """Return the average of available Stage-1 token maps for one image."""
    maps: list[torch.Tensor] = []
    for key in ("token_res_map", "token_dino_map"):
        t = out.get(key)
        if torch.is_tensor(t) and t.ndim == 4:
            maps.append(t[batch_index].float())
    if not maps:
        return None
    if len(maps) == 1:
        return maps[0]
    base_hw = maps[0].shape[-2:]
    aligned = [maps[0]]
    for m in maps[1:]:
        if m.shape[-2:] != base_hw:
            aligned.append(F.interpolate(m.unsqueeze(0), size=base_hw, mode="bilinear", align_corners=False)[0])
        else:
            aligned.append(m)
    return torch.stack(aligned, dim=0).mean(0)


def layout_feature_from_components(
    comps: dict[str, torch.Tensor],
    num_parts: int,
    *,
    slots_per_part: int = 4,
) -> torch.Tensor:
    """Fixed-length layout feature for template clustering.

    For each functional part type, keep up to ``slots_per_part`` components in
    deterministic left-to-right/top-to-bottom/large-first order and concatenate
    ``presence + geometry``.  This gives k-means a view-sensitive but identity-free
    layout descriptor.
    """
    pt = comps["part_type"].detach().cpu().long()
    geom = comps["geom"].detach().cpu().float()
    presence = comps["presence"].detach().cpu().float()
    width = int(slots_per_part)
    feat = torch.zeros(num_parts, width, 1 + GEOM_DIM)
    for k in range(int(num_parts)):
        idx = (pt == k).nonzero(as_tuple=False).flatten().tolist()
        idx = sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(geom[i, 4])))
        for j, ii in enumerate(idx[:width]):
            feat[k, j, 0] = presence[ii]
            feat[k, j, 1:] = geom[ii]
    return feat.reshape(-1)
