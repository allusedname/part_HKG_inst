from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .grammar import GEOM_FEATURE_NAMES, REL_FEATURE_NAMES


@dataclass
class TerminalExtractionConfig:
    threshold: float = 0.40
    min_area_frac: float = 1.0e-4
    min_presence: float = 0.05
    max_components_per_part: int = 4
    max_terminals: int = 32
    mask_size: int = 64


def _connected_components_cpu(binary: torch.Tensor, min_pixels: int) -> list[torch.Tensor]:
    """Dependency-free exact 4-connected components for one 2D boolean mask."""
    b = binary.detach().cpu().bool()
    if b.ndim != 2:
        raise ValueError(f"binary must be [H,W], got {tuple(b.shape)}")
    h, w = b.shape
    seen = torch.zeros_like(b, dtype=torch.bool)
    comps: list[torch.Tensor] = []
    ysxs = torch.nonzero(b, as_tuple=False)
    for yy, xx in ysxs.tolist():
        if seen[yy, xx] or not bool(b[yy, xx]):
            continue
        q: deque[tuple[int, int]] = deque([(int(yy), int(xx))])
        seen[yy, xx] = True
        pix: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if seen[ny, nx] or not bool(b[ny, nx]):
                    continue
                seen[ny, nx] = True
                q.append((ny, nx))
        if len(pix) >= int(min_pixels):
            cm = torch.zeros_like(b, dtype=torch.bool)
            idx = torch.tensor(pix, dtype=torch.long)
            cm[idx[:, 0], idx[:, 1]] = True
            comps.append(cm)
    comps.sort(key=lambda x: int(x.sum().item()), reverse=True)
    return comps


def _geometry_from_mask(mask: torch.Tensor, score_map: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    m = mask.float()
    device = m.device
    h, w = m.shape
    eps = torch.tensor(1e-6, device=device)
    area_pix = m.sum().clamp_min(eps)
    yy = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    cx = (m * xx).sum() / area_pix
    cy = (m * yy).sum() / area_pix
    cols = m.amax(0) > 0
    rows = m.amax(1) > 0
    xgrid = torch.arange(w, device=device, dtype=torch.float32)
    ygrid = torch.arange(h, device=device, dtype=torch.float32)
    minx = torch.where(cols, xgrid, torch.full_like(xgrid, float(w))).min()
    maxx = torch.where(cols, xgrid, torch.zeros_like(xgrid)).max()
    miny = torch.where(rows, ygrid, torch.full_like(ygrid, float(h))).min()
    maxy = torch.where(rows, ygrid, torch.zeros_like(ygrid)).max()
    norm_w = float(max(w - 1, 1))
    norm_h = float(max(h - 1, 1))
    bw = (maxx - minx + 1.0).clamp_min(1.0) / float(max(w, 1))
    bh = (maxy - miny + 1.0).clamp_min(1.0) / float(max(h, 1))
    area = area_pix / float(max(h * w, 1))
    if score_map is None:
        score = torch.ones((), device=device)
    else:
        score = (score_map.float().clamp(0, 1) * m).sum() / area_pix
    geom = torch.stack([
        cx / norm_w,
        cy / norm_h,
        bw.clamp(0, 1),
        bh.clamp(0, 1),
        area.clamp(0, 1),
        score.clamp(0, 1),
    ])
    return torch.nan_to_num(geom.float(), nan=0.0), score.float().clamp(0, 1)


def _pool_token(token_map: torch.Tensor | None, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if token_map is None:
        return torch.nan_to_num(fallback.float())
    if token_map.ndim != 3:
        raise ValueError(f"token_map must be [D,h,w], got {tuple(token_map.shape)}")
    d, th, tw = token_map.shape
    weights = F.interpolate(mask.float()[None, None], size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
    denom = weights.sum().clamp_min(1e-6)
    pooled = (token_map.float() * weights[None]).flatten(1).sum(-1) / denom
    if pooled.shape[-1] != fallback.shape[-1]:
        if pooled.shape[-1] > fallback.shape[-1]:
            pooled = pooled[: fallback.shape[-1]]
        else:
            pooled = F.pad(pooled, (0, fallback.shape[-1] - pooled.shape[-1]))
    return torch.nan_to_num(pooled.float(), nan=0.0)


def average_token_map(stage1_out: dict[str, torch.Tensor], b: int) -> torch.Tensor | None:
    maps: list[torch.Tensor] = []
    for key in ("token_res_map", "token_dino_map"):
        val = stage1_out.get(key)
        if torch.is_tensor(val) and val.ndim == 4:
            maps.append(val[b].float())
    if not maps:
        return None
    hw = maps[0].shape[-2:]
    aligned = []
    for m in maps:
        if m.shape[-2:] != hw:
            m = F.interpolate(m[None], size=hw, mode="bilinear", align_corners=False)[0]
        aligned.append(m)
    return torch.stack(aligned).mean(0)


def empty_terminal_tensors(max_terminals: int, token_dim: int, mask_size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    n = int(max_terminals)
    return {
        "terminal_valid": torch.zeros(n, dtype=torch.bool, device=device),
        "terminal_part": torch.full((n,), -1, dtype=torch.long, device=device),
        "terminal_score": torch.zeros(n, dtype=torch.float32, device=device),
        "terminal_geom": torch.zeros(n, len(GEOM_FEATURE_NAMES), dtype=torch.float32, device=device),
        "terminal_token": torch.zeros(n, token_dim, dtype=torch.float32, device=device),
        "terminal_mask": torch.zeros(n, int(mask_size), int(mask_size), dtype=torch.float32, device=device),
    }


def extract_terminals_from_stage1(
    part_prob: torch.Tensor,
    part_tokens: torch.Tensor,
    *,
    part_presence: torch.Tensor | None = None,
    token_map: torch.Tensor | None = None,
    cfg: TerminalExtractionConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Extract padded terminal proposals from one image's Stage-1 outputs.

    This function is meant for the offline cache/build path.  It uses exact CPU
    connected components for semantic correctness, then returns tensors on the
    same device as ``part_prob``.
    """
    cfg = cfg or TerminalExtractionConfig()
    if part_prob.ndim != 3:
        raise ValueError(f"part_prob must be [K,H,W], got {tuple(part_prob.shape)}")
    if part_tokens.ndim != 2:
        raise ValueError(f"part_tokens must be [K,D], got {tuple(part_tokens.shape)}")
    device = part_prob.device
    k_num, h, w = part_prob.shape
    token_dim = int(part_tokens.shape[-1])
    out = empty_terminal_tensors(cfg.max_terminals, token_dim, cfg.mask_size, device=device)
    min_pix = max(1, int(round(float(cfg.min_area_frac) * float(h * w))))
    prob_cpu = torch.nan_to_num(part_prob.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    pres_cpu = None if part_presence is None else torch.nan_to_num(part_presence.detach().float().cpu(), nan=0.0).clamp(0, 1)
    rows: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for k in range(k_num):
        if pres_cpu is not None and float(pres_cpu[k].item()) < float(cfg.min_presence):
            continue
        binary = prob_cpu[k] > float(cfg.threshold)
        comps = _connected_components_cpu(binary, min_pixels=min_pix)
        for comp_cpu in comps[: max(1, int(cfg.max_components_per_part))]:
            comp = comp_cpu.to(device=device)
            geom, score = _geometry_from_mask(comp, part_prob[k])
            token = _pool_token(token_map, comp, part_tokens[k])
            low_mask = F.interpolate(comp.float()[None, None], size=(cfg.mask_size, cfg.mask_size), mode="nearest")[0, 0]
            # Rank by mass and score; this favors confident large components but
            # still allows multiple repeated parts to survive.
            rank = float(score.item()) * math_safe_sqrt(float(comp.float().mean().item()))
            rows.append((rank, k, score.detach(), geom.detach(), token.detach(), low_mask.detach()))
    rows.sort(key=lambda r: r[0], reverse=True)
    for idx, (_, k, score, geom, token, low_mask) in enumerate(rows[: cfg.max_terminals]):
        out["terminal_valid"][idx] = True
        out["terminal_part"][idx] = int(k)
        out["terminal_score"][idx] = score.to(device=device)
        out["terminal_geom"][idx] = geom.to(device=device)
        out["terminal_token"][idx] = token.to(device=device)
        out["terminal_mask"][idx] = low_mask.to(device=device)
    return out


def math_safe_sqrt(x: float) -> float:
    return float(max(x, 0.0) ** 0.5)


def batch_extract_terminals(
    stage1_out: dict[str, torch.Tensor],
    *,
    cfg: TerminalExtractionConfig,
) -> dict[str, torch.Tensor]:
    part_prob = stage1_out.get("part_prob", torch.sigmoid(stage1_out["part_logits"]))
    part_presence = stage1_out.get("part_presence")
    part_tokens = stage1_out.get("part_tokens", stage1_out.get("part_tokens_res"))
    if part_tokens is None:
        raise KeyError("Stage-1 output must contain part_tokens or part_tokens_res")
    rows: list[dict[str, torch.Tensor]] = []
    for b in range(part_prob.shape[0]):
        rows.append(extract_terminals_from_stage1(
            part_prob[b],
            part_tokens[b],
            part_presence=None if part_presence is None else part_presence[b],
            token_map=average_token_map(stage1_out, b),
            cfg=cfg,
        ))
    return {k: torch.stack([r[k] for r in rows], dim=0) for k in rows[0]}


def terminal_pair_relations(geom: torch.Tensor) -> torch.Tensor:
    """Vectorized geometry-only relation tensor.

    Parameters
    ----------
    geom: ``[B,N,G]`` or ``[N,G]`` with features cx, cy, w, h, area, score.

    Returns
    -------
    ``[B,N,N,R]`` or ``[N,N,R]`` with relation features compatible with
    ``REL_FEATURE_NAMES``.
    """
    squeeze = False
    if geom.ndim == 2:
        geom = geom.unsqueeze(0)
        squeeze = True
    if geom.ndim != 3 or geom.shape[-1] < 5:
        raise ValueError(f"geom must be [B,N,G>=5], got {tuple(geom.shape)}")
    g = torch.nan_to_num(geom.float(), nan=0.0, posinf=1.0, neginf=0.0)
    ci = g[:, :, None, :]
    cj = g[:, None, :, :]
    dx = cj[..., 0] - ci[..., 0]
    dy = cj[..., 1] - ci[..., 1]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    ai = ci[..., 4].clamp_min(1e-6)
    aj = cj[..., 4].clamp_min(1e-6)
    rel = torch.stack([
        dx,
        dy,
        dist,
        ai.expand_as(dx),
        aj.expand_as(dx),
        torch.log(ai.expand_as(dx) / aj.expand_as(dx)).clamp(-8, 8),
        ci[..., 2].expand_as(dx),
        ci[..., 3].expand_as(dx),
        cj[..., 2].expand_as(dx),
        cj[..., 3].expand_as(dx),
    ], dim=-1)
    rel = torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    return rel[0] if squeeze else rel


def save_terminal_cache(records: list[dict[str, torch.Tensor | int]], path: str | Path, *, schema_payload: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "strict_aog_terminal_cache", "records": records, "schema": schema_payload}
    torch.save(payload, path)


def load_terminal_cache(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("kind") != "strict_aog_terminal_cache":
        raise ValueError(f"Expected strict_aog_terminal_cache at {path}")
    return payload
