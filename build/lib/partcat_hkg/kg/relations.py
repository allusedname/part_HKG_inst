from __future__ import annotations

import torch
import torch.nn.functional as F

RELATION_FEATURE_NAMES = [
    "dx", "dy", "dist", "area_i", "area_j", "log_area_ratio",
    "bbox_w_i", "bbox_h_i", "bbox_w_j", "bbox_h_j", "iou", "contact",
    "contain_i_in_j", "contain_j_in_i",
]

RELATION_CHANNELS = [
    "above", "below", "lateral", "near", "touching", "overlap", "contain_i", "contain_j",
]


def _mask_stats(mask: torch.Tensor, thr: float = 0.4) -> dict[str, torch.Tensor]:
    """Binary mask geometry in pixel coordinates.

    The returned coordinates are deliberately pixel-space rather than image-
    normalized.  Pairwise relation features are normalized by the union box of
    the two masks below, which is the local object-frame convention used by the
    Stage-2 HKG.
    """
    b = (torch.nan_to_num(mask.float(), nan=0.0, posinf=0.0, neginf=0.0) > thr).float()
    h, w = b.shape[-2:]
    yy = torch.arange(h, device=b.device, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, device=b.device, dtype=torch.float32).view(1, w)
    area_pix = b.sum().clamp_min(1e-6)
    x = (b * xx).sum() / area_pix
    y = (b * yy).sum() / area_pix
    cols = (b.max(dim=0).values > 0).float()
    rows = (b.max(dim=1).values > 0).float()
    xi = torch.arange(w, device=b.device, dtype=torch.float32)
    yi = torch.arange(h, device=b.device, dtype=torch.float32)
    minx = torch.where(cols > 0, xi, torch.full_like(xi, float(w))).min()
    maxx = torch.where(cols > 0, xi, torch.zeros_like(xi)).max()
    miny = torch.where(rows > 0, yi, torch.full_like(yi, float(h))).min()
    maxy = torch.where(rows > 0, yi, torch.zeros_like(yi)).max()
    has = (b.sum() > 0).float()
    # Empty masks keep a harmless degenerate box at the image center.
    cx = torch.tensor(float(max(w - 1, 1)) * 0.5, device=b.device)
    cy = torch.tensor(float(max(h - 1, 1)) * 0.5, device=b.device)
    x = torch.where(has.bool(), x, cx)
    y = torch.where(has.bool(), y, cy)
    minx = torch.where(has.bool(), minx, cx)
    maxx = torch.where(has.bool(), maxx, cx)
    miny = torch.where(has.bool(), miny, cy)
    maxy = torch.where(has.bool(), maxy, cy)
    return {
        "bin": b,
        "area_pix": area_pix,
        "x": x,
        "y": y,
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy,
    }


def relation_attributes_from_masks(mi: torch.Tensor, mj: torch.Tensor, thr: float = 0.4) -> torch.Tensor:
    """Relative relation vector for two masks.

    Geometry is normalized by the union bounding box of the two participating
    parts, not by the full image.  This keeps relation templates local to the
    object/part pair and avoids encoding absolute image position.
    """
    si, sj = _mask_stats(mi, thr), _mask_stats(mj, thr)
    eps = torch.tensor(1e-6, device=mi.device)
    minx = torch.minimum(si["minx"], sj["minx"])
    maxx = torch.maximum(si["maxx"], sj["maxx"])
    miny = torch.minimum(si["miny"], sj["miny"])
    maxy = torch.maximum(si["maxy"], sj["maxy"])
    ub_w = (maxx - minx + 1.0).clamp_min(1.0)
    ub_h = (maxy - miny + 1.0).clamp_min(1.0)
    ub_area = (ub_w * ub_h).clamp_min(1.0)

    dx = (sj["x"] - si["x"]) / ub_w
    dy = (sj["y"] - si["y"]) / ub_h
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    bi, bj = si["bin"], sj["bin"]
    inter = (bi * bj).sum()
    union = ((bi + bj) > 0).float().sum().clamp_min(1e-6)
    iou = inter / union
    dil_i = F.max_pool2d(bi[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    dil_j = F.max_pool2d(bj[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    contact = torch.maximum((dil_i * bj).sum(), (dil_j * bi).sum()) / torch.minimum(si["area_pix"], sj["area_pix"]).clamp_min(1e-6)
    bw_i = (si["maxx"] - si["minx"] + 1.0).clamp_min(0.0) / ub_w
    bh_i = (si["maxy"] - si["miny"] + 1.0).clamp_min(0.0) / ub_h
    bw_j = (sj["maxx"] - sj["minx"] + 1.0).clamp_min(0.0) / ub_w
    bh_j = (sj["maxy"] - sj["miny"] + 1.0).clamp_min(0.0) / ub_h
    area_i = si["area_pix"] / ub_area
    area_j = sj["area_pix"] / ub_area
    vals = [
        dx, dy, dist,
        area_i, area_j, torch.log((area_i + 1e-6) / (area_j + 1e-6)),
        bw_i, bh_i, bw_j, bh_j,
        iou, contact.clamp(0, 1), (inter / si["area_pix"]).clamp(0, 1), (inter / sj["area_pix"]).clamp(0, 1),
    ]
    return torch.nan_to_num(torch.stack(vals), nan=0.0, posinf=0.0, neginf=0.0)


def relation_attributes_vectorized(mi: torch.Tensor, mj: torch.Tensor, thr: float = 0.4) -> torch.Tensor:
    """Vectorized local relation features for [B,E,H,W] endpoint masks."""
    mi = torch.nan_to_num(mi.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mj = torch.nan_to_num(mj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    bi, bj = (mi > thr).float(), (mj > thr).float()
    b, e, h, w = bi.shape
    dev, eps = bi.device, 1e-6
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    si = bi.sum(dim=(-2, -1)).clamp_min(eps)
    sj = bj.sum(dim=(-2, -1)).clamp_min(eps)
    xi = (bi * xx).sum(dim=(-2, -1)) / si
    yi = (bi * yy).sum(dim=(-2, -1)) / si
    xj = (bj * xx).sum(dim=(-2, -1)) / sj
    yj = (bj * yy).sum(dim=(-2, -1)) / sj

    cols_i = bi.amax(dim=-2) > 0
    rows_i = bi.amax(dim=-1) > 0
    cols_j = bj.amax(dim=-2) > 0
    rows_j = bj.amax(dim=-1) > 0
    xgrid = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, w)
    ygrid = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h)
    infx = torch.full((b, e, w), float(w), device=dev)
    infy = torch.full((b, e, h), float(h), device=dev)
    minxi = torch.where(cols_i, xgrid, infx).amin(-1)
    maxxi = torch.where(cols_i, xgrid, torch.zeros_like(infx)).amax(-1)
    minyi = torch.where(rows_i, ygrid, infy).amin(-1)
    maxyi = torch.where(rows_i, ygrid, torch.zeros_like(infy)).amax(-1)
    minxj = torch.where(cols_j, xgrid, infx).amin(-1)
    maxxj = torch.where(cols_j, xgrid, torch.zeros_like(infx)).amax(-1)
    minyj = torch.where(rows_j, ygrid, infy).amin(-1)
    maxyj = torch.where(rows_j, ygrid, torch.zeros_like(infy)).amax(-1)
    has_i = (bi.sum(dim=(-2, -1)) > 0)
    has_j = (bj.sum(dim=(-2, -1)) > 0)
    cx = torch.tensor(float(max(w - 1, 1)) * 0.5, device=dev)
    cy = torch.tensor(float(max(h - 1, 1)) * 0.5, device=dev)
    xi, yi = torch.where(has_i, xi, cx), torch.where(has_i, yi, cy)
    xj, yj = torch.where(has_j, xj, cx), torch.where(has_j, yj, cy)
    minxi, maxxi = torch.where(has_i, minxi, cx), torch.where(has_i, maxxi, cx)
    minyi, maxyi = torch.where(has_i, minyi, cy), torch.where(has_i, maxyi, cy)
    minxj, maxxj = torch.where(has_j, minxj, cx), torch.where(has_j, maxxj, cx)
    minyj, maxyj = torch.where(has_j, minyj, cy), torch.where(has_j, maxyj, cy)

    minx, maxx = torch.minimum(minxi, minxj), torch.maximum(maxxi, maxxj)
    miny, maxy = torch.minimum(minyi, minyj), torch.maximum(maxyi, maxyj)
    ub_w = (maxx - minx + 1.0).clamp_min(1.0)
    ub_h = (maxy - miny + 1.0).clamp_min(1.0)
    ub_area = (ub_w * ub_h).clamp_min(1.0)
    dx, dy = (xj - xi) / ub_w, (yj - yi) / ub_h
    dist = torch.sqrt(dx * dx + dy * dy + eps)
    area_i, area_j = si / ub_area, sj / ub_area
    inter = (bi * bj).sum(dim=(-2, -1))
    union = ((bi + bj) > 0).float().sum(dim=(-2, -1)).clamp_min(eps)
    iou = inter / union
    dil_i = F.max_pool2d(bi.reshape(b * e, 1, h, w), 3, 1, 1).reshape(b, e, h, w)
    dil_j = F.max_pool2d(bj.reshape(b * e, 1, h, w), 3, 1, 1).reshape(b, e, h, w)
    contact = torch.maximum((dil_i * bj).sum(dim=(-2, -1)), (dil_j * bi).sum(dim=(-2, -1))) / torch.minimum(si, sj).clamp_min(eps)
    bw_i = (maxxi - minxi + 1.0).clamp_min(0.0) / ub_w
    bh_i = (maxyi - minyi + 1.0).clamp_min(0.0) / ub_h
    bw_j = (maxxj - minxj + 1.0).clamp_min(0.0) / ub_w
    bh_j = (maxyj - minyj + 1.0).clamp_min(0.0) / ub_h
    gamma = torch.stack([
        dx, dy, dist, area_i, area_j, torch.log((area_i + eps) / (area_j + eps)),
        bw_i, bh_i, bw_j, bh_j, iou, contact.clamp(0, 1), (inter / si).clamp(0, 1), (inter / sj).clamp(0, 1),
    ], dim=-1)
    return torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def relation_channel_strengths(gamma: torch.Tensor) -> torch.Tensor:
    """Map continuous attributes to explicit relation-channel strengths."""
    dx, dy, dist = gamma[..., 0], gamma[..., 1], gamma[..., 2]
    iou, contact = gamma[..., 10], gamma[..., 11]
    contain_i, contain_j = gamma[..., 12], gamma[..., 13]
    above = torch.relu(-dy)
    below = torch.relu(dy)
    lateral = torch.relu(dx.abs() - dy.abs())
    near = torch.exp(-4.0 * dist)
    return torch.stack([above, below, lateral, near, contact, iou, contain_i, contain_j], dim=-1).clamp(0, 1)


def infer_relation_type_name(part_i: str, part_j: str) -> str:
    a, b = str(part_i).lower(), str(part_j).lower()
    pair = {a, b}
    if "body" in pair:
        other = b if a == "body" else a
        if other in {"wheel", "foot", "hand", "leg"}:
            return "support/attached-to-body"
        if other in {"wing", "fin", "tail"}:
            return "lateral-or-appendage"
        if other in {"head", "mirror", "engine", "seat", "mouth", "beak"}:
            return "attached-to-body"
        return "body-context"
    if pair == {"wheel", "mirror"}:
        return "vehicle-subpart-context"
    if "head" in pair and ("tail" in pair or "foot" in pair or "leg" in pair):
        return "axis/extremity"
    return "generic-spatial"
