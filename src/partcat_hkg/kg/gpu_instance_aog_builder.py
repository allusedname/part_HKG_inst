from __future__ import annotations

from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.data.gpu_component_cache import load_component_cache_tensors
from .gpu_instance_aog import GPUInstanceAOG
from .gpu_instance_components import (
    GEOM_DIM,
    RELATION_DIM,
    relation_features_from_geometry,
    slotize_components_by_part_order,
)


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _resolve_device(device: str | torch.device) -> torch.device:
    return torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


def deterministic_kmeans_gpu(x: torch.Tensor, k: int, *, iters: int = 12) -> torch.Tensor:
    """Small deterministic k-means using torch ops on the input device."""
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if k == 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long, device=x.device)
    centers = [x[0]]
    for _ in range(1, k):
        dist = torch.stack([((x - c) ** 2).sum(-1) for c in centers], dim=0).amin(0)
        centers.append(x[int(dist.argmax().item())])
    centers_t = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers_t)
        new_assign = dist.argmin(-1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers_t[j] = x[mask].mean(0)
    return assign


def _is_anchor_name(name: str) -> bool:
    n = str(name).lower().replace("/", "_").replace("-", "_")
    return n in {"body", "frame", "body_frame", "torso", "head"} or "body" in n or "frame" in n


def _compute_template_assignments(labels: torch.Tensor, layout: torch.Tensor, cnum: int, anum: int, *, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = labels.device
    assignments = torch.full((labels.shape[0],), -1, dtype=torch.long, device=device)
    template_counts = torch.zeros(cnum, anum, device=device)
    for c in range(cnum):
        idx = (labels == c).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        k = min(anum, int(idx.numel()))
        ass = deterministic_kmeans_gpu(layout[idx], k, iters=iters)
        counts = torch.bincount(ass, minlength=k).float()
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long, device=device)
        for new_id, old_id in enumerate(order.tolist()):
            remap[int(old_id)] = int(new_id)
        ass = remap[ass]
        assignments[idx] = ass
        template_counts[c, :k].scatter_add_(0, ass, torch.ones_like(ass, dtype=torch.float32))
    return assignments, template_counts


def _relation_stats_for_edge(slot_geom: torch.Tensor, slot_valid: torch.Tensor, record_mask: torch.Tensor, si: int, sj: int, *, var_floor: float) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    mask = record_mask & slot_valid[:, si].bool() & slot_valid[:, sj].bool()
    count = int(mask.sum().item())
    if count <= 0:
        return torch.zeros(RELATION_DIM, device=slot_geom.device), torch.ones(RELATION_DIM, device=slot_geom.device), 0.0, 0
    gamma = relation_features_from_geometry(slot_geom[mask, si], slot_geom[mask, sj])
    mu = torch.nan_to_num(gamma.mean(0), nan=0.0)
    var = torch.nan_to_num(gamma.var(0, unbiased=False), nan=1.0).clamp_min(float(var_floor))
    support = float(count) / float(max(int(record_mask.sum().item()), 1))
    return mu, var, support, count


def build_gpu_instance_aog_from_tensors(
    schema: RoleSchema,
    tensors: dict[str, torch.Tensor],
    cfg: Any,
    *,
    device: str | torch.device = "cuda",
) -> GPUInstanceAOG:
    """Build a GPUInstanceAOG from cached component tensors.

    All heavy tensor work is done on ``device``. The returned grammar is moved to
    CPU for portable serialization.
    """
    dev = _resolve_device(device)
    labels = tensors["obj_label"].to(dev).long()
    component_valid = tensors["component_valid"].to(dev).bool()
    component_part = tensors["component_part"].to(dev).long()
    component_geom = tensors["component_geom"].to(dev).float()
    component_token = tensors["component_token"].to(dev).float()
    part_presence = tensors.get("part_presence")
    part_tokens = tensors.get("part_tokens")
    if part_presence is not None:
        part_presence = part_presence.to(dev).float()
    if part_tokens is not None:
        part_tokens = part_tokens.to(dev).float()

    cnum, fnum = schema.num_classes, schema.num_parts
    anum = max(1, int(_cfg(cfg, "num_templates_per_class", 2)))
    spp = max(1, int(_cfg(cfg, "max_components_per_part", 2)))
    snum = fnum * spp
    token_dim = int(component_token.shape[-1])
    slotized = slotize_components_by_part_order(
        component_valid,
        component_part,
        component_geom,
        component_token,
        num_parts=fnum,
        slots_per_part=spp,
    )
    slot_valid_obs = slotized["slot_valid"].float()
    slot_part_obs = slotized["slot_part"].long()
    slot_geom_obs = slotized["slot_geom"].float()
    slot_token_obs = slotized["slot_token"].float()

    # Layout used by the template Or-node. It is identity-free but view-sensitive.
    layout = torch.cat([slot_valid_obs.unsqueeze(-1), slot_geom_obs], dim=-1).flatten(1)
    assignments, class_template_counts = _compute_template_assignments(
        labels,
        layout,
        cnum,
        anum,
        iters=int(_cfg(cfg, "template_kmeans_iters", 12)),
    )

    min_support = max(1, int(_cfg(cfg, "min_template_support", 2)))
    template_valid = (class_template_counts >= float(min_support)).float()
    for c in range(cnum):
        if class_template_counts[c].sum() > 0 and template_valid[c].sum() == 0:
            template_valid[c, int(class_template_counts[c].argmax().item())] = 1.0
    smooth_prior = float(_cfg(cfg, "template_prior_smoothing", 1.0))
    template_prior = (class_template_counts + smooth_prior) / (class_template_counts.sum(-1, keepdim=True) + smooth_prior * anum).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    slot_part = torch.arange(snum, device=dev, dtype=torch.long).view(1, 1, snum).expand(cnum, anum, snum) // spp
    slot_family = slot_part.clone() * spp + (torch.arange(snum, device=dev).view(1, 1, snum) % spp)
    slot_valid = torch.zeros(cnum, anum, snum, device=dev)
    slot_prior = torch.zeros(cnum, anum, snum, device=dev)
    slot_required = torch.zeros(cnum, anum, snum, device=dev)
    slot_proto = torch.zeros(cnum, anum, snum, token_dim, device=dev)
    slot_geom_mean = torch.zeros(cnum, anum, snum, GEOM_DIM, device=dev)
    slot_geom_var = torch.ones(cnum, anum, snum, GEOM_DIM, device=dev)

    presence_smooth = float(_cfg(cfg, "template_presence_smoothing", 1.0))
    required_tau = float(_cfg(cfg, "template_required_tau", 0.45))
    geom_var_floor = float(_cfg(cfg, "slot_geom_var_floor", 1.0e-3))
    for c in range(cnum):
        for a in range(anum):
            mask = (labels == c) & (assignments == a)
            n = int(mask.sum().item())
            if n == 0:
                continue
            valid_mean = slot_valid_obs[mask].mean(0)
            slot_prior[c, a] = ((slot_valid_obs[mask].sum(0) + presence_smooth * 0.05) / (float(n) + presence_smooth)).clamp(0, 1)
            slot_valid[c, a] = (slot_prior[c, a] > float(_cfg(cfg, "slot_min_prior", 0.03))).float()
            slot_required[c, a] = ((slot_prior[c, a] >= required_tau) & (slot_valid[c, a] > 0)).float()
            # Weighted prototype and geometry statistics.
            w = slot_valid_obs[mask].unsqueeze(-1)
            denom = w.sum(0).clamp_min(1e-6)
            proto = (slot_token_obs[mask] * w).sum(0) / denom
            slot_proto[c, a] = _safe_norm(proto)
            geom_mean = (slot_geom_obs[mask] * w).sum(0) / denom
            slot_geom_mean[c, a] = torch.nan_to_num(geom_mean, nan=0.0)
            centered = (slot_geom_obs[mask] - slot_geom_mean[c, a].unsqueeze(0)) * w
            geom_var = (centered * centered).sum(0) / denom
            slot_geom_var[c, a] = torch.nan_to_num(geom_var, nan=1.0).clamp_min(geom_var_floor)

    edge_rows: list[list[int]] = []
    edge_means: list[torch.Tensor] = []
    edge_vars: list[torch.Tensor] = []
    edge_supports: list[float] = []
    edge_types: list[str] = []
    min_edge_support = float(_cfg(cfg, "template_edge_min_support", 0.12))
    min_edge_count = int(_cfg(cfg, "role_edge_min_count", 3))
    max_edges = int(_cfg(cfg, "template_edge_max_edges", 8))
    var_floor = float(_cfg(cfg, "relation_var_floor", 1.0e-3))
    for c in range(cnum):
        for a in range(anum):
            if template_valid[c, a] <= 0:
                continue
            active = [s for s in range(snum) if float(slot_valid[c, a, s].item()) > 0 and float(slot_prior[c, a, s].item()) >= min_edge_support]
            if not active:
                continue
            anchor_candidates = [s for s in active if _is_anchor_name(schema.part_names[int(slot_part[c, a, s].item())])]
            if anchor_candidates:
                anchor = max(anchor_candidates, key=lambda s: float(slot_prior[c, a, s].item()))
            else:
                anchor = max(active, key=lambda s: float(slot_prior[c, a, s].item() * slot_geom_mean[c, a, s, 4].item()))
            candidate_pairs: dict[tuple[int, int], tuple[str, float]] = {}
            for s in active:
                if s == anchor:
                    continue
                pair = tuple(sorted((anchor, s)))
                candidate_pairs[pair] = ("anchor-star", float(slot_prior[c, a, s].item()))
            # Repeated-part edges: same functional part, different rank slots.
            for k in range(fnum):
                ss = [s for s in active if int(slot_part[c, a, s].item()) == k]
                for ii in range(len(ss)):
                    for jj in range(ii + 1, len(ss)):
                        pair = tuple(sorted((ss[ii], ss[jj])))
                        score = min(float(slot_prior[c, a, ss[ii]].item()), float(slot_prior[c, a, ss[jj]].item()))
                        candidate_pairs[pair] = (f"repeated-{schema.part_names[k]}", score)
            ranked = sorted(candidate_pairs.items(), key=lambda z: z[1][1], reverse=True)[:max_edges]
            record_mask = (labels == c) & (assignments == a)
            for (si, sj), (etype, _) in ranked:
                mu, var, support, count = _relation_stats_for_edge(slot_geom_obs, slot_valid_obs, record_mask, si, sj, var_floor=var_floor)
                if count < min_edge_count or support < min_edge_support:
                    continue
                edge_rows.append([c, a, int(si), int(sj)])
                edge_means.append(mu)
                edge_vars.append(var)
                edge_supports.append(float(support))
                edge_types.append(etype)

    if edge_rows:
        edges = torch.tensor(edge_rows, dtype=torch.long, device=dev)
        edge_rel_mean = torch.stack(edge_means).float()
        edge_rel_var = torch.stack(edge_vars).float().clamp_min(var_floor)
        edge_support = torch.tensor(edge_supports, dtype=torch.float32, device=dev)
    else:
        edges = torch.zeros(0, 4, dtype=torch.long, device=dev)
        edge_rel_mean = torch.zeros(0, RELATION_DIM, device=dev)
        edge_rel_var = torch.ones(0, RELATION_DIM, device=dev)
        edge_support = torch.zeros(0, device=dev)

    family_names = [f"{schema.part_names[k]}:{r}" for k in range(fnum) for r in range(spp)]
    grammar = GPUInstanceAOG(
        schema=schema,
        num_templates=anum,
        max_slots=snum,
        token_dim=token_dim,
        slots_per_part=spp,
        template_prior=template_prior.detach().cpu(),
        template_valid=template_valid.detach().cpu(),
        slot_valid=slot_valid.detach().cpu(),
        slot_part=slot_part.detach().cpu(),
        slot_family=slot_family.detach().cpu(),
        slot_required=slot_required.detach().cpu(),
        slot_presence_prior=slot_prior.detach().cpu(),
        slot_proto=slot_proto.detach().cpu(),
        slot_geom_mean=slot_geom_mean.detach().cpu(),
        slot_geom_var=slot_geom_var.detach().cpu(),
        edges=edges.detach().cpu(),
        edge_rel_mean=edge_rel_mean.detach().cpu(),
        edge_rel_var=edge_rel_var.detach().cpu(),
        edge_support=edge_support.detach().cpu(),
        edge_type_names=edge_types,
        family_names=family_names,
    )
    return grammar


def build_gpu_instance_aog_from_cache(
    cache_dir: str,
    schema: RoleSchema,
    cfg: Any,
    *,
    split: str = "train",
    device: str | torch.device = "cuda",
) -> GPUInstanceAOG:
    tensors = load_component_cache_tensors(cache_dir, split=split, device=device)
    return build_gpu_instance_aog_from_tensors(schema, tensors, cfg, device=device)
