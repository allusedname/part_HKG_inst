from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.data.schema import RoleSchema
from .instance_aog import InstanceAOG
from .instance_components import GEOM_DIM, average_stage1_token_maps, extract_instance_components, layout_feature_from_components
from .relations import RELATION_FEATURE_NAMES, relation_attributes_from_masks


@dataclass
class _Record:
    c: int
    comps: dict[str, torch.Tensor]
    layout: torch.Tensor
    assignment: int = -1
    slot_to_component: dict[int, int] = field(default_factory=dict)


@dataclass
class _SlotStat:
    part_type: int
    count: int = 0
    sum_token: torch.Tensor | None = None
    sum_geom: torch.Tensor | None = None
    sum_geom2: torch.Tensor | None = None

    def add(self, token: torch.Tensor, geom: torch.Tensor) -> None:
        token = token.detach().cpu().float()
        geom = geom.detach().cpu().float()
        if self.sum_token is None:
            self.sum_token = torch.zeros_like(token)
            self.sum_geom = torch.zeros_like(geom)
            self.sum_geom2 = torch.zeros_like(geom)
        self.count += 1
        self.sum_token += token
        self.sum_geom += geom
        self.sum_geom2 += geom * geom

    def mean_token(self, token_dim: int) -> torch.Tensor:
        if self.count <= 0 or self.sum_token is None:
            return torch.zeros(token_dim)
        return F.normalize(self.sum_token / float(max(self.count, 1)), dim=0)

    def mean_geom(self) -> torch.Tensor:
        if self.count <= 0 or self.sum_geom is None:
            return torch.zeros(GEOM_DIM)
        return self.sum_geom / float(max(self.count, 1))

    def var_geom(self, floor: float) -> torch.Tensor:
        if self.count <= 1 or self.sum_geom is None or self.sum_geom2 is None:
            return torch.ones(GEOM_DIM) * float(max(floor, 1e-4))
        mean = self.mean_geom()
        var = self.sum_geom2 / float(max(self.count, 1)) - mean * mean
        return torch.nan_to_num(var, nan=float(floor)).clamp_min(float(floor))


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


def _deterministic_kmeans(x: torch.Tensor, k: int, iters: int = 12) -> torch.Tensor:
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if k == 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long)
    centers = [x[0]]
    for _ in range(1, k):
        dist = torch.stack([((x - c) ** 2).sum(-1) for c in centers], dim=0).amin(0)
        centers.append(x[int(dist.argmax().item())])
    centers = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers)
        new_assign = dist.argmin(dim=1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = x[mask].mean(0)
    return assign


def _canonical_slot_order(geoms: torch.Tensor) -> list[int]:
    if geoms.numel() == 0:
        return []
    return sorted(range(geoms.shape[0]), key=lambda i: (float(geoms[i, 0]), float(geoms[i, 1]), -float(geoms[i, 4])))


def _assign_to_centers(geoms: torch.Tensor, centers: torch.Tensor, slot_ids: list[int]) -> dict[int, int]:
    """Greedy one-to-one component -> slot assignment by geometry distance."""
    if geoms.numel() == 0 or centers.numel() == 0:
        return {}
    pairs: list[tuple[float, int, int]] = []
    for ci in range(geoms.shape[0]):
        for sj in range(centers.shape[0]):
            # Downweight score/area relative to position/extent.
            diff = geoms[ci] - centers[sj]
            w = torch.tensor([2.0, 2.0, 0.7, 0.7, 0.5, 0.2], dtype=torch.float32)
            pairs.append((float(((diff.cpu() * w) ** 2).sum().item()), ci, sj))
    pairs.sort(key=lambda z: z[0])
    used_c: set[int] = set()
    used_s: set[int] = set()
    out: dict[int, int] = {}
    for _, ci, sj in pairs:
        if ci in used_c or sj in used_s:
            continue
        used_c.add(ci)
        used_s.add(sj)
        out[int(slot_ids[sj])] = int(ci)
    return out


def _is_anchor_name(name: str) -> bool:
    n = str(name).lower().replace("/", "_").replace("-", "_")
    return n in {"body", "frame", "body_frame", "torso", "head"} or "body" in n or "frame" in n


def _edge_type_name(part_i: str, part_j: str, *, repeated: bool = False, anchor: bool = False) -> str:
    if repeated:
        return f"repeated-{part_i}"
    if anchor:
        return "anchor-star"
    return f"{part_i}-{part_j}"


@torch.no_grad()
def build_instance_aog(stage1_model, loader, schema: RoleSchema, cfg: Any, *, device: str = "cuda") -> InstanceAOG:
    """Build a compact Instance-Slot AOG from a frozen Stage-1 model.

    Compared with the current AOG-HKG builder, this builder treats repeated
    functional parts as unordered connected components.  Template-local latent
    slots are learned by clustering component layouts, and observed components
    are assigned to slots only inside a parse hypothesis.
    """
    dev = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    stage1_model.to(dev)
    stage1_model.eval()
    token_dim = int(stage1_model.cfg.token_dim)
    cnum, fnum = schema.num_classes, schema.num_parts
    anum = max(1, int(_cfg(cfg, "num_templates_per_class", 3)))
    max_per_part = max(1, int(_cfg(cfg, "max_components_per_part", 4)))
    max_total = max(1, int(_cfg(cfg, "max_total_components", 32)))
    component_thr = float(_cfg(cfg, "component_threshold", 0.40))
    min_area = float(_cfg(cfg, "min_component_area_frac", 1.0e-4))
    min_presence = float(_cfg(cfg, "component_min_presence", 0.05))
    use_pred = bool(_cfg(cfg, "use_predicted_stage1_evidence", True))
    max_images_per_class = int(_cfg(cfg, "max_images_per_class", 0) or 0)

    records: list[_Record] = []
    per_class_seen: Counter[int] = Counter()
    for batch in loader:
        labels_all = batch["obj_label"]
        keep: list[int] = []
        for b, y in enumerate(labels_all.cpu().tolist()):
            if max_images_per_class and per_class_seen[int(y)] >= max_images_per_class:
                continue
            keep.append(b)
            per_class_seen[int(y)] += 1
        if not keep:
            continue
        images = batch["image"][keep].to(dev, non_blocking=True)
        labels = batch["obj_label"][keep].to(dev, non_blocking=True)
        gt_masks = batch.get("part_masks")
        gt_presence = batch.get("presence")
        if torch.is_tensor(gt_masks):
            gt_masks = gt_masks[keep].to(dev, non_blocking=True).float()
        if torch.is_tensor(gt_presence):
            gt_presence = gt_presence[keep].to(dev, non_blocking=True).float()
        out = stage1_model(images)
        pred_prob = out.get("part_prob", torch.sigmoid(out["part_logits"])).detach().float()
        pred_presence = out.get("part_presence", None)
        if torch.is_tensor(pred_presence):
            pred_presence = pred_presence.detach().float()
        part_tokens = out.get("part_tokens", out.get("part_tokens_res"))
        if torch.is_tensor(part_tokens):
            part_tokens = part_tokens.detach().float()
        for b in range(images.shape[0]):
            c = int(labels[b].item())
            source = pred_prob[b] if (use_pred or gt_masks is None) else gt_masks[b].float()
            if use_pred and pred_presence is not None:
                presence = pred_presence[b]
            elif torch.is_tensor(gt_presence):
                presence = gt_presence[b]
            elif pred_presence is not None:
                presence = pred_presence[b]
            else:
                presence = None
            token_map = average_stage1_token_maps(out, b)
            comps = extract_instance_components(
                source,
                token_map=token_map,
                part_tokens=None if part_tokens is None else part_tokens[b],
                part_presence=presence,
                threshold=component_thr,
                min_area_frac=min_area,
                max_components_per_part=max_per_part,
                max_total_components=max_total,
                min_presence=min_presence,
            )
            layout = layout_feature_from_components(comps, fnum, slots_per_part=max_per_part)
            records.append(_Record(c=c, comps={k: v.detach().cpu() for k, v in comps.items()}, layout=layout))

    if not records:
        raise RuntimeError("Cannot build InstanceAOG: no records were collected from the loader.")

    # Template assignment: class-specific Or branches over component layouts.
    class_template_counts = torch.zeros(cnum, anum)
    rec_indices_by_class: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        rec_indices_by_class[rec.c].append(idx)
    for c, idxs in rec_indices_by_class.items():
        x = torch.stack([records[i].layout for i in idxs])
        k = min(anum, max(1, int(x.shape[0])))
        ass = _deterministic_kmeans(x, k, iters=int(_cfg(cfg, "template_kmeans_iters", 12)))
        counts = torch.bincount(ass, minlength=k)
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long)
        for new_id, old_id in enumerate(order.tolist()):
            remap[old_id] = new_id
        ass = remap[ass]
        for local, rec_idx in zip(ass.tolist(), idxs):
            records[rec_idx].assignment = int(local)
            class_template_counts[c, int(local)] += 1

    min_template_support = max(1, int(_cfg(cfg, "min_template_support", 2)))
    template_valid = (class_template_counts >= min_template_support).float()
    for c in range(cnum):
        if class_template_counts[c].sum() > 0 and template_valid[c].sum() == 0:
            template_valid[c, int(class_template_counts[c].argmax().item())] = 1.0
    smooth_prior = float(_cfg(cfg, "template_prior_smoothing", 1.0))
    template_prior = (class_template_counts + smooth_prior) / (class_template_counts.sum(-1, keepdim=True) + smooth_prior * anum).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    # Learn template-local slots by clustering same-type components inside each template.
    slot_defs: dict[tuple[int, int], list[_SlotStat]] = defaultdict(list)
    slot_centers: dict[tuple[int, int, int], torch.Tensor] = {}
    for c in range(cnum):
        for a in range(anum):
            idxs = [i for i, r in enumerate(records) if r.c == c and r.assignment == a]
            if not idxs:
                continue
            for k in range(fnum):
                geom_rows: list[torch.Tensor] = []
                for i in idxs:
                    rec = records[i]
                    ids = (rec.comps["part_type"] == k).nonzero(as_tuple=False).flatten()
                    if ids.numel() > 0:
                        geom_rows.extend([rec.comps["geom"][int(ii)] for ii in ids.tolist()])
                if not geom_rows:
                    continue
                geoms = torch.stack(geom_rows).float()
                counts = [int((records[i].comps["part_type"] == k).sum().item()) for i in idxs]
                nslots = max(1, min(max_per_part, max(counts), int(geoms.shape[0])))
                ass = _deterministic_kmeans(geoms, nslots, iters=int(_cfg(cfg, "slot_kmeans_iters", 12)))
                centers = []
                for j in range(nslots):
                    mask = ass == j
                    centers.append(geoms[mask].mean(0) if mask.any() else geoms[0])
                centers_t = torch.stack(centers)
                order = _canonical_slot_order(centers_t)
                ordered_centers = centers_t[order]
                local_slot_ids: list[int] = []
                for center in ordered_centers:
                    sid = len(slot_defs[(c, a)])
                    slot_defs[(c, a)].append(_SlotStat(part_type=k))
                    slot_centers[(c, a, sid)] = center.float()
                    local_slot_ids.append(sid)
                # Assign each record's components of this type to these slots.
                for i in idxs:
                    rec = records[i]
                    ids = (rec.comps["part_type"] == k).nonzero(as_tuple=False).flatten().tolist()
                    if not ids:
                        continue
                    rec_geoms = rec.comps["geom"][ids].float()
                    assigned = _assign_to_centers(rec_geoms, ordered_centers, local_slot_ids)
                    for sid, local_component_idx in assigned.items():
                        ci = ids[int(local_component_idx)]
                        rec.slot_to_component[int(sid)] = int(ci)
                        slot_defs[(c, a)][int(sid)].add(rec.comps["token"][ci], rec.comps["geom"][ci])

    max_slots = max(1, max((len(v) for v in slot_defs.values()), default=1))
    slot_valid = torch.zeros(cnum, anum, max_slots)
    slot_part = torch.full((cnum, anum, max_slots), -1, dtype=torch.long)
    slot_required = torch.zeros(cnum, anum, max_slots)
    slot_prior = torch.zeros(cnum, anum, max_slots)
    slot_proto = torch.zeros(cnum, anum, max_slots, token_dim)
    slot_geom_mean = torch.zeros(cnum, anum, max_slots, GEOM_DIM)
    slot_geom_var = torch.ones(cnum, anum, max_slots, GEOM_DIM) * float(_cfg(cfg, "relation_var_floor", 1e-3))
    presence_smooth = float(_cfg(cfg, "template_presence_smoothing", 1.0))
    required_tau = float(_cfg(cfg, "template_required_tau", 0.45))
    geom_floor = float(_cfg(cfg, "slot_geom_var_floor", 1.0e-3))
    for (c, a), slots in slot_defs.items():
        n_template = float(max(class_template_counts[c, a].item(), 1.0))
        for s, st in enumerate(slots):
            slot_valid[c, a, s] = 1.0
            slot_part[c, a, s] = int(st.part_type)
            prior = (float(st.count) + presence_smooth * 0.05) / (n_template + presence_smooth)
            slot_prior[c, a, s] = float(max(0.0, min(1.0, prior)))
            slot_required[c, a, s] = float(prior >= required_tau)
            slot_proto[c, a, s] = st.mean_token(token_dim)
            slot_geom_mean[c, a, s] = st.mean_geom()
            slot_geom_var[c, a, s] = st.var_geom(geom_floor)

    # Cross-template slot-family IDs: soft diagnostic links, not hard identity.
    slot_family = torch.full((cnum, anum, max_slots), -1, dtype=torch.long)
    family_names: list[str] = []
    family_centers: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    geom_tau = float(_cfg(cfg, "slot_family_geom_tau", 0.18))
    token_tau = float(_cfg(cfg, "slot_family_token_tau", 0.20))
    next_family = 0
    for k in range(fnum):
        entries: list[tuple[int, int, int]] = []
        for c in range(cnum):
            for a in range(anum):
                for s in range(max_slots):
                    if slot_valid[c, a, s] > 0 and int(slot_part[c, a, s].item()) == k:
                        entries.append((c, a, s))
        entries.sort(key=lambda t: (float(slot_geom_mean[t][0]), float(slot_geom_mean[t][1]), t[0], t[1], t[2]))
        for c, a, s in entries:
            g = slot_geom_mean[c, a, s]
            p = _safe_norm(slot_proto[c, a, s])
            chosen = -1
            for fid, rows in family_centers[k]:
                cg = torch.stack([r[0] for r in rows]).mean(0)
                cp = _safe_norm(torch.stack([r[1] for r in rows]).mean(0))
                geom_dist = float(torch.sqrt(((g[:4] - cg[:4]) ** 2).sum()).item())
                token_sim = float((p * cp).sum().item()) if p.abs().sum() > 0 and cp.abs().sum() > 0 else 1.0
                if geom_dist <= geom_tau and token_sim >= token_tau:
                    chosen = fid
                    rows.append((g, p))
                    break
            if chosen < 0:
                chosen = next_family
                next_family += 1
                family_centers[k].append((chosen, [(g, p)]))
                family_names.append(f"{schema.part_names[k]}:family{chosen}")
            slot_family[c, a, s] = int(chosen)

    # Template-local relation edges over assigned component slots.
    edge_rows: list[list[int]] = []
    edge_means: list[torch.Tensor] = []
    edge_vars: list[torch.Tensor] = []
    edge_supports: list[float] = []
    edge_types: list[str] = []
    min_edge_count = int(_cfg(cfg, "role_edge_min_count", 3))
    min_edge_support = float(_cfg(cfg, "template_edge_min_support", 0.12))
    max_edges = int(_cfg(cfg, "template_edge_max_edges", 12))
    var_floor = float(_cfg(cfg, "relation_var_floor", 1.0e-3))
    for c in range(cnum):
        for a in range(anum):
            slots = slot_defs.get((c, a), [])
            if not slots:
                continue
            n_template = float(max(class_template_counts[c, a].item(), 1.0))
            valid_slots = [s for s in range(len(slots)) if slot_prior[c, a, s] >= min_edge_support]
            if not valid_slots:
                continue
            anchors = [s for s in valid_slots if _is_anchor_name(schema.part_names[int(slot_part[c, a, s].item())])]
            if anchors:
                anchor = max(anchors, key=lambda s: float(slot_prior[c, a, s].item()))
            else:
                anchor = max(valid_slots, key=lambda s: float(slot_geom_mean[c, a, s, 4].item()))
            candidates: dict[tuple[int, int], tuple[str, float]] = {}
            for s in valid_slots:
                if s == anchor:
                    continue
                pair = tuple(sorted((anchor, s)))
                candidates[pair] = (_edge_type_name(schema.part_names[int(slot_part[c, a, anchor])], schema.part_names[int(slot_part[c, a, s])], anchor=True), float(slot_prior[c, a, s]))
            by_type: dict[int, list[int]] = defaultdict(list)
            for s in valid_slots:
                by_type[int(slot_part[c, a, s].item())].append(s)
            for k, ss in by_type.items():
                if len(ss) < 2:
                    continue
                for ii in range(len(ss)):
                    for jj in range(ii + 1, len(ss)):
                        pair = tuple(sorted((ss[ii], ss[jj])))
                        candidates[pair] = (_edge_type_name(schema.part_names[k], schema.part_names[k], repeated=True), min(float(slot_prior[c, a, ss[ii]]), float(slot_prior[c, a, ss[jj]])))
            scored_pairs = sorted(candidates.items(), key=lambda z: z[1][1], reverse=True)[:max_edges]
            idxs = [i for i, r in enumerate(records) if r.c == c and r.assignment == a]
            for (si, sj), (etype, _) in scored_pairs:
                vals: list[torch.Tensor] = []
                for ridx in idxs:
                    rec = records[ridx]
                    if si not in rec.slot_to_component or sj not in rec.slot_to_component:
                        continue
                    ci, cj = rec.slot_to_component[si], rec.slot_to_component[sj]
                    mi = rec.comps["mask"][ci]
                    mj = rec.comps["mask"][cj]
                    vals.append(relation_attributes_from_masks(mi, mj).detach().cpu())
                support = float(len(vals)) / max(n_template, 1.0)
                if len(vals) < min_edge_count or support < min_edge_support:
                    continue
                V = torch.stack(vals).float()
                edge_rows.append([c, a, int(si), int(sj)])
                edge_means.append(torch.nan_to_num(V.mean(0), nan=0.0))
                edge_vars.append(torch.nan_to_num(V.var(0, unbiased=False), nan=1.0).clamp_min(var_floor))
                edge_supports.append(float(support))
                edge_types.append(etype)

    rdim = len(RELATION_FEATURE_NAMES)
    if edge_rows:
        edges = torch.tensor(edge_rows, dtype=torch.long)
        edge_rel_mean = torch.stack(edge_means)
        edge_rel_var = torch.stack(edge_vars).clamp_min(var_floor)
        edge_support = torch.tensor(edge_supports, dtype=torch.float32)
    else:
        edges = torch.zeros(0, 4, dtype=torch.long)
        edge_rel_mean = torch.zeros(0, rdim)
        edge_rel_var = torch.ones(0, rdim)
        edge_support = torch.zeros(0)

    return InstanceAOG(
        schema=schema,
        num_templates=anum,
        max_slots=max_slots,
        token_dim=token_dim,
        template_prior=template_prior.float(),
        template_valid=template_valid.float(),
        slot_valid=slot_valid.float(),
        slot_part=slot_part.long(),
        slot_family=slot_family.long(),
        slot_required=slot_required.float(),
        slot_presence_prior=slot_prior.float(),
        slot_proto=slot_proto.float(),
        slot_geom_mean=slot_geom_mean.float(),
        slot_geom_var=slot_geom_var.float(),
        edges=edges,
        edge_rel_mean=edge_rel_mean.float(),
        edge_rel_var=edge_rel_var.float(),
        edge_support=edge_support.float(),
        edge_type_names=edge_types,
        family_names=family_names,
    )
