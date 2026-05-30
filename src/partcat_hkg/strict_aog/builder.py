from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .grammar import GEOM_FEATURE_NAMES, REL_FEATURE_NAMES, StrictAOGGrammar, save_strict_aog
from .terminals import load_terminal_cache, terminal_pair_relations


@dataclass
class StrictAOGBuildConfig:
    num_templates_per_class: int = 3
    max_slots_per_template: int = 12
    max_slots_per_part: int = 4
    kmeans_iters: int = 20
    min_template_support: int = 2
    required_tau: float = 0.45
    min_slot_support: float = 0.08
    min_edge_support: float = 0.12
    min_edge_count: int = 3
    max_edges_per_template: int = 12
    relation_var_floor: float = 1e-3
    geom_var_floor: float = 1e-3
    template_prior_smoothing: float = 1.0
    slot_prior_smoothing: float = 1.0


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


def _deterministic_kmeans(x: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if n <= 1 or k <= 1:
        return torch.zeros(n, dtype=torch.long)
    centers = [x[0]]
    for _ in range(1, k):
        dist = torch.stack([((x - c) ** 2).sum(-1) for c in centers], dim=0).amin(0)
        centers.append(x[int(dist.argmax().item())])
    centers = torch.stack(centers)
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        dist = torch.cdist(x, centers)
        new_assign = dist.argmin(-1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(0)
    return assign


def _record_layout(rec: dict[str, Any], num_parts: int, max_slots_per_part: int) -> torch.Tensor:
    valid = rec["terminal_valid"].bool()
    part = rec["terminal_part"].long()
    geom = rec["terminal_geom"].float()
    feat = torch.zeros(num_parts, max_slots_per_part, 1 + len(GEOM_FEATURE_NAMES))
    for k in range(num_parts):
        idx = ((part == k) & valid).nonzero(as_tuple=False).flatten().tolist()
        idx = sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(geom[i, 4])))
        for j, ii in enumerate(idx[:max_slots_per_part]):
            feat[k, j, 0] = 1.0
            feat[k, j, 1:] = geom[ii]
    return feat.reshape(-1)


def _is_anchor(name: str) -> bool:
    n = str(name).lower().replace("-", "_").replace("/", "_")
    return n in {"body", "frame", "body_frame", "torso", "head"} or "body" in n or "frame" in n


def _sort_component_indices(rec: dict[str, Any], k: int) -> list[int]:
    valid = rec["terminal_valid"].bool()
    part = rec["terminal_part"].long()
    geom = rec["terminal_geom"].float()
    idx = ((part == k) & valid).nonzero(as_tuple=False).flatten().tolist()
    return sorted(idx, key=lambda i: (float(geom[i, 0]), float(geom[i, 1]), -float(geom[i, 4])))


@dataclass
class _SlotStat:
    part: int
    count: int = 0
    token_sum: torch.Tensor | None = None
    geom_sum: torch.Tensor | None = None
    geom2_sum: torch.Tensor | None = None

    def add(self, token: torch.Tensor, geom: torch.Tensor) -> None:
        token = token.float().cpu()
        geom = geom.float().cpu()
        if self.token_sum is None:
            self.token_sum = torch.zeros_like(token)
            self.geom_sum = torch.zeros_like(geom)
            self.geom2_sum = torch.zeros_like(geom)
        self.count += 1
        self.token_sum += token
        self.geom_sum += geom
        self.geom2_sum += geom * geom

    def token_mean(self, token_dim: int) -> torch.Tensor:
        if self.count <= 0 or self.token_sum is None:
            return torch.zeros(token_dim)
        return _safe_norm(self.token_sum / float(self.count))

    def geom_mean(self) -> torch.Tensor:
        if self.count <= 0 or self.geom_sum is None:
            return torch.zeros(len(GEOM_FEATURE_NAMES))
        return self.geom_sum / float(self.count)

    def geom_var(self, floor: float) -> torch.Tensor:
        if self.count <= 1 or self.geom_sum is None or self.geom2_sum is None:
            return torch.ones(len(GEOM_FEATURE_NAMES)) * float(floor)
        mu = self.geom_mean()
        var = self.geom2_sum / float(self.count) - mu * mu
        return torch.nan_to_num(var, nan=float(floor)).clamp_min(float(floor))


@dataclass
class _TemplateRecords:
    records: list[int] = field(default_factory=list)
    slots: list[_SlotStat] = field(default_factory=list)
    # record index -> slot -> terminal index
    assignments: dict[int, dict[int, int]] = field(default_factory=dict)


def _schema_names(schema: Any, num_classes: int, num_parts: int) -> tuple[list[str], list[str]]:
    return (
        list(getattr(schema, "obj_names", [str(i) for i in range(num_classes)])),
        list(getattr(schema, "part_names", [str(i) for i in range(num_parts)])),
    )


def build_strict_aog_from_records(
    records: list[dict[str, Any]],
    *,
    schema: Any,
    token_dim: int,
    num_parts: int,
    cfg: StrictAOGBuildConfig | None = None,
) -> StrictAOGGrammar:
    cfg = cfg or StrictAOGBuildConfig()
    if not records:
        raise ValueError("No terminal records were provided to build_strict_aog_from_records")
    labels = torch.tensor([int(r["obj_label"]) for r in records], dtype=torch.long)
    num_classes = int(max(int(labels.max().item()) + 1, len(getattr(schema, "obj_names", [])) or 0))
    class_names, part_names = _schema_names(schema, num_classes, num_parts)
    A = max(1, int(cfg.num_templates_per_class))
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_prior = (class_counts + 1.0) / (class_counts.sum() + float(num_classes))

    # 1. Template Or-branch assignment per class using layout descriptors.
    layouts = [_record_layout(r, num_parts, cfg.max_slots_per_part) for r in records]
    assignment = torch.full((len(records),), -1, dtype=torch.long)
    template_counts = torch.zeros(num_classes, A)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        by_class[int(c)].append(i)
    for c, idxs in by_class.items():
        x = torch.stack([layouts[i] for i in idxs])
        k = min(A, max(1, int(x.shape[0])))
        ass = _deterministic_kmeans(x, k, iters=cfg.kmeans_iters)
        counts = torch.bincount(ass, minlength=k)
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long)
        for new, old in enumerate(order.tolist()):
            remap[old] = new
        ass = remap[ass]
        for local, ri in zip(ass.tolist(), idxs):
            assignment[ri] = int(local)
            template_counts[c, int(local)] += 1

    template_valid = (template_counts >= max(1, int(cfg.min_template_support))).float()
    for c in range(num_classes):
        if template_counts[c].sum() > 0 and template_valid[c].sum() == 0:
            template_valid[c, int(template_counts[c].argmax().item())] = 1.0
    smooth = float(cfg.template_prior_smoothing)
    template_prior = (template_counts + smooth) / (template_counts.sum(-1, keepdim=True) + smooth * A).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    # 2. Create template-local slots. Slots are latent roles; repeated parts
    #    become multiple slots ordered by canonical component position.
    templates: dict[tuple[int, int], _TemplateRecords] = defaultdict(_TemplateRecords)
    for i, rec in enumerate(records):
        a = int(assignment[i].item())
        if a >= 0:
            templates[(int(labels[i].item()), a)].records.append(i)

    for (c, a), tr in templates.items():
        idxs = tr.records
        for k in range(num_parts):
            counts = [len(_sort_component_indices(records[i], k)) for i in idxs]
            if not counts or max(counts) <= 0:
                continue
            nslots = min(max(counts), int(cfg.max_slots_per_part), max(1, int(cfg.max_slots_per_template) - len(tr.slots)))
            if nslots <= 0:
                break
            local_slot_ids = []
            for _ in range(nslots):
                sid = len(tr.slots)
                tr.slots.append(_SlotStat(part=k))
                local_slot_ids.append(sid)
            # Assign j-th left-to-right component to j-th slot for this part.
            for ri in idxs:
                comp_ids = _sort_component_indices(records[ri], k)
                if ri not in tr.assignments:
                    tr.assignments[ri] = {}
                for j, cid in enumerate(comp_ids[:nslots]):
                    sid = local_slot_ids[j]
                    tr.assignments[ri][sid] = int(cid)
                    tr.slots[sid].add(records[ri]["terminal_token"][cid], records[ri]["terminal_geom"][cid])

    max_slots = max(1, max((len(tr.slots) for tr in templates.values()), default=1))
    max_slots = min(max_slots, int(cfg.max_slots_per_template))
    slot_valid = torch.zeros(num_classes, A, max_slots)
    slot_part = torch.full((num_classes, A, max_slots), -1, dtype=torch.long)
    slot_required = torch.zeros(num_classes, A, max_slots)
    slot_presence = torch.zeros(num_classes, A, max_slots)
    slot_proto = torch.zeros(num_classes, A, max_slots, token_dim)
    slot_geom_mean = torch.zeros(num_classes, A, max_slots, len(GEOM_FEATURE_NAMES))
    slot_geom_var = torch.ones(num_classes, A, max_slots, len(GEOM_FEATURE_NAMES)) * float(cfg.geom_var_floor)

    for (c, a), tr in templates.items():
        ntemp = float(max(len(tr.records), 1))
        for s, st in enumerate(tr.slots[:max_slots]):
            prior = (float(st.count) + float(cfg.slot_prior_smoothing) * 0.05) / (ntemp + float(cfg.slot_prior_smoothing))
            if prior < float(cfg.min_slot_support):
                continue
            slot_valid[c, a, s] = 1.0
            slot_part[c, a, s] = int(st.part)
            slot_presence[c, a, s] = float(max(0.0, min(1.0, prior)))
            slot_required[c, a, s] = float(prior >= float(cfg.required_tau))
            slot_proto[c, a, s] = st.token_mean(token_dim)
            slot_geom_mean[c, a, s] = st.geom_mean()
            slot_geom_var[c, a, s] = st.geom_var(cfg.geom_var_floor)

    # 3. Horizontal relation factors R for each And-production.
    edge_rows: list[list[int]] = []
    edge_types: list[int] = []
    edge_supports: list[float] = []
    edge_means: list[torch.Tensor] = []
    edge_vars: list[torch.Tensor] = []
    for (c, a), tr in templates.items():
        valid_slots = [s for s in range(min(len(tr.slots), max_slots)) if slot_valid[c, a, s] > 0]
        if len(valid_slots) < 2:
            continue
        anchors = [s for s in valid_slots if _is_anchor(part_names[int(slot_part[c, a, s].item())])]
        if anchors:
            anchor = max(anchors, key=lambda s: float(slot_presence[c, a, s].item()))
        else:
            anchor = max(valid_slots, key=lambda s: float(slot_geom_mean[c, a, s, 4].item()))
        candidates: dict[tuple[int, int], int] = {}
        for s in valid_slots:
            if s != anchor:
                candidates[tuple(sorted((anchor, s)))] = 0
        by_part: dict[int, list[int]] = defaultdict(list)
        for s in valid_slots:
            by_part[int(slot_part[c, a, s].item())].append(s)
        for _, ss in by_part.items():
            if len(ss) >= 2:
                for ii in range(len(ss)):
                    for jj in range(ii + 1, len(ss)):
                        candidates[tuple(sorted((ss[ii], ss[jj])))] = 1
        # Rank by joint presence. This is a simple pursuit of stable relations.
        ranked = sorted(candidates.items(), key=lambda kv: min(float(slot_presence[c, a, kv[0][0]]), float(slot_presence[c, a, kv[0][1]])), reverse=True)
        for (si, sj), etype in ranked[: int(cfg.max_edges_per_template)]:
            vals: list[torch.Tensor] = []
            for ri in tr.records:
                ass = tr.assignments.get(ri, {})
                if si not in ass or sj not in ass:
                    continue
                g = records[ri]["terminal_geom"].unsqueeze(0)
                rel = terminal_pair_relations(g)[0, ass[si], ass[sj]].detach().cpu()
                vals.append(rel)
            support = float(len(vals)) / float(max(len(tr.records), 1))
            if len(vals) < int(cfg.min_edge_count) or support < float(cfg.min_edge_support):
                continue
            V = torch.stack(vals).float()
            edge_rows.append([c, a, int(si), int(sj)])
            edge_types.append(int(etype))
            edge_supports.append(float(support))
            edge_means.append(torch.nan_to_num(V.mean(0), nan=0.0))
            edge_vars.append(torch.nan_to_num(V.var(0, unbiased=False), nan=1.0).clamp_min(float(cfg.relation_var_floor)))

    if edge_rows:
        edges = torch.tensor(edge_rows, dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_support = torch.tensor(edge_supports, dtype=torch.float32)
        edge_rel_mean = torch.stack(edge_means).float()
        edge_rel_var = torch.stack(edge_vars).float().clamp_min(float(cfg.relation_var_floor))
    else:
        edges = torch.zeros(0, 4, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        edge_support = torch.zeros(0)
        edge_rel_mean = torch.zeros(0, len(REL_FEATURE_NAMES))
        edge_rel_var = torch.ones(0, len(REL_FEATURE_NAMES))

    return StrictAOGGrammar(
        schema=schema,
        token_dim=int(token_dim),
        num_classes=num_classes,
        num_templates=A,
        max_slots=max_slots,
        class_prior=class_prior.float(),
        template_prior=template_prior.float(),
        template_valid=template_valid.float(),
        slot_valid=slot_valid.float(),
        slot_part=slot_part.long(),
        slot_required=slot_required.float(),
        slot_presence=slot_presence.float(),
        slot_proto=slot_proto.float(),
        slot_geom_mean=slot_geom_mean.float(),
        slot_geom_var=slot_geom_var.float(),
        edges=edges,
        edge_type=edge_type,
        edge_support=edge_support,
        edge_rel_mean=edge_rel_mean,
        edge_rel_var=edge_rel_var,
        part_names=part_names,
        class_names=class_names,
    )


def build_strict_aog_from_cache(
    cache_path: str | Path,
    *,
    schema: Any,
    cfg: StrictAOGBuildConfig | None = None,
) -> StrictAOGGrammar:
    payload = load_terminal_cache(cache_path, map_location="cpu")
    records = payload["records"]
    if not records:
        raise ValueError(f"No records in terminal cache: {cache_path}")
    token_dim = int(records[0]["terminal_token"].shape[-1])
    num_parts = int(max(int(r["terminal_part"].max().item()) for r in records if torch.is_tensor(r["terminal_part"])) + 1)
    return build_strict_aog_from_records(records, schema=schema, token_dim=token_dim, num_parts=num_parts, cfg=cfg)


def _records_from_batches(loader: Iterable[dict[str, torch.Tensor]], *, max_batches: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        B = int(batch["terminal_valid"].shape[0])
        for b in range(B):
            rec = {k: v[b].detach().cpu() for k, v in batch.items() if k.startswith("terminal_")}
            rec["obj_label"] = int(batch["obj_label"][b].detach().cpu().item())
            records.append(rec)
    return records


def save_builder_output(grammar: StrictAOGGrammar, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_strict_aog(grammar, str(out_path))
