from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from partcat_hkg.config import HKGConfig
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.pooling import masked_pool, topmean_presence
from .datatypes import AOGHierarchicalKG
from .relations import RELATION_FEATURE_NAMES, infer_relation_type_name, relation_attributes_from_masks, relation_channel_strengths


@dataclass
class _EvidenceRecord:
    c: int
    presence: torch.Tensor       # [F]
    cluster_feat: torch.Tensor   # [3F] = presence, area, centroid xy flattened
    func_r: torch.Tensor         # [F,D]
    func_d: torch.Tensor         # [F,D]
    role_r: torch.Tensor         # [F,D], indexed by functional part for this class
    role_d: torch.Tensor         # [F,D]
    quality: torch.Tensor        # [F] prototype/relation quality weight
    relations: dict[tuple[int, int], torch.Tensor]


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


def _mask_cluster_features(masks: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """Return [3F] vector: presence, area, x/y centroids for GT masks."""
    fnum, h, w = masks.shape
    dev = masks.device
    yy = torch.linspace(0, 1, h, device=dev).view(1, h, 1)
    xx = torch.linspace(0, 1, w, device=dev).view(1, 1, w)
    m = masks.float().clamp(0, 1)
    area_pix = m.flatten(1).sum(-1).clamp_min(1e-6)
    area = area_pix / float(max(h * w, 1))
    cx = (m * xx).flatten(1).sum(-1) / area_pix
    cy = (m * yy).flatten(1).sum(-1) / area_pix
    pres = presence.float().clamp(0, 1)
    area = area * pres
    cx = torch.where(pres > 0, cx, torch.zeros_like(cx))
    cy = torch.where(pres > 0, cy, torch.zeros_like(cy))
    return torch.cat([pres, area, cx * pres, cy * pres], dim=0).detach().cpu()


def _prototype_quality_from_prediction(part_prob: torch.Tensor, pred_presence: torch.Tensor, gt_presence: torch.Tensor) -> torch.Tensor:
    """Quality weights for HKG prototype statistics from predicted masks.

    The HKG should not average tokens from very weak predicted masks.  This
    weight is nonzero only for GT-present parts, then softly favors high
    predicted presence and high top-mean mask confidence.  A floor is kept so
    valid small parts can still contribute while learning improves.
    """
    top = topmean_presence(part_prob.float().clamp(0, 1).unsqueeze(0), q=0.02).squeeze(0)
    pred = pred_presence.float().clamp(0, 1)
    gt = gt_presence.float().clamp(0, 1)
    top_score = (top / 0.20).clamp(0, 1)
    q = gt * (0.25 + 0.75 * pred) * (0.50 + 0.50 * top_score)
    return q.detach().cpu().clamp(0, 1)


def _deterministic_kmeans(x: torch.Tensor, k: int, iters: int = 12) -> torch.Tensor:
    """Small deterministic CPU k-means for class alternatives.

    Returns assignment indices in [0,k).  If there are fewer distinct rows than k,
    some clusters may be empty; the builder will mark them invalid.
    """
    n = int(x.shape[0])
    k = max(1, min(int(k), n))
    x = torch.nan_to_num(x.float(), nan=0.0)
    if k == 1 or n <= 1:
        return torch.zeros(n, dtype=torch.long)
    # Farthest-first initialization from the first row, deterministic and robust.
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


def _is_anchor_name(name: str) -> bool:
    n = str(name).lower().replace("/", "_").replace("-", "_")
    return n in {"body", "frame", "body_frame", "torso", "head"} or "body" in n or "frame" in n


def _is_appendage_name(name: str) -> bool:
    n = str(name).lower().replace("/", "_").replace("-", "_")
    return any(tok in n for tok in ["wheel", "wing", "tail", "leg", "foot", "fin", "engine", "mirror", "beak", "horn", "handle", "seat"])


def _motif_type(part_i: str, part_j: str, rel_mean: torch.Tensor) -> int:
    """Map pair statistics into a small reusable motif vocabulary.

    The first debug run over-promoted almost every edge to a motif.  The next
    conservative run removed too much: useful structures such as car
    body--wheel and aeroplane body--wing often became ordinary edges with no
    motif factor.  HKG-v2 therefore keeps motifs sparse, but explicitly allows
    central-body/appendage motifs when the relation is stable enough.
    """
    a, b = str(part_i).lower(), str(part_j).lower()
    ch = relation_channel_strengths(rel_mean)
    contain = float(torch.maximum(ch[6], ch[7]).item())
    lateral = float(ch[2].item())
    near = float(ch[3].item())
    contact = float(ch[4].item())
    vertical = float(max(ch[0].item(), ch[1].item()))
    if contain > 0.20:
        return 2  # containment
    if (_is_anchor_name(a) or _is_anchor_name(b)) and max(near, contact) > 0.25:
        return 1  # genuine attachment-like hub relation
    if ((_is_anchor_name(a) and _is_appendage_name(b)) or (_is_anchor_name(b) and _is_appendage_name(a))) and max(near, contact, vertical, lateral) > 0.18:
        return 4  # body/frame-to-appendage motif, e.g. body-wheel or body-wing
    if lateral > 0.35:
        return 3  # lateral / axis / symmetry-like
    return 0  # generic pair; keep as an edge, not as a motif


def _fallback_proto(template: torch.Tensor, counts: torch.Tensor, class_proto: torch.Tensor, global_proto: torch.Tensor) -> torch.Tensor:
    out = template.clone()
    cnum, anum, fnum = counts.shape
    for c in range(cnum):
        for a in range(anum):
            for f in range(fnum):
                if counts[c, a, f] > 0:
                    continue
                if torch.isfinite(class_proto[c, f]).all() and class_proto[c, f].abs().sum() > 0:
                    out[c, a, f] = class_proto[c, f]
                elif torch.isfinite(global_proto[f]).all():
                    out[c, a, f] = global_proto[f]
    return out


@torch.no_grad()
def build_aog_hkg(stage1_model, loader, schema: RoleSchema, cfg: HKGConfig, *, device: str = "cuda") -> AOGHierarchicalKG:
    """Build the AOG-inspired HKG from a trained Stage-1 checkpoint.

    The builder uses GT masks from the Stage-1 training loader for stable grammar
    statistics, while using frozen Stage-1 feature maps to pool appearance
    prototypes.  This matches the current Stage-1 training contract: Stage 1 is
    trained on functional masks and object-aware role masks, then frozen before
    Stage 2.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    stage1_model.to(device)
    stage1_model.eval()
    token_dim = int(stage1_model.cfg.token_dim)
    cnum, fnum = schema.num_classes, schema.num_parts
    anum = max(1, int(cfg.num_templates_per_class))
    rdim = len(RELATION_FEATURE_NAMES)

    records: list[_EvidenceRecord] = []
    per_class_seen = Counter()
    count_c = torch.zeros(cnum)
    count_cf = torch.zeros(cnum, fnum)
    global_f = torch.zeros(fnum)

    sum_func_r = torch.zeros(fnum, token_dim)
    sum_func_d = torch.zeros(fnum, token_dim)
    cnt_func = torch.zeros(fnum)
    sum_class_r = torch.zeros(cnum, fnum, token_dim)
    sum_class_d = torch.zeros(cnum, fnum, token_dim)
    cnt_class = torch.zeros(cnum, fnum)

    global_pair_rel_vals: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)

    for batch in loader:
        labels_all = batch["obj_label"]
        keep: list[int] = []
        for b, y in enumerate(labels_all.cpu().tolist()):
            if cfg.max_images_per_class and per_class_seen[int(y)] >= int(cfg.max_images_per_class):
                continue
            keep.append(b)
            per_class_seen[int(y)] += 1
        if not keep:
            continue
        images = batch["image"][keep].to(device, non_blocking=True)
        labels = batch["obj_label"][keep].to(device, non_blocking=True)
        func_masks = batch["part_masks"][keep].to(device, non_blocking=True).float()
        role_masks = batch["role_masks"][keep].to(device, non_blocking=True).float()
        presence = batch["presence"][keep].to(device, non_blocking=True).float()
        role_presence = batch["role_presence"][keep].to(device, non_blocking=True).float()

        out = stage1_model(images)
        use_pred = bool(getattr(cfg, "use_predicted_stage1_evidence", True))

        # Use the same Stage-1 evidence path that Stage 2 will see at runtime.
        # The earlier builder pooled tokens and relation geometry from GT masks,
        # while inference scored predicted masks/tokens.  That mismatch makes
        # relation templates look clean offline but weak in parse visualizations.
        part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"]))
        role_prob = out.get("role_prob", torch.sigmoid(out["role_logits"]))
        pred_part_presence = out.get("part_presence", presence)
        pred_role_presence = out.get("role_presence", role_presence)

        if use_pred:
            quality_all = torch.stack([
                _prototype_quality_from_prediction(part_prob[b], pred_part_presence[b], presence[b])
                for b in range(images.shape[0])
            ], dim=0)
        else:
            quality_all = presence.detach().cpu().float().clamp(0, 1)

        if use_pred and "part_tokens_res" in out:
            func_r = out.get("part_tokens_res", out.get("part_tokens")).detach().cpu()
            func_d = out.get("part_tokens_dino", out.get("part_tokens", out.get("part_tokens_res"))).detach().cpu()
            role_r_all = out.get("role_tokens_res", out.get("role_tokens", masked_pool(out["token_res_map"], role_masks))).detach().cpu()
            role_d_all = out.get("role_tokens_dino", out.get("role_tokens", role_r_all.to(device) if torch.is_tensor(role_r_all) else masked_pool(out["token_dino_map"], role_masks))).detach().cpu()
        else:
            func_r = masked_pool(out["token_res_map"], func_masks).detach().cpu()
            func_d = masked_pool(out["token_dino_map"], func_masks).detach().cpu()
            role_r_all = masked_pool(out["token_res_map"], role_masks).detach().cpu()
            role_d_all = masked_pool(out["token_dino_map"], role_masks).detach().cpu()

        for b in range(images.shape[0]):
            c = int(labels[b].item())
            count_c[c] += 1
            pres_b = presence[b].detach().cpu().float()
            rels: dict[tuple[int, int], torch.Tensor] = {}
            role_r = torch.zeros(fnum, token_dim)
            role_d = torch.zeros(fnum, token_dim)
            active_parts = pres_b.nonzero(as_tuple=False).flatten().tolist()
            for k in active_parts:
                qk = float(quality_all[b, k].item())
                global_f[k] += 1
                count_cf[c, k] += 1
                cnt_func[k] += max(qk, 1e-4)
                sum_func_r[k] += qk * func_r[b, k]
                sum_func_d[k] += qk * func_d[b, k]
                rid = schema.role_for(c, k)
                if rid >= 0 and bool(pred_role_presence[b, rid].item() > 0.5):
                    cnt_class[c, k] += max(qk, 1e-4)
                    role_r[k] = role_r_all[b, rid]
                    role_d[k] = role_d_all[b, rid]
                    sum_class_r[c, k] += qk * role_r[k]
                    sum_class_d[c, k] += qk * role_d[k]
            # If a role-specific token was unavailable, fall back to the functional token.
            for k in active_parts:
                if role_r[k].abs().sum() == 0:
                    role_r[k] = func_r[b, k]
                    role_d[k] = func_d[b, k]
            for aa in range(len(active_parts)):
                for bb in range(aa + 1, len(active_parts)):
                    i, j = sorted([int(active_parts[aa]), int(active_parts[bb])])
                    ri, rj = schema.role_for(c, i), schema.role_for(c, j)
                    if ri < 0 or rj < 0:
                        continue
                    if use_pred:
                        gamma = relation_attributes_from_masks(role_prob[b, ri], role_prob[b, rj]).detach().cpu()
                    else:
                        gamma = relation_attributes_from_masks(role_masks[b, ri], role_masks[b, rj]).detach().cpu()
                    rels[(i, j)] = gamma
                    global_pair_rel_vals[(i, j)].append(gamma)
            records.append(_EvidenceRecord(
                c=c,
                presence=pres_b,
                cluster_feat=_mask_cluster_features((part_prob[b] if use_pred else func_masks[b]).detach(), presence[b].detach()),
                func_r=func_r[b],
                func_d=func_d[b],
                role_r=role_r,
                role_d=role_d,
                quality=quality_all[b],
                relations=rels,
            ))

    if not records:
        raise RuntimeError("Cannot build AOG-HKG: no training records were collected from the loader.")

    p_cf = (count_cf + 1.0) / (count_c.view(-1, 1) + 2.0)
    p_f = (global_f + 1.0) / (count_c.sum() + 2.0)
    pmi = (p_cf.log() - p_f.view(1, -1).log()).clamp(-5, 5)
    role_prior = p_cf.clamp(0, 1)
    func_proto_r = _safe_norm(sum_func_r / cnt_func.clamp_min(1).view(-1, 1))
    func_proto_d = _safe_norm(sum_func_d / cnt_func.clamp_min(1).view(-1, 1))
    class_role_proto_r = _safe_norm(sum_class_r / cnt_class.clamp_min(1).unsqueeze(-1))
    class_role_proto_d = _safe_norm(sum_class_d / cnt_class.clamp_min(1).unsqueeze(-1))

    # Class-specific template assignment: Or-node branch selection learned from
    # part visibility and coarse part geometry.
    assignments = torch.full((len(records),), -1, dtype=torch.long)
    class_template_counts = torch.zeros(cnum, anum)
    rec_indices_by_class: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        rec_indices_by_class[rec.c].append(idx)
    for c, idxs in rec_indices_by_class.items():
        x = torch.stack([records[i].cluster_feat for i in idxs])
        k = min(anum, max(1, int(x.shape[0])))
        ass = _deterministic_kmeans(x, k, iters=int(cfg.template_kmeans_iters))
        # Reorder clusters by descending support for stable template IDs.
        counts = torch.bincount(ass, minlength=k)
        order = torch.argsort(counts, descending=True)
        remap = torch.zeros(k, dtype=torch.long)
        for new_id, old_id in enumerate(order.tolist()):
            remap[old_id] = new_id
        ass = remap[ass]
        for local, rec_idx in zip(ass.tolist(), idxs):
            assignments[rec_idx] = int(local)
            class_template_counts[c, int(local)] += 1

    template_valid = (class_template_counts >= max(1, int(cfg.min_template_support))).float()
    # Always keep the first non-empty template valid so classes with few samples
    # still have a usable parse branch.
    for c in range(cnum):
        if class_template_counts[c].sum() > 0 and template_valid[c].sum() == 0:
            template_valid[c, int(class_template_counts[c].argmax().item())] = 1.0

    smooth_prior = float(cfg.template_prior_smoothing)
    template_prior = (class_template_counts + smooth_prior) / (class_template_counts.sum(-1, keepdim=True) + smooth_prior * anum).clamp_min(1e-6)
    template_prior = template_prior * template_valid
    template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-6)

    # Template node prototypes and role visibility priors.
    t_count = torch.zeros(cnum, anum, fnum)
    t_proto_count = torch.zeros(cnum, anum, fnum)
    t_sum_r = torch.zeros(cnum, anum, fnum, token_dim)
    t_sum_d = torch.zeros(cnum, anum, fnum, token_dim)
    template_pair_vals: dict[tuple[int, int, int, int], list[torch.Tensor]] = defaultdict(list)
    for idx, rec in enumerate(records):
        a = int(assignments[idx].item())
        if a < 0:
            continue
        c = rec.c
        active = rec.presence.nonzero(as_tuple=False).flatten().tolist()
        for k in active:
            qk = float(rec.quality[k].item())
            t_count[c, a, k] += 1
            t_proto_count[c, a, k] += max(qk, 1e-4)
            t_sum_r[c, a, k] += qk * rec.role_r[k]
            t_sum_d[c, a, k] += qk * rec.role_d[k]
        for (i, j), gamma in rec.relations.items():
            template_pair_vals[(c, a, i, j)].append(gamma)

    t_role_prior = (t_count + float(cfg.template_presence_smoothing) * role_prior.unsqueeze(1)) / (
        class_template_counts.unsqueeze(-1) + float(cfg.template_presence_smoothing)
    ).clamp_min(1e-6)
    t_role_prior = t_role_prior.clamp(0, 1) * template_valid.unsqueeze(-1)
    t_required = ((t_role_prior >= float(cfg.template_required_tau)) & (template_valid.unsqueeze(-1) > 0)).float()
    t_proto_r = _safe_norm(t_sum_r / t_proto_count.clamp_min(1e-4).unsqueeze(-1))
    t_proto_d = _safe_norm(t_sum_d / t_proto_count.clamp_min(1e-4).unsqueeze(-1))
    t_proto_r = _fallback_proto(t_proto_r, t_proto_count, class_role_proto_r, func_proto_r)
    t_proto_d = _fallback_proto(t_proto_d, t_proto_count, class_role_proto_d, func_proto_d)

    edge_rows: list[list[int]] = []
    means: list[torch.Tensor] = []
    vars_: list[torch.Tensor] = []
    gmeans: list[torch.Tensor] = []
    gvars: list[torch.Tensor] = []
    supports: list[float] = []
    igs: list[float] = []
    type_names: list[str] = []
    motif_rows: list[list[int]] = []
    motif_supports: list[float] = []

    candidates_by_template: dict[tuple[int, int], list[tuple[float, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, str, bool]]] = defaultdict(list)
    for (c, a, i, j), vals in sorted(template_pair_vals.items()):
        n_template = float(class_template_counts[c, a].item())
        if n_template <= 0:
            continue
        support = len(vals) / max(n_template, 1.0)
        if len(vals) < int(cfg.role_edge_min_count) or support < float(cfg.template_edge_min_support):
            continue
        V = torch.stack(vals)
        G = torch.stack(global_pair_rel_vals.get((i, j), vals))
        obs_mu = torch.nan_to_num(V.mean(0), nan=0.0)
        obs_var = torch.nan_to_num(V.var(0, unbiased=False), nan=1.0).clamp_min(float(cfg.relation_var_floor))
        g_mu = torch.nan_to_num(G.mean(0), nan=0.0)
        g_var = torch.nan_to_num(G.var(0, unbiased=False), nan=1.0).clamp_min(float(cfg.relation_var_floor))
        eta = float(len(vals)) / (float(len(vals)) + float(cfg.template_edge_shrink_kappa))
        mu = eta * obs_mu + (1.0 - eta) * g_mu
        var = eta * obs_var + (1.0 - eta) * g_var
        ig = float(support * (((mu - g_mu) ** 2) / g_var).mean().clamp(0, 50).item())
        # Keep relations by information gain, but rescue stable anchor edges.
        # The diagnostic run showed that strict IG-only pursuit made several
        # templates too thin (e.g. car often had only one relation).  AOG-style
        # configurations still need central bonds from body/frame to reliable
        # appendages, even when the relation is common globally and therefore
        # has modest class-vs-global IG.
        min_ig = float(getattr(cfg, "edge_min_information_gain", 0.0))
        anchor_pair = (_is_anchor_name(schema.part_names[i]) or _is_anchor_name(schema.part_names[j]))
        both_role_prior_ok = bool(
            t_role_prior[c, a, i] >= float(getattr(cfg, "anchor_edge_required_prior", 0.25))
            and t_role_prior[c, a, j] >= float(getattr(cfg, "anchor_edge_required_prior", 0.25))
        )
        anchor_keep = bool(
            anchor_pair
            and both_role_prior_ok
            and support >= float(getattr(cfg, "anchor_edge_min_support", 0.18))
        )
        if ig < min_ig and not anchor_keep:
            continue
        score = ig + 0.25 * support + (0.15 if anchor_keep else 0.0)
        rtype = infer_relation_type_name(schema.part_names[i], schema.part_names[j])
        if anchor_keep and rtype == "generic-spatial":
            rtype = "anchor-bond"
        candidates_by_template[(c, a)].append((score, i, j, mu, var, g_mu, g_var, support, ig, rtype, anchor_keep))

    for (c, a), cand in candidates_by_template.items():
        cand = sorted(cand, key=lambda z: z[0], reverse=True)
        deg = Counter()
        kept = 0
        anchor_kept = 0
        for score, i, j, mu, var, g_mu, g_var, support, ig, rtype, was_anchor_keep in cand:
            if kept >= int(cfg.template_edge_max_edges):
                break
            if deg[i] >= int(cfg.template_edge_degree_cap) or deg[j] >= int(cfg.template_edge_degree_cap):
                continue
            if was_anchor_keep and anchor_kept >= int(getattr(cfg, "anchor_edge_max_per_template", 6)):
                continue
            eidx = len(edge_rows)
            edge_rows.append([c, a, i, j])
            means.append(mu)
            vars_.append(var)
            gmeans.append(g_mu)
            gvars.append(g_var)
            supports.append(float(support))
            igs.append(float(ig))
            type_names.append(rtype)
            deg[i] += 1
            deg[j] += 1
            kept += 1
            if was_anchor_keep:
                anchor_kept += 1
            if support >= float(cfg.motif_min_support):
                mtype = _motif_type(schema.part_names[i], schema.part_names[j], mu)
                # Only structural motifs are retained.  Generic high-support
                # pairs remain template edges but are not promoted to motif
                # factors or motif visualizations.
                if int(mtype) != 0:
                    motif_rows.append([c, a, i, j, int(mtype)])
                    motif_supports.append(float(support))

    if edge_rows:
        template_edges = torch.tensor(edge_rows, dtype=torch.long)
        template_rel_mean = torch.stack(means)
        template_rel_var = torch.stack(vars_).clamp_min(float(cfg.relation_var_floor))
        template_rel_global_mean = torch.stack(gmeans)
        template_rel_global_var = torch.stack(gvars).clamp_min(float(cfg.relation_var_floor))
        template_rel_support = torch.tensor(supports, dtype=torch.float32)
        template_rel_ig = torch.tensor(igs, dtype=torch.float32)
    else:
        template_edges = torch.zeros(0, 4, dtype=torch.long)
        template_rel_mean = torch.zeros(0, rdim)
        template_rel_var = torch.ones(0, rdim)
        template_rel_global_mean = torch.zeros(0, rdim)
        template_rel_global_var = torch.ones(0, rdim)
        template_rel_support = torch.zeros(0)
        template_rel_ig = torch.zeros(0)
    motif_edges = torch.tensor(motif_rows, dtype=torch.long) if motif_rows else torch.zeros(0, 5, dtype=torch.long)
    motif_support = torch.tensor(motif_supports, dtype=torch.float32) if motif_supports else torch.zeros(0)

    return AOGHierarchicalKG(
        schema=schema,
        num_templates=anum,
        pmi=pmi,
        role_prior=role_prior,
        func_proto_r=func_proto_r,
        func_proto_d=func_proto_d,
        class_role_proto_r=class_role_proto_r,
        class_role_proto_d=class_role_proto_d,
        template_prior=template_prior,
        template_valid=template_valid,
        template_role_prior=t_role_prior,
        template_role_required=t_required,
        template_role_proto_r=t_proto_r,
        template_role_proto_d=t_proto_d,
        template_edges=template_edges,
        template_rel_mean=template_rel_mean,
        template_rel_var=template_rel_var,
        template_rel_global_mean=template_rel_global_mean,
        template_rel_global_var=template_rel_global_var,
        template_rel_support=template_rel_support,
        template_rel_ig=template_rel_ig,
        template_rel_type_names=type_names,
        motif_edges=motif_edges,
        motif_support=motif_support,
    )
