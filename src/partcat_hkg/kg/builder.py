from __future__ import annotations

from collections import Counter, defaultdict
import math
import torch
import torch.nn.functional as F

from partcat_hkg.config import HKGConfig
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.pooling import masked_pool
from .datatypes import HierarchicalKG
from .relations import RELATION_FEATURE_NAMES, infer_relation_type_name, relation_attributes_from_masks


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


@torch.no_grad()
def build_hkg(stage1_model, loader, schema: RoleSchema, cfg: HKGConfig, *, device: str = "cuda") -> HierarchicalKG:
    """Build functional-role HKG from a trained Stage-1 parser.

    This skeleton follows the notebook's builder: pool functional/role prototypes,
    estimate role priors, keep PMI as diagnostic, and create role-edge Gaussian
    templates from clean GT role masks.
    """
    stage1_model.eval()
    token_dim = int(stage1_model.cfg.token_dim)
    cnum, fnum = schema.num_classes, schema.num_parts
    rdim = len(RELATION_FEATURE_NAMES)

    count_c = torch.zeros(cnum)
    count_cf = torch.zeros(cnum, fnum)
    global_f = torch.zeros(fnum)
    sum_func_r = torch.zeros(fnum, token_dim)
    sum_func_d = torch.zeros(fnum, token_dim)
    cnt_func = torch.zeros(fnum)
    sum_role_r = torch.zeros(cnum, fnum, token_dim)
    sum_role_d = torch.zeros(cnum, fnum, token_dim)
    cnt_role = torch.zeros(cnum, fnum)
    cooc = torch.zeros(fnum, fnum)
    role_rel_vals: dict[tuple[int, int, int], list[torch.Tensor]] = defaultdict(list)
    global_pair_rel_vals: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)

    per_class_seen = Counter()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["obj_label"].to(device)
        if cfg.max_images_per_class:
            keep = []
            for b, y in enumerate(labels.cpu().tolist()):
                if per_class_seen[y] < cfg.max_images_per_class:
                    keep.append(b)
                    per_class_seen[y] += 1
            if not keep:
                continue
            images = images[keep]
            labels = labels[keep]
            batch = {k: (v[keep] if torch.is_tensor(v) and v.shape[0] >= max(keep) + 1 else v) for k, v in batch.items()}

        out = stage1_model(images)
        gt_func_masks = batch["part_masks"].to(device).float()
        gt_role_masks = batch["role_masks"].to(device).float()
        gt_func = batch["presence"].bool()
        gt_role = batch["role_presence"].bool()
        tr = masked_pool(out["token_res_map"], gt_func_masks).cpu()
        td = masked_pool(out["token_dino_map"], gt_func_masks).cpu()
        rr = masked_pool(out["token_res_map"], gt_role_masks).cpu()
        rd = masked_pool(out["token_dino_map"], gt_role_masks).cpu()

        for b in range(images.shape[0]):
            c = int(labels[b].item())
            count_c[c] += 1
            active = gt_func[b].nonzero(as_tuple=False).flatten().tolist()
            valid_active = []
            for k in active:
                count_cf[c, k] += 1
                global_f[k] += 1
                cnt_func[k] += 1
                sum_func_r[k] += tr[b, k]
                sum_func_d[k] += td[b, k]
                rid = schema.role_for(c, k)
                if rid >= 0 and bool(gt_role[b, rid]):
                    valid_active.append(k)
                    cnt_role[c, k] += 1
                    sum_role_r[c, k] += rr[b, rid]
                    sum_role_d[c, k] += rd[b, rid]
            for i in active:
                for j in active:
                    if i < j:
                        cooc[i, j] += 1
                        cooc[j, i] += 1
            for aa in range(len(valid_active)):
                for bb in range(aa + 1, len(valid_active)):
                    i, j = sorted([valid_active[aa], valid_active[bb]])
                    ri, rj = schema.role_for(c, i), schema.role_for(c, j)
                    if ri < 0 or rj < 0:
                        continue
                    gamma = relation_attributes_from_masks(gt_role_masks[b, ri], gt_role_masks[b, rj]).cpu()
                    role_rel_vals[(c, i, j)].append(gamma)
                    global_pair_rel_vals[(i, j)].append(gamma)

    p_cf = (count_cf + 1.0) / (count_c.view(-1, 1) + 2.0)
    p_f = (global_f + 1.0) / (count_c.sum() + 2.0)
    pmi = (p_cf.log() - p_f.view(1, -1).log()).clamp(-5, 5)
    role_prior = p_cf
    func_proto_r = _safe_norm(sum_func_r / cnt_func.clamp_min(1).view(-1, 1))
    func_proto_d = _safe_norm(sum_func_d / cnt_func.clamp_min(1).view(-1, 1))
    role_proto_r = _safe_norm(sum_role_r / cnt_role.clamp_min(1).unsqueeze(-1))
    role_proto_d = _safe_norm(sum_role_d / cnt_role.clamp_min(1).unsqueeze(-1))

    func_edges = set()
    for i in range(fnum):
        vals = cooc[i].clone()
        vals[i] = 0
        if fnum > 1:
            for j in torch.topk(vals, k=min(int(cfg.degree_cap), fnum)).indices.tolist():
                if vals[j] > 0:
                    func_edges.add(tuple(sorted((i, j))))
    func_edges_t = torch.tensor(sorted(func_edges), dtype=torch.long) if func_edges else torch.zeros(0, 2, dtype=torch.long)

    role_edge_rows, means, vars_, supports, igs, gmeans, gvars, types = [], [], [], [], [], [], [], []
    for (c, i, j), vals in sorted(role_rel_vals.items()):
        if len(vals) < int(cfg.role_edge_min_count):
            continue
        V = torch.stack(vals)
        G = torch.stack(global_pair_rel_vals.get((i, j), vals))
        mu_role = torch.nan_to_num(V.mean(0), nan=0.0)
        var_role = torch.nan_to_num(V.var(0), nan=1.0).clamp_min(float(cfg.relation_var_floor))
        mu_global = torch.nan_to_num(G.mean(0), nan=0.0)
        var_global = torch.nan_to_num(G.var(0), nan=1.0).clamp_min(float(cfg.relation_var_floor))
        support = min(1.0, len(vals) / max(float(count_c[c]), 1.0))
        ig = float(support * (((mu_role - mu_global) ** 2) / var_global).mean().clamp(0, 50).item())
        role_edge_rows.append([c, i, j])
        means.append(mu_role)
        vars_.append(var_role)
        supports.append(support)
        igs.append(ig)
        gmeans.append(mu_global)
        gvars.append(var_global)
        types.append(infer_relation_type_name(schema.part_names[i], schema.part_names[j]))

    if role_edge_rows:
        role_edges = torch.tensor(role_edge_rows, dtype=torch.long)
        role_rel_mean = torch.stack(means)
        role_rel_var = torch.stack(vars_)
        role_rel_support = torch.tensor(supports, dtype=torch.float32)
        role_rel_ig = torch.tensor(igs, dtype=torch.float32)
        role_rel_global_mean = torch.stack(gmeans)
        role_rel_global_var = torch.stack(gvars)
    else:
        role_edges = torch.zeros(0, 3, dtype=torch.long)
        role_rel_mean = torch.zeros(0, rdim)
        role_rel_var = torch.ones(0, rdim)
        role_rel_support = torch.zeros(0)
        role_rel_ig = torch.zeros(0)
        role_rel_global_mean = torch.zeros(0, rdim)
        role_rel_global_var = torch.ones(0, rdim)

    return HierarchicalKG(
        schema=schema,
        pmi=pmi,
        func_proto_r=func_proto_r,
        func_proto_d=func_proto_d,
        role_proto_r=role_proto_r,
        role_proto_d=role_proto_d,
        role_prior=role_prior,
        func_edges=func_edges_t,
        role_edges=role_edges,
        role_rel_mean=role_rel_mean,
        role_rel_var=role_rel_var,
        role_rel_support=role_rel_support,
        role_rel_ig=role_rel_ig,
        role_rel_global_mean=role_rel_global_mean,
        role_rel_global_var=role_rel_global_var,
        role_rel_type_names=types,
    )
