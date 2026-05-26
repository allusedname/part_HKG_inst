from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.config import Stage2Config
from partcat_hkg.kg.datatypes import HierarchicalKG
from partcat_hkg.kg.relations import relation_attributes_from_masks


class SelectedEdgeScorer(nn.Module):
    """Selected explicit relation factor scorer.

    It scores role-edge observations by class-vs-global Gaussian log-likelihood
    ratio and keeps only the strongest reliable edges per candidate class.
    """

    def __init__(self, kg: HierarchicalKG, cfg: Stage2Config, mask_thr: float = 0.4):
        super().__init__()
        self.kg = kg
        self.cfg = cfg
        self.mask_thr = float(mask_thr)
        self.register_buffer("role_edges", kg.role_edges.clone())
        self.register_buffer("role_rel_mean", kg.role_rel_mean.clone())
        self.register_buffer("role_rel_var", kg.role_rel_var.clone())
        self.register_buffer("role_rel_support", kg.role_rel_support.clone())
        self.register_buffer("role_rel_ig", kg.role_rel_ig.clone())
        self.register_buffer("role_rel_global_mean", kg.role_rel_global_mean.clone())
        self.register_buffer("role_rel_global_var", kg.role_rel_global_var.clone())

    @staticmethod
    def gaussian_ll(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.clamp_min(1e-3)
        return -0.5 * (((x - mu) ** 2 / var) + var.log()).mean(-1)

    def forward(self, role_prob: torch.Tensor, role_presence: torch.Tensor, *, return_rows: bool = False):
        bsz = role_prob.shape[0]
        cnum = self.kg.schema.num_classes
        device = role_prob.device
        logits = torch.zeros(bsz, cnum, device=device)
        rows = []
        if self.role_edges.numel() == 0:
            return (logits, rows) if return_rows else logits
        candidates = [[[] for _ in range(cnum)] for _ in range(bsz)]
        role_index = self.kg.schema.role_index_table.to(device)
        for e in range(int(self.role_edges.shape[0])):
            c, i, j = [int(x) for x in self.role_edges[e].tolist()]
            if float(self.role_rel_ig[e].detach().cpu()) < float(getattr(self.cfg, "edge_min_information_gain", 0.0)):
                continue
            ri, rj = int(role_index[c, i].item()), int(role_index[c, j].item())
            if ri < 0 or rj < 0:
                continue
            rel = torch.stack([relation_attributes_from_masks(role_prob[b, ri], role_prob[b, rj], thr=self.mask_thr) for b in range(bsz)]).to(device)
            ll_c = self.gaussian_ll(rel, self.role_rel_mean[e].to(device), self.role_rel_var[e].to(device))
            ll_g = self.gaussian_ll(rel, self.role_rel_global_mean[e].to(device), self.role_rel_global_var[e].to(device))
            llr = (ll_c - ll_g).clamp(-8, 8)
            if self.cfg.edge_positive_only:
                llr_used = F.relu(llr)
            else:
                llr_used = llr
            strength = torch.sqrt((role_presence[:, ri] * role_presence[:, rj]).clamp_min(0.0) + 1e-8)
            contrib = strength * self.role_rel_support[e].to(device) * (self.role_rel_ig[e].to(device) / (self.role_rel_ig[e].to(device) + 1.0)).clamp(0, 1) * llr_used
            for b in range(bsz):
                if float(strength[b]) < 0.03:
                    continue
                candidates[b][c].append((contrib[b], e))
        topm = 2
        for b in range(bsz):
            for c in range(cnum):
                items = candidates[b][c]
                if not items:
                    continue
                chosen = sorted(items, key=lambda x: float(x[0].detach().cpu()), reverse=True)[:topm]
                vals = torch.stack([x[0] for x in chosen])
                logits[b, c] = vals.sum() / math.sqrt(float(len(chosen)))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)
        return (logits, rows) if return_rows else logits
