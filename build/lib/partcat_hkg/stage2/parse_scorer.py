from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.config import Stage2Config
from partcat_hkg.kg.datatypes import HierarchicalKG
from partcat_hkg.models.pooling import masked_pool, topk_presence
from partcat_hkg.utils.numerics import finite_center_clip_logits
from .completion import top_down_completion_score
from .edge_scorer import SelectedEdgeScorer
from .visibility import compute_visibility_states
from .calibration import ScalarCalibratedReadout


class VisibilityAwareParseGraphClassifier(nn.Module):
    """Stage 2 main proposal classifier: score the visible part-role parse graph."""

    def __init__(self, stage1_model, kg: HierarchicalKG, cfg: Stage2Config):
        super().__init__()
        self.stage1 = stage1_model
        self.kg = kg
        self.schema = kg.schema
        self.cfg = cfg
        token_dim = int(stage1_model.cfg.token_dim)
        hidden = int(cfg.hidden_dim)
        self.proj_r = nn.Linear(token_dim, hidden)
        self.proj_d = nn.Linear(token_dim, hidden)
        self.register_buffer("role_index_cf", self.schema.role_index_table.clone().long())
        self.register_buffer("valid_cf", (self.schema.role_index_table >= 0).float())
        self.register_buffer("role_proto_r_raw", kg.role_proto_r.clone())
        self.register_buffer("role_proto_d_raw", kg.role_proto_d.clone())
        self.register_buffer("func_proto_r_raw", kg.func_proto_r.clone())
        self.register_buffer("func_proto_d_raw", kg.func_proto_d.clone())
        self.edge_scorer = SelectedEdgeScorer(kg, cfg)
        self.lambda_completion_raw = nn.Parameter(torch.tensor(float(cfg.lambda_completion_init)))
        self.lambda_edge_raw = nn.Parameter(torch.tensor(float(cfg.lambda_edge_init)))
        self.lambda_contradiction_raw = nn.Parameter(torch.tensor(float(cfg.lambda_contradiction_init)))
        self.base_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.schema.num_classes))
        self.optional_readout = ScalarCalibratedReadout(num_terms=3)

    def _extract(self, batch: dict, detach_stage1: bool = True):
        image = batch["image"].to(next(self.parameters()).device, non_blocking=True)
        if detach_stage1:
            self.stage1.eval()
            with torch.no_grad():
                out = self.stage1(image)
        else:
            out = self.stage1(image)
        part_prob = torch.sigmoid(out["part_logits"])
        role_prob = torch.sigmoid(out["role_logits"])
        func_pres = topk_presence(part_prob, k=self.stage1.cfg.topk_presence_k)
        role_pres = topk_presence(role_prob, k=self.stage1.cfg.topk_presence_k)
        func_r = masked_pool(out["token_res_map"], part_prob)
        func_d = masked_pool(out["token_dino_map"], part_prob)
        role_r = masked_pool(out["token_res_map"], role_prob)
        role_d = masked_pool(out["token_dino_map"], role_prob)
        return out, part_prob, role_prob, func_pres, role_pres, func_r, func_d, role_r, role_d

    def _functional_quality(self, func_r: torch.Tensor, func_d: torch.Tensor) -> torch.Tensor:
        fr = F.normalize(self.proj_r(func_r), dim=-1)
        fd = F.normalize(self.proj_d(func_d), dim=-1)
        pr = F.normalize(self.proj_r(self.func_proto_r_raw.to(fr.device)), dim=-1)
        pd = F.normalize(self.proj_d(self.func_proto_d_raw.to(fd.device)), dim=-1)
        sim = 0.5 * (torch.einsum("bfd,fd->bf", fr, pr) + torch.einsum("bfd,fd->bf", fd, pd))
        return (0.5 + 0.5 * sim).clamp(0, 1)

    def _role_presence_cf(self, role_presence: torch.Tensor) -> torch.Tensor:
        device = role_presence.device
        rid = self.role_index_cf.to(device).clamp_min(0)
        gathered = role_presence.index_select(1, rid.reshape(-1)).reshape(role_presence.shape[0], self.schema.num_classes, self.schema.num_parts)
        return gathered * self.valid_cf.to(device).unsqueeze(0)

    def _visible_role_score(self, role_r, role_d, role_presence_cf, visibility) -> torch.Tensor:
        device = role_r.device
        rid = self.role_index_cf.to(device).clamp_min(0)
        rr = role_r.index_select(1, rid.reshape(-1)).reshape(role_r.shape[0], self.schema.num_classes, self.schema.num_parts, -1)
        rd = role_d.index_select(1, rid.reshape(-1)).reshape(role_d.shape[0], self.schema.num_classes, self.schema.num_parts, -1)
        rr = F.normalize(self.proj_r(rr), dim=-1)
        rd = F.normalize(self.proj_d(rd), dim=-1)
        pr = F.normalize(self.proj_r(self.role_proto_r_raw.to(device)), dim=-1)
        pd = F.normalize(self.proj_d(self.role_proto_d_raw.to(device)), dim=-1)
        sim = 0.5 * ((rr * pr.unsqueeze(0)).sum(-1) + (rd * pd.unsqueeze(0)).sum(-1))
        weight = visibility.visible * visibility.reliability * role_presence_cf
        return (weight * sim).sum(-1)

    def forward(self, batch: dict, *, detach_stage1: bool = True, enable_completion: bool = True, enable_edges: bool = True) -> dict[str, torch.Tensor]:
        out, part_prob, role_prob, func_pres, role_pres, func_r, func_d, role_r, role_d = self._extract(batch, detach_stage1=detach_stage1)
        device = func_pres.device
        func_quality = self._functional_quality(func_r, func_d)
        role_presence_cf = self._role_presence_cf(role_pres)
        visibility = compute_visibility_states(
            role_presence_cf,
            func_pres,
            func_quality,
            self.valid_cf.to(device),
            presence_tau=self.cfg.visibility_presence_tau,
            quality_tau=self.cfg.visibility_quality_tau,
        )
        visible_logits = self._visible_role_score(role_r, role_d, role_presence_cf, visibility)
        completion_logits = top_down_completion_score(
            self.proj_r(func_r), self.proj_d(func_d),
            self.proj_r(self.role_proto_r_raw.to(device)), self.proj_d(self.role_proto_d_raw.to(device)),
            func_pres, func_quality, visibility.unknown, self.valid_cf.to(device), eps=self.cfg.partial_parse_eps,
        ) if enable_completion else torch.zeros_like(visible_logits)
        edge_logits = self.edge_scorer(role_prob, role_pres) if enable_edges else torch.zeros_like(visible_logits)
        contradiction_logits = (visibility.contradictory * role_presence_cf).sum(-1)
        parse_logits = (
            visible_logits
            + F.softplus(self.lambda_completion_raw) * completion_logits
            + F.softplus(self.lambda_edge_raw) * edge_logits
            - F.softplus(self.lambda_contradiction_raw) * contradiction_logits
        )
        hr = F.normalize(self.proj_r(func_r), dim=-1)
        base_token = (func_pres.unsqueeze(-1) * hr).sum(1) / (func_pres.sum(1, keepdim=True) + 1e-6)
        base_logits = self.base_head(base_token)
        readout_logits = self.optional_readout(base_logits, visible_logits, completion_logits, edge_logits)
        main_logits = parse_logits if self.cfg.main_classifier == "parse_graph" else readout_logits
        return {
            "parse_logits": finite_center_clip_logits(parse_logits),
            "visible_logits": finite_center_clip_logits(visible_logits),
            "completion_logits": finite_center_clip_logits(completion_logits),
            "edge_logits": finite_center_clip_logits(edge_logits),
            "contradiction_logits": contradiction_logits,
            "base_logits": finite_center_clip_logits(base_logits),
            "readout_logits": finite_center_clip_logits(readout_logits),
            "logits": finite_center_clip_logits(main_logits),
            "func_pres": func_pres,
            "role_pres": role_pres,
            "part_prob": part_prob,
            "role_prob": role_prob,
            "visibility_state": visibility.state,
            "visible_mask": visibility.visible,
            "unknown_mask": visibility.unknown,
            "contradictory_mask": visibility.contradictory,
        }
