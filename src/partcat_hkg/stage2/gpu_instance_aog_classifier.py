from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.kg.gpu_instance_aog import GPUInstanceAOG
from partcat_hkg.kg.gpu_instance_components import GEOM_DIM, relation_features_from_geometry
from partcat_hkg.utils.numerics import finite_center_clip_logits


class GPUInstanceAOGStage2Classifier(nn.Module):
    """Fully batched GPU Stage-2 scorer for cached Instance-Slot AOG tensors.

    This model does not run Stage 1 and does not extract connected components in
    the training forward pass.  It expects a cached batch produced by
    ``scripts/cache_gpu_instance_components.py``:

      component_valid    [B,N]
      component_part     [B,N]
      component_presence [B,N]
      component_geom     [B,N,6]
      component_token    [B,N,D]
      part_presence      [B,K]
      part_tokens        [B,K,D]

    All slot-component compatibility, approximate matching, relation scoring,
    template aggregation, and fusion are vectorized torch operations on GPU.
    """

    def __init__(self, grammar: GPUInstanceAOG, cfg: Any):
        super().__init__()
        self.grammar = grammar
        self.kg = grammar  # compatibility with old trainer/checkpoint metadata
        self.schema = grammar.schema
        self.cfg = cfg
        token_dim = int(grammar.token_dim)
        hidden = int(getattr(cfg, "hidden_dim", 256))
        cnum = grammar.schema.num_classes
        self.num_classes = cnum
        self.num_templates = int(grammar.num_templates)
        self.max_slots = int(grammar.max_slots)

        self.proj = nn.Linear(token_dim, hidden)
        self.base_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, cnum),
        )

        self.register_buffer("template_prior", grammar.template_prior.clone().float())
        self.register_buffer("template_valid", grammar.template_valid.clone().float())
        self.register_buffer("slot_valid", grammar.slot_valid.clone().float())
        self.register_buffer("slot_part", grammar.slot_part.clone().long())
        self.register_buffer("slot_family", grammar.slot_family.clone().long())
        self.register_buffer("slot_required", grammar.slot_required.clone().float())
        self.register_buffer("slot_presence_prior", grammar.slot_presence_prior.clone().float())
        self.register_buffer("slot_proto_raw", grammar.slot_proto.clone().float())
        self.register_buffer("slot_geom_mean", grammar.slot_geom_mean.clone().float())
        self.register_buffer("slot_geom_var", grammar.slot_geom_var.clone().float())
        self.register_buffer("edges", grammar.edges.clone().long())
        self.register_buffer("edge_rel_mean", grammar.edge_rel_mean.clone().float())
        self.register_buffer("edge_rel_var", grammar.edge_rel_var.clone().float())
        self.register_buffer("edge_support", grammar.edge_support.clone().float())

        init_lambda = max(float(getattr(cfg, "hkg_fusion_lambda_init", 0.20)), 1e-6)
        init_raw = math.log(math.expm1(init_lambda)) if init_lambda < 20.0 else init_lambda
        if bool(getattr(cfg, "hkg_use_classwise_fusion", True)):
            self.hkg_lambda_raw = nn.Parameter(torch.full((cnum,), float(init_raw)))
            self.hkg_bias = nn.Parameter(torch.zeros(cnum))
        else:
            self.hkg_lambda_raw = nn.Parameter(torch.tensor(float(init_raw)))
            self.hkg_bias = nn.Parameter(torch.zeros(1))

        self.raw_node_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_edge_scale = nn.Parameter(torch.tensor(0.0))

    def freeze_stage1(self) -> None:
        """Compatibility no-op: this cached model has no Stage-1 submodule."""
        return None

    def _cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self.cfg, name, default))

    def _cfg_int(self, name: str, default: int) -> int:
        return int(getattr(self.cfg, name, default))

    def _cfg_str(self, name: str, default: str) -> str:
        return str(getattr(self.cfg, name, default))

    @staticmethod
    def _gaussian_ll(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.clamp_min(1e-4)
        return -0.5 * (((x - mu) ** 2) / var + var.log()).mean(-1)

    def _project_slot_prototypes(self) -> torch.Tensor:
        c, a, s, d = self.slot_proto_raw.shape
        z = self.proj(self.slot_proto_raw.reshape(c * a * s, d))
        return F.normalize(z, dim=-1).reshape(c, a, s, -1)

    def _masked_assignment(self, score: torch.Tensor, valid_pair: torch.Tensor, mode: str) -> torch.Tensor:
        """Return assignment [B,C,A,S,N] from compatibility scores.

        ``mode=max`` is fastest and uses one component per slot, but does not
        prevent duplicated component use. ``mode=softmax`` is fully differentiable.
        ``mode=sinkhorn`` does a few row/column normalizations as a cheap
        one-to-one approximation.
        """
        dtype = score.dtype
        if mode == "max":
            masked = score.masked_fill(~valid_pair, -1.0e4)
            best = masked.argmax(-1)
            has = valid_pair.any(-1)
            ass = F.one_hot(best, num_classes=score.shape[-1]).to(dtype)
            return ass * has.unsqueeze(-1).to(dtype)

        tau = max(self._cfg_float("isaog_assignment_tau", 0.35), 1e-4)
        base = score / tau
        base = base.masked_fill(~valid_pair, -1.0e4)
        maxv = base.amax(-1, keepdim=True)
        weights = torch.exp((base - maxv).clamp(-60, 60)) * valid_pair.to(dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        if mode != "sinkhorn":
            return weights

        # Approximate rectangular matching: slots sum to one; component columns
        # are discouraged from collecting mass from many slots.  This is not an
        # exact Hungarian solver, but it is batched and GPU-friendly.
        ass = weights
        iters = max(1, self._cfg_int("isaog_sinkhorn_iters", 5))
        slot_mask = valid_pair.any(-1).to(dtype)
        comp_mask = valid_pair.any(-2).to(dtype)
        for _ in range(iters):
            ass = ass / ass.sum(-1, keepdim=True).clamp_min(1e-8)
            ass = ass * slot_mask.unsqueeze(-1)
            col_sum = ass.sum(-2, keepdim=True).clamp_min(1e-8)
            ass = torch.minimum(ass / col_sum, torch.ones_like(ass)) * comp_mask.unsqueeze(-2)
            ass = ass / ass.sum(-1, keepdim=True).clamp_min(1e-8)
            ass = ass * slot_mask.unsqueeze(-1)
        return torch.nan_to_num(ass, nan=0.0, posinf=0.0, neginf=0.0)

    def _score_class_chunk(
        self,
        comp_proj: torch.Tensor,
        component_valid: torch.Tensor,
        component_part: torch.Tensor,
        component_presence: torch.Tensor,
        component_geom: torch.Tensor,
        slot_proto: torch.Tensor,
        c0: int,
        c1: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score a class chunk.

        Returns node template scores [B,Cc,A], observed slot geometry
        [B,Cc,A,S,G], observed slot presence [B,Cc,A,S], assignment mass
        [B,Cc,A,S,N].
        """
        device = comp_proj.device
        dtype = comp_proj.dtype
        bsz, ncomp, _ = comp_proj.shape
        slot_p = slot_proto[c0:c1]
        cchunk = int(c1 - c0)
        app = torch.einsum("bnh,cash->bcasn", comp_proj, slot_p)
        # Geometry likelihood [B,C,A,S,N].
        diff = component_geom[:, None, None, None, :, :] - self.slot_geom_mean[c0:c1].to(device)[None, :, :, :, None, :]
        var = self.slot_geom_var[c0:c1].to(device)[None, :, :, :, None, :].clamp_min(1e-4)
        geom_ll = -0.5 * ((diff * diff) / var + var.log()).mean(-1)
        prior = torch.log(self.slot_presence_prior[c0:c1].to(device).clamp_min(1e-4))[None, :, :, :, None]
        pres = torch.log(component_presence[:, None, None, None, :].clamp_min(1e-4))
        raw = (
            self._cfg_float("isaog_node_app_scale", getattr(self.cfg, "hkg_node_app_scale", 1.0)) * app
            + self._cfg_float("isaog_node_geom_scale", 0.50) * geom_ll
            + self._cfg_float("isaog_node_prior_scale", 0.25) * prior
            + self._cfg_float("isaog_node_presence_scale", getattr(self.cfg, "hkg_node_presence_scale", 0.50)) * pres
        )
        valid_slot = (self.slot_valid[c0:c1].to(device) > 0.5)[None, :, :, :, None]
        type_ok = component_part[:, None, None, None, :] == self.slot_part[c0:c1].to(device)[None, :, :, :, None]
        comp_ok = component_valid[:, None, None, None, :].bool()
        valid_pair = valid_slot & type_ok & comp_ok
        mode = self._cfg_str("isaog_assignment", "softmax").lower()
        ass = self._masked_assignment(raw, valid_pair, mode)
        slot_score = (ass * raw.masked_fill(~valid_pair, 0.0)).sum(-1)
        slot_presence_obs = (ass * component_presence[:, None, None, None, :]).sum(-1)
        slot_geom_obs = torch.einsum("bcasn,bng->bcasg", ass, component_geom)
        matched = (ass.sum(-1) > 0).to(dtype)
        missing = (1.0 - slot_presence_obs).clamp(0, 1)
        miss_penalty = (
            self._cfg_float("isaog_missing_penalty", getattr(self.cfg, "hkg_absence_penalty", 0.35))
            * self.slot_required[c0:c1].to(device)[None]
            * self.slot_presence_prior[c0:c1].to(device)[None]
            * missing
        )
        valid_weight = self.slot_valid[c0:c1].to(device)[None]
        node = ((slot_score - miss_penalty) * valid_weight).sum(-1)
        denom = valid_weight.sum(-1).clamp_min(1.0).sqrt()
        node = node / denom
        return node, slot_geom_obs, slot_presence_obs, ass

    def _edge_scores_from_slots(self, slot_geom: torch.Tensor, slot_presence: torch.Tensor) -> torch.Tensor:
        """Vectorized edge score scatter-add over all template edges."""
        bsz = slot_geom.shape[0]
        device = slot_geom.device
        cnum, anum = self.num_classes, self.num_templates
        scores = slot_geom.new_zeros(bsz, cnum, anum)
        if self.edges.numel() == 0:
            return scores
        rows = self.edges.to(device)
        c = rows[:, 0].long()
        a = rows[:, 1].long()
        si = rows[:, 2].long()
        sj = rows[:, 3].long()
        gi = slot_geom[:, c, a, si, :]
        gj = slot_geom[:, c, a, sj, :]
        gamma = relation_features_from_geometry(gi, gj)
        rel = self._gaussian_ll(gamma, self.edge_rel_mean.to(device).unsqueeze(0), self.edge_rel_var.to(device).unsqueeze(0)).clamp(-8, 2)
        pi = slot_presence[:, c, a, si].clamp(0, 1)
        pj = slot_presence[:, c, a, sj].clamp(0, 1)
        strength = torch.sqrt((pi * pj).clamp_min(1e-8))
        val = strength * self.edge_support.to(device).unsqueeze(0) * rel
        flat_idx = c * anum + a
        flat = torch.zeros(bsz, cnum * anum, dtype=val.dtype, device=device)
        flat.scatter_add_(1, flat_idx.view(1, -1).expand(bsz, -1), val)
        counts = torch.zeros(cnum * anum, dtype=val.dtype, device=device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=val.dtype))
        flat = flat / counts.clamp_min(1.0).sqrt().view(1, -1)
        return flat.view(bsz, cnum, anum)

    def _aggregate_templates(self, template_scores: torch.Tensor, *, include_prior: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        device = template_scores.device
        valid = self.template_valid.to(device).bool().unsqueeze(0)
        s = template_scores
        if include_prior:
            s = s + torch.log(self.template_prior.to(device).clamp_min(1e-8)).unsqueeze(0)
        s = torch.where(valid, s, torch.full_like(s, -1.0e6))
        if bool(getattr(self.cfg, "hkg_use_logsumexp_templates", True)):
            tau = max(float(getattr(self.cfg, "hkg_template_tau", 1.0)), 1e-6)
            logits = tau * torch.logsumexp(s / tau, dim=-1)
        else:
            logits = s.max(-1).values
        return logits, s.argmax(-1)

    def _base_logits(self, part_tokens: torch.Tensor, part_presence: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.proj(part_tokens.float()), dim=-1)
        w = part_presence.float().clamp_min(0.0).unsqueeze(-1)
        pooled = (z * w).sum(1) / w.sum(1).clamp_min(1e-6)
        return self.base_head(pooled)

    def forward(self, batch: dict[str, torch.Tensor], *, enable_edges: bool = True, return_parse: bool = False, **_: Any) -> dict[str, torch.Tensor | list[dict[str, object]]]:
        device = next(self.parameters()).device
        component_valid = batch["component_valid"].to(device, non_blocking=True).bool()
        component_part = batch["component_part"].to(device, non_blocking=True).long()
        component_presence = batch["component_presence"].to(device, non_blocking=True).float()
        component_geom = batch["component_geom"].to(device, non_blocking=True).float()
        component_token = batch["component_token"].to(device, non_blocking=True).float()
        part_presence = batch["part_presence"].to(device, non_blocking=True).float()
        part_tokens = batch["part_tokens"].to(device, non_blocking=True).float()

        comp_proj = F.normalize(self.proj(component_token), dim=-1)
        slot_proto = self._project_slot_prototypes()
        class_chunk = max(1, self._cfg_int("isaog_class_chunk", self.num_classes))
        node_chunks: list[torch.Tensor] = []
        geom_chunks: list[torch.Tensor] = []
        pres_chunks: list[torch.Tensor] = []
        for c0 in range(0, self.num_classes, class_chunk):
            c1 = min(self.num_classes, c0 + class_chunk)
            node, gobs, pobs, _ass = self._score_class_chunk(
                comp_proj,
                component_valid,
                component_part,
                component_presence,
                component_geom,
                slot_proto,
                c0,
                c1,
            )
            node_chunks.append(node)
            geom_chunks.append(gobs)
            pres_chunks.append(pobs)
        node_scores = torch.cat(node_chunks, dim=1)
        slot_geom_obs = torch.cat(geom_chunks, dim=1)
        slot_presence_obs = torch.cat(pres_chunks, dim=1)
        if enable_edges:
            edge_scores = self._edge_scores_from_slots(slot_geom_obs, slot_presence_obs)
        else:
            edge_scores = torch.zeros_like(node_scores)
        template_scores = (
            F.softplus(self.raw_node_scale) * node_scores
            + self._cfg_float("isaog_edge_scale", getattr(self.cfg, "hkg_edge_scale", 0.55)) * F.softplus(self.raw_edge_scale) * edge_scores
        )
        hkg_logits, best_template = self._aggregate_templates(template_scores, include_prior=True)
        edge_logits, _ = self._aggregate_templates(edge_scores, include_prior=False)
        clip = float(getattr(self.cfg, "hkg_score_clip", 30.0))
        hkg_logits = torch.nan_to_num(hkg_logits, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        edge_logits = torch.nan_to_num(edge_logits, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        base_logits = self._base_logits(part_tokens, part_presence)
        lam = F.softplus(self.hkg_lambda_raw)
        if lam.ndim == 0:
            fused = base_logits + lam * hkg_logits + self.hkg_bias.view(1, -1)
        else:
            fused = base_logits + lam.view(1, -1) * hkg_logits + self.hkg_bias.view(1, -1)
        out: dict[str, torch.Tensor | list[dict[str, object]]] = {
            "logits": finite_center_clip_logits(fused),
            "base_logits": finite_center_clip_logits(base_logits),
            "hkg_logits": finite_center_clip_logits(hkg_logits),
            "hkg_fusion_logits": finite_center_clip_logits(hkg_logits),
            "node_logits": finite_center_clip_logits(self._aggregate_templates(node_scores, include_prior=True)[0]),
            "edge_logits": finite_center_clip_logits(edge_logits),
            "motif_logits": finite_center_clip_logits(torch.zeros_like(edge_logits)),
            "template_scores": template_scores,
            "best_template": best_template,
            "part_presence": part_presence,
            "component_presence": component_presence,
            "component_valid": component_valid.float(),
            "absence_penalty": torch.zeros_like(hkg_logits),
            "conflict_penalty": torch.zeros_like(hkg_logits),
            "edges_enabled": torch.tensor(float(bool(enable_edges)), device=device),
        }
        if return_parse:
            out["parse_graph"] = self.decode_best_parse(out, batch)
        return out

    @torch.no_grad()
    def decode_best_parse(self, out: dict[str, Any], batch: dict[str, torch.Tensor]) -> list[dict[str, object]]:
        pred = out["logits"].argmax(-1).detach().cpu().tolist()
        best_template = out["best_template"].detach().cpu()
        comp_part = batch["component_part"].detach().cpu()
        comp_presence = batch["component_presence"].detach().cpu()
        comp_geom = batch["component_geom"].detach().cpu()
        summaries: list[dict[str, object]] = []
        for b, c in enumerate(pred):
            a = int(best_template[b, c].item())
            active_components = []
            for n in range(comp_part.shape[1]):
                p = float(comp_presence[b, n].item())
                if p <= 0:
                    continue
                k = int(comp_part[b, n].item())
                active_components.append({
                    "component": int(n),
                    "part": self.schema.part_names[k] if 0 <= k < len(self.schema.part_names) else str(k),
                    "presence": p,
                    "geom": [float(x) for x in comp_geom[b, n].tolist()],
                })
            edges = []
            rows = self.edges.detach().cpu()
            for e, row in enumerate(rows.tolist()):
                if row[0] == c and row[1] == a:
                    edges.append({
                        "slot_i": int(row[2]),
                        "slot_j": int(row[3]),
                        "type": self.grammar.edge_type_names[e] if e < len(self.grammar.edge_type_names) else "relation",
                        "support": float(self.edge_support[e].detach().cpu().item()),
                    })
            summaries.append({
                "pred_class": self.schema.obj_names[c],
                "template": a,
                "active_components": active_components,
                "template_edges": edges,
            })
        return summaries
