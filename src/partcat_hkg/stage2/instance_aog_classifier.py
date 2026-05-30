from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.kg.instance_aog import InstanceAOG
from partcat_hkg.kg.instance_components import average_stage1_token_maps, extract_instance_components
from partcat_hkg.kg.relations import relation_attributes_from_masks
from partcat_hkg.utils.numerics import finite_center_clip_logits


class InstanceAOGStage2Classifier(nn.Module):
    """Differentiable scorer for the Instance-Slot AOG grammar.

    Stage 1 provides functional part masks and tokens.  This classifier splits
    each part mask into unordered connected components, assigns components to
    template-local latent slots under each class/template hypothesis, scores node
    and relation compatibility, and fuses the explicit AOG logit with a learned
    base part-token classifier.
    """

    def __init__(self, stage1_model: nn.Module, grammar: InstanceAOG, cfg: Any):
        super().__init__()
        self.stage1 = stage1_model
        self.grammar = grammar
        self.kg = grammar  # compatibility with existing trainer/checkpoint extras
        self.schema = grammar.schema
        self.cfg = cfg
        token_dim = int(grammar.token_dim)
        hidden = int(getattr(cfg, "hidden_dim", 256))
        cnum, anum, snum = grammar.schema.num_classes, int(grammar.num_templates), int(grammar.max_slots)

        self.proj = nn.Linear(token_dim, hidden)
        self.base_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.05), nn.Linear(hidden, cnum))

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

        # Python index for fast edge lookup.  Buffers remain the source of truth.
        by_template: list[list[list[int]]] = [[[] for _ in range(anum)] for _ in range(cnum)]
        for e, row in enumerate(grammar.edges.detach().cpu().tolist()):
            c, a = int(row[0]), int(row[1])
            if 0 <= c < cnum and 0 <= a < anum:
                by_template[c][a].append(e)
        self._edges_by_template = by_template
        self._num_classes = cnum
        self._num_templates = anum
        self._max_slots = snum

    def freeze_stage1(self) -> None:
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad_(False)

    def _cfg_float(self, name: str, default: float) -> float:
        return float(getattr(self.cfg, name, default))

    def _cfg_int(self, name: str, default: int) -> int:
        return int(getattr(self.cfg, name, default))

    def _stage1_extract(self, batch: dict, detach_stage1: bool = True) -> dict[str, Any]:
        device = next(self.parameters()).device
        image = batch["image"].to(device, non_blocking=True)
        if detach_stage1:
            self.stage1.eval()
            with torch.no_grad():
                out = self.stage1(image)
        else:
            out = self.stage1(image)
        part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"]))
        part_presence = out.get("part_presence")
        if part_presence is None:
            part_presence = part_prob.flatten(2).amax(-1)
        part_tokens = out.get("part_tokens", out.get("part_tokens_res"))
        if part_tokens is None:
            raise KeyError("Stage1 output must include part_tokens or part_tokens_res for InstanceAOG Stage2")
        return {"stage1": out, "part_prob": part_prob, "part_presence": part_presence, "part_tokens": part_tokens}

    def _components_for_batch(self, ex: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        out = ex["stage1"]
        part_prob = ex["part_prob"]
        part_presence = ex["part_presence"]
        part_tokens = ex["part_tokens"]
        comps: list[dict[str, torch.Tensor]] = []
        for b in range(part_prob.shape[0]):
            token_map = average_stage1_token_maps(out, b)
            if token_map is not None:
                token_map = token_map.to(part_prob.device)
            comp = extract_instance_components(
                part_prob[b],
                token_map=token_map,
                part_tokens=part_tokens[b],
                part_presence=part_presence[b],
                threshold=self._cfg_float("component_threshold", 0.40),
                min_area_frac=self._cfg_float("min_component_area_frac", 1.0e-4),
                max_components_per_part=self._cfg_int("max_components_per_part", 4),
                max_total_components=self._cfg_int("max_total_components", 32),
                min_presence=self._cfg_float("component_min_presence", 0.05),
            )
            comps.append(comp)
        return comps

    @staticmethod
    def _gaussian_ll(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.clamp_min(1e-4)
        return -0.5 * (((x - mu) ** 2) / var + var.log()).mean(-1)

    def _project_slot_prototypes(self) -> torch.Tensor:
        c, a, s, d = self.slot_proto_raw.shape
        flat = self.slot_proto_raw.reshape(c * a * s, d)
        return F.normalize(self.proj(flat), dim=-1).reshape(c, a, s, -1)

    def _score_template(
        self,
        comps: dict[str, torch.Tensor],
        slot_proto: torch.Tensor,
        c: int,
        a: int,
        *,
        enable_edges: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
        device = self.template_prior.device
        if float(self.template_valid[c, a].item()) <= 0:
            neg = torch.full((), -1.0e6, device=device)
            return neg, neg, {}
        ncomp = int(comps["part_type"].shape[0])
        comp_proj = F.normalize(self.proj(comps["token"].to(device)), dim=-1) if ncomp else comps["token"].new_zeros(0, slot_proto.shape[-1]).to(device)
        comp_geom = comps["geom"].to(device)
        comp_presence = comps["presence"].to(device).clamp(1e-4, 1.0)
        comp_part = comps["part_type"].to(device).long()

        valid_slots = [s for s in range(self._max_slots) if float(self.slot_valid[c, a, s].item()) > 0.5]
        valid_slots.sort(key=lambda s: (-float(self.slot_required[c, a, s].item()), -float(self.slot_presence_prior[c, a, s].item()), s))
        used: set[int] = set()
        assignment: dict[int, int] = {}
        node_score = torch.zeros((), device=device)
        missing_penalty = torch.zeros((), device=device)
        app_scale = self._cfg_float("isaog_node_app_scale", getattr(self.cfg, "hkg_node_app_scale", 1.0))
        geom_scale = self._cfg_float("isaog_node_geom_scale", 0.50)
        prior_scale = self._cfg_float("isaog_node_prior_scale", 0.25)
        presence_scale = self._cfg_float("isaog_node_presence_scale", getattr(self.cfg, "hkg_node_presence_scale", 0.50))
        miss_weight = self._cfg_float("isaog_missing_penalty", getattr(self.cfg, "hkg_absence_penalty", 0.35))
        for s in valid_slots:
            k = int(self.slot_part[c, a, s].item())
            candidates = [i for i in range(ncomp) if i not in used and int(comp_part[i].item()) == k]
            if not candidates:
                required = self.slot_required[c, a, s].to(device)
                prior = self.slot_presence_prior[c, a, s].to(device).clamp(0, 1)
                missing_penalty = missing_penalty + miss_weight * required * prior
                continue
            cand_scores: list[torch.Tensor] = []
            for i in candidates:
                app = (comp_proj[i] * slot_proto[c, a, s]).sum(-1)
                geom = self._gaussian_ll(comp_geom[i], self.slot_geom_mean[c, a, s].to(device), self.slot_geom_var[c, a, s].to(device))
                prior = torch.log(self.slot_presence_prior[c, a, s].to(device).clamp_min(1e-4))
                pres = torch.log(comp_presence[i].clamp_min(1e-4))
                cand_scores.append(app_scale * app + geom_scale * geom + prior_scale * prior + presence_scale * pres)
            stacked = torch.stack(cand_scores)
            best_local = int(stacked.argmax().item())
            best_comp = int(candidates[best_local])
            assignment[int(s)] = best_comp
            used.add(best_comp)
            node_score = node_score + stacked[best_local]

        # Penalize unassigned components only when their part type is not expected
        # or the template has already used all slots of that type.
        spurious_penalty = torch.zeros((), device=device)
        spurious_weight = self._cfg_float("isaog_spurious_penalty", getattr(self.cfg, "hkg_spurious_template_penalty", 0.25))
        expected_counts: defaultdict[int, int] = defaultdict(int)
        used_counts: defaultdict[int, int] = defaultdict(int)
        for s in valid_slots:
            expected_counts[int(self.slot_part[c, a, s].item())] += 1
        for s, i in assignment.items():
            used_counts[int(comp_part[i].item())] += 1
        for i in range(ncomp):
            if i in used:
                continue
            k = int(comp_part[i].item())
            if expected_counts[k] <= used_counts[k]:
                spurious_penalty = spurious_penalty + spurious_weight * comp_presence[i]

        edge_score = torch.zeros((), device=device)
        if enable_edges:
            for e in self._edges_by_template[c][a]:
                si = int(self.edges[e, 2].item())
                sj = int(self.edges[e, 3].item())
                if si not in assignment or sj not in assignment:
                    continue
                ci, cj = assignment[si], assignment[sj]
                gamma = relation_attributes_from_masks(comps["mask"][ci].to(device), comps["mask"][cj].to(device)).to(device)
                rel = self._gaussian_ll(gamma, self.edge_rel_mean[e].to(device), self.edge_rel_var[e].to(device)).clamp(-8, 2)
                strength = torch.sqrt((comp_presence[ci] * comp_presence[cj]).clamp_min(1e-8))
                edge_score = edge_score + self.edge_support[e].to(device) * strength * rel
            edge_count = max(1, len(self._edges_by_template[c][a]))
            edge_score = edge_score / math.sqrt(float(edge_count))
        total = (
            F.softplus(self.raw_node_scale) * (node_score - missing_penalty - spurious_penalty)
            + self._cfg_float("isaog_edge_scale", getattr(self.cfg, "hkg_edge_scale", 0.55)) * F.softplus(self.raw_edge_scale) * edge_score
        )
        return total, edge_score, assignment

    def _aggregate_templates(self, scores: torch.Tensor, *, include_prior: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        device = scores.device
        valid = self.template_valid.to(device).bool().unsqueeze(0)
        s = scores
        if include_prior:
            s = s + torch.log(self.template_prior.to(device).clamp_min(1e-8)).unsqueeze(0)
        s = torch.where(valid, s, torch.full_like(s, -1e6))
        if bool(getattr(self.cfg, "hkg_use_logsumexp_templates", True)):
            tau = max(float(getattr(self.cfg, "hkg_template_tau", 1.0)), 1e-6)
            logits = tau * torch.logsumexp(s / tau, dim=-1)
        else:
            logits = s.max(dim=-1).values
        return logits, s.argmax(dim=-1)

    def _grammar_scores(self, comps_batch: list[dict[str, torch.Tensor]], *, enable_edges: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[dict[int, int]]]]:
        device = self.template_prior.device
        slot_proto = self._project_slot_prototypes()
        bsz = len(comps_batch)
        template_scores = torch.zeros(bsz, self._num_classes, self._num_templates, device=device)
        edge_scores = torch.zeros_like(template_scores)
        assignments: list[list[dict[int, int]]] = []
        for b, comps in enumerate(comps_batch):
            per_b: list[dict[int, int]] = []
            for c in range(self._num_classes):
                for a in range(self._num_templates):
                    total, edge, ass = self._score_template(comps, slot_proto, c, a, enable_edges=enable_edges)
                    template_scores[b, c, a] = total
                    edge_scores[b, c, a] = edge
                    if len(per_b) <= c * self._num_templates + a:
                        per_b.append(ass)
            assignments.append(per_b)
        hkg_logits, best_template = self._aggregate_templates(template_scores, include_prior=True)
        edge_logits, _ = self._aggregate_templates(edge_scores, include_prior=False)
        return hkg_logits, edge_logits, best_template, template_scores, assignments

    def forward(self, batch: dict, *, detach_stage1: bool = True, enable_edges: bool = True, return_parse: bool = False) -> dict[str, torch.Tensor | list[dict[str, object]]]:
        ex = self._stage1_extract(batch, detach_stage1=detach_stage1)
        comps = self._components_for_batch(ex)
        hkg_logits, edge_logits, best_template, raw_template_scores, assignments = self._grammar_scores(comps, enable_edges=enable_edges)
        clip = float(getattr(self.cfg, "hkg_score_clip", 30.0))
        hkg_logits = torch.nan_to_num(hkg_logits, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        edge_logits = torch.nan_to_num(edge_logits, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

        # Base part-token classifier.  This is intentionally the same simple
        # token-pooling idea used by the AOG-HKG Stage-2 model.
        part_tokens = ex["part_tokens"]
        part_presence = ex["part_presence"].clamp_min(0.0)
        part_proj = F.normalize(self.proj(part_tokens), dim=-1)
        base_token = (part_proj * part_presence.unsqueeze(-1)).sum(1) / part_presence.sum(1, keepdim=True).clamp_min(1e-6)
        base_logits = self.base_head(base_token)

        lam = F.softplus(self.hkg_lambda_raw)
        if lam.ndim == 0:
            logits = base_logits + lam * hkg_logits + self.hkg_bias.view(1, -1)
        else:
            logits = base_logits + lam.view(1, -1) * hkg_logits + self.hkg_bias.view(1, -1)
        out: dict[str, torch.Tensor | list[dict[str, object]]] = {
            "logits": finite_center_clip_logits(logits),
            "base_logits": finite_center_clip_logits(base_logits),
            "hkg_logits": finite_center_clip_logits(hkg_logits),
            "hkg_fusion_logits": finite_center_clip_logits(hkg_logits),
            "node_logits": finite_center_clip_logits(hkg_logits),
            "edge_logits": finite_center_clip_logits(edge_logits),
            "motif_logits": finite_center_clip_logits(torch.zeros_like(edge_logits)),
            "template_scores": raw_template_scores,
            "best_template": best_template,
            "part_presence": part_presence,
            "part_prob": ex["part_prob"],
            "absence_penalty": torch.zeros_like(hkg_logits),
            "conflict_penalty": torch.zeros_like(hkg_logits),
            "edges_enabled": torch.tensor(float(bool(enable_edges)), device=part_presence.device),
        }
        if return_parse:
            out["parse_graph"] = self.decode_best_parse(out, comps, assignments)
        return out

    @torch.no_grad()
    def decode_best_parse(self, out: dict[str, Any], comps_batch: list[dict[str, torch.Tensor]], assignments: list[list[dict[int, int]]]) -> list[dict[str, object]]:
        pred = out["logits"].argmax(-1).detach().cpu().tolist()
        best_template = out["best_template"].detach().cpu()
        summaries: list[dict[str, object]] = []
        for b, c in enumerate(pred):
            a = int(best_template[b, c].item())
            ass = assignments[b][c * self._num_templates + a] if assignments and assignments[b] else {}
            comps = comps_batch[b]
            active = []
            for s, ci in sorted(ass.items()):
                k = int(self.slot_part[c, a, s].detach().cpu().item())
                active.append({
                    "slot": int(s),
                    "family": int(self.slot_family[c, a, s].detach().cpu().item()),
                    "part": self.schema.part_names[k] if 0 <= k < len(self.schema.part_names) else str(k),
                    "component": int(ci),
                    "presence": float(comps["presence"][ci].detach().cpu().item()),
                    "geom": [float(x) for x in comps["geom"][ci].detach().cpu().tolist()],
                })
            edges = []
            for e in self._edges_by_template[c][a]:
                row = self.edges[e].detach().cpu().tolist()
                edges.append({
                    "slot_i": int(row[2]),
                    "slot_j": int(row[3]),
                    "type": self.grammar.edge_type_names[e] if e < len(self.grammar.edge_type_names) else "relation",
                    "support": float(self.edge_support[e].detach().cpu().item()),
                })
            summaries.append({
                "pred_class": self.schema.obj_names[c],
                "template": a,
                "active_slots": active,
                "template_edges": edges,
            })
        return summaries
