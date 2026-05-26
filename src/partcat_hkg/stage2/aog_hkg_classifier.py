from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.config import Stage2Config
from partcat_hkg.kg.datatypes import AOGHierarchicalKG, MOTIF_TYPE_NAMES
from partcat_hkg.kg.relations import relation_attributes_vectorized, relation_channel_strengths
from partcat_hkg.utils.numerics import finite_center_clip_logits


class AOGHKGStage2Classifier(nn.Module):
    """Stage 2 classifier as AOG-style HKG parsing.

    The HKG is the class-level grammar.  For each candidate class, the model
    evaluates alternative template branches, instantiates role nodes from the
    frozen Stage-1 masks/tokens, scores horizontal relation/motif factors, and
    fuses the resulting HKG logit with a learned base classifier.
    """

    def __init__(self, stage1_model: nn.Module, kg: AOGHierarchicalKG, cfg: Stage2Config):
        super().__init__()
        self.stage1 = stage1_model
        self.kg = kg
        self.schema = kg.schema
        self.cfg = cfg
        token_dim = int(stage1_model.cfg.token_dim)
        hidden = int(cfg.hidden_dim)
        cnum, fnum, anum = kg.schema.num_classes, kg.schema.num_parts, int(kg.num_templates)

        self.proj_r = nn.Linear(token_dim, hidden)
        self.proj_d = nn.Linear(token_dim, hidden)
        self.base_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.05), nn.Linear(hidden, cnum)
        )

        # Grammar/statistic buffers.
        self.register_buffer("role_index_cf", self.schema.role_index_table.clone().long())
        self.register_buffer("valid_cf", (self.schema.role_index_table >= 0).float())
        self.register_buffer("pmi", kg.pmi.clone().float())
        self.register_buffer("role_prior", kg.role_prior.clone().float())
        self.register_buffer("func_proto_r_raw", kg.func_proto_r.clone().float())
        self.register_buffer("func_proto_d_raw", kg.func_proto_d.clone().float())
        self.register_buffer("class_role_proto_r_raw", kg.class_role_proto_r.clone().float())
        self.register_buffer("class_role_proto_d_raw", kg.class_role_proto_d.clone().float())
        self.register_buffer("template_prior", kg.template_prior.clone().float())
        self.register_buffer("template_valid", kg.template_valid.clone().float())
        self.register_buffer("template_role_prior", kg.template_role_prior.clone().float())
        self.register_buffer("template_role_required", kg.template_role_required.clone().float())
        self.register_buffer("template_role_proto_r_raw", kg.template_role_proto_r.clone().float())
        self.register_buffer("template_role_proto_d_raw", kg.template_role_proto_d.clone().float())
        self.register_buffer("template_edges", kg.template_edges.clone().long())
        self.register_buffer("template_rel_mean", kg.template_rel_mean.clone().float())
        self.register_buffer("template_rel_var", kg.template_rel_var.clone().float())
        self.register_buffer("template_rel_global_mean", kg.template_rel_global_mean.clone().float())
        self.register_buffer("template_rel_global_var", kg.template_rel_global_var.clone().float())
        self.register_buffer("template_rel_support", kg.template_rel_support.clone().float())
        self.register_buffer("template_rel_ig", kg.template_rel_ig.clone().float())
        self.register_buffer("motif_edges", kg.motif_edges.clone().long())
        self.register_buffer("motif_support", kg.motif_support.clone().float())

        # HKG score normalization is a lightweight calibration layer; callers can
        # overwrite these buffers after a calibration pass if desired.
        self.register_buffer("hkg_score_mean", torch.zeros(cnum))
        self.register_buffer("hkg_score_std", torch.ones(cnum))

        # hkg_fusion_lambda_init is the *actual* desired initial positive
        # fusion weight.  Since the forward pass uses softplus(raw), initialize
        # raw with the inverse-softplus.  The previous raw=0.20 initialization
        # actually produced softplus(0.20) ~= 0.80, which made the HKG branch
        # dominate the base classifier from epoch 1 and let the standalone base
        # head drift into a residual-correction role.
        init_lambda = max(float(cfg.hkg_fusion_lambda_init), 1e-6)
        init_raw = math.log(math.expm1(init_lambda)) if init_lambda < 20.0 else init_lambda
        if bool(cfg.hkg_use_classwise_fusion):
            self.hkg_lambda_raw = nn.Parameter(torch.full((cnum,), float(init_raw)))
            self.hkg_bias = nn.Parameter(torch.zeros(cnum))
        else:
            self.hkg_lambda_raw = nn.Parameter(torch.tensor(float(init_raw)))
            self.hkg_bias = nn.Parameter(torch.zeros(1))

        # Optional learned rescaling of terms.  Softplus keeps weights positive
        # and makes the interpretation close to an energy model.
        self.raw_node_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_edge_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_motif_scale = nn.Parameter(torch.tensor(0.0))

        # HKG-v2: make relation and motif utility trainable.  The previous
        # implementation used fixed offline templates only; edge/motif CE could
        # not actually improve the relation branch because there were no
        # relation-specific trainable parameters.
        init_one = math.log(math.expm1(1.0))
        self.edge_weight_raw = nn.Parameter(torch.full((int(kg.template_edges.shape[0]),), float(init_one)))
        self.motif_weight_raw = nn.Parameter(torch.full((int(kg.motif_edges.shape[0]),), float(init_one)))

    def freeze_stage1(self) -> None:
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad_(False)

    def _stage1_extract(self, batch: dict, detach_stage1: bool = True) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        image = batch["image"].to(device, non_blocking=True)
        if detach_stage1:
            self.stage1.eval()
            with torch.no_grad():
                out = self.stage1(image)
        else:
            out = self.stage1(image)
        # Prefer the Stage-1 products exactly as trained.  Fallbacks keep older
        # checkpoints and synthetic tests working.
        part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"]))
        role_prob = out.get("role_prob", torch.sigmoid(out["role_logits"]))
        part_presence = out.get("part_presence")
        role_presence = out.get("role_presence")
        part_tokens_r = out.get("part_tokens_res", out.get("part_tokens"))
        part_tokens_d = out.get("part_tokens_dino", out.get("part_tokens"))
        role_tokens_r = out.get("role_tokens_res", out.get("role_tokens", part_tokens_r.new_zeros(part_tokens_r.shape[0], self.schema.num_roles, part_tokens_r.shape[-1])))
        role_tokens_d = out.get("role_tokens_dino", out.get("role_tokens", role_tokens_r))
        return {
            "stage1": out,
            "part_prob": part_prob,
            "role_prob": role_prob,
            "part_presence": part_presence,
            "role_presence": role_presence,
            "part_tokens_r": part_tokens_r,
            "part_tokens_d": part_tokens_d,
            "role_tokens_r": role_tokens_r,
            "role_tokens_d": role_tokens_d,
        }

    def _gather_role_cf(self, tensor_br: torch.Tensor) -> torch.Tensor:
        """Gather role-indexed tensor [B,R,...] into class/part slots [B,C,F,...]."""
        bsz = tensor_br.shape[0]
        rid = self.role_index_cf.to(tensor_br.device).clamp_min(0)
        out = tensor_br.index_select(1, rid.reshape(-1)).reshape(bsz, self.schema.num_classes, self.schema.num_parts, *tensor_br.shape[2:])
        return out * self.valid_cf.to(tensor_br.device).view(1, self.schema.num_classes, self.schema.num_parts, *([1] * (tensor_br.ndim - 2)))

    @staticmethod
    def _gaussian_ll(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.clamp_min(1e-3)
        return -0.5 * (((x - mu) ** 2 / var) + var.log()).mean(-1)

    def _node_scores(
        self,
        part_presence: torch.Tensor,
        role_presence_cf: torch.Tensor,
        part_tokens_r: torch.Tensor,
        part_tokens_d: torch.Tensor,
        role_tokens_r_cf: torch.Tensor,
        role_tokens_d_cf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = part_presence.device
        # Functional token similarity to class-template role prototypes.
        fr = F.normalize(self.proj_r(part_tokens_r), dim=-1)
        fd = F.normalize(self.proj_d(part_tokens_d), dim=-1)
        tpr = F.normalize(self.proj_r(self.template_role_proto_r_raw.to(device)), dim=-1)
        tpd = F.normalize(self.proj_d(self.template_role_proto_d_raw.to(device)), dim=-1)
        func_sim = 0.5 * (
            torch.einsum("bfd,cafd->bcaf", fr, tpr)
            + torch.einsum("bfd,cafd->bcaf", fd, tpd)
        )

        # Object-aware role-token similarity.  This is useful after Stage 1 has
        # learned role channels, but it is softly combined with functional tokens
        # to avoid early role-channel brittleness.
        rr = F.normalize(self.proj_r(role_tokens_r_cf), dim=-1)
        rd = F.normalize(self.proj_d(role_tokens_d_cf), dim=-1)
        role_sim = 0.5 * ((rr.unsqueeze(2) * tpr.unsqueeze(0)).sum(-1) + (rd.unsqueeze(2) * tpd.unsqueeze(0)).sum(-1))
        app_sim = 0.5 * (func_sim + role_sim)

        obs_presence = torch.maximum(part_presence.unsqueeze(1), role_presence_cf).clamp_min(float(self.cfg.hkg_presence_floor))
        obs_presence_t = obs_presence.unsqueeze(2)
        t_prior = self.template_role_prior.to(device).unsqueeze(0)
        valid = self.template_valid.to(device).unsqueeze(-1).unsqueeze(0)
        pmi = self.pmi.to(device).unsqueeze(0).unsqueeze(2)
        required = self.template_role_required.to(device).unsqueeze(0)

        node_evidence = obs_presence_t * t_prior * valid * (
            float(self.cfg.hkg_node_presence_scale) * torch.log(obs_presence_t.clamp_min(1e-4))
            + float(self.cfg.hkg_node_pmi_scale) * pmi
            + float(self.cfg.hkg_node_app_scale) * app_sim
        )
        node_score = node_evidence.sum(-1)
        missing = (1.0 - obs_presence_t).clamp(0, 1)
        absence_penalty = float(self.cfg.hkg_absence_penalty) * (required * t_prior * missing * valid).sum(-1)
        conflict = float(self.cfg.hkg_conflict_penalty) * (
            part_presence.unsqueeze(1).unsqueeze(2)
            * (1.0 - self.role_prior.to(device).unsqueeze(0).unsqueeze(2)).clamp(0, 1)
            * valid
        ).sum(-1)
        # Template-level spurious part penalty: the diagnostics showed that
        # Stage 1 often activates plausible-looking but wrong parts such as
        # foot/head/tail on car images.  Penalize high-presence parts that are
        # not expected in a candidate template, while keeping the penalty soft so
        # occlusion and segmentation noise are tolerated.
        spurious_mask = (t_prior < float(getattr(self.cfg, "hkg_spurious_template_tau", 0.08))).float()
        spurious = float(getattr(self.cfg, "hkg_spurious_template_penalty", 0.0)) * (
            part_presence.unsqueeze(1).unsqueeze(2) * spurious_mask * valid
        ).sum(-1)
        return node_score - absence_penalty - conflict - spurious, absence_penalty, conflict + spurious

    def _center_template_scores(self, scores: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Center relation-only scores without rewarding templates that have no factors.

        Relation-fit scores are often mostly negative because they are energies.
        Centering converts them into relative evidence.  However, the previous
        implementation centered over every valid template, including templates
        with *zero* edge/motif factors.  Those empty templates could receive a
        positive score simply because the mean relation energy was negative.
        ``active_mask`` prevents this: templates without the relevant factors are
        forced back to zero after centering.
        """
        valid = self.template_valid.to(scores.device).unsqueeze(0).clamp(0, 1)
        if active_mask is not None:
            valid = valid * active_mask.to(scores.device).unsqueeze(0).clamp(0, 1)
        if not bool(getattr(self.cfg, "hkg_center_relation_scores", False)):
            return scores * valid if active_mask is not None else scores
        denom = valid.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = (scores * valid).sum(dim=(1, 2), keepdim=True) / denom
        centered = (scores - mean) * valid
        return torch.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    def _edge_scores(self, role_prob: torch.Tensor, role_presence: torch.Tensor) -> torch.Tensor:
        bsz = role_prob.shape[0]
        device = role_prob.device
        cnum, anum = self.schema.num_classes, int(self.kg.num_templates)
        scores = role_prob.new_zeros(bsz, cnum, anum)
        if self.template_edges.numel() == 0:
            return scores
        rows = self.template_edges.to(device)
        rid = self.role_index_cf.to(device)
        for e in range(rows.shape[0]):
            c, a, i, j = [int(v) for v in rows[e].tolist()]
            ri, rj = int(rid[c, i].item()), int(rid[c, j].item())
            if ri < 0 or rj < 0:
                continue
            mi = role_prob[:, ri].unsqueeze(1)
            mj = role_prob[:, rj].unsqueeze(1)
            gamma = relation_attributes_vectorized(mi, mj)[:, 0, :]
            ll_t = self._gaussian_ll(gamma, self.template_rel_mean[e].to(device), self.template_rel_var[e].to(device))
            ll_g = self._gaussian_ll(gamma, self.template_rel_global_mean[e].to(device), self.template_rel_global_var[e].to(device))
            mode = str(getattr(self.cfg, "edge_score_mode", "template_fit"))
            if mode == "template_fit":
                # Use direct template compatibility, not only improvement over a
                # global prior.  The earlier LLR+ReLU score suppressed many
                # useful but common relations to exactly zero.
                var = self.template_rel_var[e].to(device).clamp_min(1e-3)
                mu = self.template_rel_mean[e].to(device)
                edge_val = (-0.5 * (((gamma - mu) ** 2) / var).mean(-1)).clamp(-8, 0)
            else:
                edge_val = (ll_t - ll_g).clamp(-8, 8)
                if bool(self.cfg.hkg_edge_positive_only):
                    edge_val = F.relu(edge_val)
            strength = torch.sqrt((role_presence[:, ri] * role_presence[:, rj]).clamp_min(0.0) + 1e-8)
            support = self.template_rel_support[e].to(device)
            ig_gate = (self.template_rel_ig[e].to(device) / (self.template_rel_ig[e].to(device) + 1.0)).clamp(0, 1)
            learned_w = F.softplus(self.edge_weight_raw[e]).to(device)
            scores[:, c, a] += learned_w * strength * support * (0.5 + 0.5 * ig_gate) * edge_val
        # Normalize templates with many edges to prevent edge count from becoming
        # a proxy for class frequency.
        edge_counts = torch.zeros(cnum, anum, device=device)
        for c, a in rows[:, :2].tolist():
            edge_counts[int(c), int(a)] += 1.0
        active_mask = (edge_counts > 0).float()
        scores = scores / torch.sqrt(edge_counts.clamp_min(1.0)).unsqueeze(0)
        return self._center_template_scores(scores, active_mask=active_mask)

    def _motif_scores(self, role_prob: torch.Tensor, role_presence: torch.Tensor) -> torch.Tensor:
        bsz = role_prob.shape[0]
        device = role_prob.device
        cnum, anum = self.schema.num_classes, int(self.kg.num_templates)
        scores = role_prob.new_zeros(bsz, cnum, anum)
        if self.motif_edges.numel() == 0:
            return scores
        rows = self.motif_edges.to(device)
        rid = self.role_index_cf.to(device)
        for m in range(rows.shape[0]):
            c, a, i, j, mtype = [int(v) for v in rows[m].tolist()]
            ri, rj = int(rid[c, i].item()), int(rid[c, j].item())
            if ri < 0 or rj < 0:
                continue
            gamma = relation_attributes_vectorized(role_prob[:, ri].unsqueeze(1), role_prob[:, rj].unsqueeze(1))[:, 0, :]
            ch = relation_channel_strengths(gamma)
            # channel order: above, below, lateral, near, touching, overlap, contain_i, contain_j
            if mtype == 1:      # attached
                val = 0.5 * ch[:, 3] + 0.5 * ch[:, 4]
            elif mtype == 2:    # containment
                val = torch.maximum(ch[:, 6], ch[:, 7])
            elif mtype == 3:    # lateral / symmetry-like
                val = ch[:, 2]
            elif mtype == 4:    # body/frame appendage bond
                val = torch.maximum(ch[:, 3], torch.maximum(ch[:, 4], 0.5 * torch.maximum(ch[:, 0], ch[:, 1]) + 0.25 * ch[:, 2]))
            else:
                val = ch[:, 3]
            strength = torch.sqrt((role_presence[:, ri] * role_presence[:, rj]).clamp_min(0.0) + 1e-8)
            learned_w = F.softplus(self.motif_weight_raw[m]).to(device)
            scores[:, c, a] += learned_w * strength * self.motif_support[m].to(device) * val.clamp(0, 1)
        counts = torch.zeros(cnum, anum, device=device)
        for c, a in rows[:, :2].tolist():
            counts[int(c), int(a)] += 1.0
        active_mask = (counts > 0).float()
        scores = scores / torch.sqrt(counts.clamp_min(1.0)).unsqueeze(0)
        return self._center_template_scores(scores, active_mask=active_mask)

    def _aggregate_templates(self, template_scores: torch.Tensor, *, include_prior: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        device = template_scores.device
        valid = self.template_valid.to(device).bool().unsqueeze(0)
        if include_prior:
            log_prior = torch.log(self.template_prior.to(device).clamp_min(1e-8)).unsqueeze(0)
            s = template_scores + log_prior
        else:
            s = template_scores
        s = torch.where(valid, s, torch.full_like(s, -1e6))
        if bool(self.cfg.hkg_use_logsumexp_templates):
            tau = max(float(self.cfg.hkg_template_tau), 1e-6)
            logits = tau * torch.logsumexp(s / tau, dim=-1)
        else:
            logits = s.max(dim=-1).values
        best = s.argmax(dim=-1)
        return logits, best

    def set_hkg_score_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.hkg_score_mean.copy_(mean.detach().to(self.hkg_score_mean.device).float())
        self.hkg_score_std.copy_(std.detach().to(self.hkg_score_std.device).float().clamp_min(1e-4))

    def forward(self, batch: dict, *, detach_stage1: bool = True, enable_edges: bool = True, return_parse: bool = False) -> dict[str, torch.Tensor]:
        ex = self._stage1_extract(batch, detach_stage1=detach_stage1)
        part_presence = ex["part_presence"]
        role_presence = ex["role_presence"]
        role_presence_cf = self._gather_role_cf(role_presence).squeeze(-1) if role_presence.ndim == 3 else self._gather_role_cf(role_presence)
        role_tokens_r_cf = self._gather_role_cf(ex["role_tokens_r"])
        role_tokens_d_cf = self._gather_role_cf(ex["role_tokens_d"])

        node_scores, absence_penalty, conflict_penalty = self._node_scores(
            part_presence,
            role_presence_cf,
            ex["part_tokens_r"],
            ex["part_tokens_d"],
            role_tokens_r_cf,
            role_tokens_d_cf,
        )
        edge_scores = self._edge_scores(ex["role_prob"], role_presence) if enable_edges else torch.zeros_like(node_scores)
        motif_scores = self._motif_scores(ex["role_prob"], role_presence) if enable_edges else torch.zeros_like(node_scores)
        template_scores = (
            F.softplus(self.raw_node_scale) * node_scores
            + float(self.cfg.hkg_edge_scale) * F.softplus(self.raw_edge_scale) * edge_scores
            + float(self.cfg.hkg_motif_scale) * F.softplus(self.raw_motif_scale) * motif_scores
        )
        hkg_logits, best_template = self._aggregate_templates(template_scores)
        hkg_logits = torch.nan_to_num(hkg_logits, nan=0.0, posinf=float(self.cfg.hkg_score_clip), neginf=-float(self.cfg.hkg_score_clip)).clamp(-float(self.cfg.hkg_score_clip), float(self.cfg.hkg_score_clip))

        # Base image classifier from Stage-1 part tokens.
        fr = F.normalize(self.proj_r(ex["part_tokens_r"]), dim=-1)
        token_weight = part_presence.unsqueeze(-1).clamp_min(0.0)
        base_token = (token_weight * fr).sum(1) / token_weight.sum(1).clamp_min(1e-6)
        base_logits = self.base_head(base_token)

        if bool(self.cfg.hkg_normalize_scores):
            hkg_for_fusion = (hkg_logits - self.hkg_score_mean.to(hkg_logits.device).view(1, -1)) / self.hkg_score_std.to(hkg_logits.device).view(1, -1).clamp_min(1e-4)
        else:
            hkg_for_fusion = hkg_logits
        if bool(self.cfg.hkg_calibrated_fusion):
            lam = F.softplus(self.hkg_lambda_raw)
            logits = base_logits + lam.view(1, -1) * hkg_for_fusion + self.hkg_bias.view(1, -1)
        else:
            logits = hkg_logits

        out = {
            "logits": finite_center_clip_logits(logits),
            "base_logits": finite_center_clip_logits(base_logits),
            "hkg_logits": finite_center_clip_logits(hkg_logits),
            "hkg_fusion_logits": finite_center_clip_logits(hkg_for_fusion),
            "node_logits": finite_center_clip_logits(self._aggregate_templates(node_scores, include_prior=True)[0]),
            # Branch diagnostics/auxiliary losses should not be dominated by the
            # static template prior.  Otherwise edge-only accuracy can mostly be
            # an Or-node prior rather than relation evidence.
            "edge_logits": finite_center_clip_logits(self._aggregate_templates(edge_scores, include_prior=False)[0]),
            "motif_logits": finite_center_clip_logits(self._aggregate_templates(motif_scores, include_prior=False)[0]),
            "template_scores": template_scores,
            "best_template": best_template,
            "part_presence": part_presence,
            "role_presence": role_presence,
            "part_prob": ex["part_prob"],
            "role_prob": ex["role_prob"],
            "absence_penalty": absence_penalty,
            "conflict_penalty": conflict_penalty,
            "edges_enabled": torch.tensor(float(bool(enable_edges)), device=part_presence.device),
        }
        if return_parse:
            out["parse_graph"] = self.decode_best_parse(out)
        return out

    @torch.no_grad()
    def decode_best_parse(self, out: dict[str, torch.Tensor]) -> list[dict[str, object]]:
        """Return a compact, human-readable parse graph summary per image."""
        pred = out["logits"].argmax(-1).detach().cpu().tolist()
        best_template = out["best_template"].detach().cpu()
        part_presence = out["part_presence"].detach().cpu()
        summaries: list[dict[str, object]] = []
        for b, c in enumerate(pred):
            a = int(best_template[b, c].item())
            active = [
                {"part": self.schema.part_names[k], "presence": float(part_presence[b, k].item())}
                for k in range(self.schema.num_parts)
                if float(part_presence[b, k].item()) >= float(self.stage1.cfg.presence_threshold)
            ]
            edges = []
            rows = self.template_edges.detach().cpu()
            for e, row in enumerate(rows.tolist()):
                if row[0] == c and row[1] == a:
                    edges.append({
                        "part_i": self.schema.part_names[row[2]],
                        "part_j": self.schema.part_names[row[3]],
                        "type": self.kg.template_rel_type_names[e] if e < len(self.kg.template_rel_type_names) else "relation",
                        "support": float(self.template_rel_support[e].detach().cpu().item()),
                    })
            summaries.append({
                "pred_class": self.schema.obj_names[c],
                "template": a,
                "active_parts": active,
                "template_edges": edges,
            })
        return summaries
