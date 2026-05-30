from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import StrictAOGGrammar
from .terminals import terminal_pair_relations


@dataclass
class ParserConfig:
    assignment: str = "sinkhorn"  # sinkhorn | max
    assignment_tau: float = 0.35
    sinkhorn_iters: int = 16
    node_app_weight: float = 1.0
    node_geom_weight: float = 0.50
    node_presence_weight: float = 0.25
    relation_weight: float = 0.35
    missing_weight: float = 0.35
    spurious_weight: float = 0.05
    score_clip: float = 50.0
    use_template_logsumexp: bool = True
    template_tau: float = 1.0
    class_chunk: int = 0  # 0 = all classes at once


def _get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _as_parser_config(cfg: Any) -> ParserConfig:
    if isinstance(cfg, ParserConfig):
        return cfg
    return ParserConfig(
        assignment=str(_get(cfg, "strict_aog_assignment", _get(cfg, "isaog_assignment", "sinkhorn"))),
        assignment_tau=float(_get(cfg, "strict_aog_assignment_tau", _get(cfg, "isaog_assignment_tau", 0.35))),
        sinkhorn_iters=int(_get(cfg, "strict_aog_sinkhorn_iters", 16)),
        node_app_weight=float(_get(cfg, "strict_aog_node_app_weight", 1.0)),
        node_geom_weight=float(_get(cfg, "strict_aog_node_geom_weight", 0.50)),
        node_presence_weight=float(_get(cfg, "strict_aog_node_presence_weight", 0.25)),
        relation_weight=float(_get(cfg, "strict_aog_relation_weight", 0.35)),
        missing_weight=float(_get(cfg, "strict_aog_missing_weight", 0.35)),
        spurious_weight=float(_get(cfg, "strict_aog_spurious_weight", 0.05)),
        score_clip=float(_get(cfg, "strict_aog_score_clip", 50.0)),
        use_template_logsumexp=bool(_get(cfg, "strict_aog_use_template_logsumexp", True)),
        template_tau=float(_get(cfg, "strict_aog_template_tau", 1.0)),
        class_chunk=int(_get(cfg, "strict_aog_class_chunk", 0)),
    )


class StrictAOGParser(nn.Module):
    """GPU parse scorer for a strict neural Spatial AOG.

    The parser is not a generic classifier head.  It implements the AOG energy:

    ``class Or switch + template And production + terminal singleton energies + horizontal relation energies``.

    The address variables from slots to observed terminals are approximated with
    max or entropic Sinkhorn assignment.  This is the differentiable analogue of
    AOG parse-graph inference while remaining GPU-friendly.
    """

    def __init__(self, grammar: StrictAOGGrammar, cfg: Any | None = None):
        super().__init__()
        self.grammar = grammar
        self.cfg = _as_parser_config(cfg)
        d = int(grammar.token_dim)
        # A learnable identity-initialized projection permits mild terminal feature
        # calibration without replacing the grammar with a black-box classifier.
        self.token_proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.token_proj.weight)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.class_bias = nn.Parameter(torch.zeros(grammar.num_classes))
        self.register_buffer("class_prior", grammar.class_prior.float().clamp_min(1e-8))
        self.register_buffer("template_prior", grammar.template_prior.float().clamp_min(1e-8))
        self.register_buffer("template_valid", grammar.template_valid.float())
        self.register_buffer("slot_valid", grammar.slot_valid.float())
        self.register_buffer("slot_part", grammar.slot_part.long())
        self.register_buffer("slot_required", grammar.slot_required.float())
        self.register_buffer("slot_presence", grammar.slot_presence.float().clamp(0, 1))
        self.register_buffer("slot_proto", grammar.slot_proto.float())
        self.register_buffer("slot_geom_mean", grammar.slot_geom_mean.float())
        self.register_buffer("slot_geom_var", grammar.slot_geom_var.float().clamp_min(1e-4))
        self.register_buffer("edges", grammar.edges.long())
        self.register_buffer("edge_type", grammar.edge_type.long())
        self.register_buffer("edge_support", grammar.edge_support.float().clamp(0, 1))
        self.register_buffer("edge_rel_mean", grammar.edge_rel_mean.float())
        self.register_buffer("edge_rel_var", grammar.edge_rel_var.float().clamp_min(1e-4))

    @property
    def num_classes(self) -> int:
        return int(self.grammar.num_classes)

    @staticmethod
    def _geom_ll(comp_geom: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # comp_geom: [B,1,1,1,N,G], mu/var: [1,C,A,S,1,G]
        return -0.5 * (((comp_geom - mu) ** 2) / var.clamp_min(1e-4) + var.clamp_min(1e-4).log()).mean(-1)

    def _node_compatibility(self, batch: dict[str, torch.Tensor], c0: int, c1: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = batch["terminal_valid"].bool()
        part = batch["terminal_part"].long()
        score = batch["terminal_score"].float().clamp(1e-4, 1.0)
        geom = batch["terminal_geom"].float()
        token = batch["terminal_token"].float()
        B, N, D = token.shape
        comp_tok = F.normalize(self.token_proj(token), dim=-1)
        slot_proto = F.normalize(self.token_proj(self.slot_proto[c0:c1].reshape(-1, D)), dim=-1).reshape(c1 - c0, self.grammar.num_templates, self.grammar.max_slots, D)
        # [B,C,A,S,N]
        app = torch.einsum("bnd,casd->bcasn", comp_tok, slot_proto)
        comp_geom = geom[:, None, None, None, :, :]
        mu = self.slot_geom_mean[c0:c1][None, :, :, :, None, :]
        var = self.slot_geom_var[c0:c1][None, :, :, :, None, :]
        geom_ll = self._geom_ll(comp_geom, mu, var)
        type_ok = part[:, None, None, None, :] == self.slot_part[c0:c1][None, :, :, :, None]
        valid_ok = valid[:, None, None, None, :]
        slot_ok = self.slot_valid[c0:c1][None, :, :, :, None] > 0.5
        mask = type_ok & valid_ok & slot_ok
        pres = torch.log(score[:, None, None, None, :].clamp_min(1e-4))
        slot_prior = torch.log(self.slot_presence[c0:c1][None, :, :, :, None].clamp_min(1e-4))
        compat = (
            self.cfg.node_app_weight * app
            + self.cfg.node_geom_weight * geom_ll
            + self.cfg.node_presence_weight * pres
            + 0.15 * slot_prior
        )
        compat = torch.nan_to_num(compat, nan=-1e6, posinf=1e6, neginf=-1e6)
        compat = compat.masked_fill(~mask, -1e6)
        return compat, mask, slot_ok.squeeze(-1)

    def _sinkhorn_assign(self, compat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # compat: [B,C,A,S,N]. Entropic matching with row/column capacities <= 1.
        # Add a dummy terminal so missing slots have a safe sink and no real
        # terminal is forced to explain a bad slot.
        tau = max(float(self.cfg.assignment_tau), 1e-4)
        B, C, A, S, N = compat.shape
        dummy = torch.zeros(B, C, A, S, 1, device=compat.device, dtype=compat.dtype)
        logK = torch.cat([compat / tau, dummy], dim=-1)
        dummy_mask = torch.ones(B, C, A, S, 1, device=mask.device, dtype=torch.bool)
        full_mask = torch.cat([mask, dummy_mask], dim=-1)
        logK = logK.masked_fill(~full_mask, -1e9)
        # Each real slot has mass 1 if valid; invalid slots mass 0.
        row_valid = full_mask.any(-1)
        log_r = torch.where(row_valid, torch.zeros_like(row_valid, dtype=compat.dtype), torch.full_like(row_valid, -1e9, dtype=compat.dtype))
        # Real terminals can be used at most once, dummy can absorb all slots.
        col_valid = full_mask.any(-2)
        log_c = torch.where(col_valid, torch.zeros_like(col_valid, dtype=compat.dtype), torch.full_like(col_valid, -1e9, dtype=compat.dtype))
        # Make dummy column capacity S, so all missing slots may choose it.
        log_c[..., -1] = torch.log(torch.tensor(float(max(S, 1)), device=compat.device, dtype=compat.dtype))
        u = torch.zeros_like(log_r)
        v = torch.zeros_like(log_c)
        for _ in range(max(1, int(self.cfg.sinkhorn_iters))):
            u = log_r - torch.logsumexp(logK + v.unsqueeze(-2), dim=-1)
            v = log_c - torch.logsumexp(logK + u.unsqueeze(-1), dim=-2)
        logA = logK + u.unsqueeze(-1) + v.unsqueeze(-2)
        assign = torch.exp(logA).clamp(0, 1)
        return assign[..., :N]

    def _max_assign(self, compat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Fast independent address choice. Duplicates are allowed; this is an
        # approximation useful for debugging and very large class sets.
        idx = compat.argmax(dim=-1)
        best = compat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        ok = best > -1e5
        assign = torch.zeros_like(compat)
        assign.scatter_(-1, idx.unsqueeze(-1), ok.unsqueeze(-1).to(compat.dtype))
        return assign * mask.to(compat.dtype)

    def _node_scores(self, compat: torch.Tensor, mask: torch.Tensor, slot_ok: torch.Tensor, c0: int, c1: int) -> tuple[torch.Tensor, torch.Tensor]:
        assignment_mode = str(self.cfg.assignment).lower()
        if assignment_mode == "max":
            assign = self._max_assign(compat, mask)
        elif assignment_mode == "sinkhorn":
            assign = self._sinkhorn_assign(compat, mask)
        else:
            raise ValueError(f"Unknown strict AOG assignment mode: {self.cfg.assignment}")
        assigned = assign.sum(-1).clamp(0, 1)  # [B,C,A,S]
        node = (assign * compat.clamp_min(-1e4)).sum(dim=(-1, -2))  # [B,C,A]
        missing = (slot_ok.float() * self.slot_required[c0:c1][None] * self.slot_presence[c0:c1][None] * (1.0 - assigned)).sum(-1)
        node = node - self.cfg.missing_weight * missing
        return node, assign

    def _edge_scores(self, batch: dict[str, torch.Tensor], assign_all: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
        # assign_all: [B,Cchunk,A,S,N]
        B = assign_all.shape[0]
        Cc = assign_all.shape[1]
        A = self.grammar.num_templates
        device = assign_all.device
        out = torch.zeros(B, Cc, A, device=device)
        if self.edges.numel() == 0 or self.cfg.relation_weight == 0:
            return out
        rel_pair = terminal_pair_relations(batch["terminal_geom"].to(device))  # [B,N,N,R]
        valid_pair = (batch["terminal_valid"].bool().to(device)[:, :, None] & batch["terminal_valid"].bool().to(device)[:, None, :]).float()
        rows = self.edges.to(device)
        # Filter edges in class chunk.
        keep = (rows[:, 0] >= c0) & (rows[:, 0] < c1)
        if not bool(keep.any()):
            return out
        idxs = torch.nonzero(keep, as_tuple=False).flatten()
        for e in idxs.tolist():
            c_abs = int(rows[e, 0].item())
            c = c_abs - c0
            a = int(rows[e, 1].item())
            si = int(rows[e, 2].item())
            sj = int(rows[e, 3].item())
            wi = assign_all[:, c, a, si, :]  # [B,N]
            wj = assign_all[:, c, a, sj, :]
            pair_w = wi[:, :, None] * wj[:, None, :] * valid_pair
            denom = pair_w.sum(dim=(1, 2)).clamp_min(1e-6)
            exp_rel = torch.einsum("bnm,bnmr->br", pair_w, rel_pair) / denom[:, None]
            var = self.edge_rel_var[e].to(device).clamp_min(1e-4)
            mu = self.edge_rel_mean[e].to(device)
            val = -0.5 * (((exp_rel - mu) ** 2) / var + var.log()).mean(-1)
            out[:, c, a] += self.edge_support[e].to(device) * val.clamp(-8, 4)
        # normalize by sqrt edge count per template
        counts = torch.zeros(Cc, A, device=device)
        for e in idxs.tolist():
            c = int(rows[e, 0].item()) - c0
            a = int(rows[e, 1].item())
            counts[c, a] += 1.0
        return out / torch.sqrt(counts.clamp_min(1.0))[None]

    def _aggregate_templates(self, scores: torch.Tensor, c0: int, c1: int) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self.template_valid[c0:c1].to(scores.device).bool()[None]
        s = scores + torch.log(self.template_prior[c0:c1].to(scores.device).clamp_min(1e-8))[None]
        s = torch.where(valid, s, torch.full_like(s, -1e6))
        if self.cfg.use_template_logsumexp:
            tau = max(float(self.cfg.template_tau), 1e-6)
            logits = tau * torch.logsumexp(s / tau, dim=-1)
        else:
            logits = s.max(-1).values
        return logits, s.argmax(-1)

    def _score_chunk(self, batch: dict[str, torch.Tensor], c0: int, c1: int, *, enable_edges: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compat, mask, slot_ok = self._node_compatibility(batch, c0, c1)
        node, assign = self._node_scores(compat, mask, slot_ok, c0, c1)
        edge = self._edge_scores(batch, assign, c0, c1) if enable_edges else torch.zeros_like(node)
        template_scores = node + self.cfg.relation_weight * edge
        logits, best = self._aggregate_templates(template_scores, c0, c1)
        return logits, best, template_scores, edge

    def forward(self, batch: dict[str, torch.Tensor], *, enable_edges: bool = True, return_parse: bool = False) -> dict[str, Any]:
        required = ["terminal_valid", "terminal_part", "terminal_score", "terminal_geom", "terminal_token"]
        missing = [k for k in required if k not in batch]
        if missing:
            raise KeyError(f"StrictAOGParser batch is missing keys: {missing}")
        device = self.class_prior.device
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        C = self.num_classes
        chunk = int(self.cfg.class_chunk or C)
        logits_chunks: list[torch.Tensor] = []
        best_chunks: list[torch.Tensor] = []
        template_chunks: list[torch.Tensor] = []
        edge_chunks: list[torch.Tensor] = []
        for c0 in range(0, C, chunk):
            c1 = min(C, c0 + chunk)
            lg, best, ts, es = self._score_chunk(batch, c0, c1, enable_edges=enable_edges)
            logits_chunks.append(lg)
            best_chunks.append(best)
            template_chunks.append(ts)
            edge_chunks.append(es)
        aog_logits = torch.cat(logits_chunks, dim=1)
        best_template = torch.cat(best_chunks, dim=1)
        template_scores = torch.cat(template_chunks, dim=1)
        edge_scores = torch.cat(edge_chunks, dim=1)
        aog_logits = aog_logits + torch.log(self.class_prior.to(device).clamp_min(1e-8))[None]
        scaled = F.softplus(self.logit_scale) * aog_logits + self.class_bias[None]
        clip = float(self.cfg.score_clip)
        scaled = torch.nan_to_num(scaled, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        # Keep keys compatible with the current Stage-2 trainer while making AOG
        # the primary model rather than a residual branch.
        out: dict[str, Any] = {
            "logits": scaled,
            "aog_logits": aog_logits,
            "hkg_logits": aog_logits,
            "base_logits": aog_logits.detach(),  # no black-box base branch in strict AOG mode
            "edge_logits": self._aggregate_edge_logits(edge_scores),
            "template_scores": template_scores,
            "best_template": best_template,
            "edges_enabled": torch.tensor(float(bool(enable_edges)), device=device),
        }
        if return_parse:
            out["parse_graph"] = self.decode_best_parse(batch, out)
        return out

    def _aggregate_edge_logits(self, edge_scores: torch.Tensor) -> torch.Tensor:
        valid = self.template_valid.to(edge_scores.device).bool()[None]
        s = torch.where(valid, edge_scores, torch.full_like(edge_scores, -1e6))
        if self.cfg.use_template_logsumexp:
            tau = max(float(self.cfg.template_tau), 1e-6)
            return tau * torch.logsumexp(s / tau, dim=-1)
        return s.max(-1).values

    @torch.no_grad()
    def decode_best_parse(self, batch: dict[str, torch.Tensor], out: dict[str, Any]) -> list[dict[str, Any]]:
        pred = out["logits"].argmax(-1).detach().cpu().tolist()
        best_t = out["best_template"].detach().cpu()
        summaries: list[dict[str, Any]] = []
        # Decode using max mode for readability; this does not affect training.
        old = self.cfg.assignment
        self.cfg.assignment = "max"
        try:
            for b, c in enumerate(pred):
                a = int(best_t[b, c].item())
                single = {k: v[b:b+1] for k, v in batch.items() if torch.is_tensor(v)}
                compat, mask, _ = self._node_compatibility(single, c, c + 1)
                assign = self._max_assign(compat, mask)[0, 0, a]
                slots = []
                for s in range(self.grammar.max_slots):
                    if float(self.slot_valid[c, a, s].item()) <= 0.5:
                        continue
                    row = assign[s]
                    if float(row.max().item()) <= 0:
                        status = "missing" if float(self.slot_required[c, a, s].item()) > 0.5 else "optional_absent"
                        slots.append({"slot": s, "part": self.grammar.part_names[int(self.slot_part[c, a, s].item())], "status": status})
                        continue
                    n = int(row.argmax().item())
                    slots.append({
                        "slot": s,
                        "part": self.grammar.part_names[int(self.slot_part[c, a, s].item())],
                        "terminal": n,
                        "score": float(batch["terminal_score"][b, n].detach().cpu().item()),
                        "geom": [float(x) for x in batch["terminal_geom"][b, n].detach().cpu().tolist()],
                    })
                edges = []
                for e, row in enumerate(self.edges.detach().cpu().tolist()):
                    if int(row[0]) == c and int(row[1]) == a:
                        edges.append({"slot_i": int(row[2]), "slot_j": int(row[3]), "support": float(self.edge_support[e].detach().cpu().item())})
                summaries.append({"class": self.grammar.class_names[c], "template": a, "slots": slots, "edges": edges})
        finally:
            self.cfg.assignment = old
        return summaries


def strict_aog_loss(out: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.cross_entropy(out["logits"], labels)
    with torch.no_grad():
        pred = out["logits"].argmax(-1)
        acc = (pred == labels).float().mean()
        std = out["logits"].std(dim=-1).mean()
    return loss, {"loss": float(loss.detach().cpu()), "acc": float(acc.cpu()), "logit_std": float(std.cpu())}
