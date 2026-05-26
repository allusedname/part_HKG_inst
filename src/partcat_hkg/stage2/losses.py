from __future__ import annotations

import torch
import torch.nn.functional as F

from partcat_hkg.config import Stage2LossConfig


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weight)


def stage2_parse_loss(out: dict[str, torch.Tensor], labels: torch.Tensor, cfg: Stage2LossConfig, class_weight: torch.Tensor | None = None):
    loss = cfg.parse_ce * classification_loss(out["logits"], labels, class_weight)
    logs = {"ce_parse": float(loss.detach().cpu())}
    if cfg.visible_aux > 0:
        aux = classification_loss(out["visible_logits"], labels, class_weight)
        loss = loss + cfg.visible_aux * aux
        logs["ce_visible"] = float(aux.detach().cpu())
    if cfg.completion_aux > 0:
        aux = classification_loss(out["completion_logits"], labels, class_weight)
        loss = loss + cfg.completion_aux * aux
        logs["ce_completion"] = float(aux.detach().cpu())
    if cfg.edge_aux > 0:
        aux = classification_loss(out["edge_logits"], labels, class_weight)
        loss = loss + cfg.edge_aux * aux
        logs["ce_edge"] = float(aux.detach().cpu())
    if cfg.base_aux > 0:
        aux = classification_loss(out["base_logits"], labels, class_weight)
        loss = loss + cfg.base_aux * aux
        logs["ce_base"] = float(aux.detach().cpu())
    logs["loss"] = float(loss.detach().cpu())
    return loss, logs


def stage2_aog_hkg_loss(out: dict[str, torch.Tensor], labels: torch.Tensor, cfg: Stage2LossConfig, class_weight: torch.Tensor | None = None):
    """Loss for the AOG-HKG Stage-2 classifier.

    The final fused classifier is the main objective.  Auxiliary CE terms keep
    the HKG parser and base classifier independently useful, which makes the
    parse graph interpretable rather than a dead residual branch.
    """
    loss = cfg.parse_ce * classification_loss(out["logits"], labels, class_weight)
    logs = {"ce_final": float(loss.detach().cpu())}
    if cfg.hkg_aux > 0 and "hkg_logits" in out:
        aux = classification_loss(out["hkg_logits"], labels, class_weight)
        loss = loss + cfg.hkg_aux * aux
        logs["ce_hkg"] = float(aux.detach().cpu())
    if cfg.base_aux > 0 and "base_logits" in out:
        aux = classification_loss(out["base_logits"], labels, class_weight)
        loss = loss + cfg.base_aux * aux
        logs["ce_base"] = float(aux.detach().cpu())
    edges_enabled = True
    if "edges_enabled" in out:
        flag = out["edges_enabled"]
        edges_enabled = bool(float(flag.detach().cpu().item()) > 0.5) if torch.is_tensor(flag) else bool(flag)
    if edges_enabled and cfg.edge_aux > 0 and "edge_logits" in out:
        aux = classification_loss(out["edge_logits"], labels, class_weight)
        loss = loss + cfg.edge_aux * aux
        logs["ce_edge"] = float(aux.detach().cpu())
    if edges_enabled and cfg.motif_aux > 0 and "motif_logits" in out:
        aux = classification_loss(out["motif_logits"], labels, class_weight)
        loss = loss + cfg.motif_aux * aux
        logs["ce_motif"] = float(aux.detach().cpu())
    logs["loss"] = float(loss.detach().cpu())
    return loss, logs
