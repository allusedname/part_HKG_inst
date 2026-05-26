from __future__ import annotations

import torch
import torch.nn.functional as F

from partcat_hkg.config import Stage1LossConfig
from partcat_hkg.data.schema import RoleSchema
from .pooling import topk_presence
from .quality import aggregate_role_prob_to_func, role_valid_mask_for_batch


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20))
    target = torch.nan_to_num(target.float(), nan=0.0).clamp(0, 1)
    if target.shape[-2:] != prob.shape[-2:]:
        target = F.interpolate(target, size=prob.shape[-2:], mode="nearest")
    inter = (prob * target).flatten(2).sum(-1)
    den = prob.flatten(2).sum(-1) + target.flatten(2).sum(-1)
    loss = 1.0 - (2 * inter + eps) / (den + eps)
    if weight is not None:
        loss = loss * weight.to(device=loss.device, dtype=loss.dtype).view(1, -1)
    return loss.mean()


def weighted_bce_logits(logits: torch.Tensor, target: torch.Tensor, channel_weight: torch.Tensor | None = None, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
    target = torch.nan_to_num(target.float(), nan=0.0).clamp(0, 1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if pos_weight is not None:
        bce = torch.where(target > 0.5, bce * pos_weight.to(logits.device, logits.dtype).view(1, -1, 1, 1), bce)
    if channel_weight is not None:
        bce = bce * channel_weight.to(logits.device, logits.dtype).view(1, -1, 1, 1)
    return bce.mean()


def binary_cross_entropy_prob(input_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary cross-entropy for probability inputs, implemented without BCELoss.

    ``torch.nn.functional.binary_cross_entropy`` is intentionally disallowed
    inside CUDA autocast regions because it receives post-sigmoid probabilities.
    Stage 1 has two losses that are naturally probability-domain quantities:
    invalid-role top-k presence and role-union composition.  Computing the BCE
    formula directly in float32 keeps AMP training safe while preserving the same
    objective.
    """
    p = torch.nan_to_num(input_prob.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
    y = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p)).mean()


def masked_soft_dice_loss_logits(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, channel_weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
    target = torch.nan_to_num(target.float(), nan=0.0).clamp(0, 1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    valid = valid_mask.float()
    prob = torch.sigmoid(logits)
    inter = (prob * target).flatten(2).sum(-1)
    den = prob.flatten(2).sum(-1) + target.flatten(2).sum(-1)
    loss = 1.0 - (2 * inter + eps) / (den + eps)
    if channel_weight is not None:
        loss = loss * channel_weight.to(logits.device, logits.dtype).view(1, -1)
    valid2 = valid.flatten(2).amax(-1)
    return (loss * valid2).sum() / (valid2.sum() + eps)


def dino_affinity_smooth_loss(part_prob: torch.Tensor, token_dino_map: torch.Tensor) -> torch.Tensor:
    """Lightweight edge-aware total variation guided by DINO token-map similarity."""
    p = F.interpolate(part_prob.float(), size=token_dino_map.shape[-2:], mode="bilinear", align_corners=False)
    d = F.normalize(token_dino_map.float(), dim=1)
    pdx = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs()
    pdy = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs()
    ddx = (d[:, :, :, 1:] - d[:, :, :, :-1]).pow(2).mean(1, keepdim=True)
    ddy = (d[:, :, 1:, :] - d[:, :, :-1, :]).pow(2).mean(1, keepdim=True)
    return (torch.exp(-5.0 * ddx).clamp(0, 1) * pdx).mean() + (torch.exp(-5.0 * ddy).clamp(0, 1) * pdy).mean()


def area_and_coverage_losses(part_logits: torch.Tensor, func_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Auxiliary losses from the previous notebook to reduce tiny/noisy present masks."""
    func_target = func_target.float().clamp(0, 1)
    if func_target.shape[-2:] != part_logits.shape[-2:]:
        func_target = F.interpolate(func_target, size=part_logits.shape[-2:], mode="nearest")
    fp = torch.sigmoid(torch.nan_to_num(part_logits.float(), nan=0.0).clamp(-20, 20))
    area_pred = fp.flatten(2).mean(-1)
    area_gt = func_target.flatten(2).mean(-1)
    present = (area_gt > 0).float()
    loss_area = ((area_pred - area_gt).abs() * present).sum() / (present.sum() + 1e-6)
    coverage = (fp * func_target).flatten(2).sum(-1) / (func_target.flatten(2).sum(-1) + 1e-6)
    loss_cov = ((1.0 - coverage) * present).sum() / (present.sum() + 1e-6)
    return loss_area, loss_cov


def stage1_loss(
    out: dict[str, torch.Tensor],
    batch: dict,
    schema: RoleSchema,
    cfg: Stage1LossConfig,
    *,
    part_loss_weight: torch.Tensor | None = None,
    part_pos_weight: torch.Tensor | None = None,
    role_loss_weight: torch.Tensor | None = None,
    topk_presence_k: int = 64,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = out["part_logits"].device
    func_target = batch["part_masks"].to(device).float()
    role_target = batch["role_masks"].to(device).float()
    union = batch["union_mask"].to(device).float()
    labels = batch["obj_label"].to(device)
    valid_role = role_valid_mask_for_batch(labels, schema).view(labels.shape[0], schema.num_roles, 1, 1)
    invalid_role = 1.0 - valid_role

    func_logits = out["part_logits"]
    role_logits = out["role_logits"]
    support_logits = out["support_logits"]

    union_for_support = union
    if union_for_support.shape[-2:] != support_logits.shape[-2:]:
        union_for_support = F.interpolate(union_for_support, size=support_logits.shape[-2:], mode="nearest")
    loss_support = (
        F.binary_cross_entropy_with_logits(torch.nan_to_num(support_logits.float()).clamp(-20, 20), union_for_support)
        + soft_dice_loss(support_logits, union)
    )
    loss_func = weighted_bce_logits(func_logits, func_target, part_loss_weight, part_pos_weight) + soft_dice_loss(func_logits, func_target, part_loss_weight)

    # Role loss: full supervision only on roles belonging to the GT object.
    rlog = torch.nan_to_num(role_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
    rt = role_target.float().clamp(0, 1)
    if rt.shape[-2:] != rlog.shape[-2:]:
        rt = F.interpolate(rt, size=rlog.shape[-2:], mode="nearest")
    rbce = F.binary_cross_entropy_with_logits(rlog, rt, reduction="none") * valid_role
    rw = role_loss_weight.to(device, rlog.dtype).view(1, -1, 1, 1) if role_loss_weight is not None else 1.0
    loss_role = (rbce * rw).sum() / (valid_role.sum() * rlog.shape[-1] * rlog.shape[-2] + 1e-6)
    loss_role = loss_role + masked_soft_dice_loss_logits(rlog, rt, valid_role, role_loss_weight)

    # Weak negative pressure for invalid object-aware role channels.
    inv_bce = F.binary_cross_entropy_with_logits(rlog, torch.zeros_like(rlog), reduction="none") * invalid_role
    loss_invalid = inv_bce.sum() / (invalid_role.sum() * rlog.shape[-1] * rlog.shape[-2] + 1e-6)
    invalid_pres = topk_presence(torch.sigmoid(rlog), k=topk_presence_k)
    invalid_bool = invalid_role.squeeze(-1).squeeze(-1).bool()
    loss_invalid_topk = binary_cross_entropy_prob(
        invalid_pres[invalid_bool],
        torch.zeros_like(invalid_pres[invalid_bool]),
    ) if invalid_bool.any() else rlog.sum() * 0.0

    # Functional/role consistency aggregates only roles valid for the image's GT object.
    role_prob_valid = torch.sigmoid(rlog) * valid_role
    role_agg = aggregate_role_prob_to_func(role_prob_valid, schema)
    func_prob = torch.sigmoid(func_logits.float())
    if func_prob.shape[-2:] != role_agg.shape[-2:]:
        func_prob = F.interpolate(func_prob, size=role_agg.shape[-2:], mode="bilinear", align_corners=False)
    loss_cons = F.mse_loss(func_prob, role_agg.detach())

    role_union = role_prob_valid.amax(dim=1, keepdim=True)
    union_comp = union.float()
    if union_comp.shape[-2:] != role_union.shape[-2:]:
        union_comp = F.interpolate(union_comp, size=role_union.shape[-2:], mode="nearest")
    loss_comp = binary_cross_entropy_prob(role_union, union_comp)

    fp = torch.sigmoid(func_logits.float())
    loss_area, loss_cov = area_and_coverage_losses(func_logits, func_target)
    loss_aff = dino_affinity_smooth_loss(fp, out["token_dino_map"]) if cfg.dino_affinity > 0 else func_logits.sum() * 0.0
    loss = (
        cfg.support * loss_support
        + cfg.functional * loss_func
        + cfg.role * loss_role
        + cfg.invalid_role_negative * loss_invalid
        + cfg.invalid_role_topk * loss_invalid_topk
        + cfg.functional_role_consistency * loss_cons
        + cfg.object_part_composition * loss_comp
        + cfg.area_mass * loss_area
        + cfg.present_coverage * loss_cov
        + cfg.dino_affinity * loss_aff
    )
    logs = {
        "loss": float(loss.detach().cpu()),
        "support": float(loss_support.detach().cpu()),
        "func": float(loss_func.detach().cpu()),
        "role": float(loss_role.detach().cpu()),
        "invalid_role": float(loss_invalid.detach().cpu()),
        "invalid_role_topk": float(loss_invalid_topk.detach().cpu()),
        "cons": float(loss_cons.detach().cpu()),
        "comp": float(loss_comp.detach().cpu()),
        "area": float(loss_area.detach().cpu()),
        "cov": float(loss_cov.detach().cpu()),
        "aff": float(loss_aff.detach().cpu()),
    }
    return loss, logs
