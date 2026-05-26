from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.config import ProjectConfig
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.evaluation.metrics import binary_segmentation_stats
from partcat_hkg.models.losses import binary_cross_entropy_prob, stage1_loss
from partcat_hkg.models.pooling import topk_presence, topmean_presence
from partcat_hkg.training.stage1_trainer import _loss_weights
from partcat_hkg.utils.amp import autocast_cuda, make_scaler
from partcat_hkg.utils.io import save_checkpoint, save_json


@dataclass
class Stage1QualityLossWeights:
    """Extra Stage-1 losses used by the quality-upgrade notebook.

    These terms are intentionally implemented as an add-on instead of changing
    the existing default Stage-1 loss.  This lets us compare the original Stage-1
    checkpoint against a stronger fine-tuned checkpoint without disturbing the
    existing scripts.
    """

    # Presence calibration: directly supervise image-level part presence.
    presence_bce: float = 0.40

    # Confident false-positive suppression for absent parts.  Valid-but-absent
    # parts receive a much weaker penalty than class-invalid parts so that rare
    # small parts such as mirror/engine/beak are not over-suppressed before they
    # learn good localization.  The old absent_* names are kept as aliases for
    # backward compatibility and are used only if the valid_absent_* values are
    # set to None by external code.
    absent_topmean_fp: float = 0.08
    absent_mean_fp: float = 0.02
    valid_absent_topmean_fp: float = 0.08
    valid_absent_mean_fp: float = 0.02

    # Class-aware invalid functional part suppression.
    invalid_part_topmean: float = 0.35
    invalid_part_mean: float = 0.08

    # Small present parts get an adaptive boost for mask/boundary losses.
    small_part_area_tau: float = 0.015
    small_part_weight_max: float = 6.0
    small_part_weight_power: float = 0.5

    # Support containment: keep part masks within object support.
    gt_support_leak: float = 0.35
    pred_support_containment: float = 0.25

    # Small/boundary quality improvements.
    boundary: float = 0.08
    focal_functional: float = 0.12
    tversky_functional: float = 0.12

    # Hyperparameters.
    topq: float = 0.02
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_alpha: float = 0.35
    tversky_beta: float = 0.65
    boundary_kernel: int = 3


def _resize_like(target: torch.Tensor, ref: torch.Tensor, *, mode: str = "nearest") -> torch.Tensor:
    if target.shape[-2:] != ref.shape[-2:]:
        return F.interpolate(target.float(), size=ref.shape[-2:], mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)
    return target.float()


def _safe_sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20))


def topmean_mask_probability(prob: torch.Tensor, q: float = 0.02) -> torch.Tensor:
    """Top-q mean over spatial dimensions for [B,K,H,W] probability masks."""
    if q <= 0:
        return topk_presence(prob, k=64)
    return topmean_presence(prob, q=float(q))


def valid_part_mask_for_batch(labels: torch.Tensor, schema: RoleSchema) -> torch.Tensor:
    """Return [B,K] bool mask: functional parts valid for each object class.

    A part is valid for class c if there exists an object-aware role slot (c,k)
    in ``schema.role_index_table``.  This uses annotation schema knowledge only;
    it does not use prediction or image content.
    """
    table = (schema.role_index_table.to(labels.device) >= 0)
    return table[labels.long()].bool()


def _apply_channel_weight(loss: torch.Tensor, channel_weight: torch.Tensor | None) -> torch.Tensor:
    """Apply either [K] or [B,K] weights to a dense/per-channel loss tensor."""
    if channel_weight is None:
        return loss
    w = channel_weight.to(loss.device, loss.dtype)
    if w.ndim == 1:
        return loss * w.view(1, -1, *([1] * max(loss.ndim - 2, 0)))
    if w.ndim == 2:
        return loss * w.view(w.shape[0], w.shape[1], *([1] * max(loss.ndim - 2, 0)))
    raise ValueError(f"channel_weight must have shape [K] or [B,K], got {tuple(w.shape)}")


def combine_channel_weights(
    base: torch.Tensor | None,
    adaptive: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Combine static [K] weights with adaptive [B,K] weights if both exist."""
    if base is None and adaptive is None:
        return None
    if adaptive is None:
        return base.to(device=device, dtype=dtype) if base is not None else None
    out = adaptive.to(device=device, dtype=dtype)
    if base is not None:
        out = out * base.to(device=device, dtype=dtype).view(1, -1)
    return out


def small_part_adaptive_weights(
    target: torch.Tensor,
    gt_presence: torch.Tensor,
    *,
    area_tau: float = 0.015,
    max_weight: float = 6.0,
    power: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return [B,K] weights that boost present small parts.

    The weight is 1 for absent parts and for large present parts.  For present
    parts with area below ``area_tau``, the weight grows smoothly up to
    ``max_weight``.  This helps small but real structures without making absent
    channels harder to suppress.
    """
    tgt = target.float().clamp(0, 1)
    pres = gt_presence.to(tgt.device, tgt.dtype).clamp(0, 1)
    area = tgt.flatten(2).mean(-1).clamp_min(eps)
    tau = max(float(area_tau), eps)
    boost = (tau / area).pow(float(power)).clamp(1.0, float(max_weight))
    return torch.where(pres > 0.5, boost, torch.ones_like(boost))


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move tensor fields to device; keeps metadata/list fields untouched."""
    dev = torch.device(device)
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def safe_nanmean(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not finite.any():
        return values.new_tensor(float("nan"))
    return values[finite].mean()


def binary_focal_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    channel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal BCE for dense part masks, useful for confident false positives."""
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
    target = _resize_like(target.float().clamp(0, 1), logits, mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    alpha_t = torch.where(target > 0.5, torch.full_like(prob, float(alpha)), torch.full_like(prob, 1.0 - float(alpha)))
    loss = alpha_t * (1.0 - pt).clamp_min(0.0).pow(float(gamma)) * bce
    loss = _apply_channel_weight(loss, channel_weight)
    return loss.mean()


def tversky_loss_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.35,
    beta: float = 0.65,
    channel_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tversky loss emphasizing recall of small/rare parts through beta > alpha."""
    prob = _safe_sigmoid_logits(logits)
    target = _resize_like(target.float().clamp(0, 1), prob, mode="nearest")
    tp = (prob * target).flatten(2).sum(-1)
    fp = (prob * (1.0 - target)).flatten(2).sum(-1)
    fn = ((1.0 - prob) * target).flatten(2).sum(-1)
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    loss = 1.0 - score
    loss = _apply_channel_weight(loss, channel_weight)
    return loss.mean()


def soft_boundary_map(prob_or_mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Differentiable boundary proxy: soft dilation minus soft erosion."""
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    x = prob_or_mask.float().clamp(0, 1)
    pad = k // 2
    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def boundary_dice_loss_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gt_presence: torch.Tensor | None = None,
    *,
    kernel_size: int = 3,
    channel_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Boundary Dice over present parts only.

    Boundary supervision helps small structures whose area Dice can be dominated
    by background.  Absent parts are excluded to avoid rewarding empty boundaries.
    """
    prob = _safe_sigmoid_logits(logits)
    target = _resize_like(target.float().clamp(0, 1), prob, mode="nearest")
    pb = soft_boundary_map(prob, kernel_size=kernel_size)
    tb = soft_boundary_map(target, kernel_size=kernel_size)
    inter = (pb * tb).flatten(2).sum(-1)
    den = pb.flatten(2).sum(-1) + tb.flatten(2).sum(-1)
    loss = 1.0 - (2.0 * inter + eps) / (den + eps)
    if channel_weight is not None:
        loss = _apply_channel_weight(loss, channel_weight)
    if gt_presence is not None:
        present = gt_presence.to(loss.device).float().clamp(0, 1)
        # If a [B,K] adaptive weight is provided, use it in the denominator so
        # small present parts are boosted without changing the loss scale wildly.
        denom_weight = present
        if channel_weight is not None:
            w = channel_weight.to(loss.device, loss.dtype)
            if w.ndim == 1:
                denom_weight = denom_weight * w.view(1, -1)
            elif w.ndim == 2:
                denom_weight = denom_weight * w
        loss = (loss * present).sum() / (denom_weight.sum() + eps)
    else:
        loss = loss.mean()
    return loss


def support_containment_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Two containment losses: GT support leak and predicted support containment."""
    part_prob = out["part_prob"].float() if "part_prob" in out else _safe_sigmoid_logits(out["part_logits"])
    support_prob = out.get("support_prob")
    if support_prob is None:
        support_prob = _safe_sigmoid_logits(out["support_logits"])
    else:
        support_prob = support_prob.float().clamp(0, 1)
    gt_union = batch["union_mask"].to(part_prob.device).float().clamp(0, 1)
    gt_union = _resize_like(gt_union, part_prob, mode="nearest")
    support_prob = _resize_like(support_prob, part_prob, mode="bilinear")
    leak_gt = (part_prob * (1.0 - gt_union)).mean()
    pred_containment = F.relu(part_prob - support_prob).mean()
    return leak_gt, pred_containment


def valid_absent_part_false_positive_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    schema: RoleSchema,
    *,
    q: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress false positives for parts that are valid for the class but absent.

    This is intentionally weaker than invalid-part suppression.  It reduces
    image-level overactivation while avoiding the previous failure mode where
    rare valid small parts were suppressed before they could localize.
    """
    prob = out["part_prob"].float() if "part_prob" in out else _safe_sigmoid_logits(out["part_logits"])
    labels = batch["obj_label"].to(prob.device).long()
    gt_presence = batch["presence"].to(prob.device).float().clamp(0, 1)
    valid = valid_part_mask_for_batch(labels, schema)
    valid_absent = valid & (gt_presence < 0.5)
    if not valid_absent.any():
        z = prob.sum() * 0.0
        return z, z
    top = topmean_mask_probability(prob, q=q)
    mean = prob.flatten(2).mean(-1)
    return top[valid_absent].mean(), mean[valid_absent].mean()


def absent_part_false_positive_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, q: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible all-absent suppression helper.

    The quality loss now prefers ``valid_absent_part_false_positive_losses`` plus
    class-aware invalid suppression, but this helper remains for old notebooks.
    """
    prob = out["part_prob"].float() if "part_prob" in out else _safe_sigmoid_logits(out["part_logits"])
    gt_presence = batch["presence"].to(prob.device).float().clamp(0, 1)
    absent = gt_presence < 0.5
    if not absent.any():
        z = prob.sum() * 0.0
        return z, z
    top = topmean_mask_probability(prob, q=q)
    mean = prob.flatten(2).mean(-1)
    return top[absent].mean(), mean[absent].mean()


def invalid_part_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], schema: RoleSchema, *, q: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress functional parts invalid for the object class, except GT-present parts."""
    prob = out["part_prob"].float() if "part_prob" in out else _safe_sigmoid_logits(out["part_logits"])
    labels = batch["obj_label"].to(prob.device).long()
    gt_presence = batch["presence"].to(prob.device).float().clamp(0, 1)
    valid = valid_part_mask_for_batch(labels, schema)
    invalid = (~valid) & (gt_presence < 0.5)
    if not invalid.any():
        z = prob.sum() * 0.0
        return z, z
    top = topmean_mask_probability(prob, q=q)
    mean = prob.flatten(2).mean(-1)
    return top[invalid].mean(), mean[invalid].mean()


def explicit_presence_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """BCE over Stage-1 image-level functional-part presence probabilities."""
    pred = out["part_presence"].float().clamp(1e-6, 1.0 - 1e-6)
    gt = batch["presence"].to(pred.device).float().clamp(0, 1)
    return binary_cross_entropy_prob(pred, gt)


def stage1_quality_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    schema: RoleSchema,
    cfg: ProjectConfig,
    weights: Stage1QualityLossWeights | None = None,
    *,
    part_loss_weight: torch.Tensor | None = None,
    part_pos_weight: torch.Tensor | None = None,
    role_loss_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Existing Stage-1 loss plus the quality-upgrade terms.

    The original loss remains the backbone objective.  Added terms specifically
    target the failure modes seen in diagnostics: overconfident hallucinated
    presence, part masks leaking outside object support, poor small-part
    localization, and invalid class-part activations.
    """
    weights = weights or Stage1QualityLossWeights()
    base_loss, base_logs = stage1_loss(
        out,
        batch,
        schema,
        cfg.loss.stage1,
        part_loss_weight=part_loss_weight,
        part_pos_weight=part_pos_weight,
        role_loss_weight=role_loss_weight,
        topk_presence_k=cfg.model.stage1.topk_presence_k,
    )
    device = out["part_logits"].device
    gt_part = batch["part_masks"].to(device).float()
    gt_presence = batch["presence"].to(device).float()

    loss_presence = explicit_presence_loss(out, batch)
    loss_abs_top, loss_abs_mean = valid_absent_part_false_positive_losses(out, batch, schema, q=weights.topq)
    loss_inv_top, loss_inv_mean = invalid_part_losses(out, batch, schema, q=weights.topq)
    loss_leak_gt, loss_pred_contain = support_containment_losses(out, batch)

    # Boost present small parts for the shape-quality losses only.  This does not
    # increase absent penalties and therefore avoids the previous suppression of
    # valid small parts such as mirrors.
    area_tau = float(getattr(cfg.model.stage1, "small_part_area_tau", weights.small_part_area_tau))
    max_w = float(getattr(cfg.model.stage1, "small_part_weight_max", weights.small_part_weight_max))
    power = float(getattr(cfg.model.stage1, "small_part_weight_power", weights.small_part_weight_power))
    adaptive_w = small_part_adaptive_weights(gt_part, gt_presence, area_tau=area_tau, max_weight=max_w, power=power)
    dense_w = combine_channel_weights(part_loss_weight, adaptive_w, device=device, dtype=gt_part.dtype)

    loss_bdry = boundary_dice_loss_logits(
        out["part_logits"],
        gt_part,
        gt_presence,
        kernel_size=weights.boundary_kernel,
        channel_weight=dense_w,
    )
    loss_focal = binary_focal_bce_logits(
        out["part_logits"],
        gt_part,
        gamma=weights.focal_gamma,
        alpha=weights.focal_alpha,
        channel_weight=dense_w,
    )
    loss_tversky = tversky_loss_logits(
        out["part_logits"],
        gt_part,
        alpha=weights.tversky_alpha,
        beta=weights.tversky_beta,
        channel_weight=dense_w,
    )

    extra = (
        weights.presence_bce * loss_presence
        + weights.valid_absent_topmean_fp * loss_abs_top
        + weights.valid_absent_mean_fp * loss_abs_mean
        + weights.invalid_part_topmean * loss_inv_top
        + weights.invalid_part_mean * loss_inv_mean
        + weights.gt_support_leak * loss_leak_gt
        + weights.pred_support_containment * loss_pred_contain
        + weights.boundary * loss_bdry
        + weights.focal_functional * loss_focal
        + weights.tversky_functional * loss_tversky
    )
    loss = base_loss + extra
    logs: dict[str, float] = {**base_logs}
    logs.update({
        "quality_extra": float(extra.detach().cpu()),
        "presence_bce": float(loss_presence.detach().cpu()),
        "valid_absent_topmean_fp": float(loss_abs_top.detach().cpu()),
        "valid_absent_mean_fp": float(loss_abs_mean.detach().cpu()),
        "small_part_weight_mean": float(adaptive_w.detach().mean().cpu()),
        "small_part_weight_max": float(adaptive_w.detach().amax().cpu()),
        "invalid_part_topmean": float(loss_inv_top.detach().cpu()),
        "invalid_part_mean": float(loss_inv_mean.detach().cpu()),
        "gt_support_leak": float(loss_leak_gt.detach().cpu()),
        "pred_support_containment": float(loss_pred_contain.detach().cpu()),
        "boundary": float(loss_bdry.detach().cpu()),
        "focal_func": float(loss_focal.detach().cpu()),
        "tversky_func": float(loss_tversky.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    })
    return loss, logs


@torch.no_grad()
def evaluate_stage1_quality_detailed(
    model: torch.nn.Module,
    loader,
    cfg: ProjectConfig,
    *,
    device: str = "cuda",
    max_batches: int | None = None,
    loss_weights: Stage1QualityLossWeights | None = None,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate Stage-1 quality with reliable small-part metrics.

    Key fixes compared with the first notebook version:
    * empty-empty masks are not counted as IoU=1;
    * present-part IoU/Dice are reported separately from hallucination;
    * tensors are moved to the model device before loss computation;
    * per-class/per-part rows are returned so car-only diagnostics cannot be
      misread as global part performance.
    """
    model.eval()
    lw = _loss_weights(loader)
    run = defaultdict(float)
    n_batches = 0
    n_samples = 0
    num_parts = model.schema.num_parts
    num_classes = model.schema.num_classes
    part_names = list(model.schema.part_names)
    class_names = list(model.schema.obj_names)

    inter = torch.zeros(num_parts, dtype=torch.float64)
    union = torch.zeros(num_parts, dtype=torch.float64)
    dice_num = torch.zeros(num_parts, dtype=torch.float64)
    dice_den = torch.zeros(num_parts, dtype=torch.float64)
    union_nonempty_count = torch.zeros(num_parts, dtype=torch.float64)
    present_count = torch.zeros(num_parts, dtype=torch.float64)
    iou_present_sum = torch.zeros(num_parts, dtype=torch.float64)
    dice_present_sum = torch.zeros(num_parts, dtype=torch.float64)
    pred_area_sum = torch.zeros(num_parts, dtype=torch.float64)
    gt_area_sum = torch.zeros(num_parts, dtype=torch.float64)
    topmean_sum = torch.zeros(num_parts, dtype=torch.float64)
    presence_sum = torch.zeros(num_parts, dtype=torch.float64)
    maxprob_sum = torch.zeros(num_parts, dtype=torch.float64)
    tp = torch.zeros(num_parts, dtype=torch.float64)
    fp = torch.zeros(num_parts, dtype=torch.float64)
    fn = torch.zeros(num_parts, dtype=torch.float64)
    tn = torch.zeros(num_parts, dtype=torch.float64)

    pc_present = torch.zeros(num_classes, num_parts, dtype=torch.float64)
    pc_iou_sum = torch.zeros(num_classes, num_parts, dtype=torch.float64)
    pc_dice_sum = torch.zeros(num_classes, num_parts, dtype=torch.float64)
    pc_tp = torch.zeros(num_classes, num_parts, dtype=torch.float64)
    pc_fp = torch.zeros(num_classes, num_parts, dtype=torch.float64)
    pc_fn = torch.zeros(num_classes, num_parts, dtype=torch.float64)
    pc_tn = torch.zeros(num_classes, num_parts, dtype=torch.float64)

    support_leak_total = 0.0
    support_contain_total = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch_device = move_batch_to_device(batch, device)
        image = batch_device["image"]
        with autocast_cuda(False):
            out = model(image)
            loss, logs = stage1_quality_loss(
                out,
                batch_device,
                model.schema,
                cfg,
                loss_weights,
                part_loss_weight=lw.get("part_loss_weight"),
                part_pos_weight=lw.get("part_pos_weight"),
                role_loss_weight=lw.get("role_loss_weight"),
            )
        for k, v in logs.items():
            run[k] += float(v)

        prob_dev = out["part_prob"].detach().float()
        target_dev = batch_device["part_masks"].float().clamp(0, 1)
        if target_dev.shape[-2:] != prob_dev.shape[-2:]:
            target_dev = F.interpolate(target_dev, size=prob_dev.shape[-2:], mode="nearest")
        pred_dev = (prob_dev >= float(mask_threshold)).float()
        tgt_dev = target_dev

        b_inter_dev = (pred_dev * tgt_dev).flatten(2).sum(-1)
        b_pred_dev = pred_dev.flatten(2).sum(-1)
        b_tgt_dev = tgt_dev.flatten(2).sum(-1)
        b_union_dev = b_pred_dev + b_tgt_dev - b_inter_dev
        b_dice_den_dev = b_pred_dev + b_tgt_dev
        gt_presence_dev = batch_device["presence"].float().clamp(0, 1) > 0.5
        pred_presence_dev = out["part_presence"].detach().float() >= float(cfg.model.stage1.presence_threshold)
        labels_dev = batch_device["obj_label"].long()

        eps = 1e-6
        iou_sample = torch.full_like(b_union_dev, float("nan"), dtype=torch.float32)
        dice_sample = torch.full_like(b_union_dev, float("nan"), dtype=torch.float32)
        nonempty = b_union_dev > 0
        iou_sample[nonempty] = (b_inter_dev[nonempty] / (b_union_dev[nonempty] + eps)).float()
        dice_nonempty = b_dice_den_dev > 0
        dice_sample[dice_nonempty] = (2.0 * b_inter_dev[dice_nonempty] / (b_dice_den_dev[dice_nonempty] + eps)).float()
        present = gt_presence_dev

        # Aggregate global/nonempty statistics on CPU.
        b_inter = b_inter_dev.detach().cpu().double()
        b_pred = b_pred_dev.detach().cpu().double()
        b_tgt = b_tgt_dev.detach().cpu().double()
        b_union = b_union_dev.detach().cpu().double()
        b_dice_den = b_dice_den_dev.detach().cpu().double()
        inter += b_inter.sum(0)
        union += b_union.sum(0)
        dice_num += (2.0 * b_inter).sum(0)
        dice_den += b_dice_den.sum(0)
        union_nonempty_count += (b_union > 0).sum(0).double()
        present_cpu = present.detach().cpu()
        present_count += present_cpu.sum(0).double()

        iou_cpu = iou_sample.detach().cpu().double()
        dice_cpu = dice_sample.detach().cpu().double()
        iou_present_sum += torch.nan_to_num(torch.where(present_cpu, iou_cpu, torch.full_like(iou_cpu, float("nan"))), nan=0.0).sum(0)
        dice_present_sum += torch.nan_to_num(torch.where(present_cpu, dice_cpu, torch.full_like(dice_cpu, float("nan"))), nan=0.0).sum(0)

        presence_cpu = out["part_presence"].detach().cpu().double()
        presence_sum += presence_cpu.sum(0)
        pred_area_sum += prob_dev.detach().cpu().flatten(2).mean(-1).sum(0).double()
        gt_area_sum += tgt_dev.detach().cpu().flatten(2).mean(-1).sum(0).double()
        topmean_sum += topmean_mask_probability(prob_dev.detach(), q=float(loss_weights.topq if loss_weights else 0.02)).cpu().double().sum(0)
        maxprob_sum += prob_dev.detach().cpu().flatten(2).amax(-1).sum(0).double()

        pred_presence_cpu = pred_presence_dev.detach().cpu()
        tgt_presence_cpu = present_cpu
        tp += (pred_presence_cpu & tgt_presence_cpu).sum(0).double()
        fp += (pred_presence_cpu & ~tgt_presence_cpu).sum(0).double()
        fn += (~pred_presence_cpu & tgt_presence_cpu).sum(0).double()
        tn += (~pred_presence_cpu & ~tgt_presence_cpu).sum(0).double()

        labels_cpu = labels_dev.detach().cpu()
        for c in range(num_classes):
            m = labels_cpu == c
            if not bool(m.any()):
                continue
            pc_present[c] += tgt_presence_cpu[m].sum(0).double()
            pc_iou_sum[c] += torch.nan_to_num(torch.where(tgt_presence_cpu[m], iou_cpu[m], torch.full_like(iou_cpu[m], float("nan"))), nan=0.0).sum(0)
            pc_dice_sum[c] += torch.nan_to_num(torch.where(tgt_presence_cpu[m], dice_cpu[m], torch.full_like(dice_cpu[m], float("nan"))), nan=0.0).sum(0)
            pc_tp[c] += (pred_presence_cpu[m] & tgt_presence_cpu[m]).sum(0).double()
            pc_fp[c] += (pred_presence_cpu[m] & ~tgt_presence_cpu[m]).sum(0).double()
            pc_fn[c] += (~pred_presence_cpu[m] & tgt_presence_cpu[m]).sum(0).double()
            pc_tn[c] += (~pred_presence_cpu[m] & ~tgt_presence_cpu[m]).sum(0).double()

        leak, cont = support_containment_losses(
            {k: v.detach() for k, v in out.items() if torch.is_tensor(v)},
            batch_device,
        )
        support_leak_total += float(leak.detach().cpu())
        support_contain_total += float(cont.detach().cpu())
        n_batches += 1
        n_samples += int(image.shape[0])

    eps = 1e-6
    iou_global = torch.full_like(union, float("nan"))
    dice_global = torch.full_like(dice_den, float("nan"))
    iou_global[union > 0] = inter[union > 0] / (union[union > 0] + eps)
    dice_global[dice_den > 0] = dice_num[dice_den > 0] / (dice_den[dice_den > 0] + eps)
    iou_present = torch.full_like(present_count, float("nan"))
    dice_present = torch.full_like(present_count, float("nan"))
    iou_present[present_count > 0] = iou_present_sum[present_count > 0] / (present_count[present_count > 0] + eps)
    dice_present[present_count > 0] = dice_present_sum[present_count > 0] / (present_count[present_count > 0] + eps)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    halluc = fp / (fp + tn + eps)
    miss = fn / (tp + fn + eps)
    row = {f"val_{k}": v / max(n_batches, 1) for k, v in run.items()}
    row.update({
        "num_batches": n_batches,
        "num_samples": n_samples,
        "val_miou_global_nonempty": float(safe_nanmean(iou_global).item()),
        "val_dice_global_nonempty": float(safe_nanmean(dice_global).item()),
        "val_miou_present_parts": float(safe_nanmean(iou_present).item()),
        "val_dice_present_parts": float(safe_nanmean(dice_present).item()),
        # Backward-compatible aliases; they now refer to GT-present part masks,
        # not empty-empty channels.
        "val_miou_all": float(safe_nanmean(iou_present).item()),
        "val_dice_all": float(safe_nanmean(dice_present).item()),
        "val_presence_precision_macro": float(precision.mean().item()),
        "val_presence_recall_macro": float(recall.mean().item()),
        "val_presence_f1_macro": float(f1.mean().item()),
        "val_hallucination_macro": float(halluc.mean().item()),
        "val_miss_macro": float(miss.mean().item()),
        "val_gt_support_leak": support_leak_total / max(n_batches, 1),
        "val_pred_support_containment": support_contain_total / max(n_batches, 1),
    })
    per_part = []
    for k, name in enumerate(part_names):
        per_part.append({
            "part_idx": k,
            "part": name,
            "iou": float(iou_present[k].item()) if torch.isfinite(iou_present[k]) else float("nan"),
            "dice": float(dice_present[k].item()) if torch.isfinite(dice_present[k]) else float("nan"),
            "iou_present": float(iou_present[k].item()) if torch.isfinite(iou_present[k]) else float("nan"),
            "dice_present": float(dice_present[k].item()) if torch.isfinite(dice_present[k]) else float("nan"),
            "iou_global_nonempty": float(iou_global[k].item()) if torch.isfinite(iou_global[k]) else float("nan"),
            "dice_global_nonempty": float(dice_global[k].item()) if torch.isfinite(dice_global[k]) else float("nan"),
            "presence_precision": float(precision[k].item()),
            "presence_recall": float(recall[k].item()),
            "presence_f1": float(f1[k].item()),
            "hallucination_rate": float(halluc[k].item()),
            "miss_rate": float(miss[k].item()),
            "present_count": float(present_count[k].item()),
            "union_nonempty_count": float(union_nonempty_count[k].item()),
            "pred_area_mean": float(pred_area_sum[k].item() / max(n_samples, 1)),
            "gt_area_mean": float(gt_area_sum[k].item() / max(n_samples, 1)),
            "pred_topmean_mean": float(topmean_sum[k].item() / max(n_samples, 1)),
            "pred_presence_mean": float(presence_sum[k].item() / max(n_samples, 1)),
            "pred_maxprob_mean": float(maxprob_sum[k].item() / max(n_samples, 1)),
        })
    per_class_part = []
    for c, cname in enumerate(class_names):
        for k, pname in enumerate(part_names):
            pcount = pc_present[c, k]
            iou_ck = pc_iou_sum[c, k] / (pcount + eps) if pcount > 0 else torch.tensor(float("nan"))
            dice_ck = pc_dice_sum[c, k] / (pcount + eps) if pcount > 0 else torch.tensor(float("nan"))
            prec_ck = pc_tp[c, k] / (pc_tp[c, k] + pc_fp[c, k] + eps)
            rec_ck = pc_tp[c, k] / (pc_tp[c, k] + pc_fn[c, k] + eps)
            f1_ck = 2 * prec_ck * rec_ck / (prec_ck + rec_ck + eps)
            halluc_ck = pc_fp[c, k] / (pc_fp[c, k] + pc_tn[c, k] + eps)
            miss_ck = pc_fn[c, k] / (pc_tp[c, k] + pc_fn[c, k] + eps)
            if pcount > 0 or pc_fp[c, k] > 0:
                per_class_part.append({
                    "class_idx": c,
                    "class": cname,
                    "part_idx": k,
                    "part": pname,
                    "present_count": float(pcount.item()),
                    "iou_present": float(iou_ck.item()) if torch.isfinite(iou_ck) else float("nan"),
                    "dice_present": float(dice_ck.item()) if torch.isfinite(dice_ck) else float("nan"),
                    "presence_precision": float(prec_ck.item()),
                    "presence_recall": float(rec_ck.item()),
                    "presence_f1": float(f1_ck.item()),
                    "hallucination_rate": float(halluc_ck.item()),
                    "miss_rate": float(miss_ck.item()),
                })
    row["per_part"] = per_part
    row["per_class_part"] = per_class_part
    return row


def _save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in history:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k in fields} for row in history])


def train_stage1_quality_upgrade(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: ProjectConfig,
    *,
    device: str = "cuda",
    loss_weights: Stage1QualityLossWeights | None = None,
    start_epoch: int = 1,
    max_epochs: int | None = None,
) -> list[dict[str, float]]:
    """Train or fine-tune Stage 1 with the upgraded quality loss."""
    loss_weights = loss_weights or Stage1QualityLossWeights()
    model.to(device)
    if hasattr(model, "set_stage1_trainable"):
        model.set_stage1_trainable()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.training.lr_stage1),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = make_scaler(cfg.training.use_amp)
    best = -float("inf")
    history: list[dict[str, float]] = []
    train_lw = _loss_weights(train_loader)
    ckpt_dir = Path(cfg.paths.save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = int(max_epochs or cfg.training.stage1_epochs)

    for epoch in range(int(start_epoch), total_epochs + 1):
        model.train()
        run = defaultdict(float)
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            batch_device = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with autocast_cuda(cfg.training.use_amp):
                out = model(image)
                loss, logs = stage1_quality_loss(
                    out,
                    batch_device,
                    model.schema,
                    cfg,
                    loss_weights,
                    part_loss_weight=train_lw.get("part_loss_weight"),
                    part_pos_weight=train_lw.get("part_pos_weight"),
                    role_loss_weight=train_lw.get("role_loss_weight"),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Stage1 quality loss at epoch {epoch}: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for key, val in logs.items():
                run[key] += float(val)
            n_batches += 1
        row: dict[str, float] = {f"train_{k}": v / max(n_batches, 1) for k, v in run.items()}
        row["epoch"] = float(epoch)
        row["wall_sec"] = time.time() - t0
        if val_loader is not None:
            val = evaluate_stage1_quality_detailed(model, val_loader, cfg, device=device, loss_weights=loss_weights)
            # Keep scalar fields in row; detailed per-part table is saved separately below.
            for k, v in val.items():
                if isinstance(v, (int, float)):
                    row[k] = float(v)
            per_part_path = Path(cfg.paths.save_dir) / f"stage1_quality_per_part_epoch_{epoch:03d}.json"
            save_json(per_part_path, val.get("per_part", []))
            per_class_part_path = Path(cfg.paths.save_dir) / f"stage1_quality_per_class_part_epoch_{epoch:03d}.json"
            save_json(per_class_part_path, val.get("per_class_part", []))
        history.append(row)
        save_json(Path(cfg.paths.save_dir) / "stage1_quality_history.json", history)
        _save_history_csv(Path(cfg.paths.save_dir) / "stage1_quality_history.csv", history)
        score = row.get("val_miou_present_parts", row.get("val_miou_all", 0.0)) + 0.25 * row.get("val_presence_f1_macro", 0.0) - 0.10 * row.get("val_hallucination_macro", 0.0)
        extra = {
            "epoch": epoch,
            "history": history,
            "score": float(score),
            "schema": model.schema.to_payload(),
            "config": cfg.to_dict(),
            "stage1_quality_loss_weights": asdict(loss_weights),
        }
        save_checkpoint(ckpt_dir / "stage1_quality_last.pt", model, extra=extra)
        if score > best:
            best = float(score)
            save_checkpoint(ckpt_dir / "stage1_quality_best.pt", model, extra=extra)
        print(
            f"[stage1-quality] epoch={epoch} train_loss={row.get('train_loss', float('nan')):.4f} "
            f"val_mIoU={row.get('val_miou_present_parts', float('nan')):.4f} "
            f"presence_f1={row.get('val_presence_f1_macro', float('nan')):.4f} "
            f"halluc={row.get('val_hallucination_macro', float('nan')):.4f} "
            f"score={score:.4f}"
        )
    return history


@torch.no_grad()
def summarize_stage1_quality_for_hkg(model: torch.nn.Module, loader, *, device: str = "cuda", max_batches: int | None = None, q: float = 0.02) -> dict[str, Any]:
    """Collect mask-quality weights useful for rebuilding HKG prototypes.

    This function does not rebuild the HKG directly; instead it reports which
    part channels are reliable enough for prototype/relation statistics and gives
    per-part quality factors that can be used for filtering or weighting.
    """
    model.eval()
    part_names = model.schema.part_names
    acc = {name: {"count": 0.0, "present": 0.0, "mean_presence": 0.0, "mean_area": 0.0, "mean_top": 0.0} for name in part_names}
    for bidx, batch in enumerate(loader):
        if max_batches is not None and bidx >= int(max_batches):
            break
        image = batch["image"].to(device, non_blocking=True)
        out = model(image)
        prob = out["part_prob"].float()
        top = topmean_mask_probability(prob, q=q)
        area = prob.flatten(2).mean(-1)
        presence = out["part_presence"].float()
        gt_presence = batch.get("presence")
        gt_presence = gt_presence.to(device).float() if torch.is_tensor(gt_presence) else (presence > 0.15).float()
        for k, name in enumerate(part_names):
            m = gt_presence[:, k] > 0.5
            n = float(m.numel())
            acc[name]["count"] += n
            acc[name]["present"] += float(m.sum().item())
            acc[name]["mean_presence"] += float(presence[:, k].sum().item())
            acc[name]["mean_area"] += float(area[:, k].sum().item())
            acc[name]["mean_top"] += float(top[:, k].sum().item())
    rows = []
    for name, d in acc.items():
        count = max(d["count"], 1.0)
        rows.append({
            "part": name,
            "gt_present_rate": d["present"] / count,
            "pred_presence_mean": d["mean_presence"] / count,
            "pred_area_mean": d["mean_area"] / count,
            "pred_topmean_mean": d["mean_top"] / count,
            "prototype_quality_hint": float(max(0.0, min(1.0, (d["mean_top"] / count) * (0.5 + d["present"] / count))))
        })
    return {"per_part_quality_hint": rows}
