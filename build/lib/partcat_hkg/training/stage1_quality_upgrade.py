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

    # Confident false-positive suppression for absent parts.
    absent_topmean_fp: float = 0.30
    absent_mean_fp: float = 0.06

    # Class-aware invalid functional part suppression.
    invalid_part_topmean: float = 0.35
    invalid_part_mean: float = 0.08

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
    if channel_weight is not None:
        loss = loss * channel_weight.to(loss.device, loss.dtype).view(1, -1, 1, 1)
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
    if channel_weight is not None:
        loss = loss * channel_weight.to(loss.device, loss.dtype).view(1, -1)
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
    if gt_presence is not None:
        present = gt_presence.to(loss.device).float().clamp(0, 1)
        loss = (loss * present).sum() / (present.sum() + eps)
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


def absent_part_false_positive_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, q: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress confident masks for parts that are absent in GT."""
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
    loss_abs_top, loss_abs_mean = absent_part_false_positive_losses(out, batch, q=weights.topq)
    loss_inv_top, loss_inv_mean = invalid_part_losses(out, batch, schema, q=weights.topq)
    loss_leak_gt, loss_pred_contain = support_containment_losses(out, batch)
    loss_bdry = boundary_dice_loss_logits(out["part_logits"], gt_part, gt_presence, kernel_size=weights.boundary_kernel)
    loss_focal = binary_focal_bce_logits(
        out["part_logits"],
        gt_part,
        gamma=weights.focal_gamma,
        alpha=weights.focal_alpha,
        channel_weight=part_loss_weight,
    )
    loss_tversky = tversky_loss_logits(
        out["part_logits"],
        gt_part,
        alpha=weights.tversky_alpha,
        beta=weights.tversky_beta,
        channel_weight=part_loss_weight,
    )

    extra = (
        weights.presence_bce * loss_presence
        + weights.absent_topmean_fp * loss_abs_top
        + weights.absent_mean_fp * loss_abs_mean
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
        "absent_topmean_fp": float(loss_abs_top.detach().cpu()),
        "absent_mean_fp": float(loss_abs_mean.detach().cpu()),
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
) -> dict[str, Any]:
    """Evaluate segmentation, presence calibration, hallucination, and containment."""
    model.eval()
    lw = _loss_weights(loader)
    run = defaultdict(float)
    n_batches = 0
    n_samples = 0
    num_parts = model.schema.num_parts
    inter = torch.zeros(num_parts, dtype=torch.float64)
    union = torch.zeros(num_parts, dtype=torch.float64)
    dice_num = torch.zeros(num_parts, dtype=torch.float64)
    dice_den = torch.zeros(num_parts, dtype=torch.float64)
    present_count = torch.zeros(num_parts, dtype=torch.float64)
    pred_area_sum = torch.zeros(num_parts, dtype=torch.float64)
    gt_area_sum = torch.zeros(num_parts, dtype=torch.float64)
    tp = torch.zeros(num_parts, dtype=torch.float64)
    fp = torch.zeros(num_parts, dtype=torch.float64)
    fn = torch.zeros(num_parts, dtype=torch.float64)
    tn = torch.zeros(num_parts, dtype=torch.float64)
    support_leak_total = 0.0
    support_contain_total = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        image = batch["image"].to(device, non_blocking=True)
        with autocast_cuda(False):
            out = model(image)
            loss, logs = stage1_quality_loss(
                out,
                batch,
                model.schema,
                cfg,
                loss_weights,
                part_loss_weight=lw.get("part_loss_weight"),
                part_pos_weight=lw.get("part_pos_weight"),
                role_loss_weight=lw.get("role_loss_weight"),
            )
        for k, v in logs.items():
            run[k] += float(v)
        prob = out["part_prob"].detach().cpu()
        target = batch["part_masks"].float()
        if target.shape[-2:] != prob.shape[-2:]:
            target = F.interpolate(target, size=prob.shape[-2:], mode="nearest")
        pred = (prob >= 0.5).float()
        tgt = target.clamp(0, 1)
        b_inter = (pred * tgt).flatten(2).sum(-1).double()
        b_pred = pred.flatten(2).sum(-1).double()
        b_tgt = tgt.flatten(2).sum(-1).double()
        inter += b_inter.sum(0)
        union += (b_pred + b_tgt - b_inter).sum(0)
        dice_num += (2.0 * b_inter).sum(0)
        dice_den += (b_pred + b_tgt).sum(0)
        gt_presence = batch["presence"].float()
        pred_presence = out["part_presence"].detach().cpu() >= float(cfg.model.stage1.presence_threshold)
        tgt_presence = gt_presence > 0.5
        tp += (pred_presence & tgt_presence).sum(0).double()
        fp += (pred_presence & ~tgt_presence).sum(0).double()
        fn += (~pred_presence & tgt_presence).sum(0).double()
        tn += (~pred_presence & ~tgt_presence).sum(0).double()
        present_count += tgt_presence.sum(0).double()
        pred_area_sum += prob.flatten(2).mean(-1).sum(0).double()
        gt_area_sum += tgt.flatten(2).mean(-1).sum(0).double()
        leak, cont = support_containment_losses({k: v.detach() for k, v in out.items() if torch.is_tensor(v)}, {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
        support_leak_total += float(leak.detach().cpu())
        support_contain_total += float(cont.detach().cpu())
        n_batches += 1
        n_samples += int(image.shape[0])

    eps = 1e-6
    iou = (inter + eps) / (union + eps)
    dice = (dice_num + eps) / (dice_den + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    halluc = fp / (fp + tn + eps)
    miss = fn / (tp + fn + eps)
    row = {f"val_{k}": v / max(n_batches, 1) for k, v in run.items()}
    row.update({
        "num_batches": n_batches,
        "num_samples": n_samples,
        "val_miou_all": float(iou.mean().item()),
        "val_dice_all": float(dice.mean().item()),
        "val_miou_present_parts": float((iou * (present_count > 0).double()).sum().item() / max(float((present_count > 0).sum().item()), 1.0)),
        "val_dice_present_parts": float((dice * (present_count > 0).double()).sum().item() / max(float((present_count > 0).sum().item()), 1.0)),
        "val_presence_precision_macro": float(precision.mean().item()),
        "val_presence_recall_macro": float(recall.mean().item()),
        "val_presence_f1_macro": float(f1.mean().item()),
        "val_hallucination_macro": float(halluc.mean().item()),
        "val_miss_macro": float(miss.mean().item()),
        "val_gt_support_leak": support_leak_total / max(n_batches, 1),
        "val_pred_support_containment": support_contain_total / max(n_batches, 1),
    })
    per_part = []
    for k, name in enumerate(model.schema.part_names):
        per_part.append({
            "part_idx": k,
            "part": name,
            "iou": float(iou[k].item()),
            "dice": float(dice[k].item()),
            "presence_precision": float(precision[k].item()),
            "presence_recall": float(recall[k].item()),
            "presence_f1": float(f1[k].item()),
            "hallucination_rate": float(halluc[k].item()),
            "miss_rate": float(miss[k].item()),
            "present_count": float(present_count[k].item()),
            "pred_area_mean": float(pred_area_sum[k].item() / max(n_samples, 1)),
            "gt_area_mean": float(gt_area_sum[k].item() / max(n_samples, 1)),
        })
    row["per_part"] = per_part
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
