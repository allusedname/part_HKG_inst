from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(correct: int, total: int) -> float:
    return float(correct) / max(int(total), 1)


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean().item())


def macro_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(-1)
    vals = []
    for c in range(num_classes):
        m = labels == c
        if m.any():
            vals.append(float((pred[m] == labels[m]).float().mean().item()))
    return sum(vals) / max(len(vals), 1)


def rescue_damage_rates(base_logits: torch.Tensor, parse_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    base_ok = base_logits.argmax(-1) == labels
    parse_ok = parse_logits.argmax(-1) == labels
    rescue = ((~base_ok) & parse_ok).float().mean().item()
    damage = (base_ok & (~parse_ok)).float().mean().item()
    return {"hkg_rescue_rate": float(rescue), "hkg_damage_rate": float(damage)}


@torch.no_grad()
def binary_segmentation_stats(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Per-channel IoU/Dice/F1 ingredients for Stage-1 masks."""
    if target.shape[-2:] != prob.shape[-2:]:
        target = F.interpolate(target.float(), size=prob.shape[-2:], mode="nearest")
    pred = (prob.float() >= float(threshold)).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).flatten(2).sum(-1)
    pred_sum = pred.flatten(2).sum(-1)
    target_sum = target.flatten(2).sum(-1)
    union = pred_sum + target_sum - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred_sum + target_sum + eps)
    return {"intersection": inter, "union": union, "iou": iou, "dice": dice, "pred_sum": pred_sum, "target_sum": target_sum}


@torch.no_grad()
def presence_f1(pred_presence: torch.Tensor, target_presence: torch.Tensor, threshold: float = 0.15, eps: float = 1e-6) -> dict[str, float]:
    pred = pred_presence.float() >= float(threshold)
    tgt = target_presence.float() > 0.5
    tp = (pred & tgt).sum().float()
    fp = (pred & ~tgt).sum().float()
    fn = (~pred & tgt).sum().float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    hallucination_rate = fp / ((~tgt).sum().float() + eps)
    miss_rate = fn / (tgt.sum().float() + eps)
    return {
        "presence_precision": float(precision.item()),
        "presence_recall": float(recall.item()),
        "presence_f1": float(f1.item()),
        "hallucination_rate": float(hallucination_rate.item()),
        "miss_rate": float(miss_rate.item()),
    }
