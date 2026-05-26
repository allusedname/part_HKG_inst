from __future__ import annotations

import torch
import torch.nn.functional as F

from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.pooling import topk_presence
from partcat_hkg.models.quality import role_valid_mask_for_batch


@torch.no_grad()
def stage1_audit(
    model,
    loader,
    *,
    schema: RoleSchema | None = None,
    device: str | torch.device = "cuda",
    mask_bin_thr: float = 0.40,
    presence_tau: float = 0.15,
    topk_presence_k: int = 64,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate Stage 1 masks, part presence, and invalid-role leakage."""
    schema = schema or model.schema
    device = torch.device(device)
    model.eval()
    fnum = schema.num_parts
    inter = torch.zeros(fnum)
    union = torch.zeros(fnum)
    present_inter = torch.zeros(fnum)
    present_union = torch.zeros(fnum)
    tp = fp = fn = 0.0
    hall = 0.0
    total = 0
    invalid_role_mass = 0.0
    valid_role_mass = 0.0
    nbatch = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["part_masks"].to(device, non_blocking=True).float()
        pres_gt = batch["presence"].to(device, non_blocking=True).float()
        labels = batch["obj_label"].to(device, non_blocking=True)
        out = model(image)
        prob = out.get("part_prob", torch.sigmoid(out["part_logits"]))
        if prob.shape[-2:] != target.shape[-2:]:
            prob = F.interpolate(prob.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
        pred = (prob > float(mask_bin_thr)).float()
        pres_pred = (topk_presence(prob, k=topk_presence_k) > float(presence_tau)).float()
        i = (pred * target).flatten(2).sum(-1).cpu()
        u = ((pred + target) > 0).float().flatten(2).sum(-1).cpu()
        inter += i.sum(0)
        union += u.sum(0)
        present = pres_gt.cpu() > 0.5
        present_inter += (i * present).sum(0)
        present_union += (u * present).sum(0)
        tp += float(((pres_pred == 1) & (pres_gt == 1)).sum().item())
        fp += float(((pres_pred == 1) & (pres_gt == 0)).sum().item())
        fn += float(((pres_pred == 0) & (pres_gt == 1)).sum().item())
        hall += float(((pres_pred == 1) & (pres_gt == 0)).sum().item())
        total += int(pres_gt.numel())
        if "role_prob" in out:
            rp = out["role_prob"].float()
            valid = role_valid_mask_for_batch(labels, schema).view(labels.shape[0], schema.num_roles, 1, 1)
            invalid_role_mass += float((rp * (1.0 - valid)).mean().detach().cpu())
            valid_role_mass += float((rp * valid).sum().detach().cpu() / (valid.sum().detach().cpu() * rp.shape[-1] * rp.shape[-2] + 1e-6))
            nbatch += 1
    iou = inter / union.clamp_min(1)
    present_iou = present_inter / present_union.clamp_min(1)
    p_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    part_weight = getattr(loader.dataset, "part_loss_weight", None)
    if part_weight is not None:
        small_w = torch.as_tensor(part_weight).float()
        small_w = small_w / small_w.mean().clamp_min(1e-6)
        small_weighted_miou = float((iou * small_w).sum() / small_w.sum().clamp_min(1e-6))
    else:
        small_weighted_miou = float(iou.mean())
    return {
        "stage1_miou": float(iou.mean()),
        "stage1_present_miou": float(present_iou.mean()),
        "stage1_small_weighted_miou": small_weighted_miou,
        "stage1_presence_f1": float(p_f1),
        "stage1_hallucination_rate": float(hall / max(total, 1)),
        "stage1_valid_role_mass": float(valid_role_mass / max(nbatch, 1)),
        "stage1_invalid_role_mass": float(invalid_role_mass / max(nbatch, 1)),
    }
