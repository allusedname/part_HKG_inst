from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from partcat_hkg.models.pooling import topk_presence


def _topk_masked(scores: torch.Tensor, mask: torch.Tensor, k: int) -> list[int]:
    scores = scores.detach().float().cpu().clone()
    mask = mask.detach().bool().cpu()
    if not bool(mask.any()):
        return []
    scores[~mask] = -1e9
    return torch.topk(scores, min(int(k), int(mask.sum()))).indices.tolist()


def _overlay(prob_map: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros_like(prob_map[0])
    out = torch.zeros_like(prob_map[0])
    for idx in indices:
        out = torch.maximum(out, prob_map[int(idx)])
    return out


def visualize_stage1_batch(model, batch: dict, *, max_items: int = 3, path=None, title: str = "Stage 1"):
    """Visualize Stage-1 support, functional masks, and valid/invalid role leakage."""

    model.eval()
    device = next(model.parameters()).device
    n = min(int(max_items), batch["image"].shape[0])
    with torch.no_grad():
        out = model(batch["image"][:n].to(device))
    part_prob = out["part_prob"].detach().cpu()
    role_prob = out["role_prob"].detach().cpu()
    if role_prob.shape[-2:] != part_prob.shape[-2:]:
        role_prob = F.interpolate(role_prob, size=part_prob.shape[-2:], mode="bilinear", align_corners=False)
    support = out["support_prob"].detach().cpu()
    schema = model.schema
    topn = min(4, schema.num_parts)
    fig, axes = plt.subplots(n, 6, figsize=(22, max(3.5, 3.5 * n)), squeeze=False)
    for i in range(n):
        image = batch["image_raw"][i].permute(1, 2, 0).cpu()
        obj_idx = int(batch["obj_label"][i])
        obj_name = schema.obj_names[obj_idx]
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"image\nGT={obj_name}")
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(batch["union_mask"][i, 0].cpu(), alpha=0.45)
        axes[i, 1].set_title("GT union")
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(support[i, 0], alpha=0.45)
        axes[i, 2].set_title("pred support")

        pscore = topk_presence(part_prob[i:i + 1], k=model.cfg.topk_presence_k).squeeze(0)
        gt_present = batch.get("presence", torch.zeros_like(pscore)).detach().bool().cpu()
        if gt_present.ndim == 2:
            gt_present = gt_present[i]
        pidx = torch.topk(pscore + 0.25 * gt_present.float(), k=topn).indices.tolist()
        axes[i, 3].imshow(image)
        axes[i, 3].imshow(_overlay(part_prob[i], pidx), alpha=0.5)
        axes[i, 3].set_title("top functional\n" + ", ".join(f"{schema.part_names[j]}:{float(pscore[j]):.2f}" for j in pidx[:3]))

        rscore = topk_presence(role_prob[i:i + 1], k=model.cfg.topk_presence_k).squeeze(0)
        valid = (schema.role_to_obj.cpu() == obj_idx)
        ridx = _topk_masked(rscore, valid, topn)
        axes[i, 4].imshow(image)
        axes[i, 4].imshow(_overlay(role_prob[i], ridx), alpha=0.5)
        axes[i, 4].set_title("valid roles\n" + ", ".join(f"{schema.role_names[j]}:{float(rscore[j]):.2f}" for j in ridx[:2]))

        inv = ~valid
        inv_idx = _topk_masked(rscore, inv, min(3, schema.num_roles))
        axes[i, 5].imshow(image)
        axes[i, 5].imshow(_overlay(role_prob[i], inv_idx), alpha=0.5)
        axes[i, 5].set_title("invalid-role leakage\n" + ", ".join(f"{schema.role_names[j]}:{float(rscore[j]):.2f}" for j in inv_idx[:2]))
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
