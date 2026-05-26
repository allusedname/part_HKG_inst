from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from partcat_hkg.data.canonicalization import display_object_name


def visualize_parse_batch(model, batch: dict, *, max_items: int = 3, path=None):
    model.eval()
    n = min(max_items, batch["image"].shape[0])
    small = {k: (v[:n] if torch.is_tensor(v) else v[:n]) for k, v in batch.items()}
    with torch.no_grad():
        out = model(small, detach_stage1=True)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n), squeeze=False)
    for i in range(n):
        pred = int(out["logits"][i].argmax().item())
        axes[i, 0].imshow(batch["image_raw"][i].permute(1, 2, 0).cpu())
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(out["part_prob"][i].amax(0).detach().cpu(), cmap="gray")
        axes[i, 1].set_title("active parts")
        top = out["logits"][i].detach().cpu().topk(min(5, out["logits"].shape[1]))
        labels = [display_object_name(model.schema.obj_names[int(j)]) for j in top.indices]
        axes[i, 2].barh(labels, top.values)
        axes[i, 2].set_title(f"parse pred={display_object_name(model.schema.obj_names[pred])}")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig
