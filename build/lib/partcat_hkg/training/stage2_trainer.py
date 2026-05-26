from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
import torch

from partcat_hkg.config import ProjectConfig
from partcat_hkg.stage2.losses import stage2_parse_loss
from partcat_hkg.training.curriculum import expand_curriculum
from partcat_hkg.utils.amp import autocast_cuda, make_scaler
from partcat_hkg.utils.io import save_checkpoint
from partcat_hkg.evaluation.metrics import accuracy


def train_stage2(model, train_loader, val_loader, cfg: ProjectConfig, *, device: str = "cuda"):
    model.to(device)
    for p in model.stage1.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.training.lr_stage2, weight_decay=cfg.training.weight_decay)
    scaler = make_scaler(cfg.training.use_amp)
    curriculum = expand_curriculum(cfg.training.curriculum)
    best = -1.0
    history = []
    for epoch, flags in enumerate(curriculum, start=1):
        model.train()
        run = defaultdict(float)
        n = 0
        t0 = time.time()
        for batch in train_loader:
            labels = batch["obj_label"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast_cuda(cfg.training.use_amp):
                out = model(batch, detach_stage1=True, enable_completion=flags.enable_completion, enable_edges=flags.enable_edges)
                loss, logs = stage2_parse_loss(out, labels, cfg.loss.stage2)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in logs.items():
                run[k] += v
            n += 1
        val_acc = evaluate_stage2_quick(model, val_loader, device=device, enable_completion=flags.enable_completion, enable_edges=flags.enable_edges)
        row = {f"train_{k}": v / max(n, 1) for k, v in run.items()}
        row.update({"epoch": epoch, "val_acc": val_acc, "wall_sec": time.time() - t0, "completion": flags.enable_completion, "edges": flags.enable_edges})
        history.append(row)
        if val_acc > best:
            best = val_acc
            save_checkpoint(Path(cfg.paths.save_dir) / "checkpoints" / "stage2_best.pt", model, extra={"epoch": epoch, "val_acc": val_acc})
    return history


@torch.no_grad()
def evaluate_stage2_quick(model, loader, *, device: str = "cuda", enable_completion: bool = True, enable_edges: bool = True) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        labels = batch["obj_label"].to(device)
        out = model(batch, detach_stage1=True, enable_completion=enable_completion, enable_edges=enable_edges)
        correct += int((out["logits"].argmax(-1) == labels).sum().item())
        total += int(labels.numel())
    return accuracy(correct, total)
