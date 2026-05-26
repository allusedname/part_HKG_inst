from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import time

import torch

from partcat_hkg.config import ProjectConfig
from partcat_hkg.evaluation.metrics import accuracy
from partcat_hkg.stage2.losses import stage2_aog_hkg_loss
from partcat_hkg.utils.amp import autocast_cuda, make_scaler
from partcat_hkg.utils.io import save_checkpoint, save_json


def _save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def evaluate_aog_hkg_stage2(model, loader, cfg: ProjectConfig, *, device: str = "cuda", enable_edges: bool = True, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    correct = base_correct = hkg_correct = total = 0
    run = defaultdict(float)
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        labels = batch["obj_label"].to(device, non_blocking=True)
        out = model(batch, detach_stage1=True, enable_edges=enable_edges)
        loss, logs = stage2_aog_hkg_loss(out, labels, cfg.loss.stage2)
        for k, v in logs.items():
            run[k] += float(v)
        correct += int((out["logits"].argmax(-1) == labels).sum().item())
        base_correct += int((out["base_logits"].argmax(-1) == labels).sum().item())
        hkg_correct += int((out["hkg_logits"].argmax(-1) == labels).sum().item())
        total += int(labels.numel())
        n += 1
    row = {f"val_{k}": v / max(n, 1) for k, v in run.items()}
    row.update({
        "val_acc": accuracy(correct, total),
        "val_base_acc": accuracy(base_correct, total),
        "val_hkg_acc": accuracy(hkg_correct, total),
    })
    return row


def _epoch_edges_enabled(cfg: ProjectConfig, epoch: int) -> bool:
    """Return whether edge/motif factors should be active at this epoch."""
    steps = getattr(cfg.training, "curriculum", None) or []
    cursor = 0
    for step in steps:
        cursor += int(getattr(step, "epochs", 0))
        if int(epoch) <= cursor:
            return bool(getattr(step, "enable_edges", True))
    return True


def train_aog_hkg_stage2(model, train_loader, val_loader, cfg: ProjectConfig, *, device: str = "cuda") -> list[dict[str, float]]:
    model.to(device)
    model.freeze_stage1()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.training.lr_stage2, weight_decay=cfg.training.weight_decay)
    scaler = make_scaler(cfg.training.use_amp)
    best = -1.0
    history: list[dict[str, float]] = []
    ckpt_dir = Path(cfg.paths.save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg.training.stage2_epochs) + 1):
        model.train()
        model.stage1.eval()
        run = defaultdict(float)
        n = 0
        t0 = time.time()
        enable_edges_epoch = _epoch_edges_enabled(cfg, epoch)
        for batch in train_loader:
            labels = batch["obj_label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_cuda(cfg.training.use_amp):
                out = model(batch, detach_stage1=True, enable_edges=enable_edges_epoch)
                loss, logs = stage2_aog_hkg_loss(out, labels, cfg.loss.stage2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"AOG-HKG Stage2 non-finite loss at epoch {epoch}: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in logs.items():
                run[k] += float(v)
            n += 1
        row = {f"train_{k}": v / max(n, 1) for k, v in run.items()}
        row.update({"epoch": float(epoch), "wall_sec": time.time() - t0, "train_edges_enabled": float(enable_edges_epoch)})
        if val_loader is not None:
            row.update(evaluate_aog_hkg_stage2(model, val_loader, cfg, device=device, enable_edges=True))
        history.append(row)
        save_json(Path(cfg.paths.save_dir) / "stage2_aog_hkg_history.json", history)
        _save_history_csv(Path(cfg.paths.save_dir) / "stage2_aog_hkg_history.csv", history)
        score = float(row.get("val_acc", 0.0))
        extra = {"epoch": epoch, "history": history, "score": score, "schema": model.schema.to_payload(), "config": cfg.to_dict()}
        save_checkpoint(ckpt_dir / "stage2_aog_hkg_last.pt", model, extra=extra)
        if score >= best:
            best = score
            save_checkpoint(ckpt_dir / "stage2_aog_hkg_best.pt", model, extra=extra)
        print(
            f"[stage2-aog-hkg] epoch={epoch} "
            f"train_loss={row.get('train_loss', float('nan')):.4f} "
            f"val_acc={row.get('val_acc', float('nan')):.4f} "
            f"val_base={row.get('val_base_acc', float('nan')):.4f} "
            f"val_hkg={row.get('val_hkg_acc', float('nan')):.4f} "
            f"edges={int(enable_edges_epoch)}"
        )
    return history
