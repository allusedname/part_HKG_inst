from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import time
from typing import Any

import torch

from .parser import StrictAOGParser, strict_aog_loss


def _accuracy(correct: int, total: int) -> float:
    return float(correct) / float(max(total, 1))


def _save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def evaluate_strict_aog(model: StrictAOGParser, loader, *, device: str | torch.device = "cuda", enable_edges: bool = True) -> dict[str, float]:
    model.eval()
    total = correct = 0
    logit_std_sum = 0.0
    n_batches = 0
    for batch in loader:
        labels = batch["obj_label"].to(device, non_blocking=True)
        out = model(batch, enable_edges=enable_edges)
        pred = out["logits"].argmax(-1)
        total += int(labels.numel())
        correct += int((pred == labels).sum().item())
        logit_std_sum += float(out["logits"].std(dim=-1).mean().detach().cpu())
        n_batches += 1
    return {"val_acc": _accuracy(correct, total), "val_logit_std": logit_std_sum / max(n_batches, 1)}


def train_strict_aog(
    model: StrictAOGParser,
    train_loader,
    val_loader,
    *,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cuda",
    save_dir: str | Path = "runs/strict_aog",
    enable_edges: bool = True,
    fail_on_uniform: bool = True,
) -> list[dict[str, float]]:
    """Train only the grammar calibration/projection parameters.

    The AOG grammar statistics are fixed.  This avoids the old failure mode where
    a black-box base branch silently dominates or all logits collapse to a
    constant.  The loop logs and optionally stops if the logits are uniform.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    save_dir = Path(save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best = -1.0
    uniform_floor = 1e-5
    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        run = defaultdict(float)
        nb = 0
        total = correct = 0
        for batch in train_loader:
            labels = batch["obj_label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(batch, enable_edges=enable_edges)
            loss, logs = strict_aog_loss(out, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Strict AOG non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            pred = out["logits"].argmax(-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            for k, v in logs.items():
                run[k] += float(v)
            nb += 1
        row = {f"train_{k}": v / max(nb, 1) for k, v in run.items()}
        row["train_acc"] = _accuracy(correct, total)
        row["epoch"] = float(epoch)
        row["wall_sec"] = time.time() - t0
        if val_loader is not None:
            row.update(evaluate_strict_aog(model, val_loader, device=device, enable_edges=enable_edges))
        history.append(row)
        _save_csv(save_dir / "strict_aog_history.csv", history)
        score = float(row.get("val_acc", row.get("train_acc", 0.0)))
        ckpt = {"model": model.state_dict(), "epoch": epoch, "history": history, "grammar": model.grammar.to_payload()}
        torch.save(ckpt, save_dir / "checkpoints" / "strict_aog_last.pt")
        if score >= best:
            best = score
            torch.save(ckpt, save_dir / "checkpoints" / "strict_aog_best.pt")
        msg = (
            f"[strict-aog] epoch={epoch} train_loss={row.get('train_loss', float('nan')):.4f} "
            f"train_acc={row.get('train_acc', float('nan')):.4f} "
            f"val_acc={row.get('val_acc', float('nan')):.4f} "
            f"logit_std={row.get('val_logit_std', row.get('train_logit_std', float('nan'))):.6f}"
        )
        print(msg)
        if fail_on_uniform and epoch >= 2:
            std = float(row.get("val_logit_std", row.get("train_logit_std", 0.0)))
            if std < uniform_floor:
                raise RuntimeError(
                    "Strict AOG logits are effectively uniform. Check terminal cache coverage, "
                    "grammar slot_valid/template_valid, and label/schema alignment."
                )
    return history
