from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import time

import torch

from partcat_hkg.config import ProjectConfig
from partcat_hkg.evaluation.metrics import binary_segmentation_stats, presence_f1
from partcat_hkg.models.losses import stage1_loss
from partcat_hkg.utils.amp import autocast_cuda, make_scaler
from partcat_hkg.utils.io import save_checkpoint, save_json


def _dataset_tensor_attr(loader, name: str) -> torch.Tensor | None:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return None
    val = getattr(ds, name, None)
    if val is None and hasattr(ds, "base"):
        val = getattr(ds.base, name, None)
    return torch.as_tensor(val) if val is not None else None


def _loss_weights(loader) -> dict[str, torch.Tensor | None]:
    return {
        "part_loss_weight": _dataset_tensor_attr(loader, "part_loss_weight"),
        "part_pos_weight": _dataset_tensor_attr(loader, "part_pos_weight"),
        "role_loss_weight": _dataset_tensor_attr(loader, "role_loss_weight"),
    }


def _quality_loss_enabled(cfg: ProjectConfig) -> bool:
    return bool(getattr(cfg.loss.stage1, "quality_enable", False))


def _quality_weights_from_config(cfg: ProjectConfig):
    # Lazy import avoids a circular import: stage1_quality_upgrade also imports
    # _loss_weights for its detailed diagnostic evaluator.
    from partcat_hkg.training.stage1_quality_upgrade import Stage1QualityLossWeights

    lc = cfg.loss.stage1
    sc = cfg.model.stage1
    return Stage1QualityLossWeights(
        presence_bce=float(getattr(lc, "quality_presence_bce", 0.40)),
        valid_absent_topmean_fp=float(getattr(lc, "valid_absent_topmean_fp", 0.08)),
        valid_absent_mean_fp=float(getattr(lc, "valid_absent_mean_fp", 0.02)),
        invalid_part_topmean=float(getattr(lc, "invalid_part_topmean", 0.35)),
        invalid_part_mean=float(getattr(lc, "invalid_part_mean", 0.08)),
        gt_support_leak=float(getattr(lc, "gt_support_leak", 0.35)),
        pred_support_containment=float(getattr(lc, "pred_support_containment", 0.25)),
        boundary=float(getattr(lc, "boundary", 0.08)),
        focal_functional=float(getattr(lc, "focal_functional", 0.12)),
        tversky_functional=float(getattr(lc, "tversky_functional", 0.12)),
        topq=float(getattr(lc, "quality_topq", 0.02)),
        focal_gamma=float(getattr(lc, "focal_gamma", 2.0)),
        focal_alpha=float(getattr(lc, "focal_alpha", 0.25)),
        tversky_alpha=float(getattr(lc, "tversky_alpha", 0.35)),
        tversky_beta=float(getattr(lc, "tversky_beta", 0.65)),
        boundary_kernel=int(getattr(lc, "boundary_kernel", 3)),
        small_part_area_tau=float(getattr(sc, "small_part_area_tau", 0.015)),
        small_part_weight_max=float(getattr(sc, "small_part_weight_max", 6.0)),
        small_part_weight_power=float(getattr(sc, "small_part_weight_power", 0.5)),
    )


def compute_stage1_training_loss(out, batch, model, cfg: ProjectConfig, weights: dict[str, torch.Tensor | None]):
    if _quality_loss_enabled(cfg):
        from partcat_hkg.training.stage1_quality_upgrade import stage1_quality_loss

        return stage1_quality_loss(
            out,
            batch,
            model.schema,
            cfg,
            _quality_weights_from_config(cfg),
            **weights,
        )
    return stage1_loss(
        out,
        batch,
        model.schema,
        cfg.loss.stage1,
        **weights,
        topk_presence_k=cfg.model.stage1.topk_presence_k,
    )


@torch.no_grad()
def evaluate_stage1(model, loader, cfg: ProjectConfig, *, device: str = "cuda", max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    weights = _loss_weights(loader)
    run = defaultdict(float)
    n_batches = 0
    iou_sum = dice_sum = present_iou_sum = present_dice_sum = 0.0
    iou_count = dice_count = present_count = 0.0
    pres_tp = pres_fp = pres_fn = pres_neg = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        image = batch["image"].to(device, non_blocking=True)
        with autocast_cuda(False):
            out = model(image)
            loss, logs = compute_stage1_training_loss(out, batch, model, cfg, weights)
        for key, val in logs.items():
            run[key] += float(val)
        n_batches += 1

        target = batch["part_masks"].to(device).float()
        stats = binary_segmentation_stats(out["part_prob"], target)
        gt_presence = batch["presence"].to(device).float()
        present = gt_presence > 0.5
        iou_sum += float(stats["iou"].sum().item())
        dice_sum += float(stats["dice"].sum().item())
        iou_count += float(stats["iou"].numel())
        dice_count += float(stats["dice"].numel())
        if present.any():
            present_iou_sum += float(stats["iou"][present].sum().item())
            present_dice_sum += float(stats["dice"][present].sum().item())
            present_count += float(present.sum().item())

        pred_present = out["part_presence"] >= float(cfg.model.stage1.presence_threshold)
        tgt_present = gt_presence > 0.5
        pres_tp += float((pred_present & tgt_present).sum().item())
        pres_fp += float((pred_present & ~tgt_present).sum().item())
        pres_fn += float((~pred_present & tgt_present).sum().item())
        pres_neg += float((~tgt_present).sum().item())

    row = {f"val_{k}": v / max(n_batches, 1) for k, v in run.items()}
    row.update({
        "val_miou_all": iou_sum / max(iou_count, 1.0),
        "val_dice_all": dice_sum / max(dice_count, 1.0),
        "val_miou_present": present_iou_sum / max(present_count, 1.0),
        "val_dice_present": present_dice_sum / max(present_count, 1.0),
    })
    precision = pres_tp / max(pres_tp + pres_fp, 1e-6)
    recall = pres_tp / max(pres_tp + pres_fn, 1e-6)
    row["val_presence_precision"] = precision
    row["val_presence_recall"] = recall
    row["val_presence_f1"] = 2 * precision * recall / max(precision + recall, 1e-6)
    row["val_hallucination_rate"] = pres_fp / max(pres_neg, 1e-6)
    return row


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


def train_stage1(model, train_loader, val_loader, cfg: ProjectConfig, *, device: str = "cuda") -> list[dict[str, float]]:
    model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr_stage1,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = make_scaler(cfg.training.use_amp)
    best = float("inf")
    history: list[dict[str, float]] = []
    train_weights = _loss_weights(train_loader)
    ckpt_dir = Path(cfg.paths.save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.stage1_epochs + 1):
        model.train()
        run = defaultdict(float)
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            with autocast_cuda(cfg.training.use_amp):
                out = model(image)
                loss, logs = compute_stage1_training_loss(out, batch, model, cfg, train_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Stage1 non-finite loss at epoch {epoch}: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for key, val in logs.items():
                run[key] += float(val)
            n_batches += 1

        row = {f"train_{k}": v / max(n_batches, 1) for k, v in run.items()}
        row["epoch"] = float(epoch)
        row["wall_sec"] = time.time() - t0
        if val_loader is not None:
            row.update(evaluate_stage1(model, val_loader, cfg, device=device))
        history.append(row)
        save_json(Path(cfg.paths.save_dir) / "stage1_history.json", history)
        _save_history_csv(Path(cfg.paths.save_dir) / "stage1_history.csv", history)

        score = row.get("val_loss", row.get("train_loss", float("inf")))
        extra = {"epoch": epoch, "history": history, "score": score, "schema": model.schema.to_payload(), "config": cfg.to_dict()}
        save_checkpoint(ckpt_dir / "stage1_last.pt", model, extra=extra)
        if score < best:
            best = score
            save_checkpoint(ckpt_dir / "stage1_best.pt", model, extra=extra)
        print(
            f"[stage1] epoch={epoch} train_loss={row.get('train_loss', float('nan')):.4f} "
            f"val_loss={row.get('val_loss', float('nan')):.4f} "
            f"val_mIoU={row.get('val_miou_present', float('nan')):.4f} "
            f"presence_f1={row.get('val_presence_f1', float('nan')):.4f} "
            f"quality={int(_quality_loss_enabled(cfg))}"
        )
    return history
