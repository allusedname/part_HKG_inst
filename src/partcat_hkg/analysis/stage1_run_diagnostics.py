from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

# Always use a non-interactive backend for script/notebook execution.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from partcat_hkg.data.collate import collate_part_batch
from partcat_hkg.training.stage1_quality_upgrade import (
    Stage1QualityLossWeights,
    evaluate_stage1_quality_detailed,
    move_batch_to_device,
    summarize_stage1_quality_for_hkg,
    topmean_mask_probability,
)
from partcat_hkg.utils.io import save_json


def _as_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_float(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


def _balanced_subset_indices(dataset, samples_per_class: int) -> list[int]:
    """Return up to N validation indices per class, preserving dataset order."""
    n = int(samples_per_class)
    if n <= 0:
        return list(range(len(dataset)))
    counts: dict[int, int] = {}
    selected: list[int] = []
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return list(range(min(len(dataset), n)))
    for i, rec in enumerate(samples):
        c = int(rec.get("obj_label", -1))
        if counts.get(c, 0) < n:
            selected.append(i)
            counts[c] = counts.get(c, 0) + 1
    return selected


def make_diagnostic_loader(val_ds, *, batch_size: int, num_workers: int = 0, balanced_samples_per_class: int = 0):
    if balanced_samples_per_class and balanced_samples_per_class > 0:
        indices = _balanced_subset_indices(val_ds, int(balanced_samples_per_class))
        ds = Subset(val_ds, indices)
    else:
        ds = val_ds
    kwargs = dict(batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=torch.cuda.is_available(), collate_fn=collate_part_batch)
    if num_workers and num_workers > 0:
        kwargs.update(persistent_workers=False)
    return DataLoader(ds, **kwargs)


def _get_image_for_display(batch: dict[str, Any], idx: int) -> torch.Tensor:
    img = batch.get("image_raw")
    if torch.is_tensor(img):
        out = img[idx].detach().cpu().float()
    else:
        out = batch["image"][idx].detach().cpu().float()
        # Best-effort denormalization for ImageNet normalized tensors.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        out = out * std + mean
    return out.clamp(0, 1)


def _overlay_rgb(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> torch.Tensor:
    p = pred_bin.detach().cpu().bool()
    g = gt_bin.detach().cpu().bool()
    both = p & g
    pred_only = p & ~g
    gt_only = g & ~p
    overlay = torch.zeros(3, *p.shape, dtype=torch.float32)
    overlay[0][pred_only] = 1.0  # red = prediction only
    overlay[1][gt_only] = 1.0    # green = GT only
    overlay[2][both] = 1.0       # blue = overlap
    return overlay


def _class_part_valid_mask(schema, class_idx: int) -> torch.Tensor:
    table = getattr(schema, "role_index_table", None)
    if table is None:
        return torch.ones(schema.num_parts, dtype=torch.bool)
    return (table[int(class_idx)].cpu() >= 0)


def _select_parts_for_sample(
    schema,
    presence: torch.Tensor,
    gt_presence: torch.Tensor,
    class_idx: int,
    *,
    max_parts: int = 8,
    include_valid_small_parts: bool = True,
) -> list[int]:
    """Show GT-present parts first, then top predicted, then valid class parts."""
    pres = presence.detach().cpu().float()
    gt = gt_presence.detach().cpu().bool()
    selected: list[int] = []
    for k in torch.where(gt)[0].tolist():
        if k not in selected:
            selected.append(int(k))
    for k in torch.argsort(pres, descending=True).tolist():
        if k not in selected:
            selected.append(int(k))
        if len(selected) >= max_parts:
            break
    if include_valid_small_parts and len(selected) < max_parts:
        valid = _class_part_valid_mask(schema, class_idx)
        for k in torch.where(valid)[0].tolist():
            if k not in selected:
                selected.append(int(k))
            if len(selected) >= max_parts:
                break
    return selected[:max_parts]


@torch.no_grad()
def save_stage1_sample_visualizations(
    model,
    loader,
    out_dir: Path,
    *,
    device: str,
    num_samples: int = 6,
    max_parts: int = 8,
    mask_threshold: float = 0.4,
    topq: float = 0.02,
) -> list[dict[str, Any]]:
    """Save detailed per-sample visualizations with individual part masks."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = model.schema
    saved: list[dict[str, Any]] = []
    sample_i = 0
    for batch in loader:
        batch_dev = move_batch_to_device(batch, device)
        out = model(batch_dev["image"])
        part_prob = out["part_prob"].float().detach()
        if batch_dev["part_masks"].shape[-2:] != part_prob.shape[-2:]:
            gt_masks = F.interpolate(batch_dev["part_masks"].float(), size=part_prob.shape[-2:], mode="nearest")
        else:
            gt_masks = batch_dev["part_masks"].float()
        presence = out["part_presence"].detach().float().cpu()
        topmean = topmean_mask_probability(part_prob, q=topq).detach().cpu()
        maxprob = part_prob.flatten(2).amax(-1).detach().cpu()
        pred_area = (part_prob >= mask_threshold).float().flatten(2).mean(-1).detach().cpu()
        gt_presence = batch_dev["presence"].detach().bool().cpu()
        labels = batch_dev["obj_label"].detach().cpu().long()
        bsz = int(part_prob.shape[0])
        for b in range(bsz):
            if sample_i >= int(num_samples):
                return saved
            class_idx = int(labels[b].item())
            class_name = schema.obj_names[class_idx]
            part_indices = _select_parts_for_sample(
                schema,
                presence[b],
                gt_presence[b],
                class_idx,
                max_parts=max_parts,
            )
            rows = max(1, len(part_indices))
            fig, axes = plt.subplots(rows, 5, figsize=(18, max(3.2, rows * 2.5)), squeeze=False)
            raw = _get_image_for_display(batch, b).permute(1, 2, 0).numpy()
            for r, k in enumerate(part_indices):
                prob = part_prob[b, k].detach().cpu()
                gt = gt_masks[b, k].detach().cpu() > 0.5
                pred = prob >= float(mask_threshold)
                inter = (pred & gt).sum().item()
                union = (pred | gt).sum().item()
                iou = float(inter / union) if union > 0 else float("nan")
                axes[r, 0].imshow(raw)
                axes[r, 0].set_title(f"image | GT={class_name}" if r == 0 else "image")
                axes[r, 1].imshow(prob, vmin=0.0, vmax=1.0)
                axes[r, 1].set_title(
                    f"{schema.part_names[k]} prob\n"
                    f"p={float(presence[b,k]):.2f}, top={float(topmean[b,k]):.2f}, max={float(maxprob[b,k]):.2f}\n"
                    f"gt={int(gt_presence[b,k])}, area={float(pred_area[b,k]):.3f}, IoU={iou:.2f}" if math.isfinite(iou) else
                    f"{schema.part_names[k]} prob\n"
                    f"p={float(presence[b,k]):.2f}, top={float(topmean[b,k]):.2f}, max={float(maxprob[b,k]):.2f}\n"
                    f"gt={int(gt_presence[b,k])}, area={float(pred_area[b,k]):.3f}, IoU=nan"
                )
                axes[r, 2].imshow(gt.float(), vmin=0.0, vmax=1.0)
                axes[r, 2].set_title("GT")
                axes[r, 3].imshow(pred.float(), vmin=0.0, vmax=1.0)
                axes[r, 3].set_title(f"pred >= {mask_threshold:.2f}")
                axes[r, 4].imshow(_overlay_rgb(pred, gt).permute(1, 2, 0).numpy())
                axes[r, 4].set_title("red=pred, green=GT, blue=overlap")
                for ax in axes[r]:
                    ax.axis("off")
            fig.tight_layout()
            path = out_dir / f"stage1_sample_{sample_i:03d}_{class_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append({"kind": "sample", "path": str(path), "sample_index": sample_i, "class": class_name})
            sample_i += 1
    return saved


def plot_training_curves(save_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Plot stage1_history.csv if it exists."""
    paths = [save_dir / "stage1_history.csv", save_dir / "stage1_quality_history.csv"]
    hist_path = next((p for p in paths if p.exists()), None)
    if hist_path is None:
        return []
    rows: list[dict[str, float]] = []
    with hist_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _as_float(v) for k, v in row.items()})
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]
    metrics = [
        ("train_loss", "Train loss"),
        ("val_loss", "Val loss"),
        ("val_miou_present", "Val mIoU present"),
        ("val_miou_present_parts", "Val mIoU present parts"),
        ("val_presence_f1", "Val presence F1"),
        ("val_presence_f1_macro", "Val presence F1 macro"),
        ("val_hallucination_rate", "Val hallucination"),
        ("val_hallucination_macro", "Val hallucination macro"),
    ]
    plotted = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    flat = axes.flatten()
    for key, label in metrics:
        vals = [r.get(key, float("nan")) for r in rows]
        if not any(math.isfinite(v) for v in vals):
            continue
        ax = flat[min(plotted, len(flat) - 1)]
        ax.plot(epochs, vals, marker="o")
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        plotted += 1
        if plotted >= len(flat):
            break
    for j in range(plotted, len(flat)):
        flat[j].axis("off")
    fig.tight_layout()
    path = out_dir / "stage1_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [{"kind": "training_curves", "path": str(path), "source": str(hist_path)}]


def plot_per_part_quality(per_part: list[dict[str, Any]], out_dir: Path, *, top_n: int = 14) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in per_part if math.isfinite(_as_float(r.get("iou_present")))]
    low_iou = sorted(rows, key=lambda r: _as_float(r.get("iou_present")))[:top_n]
    hall = sorted(per_part, key=lambda r: _as_float(r.get("hallucination_rate")), reverse=True)[:top_n]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    if low_iou:
        axes[0].bar([r["part"] for r in low_iou], [_as_float(r.get("iou_present")) for r in low_iou])
        axes[0].set_title("Lowest IoU on GT-present parts only")
        axes[0].set_ylabel("IoU | GT-present")
        axes[0].tick_params(axis="x", rotation=45)
    else:
        axes[0].text(0.5, 0.5, "No GT-present parts found in diagnostic subset", ha="center")
        axes[0].axis("off")
    if hall:
        axes[1].bar([r["part"] for r in hall], [_as_float(r.get("hallucination_rate")) for r in hall])
        axes[1].set_title("Highest hallucination rate on GT-absent images")
        axes[1].set_ylabel("FP / GT-absent")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].text(0.5, 0.5, "No hallucination rows", ha="center")
        axes[1].axis("off")
    fig.tight_layout()
    path = out_dir / "stage1_per_part_quality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [{"kind": "per_part_quality", "path": str(path)}]


def plot_per_class_part_heatmaps(per_class_part: list[dict[str, Any]], out_dir: Path, *, min_present: int = 1) -> list[dict[str, Any]]:
    if not per_class_part:
        return []
    classes = sorted({str(r["class"]) for r in per_class_part})
    parts = sorted({str(r["part"]) for r in per_class_part})
    cidx = {c: i for i, c in enumerate(classes)}
    pidx = {p: i for i, p in enumerate(parts)}
    mat_iou = torch.full((len(classes), len(parts)), float("nan"))
    mat_f1 = torch.full((len(classes), len(parts)), float("nan"))
    for r in per_class_part:
        if _as_float(r.get("present_count")) < min_present and _as_float(r.get("hallucination_rate")) <= 0:
            continue
        i, j = cidx[str(r["class"])], pidx[str(r["part"])]
        val = _as_float(r.get("iou_present"))
        mat_iou[i, j] = val if math.isfinite(val) else float("nan")
        mat_f1[i, j] = _as_float(r.get("presence_f1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, mat, title in [
        ("stage1_per_class_part_iou_heatmap.png", mat_iou, "Per-class / part IoU on GT-present masks"),
        ("stage1_per_class_part_presence_f1_heatmap.png", mat_f1, "Per-class / part presence F1"),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(parts)), max(6, 0.45 * len(classes))))
        im = ax.imshow(mat.numpy(), aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels(parts, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append({"kind": name.replace(".png", ""), "path": str(path)})
    return saved


@torch.no_grad()
def presence_threshold_sweep(
    model,
    loader,
    out_dir: Path,
    *,
    device: str,
    thresholds: Iterable[float],
    max_batches: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    thresholds = [float(t) for t in thresholds]
    tp = torch.zeros(len(thresholds), dtype=torch.float64)
    fp = torch.zeros(len(thresholds), dtype=torch.float64)
    fn = torch.zeros(len(thresholds), dtype=torch.float64)
    tn = torch.zeros(len(thresholds), dtype=torch.float64)
    for bidx, batch in enumerate(loader):
        if max_batches is not None and bidx >= int(max_batches):
            break
        bd = move_batch_to_device(batch, device)
        out = model(bd["image"])
        pres = out["part_presence"].detach().float().cpu()
        gt = bd["presence"].detach().bool().cpu()
        for i, tau in enumerate(thresholds):
            pred = pres >= tau
            tp[i] += (pred & gt).sum().item()
            fp[i] += (pred & ~gt).sum().item()
            fn[i] += (~pred & gt).sum().item()
            tn[i] += (~pred & ~gt).sum().item()
    rows = []
    for i, tau in enumerate(thresholds):
        precision = float(tp[i] / (tp[i] + fp[i] + 1e-6))
        recall = float(tp[i] / (tp[i] + fn[i] + 1e-6))
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        halluc = float(fp[i] / (fp[i] + tn[i] + 1e-6))
        miss = float(fn[i] / (tp[i] + fn[i] + 1e-6))
        rows.append({"threshold": tau, "precision": precision, "recall": recall, "f1": f1, "hallucination": halluc, "miss": miss})
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([r["threshold"] for r in rows], [r["f1"] for r in rows], marker="o", label="F1")
    ax.plot([r["threshold"] for r in rows], [r["hallucination"] for r in rows], marker="o", label="hallucination")
    ax.plot([r["threshold"] for r in rows], [r["miss"] for r in rows], marker="o", label="miss")
    ax.set_title("Presence threshold sweep")
    ax.set_xlabel("presence threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "stage1_presence_threshold_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows, [{"kind": "presence_threshold_sweep", "path": str(path)}]


def _parse_thresholds(s: str | None) -> list[float]:
    if not s:
        return [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    out = []
    for item in s.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def run_stage1_diagnostics(
    model,
    val_loader,
    cfg,
    *,
    device: str,
    output_dir: str | Path,
    max_batches: int | None = 50,
    num_samples: int = 8,
    max_parts_per_sample: int = 8,
    mask_threshold: float = 0.4,
    thresholds: str | None = None,
    save_dir_for_history: str | Path | None = None,
    make_zip: bool = True,
) -> dict[str, Any]:
    """Run Stage-1 diagnostics and save local figures/tables.

    This function is designed to be called from ``scripts/train_stage1.py`` via
    ``--diagnostics-only`` so the notebook still uses the same single Stage-1
    entry point.  It saves everything under ``output_dir``.
    """
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    sample_dir = fig_dir / "samples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    weights = Stage1QualityLossWeights(
        topq=float(getattr(cfg.loss.stage1, "quality_topq", 0.02)),
        small_part_area_tau=float(getattr(cfg.model.stage1, "small_part_area_tau", 0.015)),
        small_part_weight_max=float(getattr(cfg.model.stage1, "small_part_weight_max", 6.0)),
        small_part_weight_power=float(getattr(cfg.model.stage1, "small_part_weight_power", 0.5)),
    )
    metrics = evaluate_stage1_quality_detailed(
        model,
        val_loader,
        cfg,
        device=device,
        max_batches=max_batches,
        loss_weights=weights,
        mask_threshold=mask_threshold,
    )
    scalar = {k: _maybe_float(v) for k, v in metrics.items() if not isinstance(v, (list, dict))}
    save_json(out_dir / "stage1_diagnostic_summary.json", scalar)
    save_json(out_dir / "stage1_per_part.json", metrics.get("per_part", []))
    save_json(out_dir / "stage1_per_class_part.json", metrics.get("per_class_part", []))
    _write_rows_csv(table_dir / "stage1_per_part.csv", metrics.get("per_part", []))
    _write_rows_csv(table_dir / "stage1_per_class_part.csv", metrics.get("per_class_part", []))

    hkg_hint = summarize_stage1_quality_for_hkg(
        model,
        val_loader,
        device=device,
        max_batches=max_batches,
        q=float(getattr(cfg.loss.stage1, "quality_topq", 0.02)),
    )
    save_json(out_dir / "hkg_part_quality_hint.json", hkg_hint)
    _write_rows_csv(table_dir / "hkg_part_quality_hint.csv", hkg_hint.get("per_part_quality_hint", []))

    manifest: list[dict[str, Any]] = []
    if save_dir_for_history is not None:
        manifest.extend(plot_training_curves(Path(save_dir_for_history), fig_dir))
    manifest.extend(plot_per_part_quality(metrics.get("per_part", []), fig_dir))
    manifest.extend(plot_per_class_part_heatmaps(metrics.get("per_class_part", []), fig_dir))
    sweep_rows, sweep_figs = presence_threshold_sweep(
        model,
        val_loader,
        fig_dir,
        device=device,
        thresholds=_parse_thresholds(thresholds),
        max_batches=max_batches,
    )
    save_json(out_dir / "presence_threshold_sweep.json", sweep_rows)
    _write_rows_csv(table_dir / "presence_threshold_sweep.csv", sweep_rows)
    manifest.extend(sweep_figs)
    manifest.extend(save_stage1_sample_visualizations(
        model,
        val_loader,
        sample_dir,
        device=device,
        num_samples=int(num_samples),
        max_parts=int(max_parts_per_sample),
        mask_threshold=float(mask_threshold),
        topq=float(getattr(cfg.loss.stage1, "quality_topq", 0.02)),
    ))

    manifest_path = out_dir / "figure_manifest.json"
    save_json(manifest_path, manifest)
    _write_rows_csv(out_dir / "figure_manifest.csv", manifest)

    zip_path = None
    if make_zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(out_dir.parent)))
    print(f"[stage1 diagnostics] output_dir={out_dir}")
    print(f"[stage1 diagnostics] summary={out_dir / 'stage1_diagnostic_summary.json'}")
    print(f"[stage1 diagnostics] per_part_csv={table_dir / 'stage1_per_part.csv'}")
    print(f"[stage1 diagnostics] figures={fig_dir}")
    if zip_path is not None:
        print(f"[stage1 diagnostics] zip={zip_path}")
    return {"summary": scalar, "output_dir": str(out_dir), "zip": str(zip_path) if zip_path else None, "manifest": manifest}
