from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from partcat_hkg.config import ProjectConfig, load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.data.collate import collate_part_batch
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.evaluation.metrics import macro_accuracy, top1_accuracy
from partcat_hkg.kg.datatypes import AOGHierarchicalKG, MOTIF_TYPE_NAMES
from partcat_hkg.kg.relations import (
    RELATION_CHANNELS,
    RELATION_FEATURE_NAMES,
    relation_attributes_vectorized,
    relation_channel_strengths,
)
from partcat_hkg.kg.serialization import load_hkg
from partcat_hkg.models.pooling import topk_presence
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.stage2.aog_hkg_classifier import AOGHKGStage2Classifier
from partcat_hkg.utils.io import load_checkpoint


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _slice_batch(batch: dict[str, Any], index: int, n: int = 1) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[index:index + n]
        elif isinstance(v, list):
            out[k] = v[index:index + n]
        else:
            out[k] = v
    return out


def _image_from_batch(batch: dict[str, Any], index: int = 0) -> np.ndarray:
    img = batch.get("image_raw", batch.get("image"))[index].detach().cpu()
    if img.ndim == 3 and img.shape[0] in {1, 3}:
        img = img.permute(1, 2, 0)
    arr = img.float().numpy()
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0, 1)


def _resize_prob_to_image(prob: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(prob.shape[-2:]) == tuple(image_hw):
        return prob
    return F.interpolate(prob.unsqueeze(0).float(), size=image_hw, mode="bilinear", align_corners=False)[0]


def _mask_centroid(mask: torch.Tensor, thr: float = 0.4) -> tuple[float, float] | None:
    m = (mask.detach().float().cpu() > float(thr)).float()
    if float(m.sum()) <= 1.0:
        return None
    h, w = m.shape[-2:]
    yy = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, w)
    area = m.sum().clamp_min(1.0)
    return float((m * xx).sum() / area), float((m * yy).sum() / area)


def _top_names(values: torch.Tensor, names: list[str], k: int = 5, threshold: float | None = None) -> str:
    vals = values.detach().float().cpu()
    if vals.numel() == 0:
        return ""
    if threshold is not None:
        idx = (vals >= float(threshold)).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.topk(vals, min(k, vals.numel())).indices
        idx = idx[torch.argsort(vals[idx], descending=True)][:k]
    else:
        idx = torch.topk(vals, min(k, vals.numel())).indices
    return ", ".join(f"{names[int(i)]}:{float(vals[int(i)]):.2f}" for i in idx)




def make_balanced_diagnostic_loader(
    dataset,
    *,
    batch_size: int = 4,
    samples_per_class: int = 8,
    seed: int = 0,
    shuffle_within_class: bool = True,
    num_workers: int = 0,
):
    """Build a class-balanced diagnostic loader from a PartImageNet dataset.

    The normal validation loader is deterministic and often class-ordered.  If a
    notebook inspects only the first few batches, metrics and confusion matrices
    can silently cover just one class.  This helper samples up to
    ``samples_per_class`` examples from every class and returns a regular full
    mask loader suitable for Stage-1 and Stage-2 diagnostics.
    """

    if not hasattr(dataset, "samples"):
        raise TypeError("make_balanced_diagnostic_loader expects a dataset with a .samples list")
    rng = np.random.default_rng(int(seed))
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(dataset.samples):
        by_class[int(rec["obj_label"])].append(int(idx))
    chosen: list[int] = []
    for c in sorted(by_class):
        inds = list(by_class[c])
        if shuffle_within_class:
            rng.shuffle(inds)
        chosen.extend(inds[: int(samples_per_class)])
    if shuffle_within_class:
        rng.shuffle(chosen)
    subset = Subset(dataset, chosen)
    common = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_part_batch,
    )
    return DataLoader(subset, **common)


def summarize_diagnostic_loader(loader) -> pd.DataFrame:
    """Return class counts for a diagnostic loader.

    This is intentionally cheap and should be displayed before metrics.  If it
    shows a single class, the downstream confusion matrix, calibration curve,
    template usage, and hallucination statistics are not globally meaningful.
    """

    ds = getattr(loader, "dataset", None)
    base = getattr(ds, "dataset", ds)
    indices = list(getattr(ds, "indices", range(len(ds) if ds is not None else 0)))
    schema = getattr(base, "schema", None)
    rows: list[dict[str, Any]] = []
    if not hasattr(base, "samples"):
        return pd.DataFrame(rows)
    counts: Counter[int] = Counter()
    for idx in indices:
        counts[int(base.samples[int(idx)]["obj_label"])] += 1
    for c, n in sorted(counts.items()):
        rows.append({
            "class_idx": int(c),
            "class": schema.obj_names[c] if schema is not None and c < len(schema.obj_names) else str(c),
            "count": int(n),
        })
    return pd.DataFrame(rows)

def load_aog_hkg_diagnostic_context(
    config_path: str | Path = "configs/default.yaml",
    *,
    stage1_ckpt: str | Path,
    hkg_path: str | Path,
    stage2_ckpt: str | Path | None = None,
    partimagenet_root: str | Path | None = None,
    save_dir: str | Path | None = None,
    device: str = "auto",
    num_workers: int = 0,
    batch_size: int | None = None,
    max_val_samples: int | None = None,
    strict_stage2: bool = True,
) -> dict[str, Any]:
    """Load config, datasets, frozen Stage 1, AOG-HKG, and optional Stage-2 checkpoint.

    The returned dictionary is designed for notebook use.  It includes full-mask
    loaders so Stage-1 segmentation metrics and Stage-2 parse diagnostics can be
    computed from the same samples.
    """

    cfg = load_config(config_path)
    if partimagenet_root is not None and str(partimagenet_root):
        cfg.paths.partimagenet_root = str(partimagenet_root)
    if save_dir is not None and str(save_dir):
        cfg.paths.save_dir = str(save_dir)
    if max_val_samples is not None:
        cfg.data.max_val_samples = int(max_val_samples)
    cfg.data.num_workers = int(num_workers)
    cfg.data.persistent_workers = bool(cfg.data.num_workers > 0 and cfg.data.persistent_workers)
    if batch_size is not None:
        cfg.training.batch_size_stage1 = int(batch_size)
        cfg.training.batch_size_stage2 = int(batch_size)
    # Diagnostics need the full target masks; keep the normal full loader.
    cfg.data.use_stage2_image_only_loader = False

    dev = resolve_device(device)
    train_ds, val_ds = make_datasets(cfg)
    stage1_train, stage1_val, stage2_train, stage2_val = make_loaders(cfg, train_ds, val_ds)
    kg = load_hkg(hkg_path)
    if not isinstance(kg, AOGHierarchicalKG):
        raise TypeError(f"Expected AOGHierarchicalKG at {hkg_path}, got {type(kg).__name__}")

    stage1 = PartCATHKGStage1(kg.schema, cfg.model.stage1)
    load_checkpoint(stage1_ckpt, stage1, strict=True)
    model = AOGHKGStage2Classifier(stage1, kg, cfg.model.stage2)
    if stage2_ckpt is not None and str(stage2_ckpt):
        load_checkpoint(stage2_ckpt, model, strict=strict_stage2)
    model.to(dev)
    model.freeze_stage1()
    model.eval()

    return {
        "cfg": cfg,
        "device": dev,
        "schema": kg.schema,
        "kg": kg,
        "stage1": stage1,
        "model": model,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "stage1_train": stage1_train,
        "stage1_val": stage1_val,
        "stage2_train": stage2_train,
        "stage2_val": stage2_val,
    }


@torch.no_grad()
def compute_stage1_statistics(
    stage1_model: PartCATHKGStage1,
    loader,
    *,
    device: str | torch.device = "cuda",
    mask_threshold: float = 0.40,
    presence_threshold: float = 0.15,
    max_batches: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, per_part_df) for Stage-1 segmentation/presence diagnostics."""

    schema = stage1_model.schema
    dev = torch.device(device)
    stage1_model.eval()
    fnum = schema.num_parts
    eps = 1e-6
    inter = torch.zeros(fnum)
    union = torch.zeros(fnum)
    pred_sum = torch.zeros(fnum)
    target_sum = torch.zeros(fnum)
    dice_num = torch.zeros(fnum)
    dice_den = torch.zeros(fnum)
    tp = torch.zeros(fnum)
    fp = torch.zeros(fnum)
    fn = torch.zeros(fnum)
    tn = torch.zeros(fnum)
    pred_presence_sum = torch.zeros(fnum)
    gt_presence_sum = torch.zeros(fnum)
    invalid_role_mass = 0.0
    valid_role_mass = 0.0
    nb = 0
    nsamples = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        image = batch["image"].to(dev, non_blocking=True)
        target = batch["part_masks"].float().to(dev, non_blocking=True)
        labels = batch["obj_label"].to(dev, non_blocking=True)
        gt_presence = batch.get("presence")
        if gt_presence is None:
            gt_presence = (target.flatten(2).amax(-1) > 0).float().cpu()
        gt_presence = gt_presence.float().to(dev, non_blocking=True)

        out = stage1_model(image)
        prob = out.get("part_prob", torch.sigmoid(out["part_logits"])).float()
        if prob.shape[-2:] != target.shape[-2:]:
            prob = F.interpolate(prob, size=target.shape[-2:], mode="bilinear", align_corners=False)
        pred = (prob >= float(mask_threshold)).float()
        i = (pred * target).flatten(2).sum(-1)
        ps = pred.flatten(2).sum(-1)
        ts = target.flatten(2).sum(-1)
        u = ps + ts - i
        inter += i.sum(0).detach().cpu()
        union += u.sum(0).detach().cpu()
        pred_sum += ps.sum(0).detach().cpu()
        target_sum += ts.sum(0).detach().cpu()
        dice_num += (2.0 * i).sum(0).detach().cpu()
        dice_den += (ps + ts).sum(0).detach().cpu()

        presence = out.get("part_presence")
        if presence is None:
            presence = topk_presence(prob, k=int(stage1_model.cfg.topk_presence_k))
        ppred = presence >= float(presence_threshold)
        tgt = gt_presence > 0.5
        tp += (ppred & tgt).sum(0).detach().cpu()
        fp += (ppred & ~tgt).sum(0).detach().cpu()
        fn += (~ppred & tgt).sum(0).detach().cpu()
        tn += (~ppred & ~tgt).sum(0).detach().cpu()
        pred_presence_sum += presence.detach().cpu().sum(0)
        gt_presence_sum += gt_presence.detach().cpu().sum(0)
        nsamples += int(image.shape[0])

        if "role_prob" in out:
            rp = out["role_prob"].float()
            valid = (schema.role_to_obj.to(dev).view(1, -1, 1, 1) == labels.view(-1, 1, 1, 1)).float()
            invalid_role_mass += float((rp * (1.0 - valid)).mean().detach().cpu())
            denom = (valid.sum() * rp.shape[-1] * rp.shape[-2]).clamp_min(1.0)
            valid_role_mass += float((rp * valid).sum().detach().cpu() / denom.detach().cpu())
        nb += 1

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = inter / union.clamp_min(1)
    dice = dice_num / dice_den.clamp_min(1)
    hallucination = fp / (fp + tn + eps)
    miss = fn / (fn + tp + eps)
    per_part = pd.DataFrame({
        "part": schema.part_names,
        "iou": iou.numpy(),
        "dice": dice.numpy(),
        "presence_precision": precision.numpy(),
        "presence_recall": recall.numpy(),
        "presence_f1": f1.numpy(),
        "hallucination_rate": hallucination.numpy(),
        "miss_rate": miss.numpy(),
        "gt_presence_rate": (gt_presence_sum / max(nsamples, 1)).numpy(),
        "mean_pred_presence": (pred_presence_sum / max(nsamples, 1)).numpy(),
        "pred_pixels": pred_sum.numpy(),
        "gt_pixels": target_sum.numpy(),
    })
    summary = pd.DataFrame([{
        "n_samples": nsamples,
        "mIoU": float(iou.mean().item()),
        "mean_dice": float(dice.mean().item()),
        "mean_presence_precision": float(precision.mean().item()),
        "mean_presence_recall": float(recall.mean().item()),
        "mean_presence_f1": float(f1.mean().item()),
        "mean_hallucination_rate": float(hallucination.mean().item()),
        "mean_miss_rate": float(miss.mean().item()),
        "valid_role_mass": float(valid_role_mass / max(nb, 1)),
        "invalid_role_mass": float(invalid_role_mass / max(nb, 1)),
    }])
    return summary, per_part.sort_values("iou")


def summarize_aog_hkg(kg: AOGHierarchicalKG) -> dict[str, pd.DataFrame]:
    """Create interpretable HKG structure tables for templates, roles, edges, and motifs."""

    schema = kg.schema
    cnum, anum, fnum = schema.num_classes, int(kg.num_templates), schema.num_parts
    edge_counts = Counter((int(c), int(a)) for c, a, _, _ in kg.template_edges.cpu().tolist())
    motif_counts = Counter((int(c), int(a)) for c, a, *_ in kg.motif_edges.cpu().tolist())
    template_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for c in range(cnum):
        for a in range(anum):
            pri = float(kg.template_prior[c, a])
            valid = bool(float(kg.template_valid[c, a]) > 0)
            role_prior = kg.template_role_prior[c, a].detach().float().cpu()
            required = kg.template_role_required[c, a].detach().float().cpu() > 0.5
            top_idx = torch.topk(role_prior, min(5, fnum)).indices.tolist()
            template_rows.append({
                "class_idx": c,
                "class": schema.obj_names[c],
                "template": a,
                "valid": valid,
                "prior": pri,
                "required_count": int(required.sum().item()),
                "mean_role_prior": float(role_prior.mean().item()),
                "edge_count": int(edge_counts[(c, a)]),
                "motif_count": int(motif_counts[(c, a)]),
                "top_roles": ", ".join(f"{schema.part_names[k]}:{float(role_prior[k]):.2f}" for k in top_idx),
            })
            for k in range(fnum):
                role_rows.append({
                    "class_idx": c,
                    "class": schema.obj_names[c],
                    "template": a,
                    "part_idx": k,
                    "part": schema.part_names[k],
                    "role_prior": float(role_prior[k]),
                    "required": bool(required[k]),
                    "pmi": float(kg.pmi[c, k]),
                })

    edge_rows: list[dict[str, Any]] = []
    if kg.template_edges.numel() > 0:
        ch = relation_channel_strengths(kg.template_rel_mean.detach().float().cpu())
        for e, row in enumerate(kg.template_edges.detach().cpu().tolist()):
            c, a, i, j = [int(x) for x in row]
            top_ch = int(ch[e].argmax().item())
            rec: dict[str, Any] = {
                "edge_idx": e,
                "class_idx": c,
                "class": schema.obj_names[c],
                "template": a,
                "part_i_idx": i,
                "part_i": schema.part_names[i],
                "part_j_idx": j,
                "part_j": schema.part_names[j],
                "relation_type": kg.template_rel_type_names[e] if e < len(kg.template_rel_type_names) else "relation",
                "support": float(kg.template_rel_support[e]),
                "information_gain": float(kg.template_rel_ig[e]),
                "top_template_channel": RELATION_CHANNELS[top_ch],
                "top_template_channel_strength": float(ch[e, top_ch]),
            }
            for d, name in enumerate(RELATION_FEATURE_NAMES):
                rec[f"mu_{name}"] = float(kg.template_rel_mean[e, d])
                rec[f"var_{name}"] = float(kg.template_rel_var[e, d])
            edge_rows.append(rec)
    edge_df = pd.DataFrame(edge_rows)

    motif_rows: list[dict[str, Any]] = []
    for m, row in enumerate(kg.motif_edges.detach().cpu().tolist()):
        c, a, i, j, mt = [int(x) for x in row]
        motif_rows.append({
            "motif_idx": m,
            "class_idx": c,
            "class": schema.obj_names[c],
            "template": a,
            "part_i": schema.part_names[i],
            "part_j": schema.part_names[j],
            "motif_type": MOTIF_TYPE_NAMES[mt] if mt < len(MOTIF_TYPE_NAMES) else str(mt),
            "support": float(kg.motif_support[m]),
        })

    return {
        "templates": pd.DataFrame(template_rows).sort_values(["class", "template"]),
        "roles": pd.DataFrame(role_rows).sort_values(["class", "template", "role_prior"], ascending=[True, True, False]),
        "edges": edge_df.sort_values(["class", "template", "support"], ascending=[True, True, False]) if len(edge_df) else edge_df,
        "motifs": pd.DataFrame(motif_rows),
    }


@torch.no_grad()
def collect_stage2_predictions(
    model: AOGHKGStage2Classifier,
    loader,
    *,
    device: str | torch.device = "cuda",
    max_batches: int | None = None,
    presence_threshold: float | None = None,
    enable_edges: bool = True,
) -> pd.DataFrame:
    """Collect per-image Stage-2 branch predictions and parse bookkeeping."""

    dev = torch.device(device)
    schema = model.schema
    presence_threshold = float(presence_threshold if presence_threshold is not None else model.stage1.cfg.presence_threshold)
    rows: list[dict[str, Any]] = []
    sample_idx = 0
    branch_keys = {
        "final": "logits",
        "base": "base_logits",
        "hkg": "hkg_logits",
        "node": "node_logits",
        "edge": "edge_logits",
        "motif": "motif_logits",
    }
    model.eval()
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        labels = batch["obj_label"].to(dev, non_blocking=True)
        out = model(batch, detach_stage1=True, enable_edges=enable_edges)
        bsz = int(labels.shape[0])
        probs = {name: torch.softmax(out[key].detach().float(), dim=-1).cpu() for name, key in branch_keys.items() if key in out}
        logits = {name: out[key].detach().float().cpu() for name, key in branch_keys.items() if key in out}
        best_template = out["best_template"].detach().cpu()
        part_presence = out["part_presence"].detach().float().cpu()
        template_scores = out["template_scores"].detach().float().cpu()
        for i in range(bsz):
            y = int(labels[i].detach().cpu().item())
            meta = batch.get("meta", [{} for _ in range(bsz)])[i] if isinstance(batch.get("meta"), list) else {}
            rec: dict[str, Any] = {
                "sample_index": sample_idx,
                "batch_index": bi,
                "batch_row": i,
                "image_id": meta.get("image_id", ""),
                "path": meta.get("path", ""),
                "gt_idx": y,
                "gt": schema.obj_names[y],
                "active_part_count": int((part_presence[i] >= presence_threshold).sum().item()),
                "mean_part_presence": float(part_presence[i].mean().item()),
                "max_part_presence": float(part_presence[i].max().item()),
                "active_parts_top": _top_names(part_presence[i], schema.part_names, k=8, threshold=presence_threshold),
            }
            for name in logits:
                logit_i = logits[name][i]
                prob_i = probs[name][i]
                pred = int(logit_i.argmax().item())
                top2 = torch.topk(logit_i, min(2, logit_i.numel())).values
                rec[f"pred_{name}_idx"] = pred
                rec[f"pred_{name}"] = schema.obj_names[pred]
                rec[f"correct_{name}"] = bool(pred == y)
                rec[f"conf_{name}"] = float(prob_i[pred].item())
                rec[f"prob_true_{name}"] = float(prob_i[y].item())
                rec[f"logit_true_{name}"] = float(logit_i[y].item())
                rec[f"margin_{name}"] = float((top2[0] - top2[1]).item()) if top2.numel() > 1 else float("nan")
                rec[f"rank_true_{name}"] = int((logit_i > logit_i[y]).sum().item() + 1)
            pred_final = int(rec.get("pred_final_idx", -1))
            rec["best_template_pred"] = int(best_template[i, pred_final].item()) if pred_final >= 0 else -1
            rec["best_template_gt"] = int(best_template[i, y].item())
            if pred_final >= 0:
                rec["template_score_pred"] = float(template_scores[i, pred_final, rec["best_template_pred"]].item())
            rec["template_score_gt"] = float(template_scores[i, y, rec["best_template_gt"]].item())
            rec["rescued_by_final_vs_base"] = bool((not rec.get("correct_base", False)) and rec.get("correct_final", False))
            rec["damaged_by_final_vs_base"] = bool(rec.get("correct_base", False) and (not rec.get("correct_final", False)))
            rows.append(rec)
            sample_idx += 1
    return pd.DataFrame(rows)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if not np.any(m):
            continue
        ece += float(np.abs(confidence[m].mean() - correct[m].mean()) * m.mean())
    return ece


def summarize_classification(df: pd.DataFrame, schema: RoleSchema, branches: tuple[str, ...] = ("final", "base", "hkg", "node", "edge", "motif")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return branch-level and class-level statistical summaries."""

    rows: list[dict[str, Any]] = []
    for branch in branches:
        pred_col, correct_col, conf_col = f"pred_{branch}_idx", f"correct_{branch}", f"conf_{branch}"
        if pred_col not in df:
            continue
        pred = df[pred_col].to_numpy(dtype=int)
        gt = df["gt_idx"].to_numpy(dtype=int)
        correct = df[correct_col].to_numpy(dtype=bool)
        rows.append({
            "branch": branch,
            "n": int(len(df)),
            "accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "macro_accuracy": float(np.mean([correct[gt == c].mean() for c in range(schema.num_classes) if np.any(gt == c)])) if len(correct) else float("nan"),
            "mean_confidence": float(df[conf_col].mean()) if conf_col in df else float("nan"),
            "ECE_15bins": expected_calibration_error(df[conf_col].to_numpy(), correct, n_bins=15) if conf_col in df else float("nan"),
            "mean_true_probability": float(df.get(f"prob_true_{branch}", pd.Series(dtype=float)).mean()) if f"prob_true_{branch}" in df else float("nan"),
            "mean_true_rank": float(df.get(f"rank_true_{branch}", pd.Series(dtype=float)).mean()) if f"rank_true_{branch}" in df else float("nan"),
        })
    if {"correct_base", "correct_final"}.issubset(df.columns):
        # Add the usual final-vs-base diagnostic as a separate row.
        rows.append({
            "branch": "final_vs_base_delta",
            "n": int(len(df)),
            "accuracy": float(df["correct_final"].mean() - df["correct_base"].mean()),
            "macro_accuracy": float("nan"),
            "mean_confidence": float("nan"),
            "ECE_15bins": float("nan"),
            "mean_true_probability": float("nan"),
            "mean_true_rank": float("nan"),
            "rescue_rate": float(df["rescued_by_final_vs_base"].mean()),
            "damage_rate": float(df["damaged_by_final_vs_base"].mean()),
        })
    branch_df = pd.DataFrame(rows)

    class_rows: list[dict[str, Any]] = []
    for c, cname in enumerate(schema.obj_names):
        sub = df[df["gt_idx"] == c]
        if len(sub) == 0:
            continue
        rec: dict[str, Any] = {"class_idx": c, "class": cname, "support": int(len(sub))}
        for branch in branches:
            col = f"correct_{branch}"
            if col in sub:
                rec[f"acc_{branch}"] = float(sub[col].mean())
        if {"correct_base", "correct_final"}.issubset(sub.columns):
            rec["rescue_rate"] = float(sub["rescued_by_final_vs_base"].mean())
            rec["damage_rate"] = float(sub["damaged_by_final_vs_base"].mean())
        rec["mean_active_part_count"] = float(sub["active_part_count"].mean()) if "active_part_count" in sub else float("nan")
        class_rows.append(rec)
    class_df = pd.DataFrame(class_rows).sort_values("support", ascending=False)
    return branch_df, class_df


@torch.no_grad()
def explain_aog_parse_for_sample(
    model: AOGHKGStage2Classifier,
    batch: dict[str, Any],
    *,
    index: int = 0,
    class_idx: int | None = None,
    enable_edges: bool = True,
    mask_threshold: float = 0.40,
) -> dict[str, Any]:
    """Decode one image into HKG node/edge/motif contribution tables."""

    schema = model.schema
    dev = next(model.parameters()).device
    small = _slice_batch(batch, int(index), 1)
    model.eval()
    out = model(small, detach_stage1=True, enable_edges=enable_edges)
    logits = out["logits"].detach().float().cpu()[0]
    pred = int(logits.argmax().item())
    c = int(class_idx) if class_idx is not None else pred
    y = int(small["obj_label"][0].item()) if "obj_label" in small else -1
    a = int(out["best_template"][0, c].detach().cpu().item())

    ex = model._stage1_extract(small, detach_stage1=True)
    part_presence = out["part_presence"].detach().float()[0]
    role_presence = out["role_presence"].detach().float()[0]
    role_prob = out["role_prob"].detach().float()[0]
    part_prob = out["part_prob"].detach().float()[0]

    rid_table = model.role_index_cf.to(dev)
    role_presence_cf = model._gather_role_cf(ex["role_presence"]).squeeze(0) if ex["role_presence"].ndim == 2 else model._gather_role_cf(ex["role_presence"])[0]
    role_tokens_r_cf = model._gather_role_cf(ex["role_tokens_r"])[0]
    role_tokens_d_cf = model._gather_role_cf(ex["role_tokens_d"])[0]

    fr = F.normalize(model.proj_r(ex["part_tokens_r"]), dim=-1)[0]
    fd = F.normalize(model.proj_d(ex["part_tokens_d"]), dim=-1)[0]
    tpr = F.normalize(model.proj_r(model.template_role_proto_r_raw.to(dev)), dim=-1)[c, a]
    tpd = F.normalize(model.proj_d(model.template_role_proto_d_raw.to(dev)), dim=-1)[c, a]
    func_sim = 0.5 * ((fr * tpr).sum(-1) + (fd * tpd).sum(-1))
    rr = F.normalize(model.proj_r(role_tokens_r_cf), dim=-1)[c]
    rd = F.normalize(model.proj_d(role_tokens_d_cf), dim=-1)[c]
    role_sim = 0.5 * ((rr * tpr).sum(-1) + (rd * tpd).sum(-1))
    app_sim = 0.5 * (func_sim + role_sim)

    role_prior = model.role_prior.to(dev)[c]
    t_role_prior = model.template_role_prior.to(dev)[c, a]
    required = model.template_role_required.to(dev)[c, a]
    pmi = model.pmi.to(dev)[c]
    part_pres_dev = part_presence.to(dev)
    role_pres_slots = role_presence_cf[c].to(dev)
    obs_presence = torch.maximum(part_pres_dev, role_pres_slots).clamp_min(float(model.cfg.hkg_presence_floor))
    base_evidence = obs_presence * t_role_prior * (
        float(model.cfg.hkg_node_presence_scale) * torch.log(obs_presence.clamp_min(1e-4))
        + float(model.cfg.hkg_node_pmi_scale) * pmi
        + float(model.cfg.hkg_node_app_scale) * app_sim
    )
    missing_penalty = float(model.cfg.hkg_absence_penalty) * required * t_role_prior * (1.0 - obs_presence).clamp(0, 1)
    conflict_penalty = float(model.cfg.hkg_conflict_penalty) * part_pres_dev * (1.0 - role_prior).clamp(0, 1)
    spurious_mask = (t_role_prior < float(getattr(model.cfg, "hkg_spurious_template_tau", 0.08))).float()
    spurious_penalty = float(getattr(model.cfg, "hkg_spurious_template_penalty", 0.0)) * part_pres_dev * spurious_mask
    node_contribution = F.softplus(model.raw_node_scale.detach()).to(dev) * (base_evidence - missing_penalty - conflict_penalty - spurious_penalty)

    node_rows: list[dict[str, Any]] = []
    for k, pname in enumerate(schema.part_names):
        rid = int(rid_table[c, k].item())
        node_rows.append({
            "part_idx": k,
            "part": pname,
            "role_idx": rid,
            "part_presence": float(part_pres_dev[k]),
            "role_presence": float(role_pres_slots[k]),
            "obs_presence": float(obs_presence[k]),
            "template_role_prior": float(t_role_prior[k]),
            "required": bool(float(required[k]) > 0.5),
            "pmi": float(pmi[k]),
            "appearance_similarity": float(app_sim[k]),
            "node_evidence": float(base_evidence[k]),
            "missing_penalty": float(missing_penalty[k]),
            "conflict_penalty": float(conflict_penalty[k]),
            "spurious_penalty": float(spurious_penalty[k]),
            "node_contribution": float(node_contribution[k]),
        })
    node_df = pd.DataFrame(node_rows).sort_values("node_contribution", ascending=False)

    edge_rows: list[dict[str, Any]] = []
    rows = model.template_edges.detach().cpu().tolist()
    selected_edges = [(e, row) for e, row in enumerate(rows) if int(row[0]) == c and int(row[1]) == a]
    edge_count = max(len(selected_edges), 1)
    for e, row in selected_edges:
        _, _, pi, pj = [int(x) for x in row]
        ri, rj = int(rid_table[c, pi].item()), int(rid_table[c, pj].item())
        if ri < 0 or rj < 0:
            continue
        gamma = relation_attributes_vectorized(role_prob[ri].to(dev).view(1, 1, *role_prob.shape[-2:]), role_prob[rj].to(dev).view(1, 1, *role_prob.shape[-2:]), thr=mask_threshold)[0, 0]
        ll_t = model._gaussian_ll(gamma, model.template_rel_mean[e].to(dev), model.template_rel_var[e].to(dev))
        ll_g = model._gaussian_ll(gamma, model.template_rel_global_mean[e].to(dev), model.template_rel_global_var[e].to(dev))
        llr = (ll_t - ll_g).clamp(-8, 8)

        # Mirror the actual Stage-2 edge scorer.  Earlier diagnostics always used
        # LLR, but HKG-v2 defaults to ``edge_score_mode=template_fit``.  That made
        # the overlay/table disagree with the logits used for training/inference.
        mode = str(getattr(model.cfg, "edge_score_mode", "template_fit"))
        if mode == "template_fit":
            var = model.template_rel_var[e].to(dev).clamp_min(1e-3)
            mu = model.template_rel_mean[e].to(dev)
            edge_used = (-0.5 * (((gamma - mu) ** 2) / var).mean(-1)).clamp(-8, 0)
            # Positive visualization score: 1 = excellent template fit, close to 0 = poor fit.
            edge_draw_score = torch.exp(edge_used).clamp(0, 1)
        else:
            edge_used = llr
            if bool(model.cfg.hkg_edge_positive_only):
                edge_used = F.relu(edge_used)
            # Positive visualization score for LLR mode.
            edge_draw_score = torch.sigmoid(edge_used).clamp(0, 1)

        strength = torch.sqrt((role_presence[ri].to(dev) * role_presence[rj].to(dev)).clamp_min(0.0) + 1e-8)
        support = model.template_rel_support[e].to(dev)
        ig = model.template_rel_ig[e].to(dev)
        ig_gate = (ig / (ig + 1.0)).clamp(0, 1)
        learned_w = F.softplus(model.edge_weight_raw[e].detach()).to(dev)
        raw = learned_w * strength * support * (0.5 + 0.5 * ig_gate) * edge_used
        contribution = float(model.cfg.hkg_edge_scale) * float(F.softplus(model.raw_edge_scale.detach()).cpu()) * float((raw / math.sqrt(edge_count)).detach().cpu())
        draw_score = float((strength * support * (0.5 + 0.5 * ig_gate) * edge_draw_score).detach().cpu())
        obs_ch = relation_channel_strengths(gamma.detach().cpu())
        templ_ch = relation_channel_strengths(model.template_rel_mean[e].detach().cpu())
        obs_top = int(obs_ch.argmax().item())
        templ_top = int(templ_ch.argmax().item())
        rec: dict[str, Any] = {
            "edge_idx": e,
            "part_i_idx": pi,
            "part_i": schema.part_names[pi],
            "part_j_idx": pj,
            "part_j": schema.part_names[pj],
            "role_i": ri,
            "role_j": rj,
            "relation_type": model.kg.template_rel_type_names[e] if e < len(model.kg.template_rel_type_names) else "relation",
            "support": float(support),
            "information_gain": float(ig),
            "strength": float(strength),
            "ll_template": float(ll_t),
            "ll_global": float(ll_g),
            "llr": float(llr),
            "edge_score_mode": mode,
            "edge_used_by_model": float(edge_used),
            "edge_draw_score": draw_score,
            "edge_contribution": contribution,
            "top_observed_channel": RELATION_CHANNELS[obs_top],
            "top_observed_channel_strength": float(obs_ch[obs_top]),
            "top_template_channel": RELATION_CHANNELS[templ_top],
            "top_template_channel_strength": float(templ_ch[templ_top]),
        }
        for d, name in enumerate(RELATION_FEATURE_NAMES):
            rec[f"obs_{name}"] = float(gamma[d])
            rec[f"template_{name}"] = float(model.template_rel_mean[e, d])
            rec[f"resid_z_{name}"] = float((gamma[d].detach().cpu() - model.template_rel_mean[e, d].detach().cpu()) / torch.sqrt(model.template_rel_var[e, d].detach().cpu().clamp_min(1e-6)))
        edge_rows.append(rec)
    edge_df = pd.DataFrame(edge_rows).sort_values("edge_contribution", ascending=False) if edge_rows else pd.DataFrame()

    motif_rows: list[dict[str, Any]] = []
    for m, row in enumerate(model.motif_edges.detach().cpu().tolist()):
        mc, ma, pi, pj, mt = [int(x) for x in row]
        if mc != c or ma != a:
            continue
        ri, rj = int(rid_table[c, pi].item()), int(rid_table[c, pj].item())
        if ri < 0 or rj < 0:
            continue
        gamma = relation_attributes_vectorized(role_prob[ri].to(dev).view(1, 1, *role_prob.shape[-2:]), role_prob[rj].to(dev).view(1, 1, *role_prob.shape[-2:]), thr=mask_threshold)[0, 0]
        ch = relation_channel_strengths(gamma)[0:] if gamma.ndim > 1 else relation_channel_strengths(gamma)
        if mt == 1:
            val = 0.5 * ch[3] + 0.5 * ch[4]
        elif mt == 2:
            val = torch.maximum(ch[6], ch[7])
        elif mt == 3:
            val = ch[2]
        elif mt == 4:
            val = torch.maximum(ch[3], torch.maximum(ch[4], 0.5 * torch.maximum(ch[0], ch[1]) + 0.25 * ch[2]))
        else:
            val = ch[3]
        strength = torch.sqrt((role_presence[ri].to(dev) * role_presence[rj].to(dev)).clamp_min(0.0) + 1e-8)
        motif_count = max(sum(1 for rr in model.motif_edges.detach().cpu().tolist() if int(rr[0]) == c and int(rr[1]) == a), 1)
        learned_w = F.softplus(model.motif_weight_raw[m].detach()).to(dev)
        raw_motif = learned_w * strength * model.motif_support[m].to(dev) * val.clamp(0, 1) / math.sqrt(motif_count)
        contribution = float(model.cfg.hkg_motif_scale) * float(F.softplus(model.raw_motif_scale.detach()).cpu()) * float(raw_motif.detach().cpu())
        motif_rows.append({
            "motif_idx": m,
            "part_i": schema.part_names[pi],
            "part_j": schema.part_names[pj],
            "motif_type": MOTIF_TYPE_NAMES[mt] if mt < len(MOTIF_TYPE_NAMES) else str(mt),
            "support": float(model.motif_support[m]),
            "strength": float(strength),
            "motif_value": float(val),
            "learned_motif_weight": float(learned_w),
            "motif_contribution": contribution,
        })
    motif_df = pd.DataFrame(motif_rows).sort_values("motif_contribution", ascending=False) if motif_rows else pd.DataFrame()

    summary = {
        "gt_idx": y,
        "gt": schema.obj_names[y] if y >= 0 else "",
        "pred_idx": pred,
        "pred": schema.obj_names[pred],
        "explained_class_idx": c,
        "explained_class": schema.obj_names[c],
        "template": a,
        "final_top5": _top_names(logits, schema.obj_names, k=5),
        "base_pred": schema.obj_names[int(out["base_logits"][0].detach().cpu().argmax())],
        "hkg_pred": schema.obj_names[int(out["hkg_logits"][0].detach().cpu().argmax())],
        "hkg_logit_explained": float(out["hkg_logits"][0, c].detach().cpu()),
        "node_logit_explained": float(out["node_logits"][0, c].detach().cpu()),
        "edge_logit_explained": float(out["edge_logits"][0, c].detach().cpu()),
        "motif_logit_explained": float(out["motif_logits"][0, c].detach().cpu()),
    }
    return {
        "summary": summary,
        "node_df": node_df,
        "edge_df": edge_df,
        "motif_df": motif_df,
        "batch": small,
        "out": out,
        "part_prob": part_prob.detach().cpu(),
        "role_prob": role_prob.detach().cpu(),
    }


def plot_stage1_sample(
    stage1_model: PartCATHKGStage1,
    batch: dict[str, Any],
    *,
    index: int = 0,
    top_parts: int = 6,
    mask_threshold: float = 0.40,
    show_gt: bool = True,
    path: str | Path | None = None,
):
    """Plot image, GT union, support, top predicted part masks, and presence bars."""

    schema = stage1_model.schema
    dev = next(stage1_model.parameters()).device
    small = _slice_batch(batch, int(index), 1)
    stage1_model.eval()
    with torch.no_grad():
        out = stage1_model(small["image"].to(dev))
    image = _image_from_batch(small, 0)
    h, w = image.shape[:2]
    part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"])).detach().float().cpu()[0]
    part_prob = _resize_prob_to_image(part_prob, (h, w))
    presence = out.get("part_presence")
    if presence is None:
        presence = topk_presence(part_prob.unsqueeze(0), k=int(stage1_model.cfg.topk_presence_k))
    presence = presence.detach().float().cpu()[0]
    top_idx = torch.topk(presence, min(int(top_parts), schema.num_parts)).indices.tolist()

    ncols = 5 if show_gt and "union_mask" in small else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2), squeeze=False)
    axes = axes[0]
    title_gt = schema.obj_names[int(small["obj_label"][0])] if "obj_label" in small else ""
    axes[0].imshow(image)
    axes[0].set_title(f"image\nGT={title_gt}")
    axes[0].axis("off")
    col = 1
    if show_gt and "union_mask" in small:
        axes[col].imshow(image)
        gt = small["union_mask"][0, 0].detach().cpu()
        gt = _resize_prob_to_image(gt.unsqueeze(0), (h, w))[0]
        axes[col].imshow(gt, alpha=0.45)
        axes[col].set_title("GT union mask")
        axes[col].axis("off")
        col += 1
    support = out.get("support_prob")
    axes[col].imshow(image)
    if support is not None:
        sp = _resize_prob_to_image(support.detach().float().cpu()[0], (h, w))[0]
        axes[col].imshow(sp, alpha=0.45)
    axes[col].set_title("predicted support")
    axes[col].axis("off")
    col += 1

    overlay = torch.zeros(h, w)
    for k in top_idx:
        overlay = torch.maximum(overlay, part_prob[int(k)])
    axes[col].imshow(image)
    axes[col].imshow(overlay, alpha=0.50)
    axes[col].set_title("top part-mask union\n" + _top_names(presence, schema.part_names, k=min(4, top_parts)))
    axes[col].axis("off")
    col += 1

    labels = [schema.part_names[k] for k in top_idx][::-1]
    vals = [float(presence[k]) for k in top_idx][::-1]
    axes[col].barh(labels, vals)
    axes[col].set_xlim(0, max(1.0, max(vals) if vals else 1.0))
    axes[col].axvline(float(stage1_model.cfg.presence_threshold), linestyle="--", linewidth=1)
    axes[col].set_title("top predicted presence")
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig




def plot_stage1_part_detail(
    stage1_model: PartCATHKGStage1,
    batch: dict[str, Any],
    *,
    index: int = 0,
    top_parts: int = 8,
    mask_threshold: float = 0.40,
    path: str | Path | None = None,
):
    """Show individual GT/predicted part masks instead of only a union overlay.

    The union plot can hide a serious failure mode: many part channels may light
    up on the whole object or on the same region.  This view gives one row per
    high-presence part with predicted mask, GT mask, and overlap score.
    """

    schema = stage1_model.schema
    dev = next(stage1_model.parameters()).device
    small = _slice_batch(batch, int(index), 1)
    stage1_model.eval()
    with torch.no_grad():
        out = stage1_model(small["image"].to(dev))
    image = _image_from_batch(small, 0)
    h, w = image.shape[:2]
    part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"])).detach().float().cpu()[0]
    part_prob = _resize_prob_to_image(part_prob, (h, w))
    presence = out.get("part_presence")
    if presence is None:
        presence = topk_presence(part_prob.unsqueeze(0), k=int(stage1_model.cfg.topk_presence_k))
    presence = presence.detach().float().cpu()[0]
    gt_masks = small.get("part_masks")
    if gt_masks is not None:
        gt_masks = gt_masks[0].detach().float().cpu()
        gt_masks = _resize_prob_to_image(gt_masks, (h, w))
    top_idx = torch.topk(presence, min(int(top_parts), schema.num_parts)).indices.tolist()

    nrows = len(top_idx)
    fig, axes = plt.subplots(max(nrows, 1), 4, figsize=(15, 3.0 * max(nrows, 1)), squeeze=False)
    for row, k in enumerate(top_idx):
        pred = part_prob[int(k)]
        pred_bin = pred >= float(mask_threshold)
        gt = gt_masks[int(k)] if gt_masks is not None else torch.zeros_like(pred)
        gt_bin = gt > 0.5
        inter = float((pred_bin & gt_bin).sum().item())
        union = float((pred_bin | gt_bin).sum().item())
        iou = inter / max(union, 1.0)
        pres = float(presence[int(k)].item())
        name = schema.part_names[int(k)]

        axes[row, 0].imshow(image)
        axes[row, 0].imshow(pred, alpha=0.50)
        axes[row, 0].set_title(f"pred {name}\np={pres:.2f}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(gt, alpha=0.50)
        axes[row, 1].set_title(f"GT {name}\narea={float(gt_bin.float().mean()):.3f}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_bin.float(), vmin=0, vmax=1)
        axes[row, 2].set_title(f"pred binary\nIoU={iou:.3f}")
        axes[row, 2].axis("off")

        overlap = torch.zeros(h, w, 3)
        overlap[..., 0] = pred_bin.float()
        overlap[..., 1] = gt_bin.float()
        axes[row, 3].imshow(image)
        axes[row, 3].imshow(overlap, alpha=0.55)
        axes[row, 3].set_title("red=pred, green=GT")
        axes[row, 3].axis("off")
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig

def plot_aog_parse_sample(
    model: AOGHKGStage2Classifier,
    batch: dict[str, Any],
    *,
    index: int = 0,
    class_idx: int | None = None,
    top_parts: int = 8,
    mask_threshold: float = 0.40,
    path: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Visualize one Stage-2 parse: active masks, selected template edges, and branch logits."""

    detail = explain_aog_parse_for_sample(model, batch, index=index, class_idx=class_idx, mask_threshold=mask_threshold)
    schema = model.schema
    small = detail["batch"]
    image = _image_from_batch(small, 0)
    h, w = image.shape[:2]
    part_prob = _resize_prob_to_image(detail["part_prob"], (h, w))
    role_prob = detail["role_prob"]
    if role_prob.shape[-2:] != (h, w):
        role_prob = F.interpolate(role_prob.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False)[0]
    out = detail["out"]
    part_presence = out["part_presence"].detach().float().cpu()[0]
    top_part_idx = torch.topk(part_presence, min(int(top_parts), schema.num_parts)).indices.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), squeeze=False)
    axes = axes[0]
    axes[0].imshow(image)
    overlay = torch.zeros(h, w)
    for k in top_part_idx:
        overlay = torch.maximum(overlay, part_prob[int(k)])
    axes[0].imshow(overlay, alpha=0.50)
    axes[0].set_title("Stage-1 active functional masks\n" + _top_names(part_presence, schema.part_names, k=5))
    axes[0].axis("off")

    axes[1].imshow(image)
    c = int(detail["summary"]["explained_class_idx"])
    a = int(detail["summary"]["template"])
    edge_df = detail["edge_df"]
    node_df = detail["node_df"]
    active_nodes = node_df[node_df["obs_presence"] >= float(model.stage1.cfg.presence_threshold)].copy()
    centroids: dict[int, tuple[float, float]] = {}
    rid_table = model.role_index_cf.detach().cpu()
    for _, row in active_nodes.head(top_parts).iterrows():
        k = int(row["part_idx"])
        rid = int(rid_table[c, k].item())
        if rid >= 0:
            cen = _mask_centroid(role_prob[rid], thr=mask_threshold)
        else:
            cen = _mask_centroid(part_prob[k], thr=mask_threshold)
        if cen is not None:
            centroids[k] = cen
            axes[1].scatter([cen[0]], [cen[1]], s=35)
            axes[1].text(cen[0] + 2, cen[1] + 2, schema.part_names[k], fontsize=8)
    if isinstance(edge_df, pd.DataFrame) and len(edge_df):
        # Draw only explanatory edges.  In template-fit mode edge energies are
        # non-positive, so use the positive ``edge_draw_score`` instead of testing
        # whether the signed energy contribution is > 0.
        draw_df = edge_df.copy()
        score_col = "edge_draw_score" if "edge_draw_score" in draw_df else "edge_contribution"
        if score_col in draw_df:
            draw_df = draw_df[draw_df[score_col] > 1e-4]
        if "strength" in draw_df:
            draw_df = draw_df[draw_df["strength"] >= float(model.stage1.cfg.presence_threshold)]
        for _, er in draw_df.head(6).iterrows():
            i, j = int(er["part_i_idx"]), int(er["part_j_idx"])
            if i in centroids and j in centroids:
                x0, y0 = centroids[i]
                x1, y1 = centroids[j]
                axes[1].plot([x0, x1], [y0, y1], linewidth=1.5, alpha=0.8)
                axes[1].text((x0 + x1) * 0.5, (y0 + y1) * 0.5, str(er.get("top_observed_channel", "")), fontsize=7)
    axes[1].set_title(f"selected positive HKG edges\nclass={schema.obj_names[c]}, template={a}")
    axes[1].axis("off")

    branches = ["logits", "base_logits", "hkg_logits", "node_logits", "edge_logits", "motif_logits"]
    labels = ["final", "base", "hkg", "node", "edge", "motif"]
    gt = int(detail["summary"]["gt_idx"])
    vals = []
    for key in branches:
        if key in out:
            vals.append(float(out[key][0, gt].detach().cpu()))
        else:
            vals.append(float("nan"))
    axes[2].barh(labels[::-1], vals[::-1])
    axes[2].set_title(f"branch logit for GT={schema.obj_names[gt]}\npred={detail['summary']['pred']}")
    axes[2].axvline(0.0, linewidth=1)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig, detail


def plot_confusion_matrix(
    df: pd.DataFrame,
    schema: RoleSchema,
    *,
    branch: str = "final",
    top_classes: int = 30,
    normalize: bool = True,
    path: str | Path | None = None,
):
    """Plot a compact confusion matrix for the most frequent ground-truth classes."""

    pred_col = f"pred_{branch}_idx"
    if pred_col not in df:
        raise KeyError(f"Missing {pred_col} in dataframe.")
    counts = df["gt_idx"].value_counts().head(int(top_classes)).index.tolist()
    labels = counts
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    lookup = {c: i for i, c in enumerate(labels)}
    for _, row in df.iterrows():
        g, p = int(row["gt_idx"]), int(row[pred_col])
        if g in lookup and p in lookup:
            mat[lookup[g], lookup[p]] += 1.0
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(labels)), max(5, 0.35 * len(labels))))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([schema.obj_names[c] for c in labels], rotation=90, fontsize=8)
    ax.set_yticklabels([schema.obj_names[c] for c in labels], fontsize=8)
    ax.set_xlabel(f"predicted ({branch})")
    ax.set_ylabel("ground truth")
    ax.set_title(f"{branch} confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_calibration_curve(
    df: pd.DataFrame,
    *,
    branch: str = "final",
    n_bins: int = 10,
    path: str | Path | None = None,
):
    conf_col, corr_col = f"conf_{branch}", f"correct_{branch}"
    if conf_col not in df or corr_col not in df:
        raise KeyError(f"Need {conf_col} and {corr_col} in dataframe.")
    conf = df[conf_col].to_numpy(dtype=float)
    corr = df[corr_col].to_numpy(dtype=float)
    bins = np.linspace(0, 1, int(n_bins) + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if np.any(m):
            xs.append(float(conf[m].mean()))
            ys.append(float(corr[m].mean()))
            ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.plot(xs, ys, marker="o")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y, str(n), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"{branch} calibration; ECE={expected_calibration_error(conf, corr, n_bins=n_bins):.3f}")
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


@torch.no_grad()
def collect_edge_residuals(
    model: AOGHKGStage2Classifier,
    loader,
    *,
    device: str | torch.device = "cuda",
    class_source: str = "gt",  # gt or pred
    max_batches: int | None = None,
    mask_threshold: float = 0.40,
) -> pd.DataFrame:
    """Collect selected-template edge residual statistics over a validation subset."""

    dev = torch.device(device)
    schema = model.schema
    rows: list[dict[str, Any]] = []
    model.eval()
    sample_idx = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        out = model(batch, detach_stage1=True, enable_edges=True)
        labels = batch["obj_label"].to(dev, non_blocking=True)
        preds = out["logits"].argmax(-1).to(dev)
        role_prob = out["role_prob"].detach().float()
        role_presence = out["role_presence"].detach().float()
        best_template = out["best_template"].detach().cpu()
        rid = model.role_index_cf.to(dev)
        for b in range(int(labels.shape[0])):
            c = int(labels[b].item()) if class_source == "gt" else int(preds[b].item())
            a = int(best_template[b, c].item())
            edge_rows = [(e, row) for e, row in enumerate(model.template_edges.detach().cpu().tolist()) if int(row[0]) == c and int(row[1]) == a]
            for e, row in edge_rows:
                _, _, pi, pj = [int(x) for x in row]
                ri, rj = int(rid[c, pi].item()), int(rid[c, pj].item())
                if ri < 0 or rj < 0:
                    continue
                gamma = relation_attributes_vectorized(
                    role_prob[b:b + 1, ri:ri + 1].to(dev),
                    role_prob[b:b + 1, rj:rj + 1].to(dev),
                    thr=mask_threshold,
                )[0, 0]
                var = model.template_rel_var[e].to(dev).clamp_min(1e-6)
                resid = (gamma - model.template_rel_mean[e].to(dev)) / torch.sqrt(var)
                ll_t = model._gaussian_ll(gamma, model.template_rel_mean[e].to(dev), var)
                ll_g = model._gaussian_ll(gamma, model.template_rel_global_mean[e].to(dev), model.template_rel_global_var[e].to(dev))
                rows.append({
                    "sample_index": sample_idx,
                    "batch_index": bi,
                    "gt": schema.obj_names[int(labels[b])],
                    "pred": schema.obj_names[int(preds[b])],
                    "class_source": class_source,
                    "class": schema.obj_names[c],
                    "template": a,
                    "edge_idx": e,
                    "part_i": schema.part_names[pi],
                    "part_j": schema.part_names[pj],
                    "support": float(model.template_rel_support[e].detach().cpu()),
                    "information_gain": float(model.template_rel_ig[e].detach().cpu()),
                    "mean_abs_z_residual": float(resid.abs().mean().detach().cpu()),
                    "max_abs_z_residual": float(resid.abs().max().detach().cpu()),
                    "ll_template": float(ll_t.detach().cpu()),
                    "ll_global": float(ll_g.detach().cpu()),
                    "llr": float((ll_t - ll_g).detach().cpu()),
                })
            sample_idx += 1
    return pd.DataFrame(rows)


def save_diagnostic_tables(output_dir: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and len(df):
            df.to_csv(out / f"{name}.csv", index=False)


def _resolve_class_index(schema: RoleSchema, class_or_idx: int | str) -> int:
    if isinstance(class_or_idx, str):
        lower = class_or_idx.lower()
        for i, name in enumerate(schema.obj_names):
            if name.lower() == lower:
                return i
        raise KeyError(f"Unknown class {class_or_idx!r}. Known classes: {schema.obj_names}")
    idx = int(class_or_idx)
    if idx < 0 or idx >= schema.num_classes:
        raise IndexError(f"Class index {idx} outside [0,{schema.num_classes}).")
    return idx


def _template_edge_records(kg: AOGHierarchicalKG, c: int, a: int) -> list[dict[str, Any]]:
    schema = kg.schema
    out: list[dict[str, Any]] = []
    if kg.template_edges.numel() == 0:
        return out
    ch = relation_channel_strengths(kg.template_rel_mean.detach().float().cpu())
    for e, row in enumerate(kg.template_edges.detach().cpu().tolist()):
        ec, ea, i, j = [int(x) for x in row]
        if ec != int(c) or ea != int(a):
            continue
        top_ch = int(ch[e].argmax().item())
        out.append({
            "edge_idx": e,
            "part_i_idx": i,
            "part_i": schema.part_names[i],
            "part_j_idx": j,
            "part_j": schema.part_names[j],
            "support": float(kg.template_rel_support[e]),
            "information_gain": float(kg.template_rel_ig[e]),
            "relation_type": kg.template_rel_type_names[e] if e < len(kg.template_rel_type_names) else "relation",
            "top_template_channel": RELATION_CHANNELS[top_ch],
            "top_template_channel_strength": float(ch[e, top_ch]),
            "dx": float(kg.template_rel_mean[e, 0]),
            "dy": float(kg.template_rel_mean[e, 1]),
            "dist": float(kg.template_rel_mean[e, 2]),
        })
    return sorted(out, key=lambda r: (r["information_gain"], r["support"]), reverse=True)


def _template_motif_records(kg: AOGHierarchicalKG, c: int, a: int) -> list[dict[str, Any]]:
    schema = kg.schema
    out: list[dict[str, Any]] = []
    for m, row in enumerate(kg.motif_edges.detach().cpu().tolist()):
        mc, ma, i, j, mt = [int(x) for x in row]
        if mc != int(c) or ma != int(a):
            continue
        out.append({
            "motif_idx": m,
            "part_i_idx": i,
            "part_i": schema.part_names[i],
            "part_j_idx": j,
            "part_j": schema.part_names[j],
            "motif_type": MOTIF_TYPE_NAMES[mt] if mt < len(MOTIF_TYPE_NAMES) else str(mt),
            "support": float(kg.motif_support[m]),
        })
    return sorted(out, key=lambda r: r["support"], reverse=True)


def _template_nodes(kg: AOGHierarchicalKG, c: int, a: int, *, min_role_prior: float = 0.05, top_k_if_empty: int = 6) -> list[int]:
    prior = kg.template_role_prior[c, a].detach().float().cpu()
    required = kg.template_role_required[c, a].detach().float().cpu() > 0.5
    nodes = set(int(i) for i in (prior >= float(min_role_prior)).nonzero(as_tuple=False).flatten().tolist())
    nodes.update(int(i) for i in required.nonzero(as_tuple=False).flatten().tolist())
    for r in _template_edge_records(kg, c, a):
        nodes.add(int(r["part_i_idx"]))
        nodes.add(int(r["part_j_idx"]))
    for r in _template_motif_records(kg, c, a):
        nodes.add(int(r["part_i_idx"]))
        nodes.add(int(r["part_j_idx"]))
    if not nodes:
        nodes.update(int(i) for i in torch.topk(prior, min(int(top_k_if_empty), prior.numel())).indices.tolist())
    return sorted(nodes)


def _normalize_positions(pos: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    if not pos:
        return {}
    arr = np.array(list(pos.values()), dtype=float)
    if arr.shape[0] == 1:
        return {next(iter(pos.keys())): (0.5, 0.5)}
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    out: dict[int, tuple[float, float]] = {}
    for k, (x, y) in pos.items():
        xx, yy = ((np.array([x, y]) - mn) / span) * 0.80 + 0.10
        out[int(k)] = (float(xx), float(yy))
    return out


def _canonical_template_layout(kg: AOGHierarchicalKG, c: int, a: int, nodes: list[int]) -> dict[int, tuple[float, float]]:
    """Approximate a 2-D template layout from pairwise dx/dy relation means.

    The learned relation dx/dy values are local union-box-normalized pair
    attributes, so this layout should be read as a qualitative configuration:
    relative direction and rough grouping, not metric object geometry.
    """
    nodes = list(dict.fromkeys(int(n) for n in nodes))
    if not nodes:
        return {}
    idx = {n: p for p, n in enumerate(nodes)}
    edges = [r for r in _template_edge_records(kg, c, a) if int(r["part_i_idx"]) in idx and int(r["part_j_idx"]) in idx]
    if len(nodes) == 1 or not edges:
        theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        return {n: (0.5 + 0.35 * np.cos(t), 0.5 + 0.35 * np.sin(t)) for n, t in zip(nodes, theta)}
    n = len(nodes)
    rows: list[np.ndarray] = []
    bx: list[float] = []
    by: list[float] = []
    for e in edges:
        i, j = int(e["part_i_idx"]), int(e["part_j_idx"])
        row = np.zeros(n, dtype=float)
        row[idx[j]] = 1.0
        row[idx[i]] = -1.0
        w = max(0.15, float(e["support"])) * (1.0 + min(float(e["information_gain"]), 2.0))
        row = row * w
        rows.append(row)
        bx.append(float(e["dx"]) * w)
        by.append(float(e["dy"]) * w)
    # Anchor the mean location at zero.  This fixes translation gauge.
    rows.append(np.ones(n, dtype=float) * 0.2)
    bx.append(0.0)
    by.append(0.0)
    A = np.stack(rows, axis=0)
    x = np.linalg.lstsq(A, np.array(bx), rcond=None)[0]
    y = np.linalg.lstsq(A, np.array(by), rcond=None)[0]
    pos = {node: (float(x[idx[node]]), float(y[idx[node]])) for node in nodes}
    return _normalize_positions(pos)


def _draw_configuration_graph(
    ax,
    positions: dict[int, tuple[float, float]],
    schema: RoleSchema,
    *,
    node_weights: dict[int, float] | None = None,
    required: set[int] | None = None,
    edges: list[dict[str, Any]] | pd.DataFrame | None = None,
    motifs: list[dict[str, Any]] | pd.DataFrame | None = None,
    title: str = "configuration",
    edge_label: str = "top_template_channel",
    max_edges: int = 12,
    max_motifs: int = 8,
):
    node_weights = node_weights or {}
    required = required or set()
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # image-like vertical axis: smaller y is higher
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    def _records(obj):
        if obj is None:
            return []
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return list(obj)

    edge_recs = _records(edges)[:max_edges]
    motif_recs = _records(motifs)[:max_motifs]

    for rec in edge_recs:
        i = int(rec.get("part_i_idx", -1))
        j = int(rec.get("part_j_idx", -1))
        if i not in positions or j not in positions:
            continue
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        width = 0.8 + 2.0 * min(float(rec.get("support", rec.get("strength", 0.0))), 1.0)
        ax.plot([x0, x1], [y0, y1], linewidth=width, alpha=0.45)
        lab = str(rec.get(edge_label, rec.get("relation_type", "")))
        if lab:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, fontsize=7, ha="center", va="center")

    for rec in motif_recs:
        i = int(rec.get("part_i_idx", -1))
        j = int(rec.get("part_j_idx", -1))
        if i not in positions or j not in positions:
            continue
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=2.0, alpha=0.85)
        lab = str(rec.get("motif_type", "motif"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.035, lab, fontsize=8, ha="center", va="center")

    for k, (x, y) in positions.items():
        w = float(node_weights.get(int(k), 0.25))
        size = 180.0 + 650.0 * min(max(w, 0.0), 1.0)
        marker = "s" if int(k) in required else "o"
        ax.scatter([x], [y], s=size, marker=marker, alpha=0.85)
        ax.text(x, y, schema.part_names[int(k)], fontsize=8, ha="center", va="center")


def plot_hkg_template_configuration(
    kg: AOGHierarchicalKG,
    class_or_idx: int | str,
    template: int | None = None,
    *,
    min_role_prior: float = 0.05,
    max_edges: int = 12,
    max_motifs: int = 8,
    path: str | Path | None = None,
):
    """Draw the learned class-template configuration and motif-only view.

    This is more direct than the image overlay: it visualizes the grammar branch
    itself, i.e. the class -> template -> roles configuration stored in the HKG.
    """
    schema = kg.schema
    c = _resolve_class_index(schema, class_or_idx)
    if template is None:
        valid_prior = kg.template_prior[c].detach().float().cpu().clone()
        valid_prior[kg.template_valid[c].detach().float().cpu() <= 0] = -1.0
        template = int(valid_prior.argmax().item())
    a = int(template)
    nodes = _template_nodes(kg, c, a, min_role_prior=min_role_prior)
    pos = _canonical_template_layout(kg, c, a, nodes)
    role_prior = kg.template_role_prior[c, a].detach().float().cpu()
    required = set(int(i) for i in (kg.template_role_required[c, a].detach().float().cpu() > 0.5).nonzero(as_tuple=False).flatten().tolist())
    weights = {int(i): float(role_prior[int(i)]) for i in nodes}
    edge_records = _template_edge_records(kg, c, a)
    motif_records = _template_motif_records(kg, c, a)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    axes = axes[0]
    title = f"{schema.obj_names[c]} template {a}\nprior={float(kg.template_prior[c,a]):.3f}, roles={len(nodes)}, edges={len(edge_records)}"
    _draw_configuration_graph(axes[0], pos, schema, node_weights=weights, required=required, edges=edge_records, motifs=[], title=title, max_edges=max_edges)
    _draw_configuration_graph(axes[1], pos, schema, node_weights=weights, required=required, edges=[], motifs=motif_records, title=f"motifs only\ncount={len(motif_records)}", max_motifs=max_motifs)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    tables = {
        "roles": pd.DataFrame({
            "part_idx": nodes,
            "part": [schema.part_names[i] for i in nodes],
            "role_prior": [float(role_prior[i]) for i in nodes],
            "required": [i in required for i in nodes],
        }).sort_values("role_prior", ascending=False),
        "edges": pd.DataFrame(edge_records),
        "motifs": pd.DataFrame(motif_records),
    }
    return fig, tables


def plot_hkg_class_templates_grid(
    kg: AOGHierarchicalKG,
    class_or_idx: int | str,
    *,
    min_role_prior: float = 0.05,
    max_edges: int = 8,
    path: str | Path | None = None,
):
    """Show all valid Or-node template branches for one class side-by-side."""
    schema = kg.schema
    c = _resolve_class_index(schema, class_or_idx)
    valid = [a for a in range(int(kg.num_templates)) if float(kg.template_valid[c, a]) > 0]
    if not valid:
        valid = [int(kg.template_prior[c].argmax().item())]
    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4.8), squeeze=False)
    axes = axes[0]
    for ax, a in zip(axes, valid):
        nodes = _template_nodes(kg, c, a, min_role_prior=min_role_prior)
        pos = _canonical_template_layout(kg, c, a, nodes)
        role_prior = kg.template_role_prior[c, a].detach().float().cpu()
        required = set(int(i) for i in (kg.template_role_required[c, a].detach().float().cpu() > 0.5).nonzero(as_tuple=False).flatten().tolist())
        weights = {int(i): float(role_prior[int(i)]) for i in nodes}
        edges = _template_edge_records(kg, c, a)
        title = f"template {a}\nprior={float(kg.template_prior[c,a]):.2f}, edges={len(edges)}"
        _draw_configuration_graph(ax, pos, schema, node_weights=weights, required=required, edges=edges, motifs=[], title=title, max_edges=max_edges)
    fig.suptitle(f"HKG Or-node alternatives for {schema.obj_names[c]}", y=1.02)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _observed_positions_from_detail(model: AOGHKGStage2Classifier, detail: dict[str, Any], *, min_presence: float | None = None, mask_threshold: float = 0.40, top_parts: int = 10) -> dict[int, tuple[float, float]]:
    schema = model.schema
    c = int(detail["summary"]["explained_class_idx"])
    node_df = detail["node_df"].copy()
    if min_presence is None:
        min_presence = float(model.stage1.cfg.presence_threshold)
    active = node_df[node_df["obs_presence"] >= float(min_presence)].sort_values("obs_presence", ascending=False).head(int(top_parts))
    role_prob = detail["role_prob"]
    part_prob = detail["part_prob"]
    rid_table = model.role_index_cf.detach().cpu()
    h, w = part_prob.shape[-2:]
    pos: dict[int, tuple[float, float]] = {}
    for _, row in active.iterrows():
        k = int(row["part_idx"])
        rid = int(rid_table[c, k].item())
        cen = _mask_centroid(role_prob[rid], thr=mask_threshold) if rid >= 0 else _mask_centroid(part_prob[k], thr=mask_threshold)
        if cen is None:
            continue
        x, y = cen
        pos[k] = (float(x) / max(float(w - 1), 1.0), float(y) / max(float(h - 1), 1.0))
    return pos


def plot_parse_configuration_and_motifs(
    model: AOGHKGStage2Classifier,
    batch: dict[str, Any],
    *,
    index: int = 0,
    class_idx: int | None = None,
    top_parts: int = 10,
    mask_threshold: float = 0.40,
    min_edge_contribution: float = 1e-4,
    min_motif_contribution: float = 1e-5,
    path: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Show a direct parse configuration: observed nodes, positive edges, motifs, and selected template.

    The previous overlay mixed masks, edges, template edges, and branch logits in
    one plot.  This figure separates the four concepts so the user can see:
    observed configuration C=(V,E), active motifs, and the selected template prior.
    """
    detail = explain_aog_parse_for_sample(model, batch, index=index, class_idx=class_idx, mask_threshold=mask_threshold)
    schema = model.schema
    small = detail["batch"]
    image = _image_from_batch(small, 0)
    h, w = image.shape[:2]
    part_prob = _resize_prob_to_image(detail["part_prob"], (h, w))
    part_presence = detail["out"]["part_presence"].detach().float().cpu()[0]
    top_part_idx = torch.topk(part_presence, min(int(top_parts), schema.num_parts)).indices.tolist()
    overlay = torch.zeros(h, w)
    for k in top_part_idx:
        overlay = torch.maximum(overlay, part_prob[int(k)])

    c = int(detail["summary"]["explained_class_idx"])
    a = int(detail["summary"]["template"])
    obs_pos = _observed_positions_from_detail(model, detail, mask_threshold=mask_threshold, top_parts=top_parts)
    node_weights = {int(r["part_idx"]): float(r["obs_presence"]) for _, r in detail["node_df"].iterrows() if int(r["part_idx"]) in obs_pos}
    required = set(int(i) for i in (model.template_role_required[c, a].detach().float().cpu() > 0.5).nonzero(as_tuple=False).flatten().tolist())

    edge_df = detail["edge_df"].copy() if isinstance(detail["edge_df"], pd.DataFrame) else pd.DataFrame()
    if len(edge_df):
        score_col = "edge_draw_score" if "edge_draw_score" in edge_df else "edge_contribution"
        edge_df = edge_df[edge_df[score_col] > float(min_edge_contribution)]
    motif_df = detail["motif_df"].copy() if isinstance(detail["motif_df"], pd.DataFrame) else pd.DataFrame()
    if len(motif_df):
        motif_df = motif_df[motif_df["motif_contribution"] > float(min_motif_contribution)]
        # Add endpoint indices for the generic graph drawer.
        name_to_idx = {name: i for i, name in enumerate(schema.part_names)}
        motif_df["part_i_idx"] = motif_df["part_i"].map(name_to_idx).fillna(-1).astype(int)
        motif_df["part_j_idx"] = motif_df["part_j"].map(name_to_idx).fillna(-1).astype(int)

    nodes = _template_nodes(model.kg, c, a, min_role_prior=0.05)
    tmpl_pos = _canonical_template_layout(model.kg, c, a, nodes)
    tmpl_weights = {int(i): float(model.template_role_prior[c, a, int(i)].detach().cpu()) for i in nodes}
    tmpl_edges = _template_edge_records(model.kg, c, a)
    tmpl_motifs = _template_motif_records(model.kg, c, a)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)
    axes = axes[0]
    axes[0].imshow(image)
    axes[0].imshow(overlay, alpha=0.45)
    axes[0].set_title("Stage-1 masks used by parse\n" + _top_names(part_presence, schema.part_names, k=5))
    axes[0].axis("off")

    _draw_configuration_graph(
        axes[1], obs_pos, schema,
        node_weights=node_weights,
        required=required,
        edges=edge_df,
        motifs=[],
        title=f"observed configuration C=(V,E)\nclass={schema.obj_names[c]}, template={a}",
        edge_label="top_observed_channel",
        max_edges=10,
    )
    _draw_configuration_graph(
        axes[2], obs_pos, schema,
        node_weights=node_weights,
        required=required,
        edges=[],
        motifs=motif_df,
        title=f"active motif factors\ncount={len(motif_df)}",
        max_motifs=10,
    )
    _draw_configuration_graph(
        axes[3], tmpl_pos, schema,
        node_weights=tmpl_weights,
        required=required,
        edges=tmpl_edges,
        motifs=tmpl_motifs,
        title=f"selected template prior\nedges={len(tmpl_edges)}, motifs={len(tmpl_motifs)}",
        edge_label="top_template_channel",
        max_edges=10,
        max_motifs=6,
    )
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig, detail


def summarize_template_structure_quality(kg: AOGHierarchicalKG) -> pd.DataFrame:
    """Compact table for diagnosing template/motif density and Or-node collapse."""
    schema = kg.schema
    rows: list[dict[str, Any]] = []
    for c in range(schema.num_classes):
        for a in range(int(kg.num_templates)):
            if float(kg.template_valid[c, a]) <= 0:
                continue
            edges = _template_edge_records(kg, c, a)
            motifs = _template_motif_records(kg, c, a)
            role_prior = kg.template_role_prior[c, a].detach().float().cpu()
            active_roles = int((role_prior >= 0.05).sum().item())
            rows.append({
                "class_idx": c,
                "class": schema.obj_names[c],
                "template": a,
                "prior": float(kg.template_prior[c, a]),
                "active_roles@0.05": active_roles,
                "required_roles": int((kg.template_role_required[c, a] > 0.5).sum().item()),
                "edge_count": len(edges),
                "motif_count": len(motifs),
                "mean_edge_support": float(np.mean([e["support"] for e in edges])) if edges else 0.0,
                "mean_edge_ig": float(np.mean([e["information_gain"] for e in edges])) if edges else 0.0,
                "motif_to_edge_ratio": float(len(motifs) / max(len(edges), 1)),
                "top_roles": ", ".join(f"{schema.part_names[i]}:{float(role_prior[i]):.2f}" for i in torch.topk(role_prior, min(5, role_prior.numel())).indices.tolist()),
            })
    return pd.DataFrame(rows).sort_values(["class", "template"])
