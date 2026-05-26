#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.data.partimagenet import resolve_partimagenet_split_paths
from partcat_hkg.data.partimagenet import describe_partimagenet_layout
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.evaluation.stage1 import stage1_audit
from partcat_hkg.models.losses import stage1_loss
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.training.stage1_trainer import train_stage1
from partcat_hkg.analysis.stage1_run_diagnostics import make_diagnostic_loader, run_stage1_diagnostics
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _apply_overrides(cfg, args):
    if args.save_dir:
        cfg.paths.save_dir = args.save_dir
    if args.partimagenet_root:
        cfg.paths.partimagenet_root = args.partimagenet_root
    if args.train_annotations:
        cfg.paths.train_annotations = args.train_annotations
    if args.val_annotations:
        cfg.paths.val_annotations = args.val_annotations
    if args.train_image_root:
        cfg.paths.train_image_root = args.train_image_root
    if args.val_image_root:
        cfg.paths.val_image_root = args.val_image_root
    if args.img_size is not None:
        cfg.data.img_size = int(args.img_size)
    if args.batch_size is not None:
        cfg.training.batch_size_stage1 = int(args.batch_size)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    if args.max_train_samples is not None:
        cfg.data.max_train_samples = int(args.max_train_samples)
    if args.max_val_samples is not None:
        cfg.data.max_val_samples = int(args.max_val_samples)
    if args.epochs is not None:
        cfg.training.stage1_epochs = int(args.epochs)
    if args.lr is not None:
        cfg.training.lr_stage1 = float(args.lr)
    if args.no_amp:
        cfg.training.use_amp = False
    if args.no_dino:
        cfg.model.stage1.use_dino = False
    if args.no_clip_text:
        cfg.model.stage1.use_clip_text = False

    # Stage-1 quality/small-part options. These are intentionally exposed here
    # so Stage 1 can be run only through scripts/train_stage1.py, without a
    # separate fine-tuning notebook or helper training entry point.
    if args.quality_loss:
        cfg.loss.stage1.quality_enable = True
    if args.no_quality_loss:
        cfg.loss.stage1.quality_enable = False
    if args.use_highres_refine:
        cfg.model.stage1.use_highres_refine = True
    if args.no_highres_refine:
        cfg.model.stage1.use_highres_refine = False
    if args.highres_refine_dim is not None:
        cfg.model.stage1.highres_refine_dim = int(args.highres_refine_dim)
    if args.presence_threshold is not None:
        cfg.model.stage1.presence_threshold = float(args.presence_threshold)
    if args.topk_presence_k is not None:
        cfg.model.stage1.topk_presence_k = int(args.topk_presence_k)
    if args.small_part_area_tau is not None:
        cfg.model.stage1.small_part_area_tau = float(args.small_part_area_tau)
    if args.small_part_weight_max is not None:
        cfg.model.stage1.small_part_weight_max = float(args.small_part_weight_max)
    if args.small_part_weight_power is not None:
        cfg.model.stage1.small_part_weight_power = float(args.small_part_weight_power)

    # Optional overrides for quality-loss coefficients. Leave as config values
    # unless explicitly supplied from the command line.
    qmap = {
        "quality_presence_bce": args.quality_presence_bce,
        "valid_absent_topmean_fp": args.valid_absent_topmean_fp,
        "valid_absent_mean_fp": args.valid_absent_mean_fp,
        "invalid_part_topmean": args.invalid_part_topmean,
        "invalid_part_mean": args.invalid_part_mean,
        "gt_support_leak": args.gt_support_leak,
        "pred_support_containment": args.pred_support_containment,
        "boundary": args.boundary_loss,
        "focal_functional": args.focal_functional,
        "tversky_functional": args.tversky_functional,
        "quality_topq": args.quality_topq,
    }
    for key, value in qmap.items():
        if value is not None:
            setattr(cfg.loss.stage1, key, float(value))
    return cfg


def _synthetic_schema() -> RoleSchema:
    return RoleSchema.from_names(
        ["car", "bird"],
        ["body", "wheel", "wing"],
        ["car:body", "car:wheel", "bird:body", "bird:wing"],
    )


def run_synthetic_smoke(cfg, device: str) -> None:
    cfg.model.stage1.use_dino = False
    cfg.model.stage1.use_clip_text = False
    cfg.model.stage1.backbone_name = "tiny"
    cfg.model.stage1.model_dim = min(int(cfg.model.stage1.model_dim), 64)
    cfg.model.stage1.fuse_dim = min(int(cfg.model.stage1.fuse_dim), 48)
    cfg.model.stage1.token_dim = min(int(cfg.model.stage1.token_dim), 32)
    cfg.model.stage1.cost_embed_dim = min(int(cfg.model.stage1.cost_embed_dim), 8)
    schema = _synthetic_schema()
    model = PartCATHKGStage1(schema, cfg.model.stage1).to(device)
    shapes = model.smoke_forward(batch_size=2, image_size=96, device=device)
    print("[synthetic smoke]", shapes)

    batch = {
        "image": torch.randn(2, 3, 96, 96, device=device),
        "part_masks": torch.zeros(2, schema.num_parts, 96, 96, device=device),
        "role_masks": torch.zeros(2, schema.num_roles, 96, 96, device=device),
        "union_mask": torch.zeros(2, 1, 96, 96, device=device),
        "obj_label": torch.tensor([0, 1], dtype=torch.long, device=device),
    }
    batch["part_masks"][0, 0, 24:70, 24:70] = 1
    batch["part_masks"][0, 1, 55:85, 10:35] = 1
    batch["part_masks"][1, 2, 20:76, 10:86] = 1
    batch["role_masks"][0, 0] = batch["part_masks"][0, 0]
    batch["role_masks"][0, 1] = batch["part_masks"][0, 1]
    batch["role_masks"][1, 3] = batch["part_masks"][1, 2]
    batch["union_mask"] = batch["part_masks"].amax(dim=1, keepdim=True)
    out = model(batch["image"])
    loss, logs = stage1_loss(out, batch, schema, cfg.loss.stage1, topk_presence_k=cfg.model.stage1.topk_presence_k)
    print("[synthetic smoke] loss", float(loss.detach().cpu()), logs)


def run_real_smoke(model, train_loader, cfg, device: str) -> None:
    model.eval()
    batch = next(iter(train_loader))
    with torch.no_grad():
        out = model(batch["image"][: min(2, batch["image"].shape[0])].to(device))
    print("[smoke] support_logits", tuple(out["support_logits"].shape), "finite", bool(torch.isfinite(out["support_logits"]).all()))
    print("[smoke] part_logits", tuple(out["part_logits"].shape), "finite", bool(torch.isfinite(out["part_logits"]).all()))
    print("[smoke] role_logits", tuple(out["role_logits"].shape), "finite", bool(torch.isfinite(out["role_logits"]).all()))
    print("[smoke] part_presence", tuple(out["part_presence"].shape), "part_tokens", tuple(out["part_tokens"].shape))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or smoke-test Stage 1 PartCAT-HKG segmentation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--partimagenet-root", default="")
    parser.add_argument("--train-annotations", default="", help="Relative/absolute train COCO JSON; default is annotations/train/train.json")
    parser.add_argument("--val-annotations", default="", help="Relative/absolute val COCO JSON; default is annotations/val/val.json")
    parser.add_argument("--train-image-root", default="", help="Relative/absolute train image directory; default is images/train")
    parser.add_argument("--val-image-root", default="", help="Relative/absolute val image directory; default is images/val")
    parser.add_argument("--print-data-layout", action="store_true", help="Print resolved PartImageNet root diagnostics and exit.")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0, help="Set torch CPU threads; useful for notebook CPU smoke tests.")
    parser.add_argument("--no-dino", action="store_true", help="Disable DINO branch and use zero structural guidance.")
    parser.add_argument("--no-clip-text", action="store_true", help="Disable CLIP and use deterministic random text prototypes.")
    parser.add_argument("--warm-start", default="", help="Optional Stage-1 checkpoint to initialize from before training.")
    parser.add_argument("--allow-partial-load", action="store_true", help="Load warm-start with strict=False; required when enabling new high-res heads.")
    parser.add_argument("--quality-loss", action="store_true", help="Enable small-part/presence/containment Stage-1 quality losses.")
    parser.add_argument("--no-quality-loss", action="store_true", help="Disable quality losses even if enabled in config.")
    parser.add_argument("--use-highres-refine", action="store_true", help="Enable the high-resolution refinement branch.")
    parser.add_argument("--no-highres-refine", action="store_true", help="Disable high-resolution refinement branch.")
    parser.add_argument("--highres-refine-dim", type=int, default=None)
    parser.add_argument("--presence-threshold", type=float, default=None)
    parser.add_argument("--topk-presence-k", type=int, default=None)
    parser.add_argument("--small-part-area-tau", type=float, default=None)
    parser.add_argument("--small-part-weight-max", type=float, default=None)
    parser.add_argument("--small-part-weight-power", type=float, default=None)
    parser.add_argument("--quality-presence-bce", type=float, default=None)
    parser.add_argument("--valid-absent-topmean-fp", type=float, default=None)
    parser.add_argument("--valid-absent-mean-fp", type=float, default=None)
    parser.add_argument("--invalid-part-topmean", type=float, default=None)
    parser.add_argument("--invalid-part-mean", type=float, default=None)
    parser.add_argument("--gt-support-leak", type=float, default=None)
    parser.add_argument("--pred-support-containment", type=float, default=None)
    parser.add_argument("--boundary-loss", type=float, default=None)
    parser.add_argument("--focal-functional", type=float, default=None)
    parser.add_argument("--tversky-functional", type=float, default=None)
    parser.add_argument("--quality-topq", type=float, default=None)
    parser.add_argument("--smoke-only", action="store_true", help="Run one real-data forward/audit smoke and exit.")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run a no-dataset synthetic forward/loss smoke and exit.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Load a Stage-1 checkpoint and save diagnostic figures/tables, then exit.")
    parser.add_argument("--diagnostics-after-training", action="store_true", help="Run the same diagnostics after the final training epoch.")
    parser.add_argument("--diag-checkpoint", default="", help="Checkpoint for --diagnostics-only. Defaults to --warm-start, then save-dir/checkpoints/stage1_best.pt.")
    parser.add_argument("--diag-save-dir", default="", help="Where to save Stage-1 diagnostic figures/tables. Defaults to save-dir/diagnostics_stage1.")
    parser.add_argument("--diag-max-batches", type=int, default=50, help="Maximum validation batches for diagnostics; use -1 for full validation.")
    parser.add_argument("--diag-num-samples", type=int, default=8, help="Number of individual validation samples to visualize.")
    parser.add_argument("--diag-max-parts-per-sample", type=int, default=8, help="Maximum part rows in each sample diagnostic figure.")
    parser.add_argument("--diag-mask-threshold", type=float, default=0.40, help="Mask threshold used for IoU/visualization diagnostics.")
    parser.add_argument("--diag-presence-thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50", help="Comma-separated thresholds for presence sweep.")
    parser.add_argument("--diag-balanced-samples-per-class", type=int, default=8, help="Use a class-balanced validation subset for diagnostics; 0 disables.")
    parser.add_argument("--no-diag-zip", action="store_true", help="Do not create diagnostics zip file.")
    args = parser.parse_args()

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    cfg = _apply_overrides(load_config(args.config), args)
    set_seed(cfg.seed)
    device = _resolve_device(args.device)

    if args.synthetic_smoke:
        run_synthetic_smoke(cfg, device)
        # Some CPU/PyTorch builds keep worker threads alive after large attention
        # smoke tests. Synthetic smoke is a diagnostic-only path, so exit hard
        # after flushing to keep notebooks and CI from hanging at interpreter
        # shutdown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if args.print_data_layout:
        print(describe_partimagenet_layout(cfg.paths.partimagenet_root))
        return

    train_ds, val_ds = make_datasets(cfg)
    train_loader, val_loader, _, _ = make_loaders(cfg, train_ds, val_ds)
    model = PartCATHKGStage1(train_ds.schema, cfg.model.stage1).to(device)
    print("Stage1 params:", sum(p.numel() for p in model.parameters()), "trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Stage1 status:", model.status)
    print("Stage1 quality loss enabled:", bool(cfg.loss.stage1.quality_enable))
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)

    if args.warm_start:
        ckpt = Path(args.warm_start)
        if not ckpt.exists():
            raise FileNotFoundError(f"--warm-start checkpoint not found: {ckpt}")
        extra = load_checkpoint(ckpt, model, strict=not bool(args.allow_partial_load))
        print(f"Loaded warm-start checkpoint: {ckpt} (strict={not bool(args.allow_partial_load)})")
        if isinstance(extra, dict) and extra.get("epoch") is not None:
            print(f"Warm-start source epoch: {extra.get('epoch')}")

    def _run_diagnostics_and_exit_or_continue():
        diag_batches = None if int(args.diag_max_batches) < 0 else int(args.diag_max_batches)
        diag_dir = Path(args.diag_save_dir) if args.diag_save_dir else Path(cfg.paths.save_dir) / "diagnostics_stage1"
        diag_loader = make_diagnostic_loader(
            val_ds,
            batch_size=cfg.training.batch_size_stage1,
            num_workers=0,
            balanced_samples_per_class=int(args.diag_balanced_samples_per_class),
        )
        return run_stage1_diagnostics(
            model,
            diag_loader,
            cfg,
            device=device,
            output_dir=diag_dir,
            max_batches=diag_batches,
            num_samples=int(args.diag_num_samples),
            max_parts_per_sample=int(args.diag_max_parts_per_sample),
            mask_threshold=float(args.diag_mask_threshold),
            thresholds=args.diag_presence_thresholds,
            save_dir_for_history=Path(cfg.paths.save_dir),
            make_zip=not bool(args.no_diag_zip),
        )

    if args.diagnostics_only:
        diag_ckpt = Path(args.diag_checkpoint or args.warm_start or (Path(cfg.paths.save_dir) / "checkpoints" / "stage1_best.pt"))
        if not diag_ckpt.exists():
            raise FileNotFoundError(
                f"Diagnostics checkpoint not found: {diag_ckpt}. Supply --diag-checkpoint or set --save-dir to the run directory."
            )
        extra = load_checkpoint(diag_ckpt, model, strict=not bool(args.allow_partial_load))
        print(f"Loaded diagnostics checkpoint: {diag_ckpt} (strict={not bool(args.allow_partial_load)})")
        if isinstance(extra, dict) and extra.get("epoch") is not None:
            print(f"Diagnostics checkpoint epoch: {extra.get('epoch')}")
        _run_diagnostics_and_exit_or_continue()
        return

    run_real_smoke(model, train_loader, cfg, device)
    if args.smoke_only:
        audit = stage1_audit(
            model,
            val_loader,
            schema=train_ds.schema,
            device=device,
            max_batches=1,
            mask_bin_thr=cfg.analysis.mask_bin_thr,
            presence_tau=cfg.model.stage1.presence_threshold,
        )
        print("[smoke audit]", audit)
        return

    train_stage1(model, train_loader, val_loader, cfg, device=device)
    if args.diagnostics_after_training:
        best_ckpt = Path(cfg.paths.save_dir) / "checkpoints" / "stage1_best.pt"
        if best_ckpt.exists():
            load_checkpoint(best_ckpt, model, strict=not bool(args.allow_partial_load))
            print(f"Loaded best checkpoint for post-training diagnostics: {best_ckpt}")
        _run_diagnostics_and_exit_or_continue()


if __name__ == "__main__":
    main()
