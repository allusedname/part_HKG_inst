#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.collate import collate_part_batch
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.kg.instance_aog_builder import build_instance_aog
from partcat_hkg.kg.instance_aog_io import save_instance_aog
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an Instance-Slot AOG grammar from a frozen Stage-1 checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--num-templates-per-class", type=int, default=None)
    parser.add_argument("--component-threshold", type=float, default=None)
    parser.add_argument("--max-components-per-part", type=int, default=None)
    parser.add_argument("--max-total-components", type=int, default=None)
    parser.add_argument("--min-component-area-frac", type=float, default=None)
    parser.add_argument("--component-min-presence", type=float, default=None)
    parser.add_argument("--use-gt-components", action="store_true", help="Build component geometry from GT part masks instead of Stage-1 predictions.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    if args.max_images_per_class is not None:
        cfg.model.hkg.max_images_per_class = int(args.max_images_per_class)
    if args.num_templates_per_class is not None:
        cfg.model.hkg.num_templates_per_class = int(args.num_templates_per_class)
    if args.component_threshold is not None:
        setattr(cfg.model.hkg, "component_threshold", float(args.component_threshold))
    if args.max_components_per_part is not None:
        setattr(cfg.model.hkg, "max_components_per_part", int(args.max_components_per_part))
    if args.max_total_components is not None:
        setattr(cfg.model.hkg, "max_total_components", int(args.max_total_components))
    if args.min_component_area_frac is not None:
        setattr(cfg.model.hkg, "min_component_area_frac", float(args.min_component_area_frac))
    if args.component_min_presence is not None:
        setattr(cfg.model.hkg, "component_min_presence", float(args.component_min_presence))
    if args.use_gt_components:
        cfg.model.hkg.use_predicted_stage1_evidence = False

    set_seed(cfg.seed)
    device = _resolve_device(args.device)
    train_ds, _ = make_datasets(cfg)
    if hasattr(train_ds, "transform") and hasattr(train_ds.transform, "train"):
        train_ds.transform.train = False
    common = dict(
        batch_size=int(args.batch_size or cfg.training.batch_size_stage1),
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_part_batch,
    )
    if cfg.data.num_workers > 0:
        common.update(persistent_workers=cfg.data.persistent_workers, prefetch_factor=cfg.data.prefetch_factor)
    loader = DataLoader(train_ds, **common)
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    grammar = build_instance_aog(stage1, loader, train_ds.schema, cfg.model.hkg, device=device)
    out = Path(args.out or Path(cfg.paths.save_dir) / "checkpoints" / "instance_aog.pt")
    save_instance_aog(grammar, out)
    print(f"saved Instance-AOG to {out}")
    print(
        f"templates/class={grammar.num_templates} max_slots={grammar.max_slots} "
        f"edges={grammar.edges.shape[0]} families={len(grammar.family_names)}"
    )


if __name__ == "__main__":
    main()
