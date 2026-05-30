#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.kg.instance_aog import InstanceAOG
from partcat_hkg.kg.instance_aog_io import load_instance_aog
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.stage2.instance_aog_classifier import InstanceAOGStage2Classifier
from partcat_hkg.training.instance_aog_trainer import train_instance_aog_stage2
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 Instance-Slot AOG classifier with a frozen Stage-1 checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--component-threshold", type=float, default=None)
    parser.add_argument("--max-components-per-part", type=int, default=None)
    parser.add_argument("--max-total-components", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.save_dir:
        cfg.paths.save_dir = args.save_dir
    if args.batch_size is not None:
        cfg.training.batch_size_stage2 = int(args.batch_size)
    if args.epochs is not None:
        cfg.training.stage2_epochs = int(args.epochs)
    if args.lr is not None:
        cfg.training.lr_stage2 = float(args.lr)
    if args.component_threshold is not None:
        setattr(cfg.model.stage2, "component_threshold", float(args.component_threshold))
    if args.max_components_per_part is not None:
        setattr(cfg.model.stage2, "max_components_per_part", int(args.max_components_per_part))
    if args.max_total_components is not None:
        setattr(cfg.model.stage2, "max_total_components", int(args.max_total_components))
    if args.no_amp:
        cfg.training.use_amp = False
    cfg.data.use_stage2_image_only_loader = True
    set_seed(cfg.seed)
    device = _resolve_device(args.device)

    train_ds, val_ds = make_datasets(cfg)
    _, _, stage2_train, stage2_val = make_loaders(cfg, train_ds, val_ds)
    grammar = load_instance_aog(args.grammar)
    if not isinstance(grammar, InstanceAOG):
        raise TypeError(f"Expected an InstanceAOG saved by build_instance_aog.py, got {type(grammar).__name__}")
    stage1 = PartCATHKGStage1(grammar.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    model = InstanceAOGStage2Classifier(stage1, grammar, cfg.model.stage2)
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)
    train_instance_aog_stage2(model, stage2_train, stage2_val, cfg, device=device)


if __name__ == "__main__":
    main()
