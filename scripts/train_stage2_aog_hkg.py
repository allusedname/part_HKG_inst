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
from partcat_hkg.kg.datatypes import AOGHierarchicalKG
from partcat_hkg.kg.serialization import load_hkg
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.stage2.aog_hkg_classifier import AOGHKGStage2Classifier
from partcat_hkg.training.aog_stage2_trainer import train_aog_hkg_stage2
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 AOG-HKG classifier with a frozen Stage-1 checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--hkg", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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
    if args.no_amp:
        cfg.training.use_amp = False
    cfg.data.use_stage2_image_only_loader = True
    set_seed(cfg.seed)
    device = _resolve_device(args.device)

    train_ds, val_ds = make_datasets(cfg)
    _, _, stage2_train, stage2_val = make_loaders(cfg, train_ds, val_ds)
    kg = load_hkg(args.hkg)
    if not isinstance(kg, AOGHierarchicalKG):
        raise TypeError(f"Expected an AOGHierarchicalKG saved by build_aog_hkg.py, got {type(kg).__name__}")
    stage1 = PartCATHKGStage1(kg.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    model = AOGHKGStage2Classifier(stage1, kg, cfg.model.stage2)
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)
    train_aog_hkg_stage2(model, stage2_train, stage2_val, cfg, device=device)


if __name__ == "__main__":
    main()
