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
from partcat_hkg.data.gpu_component_cache import GPUComponentCacheDataset, collate_gpu_component_batch
from partcat_hkg.kg.gpu_instance_aog import load_gpu_instance_aog
from partcat_hkg.stage2.gpu_instance_aog_classifier import GPUInstanceAOGStage2Classifier
from partcat_hkg.training.gpu_instance_aog_trainer import train_gpu_instance_aog_stage2
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _loader(cache_dir: str, split: str, batch_size: int, num_workers: int, *, shuffle: bool):
    ds = GPUComponentCacheDataset(cache_dir, split=split, load_into_memory=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_gpu_component_batch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fully-GPU cached Stage-2 Instance-Slot AOG classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--grammar", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--assignment", default="softmax", choices=["max", "softmax", "sinkhorn"])
    parser.add_argument("--class-chunk", type=int, default=None)
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
    setattr(cfg.model.stage2, "isaog_assignment", args.assignment)
    if args.class_chunk is not None:
        setattr(cfg.model.stage2, "isaog_class_chunk", int(args.class_chunk))
    set_seed(cfg.seed)
    device = _resolve_device(args.device)
    grammar = load_gpu_instance_aog(args.grammar)
    train_loader = _loader(args.cache_dir, "train", cfg.training.batch_size_stage2, int(args.num_workers), shuffle=True)
    val_loader = _loader(args.cache_dir, "val", cfg.training.batch_size_stage2, int(args.num_workers), shuffle=False)
    model = GPUInstanceAOGStage2Classifier(grammar, cfg.model.stage2)
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)
    train_gpu_instance_aog_stage2(model, train_loader, val_loader, cfg, device=device)


if __name__ == "__main__":
    main()
