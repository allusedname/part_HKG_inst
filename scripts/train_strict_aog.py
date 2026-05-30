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

from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.grammar import load_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser
from partcat_hkg.strict_aog.trainer import train_strict_aog, evaluate_strict_aog


def _device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if x.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return x


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate strict Spatial AOG calibration from cached terminals.")
    p.add_argument("--grammar", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--save-dir", default="runs/strict_aog")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--assignment", choices=["sinkhorn", "max"], default="sinkhorn")
    p.add_argument("--class-chunk", type=int, default=0)
    p.add_argument("--disable-edges", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    dev = torch.device(_device(args.device))
    train_ds = StrictAOGTerminalDataset(args.train_cache)
    val_ds = StrictAOGTerminalDataset(args.val_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_strict_aog, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_strict_aog, pin_memory=torch.cuda.is_available())
    grammar = load_strict_aog(args.grammar)
    pcfg = ParserConfig(assignment=args.assignment, class_chunk=int(args.class_chunk))
    model = StrictAOGParser(grammar, pcfg).to(dev)
    if args.eval_only:
        print(evaluate_strict_aog(model, val_loader, device=dev, enable_edges=not args.disable_edges))
        return
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_strict_aog(
        model,
        train_loader,
        val_loader,
        epochs=int(args.epochs),
        lr=float(args.lr),
        device=dev,
        save_dir=args.save_dir,
        enable_edges=not args.disable_edges,
    )


if __name__ == "__main__":
    main()
