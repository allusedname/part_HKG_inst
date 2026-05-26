#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.kg.builder import build_hkg
from partcat_hkg.kg.serialization import save_hkg
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    train_ds, val_ds = make_datasets(cfg)
    train_loader, _, _, _ = make_loaders(cfg, train_ds, val_ds)
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    kg = build_hkg(stage1, train_loader, train_ds.schema, cfg.model.hkg)
    out = Path(args.out or Path(cfg.paths.save_dir) / "checkpoints" / "hkg.pt")
    save_hkg(kg, out)
    print(f"saved HKG to {out}")


if __name__ == "__main__":
    main()
