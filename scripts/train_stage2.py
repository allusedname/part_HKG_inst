#!/usr/bin/env python
from __future__ import annotations

import argparse

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.kg.serialization import load_hkg
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.stage2.parse_scorer import VisibilityAwareParseGraphClassifier
from partcat_hkg.training.stage2_trainer import train_stage2
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--hkg", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    train_ds, val_ds = make_datasets(cfg)
    _, _, stage2_train, stage2_val = make_loaders(cfg, train_ds, val_ds)
    kg = load_hkg(args.hkg)
    stage1 = PartCATHKGStage1(kg.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    model = VisibilityAwareParseGraphClassifier(stage1, kg, cfg.model.stage2)
    train_stage2(model, stage2_train, stage2_val, cfg)


if __name__ == "__main__":
    main()
