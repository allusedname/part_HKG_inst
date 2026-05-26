#!/usr/bin/env python
from __future__ import annotations

import argparse
import torch

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets, make_loaders
from partcat_hkg.kg.serialization import load_hkg
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.stage2.parse_scorer import VisibilityAwareParseGraphClassifier
from partcat_hkg.evaluation.metrics import top1_accuracy, macro_accuracy
from partcat_hkg.utils.io import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--hkg", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_ds, val_ds = make_datasets(cfg)
    _, _, _, val_loader = make_loaders(cfg, train_ds, val_ds)
    kg = load_hkg(args.hkg)
    stage1 = PartCATHKGStage1(kg.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    model = VisibilityAwareParseGraphClassifier(stage1, kg, cfg.model.stage2)
    load_checkpoint(args.stage2_ckpt, model, strict=True)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch, detach_stage1=True)
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["obj_label"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    print({"top1": top1_accuracy(logits, labels), "macro": macro_accuracy(logits, labels, kg.schema.num_classes)})


if __name__ == "__main__":
    main()
