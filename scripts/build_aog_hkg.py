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
from partcat_hkg.kg.aog_builder import build_aog_hkg
from partcat_hkg.kg.serialization import save_hkg
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
    parser = argparse.ArgumentParser(description="Build an AOG-inspired Stage-2 HKG from a frozen Stage-1 checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None, help="HKG-build batch size; default uses Stage-1 batch size.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--num-templates-per-class", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    if args.max_images_per_class is not None:
        cfg.model.hkg.max_images_per_class = int(args.max_images_per_class)
    if args.num_templates_per_class is not None:
        cfg.model.hkg.num_templates_per_class = int(args.num_templates_per_class)
    set_seed(cfg.seed)
    device = _resolve_device(args.device)

    train_ds, _ = make_datasets(cfg)
    # HKG statistics should be built from a deterministic train split.  Stage 1
    # training uses random flips/color jitter, but grammar/template learning must
    # not: otherwise k-means alternatives and relation templates are estimated
    # from a random mixture of canonical and flipped layouts, which produces
    # unstable templates and messy motif overlays.
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
    kg = build_aog_hkg(stage1, loader, train_ds.schema, cfg.model.hkg, device=device)
    out = Path(args.out or Path(cfg.paths.save_dir) / "checkpoints" / "aog_hkg.pt")
    save_hkg(kg, out)
    print(f"saved AOG-HKG to {out}")
    print(f"templates/class={kg.num_templates} edges={kg.template_edges.shape[0]} motifs={kg.motif_edges.shape[0]}")


if __name__ == "__main__":
    main()
