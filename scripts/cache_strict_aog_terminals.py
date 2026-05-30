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
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.strict_aog.terminals import TerminalExtractionConfig, batch_extract_terminals, save_terminal_cache
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _device(x: str) -> str:
    if x == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if x.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return x


@torch.no_grad()
def _cache_split(stage1, dataset, out_path: Path, args, schema_payload):
    common = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_part_batch,
    )
    loader = DataLoader(dataset, **common)
    records = []
    cfg = TerminalExtractionConfig(
        threshold=float(args.threshold),
        min_area_frac=float(args.min_area_frac),
        min_presence=float(args.min_presence),
        max_components_per_part=int(args.max_components_per_part),
        max_terminals=int(args.max_terminals),
        mask_size=int(args.mask_size),
    )
    dev = next(stage1.parameters()).device
    stage1.eval()
    seen = 0
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= int(args.max_batches):
            break
        images = batch["image"].to(dev, non_blocking=True)
        out = stage1(images)
        terms = batch_extract_terminals(out, cfg=cfg)
        B = images.shape[0]
        for b in range(B):
            rec = {k: v[b].detach().cpu() for k, v in terms.items()}
            rec["obj_label"] = int(batch["obj_label"][b].detach().cpu().item())
            records.append(rec)
        seen += B
        if bi % 20 == 0:
            print(f"[cache-strict-aog] {out_path.name} batches={bi} images={seen}")
    save_terminal_cache(records, out_path, schema_payload=schema_payload)
    print(f"saved {len(records)} terminal records to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cache Stage-1 terminal proposals for strict Spatial AOG parsing.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--min-area-frac", type=float, default=1e-4)
    p.add_argument("--min-presence", type=float, default=0.05)
    p.add_argument("--max-components-per-part", type=int, default=4)
    p.add_argument("--max-terminals", type=int, default=32)
    p.add_argument("--mask-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.data.num_workers = int(args.num_workers)
    set_seed(cfg.seed)
    dev = torch.device(_device(args.device))
    train_ds, val_ds = make_datasets(cfg)
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1).to(dev)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_payload = train_ds.schema.to_payload()
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "train" in splits:
        _cache_split(stage1, train_ds, out_dir / "train_strict_aog_terminals.pt", args, schema_payload)
    if "val" in splits:
        _cache_split(stage1, val_ds, out_dir / "val_strict_aog_terminals.pt", args, schema_payload)


if __name__ == "__main__":
    main()
