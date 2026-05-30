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
from partcat_hkg.data.gpu_component_cache import save_cache_meta
from partcat_hkg.kg.gpu_instance_components import GPUComponentConfig, average_stage1_token_maps, extract_gpu_instance_components, save_component_cache_shard
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.utils.io import load_checkpoint
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _make_loader(ds, cfg, batch_size: int):
    common = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_part_batch,
    )
    if cfg.data.num_workers > 0:
        common.update(persistent_workers=cfg.data.persistent_workers, prefetch_factor=cfg.data.prefetch_factor)
    return DataLoader(ds, **common)


@torch.no_grad()
def cache_split(stage1, loader, split: str, out_dir: Path, cfg_comp: GPUComponentConfig, device: str, schema_payload: dict) -> int:
    stage1.to(device)
    stage1.eval()
    count = 0
    shard_id = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["obj_label"].to(device, non_blocking=True).long()
        out = stage1(image)
        part_prob = out.get("part_prob", torch.sigmoid(out["part_logits"])).detach().float()
        part_presence = out.get("part_presence")
        if torch.is_tensor(part_presence):
            part_presence = part_presence.detach().float()
        else:
            part_presence = part_prob.flatten(2).topk(k=min(64, part_prob.shape[-1] * part_prob.shape[-2]), dim=-1).values.mean(-1)
        part_tokens = out.get("part_tokens", out.get("part_tokens_res"))
        if part_tokens is None:
            raise KeyError("Stage1 output must contain part_tokens or part_tokens_res")
        part_tokens = part_tokens.detach().float()
        token_map = average_stage1_token_maps(out)
        comp = extract_gpu_instance_components(
            part_prob,
            part_tokens,
            part_presence=part_presence,
            token_map=token_map,
            cfg=cfg_comp,
            return_masks=cfg_comp.keep_component_masks,
        )
        bsz = int(label.shape[0])
        payload = {
            "image_index": torch.arange(count, count + bsz, device=label.device, dtype=torch.long),
            "obj_label": label,
            "part_presence": part_presence,
            "part_tokens": part_tokens,
            **comp,
        }
        path = out_dir / f"{split}_shard_{shard_id:05d}.pt"
        save_component_cache_shard(path, payload)
        count += bsz
        shard_id += 1
        if shard_id % 25 == 0:
            print(f"[cache-gpu-components] split={split} shards={shard_id} samples={count}")
    meta = {
        "split": split,
        "num_samples": count,
        "schema": schema_payload,
        "component_config": cfg_comp.__dict__,
    }
    save_cache_meta(out_dir, split=split, meta=meta)
    print(f"[cache-gpu-components] saved split={split} samples={count} to {out_dir}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen Stage-1 outputs and GPU-extracted instance components for fast IS-AOG Stage 2.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", default="train,val", help="Comma-separated split names: train,val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mask-size", type=int, default=64)
    parser.add_argument("--component-threshold", type=float, default=0.20)
    parser.add_argument("--max-components-per-part", type=int, default=2)
    parser.add_argument("--max-total-components", type=int, default=32)
    parser.add_argument("--gaussian-sigma", type=float, default=0.075)
    parser.add_argument("--local-max-kernel", type=int, default=7)
    parser.add_argument("--keep-component-masks", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    set_seed(cfg.seed)
    device = _resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = make_datasets(cfg)
    for ds in (train_ds, val_ds):
        if hasattr(ds, "transform") and hasattr(ds.transform, "train"):
            ds.transform.train = False
    stage1 = PartCATHKGStage1(train_ds.schema, cfg.model.stage1)
    load_checkpoint(args.stage1_ckpt, stage1, strict=True)
    comp_cfg = GPUComponentConfig(
        mask_size=int(args.mask_size),
        threshold=float(args.component_threshold),
        local_max_kernel=int(args.local_max_kernel),
        gaussian_sigma=float(args.gaussian_sigma),
        max_components_per_part=int(args.max_components_per_part),
        max_total_components=int(args.max_total_components),
        keep_component_masks=bool(args.keep_component_masks),
    )
    batch_size = int(args.batch_size or cfg.training.batch_size_stage1)
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    if "train" in splits:
        cache_split(stage1, _make_loader(train_ds, cfg, batch_size), "train", out_dir, comp_cfg, device, train_ds.schema.to_payload())
    if "val" in splits:
        cache_split(stage1, _make_loader(val_ds, cfg, batch_size), "val", out_dir, comp_cfg, device, train_ds.schema.to_payload())


if __name__ == "__main__":
    main()
