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
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.kg.gpu_instance_aog import save_gpu_instance_aog
from partcat_hkg.kg.gpu_instance_aog_builder import build_gpu_instance_aog_from_cache
from partcat_hkg.utils.seed import set_seed


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _load_schema_from_cache(cache_dir: Path, split: str) -> RoleSchema:
    meta_path = cache_dir / f"{split}_meta.pt"
    if not meta_path.exists():
        meta_path = cache_dir / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {split}_meta.pt or meta.pt in {cache_dir}")
    meta = torch.load(meta_path, map_location="cpu")
    if "schema" not in meta:
        raise KeyError(f"Cache meta {meta_path} does not contain schema")
    return RoleSchema.from_payload(meta["schema"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GPU-friendly Instance-Slot AOG grammar from cached Stage-1 components.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-templates-per-class", type=int, default=None)
    parser.add_argument("--max-components-per-part", type=int, default=None)
    parser.add_argument("--template-edge-max-edges", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Accepted for compatibility; cache generation should do sampling if needed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_templates_per_class is not None:
        cfg.model.hkg.num_templates_per_class = int(args.num_templates_per_class)
    if args.max_components_per_part is not None:
        setattr(cfg.model.hkg, "max_components_per_part", int(args.max_components_per_part))
    if args.template_edge_max_edges is not None:
        cfg.model.hkg.template_edge_max_edges = int(args.template_edge_max_edges)
    set_seed(cfg.seed)
    device = _resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    schema = _load_schema_from_cache(cache_dir, args.split)
    grammar = build_gpu_instance_aog_from_cache(str(cache_dir), schema, cfg.model.hkg, split=args.split, device=device)
    out = Path(args.out)
    save_gpu_instance_aog(grammar, out)
    print(f"saved GPU InstanceAOG to {out}")
    print(f"templates/class={grammar.num_templates} slots={grammar.max_slots} edges={grammar.edges.shape[0]}")


if __name__ == "__main__":
    main()
