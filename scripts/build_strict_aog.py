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
from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig, build_strict_aog_from_records, save_builder_output
from partcat_hkg.strict_aog.terminals import load_terminal_cache
from partcat_hkg.utils.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Build a strict Song-Chun-style Spatial AOG from cached part terminals.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--cache", required=True, help="train_strict_aog_terminals.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--num-templates-per-class", type=int, default=3)
    p.add_argument("--max-slots-per-template", type=int, default=12)
    p.add_argument("--max-slots-per-part", type=int, default=4)
    p.add_argument("--min-template-support", type=int, default=2)
    p.add_argument("--required-tau", type=float, default=0.45)
    p.add_argument("--min-slot-support", type=float, default=0.08)
    p.add_argument("--min-edge-support", type=float, default=0.12)
    p.add_argument("--min-edge-count", type=int, default=3)
    p.add_argument("--max-edges-per-template", type=int, default=12)
    args = p.parse_args()
    cfg0 = load_config(args.config)
    set_seed(cfg0.seed)
    payload = load_terminal_cache(args.cache, map_location="cpu")
    if payload.get("schema") is None:
        raise ValueError("Terminal cache does not contain a schema payload; rebuild cache with cache_strict_aog_terminals.py")
    schema = RoleSchema.from_payload(payload["schema"])
    records = payload["records"]
    token_dim = int(records[0]["terminal_token"].shape[-1])
    num_parts = schema.num_parts
    cfg = StrictAOGBuildConfig(
        num_templates_per_class=int(args.num_templates_per_class),
        max_slots_per_template=int(args.max_slots_per_template),
        max_slots_per_part=int(args.max_slots_per_part),
        min_template_support=int(args.min_template_support),
        required_tau=float(args.required_tau),
        min_slot_support=float(args.min_slot_support),
        min_edge_support=float(args.min_edge_support),
        min_edge_count=int(args.min_edge_count),
        max_edges_per_template=int(args.max_edges_per_template),
    )
    grammar = build_strict_aog_from_records(records, schema=schema, token_dim=token_dim, num_parts=num_parts, cfg=cfg)
    save_builder_output(grammar, args.out)
    print(f"saved strict AOG to {args.out}")
    print(f"classes={grammar.num_classes} templates={grammar.num_templates} slots={grammar.max_slots} edges={grammar.edges.shape[0]}")
    valid_templates = int(grammar.template_valid.sum().item())
    valid_slots = int(grammar.slot_valid.sum().item())
    print(f"valid_templates={valid_templates} valid_slots={valid_slots}")


if __name__ == "__main__":
    main()
