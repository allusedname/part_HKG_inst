#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from partcat_hkg.config import ProjectConfig, Stage1Config
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.losses import stage1_loss
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.utils.seed import set_seed


def _rect(mask: torch.Tensor, y0: int, y1: int, x0: int, x1: int) -> None:
    mask[..., y0:y1, x0:x1] = 1.0


def make_toy_batch(schema: RoleSchema, batch_size: int = 2, image_size: int = 96) -> dict:
    image = torch.randn(batch_size, 3, image_size, image_size)
    part_masks = torch.zeros(batch_size, schema.num_parts, image_size, image_size)
    role_masks = torch.zeros(batch_size, schema.num_roles, image_size, image_size)
    union = torch.zeros(batch_size, 1, image_size, image_size)
    labels = torch.empty(batch_size, dtype=torch.long)

    for b in range(batch_size):
        if b % 2 == 0:
            labels[b] = schema.obj_to_idx["bird"]
            body = schema.part_to_idx["body"]
            wing = schema.part_to_idx["wing"]
            _rect(part_masks[b, body], int(.30 * image_size), int(.70 * image_size), int(.36 * image_size), int(.66 * image_size))
            _rect(part_masks[b, wing], int(.22 * image_size), int(.55 * image_size), int(.15 * image_size), int(.40 * image_size))
            _rect(role_masks[b, schema.role_to_idx["bird:body"]], int(.30 * image_size), int(.70 * image_size), int(.36 * image_size), int(.66 * image_size))
            _rect(role_masks[b, schema.role_to_idx["bird:wing"]], int(.22 * image_size), int(.55 * image_size), int(.15 * image_size), int(.40 * image_size))
        else:
            labels[b] = schema.obj_to_idx["car"]
            body = schema.part_to_idx["body"]
            wheel = schema.part_to_idx["wheel"]
            _rect(part_masks[b, body], int(.38 * image_size), int(.68 * image_size), int(.20 * image_size), int(.82 * image_size))
            _rect(part_masks[b, wheel], int(.62 * image_size), int(.86 * image_size), int(.24 * image_size), int(.46 * image_size))
            _rect(role_masks[b, schema.role_to_idx["car:body"]], int(.38 * image_size), int(.68 * image_size), int(.20 * image_size), int(.82 * image_size))
            _rect(role_masks[b, schema.role_to_idx["car:wheel"]], int(.62 * image_size), int(.86 * image_size), int(.24 * image_size), int(.46 * image_size))
    union[:, 0] = part_masks.amax(dim=1)
    presence = (part_masks.flatten(2).amax(dim=-1) > 0).float()
    role_presence = (role_masks.flatten(2).amax(dim=-1) > 0).float()
    return {
        "image": image,
        "image_raw": image.clamp(0, 1),
        "obj_label": labels,
        "part_masks": part_masks,
        "role_masks": role_masks,
        "union_mask": union,
        "presence": presence,
        "role_presence": role_presence,
        "meta": [{"toy": True} for _ in range(batch_size)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Stage-1 forward/loss smoke test.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--with-cost-aggregation", action="store_true", help="Enable the attention aggregation path in the smoke model.")
    parser.add_argument("--backward", action="store_true", help="Also run backward; slower on CPU.")
    args = parser.parse_args()
    set_seed(7)
    schema = RoleSchema.from_names(
        obj_names=["bird", "car"],
        part_names=["body", "wheel", "wing"],
        role_names=["bird:body", "bird:wing", "car:body", "car:wheel"],
    )
    s1 = Stage1Config(
        backbone_name="tiny",
        model_dim=16,
        fuse_dim=12,
        token_dim=8,
        cost_embed_dim=4,
        cost_agg_heads=1,
        use_dino=False,
        use_clip_text=False,
        use_cost_aggregation=bool(args.with_cost_aggregation),
        use_spatial_aggregation=bool(args.with_cost_aggregation),
        use_part_aggregation=bool(args.with_cost_aggregation),
        emit_role_tokens=True,
        topk_presence_k=4,
    )
    cfg = ProjectConfig()
    cfg.model.stage1 = s1
    cfg.loss.stage1.dino_affinity = 0.0
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = PartCATHKGStage1(schema, s1).to(device)
    batch = make_toy_batch(schema, batch_size=args.batch_size, image_size=args.image_size)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch["image"])
    loss, logs = stage1_loss(out, batch, schema, cfg.loss.stage1, topk_presence_k=s1.topk_presence_k)
    if args.backward:
        loss.backward()
    shape_keys = ["support_logits", "part_logits", "role_logits", "part_presence", "part_tokens", "role_tokens"]
    print("Stage1 synthetic smoke passed")
    print("status:", model.status)
    print("shapes:", {k: tuple(out[k].shape) for k in shape_keys})
    print("loss:", float(loss.detach().cpu()), "backward:", bool(args.backward))
    print("logs:", {k: round(v, 5) for k, v in logs.items()})


if __name__ == "__main__":
    main()
