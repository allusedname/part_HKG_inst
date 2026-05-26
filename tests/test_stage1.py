from __future__ import annotations

import torch

from partcat_hkg.config import Stage1Config, Stage1LossConfig
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.losses import binary_cross_entropy_prob, stage1_loss
from partcat_hkg.models.stage1 import PartCATHKGStage1

torch.set_num_threads(1)


def _schema():
    return RoleSchema.from_names(
        ["bird", "car"],
        ["body", "wheel", "wing"],
        ["bird:body", "bird:wing", "car:body", "car:wheel"],
    )


def test_stage1_forward_outputs_presence_and_tokens():
    schema = _schema()
    cfg = Stage1Config(
        backbone_name="tiny",
        model_dim=16,
        fuse_dim=16,
        token_dim=8,
        use_dino=False,
        use_clip_text=False,
        use_cost_aggregation=True,
        cost_embed_dim=4,
        cost_agg_heads=1,
        cost_agg_blocks=1,
        spatial_attention_max_tokens=64,
        topk_presence_k=4,
    )
    model = PartCATHKGStage1(schema, cfg)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out["part_logits"].shape == (1, schema.num_parts, 32, 32)
    assert out["role_logits"].shape == (1, schema.num_roles, 32, 32)
    assert out["support_logits"].shape == (1, 1, 32, 32)
    assert out["part_presence"].shape == (1, schema.num_parts)
    assert out["role_presence"].shape == (1, schema.num_roles)
    assert out["part_tokens"].shape == (1, schema.num_parts, cfg.token_dim)
    assert out["role_tokens"].shape == (1, schema.num_roles, cfg.token_dim)
    assert torch.isfinite(out["func_agg"]).all()


def test_stage1_loss_is_finite():
    schema = _schema()
    cfg = Stage1Config(
        backbone_name="tiny",
        model_dim=16,
        fuse_dim=16,
        token_dim=8,
        use_dino=False,
        use_clip_text=False,
        use_cost_aggregation=False,
        cost_embed_dim=4,
        cost_agg_heads=1,
        cost_agg_blocks=1,
        topk_presence_k=4,
    )
    model = PartCATHKGStage1(schema, cfg)
    image = torch.randn(1, 3, 32, 32)
    out = model(image)
    part_masks = torch.zeros(1, schema.num_parts, 32, 32)
    role_masks = torch.zeros(1, schema.num_roles, 32, 32)
    part_masks[0, 0, 6:20, 6:20] = 1
    part_masks[0, 2, 12:24, 18:28] = 1
    role_masks[0, schema.role_for(0, 0), 6:20, 6:20] = 1
    role_masks[0, schema.role_for(0, 2), 12:24, 18:28] = 1
    union = (part_masks.amax(dim=1, keepdim=True) > 0).float()
    batch = {
        "image": image,
        "part_masks": part_masks,
        "role_masks": role_masks,
        "union_mask": union,
        "obj_label": torch.tensor([0]),
    }
    loss, logs = stage1_loss(
        out,
        batch,
        schema,
        Stage1LossConfig(dino_affinity=0.0),
        part_loss_weight=torch.ones(schema.num_parts),
        part_pos_weight=torch.ones(schema.num_parts),
        role_loss_weight=torch.ones(schema.num_roles),
        topk_presence_k=4,
    )
    assert torch.isfinite(loss)
    assert logs["loss"] > 0


def test_probability_bce_matches_torch_bce_outside_amp():
    prob = torch.tensor([0.01, 0.20, 0.75, 0.99], dtype=torch.float32)
    target = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    expected = torch.nn.functional.binary_cross_entropy(prob, target)
    got = binary_cross_entropy_prob(prob, target)
    assert torch.allclose(got, expected, atol=1e-6)
