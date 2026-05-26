from __future__ import annotations

import math
import torch

from partcat_hkg.config import ProjectConfig, Stage1Config
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.training import stage1_quality_upgrade as qmod
from partcat_hkg.training.stage1_quality_upgrade import (
    evaluate_stage1_quality_detailed,
    small_part_adaptive_weights,
)


def _schema() -> RoleSchema:
    return RoleSchema.from_names(
        ["car", "bird"],
        ["body", "wing"],
        ["car:body", "bird:body", "bird:wing"],
    )


class _FakeStage1(torch.nn.Module):
    def __init__(self, schema: RoleSchema):
        super().__init__()
        self.schema = schema

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = image.shape
        part_prob = image.new_zeros((b, self.schema.num_parts, h, w))
        # sample 0/body perfectly predicted; wing is empty-empty.
        part_prob[0, 0, 1:3, 1:3] = 1.0
        support = image.new_ones((b, 1, h, w))
        return {
            "part_prob": part_prob,
            "part_logits": torch.logit(part_prob.clamp(1e-4, 1 - 1e-4)),
            "support_prob": support,
            "support_logits": torch.logit(support.clamp(1e-4, 1 - 1e-4)),
            "part_presence": part_prob.flatten(2).amax(-1),
        }


def test_stage1_quality_iou_does_not_reward_empty_empty(monkeypatch):
    schema = _schema()
    model = _FakeStage1(schema)
    image = torch.zeros(2, 3, 4, 4)
    part_masks = torch.zeros(2, schema.num_parts, 4, 4)
    part_masks[0, 0, 1:3, 1:3] = 1.0
    presence = torch.zeros(2, schema.num_parts)
    presence[0, 0] = 1.0
    batch = {
        "image": image,
        "part_masks": part_masks,
        "role_masks": torch.zeros(2, schema.num_roles, 4, 4),
        "union_mask": part_masks.amax(dim=1, keepdim=True),
        "presence": presence,
        "obj_label": torch.tensor([0, 1]),
        "meta": [{}, {}],
    }

    def _fake_loss(out, batch, schema, cfg, *args, **kwargs):
        return image.new_tensor(0.0), {"loss": 0.0}

    monkeypatch.setattr(qmod, "stage1_quality_loss", _fake_loss)
    cfg = ProjectConfig()
    cfg.model.stage1.presence_threshold = 0.5
    report = evaluate_stage1_quality_detailed(model, [batch], cfg, device="cpu", mask_threshold=0.5)
    rows = {r["part"]: r for r in report["per_part"]}
    assert abs(rows["body"]["iou_present"] - 1.0) < 1e-5
    assert rows["body"]["present_count"] == 1.0
    assert math.isnan(rows["wing"]["iou_present"])
    assert rows["wing"]["present_count"] == 0.0
    assert abs(report["val_miou_present_parts"] - 1.0) < 1e-5


def test_small_part_adaptive_weights_boost_only_present_small_parts():
    target = torch.zeros(1, 3, 10, 10)
    target[0, 0, :1, :1] = 1.0  # small present part
    target[0, 1, :8, :8] = 1.0  # large present part
    presence = torch.tensor([[1.0, 1.0, 0.0]])
    w = small_part_adaptive_weights(target, presence, area_tau=0.05, max_weight=5.0, power=0.5)
    assert w.shape == (1, 3)
    assert w[0, 0] > w[0, 1]
    assert torch.isclose(w[0, 1], torch.tensor(1.0))
    assert torch.isclose(w[0, 2], torch.tensor(1.0))


def test_highres_refine_stage1_forward_shapes():
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
        use_highres_refine=True,
        highres_refine_dim=8,
        topk_presence_k=4,
    )
    model = PartCATHKGStage1(schema, cfg)
    out = model(torch.randn(1, 3, 32, 32))
    assert out["part_logits"].shape == (1, schema.num_parts, 32, 32)
    assert out["support_logits"].shape == (1, 1, 32, 32)
    assert torch.isfinite(out["part_logits"]).all()
