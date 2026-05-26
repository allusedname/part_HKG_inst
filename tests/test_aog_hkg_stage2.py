from __future__ import annotations

import torch

from partcat_hkg.config import Stage1Config, Stage2Config, Stage2LossConfig
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.kg.datatypes import empty_aog_hkg
from partcat_hkg.models.stage1 import PartCATHKGStage1
from partcat_hkg.stage2.aog_hkg_classifier import AOGHKGStage2Classifier
from partcat_hkg.stage2.losses import stage2_aog_hkg_loss


def _schema():
    return RoleSchema.from_names(
        ["bird", "car"],
        ["body", "wheel", "wing"],
        ["bird:body", "bird:wing", "car:body", "car:wheel"],
    )


def test_aog_hkg_stage2_forward_and_loss():
    torch.set_num_threads(1)
    schema = _schema()
    s1cfg = Stage1Config(
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
    stage1 = PartCATHKGStage1(schema, s1cfg)
    kg = empty_aog_hkg(schema, token_dim=s1cfg.token_dim, num_templates=2)
    kg.template_role_prior[0, 0, [0, 2]] = 1.0
    kg.template_role_required[0, 0, [0, 2]] = 1.0
    kg.template_role_prior[1, 0, [0, 1]] = 1.0
    kg.template_role_required[1, 0, [0, 1]] = 1.0
    kg.template_valid[:, 1] = 0.0
    kg.template_prior[:, 0] = 1.0
    kg.template_prior[:, 1] = 0.0
    kg.template_role_proto_r.normal_()
    kg.template_role_proto_d.normal_()
    kg.func_proto_r.normal_()
    kg.func_proto_d.normal_()
    kg.class_role_proto_r.normal_()
    kg.class_role_proto_d.normal_()

    model = AOGHKGStage2Classifier(stage1, kg, Stage2Config(hidden_dim=16))
    batch = {"image": torch.randn(2, 3, 32, 32), "obj_label": torch.tensor([0, 1])}
    out = model(batch, detach_stage1=True, enable_edges=True)
    assert out["logits"].shape == (2, schema.num_classes)
    assert out["hkg_logits"].shape == (2, schema.num_classes)
    assert out["template_scores"].shape == (2, schema.num_classes, kg.num_templates)
    loss, logs = stage2_aog_hkg_loss(out, batch["obj_label"], Stage2LossConfig(base_aux=0.1))
    assert torch.isfinite(loss)
    assert logs["loss"] > 0
