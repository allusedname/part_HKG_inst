from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.kg.instance_aog import empty_instance_aog, InstanceAOG
from partcat_hkg.kg.instance_aog_builder import build_instance_aog
from partcat_hkg.kg.instance_components import extract_instance_components
from partcat_hkg.kg.instance_aog_io import load_instance_aog, save_instance_aog
from partcat_hkg.stage2.instance_aog_classifier import InstanceAOGStage2Classifier


def _schema() -> RoleSchema:
    return RoleSchema.from_names(
        ["car", "bird"],
        ["body", "wheel", "wing"],
        ["car:body", "car:wheel", "bird:body", "bird:wing"],
    )


class FakeStage1(nn.Module):
    def __init__(self, num_parts: int, token_dim: int = 8):
        super().__init__()
        self.cfg = SimpleNamespace(token_dim=token_dim, presence_threshold=0.15)
        self.num_parts = num_parts
        self.token_dim = token_dim
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = image.shape
        part_prob = image.new_zeros(b, self.num_parts, h, w)
        part_prob[:, 0, 6:26, 8:24] = 0.95  # body
        part_prob[:, 1, 23:30, 7:13] = 0.90  # wheel component 1
        part_prob[:, 1, 23:30, 20:27] = 0.88  # wheel component 2
        if self.num_parts > 2:
            part_prob[:, 2, 5:14, 20:28] = 0.70
        token_res_map = torch.randn(b, self.token_dim, max(1, h // 4), max(1, w // 4), device=image.device)
        token_dino_map = torch.randn_like(token_res_map)
        part_tokens = torch.randn(b, self.num_parts, self.token_dim, device=image.device)
        return {
            "part_prob": part_prob,
            "part_logits": torch.logit(part_prob.clamp(1e-4, 1 - 1e-4)),
            "part_presence": part_prob.flatten(2).amax(-1),
            "part_tokens": part_tokens,
            "part_tokens_res": part_tokens,
            "part_tokens_dino": part_tokens,
            "token_res_map": token_res_map,
            "token_dino_map": token_dino_map,
        }


def test_extract_repeated_components_splits_same_part() -> None:
    prob = torch.zeros(3, 32, 32)
    prob[1, 20:27, 5:12] = 0.9
    prob[1, 20:27, 22:29] = 0.9
    comps = extract_instance_components(prob, threshold=0.4, max_components_per_part=4, min_area_frac=1e-4)
    assert comps["part_type"].shape[0] == 2
    assert torch.equal(comps["part_type"], torch.tensor([1, 1]))


def test_instance_aog_serialization_roundtrip(tmp_path) -> None:
    schema = _schema()
    grammar = empty_instance_aog(schema, token_dim=8, num_templates=2, max_slots=3)
    path = tmp_path / "instance_aog.pt"
    save_instance_aog(grammar, path)
    loaded = load_instance_aog(path)
    assert isinstance(loaded, InstanceAOG)
    assert loaded.slot_part.shape == grammar.slot_part.shape
    assert loaded.schema.num_parts == schema.num_parts


def test_instance_aog_classifier_forward_smoke() -> None:
    schema = _schema()
    grammar = empty_instance_aog(schema, token_dim=8, num_templates=2, max_slots=3)
    # Make body and wheel expected, wing optional.
    grammar.slot_required[:, :, 0] = 1.0
    grammar.slot_presence_prior[:, :, :] = 0.8
    stage1 = FakeStage1(schema.num_parts, token_dim=8)
    model = InstanceAOGStage2Classifier(stage1, grammar, SimpleNamespace(hidden_dim=16, component_threshold=0.4))
    batch = {"image": torch.randn(2, 3, 32, 32), "obj_label": torch.tensor([0, 1])}
    out = model(batch, return_parse=True)
    assert out["logits"].shape == (2, schema.num_classes)
    assert out["base_logits"].shape == (2, schema.num_classes)
    assert out["hkg_logits"].shape == (2, schema.num_classes)
    assert isinstance(out["parse_graph"], list)


def test_build_instance_aog_smoke() -> None:
    schema = _schema()
    stage1 = FakeStage1(schema.num_parts, token_dim=8)
    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "obj_label": torch.tensor([0, 1]),
        "part_masks": torch.zeros(2, schema.num_parts, 32, 32),
        "presence": torch.ones(2, schema.num_parts),
    }
    cfg = SimpleNamespace(
        num_templates_per_class=1,
        min_template_support=1,
        role_edge_min_count=1,
        template_edge_min_support=0.1,
        template_edge_max_edges=4,
        max_components_per_part=3,
        max_total_components=8,
        component_threshold=0.4,
        max_images_per_class=0,
        use_predicted_stage1_evidence=True,
    )
    grammar = build_instance_aog(stage1, [batch], schema, cfg, device="cpu")
    assert isinstance(grammar, InstanceAOG)
    assert grammar.slot_valid.sum() > 0
    assert grammar.max_slots >= 1
