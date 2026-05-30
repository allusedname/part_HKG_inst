from __future__ import annotations

import torch

from partcat_hkg.config import Stage2Config, Stage2LossConfig
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.kg.gpu_instance_aog import empty_gpu_instance_aog
from partcat_hkg.kg.gpu_instance_aog_builder import build_gpu_instance_aog_from_tensors
from partcat_hkg.kg.gpu_instance_components import GPUComponentConfig, extract_gpu_instance_components
from partcat_hkg.stage2.gpu_instance_aog_classifier import GPUInstanceAOGStage2Classifier
from partcat_hkg.stage2.losses import stage2_aog_hkg_loss


def _schema():
    return RoleSchema.from_names(
        ["bird", "car"],
        ["body", "wheel", "wing"],
        ["bird:body", "bird:wing", "car:body", "car:wheel"],
    )


def test_gpu_component_extractor_shapes_cpu_or_gpu():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    part_prob = torch.zeros(2, 3, 32, 32, device=dev)
    part_prob[:, 0, 8:20, 8:20] = 0.9
    part_prob[:, 1, 20:25, 5:10] = 0.8
    part_prob[:, 1, 20:25, 22:27] = 0.7
    part_tokens = torch.randn(2, 3, 8, device=dev)
    out = extract_gpu_instance_components(
        part_prob,
        part_tokens,
        cfg=GPUComponentConfig(mask_size=32, max_components_per_part=2, max_total_components=6, threshold=0.2),
    )
    assert out["component_valid"].shape == (2, 6)
    assert out["component_token"].shape == (2, 6, 8)
    assert out["component_geom"].shape == (2, 6, 6)
    assert out["component_valid"].any()


def test_gpu_instance_aog_forward_and_loss():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = _schema()
    grammar = empty_gpu_instance_aog(schema, token_dim=8, num_templates=2, slots_per_part=2)
    grammar.template_valid[:, 1] = 0
    grammar.template_prior[:, 0] = 1
    grammar.template_prior[:, 1] = 0
    grammar.slot_proto.normal_()
    grammar.slot_required[:, 0, :] = grammar.slot_valid[:, 0, :]
    grammar.slot_presence_prior[:, 0, :] = grammar.slot_valid[:, 0, :]
    model = GPUInstanceAOGStage2Classifier(grammar, Stage2Config(hidden_dim=16)).to(dev)
    batch = {
        "obj_label": torch.tensor([0, 1], device=dev),
        "part_presence": torch.rand(2, schema.num_parts, device=dev),
        "part_tokens": torch.randn(2, schema.num_parts, 8, device=dev),
        "component_valid": torch.ones(2, 6, dtype=torch.bool, device=dev),
        "component_part": torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 2, 2]], device=dev),
        "component_presence": torch.rand(2, 6, device=dev),
        "component_geom": torch.rand(2, 6, 6, device=dev),
        "component_token": torch.randn(2, 6, 8, device=dev),
    }
    out = model(batch, enable_edges=True)
    assert out["logits"].shape == (2, schema.num_classes)
    assert out["template_scores"].shape == (2, schema.num_classes, grammar.num_templates)
    loss, logs = stage2_aog_hkg_loss(out, batch["obj_label"], Stage2LossConfig(base_aux=0.1, hkg_aux=0.1, edge_aux=0.0, motif_aux=0.0))
    assert torch.isfinite(loss)
    assert logs["loss"] > 0


def test_gpu_instance_aog_builder_from_tensors():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = _schema()
    b, n, d = 8, 6, 8
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=dev)
    comp_part = torch.tensor([[0, 2, 2, 0, 1, 1]] * b, device=dev)
    comp_valid = torch.ones(b, n, dtype=torch.bool, device=dev)
    comp_geom = torch.rand(b, n, 6, device=dev)
    comp_geom[..., 4] = 0.05
    tensors = {
        "obj_label": labels,
        "part_presence": torch.rand(b, schema.num_parts, device=dev),
        "part_tokens": torch.randn(b, schema.num_parts, d, device=dev),
        "component_valid": comp_valid,
        "component_part": comp_part,
        "component_presence": torch.rand(b, n, device=dev),
        "component_geom": comp_geom,
        "component_token": torch.randn(b, n, d, device=dev),
    }

    class Cfg:
        num_templates_per_class = 2
        max_components_per_part = 2
        min_template_support = 1
        template_required_tau = 0.1
        template_edge_min_support = 0.1
        role_edge_min_count = 1
        template_edge_max_edges = 4

    grammar = build_gpu_instance_aog_from_tensors(schema, tensors, Cfg(), device=dev)
    assert grammar.template_prior.shape == (schema.num_classes, 2)
    assert grammar.slot_proto.shape[-1] == d
    assert grammar.max_slots == schema.num_parts * 2
