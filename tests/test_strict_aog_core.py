from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.builder import StrictAOGBuildConfig, build_strict_aog_from_records
from partcat_hkg.strict_aog.grammar import StrictAOGGrammar, load_strict_aog, save_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig, StrictAOGParser, strict_aog_loss
from partcat_hkg.strict_aog.terminals import terminal_pair_relations


class TinySchema:
    obj_names = ["bike", "bird"]
    part_names = ["body", "wheel", "wing"]
    num_classes = 2
    num_parts = 3

    def to_payload(self):
        return {"obj_names": self.obj_names, "part_names": self.part_names, "role_names": [], "role_to_obj": torch.zeros(0, dtype=torch.long), "role_to_part": torch.zeros(0, dtype=torch.long), "role_index_table": torch.full((2,3), -1, dtype=torch.long)}


# Monkeypatch RoleSchema fallback is not needed when repo schema exists. This test
# passes TinySchema directly to runtime code that only uses names/counts.

def _record(label: int, parts: list[int], xs: list[float], token_dim: int = 4) -> dict:
    nmax = 4
    valid = torch.zeros(nmax, dtype=torch.bool)
    part = torch.full((nmax,), -1, dtype=torch.long)
    score = torch.zeros(nmax)
    geom = torch.zeros(nmax, 6)
    token = torch.zeros(nmax, token_dim)
    for i, (p, x) in enumerate(zip(parts, xs)):
        valid[i] = True
        part[i] = p
        score[i] = 0.95
        geom[i] = torch.tensor([x, 0.6 if p == 1 else 0.4, 0.2, 0.2, 0.05 if p != 0 else 0.3, 0.95])
        token[i, p % token_dim] = 1.0
        if p == 1:
            token[i, 3] = x
    return {"obj_label": int(label), "terminal_valid": valid, "terminal_part": part, "terminal_score": score, "terminal_geom": geom, "terminal_token": token, "terminal_mask": torch.zeros(nmax, 16, 16)}


def test_pair_relations_shape_and_finite():
    geom = torch.rand(2, 5, 6)
    rel = terminal_pair_relations(geom)
    assert rel.shape == (2, 5, 5, 10)
    assert torch.isfinite(rel).all()


def test_strict_aog_build_and_forward_separates_classes():
    schema = TinySchema()
    records = []
    for _ in range(6):
        records.append(_record(0, [0, 1, 1], [0.5, 0.25, 0.75]))  # bike: body + two wheels
        records.append(_record(1, [0, 2, 2], [0.5, 0.25, 0.75]))  # bird: body + two wings
    grammar = build_strict_aog_from_records(
        records,
        schema=schema,
        token_dim=4,
        num_parts=3,
        cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1, min_edge_support=0.1),
    )
    assert grammar.slot_valid.sum() > 0
    model = StrictAOGParser(grammar, ParserConfig(assignment="sinkhorn", sinkhorn_iters=8))
    batch = {k: torch.stack([records[0][k], records[1][k]], 0) for k in records[0] if k.startswith("terminal_")}
    labels = torch.tensor([0, 1])
    out = model(batch, enable_edges=True, return_parse=True)
    assert out["logits"].shape == (2, 2)
    assert torch.isfinite(out["logits"]).all()
    loss, logs = strict_aog_loss(out, labels)
    assert torch.isfinite(loss)
    assert logs["logit_std"] > 0.0
    assert isinstance(out["parse_graph"], list)


def test_save_load_roundtrip(tmp_path):
    schema = TinySchema()
    records = [_record(0, [0, 1, 1], [0.5, 0.25, 0.75]), _record(1, [0, 2, 2], [0.5, 0.25, 0.75])]
    grammar = build_strict_aog_from_records(records, schema=schema, token_dim=4, num_parts=3, cfg=StrictAOGBuildConfig(num_templates_per_class=1, min_template_support=1, min_edge_count=1))
    path = tmp_path / "g.pt"
    save_strict_aog(grammar, str(path))
    loaded = load_strict_aog(str(path))
    assert loaded.num_classes == grammar.num_classes
    assert loaded.slot_valid.shape == grammar.slot_valid.shape
