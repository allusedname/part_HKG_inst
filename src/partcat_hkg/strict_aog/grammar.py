from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from partcat_hkg.data.schema import RoleSchema
except Exception:  # pragma: no cover - lets unit tests use a tiny local schema if repo is absent
    RoleSchema = Any  # type: ignore


GEOM_FEATURE_NAMES = ["cx", "cy", "w", "h", "area", "score"]
REL_FEATURE_NAMES = [
    "dx", "dy", "dist", "area_i", "area_j", "log_area_ratio",
    "w_i", "h_i", "w_j", "h_j",
]


@dataclass
class StrictAOGGrammar:
    """A strict Spatial And-Or Graph grammar over neural part terminals.

    This dataclass is intentionally close to Zhu/Wu's S-AOG tuple
    ``<S, VN, VT, R, P>``:

    * ``S`` is represented by class root Or-nodes.  The start symbol selects one
      object class, then the selected class Or-node selects one And-production
      (template/view/subtype branch).
    * ``VN`` is implicit in the dense tensors: class Or-nodes and template
      And-nodes.  ``slot_*`` tensors define each And-production's child Or/terminal
      slots.
    * ``VT`` is not a fixed image dictionary stored in the grammar; at runtime it
      is instantiated by Stage-1 part segmentation proposals.  Each terminal has
      attributes: part type, mask/geometry, appearance token, and confidence.
    * ``R`` is the set of horizontal slot-slot relation factors ``edges``.
    * ``P`` is represented by log-probabilities and energy parameters:
      class/template rule probabilities, slot presence probabilities, singleton
      appearance/geometry models, and relation Gaussian potentials.

    The grammar stores template-local slots, not global names such as wheel_1.
    A parse graph selects a class, a template branch, and address variables that
    bind observed terminals to these slots.
    """

    schema: RoleSchema
    token_dim: int
    num_classes: int
    num_templates: int
    max_slots: int

    # Or-node / production probabilities.
    class_prior: torch.Tensor          # [C]
    template_prior: torch.Tensor       # [C,A]
    template_valid: torch.Tensor       # [C,A]

    # And-node child slots.
    slot_valid: torch.Tensor           # [C,A,S]
    slot_part: torch.Tensor            # [C,A,S], -1 for padding
    slot_required: torch.Tensor        # [C,A,S]
    slot_presence: torch.Tensor        # [C,A,S]
    slot_proto: torch.Tensor           # [C,A,S,D]
    slot_geom_mean: torch.Tensor       # [C,A,S,G]
    slot_geom_var: torch.Tensor        # [C,A,S,G]

    # Horizontal relation factors. Each edge belongs to exactly one And-production.
    edges: torch.Tensor                # [E,4] = [class, template, slot_i, slot_j]
    edge_type: torch.Tensor            # [E], 0 anchor, 1 repeated-part, 2 generic
    edge_support: torch.Tensor         # [E]
    edge_rel_mean: torch.Tensor        # [E,R]
    edge_rel_var: torch.Tensor         # [E,R]

    part_names: list[str]
    class_names: list[str]

    @property
    def geom_dim(self) -> int:
        return len(GEOM_FEATURE_NAMES)

    @property
    def rel_dim(self) -> int:
        return len(REL_FEATURE_NAMES)

    def to_payload(self) -> dict[str, Any]:
        schema_payload = self.schema.to_payload() if hasattr(self.schema, "to_payload") else None
        return {
            "kind": "strict_aog",
            "schema": schema_payload,
            "token_dim": self.token_dim,
            "num_classes": self.num_classes,
            "num_templates": self.num_templates,
            "max_slots": self.max_slots,
            "class_prior": self.class_prior.cpu(),
            "template_prior": self.template_prior.cpu(),
            "template_valid": self.template_valid.cpu(),
            "slot_valid": self.slot_valid.cpu(),
            "slot_part": self.slot_part.cpu(),
            "slot_required": self.slot_required.cpu(),
            "slot_presence": self.slot_presence.cpu(),
            "slot_proto": self.slot_proto.cpu(),
            "slot_geom_mean": self.slot_geom_mean.cpu(),
            "slot_geom_var": self.slot_geom_var.cpu(),
            "edges": self.edges.cpu(),
            "edge_type": self.edge_type.cpu(),
            "edge_support": self.edge_support.cpu(),
            "edge_rel_mean": self.edge_rel_mean.cpu(),
            "edge_rel_var": self.edge_rel_var.cpu(),
            "part_names": list(self.part_names),
            "class_names": list(self.class_names),
            "geom_feature_names": GEOM_FEATURE_NAMES,
            "rel_feature_names": REL_FEATURE_NAMES,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StrictAOGGrammar":
        payload = dict(payload)
        payload.pop("kind", None)
        payload.pop("geom_feature_names", None)
        payload.pop("rel_feature_names", None)
        schema_payload = payload.pop("schema", None)
        if schema_payload is not None and hasattr(RoleSchema, "from_payload"):
            payload["schema"] = RoleSchema.from_payload(schema_payload)
        else:
            payload["schema"] = schema_payload
        for k in [
            "class_prior", "template_prior", "template_valid", "slot_valid", "slot_part",
            "slot_required", "slot_presence", "slot_proto", "slot_geom_mean", "slot_geom_var",
            "edges", "edge_type", "edge_support", "edge_rel_mean", "edge_rel_var",
        ]:
            if k in payload and torch.is_tensor(payload[k]):
                payload[k] = payload[k].clone()
        return cls(**payload)


def save_strict_aog(grammar: StrictAOGGrammar, path: str) -> None:
    torch.save(grammar.to_payload(), path)


def load_strict_aog(path: str, *, map_location: str | torch.device = "cpu") -> StrictAOGGrammar:
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, StrictAOGGrammar):
        return payload
    if not isinstance(payload, dict) or payload.get("kind") != "strict_aog":
        raise ValueError(f"Expected a strict_aog payload at {path!r}")
    return StrictAOGGrammar.from_payload(payload)


def make_empty_strict_aog(schema: RoleSchema, token_dim: int, num_templates: int = 1) -> StrictAOGGrammar:
    c = int(getattr(schema, "num_classes", len(getattr(schema, "obj_names", []))))
    k = int(getattr(schema, "num_parts", len(getattr(schema, "part_names", []))))
    a = max(1, int(num_templates))
    s = max(1, k)
    slot_part = torch.full((c, a, s), -1, dtype=torch.long)
    slot_valid = torch.zeros(c, a, s)
    for ci in range(c):
        for ai in range(a):
            for si in range(min(s, k)):
                slot_part[ci, ai, si] = si
                slot_valid[ci, ai, si] = 1.0
    return StrictAOGGrammar(
        schema=schema,
        token_dim=int(token_dim),
        num_classes=c,
        num_templates=a,
        max_slots=s,
        class_prior=torch.full((c,), 1.0 / max(c, 1)),
        template_prior=torch.full((c, a), 1.0 / float(a)),
        template_valid=torch.ones(c, a),
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_required=slot_valid.clone(),
        slot_presence=slot_valid.clone(),
        slot_proto=torch.zeros(c, a, s, int(token_dim)),
        slot_geom_mean=torch.zeros(c, a, s, len(GEOM_FEATURE_NAMES)),
        slot_geom_var=torch.ones(c, a, s, len(GEOM_FEATURE_NAMES)),
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        edge_support=torch.zeros(0),
        edge_rel_mean=torch.zeros(0, len(REL_FEATURE_NAMES)),
        edge_rel_var=torch.ones(0, len(REL_FEATURE_NAMES)),
        part_names=list(getattr(schema, "part_names", [str(i) for i in range(k)])),
        class_names=list(getattr(schema, "obj_names", [str(i) for i in range(c)])),
    )
