from __future__ import annotations

from dataclasses import dataclass

import torch

from partcat_hkg.data.schema import RoleSchema
from .instance_components import GEOM_DIM, GEOM_FEATURE_NAMES
from .relations import RELATION_FEATURE_NAMES


@dataclass
class InstanceAOG:
    """Instance-slot And-Or grammar for repeated part components.

    The grammar is class-level and template-local.  A class selects one of a
    small number of Or-branches.  Each template owns a variable number of latent
    slots.  Slots specify only a functional part type plus appearance/geometry
    statistics; observed connected components are assigned to slots at parse
    time.  ``slot_family`` links template-local slots that look relationally
    compatible across templates, but the family id is diagnostic rather than a
    hard global identity.
    """

    schema: RoleSchema
    num_templates: int
    max_slots: int
    token_dim: int
    template_prior: torch.Tensor        # [C,A]
    template_valid: torch.Tensor        # [C,A]
    slot_valid: torch.Tensor            # [C,A,S]
    slot_part: torch.Tensor             # [C,A,S], -1 for padding
    slot_family: torch.Tensor           # [C,A,S], -1 for padding
    slot_required: torch.Tensor         # [C,A,S]
    slot_presence_prior: torch.Tensor   # [C,A,S]
    slot_proto: torch.Tensor            # [C,A,S,D]
    slot_geom_mean: torch.Tensor        # [C,A,S,G]
    slot_geom_var: torch.Tensor         # [C,A,S,G]
    edges: torch.Tensor                 # [E,4] = [class, template, slot_i, slot_j]
    edge_rel_mean: torch.Tensor         # [E,R]
    edge_rel_var: torch.Tensor          # [E,R]
    edge_support: torch.Tensor          # [E]
    edge_type_names: list[str]
    family_names: list[str]

    @property
    def relation_dim(self) -> int:
        return len(RELATION_FEATURE_NAMES)

    @property
    def geom_dim(self) -> int:
        return GEOM_DIM

    def to_payload(self) -> dict:
        return {
            "kind": "instance_aog",
            "schema": self.schema.to_payload(),
            "num_templates": self.num_templates,
            "max_slots": self.max_slots,
            "token_dim": self.token_dim,
            "template_prior": self.template_prior,
            "template_valid": self.template_valid,
            "slot_valid": self.slot_valid,
            "slot_part": self.slot_part,
            "slot_family": self.slot_family,
            "slot_required": self.slot_required,
            "slot_presence_prior": self.slot_presence_prior,
            "slot_proto": self.slot_proto,
            "slot_geom_mean": self.slot_geom_mean,
            "slot_geom_var": self.slot_geom_var,
            "edges": self.edges,
            "edge_rel_mean": self.edge_rel_mean,
            "edge_rel_var": self.edge_rel_var,
            "edge_support": self.edge_support,
            "edge_type_names": self.edge_type_names,
            "family_names": self.family_names,
            "geom_feature_names": GEOM_FEATURE_NAMES,
            "relation_feature_names": RELATION_FEATURE_NAMES,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "InstanceAOG":
        payload = dict(payload)
        payload.pop("kind", None)
        payload.pop("geom_feature_names", None)
        payload.pop("relation_feature_names", None)
        payload["schema"] = RoleSchema.from_payload(payload["schema"])
        return cls(**payload)


def empty_instance_aog(
    schema: RoleSchema,
    token_dim: int,
    *,
    num_templates: int = 1,
    max_slots: int | None = None,
    relation_dim: int = len(RELATION_FEATURE_NAMES),
) -> InstanceAOG:
    cnum = schema.num_classes
    anum = max(1, int(num_templates))
    # One slot per functional part is a useful empty/smoke default.
    snum = int(max_slots or max(1, schema.num_parts))
    slot_part = torch.full((cnum, anum, snum), -1, dtype=torch.long)
    slot_valid = torch.zeros(cnum, anum, snum)
    for c in range(cnum):
        for a in range(anum):
            for s in range(min(snum, schema.num_parts)):
                slot_part[c, a, s] = s
                slot_valid[c, a, s] = 1.0
    return InstanceAOG(
        schema=schema,
        num_templates=anum,
        max_slots=snum,
        token_dim=int(token_dim),
        template_prior=torch.full((cnum, anum), 1.0 / float(anum)),
        template_valid=torch.ones(cnum, anum),
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_family=torch.full((cnum, anum, snum), -1, dtype=torch.long),
        slot_required=torch.zeros(cnum, anum, snum),
        slot_presence_prior=slot_valid.clone(),
        slot_proto=torch.zeros(cnum, anum, snum, int(token_dim)),
        slot_geom_mean=torch.zeros(cnum, anum, snum, GEOM_DIM),
        slot_geom_var=torch.ones(cnum, anum, snum, GEOM_DIM),
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_rel_mean=torch.zeros(0, relation_dim),
        edge_rel_var=torch.ones(0, relation_dim),
        edge_support=torch.zeros(0),
        edge_type_names=[],
        family_names=[],
    )
