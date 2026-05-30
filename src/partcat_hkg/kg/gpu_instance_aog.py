from __future__ import annotations

from dataclasses import dataclass

import torch

from partcat_hkg.data.schema import RoleSchema
from .gpu_instance_components import GEOM_DIM, GEOM_FEATURE_NAMES, RELATION_DIM
from .relations import RELATION_FEATURE_NAMES


@dataclass
class GPUInstanceAOG:
    """Fixed-budget, GPU-friendly instance-slot And-Or grammar.

    Shapes:
      C: number of object classes
      A: templates per class
      S: max template-local slots
      D: token dimension
      G: geometry dimension (6)
      R: relation dimension (14)
    """

    schema: RoleSchema
    num_templates: int
    max_slots: int
    token_dim: int
    slots_per_part: int
    template_prior: torch.Tensor       # [C,A]
    template_valid: torch.Tensor       # [C,A]
    slot_valid: torch.Tensor           # [C,A,S]
    slot_part: torch.Tensor            # [C,A,S], -1 for padding
    slot_family: torch.Tensor          # [C,A,S], diagnostic
    slot_required: torch.Tensor        # [C,A,S]
    slot_presence_prior: torch.Tensor  # [C,A,S]
    slot_proto: torch.Tensor           # [C,A,S,D]
    slot_geom_mean: torch.Tensor       # [C,A,S,G]
    slot_geom_var: torch.Tensor        # [C,A,S,G]
    edges: torch.Tensor                # [E,4] = [class, template, slot_i, slot_j]
    edge_rel_mean: torch.Tensor        # [E,R]
    edge_rel_var: torch.Tensor         # [E,R]
    edge_support: torch.Tensor         # [E]
    edge_type_names: list[str]
    family_names: list[str]

    @property
    def geom_dim(self) -> int:
        return GEOM_DIM

    @property
    def relation_dim(self) -> int:
        return RELATION_DIM

    def to_payload(self) -> dict:
        return {
            "kind": "gpu_instance_aog",
            "schema": self.schema.to_payload(),
            "num_templates": int(self.num_templates),
            "max_slots": int(self.max_slots),
            "token_dim": int(self.token_dim),
            "slots_per_part": int(self.slots_per_part),
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
    def from_payload(cls, payload: dict) -> "GPUInstanceAOG":
        payload = dict(payload)
        payload.pop("kind", None)
        payload.pop("geom_feature_names", None)
        payload.pop("relation_feature_names", None)
        payload["schema"] = RoleSchema.from_payload(payload["schema"])
        return cls(**payload)


def empty_gpu_instance_aog(
    schema: RoleSchema,
    token_dim: int,
    *,
    num_templates: int = 1,
    slots_per_part: int = 1,
    max_slots: int | None = None,
) -> GPUInstanceAOG:
    cnum = schema.num_classes
    anum = max(1, int(num_templates))
    spp = max(1, int(slots_per_part))
    snum = int(max_slots or (schema.num_parts * spp))
    slot_part = torch.full((cnum, anum, snum), -1, dtype=torch.long)
    slot_valid = torch.zeros(cnum, anum, snum)
    for c in range(cnum):
        for a in range(anum):
            for s in range(min(snum, schema.num_parts * spp)):
                slot_part[c, a, s] = s // spp
                slot_valid[c, a, s] = 1.0
    return GPUInstanceAOG(
        schema=schema,
        num_templates=anum,
        max_slots=snum,
        token_dim=int(token_dim),
        slots_per_part=spp,
        template_prior=torch.full((cnum, anum), 1.0 / float(anum)),
        template_valid=torch.ones(cnum, anum),
        slot_valid=slot_valid,
        slot_part=slot_part,
        slot_family=slot_part.clone(),
        slot_required=torch.zeros(cnum, anum, snum),
        slot_presence_prior=slot_valid.clone(),
        slot_proto=torch.zeros(cnum, anum, snum, int(token_dim)),
        slot_geom_mean=torch.zeros(cnum, anum, snum, GEOM_DIM),
        slot_geom_var=torch.ones(cnum, anum, snum, GEOM_DIM),
        edges=torch.zeros(0, 4, dtype=torch.long),
        edge_rel_mean=torch.zeros(0, RELATION_DIM),
        edge_rel_var=torch.ones(0, RELATION_DIM),
        edge_support=torch.zeros(0),
        edge_type_names=[],
        family_names=[],
    )


def save_gpu_instance_aog(grammar: GPUInstanceAOG, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grammar.to_payload(), p)


def load_gpu_instance_aog(path) -> GPUInstanceAOG:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, GPUInstanceAOG):
        return payload
    if not isinstance(payload, dict) or payload.get("kind") != "gpu_instance_aog":
        raise TypeError(f"Expected a gpu_instance_aog payload, got {type(payload).__name__}")
    return GPUInstanceAOG.from_payload(payload)
