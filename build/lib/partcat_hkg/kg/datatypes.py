from __future__ import annotations

from dataclasses import dataclass
import torch

from partcat_hkg.data.schema import RoleSchema
from .relations import RELATION_FEATURE_NAMES


@dataclass
class HierarchicalKG:
    """Functional-role HKG tensors.

    The tensors preserve the v51 notebook's practical fields while using names
    aligned with the simplified proposal.
    """

    schema: RoleSchema
    pmi: torch.Tensor                  # [C,F], diagnostic only by default
    func_proto_r: torch.Tensor          # [F,D]
    func_proto_d: torch.Tensor          # [F,D]
    role_proto_r: torch.Tensor          # [C,F,D]
    role_proto_d: torch.Tensor          # [C,F,D]
    role_prior: torch.Tensor            # [C,F]
    func_edges: torch.Tensor            # [E_func,2]
    role_edges: torch.Tensor            # [E_role,3] = [class, part_i, part_j]
    role_rel_mean: torch.Tensor         # [E_role,R]
    role_rel_var: torch.Tensor          # [E_role,R]
    role_rel_support: torch.Tensor      # [E_role]
    role_rel_ig: torch.Tensor           # [E_role]
    role_rel_global_mean: torch.Tensor  # [E_role,R]
    role_rel_global_var: torch.Tensor   # [E_role,R]
    role_rel_type_names: list[str]

    @property
    def relation_dim(self) -> int:
        return len(RELATION_FEATURE_NAMES)

    def to_payload(self) -> dict:
        return {
            "schema": self.schema.to_payload(),
            "pmi": self.pmi,
            "func_proto_r": self.func_proto_r,
            "func_proto_d": self.func_proto_d,
            "role_proto_r": self.role_proto_r,
            "role_proto_d": self.role_proto_d,
            "role_prior": self.role_prior,
            "func_edges": self.func_edges,
            "role_edges": self.role_edges,
            "role_rel_mean": self.role_rel_mean,
            "role_rel_var": self.role_rel_var,
            "role_rel_support": self.role_rel_support,
            "role_rel_ig": self.role_rel_ig,
            "role_rel_global_mean": self.role_rel_global_mean,
            "role_rel_global_var": self.role_rel_global_var,
            "role_rel_type_names": self.role_rel_type_names,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "HierarchicalKG":
        payload = dict(payload)
        payload["schema"] = RoleSchema.from_payload(payload["schema"])
        payload.pop("kind", None)
        return cls(**payload)


MOTIF_TYPE_NAMES = ["generic", "attached", "containment", "lateral", "appendage"]


@dataclass
class AOGHierarchicalKG:
    """AOG-inspired hierarchical KG for Stage 2.

    The graph is a compact class-level grammar.  Each class has a small set of
    alternative templates (Or-node branches).  A template decomposes into
    class-conditioned role slots over the shared functional-part vocabulary and
    owns horizontal relation/motif edges.  At inference, Stage 2 selects or
    log-sums over templates and instantiates a parse graph from Stage-1 masks.
    """

    schema: RoleSchema
    num_templates: int
    pmi: torch.Tensor                       # [C,F]
    role_prior: torch.Tensor                # [C,F]
    func_proto_r: torch.Tensor              # [F,D]
    func_proto_d: torch.Tensor              # [F,D]
    class_role_proto_r: torch.Tensor        # [C,F,D]
    class_role_proto_d: torch.Tensor        # [C,F,D]
    template_prior: torch.Tensor            # [C,A]
    template_valid: torch.Tensor            # [C,A]
    template_role_prior: torch.Tensor       # [C,A,F]
    template_role_required: torch.Tensor    # [C,A,F]
    template_role_proto_r: torch.Tensor     # [C,A,F,D]
    template_role_proto_d: torch.Tensor     # [C,A,F,D]
    template_edges: torch.Tensor            # [E,4] = [class, template, part_i, part_j]
    template_rel_mean: torch.Tensor         # [E,R]
    template_rel_var: torch.Tensor          # [E,R]
    template_rel_global_mean: torch.Tensor  # [E,R]
    template_rel_global_var: torch.Tensor   # [E,R]
    template_rel_support: torch.Tensor      # [E]
    template_rel_ig: torch.Tensor           # [E]
    template_rel_type_names: list[str]
    motif_edges: torch.Tensor               # [M,5] = [class, template, part_i, part_j, motif_type]
    motif_support: torch.Tensor             # [M]

    @property
    def relation_dim(self) -> int:
        return len(RELATION_FEATURE_NAMES)

    def to_payload(self) -> dict:
        return {
            "kind": "aog_hkg",
            "schema": self.schema.to_payload(),
            "num_templates": self.num_templates,
            "pmi": self.pmi,
            "role_prior": self.role_prior,
            "func_proto_r": self.func_proto_r,
            "func_proto_d": self.func_proto_d,
            "class_role_proto_r": self.class_role_proto_r,
            "class_role_proto_d": self.class_role_proto_d,
            "template_prior": self.template_prior,
            "template_valid": self.template_valid,
            "template_role_prior": self.template_role_prior,
            "template_role_required": self.template_role_required,
            "template_role_proto_r": self.template_role_proto_r,
            "template_role_proto_d": self.template_role_proto_d,
            "template_edges": self.template_edges,
            "template_rel_mean": self.template_rel_mean,
            "template_rel_var": self.template_rel_var,
            "template_rel_global_mean": self.template_rel_global_mean,
            "template_rel_global_var": self.template_rel_global_var,
            "template_rel_support": self.template_rel_support,
            "template_rel_ig": self.template_rel_ig,
            "template_rel_type_names": self.template_rel_type_names,
            "motif_edges": self.motif_edges,
            "motif_support": self.motif_support,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "AOGHierarchicalKG":
        payload = dict(payload)
        payload.pop("kind", None)
        payload["schema"] = RoleSchema.from_payload(payload["schema"])
        return cls(**payload)


def empty_kg(schema: RoleSchema, token_dim: int, relation_dim: int = len(RELATION_FEATURE_NAMES)) -> HierarchicalKG:
    c, f = schema.num_classes, schema.num_parts
    return HierarchicalKG(
        schema=schema,
        pmi=torch.zeros(c, f),
        func_proto_r=torch.zeros(f, token_dim),
        func_proto_d=torch.zeros(f, token_dim),
        role_proto_r=torch.zeros(c, f, token_dim),
        role_proto_d=torch.zeros(c, f, token_dim),
        role_prior=torch.zeros(c, f),
        func_edges=torch.zeros(0, 2, dtype=torch.long),
        role_edges=torch.zeros(0, 3, dtype=torch.long),
        role_rel_mean=torch.zeros(0, relation_dim),
        role_rel_var=torch.ones(0, relation_dim),
        role_rel_support=torch.zeros(0),
        role_rel_ig=torch.zeros(0),
        role_rel_global_mean=torch.zeros(0, relation_dim),
        role_rel_global_var=torch.ones(0, relation_dim),
        role_rel_type_names=[],
    )


def empty_aog_hkg(
    schema: RoleSchema,
    token_dim: int,
    num_templates: int = 1,
    relation_dim: int = len(RELATION_FEATURE_NAMES),
) -> AOGHierarchicalKG:
    c, f, a = schema.num_classes, schema.num_parts, int(max(1, num_templates))
    return AOGHierarchicalKG(
        schema=schema,
        num_templates=a,
        pmi=torch.zeros(c, f),
        role_prior=torch.zeros(c, f),
        func_proto_r=torch.zeros(f, token_dim),
        func_proto_d=torch.zeros(f, token_dim),
        class_role_proto_r=torch.zeros(c, f, token_dim),
        class_role_proto_d=torch.zeros(c, f, token_dim),
        template_prior=torch.full((c, a), 1.0 / float(a)),
        template_valid=torch.ones(c, a),
        template_role_prior=torch.zeros(c, a, f),
        template_role_required=torch.zeros(c, a, f),
        template_role_proto_r=torch.zeros(c, a, f, token_dim),
        template_role_proto_d=torch.zeros(c, a, f, token_dim),
        template_edges=torch.zeros(0, 4, dtype=torch.long),
        template_rel_mean=torch.zeros(0, relation_dim),
        template_rel_var=torch.ones(0, relation_dim),
        template_rel_global_mean=torch.zeros(0, relation_dim),
        template_rel_global_var=torch.ones(0, relation_dim),
        template_rel_support=torch.zeros(0),
        template_rel_ig=torch.zeros(0),
        template_rel_type_names=[],
        motif_edges=torch.zeros(0, 5, dtype=torch.long),
        motif_support=torch.zeros(0),
    )
