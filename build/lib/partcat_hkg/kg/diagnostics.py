from __future__ import annotations

import pandas as pd
from .datatypes import HierarchicalKG
from partcat_hkg.data.canonicalization import display_object_name


def pmi_report(kg: HierarchicalKG, topk: int = 5) -> pd.DataFrame:
    rows = []
    for c, obj in enumerate(kg.schema.obj_names):
        vals = kg.pmi[c]
        for k in vals.abs().topk(min(topk, vals.numel())).indices.tolist():
            rows.append({"class": display_object_name(obj), "part": kg.schema.part_names[k], "pmi": float(vals[k])})
    return pd.DataFrame(rows)


def role_edge_report(kg: HierarchicalKG) -> pd.DataFrame:
    rows = []
    for e in range(int(kg.role_edges.shape[0])):
        c, i, j = [int(x) for x in kg.role_edges[e].tolist()]
        rows.append({
            "edge_idx": e,
            "class": display_object_name(kg.schema.obj_names[c]),
            "part_i": kg.schema.part_names[i],
            "part_j": kg.schema.part_names[j],
            "relation_type": kg.role_rel_type_names[e] if e < len(kg.role_rel_type_names) else "unknown",
            "support": float(kg.role_rel_support[e]),
            "information_gain": float(kg.role_rel_ig[e]),
        })
    return pd.DataFrame(rows)
