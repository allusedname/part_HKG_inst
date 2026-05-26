from __future__ import annotations

from pathlib import Path
import torch
from .datatypes import AOGHierarchicalKG, HierarchicalKG


def save_hkg(kg: HierarchicalKG | AOGHierarchicalKG, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kg.to_payload(), path)


def load_hkg(path: str | Path) -> HierarchicalKG | AOGHierarchicalKG:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, (HierarchicalKG, AOGHierarchicalKG)):
        return payload
    if isinstance(payload, dict) and payload.get("kind") == "aog_hkg":
        return AOGHierarchicalKG.from_payload(payload)
    return HierarchicalKG.from_payload(payload)
