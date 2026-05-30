from __future__ import annotations

from pathlib import Path
import torch

from .instance_aog import InstanceAOG


def save_instance_aog(grammar: InstanceAOG, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grammar.to_payload(), path)


def load_instance_aog(path: str | Path) -> InstanceAOG:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, InstanceAOG):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an InstanceAOG payload dict, got {type(payload).__name__}")
    kind = payload.get("kind")
    if kind not in {None, "instance_aog"}:
        raise ValueError(f"Expected kind='instance_aog', got {kind!r}")
    return InstanceAOG.from_payload(payload)
