from pathlib import Path
from typing import Any
import json
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_checkpoint(path: str | Path, model, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, model, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=strict)
    return payload.get("extra", {})
