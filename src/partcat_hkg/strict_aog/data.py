from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .terminals import load_terminal_cache


class StrictAOGTerminalDataset(Dataset):
    """Dataset backed by cached Stage-1 terminal proposals."""

    def __init__(self, cache_path: str | Path, *, map_location: str | torch.device = "cpu"):
        payload = load_terminal_cache(cache_path, map_location=map_location)
        self.records = payload["records"]
        if not self.records:
            raise ValueError(f"No records found in cache {cache_path}")
        self.schema_payload = payload.get("schema")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[int(idx)]
        out: dict[str, torch.Tensor] = {}
        for k, v in r.items():
            if k == "obj_label":
                out[k] = torch.tensor(int(v), dtype=torch.long)
            elif torch.is_tensor(v):
                out[k] = v.clone()
        return out


def collate_strict_aog(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty strict AOG batch")
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, dim=0)
    return out
