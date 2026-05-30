from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


CACHE_KEYS = [
    "image_index",
    "obj_label",
    "part_presence",
    "part_tokens",
    "component_valid",
    "component_part",
    "component_presence",
    "component_geom",
    "component_token",
]


class GPUComponentCacheDataset(Dataset):
    """Dataset backed by Stage-1 component-cache shards.

    Each shard is a torch payload created by scripts/cache_gpu_instance_components.py.
    It stores CPU tensors, but the actual Stage-2 trainer moves each batch to GPU.
    """

    def __init__(self, cache_dir: str | Path, *, split: str = "train", load_into_memory: bool = True):
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        pattern = f"{self.split}_shard_*.pt"
        self.shards = sorted(self.cache_dir.glob(pattern))
        if not self.shards:
            # Accept a directory that contains only shard_*.pt for convenience.
            self.shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No cache shards found in {self.cache_dir} matching {pattern}")
        self.meta = {}
        meta_path = self.cache_dir / f"{self.split}_meta.pt"
        if meta_path.exists():
            self.meta = torch.load(meta_path, map_location="cpu")
        elif (self.cache_dir / "meta.pt").exists():
            self.meta = torch.load(self.cache_dir / "meta.pt", map_location="cpu")
        self.load_into_memory = bool(load_into_memory)
        self._data: dict[str, torch.Tensor] | None = None
        if self.load_into_memory:
            self._data = self._load_all()
            self.length = int(self._data["obj_label"].shape[0])
        else:
            self._lengths = []
            for p in self.shards:
                payload = torch.load(p, map_location="cpu")
                self._lengths.append(int(payload["obj_label"].shape[0]))
            self.length = int(sum(self._lengths))
            self._cum = torch.tensor([0] + list(torch.tensor(self._lengths).cumsum(0).tolist()), dtype=torch.long)

    @property
    def schema_payload(self) -> dict | None:
        value = self.meta.get("schema") if isinstance(self.meta, dict) else None
        return value if isinstance(value, dict) else None

    def _load_all(self) -> dict[str, torch.Tensor]:
        pieces: dict[str, list[torch.Tensor]] = {k: [] for k in CACHE_KEYS}
        extra: dict[str, list[torch.Tensor]] = {}
        for p in self.shards:
            payload = torch.load(p, map_location="cpu")
            for k, v in payload.items():
                if not torch.is_tensor(v):
                    continue
                if k in pieces:
                    pieces[k].append(v)
                else:
                    extra.setdefault(k, []).append(v)
        out: dict[str, torch.Tensor] = {}
        for k, vals in pieces.items():
            if vals:
                out[k] = torch.cat(vals, dim=0)
        for k, vals in extra.items():
            if vals:
                out[k] = torch.cat(vals, dim=0)
        missing = [k for k in CACHE_KEYS if k not in out]
        if missing:
            raise KeyError(f"Cache missing required keys: {missing}")
        return out

    def __len__(self) -> int:
        return self.length

    def _getitem_memory(self, idx: int) -> dict[str, torch.Tensor]:
        assert self._data is not None
        return {k: v[idx] for k, v in self._data.items() if torch.is_tensor(v)}

    def _getitem_stream(self, idx: int) -> dict[str, torch.Tensor]:
        # Simple streaming path for very large caches. It loads one shard per item,
        # so the default should be load_into_memory=True for speed.
        shard_id = int(torch.searchsorted(self._cum[1:], torch.tensor(idx), right=False).item())
        start = int(self._cum[shard_id].item())
        payload = torch.load(self.shards[shard_id], map_location="cpu")
        local = idx - start
        return {k: v[local] for k, v in payload.items() if torch.is_tensor(v)}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.load_into_memory:
            return self._getitem_memory(int(idx))
        return self._getitem_stream(int(idx))


def collate_gpu_component_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot collate an empty GPU component batch")
    keys = sorted(items[0].keys())
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        if torch.is_tensor(items[0][k]):
            out[k] = torch.stack([it[k] for it in items], dim=0)
    return out


def move_component_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device | str, *, non_blocking: bool = True) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    return {k: (v.to(dev, non_blocking=non_blocking) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_component_cache_tensors(cache_dir: str | Path, *, split: str = "train", device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    ds = GPUComponentCacheDataset(cache_dir, split=split, load_into_memory=True)
    dev = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    assert ds._data is not None
    return move_component_batch_to_device(ds._data, dev, non_blocking=False)


def save_cache_meta(cache_dir: str | Path, *, split: str, meta: dict[str, Any]) -> None:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(meta, p / f"{split}_meta.pt")
