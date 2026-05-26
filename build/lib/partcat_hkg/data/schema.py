from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

from .canonicalization import canonicalize_object_name, canonicalize_part_name, role_name


@dataclass
class RoleSchema:
    obj_names: list[str]
    part_names: list[str]
    role_names: list[str]
    role_to_obj: torch.Tensor
    role_to_part: torch.Tensor
    role_index_table: torch.Tensor

    @property
    def num_classes(self) -> int:
        return len(self.obj_names)

    @property
    def num_parts(self) -> int:
        return len(self.part_names)

    @property
    def num_roles(self) -> int:
        return len(self.role_names)

    @property
    def obj_to_idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.obj_names)}

    @property
    def part_to_idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.part_names)}

    @property
    def role_to_idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.role_names)}

    @classmethod
    def from_coco_categories(cls, categories: Iterable[dict]) -> "RoleSchema":
        cats = list(categories)
        obj_names = sorted({canonicalize_object_name(cat.get("supercategory", "unknown")) for cat in cats})
        part_names = sorted({canonicalize_part_name(cat.get("name", "unknown")) for cat in cats})
        role_names = sorted({
            role_name(cat.get("supercategory", "unknown"), canonicalize_part_name(cat.get("name", "unknown")))
            for cat in cats
        })
        return cls.from_names(obj_names, part_names, role_names)

    @classmethod
    def from_names(cls, obj_names: list[str], part_names: list[str], role_names: list[str]) -> "RoleSchema":
        obj_names = [canonicalize_object_name(x) for x in obj_names]
        part_names = list(part_names)
        role_names = list(role_names)
        obj_to_idx = {n: i for i, n in enumerate(obj_names)}
        part_to_idx = {n: i for i, n in enumerate(part_names)}
        role_to_obj = torch.full((len(role_names),), -1, dtype=torch.long)
        role_to_part = torch.full((len(role_names),), -1, dtype=torch.long)
        role_index_table = torch.full((len(obj_names), len(part_names)), -1, dtype=torch.long)
        for r, rn in enumerate(role_names):
            if ":" not in rn:
                continue
            obj, part = rn.split(":", 1)
            if obj in obj_to_idx and part in part_to_idx:
                c, k = obj_to_idx[obj], part_to_idx[part]
                role_to_obj[r] = c
                role_to_part[r] = k
                role_index_table[c, k] = r
        return cls(obj_names, part_names, role_names, role_to_obj, role_to_part, role_index_table)

    def valid_roles_for_class(self, class_idx: int) -> torch.Tensor:
        return self.role_to_obj == int(class_idx)

    def role_for(self, class_idx: int, part_idx: int) -> int:
        return int(self.role_index_table[int(class_idx), int(part_idx)].item())

    def smoke_test(self, strict: bool = True) -> dict[str, int]:
        invalid = int((self.role_to_obj < 0).sum().item())
        duplicate_slots = self._duplicate_slot_count()
        report = {"invalid_role_mappings": invalid, "duplicate_role_slots": duplicate_slots}
        if strict and (invalid or duplicate_slots):
            raise ValueError(f"Role schema failed smoke test: {report}")
        return report

    def _duplicate_slot_count(self) -> int:
        seen: set[tuple[int, int]] = set()
        duplicates = 0
        for r in range(self.num_roles):
            c = int(self.role_to_obj[r].item())
            k = int(self.role_to_part[r].item())
            if c < 0 or k < 0:
                continue
            key = (c, k)
            if key in seen:
                duplicates += 1
            seen.add(key)
        return duplicates

    def to_payload(self) -> dict:
        return {
            "obj_names": self.obj_names,
            "part_names": self.part_names,
            "role_names": self.role_names,
            "role_to_obj": self.role_to_obj,
            "role_to_part": self.role_to_part,
            "role_index_table": self.role_index_table,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "RoleSchema":
        return cls(
            list(payload["obj_names"]),
            list(payload["part_names"]),
            list(payload["role_names"]),
            payload["role_to_obj"].long(),
            payload["role_to_part"].long(),
            payload["role_index_table"].long(),
        )
