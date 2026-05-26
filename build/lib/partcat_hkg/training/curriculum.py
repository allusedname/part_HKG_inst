from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeStage2Flags:
    enable_completion: bool
    enable_edges: bool


def expand_curriculum(steps) -> list[RuntimeStage2Flags]:
    flags = []
    for step in steps:
        for _ in range(int(step.epochs)):
            flags.append(RuntimeStage2Flags(step.enable_completion, step.enable_edges))
    return flags
