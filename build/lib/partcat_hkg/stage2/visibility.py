from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import torch


class VisibilityState(IntEnum):
    ABSENT = 0
    UNKNOWN = 1
    VISIBLE = 2
    CONTRADICTORY = 3


@dataclass
class VisibilityOutput:
    state: torch.Tensor          # [B,C,F], integer states
    visible: torch.Tensor        # [B,C,F]
    unknown: torch.Tensor        # [B,C,F]
    contradictory: torch.Tensor  # [B,C,F]
    reliability: torch.Tensor    # [B,C,F]


def compute_visibility_states(
    role_presence_cf: torch.Tensor,
    functional_presence: torch.Tensor,
    functional_quality: torch.Tensor,
    valid_cf: torch.Tensor,
    *,
    presence_tau: float = 0.15,
    quality_tau: float = 0.25,
) -> VisibilityOutput:
    """Assign proposal visibility states for candidate class-role nodes."""
    valid = valid_cf.unsqueeze(0).float()
    fp = functional_presence.unsqueeze(1) * valid
    fq = functional_quality.unsqueeze(1) * valid
    rp = role_presence_cf * valid
    visible = ((rp >= presence_tau) & (fq >= quality_tau) & (valid > 0)).float()
    unknown = ((rp < presence_tau) & (fp >= presence_tau) & (valid > 0)).float()
    contradictory = ((rp >= presence_tau) & (fq < quality_tau) & (valid > 0)).float()
    state = torch.full_like(rp, int(VisibilityState.ABSENT), dtype=torch.long)
    state = torch.where(unknown.bool(), torch.full_like(state, int(VisibilityState.UNKNOWN)), state)
    state = torch.where(visible.bool(), torch.full_like(state, int(VisibilityState.VISIBLE)), state)
    state = torch.where(contradictory.bool(), torch.full_like(state, int(VisibilityState.CONTRADICTORY)), state)
    reliability = (0.35 + 0.65 * fq).clamp(0, 1) * valid
    return VisibilityOutput(state, visible, unknown, contradictory, reliability)
