from __future__ import annotations

import torch
import torch.nn as nn

from .calibration import LogOpinionPoolFusion


class LegacyV51Readout(nn.Module):
    """Thin adapter for v51-style expert fusion.

    This class is intentionally separated from the main parse graph classifier.
    Port the notebook's full relation-routing branch here if running the legacy
    ablation is required.
    """

    expert_names = ["base", "node", "pmi", "edge"]

    def __init__(self, num_experts: int = 4, hidden: int = 64):
        super().__init__()
        self.fusion = LogOpinionPoolFusion(num_experts=num_experts, hidden=hidden)

    def forward(self, expert_logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        stacked = torch.stack([expert_logits[name] for name in self.expert_names], dim=-1)
        fused, gate = self.fusion(stacked)
        return {"legacy_fused_logits": fused, "legacy_gate_weights": gate}


class RelationRoutingBranch(nn.Module):
    """Placeholder for the notebook's learned relation routing branch.

    The final proposal prefers selected explicit edge factors. Keep this branch
    disabled unless reproducing v51/v55 experiments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Port v51 relation routing here only for legacy ablations.")
