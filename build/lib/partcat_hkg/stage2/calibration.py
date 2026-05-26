from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarCalibratedReadout(nn.Module):
    """Optional readout: base + learned softplus weights over parse terms."""

    def __init__(self, num_terms: int):
        super().__init__()
        self.raw_lambda = nn.Parameter(torch.zeros(num_terms))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, base_logits: torch.Tensor, *terms: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(self.raw_lambda)
        out = base_logits
        for w, term in zip(weights, terms):
            out = out + w * term
        return out + self.bias


class LogOpinionPoolFusion(nn.Module):
    """Legacy v51-style product-of-experts fusion.

    Kept as an ablation/readout baseline, not as the main proposal classifier.
    """

    def __init__(self, num_experts: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(num_experts * 2, hidden), nn.SiLU(), nn.Linear(hidden, num_experts))

    def forward(self, expert_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # expert_logits: [B,C,E]
        margins = expert_logits - expert_logits.max(dim=1, keepdim=True).values
        gate_feat = torch.cat([expert_logits.detach(), margins.detach()], dim=-1)
        gate_w = F.softmax(self.gate(gate_feat), dim=-1)
        logp = F.log_softmax(expert_logits, dim=1)
        return (gate_w * logp).sum(dim=-1), gate_w
