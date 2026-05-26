import torch
import torch.nn.functional as F


def top_down_completion_score(
    func_tokens_r: torch.Tensor,
    func_tokens_d: torch.Tensor,
    role_proto_r: torch.Tensor,
    role_proto_d: torch.Tensor,
    functional_presence: torch.Tensor,
    functional_quality: torch.Tensor,
    unknown_mask: torch.Tensor,
    valid_cf: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Score unknown role slots using functional evidence and class-role prototypes."""
    fr = F.normalize(func_tokens_r, dim=-1)
    fd = F.normalize(func_tokens_d, dim=-1)
    pr = F.normalize(role_proto_r, dim=-1)
    pd = F.normalize(role_proto_d, dim=-1)
    sim = 0.5 * (torch.einsum("bfd,cfd->bcf", fr, pr) + torch.einsum("bfd,cfd->bcf", fd, pd))
    weight = functional_presence.unsqueeze(1) * functional_quality.unsqueeze(1) * unknown_mask * valid_cf.unsqueeze(0)
    mass = weight.sum(-1)
    valid_count = valid_cf.sum(-1).clamp_min(1.0).view(1, -1)
    score = (weight * sim).sum(-1) * torch.sqrt(valid_count) / torch.sqrt(mass + eps)
    return torch.nan_to_num(score, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)
