import torch
import torch.nn.functional as F
from partcat_hkg.data.schema import RoleSchema


def role_valid_mask_for_batch(labels: torch.Tensor, schema: RoleSchema) -> torch.Tensor:
    return (schema.role_to_obj.to(labels.device).view(1, -1) == labels.view(-1, 1)).float()


def aggregate_role_prob_to_func(role_prob: torch.Tensor, schema: RoleSchema, role_mask: torch.Tensor | None = None) -> torch.Tensor:
    if role_mask is not None:
        role_prob = role_prob * role_mask
    b, _, h, w = role_prob.shape
    out = torch.zeros(b, schema.num_parts, h, w, device=role_prob.device, dtype=role_prob.dtype)
    r2p = schema.role_to_part.to(role_prob.device)
    for k in range(schema.num_parts):
        idx = (r2p == k).nonzero(as_tuple=False).flatten()
        if idx.numel() > 0:
            out[:, k] = role_prob[:, idx].amax(dim=1)
    return out


def functional_mask_quality(func_tokens: torch.Tensor, func_proto: torch.Tensor) -> torch.Tensor:
    """Simple token/prototype quality proxy in [0,1]."""
    sim = (F.normalize(func_tokens, dim=-1) * F.normalize(func_proto, dim=-1).unsqueeze(0)).sum(-1)
    return (0.5 + 0.5 * sim).clamp(0, 1)
