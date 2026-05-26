import torch


def finite_center_clip_logits(x: torch.Tensor, max_abs: float = 30.0) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=max_abs, neginf=-max_abs)
    x = x - x.mean(dim=-1, keepdim=True)
    return x.clamp(-max_abs, max_abs)


def count_parameters(module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())
