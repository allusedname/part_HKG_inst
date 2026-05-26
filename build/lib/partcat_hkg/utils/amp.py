import torch


def autocast_cuda(enabled: bool = True):
    """Return a CUDA autocast context.

    This intentionally returns an autocast context even when ``enabled`` is
    ``False``.  Nested ``autocast(enabled=False)`` blocks are the PyTorch way to
    opt out of an outer autocast region.  Returning ``nullcontext`` for
    ``enabled=False`` leaves the outer autocast active, which is unsafe for
    probability-domain BCE and other explicitly-float32 computations.
    """
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=bool(enabled and torch.cuda.is_available()))
    return torch.cuda.amp.autocast(enabled=bool(enabled and torch.cuda.is_available()))


def make_scaler(enabled: bool = True):
    enabled = bool(enabled and torch.cuda.is_available())
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
