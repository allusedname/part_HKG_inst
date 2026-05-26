import torch
from partcat_hkg.stage2.visibility import compute_visibility_states, VisibilityState


def test_visibility_visible():
    role_cf = torch.tensor([[[0.8, 0.0]]])
    func = torch.tensor([[0.9, 0.4]])
    qual = torch.tensor([[0.9, 0.9]])
    valid = torch.tensor([[1.0, 1.0]])
    out = compute_visibility_states(role_cf, func, qual, valid)
    assert int(out.state[0, 0, 0]) == int(VisibilityState.VISIBLE)
    assert int(out.state[0, 0, 1]) == int(VisibilityState.UNKNOWN)
