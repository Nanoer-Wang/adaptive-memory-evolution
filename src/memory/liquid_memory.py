
import torch
from torch import nn
from torch.nn import functional as F


class LiquidMemory(nn.Module):
    """Time-aware gated recurrent approximation, not an ODE solver or exact CfC."""
    def __init__(self, dim, consolidation=True, fixed_time=False):
        super().__init__()
        self.dim = dim
        self.consolidation = consolidation
        self.fixed_time = fixed_time
        self.rate = nn.Linear(dim * 2, dim)
        self.candidate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())
        self.importance = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())

    def forward(self, state, experience, importance=None, delta=0.0):
        joined = torch.cat([state, experience], dim=-1)
        elapsed = torch.as_tensor(delta, dtype=state.dtype, device=state.device)
        if self.fixed_time:
            elapsed = torch.zeros_like(elapsed)
        alpha = torch.exp(-F.softplus(self.rate(joined)) * (1 + elapsed))
        proposal = alpha * state + (1 - alpha) * self.candidate(joined)
        if self.consolidation:
            weight = self.importance(joined) if importance is None else importance
            return state + weight * (proposal - state)
        return proposal
