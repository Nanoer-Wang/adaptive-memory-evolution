"""Event-driven per-version retention dynamics, separate from factual validity.

Between evidence events: dz/dt = rate * (equilibrium - z), solved exactly.
Coefficients remain fixed until a related evidence event. This is a small
piecewise-linear continuous-time model, NOT an implementation of CfC/LTC.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

FEATURES = 16
HIDDEN = 8
KINDS = ('static', 'gru', 'liquid', 'fixed_rate', 'no_time')


def exact_flow(state, equilibrium, rate, elapsed_days):
    elapsed = torch.as_tensor(elapsed_days, dtype=state.dtype, device=state.device)
    if not bool(torch.isfinite(elapsed).all()) or bool((elapsed < 0).any()):
        raise ValueError('Elapsed time must be finite and nonnegative')
    if not bool(torch.isfinite(rate).all()) or bool((rate < 0).any()):
        raise ValueError('Rates must be finite and nonnegative')
    # expm1 avoids cancellation for tiny elapsed intervals.
    return state + (-torch.expm1(-rate * elapsed)) * (equilibrium - state)


class EvidenceDynamics(nn.Module):
    def __init__(self, kind='liquid', feature_dim=FEATURES):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown dynamics control')
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(feature_dim, 16), nn.Tanh())
        self.impulse = nn.Linear(16, HIDDEN)
        self.gate = nn.Linear(16, HIDDEN)
        self.rate_head = nn.Linear(16, HIDDEN)
        self.equilibrium_head = nn.Linear(16, HIDDEN)
        self.gru = nn.GRUCell(16, HIDDEN) if kind == 'gru' else None
        self.head = nn.Linear(HIDDEN + feature_dim, 1)

    def flow(self, state, equilibrium, rate, elapsed):
        if self.kind in ('static', 'gru', 'no_time'):
            return state
        return exact_flow(state, equilibrium, rate, elapsed)

    def event(self, state, features):
        encoded = self.encoder(features)
        target = torch.sigmoid(self.impulse(encoded))
        if self.kind == 'static':
            updated = target  # Same features, no recurrent latent carry.
        elif self.kind == 'gru':
            updated = self.gru(encoded, state)
        else:
            gate = torch.sigmoid(self.gate(encoded))
            updated = state + gate * (target - state)
        rate = (F.softplus(self.rate_head(encoded)) + 1e-4) / 30.
        if self.kind == 'fixed_rate':
            rate = torch.ones_like(rate) / 30.
        equilibrium = torch.sigmoid(self.equilibrium_head(encoded))
        return updated, equilibrium, rate

    def score(self, state, features):
        # Current ledger features are identical across learned controls.
        return self.head(torch.cat((state, features), dim=-1)).squeeze(-1)
