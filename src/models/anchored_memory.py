"""Query-conditioned residual ranking; lexical evidence survives initialization."""
import torch
from torch import nn
from src.memory.liquid_memory import LiquidMemory

FEATURE_VERSION = 1
FEATURE_NAMES = ['tfidf', 'content_tfidf', 'bm25', 'query_coverage', 'best_window']


class AnchoredMemory(nn.Module):
    def __init__(self, kind='lcm_v2', memory_dim=16, consolidation=True, fixed_time=False, reset_history=False, **_):
        super().__init__()
        if kind not in ('lcm_v2', 'ranker', 'gru_v2'):
            raise ValueError('Unknown anchored model')
        self.kind = kind
        self.reset_history = reset_history
        self.local = nn.Linear(len(FEATURE_NAMES), 1, bias=False)
        nn.init.zeros_(self.local.weight)
        if kind != 'ranker':
            self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), memory_dim), nn.Tanh())
            self.cell = LiquidMemory(memory_dim, consolidation, fixed_time) if kind == 'lcm_v2' else nn.GRUCell(memory_dim, memory_dim)
            self.readout = nn.Linear(memory_dim, 1, bias=False)
            nn.init.zeros_(self.readout.weight)

    def forward(self, features, query=None, deltas=None):
        scores = features[:, 0] + self.local(features).squeeze(-1)
        if self.kind != 'ranker':
            encoded = self.encoder(features)
            state = torch.zeros_like(encoded[0])
            states = []
            for experience, delta in zip(encoded, deltas):
                if self.reset_history:
                    state = torch.zeros_like(state)
                state = self.cell(state, experience, delta=delta) if self.kind == 'lcm_v2' else self.cell(experience, state)
                states.append(state)
            scores = scores + self.readout(torch.stack(states)).squeeze(-1)
        zero = scores.sum() * 0
        return scores, zero, zero
