from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.experience_encoder import ExperienceEncoder
from src.memory.liquid_memory import LiquidMemory


class MemoryRetriever(nn.Module):
    def __init__(self, input_dim=1024, memory_dim=128, kind="lcm", consolidation=True, fixed_time=False, **_):
        super().__init__()
        if kind not in ("lcm", "gru", "transformer"):
            raise ValueError("Unknown memory model")
        self.kind, self.dim = kind, memory_dim
        self.encoder = ExperienceEncoder(input_dim, memory_dim)
        self.key = nn.Linear(memory_dim, memory_dim)
        self.query = nn.Linear(memory_dim, memory_dim)
        self.adapter = nn.Sequential(nn.Linear(memory_dim * 2, memory_dim), nn.Tanh())
        self.future = nn.Linear(memory_dim, input_dim)
        if kind == "lcm":
            self.cell = LiquidMemory(memory_dim, consolidation, fixed_time)
        elif kind == "gru":
            self.cell = nn.GRUCell(memory_dim, memory_dim)
        else:
            self.cell = nn.TransformerEncoder(nn.TransformerEncoderLayer(memory_dim, nhead=4,
                dim_feedforward=memory_dim * 2, dropout=0, batch_first=True), num_layers=1)

    def forward(self, features, query, deltas):
        encoded = self.encoder(features)
        if self.kind == "transformer":
            length = len(encoded)
            position = torch.arange(length, device=encoded.device, dtype=encoded.dtype)[:, None]
            frequency = torch.exp(torch.arange(0, self.dim, 2, device=encoded.device) * (-math.log(10000) / self.dim))
            pe = torch.zeros_like(encoded)
            pe[:, 0::2] = torch.sin(position * frequency)
            pe[:, 1::2] = torch.cos(position * frequency)
            mask = torch.triu(torch.ones(length, length, device=encoded.device, dtype=torch.bool), diagonal=1)
            states = self.cell((encoded + pe).unsqueeze(0), mask=mask)[0]
        else:
            state = torch.zeros(self.dim, device=features.device, dtype=features.dtype)
            collected = []
            for experience, delta in zip(encoded, deltas):
                state = self.cell(state, experience, delta=delta) if self.kind == "lcm" else self.cell(experience, state)
                collected.append(state)
            states = torch.stack(collected)
        q = self.query(self.encoder(query))
        conditioned = F.normalize(q + self.adapter(torch.cat([q, states[-1]])), dim=-1)
        keys = F.normalize(self.key(encoded), dim=-1)
        scores = keys @ conditioned
        future_loss = (1 - F.cosine_similarity(self.future(states[:-1]), features[1:], dim=-1)).mean() if len(states) > 1 else scores.sum() * 0
        stability_loss = (states[1:] - states[:-1]).square().mean() if len(states) > 1 else scores.sum() * 0
        return scores, future_loss, stability_loss
