"""Frozen inference adapters. Batch axes are independent streams, never time."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class FastLiquidWriter(nn.Module):
    __constants__ = ['static', 'fixed_time', 'hidden']

    def __init__(self, model):
        super().__init__()
        if model.kind not in ('liquid_content', 'static_content'):
            raise ValueError('Expected liquid cell')
        self.static = model.kind == 'static_content'
        self.fixed_time = model.cell.fixed_time
        self.hidden = model.hidden
        self.encoder = model.encoder
        self.head = model.head
        layers = [model.cell.rate, model.cell.candidate[0], model.cell.importance[0]]
        self.register_buffer('weight', torch.cat([m.weight.detach() for m in layers]))
        self.register_buffer('bias', torch.cat([m.bias.detach() for m in layers]))
        self.requires_grad_(False)

    def advance(self, encoded: torch.Tensor, state: torch.Tensor, delta: torch.Tensor):
        if self.static:
            state = torch.zeros_like(state)
        z = F.linear(torch.cat([state, encoded], dim=-1), self.weight, self.bias)
        if self.fixed_time:
            delta = torch.zeros_like(delta)
        alpha = torch.exp(-F.softplus(z[..., :self.hidden]) * (1 + delta.unsqueeze(-1)))
        candidate = torch.tanh(z[..., self.hidden:2*self.hidden])
        importance = torch.sigmoid(z[..., 2*self.hidden:])
        proposal = alpha*state + (1-alpha)*candidate
        state = state + importance*(proposal-state)
        return self.head(state).squeeze(-1), state

    @torch.jit.export
    def step(self, vector: torch.Tensor, state: torch.Tensor, delta: torch.Tensor):
        return self.advance(self.encoder(vector), state, delta)

    def forward(self, vectors: torch.Tensor, deltas: torch.Tensor):
        # [time, independent streams, embedding]; each projection is pointwise.
        encoded = self.encoder(vectors)
        state = vectors.new_zeros((vectors.size(1), self.hidden))
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        for i in range(vectors.size(0)):
            score, state = self.advance(encoded[i], state, deltas[i])
            outputs.append(score)
        return torch.stack(outputs)


class FastGRUWriter(nn.Module):
    __constants__ = ['hidden']

    def __init__(self, model):
        super().__init__()
        self.encoder, self.cell, self.head = model.encoder, model.cell, model.head
        self.hidden = model.hidden
        self.requires_grad_(False)

    @torch.jit.export
    def step(self, vector: torch.Tensor, state: torch.Tensor, delta: torch.Tensor):
        state = self.cell(self.encoder(vector), state)
        return self.head(state).squeeze(-1), state

    def forward(self, vectors: torch.Tensor, deltas: torch.Tensor):
        encoded = self.encoder(vectors)
        state = vectors.new_zeros((vectors.size(1), self.hidden))
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        for i in range(vectors.size(0)):
            state = self.cell(encoded[i], state)
            outputs.append(self.head(state).squeeze(-1))
        return torch.stack(outputs)


def frozen_writer(model, scripted=True):
    model = copy.deepcopy(model)
    adapter = FastGRUWriter(model) if model.kind == 'gru_content' else FastLiquidWriter(model)
    adapter.eval()
    return torch.jit.script(adapter) if scripted else adapter


class OnlineFrozenWriter:
    """BudgetStore-compatible inference adapter; maintains no hidden private state."""
    def __init__(self, model, scripted=True):
        self.hidden = model.hidden
        self.impl = frozen_writer(model, scripted)

    def step(self, vector, state, delta):
        elapsed = torch.as_tensor(delta, dtype=vector.dtype, device=vector.device)
        return self.impl.step(vector, state, elapsed)
