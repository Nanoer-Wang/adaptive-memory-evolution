"""Query-independent stream encoder; only its fixed-width state reaches reader."""
import torch
from torch import nn
from torch.nn import functional as F


class BoundedCodec(nn.Module):
    def __init__(self, kind, width, entities=4, values=6):
        super().__init__()
        self.kind, self.width = kind, width
        self.entities, self.values = entities, values
        inputs = entities + values
        if kind == 'gru':
            self.cell = nn.GRUCell(inputs, width)
        else:
            self.target = nn.Linear(inputs, width)
            self.rate = nn.Linear(inputs + (width if kind == 'ltc' else 0), width)
            self.leak = nn.Parameter(torch.zeros(width))
        self.reader = nn.Sequential(nn.Linear(width + entities, width), nn.Tanh(), nn.Linear(width, values))

    def write(self, state, event):
        if self.kind == 'gru':
            return self.cell(event, state)
        target = torch.tanh(self.target(event))
        # Each event is held for one logical unit. LTC uses four numerical
        # substeps, input-only flow has an exact frozen-coefficient update.
        if self.kind == 'input_flow':
            f = F.softplus(self.rate(event))
            leak = F.softplus(self.leak)
            rate = leak + f
            equilibrium = f * target / rate
            return equilibrium + (state - equilibrium) * torch.exp(-rate)
        for _ in range(4):
            f = F.softplus(self.rate(torch.cat((state, event), -1)))
            state = (state + .25 * f * target) / (1 + .25 * (F.softplus(self.leak) + f))
        return state

    def read(self, state, query):
        return self.reader(torch.cat((state, query), -1))

    def forward(self, events, valid, query):
        state = events.new_zeros((len(events), self.width))
        for t in range(events.shape[1]):
            updated = self.write(state, events[:, t])
            state = torch.where(valid[:, t, None], updated, state)
        return self.read(state, query)


def pack_state(state):
    """Only fixed float32 payload, no source, per-event cache or query."""
    return state.detach().cpu().numpy().astype('<f4').tobytes()


def unpack_state(payload, width):
    import numpy as np
    if len(payload) != 4 * width:
        raise ValueError('Incorrect memory payload size')
    return torch.from_numpy(np.frombuffer(payload, dtype='<f4').copy()).reshape(1, width)
