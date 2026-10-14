"""Shared neural value codec with explicit entity addressing, bounded state.

Source text is never retained. Slot identities are a fixed shared ontology;
value states, and optional duplicate guards, are counted per stream.
"""
import numpy as np
import torch
from torch import nn
from src.memory.bounded_codec import BoundedCodec


class AddressedCodec(nn.Module):
    def __init__(self, kind, slot_width, entities=4, values=6):
        super().__init__()
        self.entities, self.values, self.slot_width = entities, values, slot_width
        self.kind = kind
        self.cell = BoundedCodec(kind, slot_width, entities=0, values=values)

    def initial(self, batch=1):
        return next(self.parameters()).new_zeros(batch, self.entities, self.slot_width)

    def write(self, state, owner, value):
        # A new observation changes only the addressed entity's slot.
        index = torch.arange(len(state), device=state.device)
        old = state[index, owner]
        updated = self.cell.write(old, value)
        mask = torch.nn.functional.one_hot(owner, self.entities).to(state.dtype)[:, :, None]
        return state * (1 - mask) + updated[:, None, :] * mask

    def read(self, state, query):
        selected = state[torch.arange(len(state), device=state.device), query]
        return self.cell.read(selected, selected.new_zeros(len(state), 0))

    def forward(self, events, valid, query):
        state = self.initial(len(events))
        for step in range(events.shape[1]):
            e = events[:, step]
            updated = self.write(state, e[:, :self.entities].argmax(-1), e[:, self.entities:])
            state = torch.where(valid[:, step, None, None], updated, state)
        return self.read(state, query.argmax(-1))

    def pack(self, state):
        if tuple(state.shape) != (1, self.entities, self.slot_width):
            raise ValueError('Only one bounded stream can be packed')
        return state.detach().cpu().numpy().astype('<f4').tobytes()

    def unpack(self, payload):
        if len(payload) != self.entities * self.slot_width * 4:
            raise ValueError('Invalid state size')
        return torch.from_numpy(np.frombuffer(payload, dtype='<f4').copy()).reshape(1, self.entities, self.slot_width)


class GuardedWriter:
    """Exact semantic-ID guard. Strong cheap control, NOT a liquid innovation.

    Guard compares incoming observed value IDs, never answer labels. No history
    cache: at most one uint8 per entity. Neural state is still the only reader
    input; the cache is counted because it contains reusable stream information.
    """
    def __init__(self, model, deduplicate=False):
        if model.values >= 255:
            raise ValueError('uint8 guard requires fewer than 255 values')
        self.model = model
        self.state = model.initial()
        self.cache = bytearray([255] * model.entities) if deduplicate else None
        self.writes = 0
        self.observations = 0

    def write(self, owner, value, owner_tensor, value_tensor):
        self.observations += 1
        if self.cache is not None and self.cache[owner] == value:
            return False
        self.state = self.model.write(self.state, owner_tensor, value_tensor)
        if self.cache is not None:
            self.cache[owner] = value
        self.writes += 1
        return True

    def pack(self):
        return self.model.pack(self.state) + (bytes(self.cache) if self.cache is not None else b'')

    def restore(self, payload):
        size = self.model.entities * self.model.slot_width * 4
        expected = size + (self.model.entities if self.cache is not None else 0)
        if len(payload) != expected:
            raise ValueError('Invalid writer payload size')
        self.state = self.model.unpack(payload[:size])
        if self.cache is not None:
            self.cache[:] = payload[size:]

    @property
    def payload_bytes(self):
        return self.model.entities * self.model.slot_width * 4 + (self.model.entities if self.cache is not None else 0)
