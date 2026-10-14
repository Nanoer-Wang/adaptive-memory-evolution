"""Causal content storage under a serialized payload byte budget.

Dates and source lineage are retained; differing versions are never overwritten.
This stores extractive evidence, not inferred entity/relation facts.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BudgetItem:
    session_id: str
    date: str
    text: str
    vector: np.ndarray
    priority: float
    sequence: int

    @property
    def nbytes(self):
        return len(self.session_id.encode()) + len(self.date.encode()) + len(self.text.encode()) + self.vector.nbytes + 16


class BudgetStore:
    def __init__(self, budget_bytes, writer=None):
        self.budget_bytes = budget_bytes
        self.writer = writer
        self.state = torch.zeros(writer.hidden) if writer is not None else None
        self.items = []
        self.sequence = 0
        self.evicted = []
        if budget_bytes is not None and budget_bytes < self.state_bytes:
            raise ValueError('Budget smaller than persistent state')

    @property
    def state_bytes(self):
        return self.state.numel() * self.state.element_size() if self.state is not None else 0

    @property
    def payload_bytes(self):
        return sum(item.nbytes for item in self.items) + self.state_bytes

    def insert(self, session_id, date, text, vector, delta):
        priority = float(self.sequence)
        if self.writer is not None:
            with torch.no_grad():
                score, self.state = self.writer.step(torch.as_tensor(vector), self.state, delta)
                priority = float(score)
        self.items.append(BudgetItem(session_id, date, text, np.asarray(vector, dtype=np.float32).copy(), priority, self.sequence))
        self.sequence += 1
        while self.budget_bytes is not None and self.payload_bytes > self.budget_bytes:
            worst = min(range(len(self.items)), key=lambda i: (self.items[i].priority, self.items[i].sequence))
            self.evicted.append(self.items.pop(worst).session_id)


def byte_retained_indices(scores, costs, budget, state_bytes=0):
    """Online insertion/eviction simulation; evicted records cannot reappear."""
    kept = []
    used = state_bytes
    for index, (score, cost) in enumerate(zip(scores, costs)):
        kept.append(index)
        used += cost
        while used > budget and kept:
            worst = min(kept, key=lambda i: (float(scores[i]), i))
            kept.remove(worst)
            used -= costs[worst]
    return kept
