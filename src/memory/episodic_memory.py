
import torch


class EpisodicMemory:
    """Detached inference memory, optional oldest-first capacity bound."""
    def __init__(self, capacity=None):
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.storage=[]

    def add(self, embedding, metadata=None):
        self.storage.append(
            {
                "embedding": embedding.detach(),
                "metadata": metadata
            }
        )
        if self.capacity is not None and len(self.storage) > self.capacity:
            self.storage.pop(0)

    def __len__(self):
        return len(self.storage)
