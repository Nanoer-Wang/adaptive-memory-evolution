"""Causal prefix budget: at most floor(rate * seen) extra calls, no borrowing."""
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class CallBudget:
    rate: float
    seen: int = 0
    used: int = 0

    def __post_init__(self):
        if not 0 <= self.rate <= 1:
            raise ValueError('rate must be in [0, 1]')
        self._fraction = Fraction(str(self.rate))

    def step(self, requested: bool) -> bool:
        self.seen += 1
        allowance = self.seen * self._fraction.numerator // self._fraction.denominator
        granted = bool(requested and self.used < allowance)
        self.used += int(granted)
        return granted
