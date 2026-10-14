"""Bounded evidence ledger with observation-time versioning and local dynamics.

Version status is an explicit state machine; learned utility never decides truth.
An observation timestamp is not a claim's real-world effective date.
"""
from dataclasses import dataclass, field
import hashlib
import math
import numpy as np
import torch
from src.memory.evidence_dynamics import HIDDEN


@dataclass(frozen=True)
class Evidence:
    owner: str
    predicate: str
    scope: str
    value: str
    observed_at: float
    source_id: str
    turn: int
    start: int
    end: int
    quote: str
    mode: str = 'current'  # current / past / tentative / planned / retracted

    def __post_init__(self):
        if not math.isfinite(self.observed_at):
            raise ValueError('Invalid source time')
        if self.mode not in ('current', 'past', 'tentative', 'planned', 'retracted'):
            raise ValueError('Invalid assertion mode')
        if not self.value or not self.predicate or not self.owner or self.start < 0 or self.end <= self.start:
            raise ValueError('Incomplete evidence')
        if self.end - self.start != len(self.quote):
            raise ValueError('Quote offsets do not match quote length')

    @property
    def key(self):
        return (self.owner, self.predicate, self.scope)

    @property
    def identity(self):
        # Source IDs can encode annotations; opaque identities never encode
        # a label and source_id is never passed to model feature construction.
        value = f'{self.source_id}\0{self.turn}\0{self.start}\0{self.end}\0{self.owner}\0{self.predicate}\0{self.scope}'
        return 'e_' + hashlib.sha256(value.encode()).hexdigest()[:24]

    @property
    def payload_bytes(self):
        # UTF8 strings + observation float64 + 3 uint64 offsets/turn + mode byte.
        return sum(len(v.encode()) for v in (self.owner, self.predicate, self.scope, self.value, self.source_id, self.quote, self.identity)) + 33


@dataclass
class Version:
    evidence: Evidence
    sequence: int
    status: str
    known_until: float | None = None
    support: int = 1
    state: torch.Tensor = field(default_factory=lambda: torch.zeros(HIDDEN))
    equilibrium: torch.Tensor = field(default_factory=lambda: torch.zeros(HIDDEN))
    rate: torch.Tensor = field(default_factory=lambda: torch.zeros(HIDDEN))
    state_time: float = 0.
    priority_at_write: float = 0.


@dataclass
class KeyState:
    current_id: str | None = None
    # A missing current payload keeps this watermark while archives survive.
    changes: int = 0
    last_observed: float = 0.


class VersionLedger:
    """Causal ordered input. Inference writes should use torch.no_grad().

All controls reserve the same 96 state bytes/version in the main comparison;
actual tensor state is three float32[8] arrays. Rule policy uses the same reserve.
Key watermarks are charged. No unbounded evicted-content or key log is retained.
"""
    def __init__(self, budget_bytes=None, controller=None, policy='rule'):
        if policy not in ('rule', 'fifo', 'frozen_rule', 'learned'):
            raise ValueError('Invalid policy')
        if policy == 'learned' and controller is None:
            raise ValueError('Learned policy needs a controller')
        if budget_bytes is not None and budget_bytes < 16:
            raise ValueError('Budget must include global clock/sequence')
        self.budget_bytes = budget_bytes
        self.controller = controller
        self.policy = policy
        self.items = {}
        self.keys = {}
        self.now = None
        self.sequence = 0

    @property
    def payload_bytes(self):
        keys = sum(sum(len(part.encode()) for part in key) + 16 + len(state.current_id.encode() if state.current_id else b'') for key, state in self.keys.items())
        items = sum(v.evidence.payload_bytes + 8 + 8 + 8 + 1 + 8 + 8 + 3*HIDDEN*4 for v in self.items.values())
        return 16 + keys + items

    def features(self, version, event=None, now=None):
        fact = version.evidence
        now = self.now if now is None else now
        key_state = self.keys[fact.key]
        age = max(0., float(now) - fact.observed_at)
        same_key = event is not None and event.key == fact.key
        same_value = same_key and event.value.casefold() == fact.value.casefold()
        # Hash only semantic predicate, never source IDs or labels.
        predicate_hash = hashlib.sha256((fact.predicate + ':' + fact.scope).encode()).digest()
        values = [float(version.status == 'current'), float(version.status == 'superseded'),
                  float(fact.mode == 'past'), float(fact.mode in ('planned', 'tentative')),
                  math.log1p(version.support), math.log1p(key_state.changes), math.log1p(age)/6,
                  float(same_key), float(same_value), float(same_key and not same_value and event.mode == 'current'),
                  min(len(fact.quote)/256, 4.), min(len(fact.value)/64, 4.)]
        values += [(predicate_hash[i]/127.5)-1 for i in range(4)]
        return torch.tensor(values, dtype=torch.float32)

    def latent_at(self, version, now=None):
        if self.controller is None:
            return version.state
        now = self.now if now is None else now
        return self.controller.flow(version.state, version.equilibrium, version.rate, float(now)-version.state_time)

    def score(self, version):
        if self.policy == 'fifo':
            return torch.tensor(float(version.sequence))
        if self.policy == 'frozen_rule':
            return torch.tensor(version.priority_at_write)
        if self.policy == 'rule':
            f = self.features(version)
            return 2*f[0] + .2*f[4] - .3*f[6] - .5*f[3]
        return self.controller.score(self.latent_at(version), self.features(version))

    def scores(self):
        identities = list(self.items)
        if not identities:
            return {}
        if self.policy != 'learned':
            return {identity: self.score(self.items[identity]) for identity in identities}
        versions = [self.items[identity] for identity in identities]
        states = self.controller.flow(torch.stack([v.state for v in versions]),
            torch.stack([v.equilibrium for v in versions]), torch.stack([v.rate for v in versions]),
            torch.tensor([[self.now-v.state_time] for v in versions], dtype=torch.float32))
        scores = self.controller.score(states, torch.stack([self.features(v) for v in versions]))
        return dict(zip(identities, scores))

    def write(self, evidence):
        if self.now is not None and evidence.observed_at < self.now:
            raise ValueError('Input must be ordered by observation time; no silent reordering')
        if evidence.identity in self.items:
            old = self.items[evidence.identity].evidence
            if old != evidence:
                raise ValueError('Same provenance identity has conflicting content')
            return False
        self.now = evidence.observed_at
        self.sequence += 1
        key_state = self.keys.setdefault(evidence.key, KeyState())
        affected = self.affected(evidence)
        current = self.items.get(key_state.current_id)
        if evidence.mode == 'current':
            if current is not None:
                if current.evidence.value.casefold() != evidence.value.casefold():
                    key_state.changes += 1
                current.status = 'superseded'
                current.known_until = self.now
            # Watermark survives even if this new version is evicted below.
            key_state.current_id = evidence.identity
            key_state.last_observed = self.now
        elif evidence.mode == 'retracted' and current is not None and current.evidence.value.casefold() == evidence.value.casefold():
            current.status = 'retracted'
            current.known_until = self.now
            key_state.current_id = evidence.identity
            key_state.last_observed = self.now
            key_state.changes += 1
        version = Version(evidence, self.sequence, 'current' if evidence.mode == 'current' else evidence.mode, state_time=self.now)
        self.items[evidence.identity] = version
        for item in affected + [version]:
            if item is not version and item.evidence.key == evidence.key and item.evidence.value.casefold() == evidence.value.casefold():
                item.support += 1
                version.support = max(version.support, item.support)
        f = self.features(version)
        version.priority_at_write = float(2*f[0] + .2*f[4] - .3*f[6] - .5*f[3])
        if self.controller is not None:
            touched = affected + [version]
            states = self.controller.flow(torch.stack([v.state for v in touched]),
                torch.stack([v.equilibrium for v in touched]), torch.stack([v.rate for v in touched]),
                torch.tensor([[self.now-v.state_time] for v in touched], dtype=torch.float32))
            states, equilibria, rates = self.controller.event(states, torch.stack([self.features(v,evidence) for v in touched]))
            for index, item in enumerate(touched):
                item.state, item.equilibrium, item.rate = states[index], equilibria[index], rates[index]
                item.state_time = self.now
        self._evict()
        return True

    def affected(self, evidence):
        return [v for v in self.items.values() if v.evidence.key == evidence.key]

    def _evict(self):
        while self.budget_bytes is not None and self.payload_bytes > self.budget_bytes:
            if self.items:
                scores = self.scores()
                worst = min(self.items, key=lambda identity: (float(scores[identity].detach()), self.items[identity].sequence))
                key = self.items.pop(worst).evidence.key
                if not any(v.evidence.key == key for v in self.items.values()):
                    del self.keys[key]
            elif self.keys:
                self.keys.clear()
            else:
                break

    def current(self, key):
        state = self.keys.get(key)
        if state is None:
            return None
        item = self.items.get(state.current_id)
        return item.evidence if item is not None and item.status == 'current' else None

    def versions(self, key):
        return [v for v in sorted(self.items.values(), key=lambda item: item.sequence) if v.evidence.key == key]

    def known_as_of(self, key, observed_at):
        """Knowledge-time query, not inferred real-world validity or time travel."""
        if self.now is not None and observed_at > self.now:
            raise ValueError('Cannot assert a future state from current observations')
        candidates = [v for v in self.versions(key) if v.evidence.mode == 'current'
                      and v.evidence.observed_at <= observed_at
                      and (v.known_until is None or observed_at < v.known_until)]
        if not candidates:
            return None
        # If the last observation is missing, do not fall back to an older value.
        latest = max(candidates, key=lambda v: v.sequence)
        state = self.keys[key]
        if state.last_observed <= observed_at and state.current_id not in self.items:
            return None
        return latest.evidence

    def export_context(self, key=None):
        values = self.versions(key) if key is not None else sorted(self.items.values(), key=lambda item: item.sequence)
        # Include status; archived content must never be described as current.
        return '\n'.join(f'[{v.evidence.identity} status={v.status} observed_day={v.evidence.observed_at:.6f} scope={v.evidence.scope}] {v.evidence.quote}' for v in values)
