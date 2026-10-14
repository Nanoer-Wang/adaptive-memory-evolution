"""Small, auditable MemoryPolicy adapters; never read evaluator labels or truth."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import math
from typing import Protocol
import numpy as np

METHODS = ('fixed_decay', 'frequency', 'adaptive_evolution', 'adaptive_rule', 'immediate_context')


class MemoryPolicy(Protocol):
    def observe(self, event: dict) -> dict: ...
    def update_state(self, event: dict) -> None: ...
    def retrieve(self, query: dict) -> dict | None: ...
    def evict(self) -> dict: ...
    def feedback(self, item_id: str | None, value: float) -> None: ...


@dataclass
class Item:
    item_id: str
    key: int
    topic: int
    version: int
    content: str
    birth: float
    observed_at: float
    episode: int
    count: int = 1
    I: float = 1.
    C: float = 1.
    U: float = 0.
    T: float = 0.


class Policy:
    """Uniform storage/retrieval; only the retention policy and state flow differ.

    The candidate is eligible for rejection, so an overflow is not necessarily
    a resident replacement. Ledger metadata holds IDs only, never evicted text.
    """
    def __init__(self, method, capacity=8, seed=11, coefficients=None, rate=.3, decay=.001):
        if method not in METHODS or capacity < 1:
            raise ValueError('Invalid method or capacity')
        if method=='adaptive_evolution' and coefficients is None:
            raise ValueError('Explicit frozen coefficients required; no silent initialization/training')
        self.method, self.capacity, self.seed = method, capacity, seed
        self.coefficients = coefficients
        self.rate, self.decay = rate, decay
        self.items = {}; self.ledger = {}; self.now = 0.; self.focus = -1
        self.held_k, self.held_w, self.alpha = rate, 1.5, 2/3
        self.mean_feedback = self.last_feedback = .5
        self.demand = {}; self.last_query = {}; self.conflict_mean = 0.
        self.pending = None; self.serial = 0; self.lifetimes = []
        self.calls = dict(update_state=0, observe=0, retrieve=0, evict=0, feedback=0)
        self.counts = dict(item_state_updates=0, coefficient_calls=0, score_evaluations=0,
                           overflow_opportunities=0, resident_evictions=0, rejected_writes=0,
                           accepted_writes=0, duplicate_refreshes=0)
        self.slot_time = self.age_integral = 0.

    @property
    def adaptive(self):
        return self.method in ('adaptive_evolution', 'adaptive_rule')

    def bootstrap(self, records):
        if self.items or len(records) > self.capacity:
            raise ValueError('Initial snapshot must fit and bootstrap occurs once')
        for event in records:
            self._new_item(event, 1.)
            self.ledger[event['key']] = event['item_id']
        return self.snapshot()

    def _new_item(self, event, importance):
        self.serial += 1
        item = Item(event['item_id'], event['key'], event['topic'], event['version'],
                    event['content'], self.now, self.now, self.serial, I=importance)
        self.items[item.item_id] = item
        return item

    def update_state(self, event):
        self.calls['update_state'] += 1
        now = float(event['time']); dt = now-self.now
        if dt < 0 or not math.isfinite(now):
            raise ValueError('Non-monotonic/nonfinite time')
        self.age_integral += sum((self.now-i.birth)*dt + dt*dt/2 for i in self.items.values())
        self.slot_time += len(self.items)*dt
        if self.adaptive:
            for item in self.items.values():
                q = float(item.topic==self.focus)
                target = .5+self.held_w*q
                valid = float(self.ledger.get(item.key)==item.item_id)
                item.I = target+(item.I-target)*math.exp(-self.held_k*dt)
                item.C = valid+(item.C-valid)*math.exp(-4*self.held_k*dt)
                item.U *= math.exp(-self.held_k*dt/20)
                item.T += dt
            self.counts['item_state_updates'] += len(self.items)
        self.now = now
        for key in self.demand: self.demand[key] *= math.exp(-dt/30)
        key = event.get('key', -1)
        conflict = event['kind']=='observe' and key in self.ledger and self.ledger[key]!=event['item_id']
        if event['kind']=='observe': self.conflict_mean=.95*self.conflict_mean+.05*conflict
        if self.coefficients is not None:
            # Same 32 content + 8 causal features as the frozen head; exact topic
            # is observable, while stable role and current truth are not inputs.
            extra = [float(event['kind']=='observe')-float(event['kind']=='context'),
                     math.log1p(dt)/5, self.demand.get(key, 0.),
                     min((now-self.last_query.get(key, -1000.))/100, 4),
                     self.conflict_mean, self.mean_feedback, self.last_feedback,
                     float(event['topic']==self.focus)]
            x = np.asarray(event['vector']+extra, dtype=np.float32)
            raw = self.coefficients['weight'] @ x + self.coefficients['bias']
            self.alpha = float(1/(1+np.exp(-raw[0])))
            self.held_w = float(2/(1+np.exp(-raw[1])))
            self.held_k = float((np.logaddexp(0., raw[2])+1e-4)/30)
            self.counts['coefficient_calls'] += 1
        # Newly arrived context is held FORWARD, never integrated into the past.
        if event['kind']=='context': self.focus=event['topic']

    def observe(self, event):
        self.calls['observe'] += 1
        if self.pending is not None:
            raise RuntimeError('Previous candidate was not finalized by evict()')
        self.ledger[event['key']] = event['item_id']
        old = self.items.get(event['item_id'])
        initial = self.alpha*self.held_w if self.adaptive else 1.
        if old is not None:
            old.observed_at=self.now
            if self.adaptive: old.I=min(old.I+initial,20.); old.C=1.; old.T=0.
            self.counts['duplicate_refreshes'] += 1
            return dict(candidate=event['item_id'], duplicate=True)
        self._new_item(event, initial)
        self.pending=event['item_id']
        return dict(candidate=self.pending, duplicate=False)

    def score(self, item):
        if self.method=='fixed_decay': return -self.decay*(self.now-item.observed_at)
        if self.method=='frequency': return math.log(item.count)
        if self.method=='immediate_context':
            # No evolving latent: same current topic, observed consistency and
            # retrieval count, but no latent integration, confidence or forgetting.
            valid = float(self.ledger.get(item.key)==item.item_id)
            return math.log(.5+1.5*(item.topic==self.focus))+math.log(max(valid,.05))+math.log(item.count)
        return math.log(max(item.I,1e-8))+math.log(max(item.C,.05))+math.log1p(item.U)-.05*math.log1p(item.T)

    def _tie(self, item):
        return hashlib.sha256(f'{self.seed}:{item.item_id}'.encode()).hexdigest()

    def evict(self):
        self.calls['evict'] += 1
        result=dict(candidate=self.pending, victim=None, action='none', scores=None)
        if len(self.items)>self.capacity:
            self.counts['overflow_opportunities'] += 1
            self.counts['score_evaluations'] += len(self.items)
            scores={key:self.score(item) for key,item in self.items.items()}
            victim=min(self.items.values(),key=lambda i:(scores[i.item_id],self._tie(i)))
            rejected=victim.item_id==self.pending
            result.update(victim=victim.item_id, action='reject' if rejected else 'replace', scores=scores)
            if rejected:
                self.counts['rejected_writes'] += 1
            else:
                self.counts['resident_evictions'] += 1
                self.lifetimes.append(dict(item_id=victim.item_id, episode=victim.episode, key=victim.key,
                                           birth=victim.birth, end=self.now, lifetime=self.now-victim.birth,
                                           right_censored=False))
            del self.items[victim.item_id]
        if self.pending is not None and self.pending in self.items:
            self.counts['accepted_writes'] += 1
            if result['action']=='none': result['action']='admit'
        self.pending=None
        assert len(self.items)<=self.capacity
        return result

    def retrieve(self, query):
        self.calls['retrieve'] += 1
        self.demand[query['key']]=self.demand.get(query['key'],0.)+1
        self.last_query[query['key']]=self.now
        candidates=[i for i in self.items.values() if i.key==query['key']]
        if not candidates: return None
        item=max(candidates,key=lambda i:i.version)
        item.count += 1
        return dict(item_id=item.item_id, content=item.content, key=item.key)

    def feedback(self, item_id, value):
        self.calls['feedback'] += 1
        if not 0 <= value <= 1: raise ValueError('Feedback must be within [0,1]')
        item=self.items.get(item_id)
        if self.adaptive and item is not None:
            item.C=.8*item.C+.2*value; item.U+=value; item.T=0.
        self.mean_feedback=.9*self.mean_feedback+.1*value
        self.last_feedback=value

    def snapshot(self):
        return [asdict(i) for i in sorted(self.items.values(), key=lambda i:i.item_id)]

    def finish_lifetimes(self):
        return self.lifetimes+[dict(item_id=i.item_id, episode=i.episode, key=i.key,
                                   birth=i.birth, end=self.now, lifetime=self.now-i.birth,
                                   right_censored=True) for i in self.items.values()]


def frozen_head(checkpoint):
    """Load only inference parameters; no optimizer or training path."""
    import torch
    record=torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights=record['state_dict']
    result=dict(weight=weights['policy.weight'].numpy().copy(), bias=weights['policy.bias'].numpy().copy())
    assert result['weight'].shape==(3,40) and result['bias'].shape==(3,)
    return result
