"""External evidence storage controlled by learned continuous-time consolidation.

The latent state contains control context, never retrievable factual payloads.
Time flow is exact for piecewise-constant coefficients; this is not canonical LTC.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.memory.evidence_dynamics import exact_flow

FEATURES = 40
HIDDEN = 8
KINDS = ('liquid', 'gru', 'static', 'linear', 'no_forgetting', 'no_importance', 'no_time')
PROJECTION = np.random.default_rng(20261013).normal(size=(384, 32)).astype(np.float32)


def project(vector):
    v = np.asarray(vector, dtype=np.float32)
    if v.shape == (384,): v = v @ PROJECTION
    if v.shape != (32,) or not np.isfinite(v).all(): raise ValueError('Expected finite32/384 vector')
    return v / max(float(np.linalg.norm(v)), 1e-8)


class PrefixFeatures:
    """Past-only features, independent of storage admissions: same train/run process."""
    def __init__(self):
        self.previous = np.zeros(32, dtype=np.float32)
        self.first = None; self.last = None; self.count = 0; self.mean_gap = 0.

    def observe(self, text, vector, now):
        if not math.isfinite(now) or (self.last is not None and now < self.last): raise ValueError('Noncausal time')
        vector = project(vector)
        dt = 0. if self.last is None else now-self.last
        self.first = now if self.first is None else self.first
        self.count += 1; self.mean_gap += (dt-self.mean_gap)/self.count
        extra = [math.log1p(dt)/5, min(len(text.encode())/2400,4),
                 sum(c.isdigit() for c in text)/max(len(text),1), float(text.startswith('user:')),
                 float(vector@self.previous), math.log1p(self.count)/6,
                 math.log1p(self.mean_gap)/5, math.log1p(now-self.first)/6]
        self.previous = vector.copy(); self.last = now
        return np.concatenate((vector, np.asarray(extra,dtype=np.float32))), dt


class ConsolidationController(nn.Module):
    def __init__(self, kind='liquid'):
        super().__init__()
        if kind not in KINDS: raise ValueError(kind)
        self.kind = kind
        self.beta_override = None  # Explicit same-weight diagnostic; disabled in primary runs.
        self.encode = nn.Sequential(nn.Linear(FEATURES,HIDDEN),nn.Tanh())
        self.impulse = nn.Linear(HIDDEN,HIDDEN)
        self.gate = nn.Linear(HIDDEN,HIDDEN)
        self.equilibrium = nn.Linear(HIDDEN,HIDDEN)
        self.rate = nn.Linear(HIDDEN,HIDDEN)
        self.gru = nn.GRUCell(HIDDEN,HIDDEN) if kind=='gru' else None
        self.policy = nn.Linear(FEATURES+(0 if kind=='linear' else HIDDEN),3)

    def initial(self, batch=None):
        shape=(HIDDEN,) if batch is None else (batch,HIDDEN)
        z=next(self.parameters()).new_zeros(shape)
        return z,z.clone(),z.clone()

    def step(self, x, dt, state):
        h,e,r=state;elapsed=torch.as_tensor(dt,dtype=x.dtype,device=x.device)
        if not bool(torch.isfinite(elapsed).all()) or bool((elapsed<0).any()):raise ValueError('Invalid elapsed time')
        if elapsed.ndim:elapsed=elapsed.unsqueeze(-1)
        if self.kind not in ('linear','static','gru','no_time'):h=exact_flow(h,e,r,elapsed)
        encoded=self.encode(x)
        if self.kind in ('linear','static'):h=encoded
        elif self.kind=='gru':h=self.gru(encoded,h)
        else:h=h+torch.sigmoid(self.gate(encoded))*(torch.tanh(self.impulse(encoded))-h)
        e=torch.tanh(self.equilibrium(encoded));r=(F.softplus(self.rate(encoded))+1e-4)/30.
        logits=self.policy(x if self.kind=='linear' else torch.cat((h,x),-1))
        alpha=torch.sigmoid(logits[...,0]);w=2*torch.sigmoid(logits[...,1]);beta=(F.softplus(logits[...,2])+1e-4)/30.
        if self.kind=='no_importance':w=torch.ones_like(w)
        if self.kind=='no_forgetting':beta=torch.zeros_like(beta)
        if self.beta_override is not None:
            if not math.isfinite(self.beta_override) or self.beta_override<0:raise ValueError('Invalid forgetting rate')
            beta=torch.full_like(beta,float(self.beta_override))
        return dict(alpha=alpha,w=w,beta=beta,logits=logits),(h,e,r)


@dataclass
class ConsolidatedRecord:
    identity: str
    text: str
    vector: list
    key: str | None
    value: str | None
    origins: list
    strength: float
    beta: float
    alpha: float
    importance: float
    updated_at: float
    sequence: int

    def at(self,now):return self.strength*math.exp(-self.beta*max(0.,now-self.updated_at))

    @property
    def nbytes(self):
        # Defined packed payload accounting, NOT JSON file size or Python RSS.
        strings=[self.identity,self.text,self.key or '',self.value or '']+[o[0] for o in self.origins]
        return sum(len(s.encode()) for s in strings)+32*4+6*8+len(self.origins)*8


class ConsolidationMemory:
    """Causal observe/retrieve/feedback API. Version keys require trusted extraction.

    Similarity never establishes truth or supersession. Exact text repeats merge;
    all source IDs/times remain charged. Model parameters are shared, not per-store.
    """
    reserve_bytes=512  # Same allowance for latent/prefix arrays, clocks and counters in all controls.

    def __init__(self,controller=None,budget_bytes=4096,rule='fifo',admission='competitive'):
        if budget_bytes is not None and budget_bytes<self.reserve_bytes:raise ValueError('Insufficient state budget')
        if admission not in ('competitive','threshold'):raise ValueError('Invalid admission policy')
        self.admission=admission
        self.controller=controller;self.budget_bytes=budget_bytes;self.rule=rule
        self.state=controller.initial() if controller else None;self.prefix=PrefixFeatures()
        self.items={};self.current={};self.now=None;self.sequence=0
        self.stats=dict(observed=0,writes=0,rejected=0,merged=0,evicted=0,feedback=0,controller_calls=0)

    @property
    def payload_bytes(self):
        return self.reserve_bytes+sum(r.nbytes for r in self.items.values())+sum(len(k.encode())+len(v.encode()) for k,v in self.current.items())

    def _check_time(self,now):
        if not math.isfinite(now) or (self.now is not None and now<self.now):raise ValueError('Noncausal time')

    def observe(self,source_id,text,vector,now,key=None,value=None):
        self._check_time(now)
        if (key is None)!=(value is None):raise ValueError('Key/value must be supplied together')
        if not text or not source_id:raise ValueError('Empty evidence')
        key=None if key is None else str(key);source_id=str(source_id)
        for old in self.items.values():
            if any(o[0]==source_id for o in old.origins):
                if old.text!=text or old.key!=key or old.value!=value:raise ValueError('Conflicting source reuse')
                return dict(action='replay',payload_bytes=self.payload_bytes)
        x,dt=self.prefix.observe(text,vector,now);vector=self.prefix.previous.copy();self.now=now;self.sequence+=1;self.stats['observed']+=1
        if self.controller:
            with torch.no_grad():policy,self.state=self.controller.step(torch.tensor(x),dt,self.state)
            a,w,b=[float(policy[k]) for k in ('alpha','w','beta')];self.stats['controller_calls']+=1
        else:a,w,b=1.,1.,(1/30 if self.rule=='decay' else 0.)
        identity=hashlib.sha256(((key or '')+'\0'+text).encode()).hexdigest()[:24]
        if key is not None:self.current[key]=identity  # Includes rejected-new-version tombstones.
        action='reject'
        if identity in self.items:
            item=self.items[identity];item.strength=min(20.,item.at(now)+a*w);item.updated_at=now
            item.origins.append((source_id,now));item.beta=b;item.alpha=a;item.importance=w;item.sequence=self.sequence
            self.stats['merged']+=1;action='merge'
        elif self.admission=='competitive' or a>=.5:
            self.items[identity]=ConsolidatedRecord(identity,text,vector.tolist(),key,value,[(source_id,now)],a*w,b,a,w,now,self.sequence)
            self.stats['writes']+=1;action='write'
        else:self.stats['rejected']+=1
        removed=self._enforce()
        if action=='write' and identity not in self.items:self.stats['rejected']+=1
        return dict(action=action,alpha=a,importance=w,beta=b,evicted=removed,payload_bytes=self.payload_bytes)

    def _prune_index(self):
        live_keys={r.key for r in self.items.values() if r.key is not None}
        self.current={k:v for k,v in self.current.items() if k in live_keys}

    def _enforce(self):
        removed=[];self._prune_index()
        while self.budget_bytes is not None and self.payload_bytes>self.budget_bytes and self.items:
            def score(r):
                if self.controller is None and self.rule=='fifo':return (float(r.sequence),r.sequence)
                # Shared explicit version semantics, not a model truth prediction.
                current=r.key is None or self.current.get(r.key)==r.identity
                return (r.at(self.now)*(1. if current else .1),r.sequence)
            worst=min(self.items.values(),key=score);removed.append(worst.identity);del self.items[worst.identity];self.stats['evicted']+=1;self._prune_index()
        assert self.budget_bytes is None or self.payload_bytes<=self.budget_bytes
        return removed

    def retrieve(self,now,vector=None,key=None,limit=5):
        self._check_time(now);self.now=now
        if key is not None:
            identity=self.current.get(str(key));return [self.items[identity]] if identity in self.items else []
        if vector is None:raise ValueError('Provide a query vector or a trusted key')
        q=project(vector)
        eligible=[r for r in self.items.values() if r.key is None or self.current.get(r.key)==r.identity]
        return sorted(eligible,key=lambda r:(-float(np.dot(r.vector,q)),-r.sequence))[:limit]

    def reinforce(self,identities,now,reward=1.):
        self._check_time(now)
        if not math.isfinite(reward) or not 0<=reward<=1:raise ValueError('Feedback must be in [0,1]')
        self.now=now
        for identity in set(identities):
            if identity in self.items:
                r=self.items[identity];r.strength=min(20.,r.at(now)+r.alpha*r.importance*reward);r.updated_at=now;self.stats['feedback']+=1

    def checkpoint(self,path):
        signature=self.signature()
        value=dict(version=1,signature=signature,budget=self.budget_bytes,rule=self.rule,admission=self.admission,now=self.now,sequence=self.sequence,stats=self.stats,
                   prefix={**vars(self.prefix),'previous':self.prefix.previous.tolist()},state=None if self.state is None else [s.tolist() for s in self.state],
                   current=self.current,items=[asdict(r) for r in self.items.values()])
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');tmp.replace(p)

    def signature(self):
        if self.controller is None:return self.rule
        h=hashlib.sha256(self.controller.kind.encode())
        if self.controller.beta_override is not None:h.update(str(self.controller.beta_override).encode())
        for k,v in self.controller.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def restore(self,path):
        v=json.loads(Path(path).read_text(encoding='utf-8'))
        if v['version']!=1 or v['signature']!=self.signature() or v['budget']!=self.budget_bytes or v['admission']!=self.admission:raise ValueError('Incompatible checkpoint')
        candidate=ConsolidationMemory(self.controller,self.budget_bytes,self.rule,self.admission)
        candidate.now=v['now'];candidate.sequence=v['sequence'];candidate.stats=v['stats'];candidate.current=v['current']
        candidate.items={r['identity']:ConsolidatedRecord(**r) for r in v['items']}
        candidate.prefix.__dict__.update(v['prefix']);candidate.prefix.previous=np.asarray(v['prefix']['previous'],dtype=np.float32)
        candidate.state=None if v['state'] is None else tuple(torch.tensor(s,dtype=torch.float32) for s in v['state'])
        if self.budget_bytes is not None and candidate.payload_bytes>self.budget_bytes:raise ValueError('Oversized checkpoint')
        self.__dict__.update(candidate.__dict__)
