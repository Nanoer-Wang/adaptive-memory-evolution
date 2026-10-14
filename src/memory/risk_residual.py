"""Shared cheap content scores plus sparse, trajectory-trained residual memory.

Gate decisions use only observed content, elapsed time and a bounded call budget.
All writers share identical gates. Cheap scores are never copied from other text.
"""
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.memory.content_store import ContentWriter


def risk_features(vectors,deltas,cheap,scalar_history=False):
    previous=np.vstack([np.zeros_like(vectors[:1]),vectors[:-1]])
    similarity=np.sum(previous*vectors,axis=1)/(np.linalg.norm(previous,axis=1)*np.linalg.norm(vectors,axis=1)+1e-8)
    novelty=np.abs(np.diff(cheap,prepend=np.float32(0))) if scalar_history else 1-similarity
    return np.column_stack([vectors,cheap,np.abs(cheap),novelty,np.log1p(deltas),np.ones(len(vectors))]).astype(np.float32)


def call_mask(risk,threshold,policy='risk',rate=.5,capacity=2.):
    credit=1.;mask=[]
    for i,value in enumerate(risk):
        credit=min(capacity,credit+rate)
        wants=policy=='full' or (policy=='periodic' and i%2==0) or (policy=='risk' and value>=threshold)
        take=wants and (policy=='full' or credit>=1.)
        if take and policy!='full':credit-=1.
        mask.append(take)
    return np.asarray(mask,dtype=bool)


class StreamingRiskGate:
    """Algebraically fold feature normalization; no matrix allocations per event."""
    def __init__(self,shared):
        cheap=shared['cheap'].astype(np.float64)
        normalized=shared['risk'][:-1].astype(np.float64)/shared['std']
        intercept=float(shared['risk'][-1])-float(shared['mean']@normalized)
        self.projection=np.stack([cheap[:-1],normalized[:384]+normalized[384]*cheap[:-1]])
        self.cheap_bias=cheap[-1];self.risk_bias=intercept+normalized[384]*cheap[-1]
        self.abs_weight,self.novelty_weight,self.time_weight=normalized[385:388]
        self.threshold=float(shared['threshold']);self.previous=None;self.credit=1.
        self.scalar_history=bool(shared['scalar_history']) if 'scalar_history' in shared else False
        self.previous_score=0.

    def observe(self,vector,delta):
        projected=self.projection@vector;cheap=float(projected[0]+self.cheap_bias)
        if self.scalar_history:novelty=abs(cheap-self.previous_score)
        else:
            norm=float(np.linalg.norm(vector));similarity=0. if self.previous is None else float(self.previous@vector)/(float(np.linalg.norm(self.previous))*norm+1e-8)
            novelty=1-similarity
        risk=float(projected[1]+self.risk_bias+self.abs_weight*abs(cheap)+self.novelty_weight*novelty+self.time_weight*math.log1p(float(delta)))
        self.credit=min(2.,self.credit+.5);take=risk>=self.threshold and self.credit>=1.
        if take:self.credit-=1.
        if self.scalar_history:self.previous_score=cheap
        else:self.previous=vector.copy()
        return cheap,risk,take


class RiskResidualWriter(nn.Module):
    """Cheap content + state residual, with common continuous decay between reads.

    Time decay is given to GRU/static too, so access to time is not a liquid advantage.
    This is a gated recurrence with exponential decay, not an exact ODE solver.
    """
    def __init__(self,kind):
        super().__init__();self.core=ContentWriter(kind+'_content');self.hidden=self.core.hidden
        self.decay=nn.Parameter(torch.full((self.hidden,),-2.25))
        nn.init.zeros_(self.core.head.weight);nn.init.zeros_(self.core.head.bias)

    def forward(self,vectors,deltas,cheap,mask,reset=False,zero_time=False):
        state=vectors.new_zeros(self.hidden);scores=[]
        for x,dt,base,take in zip(vectors,deltas,cheap,mask):
            if not zero_time:state=state*torch.exp(-F.softplus(self.decay)*dt)
            if bool(take):
                if reset:state=torch.zeros_like(state)
                _,state=self.core.step(x,state,0.)
            # Both selected and skipped records use the SAME scoring scale.
            scores.append(base+self.core.head(state).squeeze(-1))
        return torch.stack(scores)
