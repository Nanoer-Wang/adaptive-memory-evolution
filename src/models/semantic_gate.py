"""Same full pair content and lexical signals; alternative gate mechanisms.

Each pair resets its state. Unit event steps have no physical-time meaning.
This tests semantic discrimination, NOT a claim about long-term liquid memory.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SemanticGate(nn.Module):
    def __init__(self,kind,width=64):
        super().__init__();self.kind=kind;self.width=width
        self.project=nn.Sequential(nn.Linear(384,width),nn.Tanh())
        if kind=='mlp':self.core=nn.Sequential(nn.Linear(width*4,width),nn.Tanh())
        elif kind=='gru':self.core=nn.GRUCell(width,width)
        elif kind in ('input_flow','ltc'):
            self.target=nn.Linear(width,width)
            self.rate=nn.Linear(width*(2 if kind=='ltc' else 1),width)
            self.leak=nn.Parameter(torch.zeros(width))
        else:raise ValueError('Unsupported gate')
        self.reader=nn.Sequential(nn.Linear(width*5+8,width),nn.Tanh(),nn.Linear(width,1))

    def transition(self,h,x):
        if self.kind=='gru':return self.core(x,h)
        target=torch.tanh(self.target(x));leak=F.softplus(self.leak)
        if self.kind=='input_flow':
            f=F.softplus(self.rate(x));r=leak+f;b=f*target/r
            return b+(h-b)*torch.exp(-r)
        for _ in range(4):
            f=F.softplus(self.rate(torch.cat((h,x),-1)))
            h=(h+.25*f*target)/(1+.25*(leak+f))
        return h

    def forward(self,pair,lexical):
        a,b=self.project(pair[:,0]),self.project(pair[:,1])
        joined=torch.cat((a,b,torch.abs(a-b),a*b),-1)
        if self.kind=='mlp':h=self.core(joined)
        else:
            h=self.transition(torch.zeros_like(a),a);h=self.transition(h,b)
        cos=(pair[:,0]*pair[:,1]).sum(-1,keepdim=True)
        return self.reader(torch.cat((joined,h,lexical,cos),-1)).squeeze(-1)


def wilson_upper(errors,n,z=1.96):
    if not n:return 1.
    p=errors/n
    return (p+z*z/(2*n)+z*((p*(1-p)/n+z*z/(4*n*n))**.5))/(1+z*z/n)


def calibrate(scores,labels,risk=.01):
    """Largest coverage among >=0.5 thresholds meeting a Wilson upper rule.

    This is a calibration heuristic, NOT a simultaneous or shift-robust risk
    guarantee. A reserved test set must independently check the achieved risk.
    """
    import numpy as np
    scores=np.asarray(scores);labels=np.asarray(labels);neg=labels==0
    thresholds=np.unique(np.r_[scores[scores>=.5],1.000001])
    feasible=[]
    for t in thresholds:
        skip=scores>=t;errors=int((skip&neg).sum());upper=wilson_upper(errors,int(neg.sum()))
        if upper<=risk:feasible.append((int(skip.sum()),float(t),errors,float(upper)))
    if not feasible:return dict(threshold=1.000001,skip_count=0,errors=0,wilson_upper=wilson_upper(0,int(neg.sum())),feasible=False)
    n,t,e,u=max(feasible,key=lambda v:(v[0],v[1]))
    return dict(threshold=t,skip_count=n,errors=e,wilson_upper=u,feasible=True)
