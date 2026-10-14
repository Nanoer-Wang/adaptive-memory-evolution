"""Preserve directional NLI evidence; history may veto, never override guards."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def protected_decision(forward,backward,role_equal,history_score,threshold,seen=False):
    f=np.asarray(forward);b=np.asarray(backward)
    if not np.isfinite(f).all() or not np.isfinite(b).all():return False
    if seen:return True  # Exact text previously retained or semantically skipped.
    eligible=min(f[1],b[1])>=.9 and max(f[0],b[0])<.1 and role_equal>=.5
    return bool(eligible and history_score>=threshold)

class HistoryGate(nn.Module):
    def __init__(self,kind,width=16):
        super().__init__();self.kind=kind;self.width=width
        self.project=nn.Linear(8,width)
        if kind=='gru':self.cell=nn.GRUCell(width,width)
        elif kind=='liquid':
            self.rate=nn.Linear(width*2,width);self.target=nn.Linear(width,width);self.leak=nn.Parameter(torch.zeros(width))
        elif kind!='static':raise ValueError(kind)
        self.read=nn.Linear(width+8,1)

    def forward(self,x,reset=False,unit_time=False):
        h=x.new_zeros((len(x),self.width));out=[]
        for i in range(x.shape[1]):
            z=torch.tanh(self.project(x[:,i]));dt=x[:,i,7:8]
            if reset:h=torch.zeros_like(h)
            if unit_time:dt=torch.ones_like(dt)
            if self.kind=='static':h=z
            elif self.kind=='gru':h=self.cell(z,h)
            else:
                target=torch.tanh(self.target(z));leak=F.softplus(self.leak)
                for _ in range(4):
                    rate=F.softplus(self.rate(torch.cat((h,z),-1)))
                    h=(h+dt/4*rate*target)/(1+dt/4*(leak+rate))
            out.append(self.read(torch.cat((h,x[:,i]),-1)).squeeze(-1))
        return torch.stack(out,1)
