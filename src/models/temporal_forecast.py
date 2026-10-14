"""Shared causal numeric observations; real elapsed hours, no future covariates."""
import torch
from torch import nn
from torch.nn import functional as F

class TemporalForecast(nn.Module):
    def __init__(self,kind,features=11,width=16):
        super().__init__();self.kind=kind;self.width=width
        self.project=nn.Linear(features+1,width)
        if kind=='gru':self.cell=nn.GRUCell(width,width)
        elif kind in ('liquid','input_flow'):
            self.rate=nn.Linear(width*2 if kind=='liquid' else width,width)
            self.target=nn.Linear(width,width);self.leak=nn.Parameter(torch.zeros(width))
        elif kind!='last_mlp':raise ValueError(kind)
        self.read=nn.Linear(width+features+1,1)

    def forward(self,x,valid,dt,gap,reset=False,unit_time=False):
        h=x.new_zeros(len(x),self.width);last=x.new_zeros(len(x),x.shape[-1])
        for i in range(x.shape[1]):
            time=dt[:,i:i+1]/6
            if unit_time:time=torch.ones_like(time)/6
            z=torch.tanh(self.project(torch.cat((x[:,i],time),-1)))
            previous=torch.zeros_like(h) if reset else h
            if self.kind=='last_mlp':new=z
            elif self.kind=='gru':new=self.cell(z,previous)
            else:
                target=torch.tanh(self.target(z));leak=F.softplus(self.leak);new=previous
                for _ in range(4):
                    rate=F.softplus(self.rate(torch.cat((new,z),-1) if self.kind=='liquid' else z))
                    new=(new+time/4*rate*target)/(1+time/4*(rate+leak))
            h=torch.where(valid[:,i:i+1],new,h);last=torch.where(valid[:,i:i+1],x[:,i],last)
        # Residual forecast anchored on the most recent observed pollution.
        return last[:,0]+self.read(torch.cat((h,last,gap[:,None]/6),-1)).squeeze(-1)
