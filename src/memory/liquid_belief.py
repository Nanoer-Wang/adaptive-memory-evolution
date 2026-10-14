"""Hybrid observation update and state-coupled LTC belief propagation.

The state is a belief about CURRENT location, not a compressed fact archive.
Explicit movement reports replace the observation distribution. Constraint
reports condition it. Evidence is immutable outside this controller.
Integration is a fused semi-implicit approximation, NOT an exact LTC solution.
"""
import torch
from torch import nn
from torch.nn import functional as F

LOCATIONS=6
INPUTS=12  # full likelihood6, relation3, reliability, capture rate, last log gap
KINDS=('static','input_decay','time_gru','ltc','no_hflow')


class LiquidBelief(nn.Module):
    def __init__(self,kind='ltc',hidden=16,steps=4):
        super().__init__()
        if kind not in KINDS:raise ValueError('Unknown belief mechanism')
        self.kind=kind;self.hidden=hidden;self.steps=steps
        self.register_buffer('prior',torch.full((LOCATIONS,),1/LOCATIONS))
        self.jump=nn.Linear(hidden+INPUTS,2*hidden)
        self.gru=nn.GRUCell(INPUTS,hidden) if kind=='time_gru' else None
        self.conductance=nn.Linear(hidden+INPUTS,hidden)
        self.leak=nn.Parameter(torch.zeros(hidden))
        self.attractor=nn.Parameter(torch.linspace(-.5,.5,hidden))
        self.rate=nn.Linear(hidden+INPUTS,1)
        self.destination=nn.Linear(hidden+INPUTS,LOCATIONS)

    def initial(self,batch,device=None):
        return (torch.zeros(batch,self.hidden,device=device),
                self.prior.to(device=device).expand(batch,-1).clone(),
                torch.zeros(batch,INPUTS,device=device))

    def vector_field(self,h,c):
        joined=torch.cat((h if self.kind!='input_decay' else torch.zeros_like(h),c),-1)
        conductance=F.softplus(self.conductance(joined))/4
        leak=F.softplus(self.leak)/4+1e-5
        return -(leak+conductance)*h+conductance*torch.tanh(self.attractor)

    def advance(self,h,p,c,elapsed,capture,steps=None):
        if not torch.isfinite(elapsed).all() or (elapsed<0).any():raise ValueError('Invalid elapsed time')
        if not torch.isfinite(capture).all() or (capture<0).any() or (capture>1).any():raise ValueError('Invalid capture rate')
        count=steps or self.steps;dt=elapsed.reshape(-1,1)/count
        for _ in range(count):
            if self.kind in ('ltc','input_decay'):
                joined=torch.cat((h if self.kind=='ltc' else torch.zeros_like(h),c),-1)
                f=F.softplus(self.conductance(joined))/4
                leak=F.softplus(self.leak)/4+1e-5
                h=(h+dt*f*torch.tanh(self.attractor))/(1+dt*(leak+f))
            joined=torch.cat((h,c),-1)
            hazard=F.softplus(self.rate(joined))/4
            destination=torch.softmax(self.destination(joined),-1)
            # Known channel capture is supplied equally to all methods.
            # With complete observation, silence cannot hide a movement.
            alpha=-torch.expm1(-hazard*dt*(1-capture.reshape(-1,1)))
            p=p+alpha*(destination-p)
        return h,p,c

    def observe(self,h,p,event,replace):
        likelihood=event[:,:LOCATIONS]
        if (likelihood<0).any() or not torch.isfinite(likelihood).all() or (likelihood.sum(-1)<=0).any():
            raise ValueError('Invalid evidence likelihood')
        # A small prior floor allows revising an earlier zero-probability belief.
        prior=(1-1e-6)*p+1e-6/LOCATIONS
        weighted=torch.where(replace.reshape(-1,1),likelihood,prior*likelihood)
        p=weighted/weighted.sum(-1,keepdim=True)
        if self.kind=='time_gru':h=self.gru(event,h)
        else:
            previous=torch.zeros_like(h) if self.kind=='static' else h
            gate,target=self.jump(torch.cat((previous,event),-1)).chunk(2,-1)
            target=torch.tanh(target)
            h=target if self.kind=='static' else h+torch.sigmoid(gate)*(target-h)
        return h,p,event

    def forward(self,events,times,valid,query_time,capture,steps=None):
        h,p,c=self.initial(len(events),events.device)
        c[:,10]=capture
        now=torch.zeros_like(query_time)
        for i in range(events.shape[1]):
            mask=valid[:,i];delta=torch.where(mask,times[:,i]-now,torch.zeros_like(now))
            h,p,c=self.advance(h,p,c,delta,capture,steps)
            hn,pn,cn=self.observe(h,p,events[:,i],torch.ones_like(mask))
            h=torch.where(mask[:,None],hn,h);p=torch.where(mask[:,None],pn,p);c=torch.where(mask[:,None],cn,c)
            now=torch.where(mask,times[:,i],now)
        h,p,c=self.advance(h,p,c,query_time-now,capture,steps)
        return p
