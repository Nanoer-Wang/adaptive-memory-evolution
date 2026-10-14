import numpy as np
import torch
from src.memory.liquid_belief import LiquidBelief,INPUTS


def event(allowed,kind=0):
    value=torch.zeros(1,INPUTS);value[0,:6]=torch.tensor(allowed);value[0,6+kind]=1;value[0,9]=1;value[0,10]=.6
    return value


def test_signed_evidence_revises_belief_without_rewriting_archive():
    model=LiquidBelief();h,p,c=model.initial(1)
    old=event([1.,0,0,0,0,0]);saved=old.clone()
    h,p,c=model.observe(h,p,old,torch.tensor([True]))
    _,confirmed,_=model.observe(h,p,old,torch.tensor([False]))
    _,denied,_=model.observe(h,p,event([0.,1,1,1,1,1],1),torch.tensor([False]))
    assert confirmed[0,0]>.999 and denied[0,0]==0
    assert torch.equal(old,saved)
    assert torch.allclose(denied.sum(-1),torch.ones(1))


def test_ltc_state_coupling_zero_time_and_probability_invariants():
    torch.manual_seed(7);m=LiquidBelief();h,p,c=m.initial(3);h.uniform_(-1,1);c.uniform_(-1,1)
    h0,p0,_=m.advance(h,p,c,torch.zeros(3),torch.full((3,),.5))
    assert torch.equal(h,h0) and torch.equal(p,p0)
    h1,p1,_=m.advance(h,p,c,torch.tensor([.1,3.,100.]),torch.full((3,),.5))
    assert h1.abs().max()<=1 and (p1>=0).all()
    assert torch.allclose(p1.sum(-1),torch.ones(3),atol=1e-6)
    def rates(x):return torch.nn.functional.softplus(m.conductance(torch.cat((x,c),-1)))
    assert not torch.equal(rates(h),rates(h+.2))


def test_complete_observation_silence_preserves_fact_belief():
    m=LiquidBelief();h,p,c=m.initial(1);p[0]=torch.tensor([1.,0,0,0,0,0])
    _,later,_=m.advance(h,p,c,torch.tensor([100.]),torch.ones(1))
    assert torch.equal(p,later)


def test_equal_scalar_similarity_does_not_collapse_new_event_semantics():
    torch.manual_seed(9);model=LiquidBelief();h,p,c=model.initial(1)
    first=event([.8,.2,0,0,0,0]);second=event([.8,0,.2,0,0,0])
    old=torch.tensor([1.,0,0,0,0,0])
    assert torch.dot(first[0,:6],old)==torch.dot(second[0,:6],old)
    ha,pa,_=model.observe(h,p,first,torch.tensor([True]))
    hb,pb,_=model.observe(h,p,second,torch.tensor([True]))
    assert not torch.equal(ha,hb) and not torch.equal(pa,pb)


def test_integration_refines_and_gradients_reach_state_coupling():
    torch.manual_seed(5);m=LiquidBelief();h,p,c=m.initial(1);h.uniform_(-.3,.3);c.uniform_(-.2,.2)
    args=(h,p,c,torch.tensor([5.]),torch.tensor([.5]))
    a=m.advance(*args,steps=8)[0];b=m.advance(*args,steps=32)[0];ref=m.advance(*args,steps=256)[0]
    assert torch.linalg.vector_norm(b-ref)<torch.linalg.vector_norm(a-ref)
    m.advance(*args)[1][0,0].backward()
    assert m.conductance.weight.grad is not None and m.conductance.weight.grad[:,:m.hidden].abs().sum()>0
