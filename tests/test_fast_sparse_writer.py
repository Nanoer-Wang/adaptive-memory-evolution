import copy
import numpy as np
import torch
import pytest
from src.memory.content_store import ContentWriter
from src.memory.fast_writer import frozen_writer
from src.memory.sparse_writer import sparse_scores
from src.memory.fast_writer import OnlineFrozenWriter
from src.memory.budget_store import BudgetStore


@pytest.mark.parametrize('kind', ['liquid_content','static_content','gru_content'])
def test_fused_scripted_batch_and_online_match_reference(kind):
    torch.manual_seed(81)
    model=ContentWriter(kind).eval(); fast=frozen_writer(copy.deepcopy(model))
    v=torch.randn(9,3,384);d=torch.rand(9,3)*12
    expected=torch.stack([model(v[:,i],d[:,i]) for i in range(3)],dim=1)
    assert torch.allclose(fast(v,d),expected,atol=2e-6,rtol=2e-6)
    state=torch.zeros(3,model.hidden);scores=[]
    for x,dt in zip(v,d):
        score,state=fast.step(x,state,dt);scores.append(score)
    assert torch.allclose(torch.stack(scores),expected,atol=2e-6,rtol=2e-6)
    changed=v.clone();changed[5:]+=10
    assert torch.allclose(fast(changed,d)[:5],expected[:5],atol=2e-6,rtol=2e-6)


def test_sparse_full_equivalence_causality_and_call_budget():
    torch.manual_seed(81);model=ContentWriter().eval()
    v=torch.randn(9,384);d=torch.rand(9)
    score,calls=sparse_scores(model,v,d,'full')
    assert torch.allclose(score,model(v,d));assert len(calls)==9
    for policy,expected in [('stride2',[0,2,4,6,8]),('stride4',[0,4,8])]:
        a,c=sparse_scores(model,v,d,policy);assert c.tolist()==expected
        altered=v.clone();altered[5:]+=10
        b,_=sparse_scores(model,altered,d,policy)
        assert torch.equal(a[:5],b[:5])


def test_sparse_accumulates_skipped_elapsed_time():
    class Recorder:
        hidden=1
        def __init__(self):self.gaps=[]
        def step(self,x,state,delta):
            self.gaps.append(delta);return x[0],state
    model=Recorder();sparse_scores(model,torch.ones(5,2),torch.tensor([0.,1.,2.,3.,4.]),'stride2')
    assert model.gaps==[0.,3.,7.]


@pytest.mark.parametrize('kind',['liquid_content','gru_content','static_content'])
def test_online_adapter_keeps_budget_store_contract(kind):
    torch.manual_seed(81);model=ContentWriter(kind).eval()
    a=BudgetStore(7000,model);b=BudgetStore(7000,OnlineFrozenWriter(model))
    for i in range(12):
        vector=np.random.default_rng(i).standard_normal(384).astype(np.float32)
        for store in [a,b]:store.insert(str(i),'2026-10-12','Public evidence '+str(i),vector,float(i)/2)
        assert [x.session_id for x in a.items]==[x.session_id for x in b.items]
        assert a.payload_bytes==b.payload_bytes
        assert torch.allclose(a.state,b.state,atol=2e-6,rtol=2e-6)
    assert all(p.requires_grad for p in model.parameters())


def test_content_fallback_scores_skipped_events_from_their_own_content():
    from scripts.content_fallback import fallback_scores
    class Recorder:
        hidden=1
        def __init__(self):self.gaps=[]
        def step(self,x,state,delta):
            self.gaps.append(delta);return x[0]+100,state
    v=torch.tensor([[1.,0.],[2.,0.],[3.,0.],[4.,0.]])
    d=torch.tensor([0.,1.,2.,3.]);weights=np.array([2.,0.,1.],dtype=np.float32)
    model=Recorder();out,calls=fallback_scores(model,v,d,weights,True)
    assert out.tolist()==[101.,5.,103.,9.];assert calls==2;assert model.gaps==[0.,3.]
    model=Recorder();out,calls=fallback_scores(model,v,d,weights,False)
    assert out.tolist()==[3.,5.,7.,9.];assert calls==0;assert not model.gaps
