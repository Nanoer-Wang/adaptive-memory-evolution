import numpy as np
import torch
import pytest
from src.memory.risk_residual import risk_features,call_mask,RiskResidualWriter,StreamingRiskGate


def test_gate_causal_budget_bound_and_no_future_features():
    rng=np.random.default_rng(43);v=rng.normal(size=(100,384)).astype(np.float32);dt=rng.random(100);cheap=rng.normal(size=100)
    x=risk_features(v,dt,cheap);vv=v.copy();vv[50:]+=10
    assert np.array_equal(x[:50],risk_features(vv,dt,cheap)[:50])
    risks=rng.normal(size=100);a=call_mask(risks,0);risks[50:]+=100
    assert np.array_equal(a[:50],call_mask(risks,0)[:50])
    assert np.all(np.cumsum(a)<=np.floor(1+.5*np.arange(1,101)))
    assert np.all(np.cumsum(call_mask(np.ones(100),0))<=np.floor(1+.5*np.arange(1,101)))


@pytest.mark.parametrize('kind',['liquid','gru','static'])
def test_same_sparse_training_rollout_and_calls(kind):
    torch.manual_seed(43);m=RiskResidualWriter(kind);v=torch.randn(8,384);dt=torch.rand(8);base=torch.randn(8);mask=np.array([1,0,0,1,0,1,0,0],dtype=bool)
    calls=[];hook=m.core.encoder.register_forward_hook(lambda *args:calls.append(1))
    m.train();a=m(v,dt,base,mask);assert len(calls)==3
    # Zero head starts from common cheap scorer, including all skipped observations.
    assert torch.equal(a,base);a.square().mean().backward();assert m.core.head.weight.grad.abs().sum()>0
    with torch.no_grad():m.core.head.weight.fill_(.2)
    calls.clear();m.eval();b=m(v,dt,base,mask);assert len(calls)==3
    m.train();c=m(v,dt,base,mask);assert torch.equal(b,c)
    modified=v.clone();modified[5:]+=10
    assert torch.equal(m(modified,dt,base,mask)[:5],c[:5]);hook.remove()


def test_skipped_events_keep_their_own_cheap_scores():
    torch.manual_seed(43);m=RiskResidualWriter('liquid');v=torch.randn(5,384);dt=torch.zeros(5);mask=np.array([1,0,0,0,0],dtype=bool)
    with torch.no_grad():m.core.head.weight.fill_(.5)
    a=m(v,dt,torch.zeros(5),mask);base=torch.arange(5).float();b=m(v,dt,base,mask)
    assert torch.allclose(b-a,base,atol=1e-6)


@pytest.mark.parametrize('scalar',[False,True])
def test_folded_live_gate_matches_offline_and_keeps_bound(scalar):
    rng=np.random.default_rng(43);v=rng.normal(size=(30,384)).astype(np.float32);dt=rng.random(30).astype(np.float32)
    shared=dict(cheap=rng.normal(size=385).astype(np.float32),risk=(rng.normal(size=389)*.1).astype(np.float32),mean=rng.normal(size=388).astype(np.float32),std=(rng.random(388)+.5).astype(np.float32),threshold=np.array(.1),scalar_history=np.array(scalar))
    base=v@shared['cheap'][:-1]+shared['cheap'][-1];z=risk_features(v,dt,base,scalar);z[:,:-1]=(z[:,:-1]-shared['mean'])/shared['std'];expected=z@shared['risk'];gate=StreamingRiskGate(shared);observed=[gate.observe(x,d) for x,d in zip(v,dt)]
    assert np.allclose([r[1] for r in observed],expected,atol=2e-5)
    assert np.array_equal([r[2] for r in observed],call_mask(expected,.1))
    if scalar:assert gate.previous is None
