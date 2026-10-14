import numpy as np
import torch
from src.memory.adaptive_evolution import evolve_items,build_trajectory,batch,rollout,make_model
from src.memory.consolidation import ConsolidationController
from src.memory.continual_utility import policy_loss


def public_facts():
    observations=[];vectors={}
    for key in range(4):
        for value in range(3):
            text=f'person{key} at room{value}'
            observations.append(dict(type='observation',key=str(key),value=str(value),text=text))
            vector=[0.]*32;vector[key*3+value]=1.;vectors[text]=vector
    return {'public':dict(split='train',events=observations)},vectors


def data(benchmark='reactivation'):
    stories,vectors=public_facts()
    return build_trajectory(stories,vectors,'train',benchmark,91)


def test_exact_flow_and_importance_recovery():
    initial=torch.tensor([[[1.,1.,2.,0.]]]);k=torch.tensor([.1]);valid=torch.ones(1,1)
    low=torch.full((1,1),.5);high=torch.full((1,1),2.)
    first=evolve_items(initial,torch.tensor([20.]),k,low,valid)
    later=evolve_items(first,torch.tensor([20.]),k,high,valid)
    assert first[0,0,0]<initial[0,0,0]<later[0,0,0]
    full=evolve_items(initial,torch.tensor([20.]),k,low,valid)
    split=evolve_items(evolve_items(initial,torch.tensor([7.]),k,low,valid),torch.tensor([13.]),k,low,valid)
    assert torch.allclose(full,split,atol=1e-6)
    assert torch.equal(evolve_items(initial,torch.zeros(1),k,low,valid),initial)


def test_observed_invalidation_lowers_confidence():
    initial=torch.tensor([[[1.,1.,0.,0.]]])
    result=evolve_items(initial,torch.tensor([10.]),torch.tensor([.1]),torch.ones(1,1),torch.zeros(1,1))
    assert result[0,0,1]<.02


def test_causal_targets_and_capacity():
    inputs=batch([data()]);altered={k:v.clone() for k,v in inputs.items()};altered['identity'][-1]=9999
    a=rollout(inputs,12,'adaptive_rule',record=True,seed=7)
    b=rollout(altered,12,'adaptive_rule',record=True,seed=7)
    assert a['trace'][:-1]==b['trace'][:-1]
    assert max(r['slots'][0] for r in a['trace'])<=12
    for trace in a['trace']:
        for identity,state in zip(trace['identity'][0],trace['states'][0]):
            if identity<0:assert state==[0.,0.,0.,0.]


def test_new_cue_does_not_retroactively_change_flow():
    inputs=batch([data()]);step=int((inputs['kind'][:,0]==2).nonzero()[1])
    altered={k:v.clone() for k,v in inputs.items()};altered['key'][step]=(inputs['key'][step]+8)%24
    a=rollout(inputs,12,'adaptive_rule',record=True,seed=1)
    b=rollout(altered,12,'adaptive_rule',record=True,seed=1)
    assert a['trace'][step]['states']==b['trace'][step]['states']
    assert a['trace'][step]['focus']!=b['trace'][step]['focus']


def test_same_neural_size_and_differentiable_utility():
    model=make_model('adaptive',11,.1)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in ConsolidationController('liquid').parameters())
    inputs=batch([data()]);result=rollout(inputs,8,'adaptive',model,rate=.1,seed=2)
    baseline=rollout(inputs,8,'decay',rate=.03,greedy=True)
    policy_loss(result,baseline).backward()
    assert torch.isfinite(model.policy.weight.grad).all()
    assert (model.policy.weight.grad.abs().sum(1)>0).all()


def test_public_benchmarks_are_reproducible_and_keep_targets_causal():
    stories,vectors=public_facts()
    for benchmark in ['inversion','reactivation','query_conditioned']:
        row=build_trajectory(stories,vectors,'train',benchmark,91)
        assert row==build_trajectory(stories,vectors,'train',benchmark,91)
        ledger={};last=-1
        for e in row['events']:
            assert e['time']>=last;last=e['time']
            if e['type']==1:ledger[e['key']]=e['identity']
            if e['type']==0:assert e['identity']==ledger[e['key']]
        assert sum(e['probe'] for e in row['events'])>0
