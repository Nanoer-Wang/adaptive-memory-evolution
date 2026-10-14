import numpy as np
import torch
from src.memory.consolidation import ConsolidationController
from src.memory.continual_utility import batch_trajectories,rollout,policy_loss,make_trajectory


def trajectory():
    events=[]
    for i in range(12):
        key=i%3;identity=key*10000+i+1
        x=[0.]*40;x[key]=1.;x[32]=1.
        events.append(dict(type=1,key=key,identity=identity,time=float(i*10),dt=10.,delay=0.,regime=0,features=x))
        y=x.copy();y[32]=0.
        events.append(dict(type=0,key=key,identity=identity,time=float(i*10+.5),dt=.5,delay=35.,regime=0,features=y))
    return dict(events=events)


def test_hard_budget_and_repeatability():
    data=batch_trajectories([trajectory()])
    a=rollout(data,2,seed=9,record=True);b=rollout(data,2,seed=9,record=True)
    assert a['trace']==b['trace']
    assert all(max(r['slots'])<=2 for r in a['trace'])
    assert all(0<=v<=1 for v in a['metrics']['accuracy'])


def test_future_task_utility_reaches_all_policy_heads():
    torch.manual_seed(2);model=ConsolidationController('liquid')
    data=batch_trajectories([trajectory(),trajectory()])
    baseline=rollout(data,2,greedy=True)
    result=rollout(data,2,model,seed=1)
    loss=policy_loss(result,baseline);loss.backward()
    assert model.policy.weight.grad is not None
    assert torch.isfinite(model.policy.weight.grad).all()
    assert (model.policy.weight.grad.abs().sum(1)>0).all()
    assert model.impulse.weight.grad.abs().sum()>0


def test_full_capacity_preserves_answers_and_loss_identity():
    data=batch_trajectories([trajectory()]);output=rollout(data,30)
    m=output['metrics']
    assert m['accuracy']==[1.]
    expected=m['accuracy'][0]+.25*m['delayed_accuracy'][0]-.05*m['stale_occupancy'][0]
    assert abs(m['utility'][0]-expected)<1e-6


def test_future_target_cannot_change_prior_eviction():
    data=batch_trajectories([trajectory()]);changed={k:v.clone() for k,v in data.items()}
    changed['identity'][-1]=99999
    model=ConsolidationController('liquid')
    with torch.no_grad():
        a=rollout(data,2,model,seed=8,record=True)
        b=rollout(changed,2,model,seed=8,record=True)
    assert a['trace'][:-1]==b['trace'][:-1]


def test_padding_does_not_add_reward_or_decisions():
    short=trajectory();long=trajectory();short['events']=short['events'][:12]
    a=rollout(batch_trajectories([short]),3,greedy=True)
    b=rollout(batch_trajectories([short,long]),3,greedy=True)
    assert abs(a['metrics']['utility'][0]-b['metrics']['utility'][0])<1e-6
    assert a['metrics']['evictions'][0]==b['metrics']['evictions'][0]


def test_generator_conflicts_delays_causality_and_matched_workload():
    observations=[];vectors={}
    for key in range(4):
        for value in range(3):
            text=f'person{key} at room{value}'
            observations.append(dict(type='observation',key=str(key),value=str(value),text=text))
            vector=[0.]*32;vector[key*3+value]=1.;vectors[text]=vector
    stories={'public':dict(split='train',events=observations)}
    outputs=[make_trajectory(stories,vectors,'train',env,15,180) for env in ['stationary','slow','rapid']]
    for result in outputs:
        assert result['regime_counts']==[60,60,60]
        ledger={};previous=-1;conflicts=0
        for e in result['events']:
            assert e['time']>=previous;previous=e['time']
            assert len(e['features'])==40
            if e['type']:
                ledger[e['key']]=e['identity'];conflicts+=e['conflict']
            else:
                assert e['identity']==ledger[e['key']]
                assert e['delay']>0
        assert conflicts>0
    assert len({len(r['events']) for r in outputs})==1
    assert outputs[0]==make_trajectory(stories,vectors,'train','stationary',15,180)
