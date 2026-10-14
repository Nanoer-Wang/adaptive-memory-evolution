import numpy as np
import torch
from src.memory.evolving_memory import EvolutionPolicy,make_trajectory,batch,rollout
from src.memory.continual_utility import policy_loss


def facts():
    observations=[];vectors={}
    for key in range(4):
        for value in range(3):
            text=f'person{key} at room{value}';vector=[0.]*32;vector[key*3+value]=1.;vectors[text]=vector
            observations.append(dict(type='observation',key=str(key),value=str(value),text=text))
    return {'story':dict(split='train',events=observations)},vectors


def trajectory(condition='mixture'):
    stories,vectors=facts();return make_trajectory(stories,vectors,'train',condition,81)


def test_noise_is_paired_and_does_not_change_facts_or_queries():
    clean=trajectory();noisy=trajectory('feedback_noise');misleading=trajectory('feedback_misleading')
    for other in [noisy,misleading]:
        assert [{k:v for k,v in e.items() if k!='flip'} for e in clean['events']]==[{k:v for k,v in e.items() if k!='flip'} for e in other['events']]
        assert sum(e['flip'] for e in other['events'])>0
    x=rollout(batch([clean]),8,'frequency',seed=7)
    y=rollout(batch([noisy]),8,'frequency',seed=7)
    assert x['metrics']==y['metrics']


def test_stable_facts_do_not_change_and_cycles_have_prereminder_probes():
    for condition in ['mixture','cyclic']:
        row=trajectory(condition);versions={k:set() for k in row['stable_keys']}
        for e in row['events']:
            if e['type']==1 and e['key'] in versions:versions[e['key']].add(e['identity'])
        assert all(len(v)==1 for v in versions.values())
        if condition=='cyclic':assert sum(e['return_probe'] for e in row['events'])==16


def test_noise_changes_observed_feedback_not_ground_truth_at_that_query():
    clean=batch([trajectory()]);changed={k:v.clone() for k,v in clean.items()}
    first=int((clean['kind'][:,0]==0).nonzero()[0]);changed['flip'][first]=True
    a=rollout(clean,12,'adaptive_rule',rate=.1,seed=7,record=True)
    b=rollout(changed,12,'adaptive_rule',rate=.1,seed=7,record=True)
    assert a['trace'][:first]==b['trace'][:first]
    assert a['trace'][first]['correct']==b['trace'][first]['correct']
    if a['trace'][first]['answered'][0]:assert a['trace'][first]['feedback']!=b['trace'][first]['feedback']


def test_capacity_objective_and_small_learned_head():
    model=EvolutionPolicy(.1);assert sum(p.numel() for p in model.parameters())==123
    inputs=batch([trajectory('cyclic')]);result=rollout(inputs,8,'adaptive_learned',.1,model,seed=2)
    baseline=rollout(inputs,8,'fixed_decay',.001,greedy=True)
    policy_loss(result,baseline).backward()
    assert torch.isfinite(model.policy.weight.grad).all()
    assert model.policy.weight.grad.abs().sum()>0
    m=result['metrics'];assert m['peak_slots'][0]<=8
    assert abs(m['utility'][0]-(m['accuracy'][0]+.25*m['stable_retention'][0]-.25*m['stale_error'][0]))<1e-6
    assert 5<=m['adaptation_lag'][0]<=12


def test_future_target_does_not_change_past_states():
    original=batch([trajectory()]);modified={k:v.clone() for k,v in original.items()};modified['identity'][-1]=99999
    a=rollout(original,8,'adaptive_rule',record=True,seed=4)
    b=rollout(modified,8,'adaptive_rule',record=True,seed=4)
    assert a['trace'][:-1]==b['trace'][:-1]


def test_stale_error_is_measured_and_filter_uses_observed_ledger():
    events=[]
    for i in range(23):
        kind=1 if i in [0,21] else 0;identity=1 if i<21 else 2
        events.append(dict(type=kind,key=0,identity=identity,time=float(i),dt=1.,delay=0.,regime=0,
                           features=[0.]*40,stable=False,target_query=kind==0,return_probe=False,flip=False))
    result=rollout(batch([dict(events=events)]),1,'frequency',greedy=True,record=True)
    assert result['trace'][-1]['stale']==[True]
    m=result['metrics'];assert m['stale_error'][0]>0
    assert m['guarded_coverage'][0]<m['answer_coverage'][0]
