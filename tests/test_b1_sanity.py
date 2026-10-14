import copy
import json
import math
import numpy as np
import pytest
from src.memory.b1_environment import generate, audit_environment, digest
from src.memory.b1_policy import Policy, METHODS
from src.evaluation.b1_sanity import run_trajectory, audit_run


def facts():
    events=[];vectors={}
    for key in range(4):
        for value in range(3):
            text=f'person{key} in room{value}'; v=[0.]*32;v[key*3+value]=1.;vectors[text]=v
            events.append(dict(type='observation',key=str(key),value=str(value),text=text))
    return dict(story=dict(split='train',events=events)),vectors


def tape(ratio=.5,regime='rapid'):
    return generate(*facts(),60001,160001,ratio,regime,phases=2)


def event(kind,time,key=0,topic=0,version=1):
    return dict(kind=kind,time=time,key=key,topic=topic,version=version,
                item_id=f'{key}:{version}',content=f'fact {key} version{version}',vector=[0.]*32)


def zero_head():
    return dict(weight=np.zeros((3,40),dtype=np.float32),
                bias=np.array([np.log(2),np.log(3),np.log(np.expm1(.3*30-1e-4))],dtype=np.float32))


@pytest.mark.parametrize('fraction',[.2,.5,.8])
def test_deterministic_balanced_public_inputs_and_serialization(fraction):
    row=tape(fraction);assert row==tape(fraction)
    assert audit_environment(row)['passed']
    assert digest(row)==digest(json.loads(json.dumps(row)))
    assert row['world_seed']==60001 and row['trajectory_seed']==160001
    assert not any('target_id' in e['input'] or 'stable' in e['input'] for e in row['events'])


def test_shared_opportunities_lifetime_and_exact_trace_accounting():
    row=tape();results=[]
    for method in METHODS:
        r=run_trajectory(row,method,11,zero_head() if method=='adaptive_evolution' else None)
        assert audit_run(row,r)['passed'];results.append(r)
        m=r['metrics']
        assert m['eviction_count']+m['rejected_write_count']==m['overflow_opportunities']
        assert m['lifetime_episodes']==8+m['accepted_writes']
        assert m['lifetime_right_censored']==8
    for field in ('initial_memory_sha256','input_sha256','query_sha256','overflow_opportunities','queries'):
        assert len({r['metrics'][field] for r in results})==1


def test_held_flow_uses_previous_context_and_matches_reference_math():
    import torch
    from src.memory.adaptive_evolution import evolve_items
    p=Policy('adaptive_rule');p.bootstrap([event('observe',0)])
    p.update_state(event('context',0,topic=0))
    p.update_state(event('context',10,topic=1))
    expected=evolve_items(torch.tensor([[[1.,1.,0.,0.]]]),torch.tensor([10.]),
                          torch.tensor([.3]),torch.tensor([[2.]]),torch.tensor([[1.]]))[0,0].numpy()
    np.testing.assert_allclose([p.items['0:1'].I,p.items['0:1'].C,p.items['0:1'].U,p.items['0:1'].T],expected,rtol=1e-6)
    before=p.items['0:1'].I;p.update_state(event('context',20,topic=0))
    assert p.items['0:1'].I<before
    with pytest.raises(ValueError):p.update_state(event('context',19))


def test_pure_fixed_decay_and_frequency_do_not_read_correctness_feedback():
    for method in ('fixed_decay','frequency'):
        left=Policy(method);right=Policy(method)
        for p in (left,right):
            p.bootstrap([event('observe',0)]);p.update_state(event('query',10));p.retrieve(event('query',10))
        left.feedback('0:1',0.);right.feedback('0:1',1.)
        assert left.score(left.items['0:1'])==right.score(right.items['0:1'])
        expected=-.01 if method=='fixed_decay' else math.log(2)
        assert abs(left.score(left.items['0:1'])-expected)<1e-8


def test_rejected_candidate_is_not_resident_eviction_or_lifetime():
    p=Policy('frequency',capacity=1)
    p.bootstrap([event('observe',0)])
    p.retrieve(event('query',0))
    p.update_state(event('observe',1,key=1));p.observe(event('observe',1,key=1));a=p.evict()
    assert a['action']=='reject' and p.counts['resident_evictions']==0
    assert p.counts['rejected_writes']==1 and len(p.finish_lifetimes())==1


def test_metadata_cannot_resurrect_evicted_content_and_stale_answers_are_visible():
    p=Policy('fixed_decay',capacity=1);p.bootstrap([event('observe',0)])
    p.update_state(event('observe',1,key=1));p.observe(event('observe',1,key=1));p.evict()
    assert 0 in p.ledger and p.retrieve(event('query',1,key=0)) is None
    p=Policy('frequency',capacity=1);p.bootstrap([event('observe',0)])
    p.retrieve(event('query',0));p.update_state(event('observe',1,version=2));p.observe(event('observe',1,version=2));p.evict()
    assert p.retrieve(event('query',1))['item_id']=='0:1' and p.ledger[0]=='0:2'


def test_future_reference_cannot_change_earlier_states():
    row=tape();changed=copy.deepcopy(row)
    changed['events'][-1]['evaluation']['target_id']='unavailable'
    a=run_trajectory(row,'adaptive_rule',11);b=run_trajectory(changed,'adaptive_rule',11)
    assert a['trace'][:-1]==b['trace'][:-1]
    assert a['metrics']['input_sha256']==b['metrics']['input_sha256']


def test_trace_tampering_fails_audit():
    row=tape();r=run_trajectory(row,'fixed_decay',11)
    r['trace'][0]['resident'][0]['content']='fabricated'
    with pytest.raises(AssertionError):audit_run(row,r)


def test_no_shift_adaptation_is_null():
    r=run_trajectory(tape(regime='stationary'),'adaptive_rule',11)
    assert r['metrics']['adaptation_lag'] is None
    assert r['metrics']['adaptation_score'] is None


def test_numpy_frozen_head_matches_original_123parameter_head():
    import torch
    from src.memory.evolving_memory import EvolutionPolicy
    torch.manual_seed(11);m=EvolutionPolicy(.3)
    with torch.no_grad():m.policy.weight.normal_(0,.03)
    head=dict(weight=m.policy.weight.detach().numpy(),bias=m.policy.bias.detach().numpy())
    p=Policy('adaptive_evolution',coefficients=head);p.bootstrap([event('observe',0)])
    e=event('query',2);e['vector']=np.random.default_rng(1).normal(size=32).tolist()
    x=e['vector']+[0.,math.log1p(2)/5,0.,4.,0.,.5,.5,0.]
    p.update_state(e)
    with torch.no_grad():expected=[float(v) for v in m(torch.tensor(x,dtype=torch.float32))]
    np.testing.assert_allclose([p.alpha,p.held_w,p.held_k],expected,rtol=1e-6)
