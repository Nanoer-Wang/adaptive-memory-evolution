import copy
import numpy as np
import pytest
from tests.test_b1_sanity import facts,zero_head
from src.memory.b1_interleaved import generate,audit_environment
from src.memory.b1_attribution import AttributionPolicy,matching_loss
from src.evaluation.b1_interleaved import run,audit


def tape(dynamics='rapid_shift'):
    return generate(*facts(),73001,373001,dynamics=dynamics)


@pytest.mark.parametrize('dynamics,revisions',[('stable',0),('slow_drift',8),('rapid_shift',16)])
def test_dynamics_are_causal_and_counted(dynamics,revisions):
    row=tape(dynamics);assert audit_environment(row)['passed']
    assert sum(e['evaluation'].get('revision',False) for e in row['events'])==revisions


def test_slow_and_rapid_share_queries_clocks_initial_and_core_keys():
    a,b=tape('slow_drift'),tape('rapid_shift')
    assert a['initial_memory']==b['initial_memory'] and a['query_sequence']==b['query_sequence']
    assert [(e['input']['kind'],e['input']['time'],e['input'].get('key')) for e in a['events']]==[(e['input']['kind'],e['input']['time'],e['input'].get('key')) for e in b['events']]


def test_static_freezes_every_residency_and_keeps_insertion_formula():
    row=tape();p=AttributionPolicy('static_state_adaptive',coefficients=zero_head())
    r=run(row,p.method,11,policy=p);assert audit(row,r)['passed']
    states={}
    for f in r['trace']:
        for i in f['resident']:
            z=tuple(i[k] for k in ('I','C','U','T'))
            assert z==states.setdefault(i['episode'],z)
    # zero logits: alpha=.5,w=1, so inserted I=.5 (bootstrap I=1).
    # With frozen bootstrap I=1, the .5 candidates may all be rejected.
    # Test the insertion state before eviction, not surviving-only selection.
    q=AttributionPolicy('static_state_adaptive',coefficients=dict(weight=np.zeros((3,40)),bias=np.zeros(3)))
    event=next(e['input'] for e in row['events'] if e['input']['kind']=='observe')
    q.update_state(event);q.observe(event)
    assert q.items[event['item_id']].I==.5
    assert r['metrics']['state_change_without_write_events']==0


def test_decay_scale_cannot_tune_replacement_and_gate_endpoints():
    row=tape();results=[]
    for decay in (.0001,.001,1.):
        p=AttributionPolicy('fixed_decay',decay=decay);results.append(run(row,'fixed_decay',11,policy=p))
    assert all(r['queries']==results[0]['queries'] and r['lifetimes']==results[0]['lifetimes'] for r in results)
    full=run(row,'replacement_matched',11,policy=AttributionPolicy('replacement_matched',admission=1.))
    assert full['queries']==results[0]['queries'] and full['lifetimes']==results[0]['lifetimes']
    zero=run(row,'replacement_matched',11,policy=AttributionPolicy('replacement_matched',admission=0.))
    assert audit(row,zero)['passed'] and zero['metrics']['eviction_count']==0
    assert {i['item_id'] for i in zero['trace'][-1]['resident']}=={i['item_id'] for i in row['initial_memory']}


def test_matching_ignores_task_labels_and_utility_metrics():
    target=dict(eviction_count=20,mean_observed_lifetime=100)
    a=dict(eviction_count=21,mean_observed_lifetime=105,accuracy=0.)
    b=dict(a,accuracy=1.,stale_error=0.)
    assert matching_loss(a,target)==matching_loss(b,target)


def test_full_storage_lifetime_identity_and_feedback_causality():
    row=tape();p=AttributionPolicy('adaptive_evolution',coefficients=zero_head())
    r=run(row,p.method,11,policy=p)
    assert sum(i['lifetime'] for i in r['lifetimes'])==pytest.approx(8*row['events'][-1]['input']['time'])
    altered=copy.deepcopy(row);idx=next(n for n,e in enumerate(row['events']) if e['input']['kind']=='query')
    altered['events'][idx]['evaluation']['target_id']='wrong'
    other=run(altered,p.method,11,policy=AttributionPolicy(p.method,coefficients=zero_head()))
    assert other['trace'][:idx]==r['trace'][:idx]
    assert other['trace'][idx]['answer']==r['trace'][idx]['answer']
    assert other['trace'][idx]['resident']==r['trace'][idx]['resident']


def test_uniform_static_removes_bootstrap_insertion_state_asymmetry():
    row=tape();p=AttributionPolicy('static_uniform_state')
    r=run(row,p.method,11,policy=p);assert audit(row,r)['passed']
    assert all((i['I'],i['C'],i['U'],i['T'])==(1.,1.,0.,0.) for f in r['trace'] for i in f['resident'])
