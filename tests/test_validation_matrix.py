import copy
import numpy as np
import pytest
from tests.test_b1_sanity import facts,zero_head
from src.memory.validation_matrix import generate,MatrixPolicy,BASELINES
from src.evaluation.b1_interleaved import run,audit
from scripts.validation_matrix import match_distance,ks,execute


def tape(**kwargs):return generate(*facts(),85101,485101,**kwargs)


def test_capacity_pressure_changes_only_noise_ids_not_task_schedule():
    a=tape(capacity=4,noise_count=8,repetitions=4);b=tape(capacity=64,noise_count=128,repetitions=4)
    assert a['initial_memory']==b['initial_memory'] and len(a['initial_memory'])==4
    def core(row):return [e for e in row['events'] if not e['evaluation'].get('noise_observation')]
    assert core(a)==core(b)
    for phase in range(3):
        noise={e['input']['key'] for e in b['events'] if e['evaluation']['phase']==phase and e['evaluation'].get('noise_observation')}
        assert len(noise)==128


def test_prereminder_queries_precede_every_return_observation():
    r=tape(cyclic_probe=True,versions=False);events=[e for e in r['events'] if e['evaluation']['phase']==2]
    probe=[e for e in events if e['evaluation'].get('pre_reminder')]
    assert len(probe)==4 and max(e['input']['time'] for e in probe)<min(e['input']['time'] for e in events if e['input']['kind']=='observe')
    assert not any(e['evaluation'].get('revision') for e in r['events'])


@pytest.mark.parametrize('method',BASELINES+('no_importance','no_consistency','no_usage','no_time','event_only','discrete_euler','query_conditioned'))
def test_policies_share_capacity_causality_and_full_audit(method):
    r=tape(capacity=4);p=MatrixPolicy(method,4,11,zero_head() if method not in ('fixed_decay','frequency','lru','adaptive_rule','replacement_matched') else None,.2,24.)
    result=run(r,method,11,policy=p);assert audit(r,result)['passed']
    field={'no_importance':('I',1),'no_consistency':('C',1),'no_usage':('U',0),'no_time':('T',0)}.get(method)
    if field:assert all(i[field[0]]==field[1] for f in result['trace'] for i in f['resident'])


def test_lru_tracks_access_rather_than_only_observation():
    a=tape();initial=a['initial_memory'][:2]
    p=MatrixPolicy('lru',2,11);p.bootstrap(initial)
    p.now=10.;p.retrieve(dict(key=initial[0]['key']))
    assert p.score(p.items[initial[0]['item_id']])>p.score(p.items[initial[1]['item_id']])


def test_feedback_flip_common_randomness_and_no_truth_rewrite():
    clean=tape();noisy=tape(feedback_noise=.5)
    assert clean['query_sequence']==noisy['query_sequence']
    assert [e['evaluation'] for e in clean['events']]==[e['evaluation'] for e in noisy['events']]
    for method in ('fixed_decay','frequency','lru'):
        a=run(clean,method,11,policy=MatrixPolicy(method,8,11));b=run(noisy,method,11,policy=MatrixPolicy(method,8,11))
        assert a['queries']==b['queries'] and a['lifetimes']==b['lifetimes']


def test_distribution_matching_detects_different_shapes_with_same_mean():
    a=dict(eviction_count=10,mean_observed_lifetime=10,lifetime_censor_fraction=.1,lifetimes=[0,20],uncensored_lifetimes=[0,20])
    b=dict(a,lifetimes=[10,10],uncensored_lifetimes=[10,10])
    assert match_distance(a,b)['eviction_relative_error']==0 and not match_distance(a,b)['matched']
    assert ks(a['lifetimes'],b['lifetimes'])==.5


def test_new_policy_full_matches_frozen_b1_state_dynamics():
    from src.memory.b1_attribution import AttributionPolicy
    r=tape();a=run(r,'adaptive_evolution',11,policy=MatrixPolicy('adaptive_evolution',8,11,zero_head()))
    b=run(r,'adaptive_evolution',11,policy=AttributionPolicy('adaptive_evolution',8,11,zero_head()))
    assert a['trace']==b['trace'] and a['lifetimes']==b['lifetimes']
