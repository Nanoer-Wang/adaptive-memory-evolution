import copy
import numpy as np
import pytest
from tests.test_b1_sanity import facts,zero_head
from src.memory.b1_interleaved import generate,audit_environment
from src.memory.b1_policy import METHODS
from src.evaluation.b1_interleaved import run,audit,lag_episode


def tape(pressure='medium',capacity=8,shift=True):
    return generate(*facts(),60401,260401,pressure,capacity,shift)


@pytest.mark.parametrize('pressure',['low','medium','high'])
def test_ratios_interleaving_determinism_and_causal_truth(pressure):
    r=tape(pressure);assert r==tape(pressure);a=audit_environment(r)
    assert a['passed'] and a['max_consecutive_observations']==1
    for phase in range(3):
        noise={e['input']['key'] for e in r['events'] if e['evaluation']['phase']==phase and e['input']['kind']=='observe' and e['evaluation']['noise_observation']}
        assert len(noise)==r['noise_memory_count']


def test_pressure_changes_noise_pool_only_not_task_clock_or_query_order():
    rows=[tape(p) for p in ['low','medium','high']]
    def core_tape(row):
        return [e for e in row['events'] if not e['evaluation'].get('noise_observation',False)]
    assert rows[0]['initial_memory']==rows[1]['initial_memory']==rows[2]['initial_memory']
    assert core_tape(rows[0])==core_tape(rows[1])==core_tape(rows[2])
    assert all(r['query_sequence']==rows[0]['query_sequence'] for r in rows)


@pytest.mark.parametrize('method',METHODS)
def test_adapters_capacity_16_and_duplicate_refresh_accounting(method):
    row=tape(capacity=16);result=run(row,method,11,zero_head() if method=='adaptive_evolution' else None)
    assert audit(row,result)['passed'];m=result['metrics']
    assert m['peak_slots']<=16
    assert m['observation_opportunities']==m['accepted_writes']+m['rejected_writes']+m['duplicate_refreshes']
    assert m['policy_calls']['feedback']==m['policy_calls']['retrieve']==204


def test_no_shift_has_no_adaptation_or_acquisition_claim():
    r=run(tape(shift=False),'adaptive_rule',11)
    assert r['metrics']['adaptation_lag'] is None and r['metrics']['revision_count']==0


def test_lag_metric_detects_delayed_recovery_and_censors_failure():
    def queries(v):return [dict(correct=x,time=i+1) for i,x in enumerate(v)]
    immediate=lag_episode(queries([True]*20),0)
    delayed=lag_episode(queries([False]*6+[True]*14),0)
    failure=lag_episode(queries([False]*20),0)
    assert immediate['lag_queries']==5 and delayed['lag_queries']==10
    assert failure['lag_queries']==20 and not failure['recovered']


def test_future_reference_does_not_change_past_or_current_retrieval():
    row=tape();altered=copy.deepcopy(row);idx=next(i for i,e in reversed(list(enumerate(row['events']))) if e['input']['kind']=='query')
    altered['events'][idx]['evaluation']['target_id']='changed_future_reference'
    a=run(row,'adaptive_rule',11);b=run(altered,'adaptive_rule',11)
    assert a['trace'][:idx]==b['trace'][:idx]
    assert a['trace'][idx]['resident']==b['trace'][idx]['resident']
    assert a['trace'][idx]['answer']==b['trace'][idx]['answer']


def test_reacquisition_changes_what_can_be_answered_after_shift():
    row=tape();r=run(row,'fixed_decay',11)
    assert audit(row,r)['correct_answers_unavailable_at_phase_start']>0
    assert r['metrics']['revision_count']==16
