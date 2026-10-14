"""Policy-independent evaluator and auditable accounting for B1 sanity runs."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
from src.memory.b1_environment import digest
from src.memory.b1_policy import Policy


def ratio(a,b): return a/b if b else None


def run_trajectory(row, method, seed, coefficients=None):
    capacity=len(row['initial_memory'])
    policy=Policy(method, capacity, seed, coefficients)
    initial=policy.bootstrap(copy.deepcopy(row['initial_memory']))
    trace=[]; answers=[]; ages=[]; timing={k:0. for k in policy.calls}
    policy_elapsed=0.
    def call(name,*args):
        start=time.perf_counter(); value=getattr(policy,name)(*args)
        timing[name]+=time.perf_counter()-start
        return value
    for index, record in enumerate(row['events']):
        event=copy.deepcopy(record['input'])
        # Evaluator labels remain here. Policy receives only an observed input,
        # then the scalar feedback that would be available after its answer.
        call('update_state',event)
        action=None; answer=None
        if event['kind']=='observe':
            call('observe',event); action=call('evict')
        if event['kind']=='query':
            answer=call('retrieve',event)
            reference=record['evaluation']['target_id']
            correct=answer is not None and answer['item_id']==reference
            stale=answer is not None and answer['item_id']!=reference
            call('feedback',answer['item_id'] if answer else None,float(correct))
            answers.append(dict(event_id=index, **record['evaluation'], key=event['key'],
                                answer=answer, correct=correct, stale=stale, hit=answer is not None,
                                guard_accept=answer is not None and policy.ledger[event['key']]==answer['item_id']))
        resident=policy.snapshot()
        ages.append(float(np.mean([policy.now-i['birth'] for i in resident])) if resident else 0.)
        assert len(resident)<=capacity
        assert all(math.isfinite(i[field]) for i in resident for field in ('I','C','U','T'))
        trace.append(dict(event_id=index,time=event['time'],kind=event['kind'],
                          context=policy.focus,action=action,answer=answer,resident=resident,
                          counts=dict(policy.counts)))
    count=len(answers); hits=sum(q['hit'] for q in answers); correct=sum(q['correct'] for q in answers)
    stable=[q for q in answers if q['stable']]; volatile=[q for q in answers if not q['stable']]
    episodes=[]
    for phase in range(row['phases']):
        queries=[q for q in volatile if q['phase']==phase]
        if not queries or not queries[0]['shifted']:
            episodes.append(dict(phase=phase, applicable=False, lag=None, recovered=None, adaptation_score=None))
            continue
        values=[q['correct'] for q in queries]; lag=len(values); success=False
        for j in range(4,len(values)):
            if sum(values[j-4:j])>=3 and sum(values[j-3:j+1])>=3:
                lag=j+1; success=True; break
        episodes.append(dict(phase=phase, applicable=True, lag=lag,recovered=success,
                             adaptation_score=1-lag/len(values)))
    applicable=[e for e in episodes if e['applicable']]
    lifetimes=policy.finish_lifetimes(); complete=[e['lifetime'] for e in lifetimes if not e['right_censored']]
    observed_lifetimes=[e['lifetime'] for e in lifetimes]
    nobs=policy.calls['observe']; overflow=policy.counts['overflow_opportunities']
    assert nobs==policy.counts['accepted_writes']+policy.counts['rejected_writes']+policy.counts['duplicate_refreshes']
    assert overflow==policy.counts['resident_evictions']+policy.counts['rejected_writes']
    assert len(lifetimes)==capacity+policy.counts['accepted_writes']
    assert sum(e['right_censored'] for e in lifetimes)==len(policy.items)
    metrics=dict(method=method,policy_seed=seed,world_seed=row['world_seed'],trajectory_seed=row['trajectory_seed'],
        stable_fraction=row['stable_fraction'],regime=row['regime'],capacity=capacity,
        environment_sha256=row['environment_sha256'],initial_memory_sha256=digest(initial),
        input_sha256=digest([e['input'] for e in row['events']]),query_sha256=digest(row['query_sequence']),
        queries=count,correct=correct,hits=hits,stale=sum(q['stale'] for q in answers),
        accuracy=ratio(correct,count),retrieval_hit_rate=ratio(hits,count),
        stable_queries=len(stable),stable_correct=sum(q['correct'] for q in stable),
        stable_retention=ratio(sum(q['correct'] for q in stable),len(stable)),
        volatile_accuracy=ratio(sum(q['correct'] for q in volatile),len(volatile)),
        stale_retrieval=ratio(sum(q['stale'] for q in answers),count),
        stale_given_hit=ratio(sum(q['stale'] for q in answers),hits),
        guarded_coverage=ratio(sum(q['guard_accept'] for q in answers),count),
        adaptation_lag=float(np.mean([e['lag'] for e in applicable])) if applicable else None,
        adaptation_success=float(np.mean([e['recovered'] for e in applicable])) if applicable else None,
        adaptation_score=float(np.mean([e['adaptation_score'] for e in applicable])) if applicable else None,
        observation_opportunities=nobs,**policy.counts,
        eviction_count=policy.counts['resident_evictions'],
        replacement_rate=ratio(policy.counts['resident_evictions'],overflow),
        rejected_write_count=policy.counts['rejected_writes'],
        average_resident_age=float(np.mean(ages)),
        time_weighted_resident_age=ratio(policy.age_integral,policy.slot_time),
        lifetime_observed_quantiles=np.quantile(observed_lifetimes,[0,.25,.5,.75,1]).tolist(),
        lifetime_complete_quantiles=np.quantile(complete,[0,.25,.5,.75,1]).tolist() if complete else None,
        lifetime_right_censored=sum(e['right_censored'] for e in lifetimes),
        lifetime_episodes=len(lifetimes),
        policy_calls=policy.calls,policy_seconds=sum(timing.values()),timing_seconds=timing,
        nominal_update_calls=policy.calls['update_state'],peak_slots=max(len(t['resident']) for t in trace),
        max_ledger_keys=len(policy.ledger),
        nominal_adaptive_state_bytes=capacity*4*8 if policy.adaptive else 0)
    return dict(metrics=metrics,trace=trace,queries=answers,lifetimes=lifetimes,adaptation_episodes=episodes)


def audit_run(row, result):
    """Independent trace reconstruction of membership, observed versions and counts."""
    live={i['item_id']:i for i in row['initial_memory']}; ledger={i['key']:i['item_id'] for i in live.values()}
    replacements=rejections=overflows=0
    for entry, frame in zip(row['events'],result['trace'],strict=True):
        e=entry['input']; action=frame['action']
        if e['kind']=='observe':
            ledger[e['key']]=e['item_id']; live[e['item_id']]=e
            if len(live)>len(row['initial_memory']):
                overflows+=1
                assert action['victim'] in live
                assert action['action']==('reject' if action['victim']==e['item_id'] else 'replace')
                if action['action']=='reject': rejections+=1
                else: replacements+=1
                del live[action['victim']]
        if e['kind']=='query':
            options=[i for i in live.values() if i['key']==e['key']]
            predicted=max(options,key=lambda i:i['version'])['item_id'] if options else None
            assert (frame['answer']['item_id'] if frame['answer'] else None)==predicted
            assert entry['evaluation']['target_id']==ledger[e['key']]
        assert set(live)=={i['item_id'] for i in frame['resident']}
        assert all(i['content']==live[i['item_id']]['content'] for i in frame['resident'])
    m=result['metrics']
    assert replacements==m['eviction_count'] and rejections==m['rejected_write_count']
    assert overflows==m['overflow_opportunities']
    assert m['peak_slots']<=len(row['initial_memory'])
    queries=result['queries']
    assert m['correct']==sum(q['answer'] is not None and q['answer']['item_id']==q['target_id'] for q in queries)
    return dict(passed=True,frames=len(result['trace']),replacements=replacements,rejections=rejections)
