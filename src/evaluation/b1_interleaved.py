"""Explicit feedback, exposure-aware acquisition, and independent stream audits."""
import copy
import math
import time
import numpy as np
from src.memory.b1_environment import digest
from src.memory.b1_policy import Policy


def fraction(a,b):return a/b if b else None


def lag_episode(queries,shift_time):
    vals=[q['correct'] for q in queries];lag=len(vals);recovered=False
    for j in range(4,len(vals)):
        if sum(vals[j-4:j])>=3 and sum(vals[j-3:j+1])>=3:lag=j+1;recovered=True;break
    return dict(query_opportunities=len(vals),lag_queries=lag,recovered=recovered,
                lag_time=(queries[lag-1]['time']-shift_time) if queries else None,
                score=1-lag/len(vals) if vals else None,
                initial_window_already_good=sum(vals[:4])>=3 if len(vals)>=4 else None)


def run(row,method,seed,head=None,policy=None):
    p=policy if policy is not None else Policy(method,row['capacity'],seed,head)
    initial=p.bootstrap(copy.deepcopy(row['initial_memory']))
    frames=[];answers={};timings={k:0. for k in p.calls};ages=[];revisions=[];state_only_changes=0;shift_times={}
    def call(name,*args):
        start=time.perf_counter();v=getattr(p,name)(*args);timings[name]+=time.perf_counter()-start;return v
    for wrapped in row['events']:
        event=copy.deepcopy(wrapped['input']);labels=wrapped['evaluation'];kind=event['kind']
        before=p.snapshot();call('update_state',event)
        flowed=p.snapshot();action=None;answer=None;feedback_value=None
        if p.adaptive and before and kind!='observe':
            state_only_changes+=int(any(a[f]!=b[f] for a,b in zip(before,flowed) for f in ('I','C','U','T')))
        if kind=='context':shift_times[labels['phase']]=event['time']
        if kind=='observe':
            if labels['revision']:revisions.append(dict(event_id=event['event_id'],key=event['key'],item_id=event['item_id'],time=event['time']))
            call('observe',event);action=call('evict')
        if kind=='query':
            answer=call('retrieve',event);correct=answer is not None and answer['item_id']==labels['target_id']
            answers[event['event_id']]=dict(event_id=event['event_id'],time=event['time'],key=event['key'],**labels,
                answer=answer,correct=correct,stale=answer is not None and not correct,hit=answer is not None,
                guarded=answer is not None and answer['item_id']==p.ledger[event['key']])
        if kind=='feedback':
            q=answers[event['query_event_id']];feedback_value=float(q['correct'])
            call('feedback',q['answer']['item_id'] if q['answer'] else None,feedback_value)
        resident=p.snapshot();assert len(resident)<=row['capacity']
        active=set(row['stable_keys']+(row['a_keys'] if p.focus==0 else row['b_keys']))
        for i in resident:assert all(math.isfinite(i[f]) for f in ('I','C','U','T'))
        rages=[p.now-i['birth'] for i in resident];ages.extend(rages)
        frames.append(dict(event=event,evaluation=labels,flow_state=flowed if p.adaptive else None,
            resident=resident,action=action,answer=answer,feedback_value=feedback_value,focus=p.focus,
            active_residents=sum(i['key'] in active for i in resident),noise_residents=sum(i['key'] in row['noise_keys'] for i in resident),
            resident_ages=rages,counts=dict(p.counts)))
    queries=list(answers.values());scored=[q for q in queries if q['phase']>=0]
    stable=[q for q in scored if q['stable']];hits=sum(q['hit'] for q in scored)
    episodes=[]
    for phase in range(3):
        qq=[q for q in scored if q['phase']==phase and q['target_query']]
        episodes.append(dict(phase=phase,applicable=bool(qq[0]['shifted']),
            **(lag_episode(qq,shift_times[phase]) if qq[0]['shifted'] else dict(lag_queries=None,lag_time=None,recovered=None,score=None))))
    acquisitions=[]
    for revision in revisions:
        qq=[q for q in scored if q['event_id']>revision['event_id'] and q['key']==revision['key'] and q['target_id']==revision['item_id']]
        first=next((n for n,q in enumerate(qq,1) if q['correct']),None)
        acquisitions.append(dict(**revision,eligible_queries=len(qq),success=first is not None,
            first_correct_opportunity=first,censored=first is None,
            latency_time=qq[first-1]['time']-revision['time'] if first else None,
            failed_before_recovery=bool(first and first>1)))
    lifetimes=p.finish_lifetimes();nobs=p.calls['observe'];overflow=p.counts['overflow_opportunities'];nevents=len(frames)
    assert nobs==p.counts['accepted_writes']+p.counts['rejected_writes']+p.counts['duplicate_refreshes']
    assert overflow==p.counts['resident_evictions']+p.counts['rejected_writes']
    assert len(lifetimes)==len(initial)+p.counts['accepted_writes']
    shifted=[e for e in episodes if e['applicable']]
    avg=lambda values:float(np.mean(values)) if values else None
    metrics=dict(method=method,policy_seed=seed,world_seed=row['world_seed'],trajectory_seed=row['trajectory_seed'],
        pressure=row['pressure'],noise_ratio=row['noise_ratio'],shift=row['shift'],capacity=row['capacity'],
        active_memory_count=8,noise_memory_count=row['noise_memory_count'],
        initial_sha256=digest(initial),input_sha256=digest([w['input'] for w in row['events']]),
        query_sha256=digest(row['query_sequence']),environment_sha256=row['environment_sha256'],
        queries=len(scored),correct=sum(q['correct'] for q in scored),accuracy=avg([q['correct'] for q in scored]),
        stable_retention=avg([q['correct'] for q in stable]),stale_error=avg([q['stale'] for q in scored]),
        hit_rate=fraction(hits,len(scored)),stale_given_hit=fraction(sum(q['stale'] for q in scored),hits),
        guarded_coverage=avg([q['guarded'] for q in scored]),
        adaptation_lag=avg([e['lag_queries'] for e in shifted]),adaptation_lag_time=avg([e['lag_time'] for e in shifted]),
        adaptation_success=avg([e['recovered'] for e in shifted]),
        recovery_query_lag=episodes[2]['lag_queries'],recovery_time=episodes[2]['lag_time'],
        acquisition_success=avg([a['success'] for a in acquisitions if a['eligible_queries']]),
        acquisition_recoveries_after_miss=sum(a['failed_before_recovery'] for a in acquisitions),
        revision_count=len(acquisitions),revision_without_query=sum(a['eligible_queries']==0 for a in acquisitions),
        observation_opportunities=nobs,**p.counts,eviction_count=p.counts['resident_evictions'],
        eviction_rate_per_observation=fraction(p.counts['resident_evictions'],nobs),
        replacement_rate_per_overflow=fraction(p.counts['resident_evictions'],overflow),
        eviction_rate_per_time=p.counts['resident_evictions']/p.now,
        resident_age_quantiles=np.quantile(ages,[0,.25,.5,.75,1]).tolist(),
        time_weighted_resident_age=fraction(p.age_integral,p.slot_time),
        lifetime_observed_quantiles=np.quantile([r['lifetime'] for r in lifetimes],[0,.25,.5,.75,1]).tolist(),
        lifetime_right_censored=sum(r['right_censored'] for r in lifetimes),
        mean_noise_residents=avg([f['noise_residents'] for f in frames]),
        mean_active_residents=avg([f['active_residents'] for f in frames]),
        state_change_without_write_events=state_only_changes,policy_calls=p.calls,
        policy_seconds=sum(timings.values()),timing_seconds=timings,peak_slots=max(len(f['resident']) for f in frames))
    return dict(metrics=metrics,trace=frames,queries=queries,lifetimes=lifetimes,episodes=episodes,acquisition=acquisitions)


def audit(row,result):
    live={i['item_id']:i for i in row['initial_memory']};ledger={i['key']:i['item_id'] for i in live.values()}
    replacements=rejections=0;available_at_shift={};newly_available_correct=0;rounds=[]
    for wrapper,frame in zip(row['events'],result['trace'],strict=True):
        e=wrapper['input'];phase=wrapper['evaluation']['phase'];kind=e['kind']
        assert e==frame['event']
        if kind=='context':available_at_shift[phase]=set(live)
        if kind=='observe':
            duplicate=e['item_id'] in live;ledger[e['key']]=e['item_id'];live[e['item_id']]=e
            if len(live)>row['capacity']:
                victim=frame['action']['victim'];assert victim in live
                if victim==e['item_id']:rejections+=1
                else:replacements+=1
                del live[victim]
            elif duplicate:assert frame['action']['action']=='none'
        if kind=='query':
            options=[i for i in live.values() if i['key']==e['key']]
            predicted=max(options,key=lambda i:i['version'])['item_id'] if options else None
            assert (frame['answer']['item_id'] if frame['answer'] else None)==predicted
            assert wrapper['evaluation']['target_id']==ledger[e['key']]
            if phase>0 and predicted==ledger[e['key']] and predicted not in available_at_shift[phase]:newly_available_correct+=1
        assert set(live)=={i['item_id'] for i in frame['resident']}
        assert all(i['content']==live[i['item_id']]['content'] for i in frame['resident'])
        if kind=='feedback':
            q=next(q for q in result['queries'] if q['event_id']==e['query_event_id'])
            assert frame['feedback_value']==float(q['correct'])
    m=result['metrics'];assert replacements==m['eviction_count'] and rejections==m['rejected_writes']
    assert m['peak_slots']<=row['capacity']
    return dict(passed=True,correct_answers_unavailable_at_phase_start=newly_available_correct,
                eviction_count=replacements,rejected_writes=rejections,frames=len(result['trace']))
