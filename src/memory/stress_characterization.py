"""C4 matched-prefix stress with fixed context-switch cadence."""
import copy
import numpy as np
from src.memory.validation_matrix import generate
from src.memory.characterization import execute as base_execute
from src.evaluation.b1_interleaved import lag_episode
from src.memory.b1_environment import digest

def stream(stories,vectors,world,capacity,length):
    base=generate(stories,vectors,world,world+800000,capacity=capacity,noise_count=16,rounds=2)
    initial=copy.deepcopy(base['initial_memory']);events=[];content={e['key']:e['content'] for e in initial};versions={e['key']:1 for e in initial};ledger={e['key']:e['item_id'] for e in initial};last_query=None
    pools={}
    for story in stories.values():
        if story['split']=='train':
            for e in story['events']:
                if e['type']=='observation':pools.setdefault(e['key'],set()).add(e['text'])
    choices={text:sorted(pool) for pool in pools.values() for text in pool}
    revised=set();scope=np.random.default_rng(world+500).normal(size=(140,32));scope/=np.linalg.norm(scope,axis=1,keepdims=True)
    def emit(wrapper,phase):
        nonlocal last_query
        w=copy.deepcopy(wrapper);e=w['input'];label=w['evaluation'];e['event_id']=len(events);e['time']=float(len(events));label['phase']=phase
        if 'shifted' in label:label['shifted']=phase>0
        if e['kind']=='observe':
            k=e['key']
            if phase>=0 and k in (base['a_keys'] if phase%2==0 else base['b_keys']):
                if phase>0 and (phase,k) not in revised:
                    options=choices[content[k]];text=options[(options.index(content[k])+1)%len(options)];revised.add((phase,k))
                else:text=content[k]
                e['content']=text;v=np.asarray(vectors[text])+.35*scope[k];e['vector']=(v/max(np.linalg.norm(v),1e-8)).tolist()
            k=e['key'];changed=k in content and content[k]!=e['content'];versions[k]=versions.get(k,0)+int(k not in versions or changed)
            content[k]=e['content'];e['version']=versions[k];e['item_id']=f'{k}:{versions[k]}';ledger[k]=e['item_id'];label['revision']=changed
        if e['kind']=='query':label['target_id']=ledger[e['key']];last_query=e['event_id']
        if e['kind']=='feedback':e['query_event_id']=last_query
        events.append(w)
    for w in base['events']:
        if w['evaluation']['phase']==-1:emit(w,-1)
    phase=0;block=0
    while len(events)<length:
        row=generate(stories,vectors,world,world+800000+block*997,capacity=capacity,noise_count=16,rounds=2)
        for local in (0,1):
            for w in row['events']:
                if w['evaluation']['phase']==local:
                    emit(w,phase)
                    if len(events)==length:break
            phase+=1
            if len(events)==length:break
        block+=1
    base.update(initial_memory=initial,events=events,query_sequence=[w['input']['key'] for w in events if w['input']['kind']=='query'],
        trajectory_seed=world+800000,phases=phase,stage_topics=[j%2 for j in range(phase)],setting='fixed_cadence',length=length,
        condition=f'fixed_cadence-c{capacity}-n{length}')
    base['environment_sha256']=digest({k:v for k,v in base.items() if k!='environment_sha256'});return base

def execute(row,method,seed,head):
    result,check=base_execute(row,method,seed,head)
    cues=[(f['evaluation']['phase'],f['event']['time'],f['event']['event_id']) for f in result['trace'] if f['event']['kind']=='context' and f['evaluation']['phase']>=0]
    episodes=[]
    for j,(phase,time,event_id) in enumerate(cues):
        qq=[q for q in result['queries'] if q['phase']==phase and q['target_query']]
        complete=j<len(cues)-1 or len(row['events'])-event_id==97
        episodes.append(dict(phase=phase,complete_episode=complete,**lag_episode(qq,time)))
    shifted=[e for e in episodes if e['phase']>0 and e['complete_episode']]
    assert shifted
    m=result['metrics'];m.update(adaptation_lag=float(np.mean([e['lag_queries'] for e in shifted])),
        adaptation_lag_time=float(np.mean([e['lag_time'] for e in shifted])),adaptation_score=float(np.mean([e['score'] for e in shifted])),
        adaptation_success=float(np.mean([e['recovered'] for e in shifted])),complete_shift_episodes=len(shifted),
        final_partial_episode_excluded=not episodes[-1]['complete_episode'])
    result['episodes']=episodes
    windows=result['window_metrics'];collapsed=[w['accuracy']<.1 for w in windows]
    m['sustained_low_accuracy']=any(a and b for a,b in zip(collapsed,collapsed[1:]))
    m['final_window_accuracy']=windows[-1]['accuracy'];m['final_stale_occupancy']=windows[-1]['stale_occupancy']
    return result,check
