"""Controlled distinct-key pressure and causal observe/query/feedback streams."""
from __future__ import annotations
import numpy as np
from src.memory.b1_environment import digest

PRESSURES={'low':.25,'medium':1.,'high':2.}


def generate(stories,vectors,world_seed,trajectory_seed,pressure='medium',capacity=8,shift=True,dynamics=None):
    if dynamics not in (None,'stable','slow_drift','rapid_shift'):raise ValueError('Invalid dynamics')
    if dynamics is not None:shift=dynamics!='stable'
    if pressure not in PRESSURES or capacity not in (8,16):raise ValueError('Invalid registered condition')
    pools={}
    for story in stories.values():
        if story['split']=='train':
            for e in story['events']:
                if e['type']=='observation':pools.setdefault(e['key'],{})[e['value']]=e['text']
    subjects=sorted(k for k,v in pools.items() if len(v)>1)
    if not subjects:raise ValueError('Public multi-version facts needed')
    rng=np.random.default_rng(world_seed);order=list(map(int,rng.permutation(28)))
    stable=order[:4];topics={k:i%2 for i,k in enumerate(stable)}
    a,b=order[4:8],order[8:12];noise_master=order[12:]
    topics.update({k:0 for k in a});topics.update({k:1 for k in b})
    topics.update({k:i%2 for i,k in enumerate(noise_master)})
    noise=noise_master[:int(8*PRESSURES[pressure])]
    subject={k:subjects[i%len(subjects)] for i,k in enumerate(order)}
    scope=np.random.default_rng(world_seed+500).normal(size=(28,32));scope/=np.linalg.norm(scope,axis=1,keepdims=True)
    current={k:str(rng.choice(sorted(pools[subject[k]]))) for k in order};versions={k:1 for k in order}
    def vec(k,text=None):
        v=np.asarray(vectors[text]) if text else np.mean([vectors[t] for t in pools[subject[k]].values()],axis=0)
        v=v+.35*scope[k];return (v/max(np.linalg.norm(v),1e-8)).tolist()
    def obs(k,t,change=False):
        if change:
            values=sorted(pools[subject[k]]);current[k]=values[(values.index(current[k])+1)%len(values)];versions[k]+=1
        text=pools[subject[k]][current[k]]
        return dict(kind='observe',time=t,key=k,topic=topics[k],item_id=f'{k}:{versions[k]}',
                    version=versions[k],content=text,vector=vec(k,text))
    # Identical initial content across pressures AND supported capacities.
    initial=[obs(k,0.) for k in stable+a[:2]+b[:2]]
    known={e['key']:e['item_id'] for e in initial}
    tape=[];cycle=0;catalog={str(k):dict(stable=k in stable,noise=k in noise_master,topic=topics[k]) for k in order}
    def emit(inp,phase,**evaluation):
        inp=dict(inp,event_id=len(tape));tape.append(dict(input=inp,evaluation=dict(phase=phase,**evaluation)))
        return inp['event_id']
    def cue(topic,phase):
        t=cycle*3.+.01
        emit(dict(kind='context',time=t,topic=topic,vector=np.mean([vec(k) for k in (a if topic==0 else b)],axis=0).tolist(),
                  content=f'Current work topic is {topic}; the other work topic is less relevant.'),phase,
             shifted=shift and phase>0)
    def triple(k,q,phase,change=False,shifted=False,round_index=-1):
        nonlocal cycle
        t=cycle*3.+.2
        e=obs(k,t,change);emit(e,phase,noise_observation=k in noise_master,revision=change,round=round_index)
        known[k]=e['item_id']
        if q not in known:raise AssertionError('Query asks for never-observed content')
        qid=emit(dict(kind='query',time=t+.9,key=q,topic=topics[q],vector=vec(q),content=f'Where is scoped subject {q}?'),
                 phase,target_id=known[q],stable=q in stable,target_query=q not in stable,shifted=shifted,round=round_index)
        emit(dict(kind='feedback',time=t+1.,key=q,topic=topics[q],vector=vec(q),query_event_id=qid),phase,round=round_index)
        cycle+=1
    # Knowledge introduction is also interleaved. Query only the supplied known
    # initial facts and never the immediately observed key. No distractor burst.
    cue(0,-1)
    scheduler=np.random.default_rng(trajectory_seed)
    for n,k in enumerate(scheduler.permutation(stable+a+b)):
        q=initial[n%len(initial)]['key']
        if q==k:q=initial[(n+1)%len(initial)]['key']
        triple(int(k),q,-1)
    stage_topics=[0,1,0] if shift else [0,0,0]
    # Pressure RNG is isolated: relevant observations, query targets and clocks
    # remain identical across pressure levels.
    noise_rng=np.random.default_rng(trajectory_seed+333)
    for phase,focus in enumerate(stage_topics):
        cue(focus,phase);active=stable+(a if focus==0 else b)
        phase_noise=[]
        while len(phase_noise)<32:phase_noise.extend(map(int,noise_rng.permutation(noise)))
        for round_index in range(4):
            observed=list(map(int,scheduler.permutation(active)))
            slots=list(map(int,scheduler.permutation([0]*8+[1]*8)))
            sq=list(map(int,scheduler.permutation(stable)))*2
            vq=list(map(int,scheduler.permutation(a if focus==0 else b)))*2
            queries=[v for pair in zip(sq,vq) for v in pair]
            # Construct the query schedule without inspecting which noise key is
            # chosen. Shift a within-role query cycle when it would copy the input.
            core_index=0;noises=phase_noise[round_index*8:(round_index+1)*8]
            noise_index=0
            for j,is_noise in enumerate(slots):
                if is_noise:k=noises[noise_index];noise_index+=1
                else:k=observed[core_index];core_index+=1
                q=queries[j]
                if q==k:
                    pool=stable if q in stable else (a if focus==0 else b)
                    q=pool[(pool.index(q)+1)%len(pool)]
                # True revisions happen only at observed arrivals. Stable facts
                # receive exactly the same number of refreshes as active volatile facts.
                change=(shift and phase>0 and not is_noise and k not in stable and round_index in (0,2))
                if dynamics=='slow_drift':
                    change=bool(phase>0 and not is_noise and k not in stable and k==(a if focus==0 else b)[round_index])
                triple(k,q,phase,change,shift and phase>0,round_index)
    row=dict(world_seed=world_seed,trajectory_seed=trajectory_seed,pressure=pressure,noise_ratio=PRESSURES[pressure],
             capacity=capacity,shift=shift,phases=3,stage_topics=stage_topics,active_memory_count=8,
             noise_memory_count=len(noise),stable_keys=stable,a_keys=a,b_keys=b,noise_keys=noise,
             catalog=catalog,initial_memory=initial,events=tape,
             query_sequence=[e['input']['key'] for e in tape if e['input']['kind']=='query'],
             pressure_definition='distinct noise keys /8active keys; arrival traffic held at8noise+8active observations per round',
             source='Public bAbI train facts, synthetic scoped schedule; no formal test data')
    if dynamics is not None:row['dynamics']=dynamics
    row['environment_sha256']=digest(row)
    return row


def audit_environment(row):
    known={i['key']:i['item_id'] for i in row['initial_memory']};keys=set(known);previous=None
    queries=0;feedbacks=0;phase_counts={};max_observation_run=0;run=0
    for index,e in enumerate(row['events']):
        p=e['input'];phase=e['evaluation']['phase'];counts=phase_counts.setdefault(str(phase),dict(active_observations=0,noise_observations=0,queries=0,feedbacks=0))
        assert p['event_id']==index and not {'stable','target_id','revision','noise_observation'} & p.keys()
        if previous is not None:assert p['time']>previous['time']
        if p['kind']=='observe':
            known[p['key']]=p['item_id'];keys.add(p['key']);run+=1;max_observation_run=max(run,max_observation_run)
            counts['noise_observations' if e['evaluation']['noise_observation'] else 'active_observations']+=1
            if p['key'] in row['stable_keys']:assert p['version']==1
        else:run=0
        if p['kind']=='query':
            assert previous['kind']=='observe' and previous['key']!=p['key']
            assert e['evaluation']['target_id']==known[p['key']]
            queries+=1;counts['queries']+=1
        if p['kind']=='feedback':
            assert previous['kind']=='query' and p['query_event_id']==previous['event_id']
            assert 'value' not in p  # Actual feedback is resolved per-policy after answering.
            feedbacks+=1;counts['feedbacks']+=1
        previous=p
    assert queries==feedbacks==204 and max_observation_run==1
    assert row['noise_memory_count']/row['active_memory_count']==row['noise_ratio']
    for phase in range(3):
        c=phase_counts[str(phase)];assert c['active_observations']==c['noise_observations']==32 and c['queries']==64
    return dict(passed=True,world_seed=row['world_seed'],pressure=row['pressure'],capacity=row['capacity'],
                active_memory_count=8,noise_memory_count=row['noise_memory_count'],noise_ratio=row['noise_ratio'],
                max_consecutive_observations=max_observation_run,phase_counts=phase_counts,
                input_sha256=digest([e['input'] for e in row['events']]),query_sha256=digest(row['query_sequence']),
                initial_sha256=digest(row['initial_memory']),queries=queries,feedbacks=feedbacks)
