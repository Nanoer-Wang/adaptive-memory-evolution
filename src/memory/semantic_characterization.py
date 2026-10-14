"""Public MemOps-derived controlled migration; gold keys never enter adapter."""
import copy,hashlib,math,time
import numpy as np
from src.memory.characterization import StrengthPolicy
from src.memory.consolidation import project
from src.memory.b1_environment import digest
from src.evaluation.b1_interleaved import lag_episode

DOMAINS={
 'D17':('work_laptop_default_browser','Which browser is my default on my work laptop?'),
 'D18':('gboard_keyboard_height','How high do I want the Gboard keyboard to be?'),
 'D20':('favorite_author','Who is my favorite writer at the moment?'),
 'D22':('preferred_hotel_brand','Which hotel chain do I currently prefer?'),
 'B05':('daughter_first_choice_name','What name is our first choice for our daughter?'),
 'B30':('marcus_preferred_name','What name should I use for my college friend Marcus?'),
 'C05':('iceland_return_date','When do we fly home from Iceland?'),
 'C07':('kyoto_hotel','At which hotel am I staying in Kyoto?'),
 'C10':('cpa_far_exam_date','When is my FAR accounting exam scheduled?'),
 'D19':('google_one_storage_plan','How much storage is in my Google One plan?')}

class SemanticStrengthPolicy(StrengthPolicy):
    def topic_similarity(self,item):
        memory=self.addressing
        return max(0.,float(memory.vectors[item.item_id]@memory.context)) if memory.context is not None else 0.

class SemanticMemory:
    def __init__(self,method,capacity,seed,head,threshold=.65):
        self.policy=SemanticStrengthPolicy('full_1' if method=='adaptive_evolution' else method,capacity,seed,head)
        self.policy.addressing=self
        self.vectors={};self.context=None;self.serial=0;self.threshold=threshold
        self.trace=[];self.last_answer=None

    def event(self,raw):
        # Construct an explicit allowlist. Evaluator identity/version/value cannot
        # leak even if a caller adds them to the wrapper by mistake.
        v=np.asarray(raw['vector'],dtype=np.float32)
        e=dict(kind=raw['kind'],time=float(raw['time']),vector=project(v).tolist(),
               topic=int(self.context is not None and float(v@self.context)>=.35))
        return e,v

    def process(self,raw,feedback=None):
        p=self.policy;e,v=self.event(raw);kind=e['kind'];action=None;answer=None;similarity=None
        if kind=='observe':
            item_id=hashlib.sha256(raw['text'].encode()).hexdigest()[:20]
            if item_id in p.items:key=p.items[item_id].key;version=p.items[item_id].version
            else:
                nearest=max(self.vectors,key=lambda k:float(self.vectors[k]@v)) if self.vectors else None
                similarity=float(self.vectors[nearest]@v) if nearest else -1.
                if nearest is not None and similarity>=self.threshold:
                    key=p.items[nearest].key;version=max(i.version for i in p.items.values() if i.key==key)+1
                else:self.serial+=1;key=self.serial;version=1
            e.update(key=key,item_id=item_id,version=version,content=raw['text'])
        elif kind=='query':
            nearest=max(self.vectors,key=lambda k:(float(self.vectors[k]@v),k)) if self.vectors else None
            similarity=float(self.vectors[nearest]@v) if nearest else None
            e['key']=p.items[nearest].key if nearest else -1
        else:e['key']=-1
        p.update_state(e);flow=p.snapshot()
        if kind=='context':self.context=v.copy();p.focus=1
        elif kind=='observe':
            p.observe(e);self.vectors[e['item_id']]=v;action=p.evict()
            if action['victim'] is not None:self.vectors.pop(action['victim'])
        elif kind=='query':answer=p.retrieve(e);self.last_answer=answer
        elif kind=='feedback':
            if feedback is None:raise ValueError('Explicit feedback required')
            p.feedback(self.last_answer['item_id'] if self.last_answer else None,feedback)
        assert set(self.vectors)==set(p.items) and len(p.items)<=p.capacity
        frame=dict(event=e,source_text=raw['text'],flow_state=flow,resident=p.snapshot(),answer=answer,action=action,similarity=similarity,
            feedback=feedback,counts=dict(p.counts))
        self.trace.append(frame);return answer

def generate(corpus,world,capacity,task):
    rng=np.random.default_rng(world);schedule=np.random.default_rng(world+700000)
    pref=list(DOMAINS)[:6];rng.shuffle(pref);a=pref[:4];b=list(DOMAINS)[6:]
    control=list(corpus['stable']);rng.shuffle(control);stable=control[:4]
    noise=[k for k in corpus['noise'] if k not in stable];rng.shuffle(noise);noise=noise[:16]
    truth={};events=[];clock=0
    def emit(kind,text,phase,**evaluation):
        nonlocal clock
        events.append(dict(input=dict(kind=kind,time=float(clock),text=text,vector=corpus['vectors'][text]),evaluation=dict(phase=phase,**evaluation)));clock+=1
    def observe(key,index,phase,historical=False):
        op=corpus['facts'][key][min(index,len(corpus['facts'][key])-1)];text=op['text']
        if historical:text='Historical note, no longer current: '+text
        else:truth[key]=op['value']
        emit('observe',text,phase,key=key,value=op['value'],changes_truth=not historical,stable=key in stable)
    def query(key,phase,probe=False):
        text=corpus['queries'][key]
        emit('query',text,phase,key=key,target_value=truth[key],stable=key in stable,target_query=key not in stable,pre_reminder=probe)
        emit('feedback',text,phase)
    for key in stable+a+b:observe(key,0,-1)
    for phase in range(3):
        active=b if task in ('preference_revival','temporary_task') and phase==1 else a
        if task=='temporary_task' and phase==2:active=b
        cue='My current questions concern: '+' '.join(corpus['queries'][k] for k in active)
        emit('context',cue,phase)
        if phase==2 and task=='preference_revival':
            for k in a:query(k,phase,True)
        # Shared16core/16interference observations, queries alternately stable/active.
        for round_index in range(3):
            obs=list(schedule.permutation(stable+active))*2;distractors=list(schedule.permutation(noise))
            slots=list(schedule.permutation([False]*16+[True]*16));ci=ni=0
            sq=list(schedule.permutation(stable))*4;vq=list(schedule.permutation(active))*4
            qq=[k for pair in zip(sq,vq) for k in pair]
            for j,is_noise in enumerate(slots):
                if is_noise:k=str(distractors[ni]);ni+=1
                else:k=str(obs[ci]);ci+=1
                index=0 if k in stable or k in noise else (min(phase,2) if task in ('preference_revival','temporary_task') else int(phase>0))
                # Dormant A does not change while B is active; returning reminders
                # use the released third confirmed statement (often a revival).
                historical=task=='contradictory_preference' and phase>0 and k in active and j%7==0
                observe(k,0 if historical else index,phase,historical)
                query(str(qq[j]),phase)
    row=dict(world_seed=world,trajectory_seed=world+700000,capacity=capacity,task=task,stable_keys=stable,active_a=a,active_b=b,
             events=events,initial_memory=[],condition=f'{task}-c{capacity}')
    row['environment_sha256']=digest(row);return row

def execute(row,method,seed,head):
    memory=SemanticMemory(method,row['capacity'],seed,head);truth={};origins={};queries=[];phase_starts={};last_correct=False
    empty=hashlib.sha256(b'empty initial memory').hexdigest();elapsed=0.;mismerge=merges=0;phase_truth={}
    for event_id,w in enumerate(row['events']):
        e=w['input'];label=w['evaluation'];kind=e['kind']
        if kind=='context':phase_starts[label['phase']]=e['time'];phase_truth[label['phase']]=dict(truth)
        if kind=='observe':
            item_id=hashlib.sha256(e['text'].encode()).hexdigest()[:20];origins[item_id]=dict(key=label['key'],value=label['value'])
            if label['changes_truth']:truth[label['key']]=label['value']
        before={i.item_id:i.key for i in memory.policy.items.values()}
        start=time.perf_counter();answer=memory.process(e,float(last_correct) if kind=='feedback' else None);elapsed+=time.perf_counter()-start
        f=memory.trace[-1];f.update(evaluation=label,event_id=event_id)
        if kind=='observe':
            inferred=f['event']['key'];same=[i for i,k in before.items() if k==inferred]
            if same:merges+=1;mismerge+=any(origins[i]['key']!=label['key'] for i in same)
        if kind=='query':
            origin=origins.get(answer['item_id']) if answer else None
            correct=origin is not None and origin['key']==label['key'] and origin['value']==label['target_value'];last_correct=correct
            stale=origin is not None and origin['key']==label['key'] and origin['value']!=label['target_value']
            available=any(origins[i]['key']==label['key'] and origins[i]['value']==label['target_value'] for i in memory.policy.items)
            queries.append(dict(event_id=event_id,time=e['time'],**label,correct=correct,stale=stale,hit=answer is not None,
                wrong_fact=origin is not None and origin['key']!=label['key'],resident_correct_available=available,answer=answer))
        f['obsolete_resident_fraction']=sum(origins[i]['value']!=truth.get(origins[i]['key']) for i in memory.policy.items)/max(1,len(memory.policy.items))
    scored=[q for q in queries if q['phase']>=0];stable=[q for q in scored if q['stable']]
    phases=(1,) if row['task'] in ('preference_update','contradictory_preference') else (1,2)
    episodes=[]
    for phase in phases:
        qq=[q for q in scored if q['phase']==phase and q['target_query']]
        version_window=row['task'] in ('preference_update','contradictory_preference') or (row['task']=='temporary_task' and phase==2)
        start=phase_starts[phase]
        if version_window:
            # Count opportunities only after a fact's new value is observed;
            # old-but-still-valid pre-update answers are not fast adaptation.
            qq=[q for q in qq if q['target_value']!=phase_truth[phase].get(q['key'])]
            keys={q['key'] for q in qq}
            times=[w['input']['time'] for w in row['events'] if w['input']['kind']=='observe' and w['evaluation']['phase']==phase
                   and w['evaluation']['changes_truth'] and w['evaluation']['key'] in keys
                   and w['evaluation']['value']!=phase_truth[phase].get(w['evaluation']['key'])]
            if times:start=min(times)
        assert qq,'Each planned shift must provide eligible target queries'
        episodes.append(dict(phase=phase,window='observed_new_values' if version_window else 'demand_shift',**lag_episode(qq,start)))
    p=memory.policy;mean=lambda vals:float(np.mean(vals)) if vals else None
    metrics=dict(method=method,world_seed=row['world_seed'],policy_seed=seed,capacity=row['capacity'],condition=row['condition'],task=row['task'],
        accuracy=mean([q['correct'] for q in scored]),stale_error=mean([q['stale'] for q in scored]),stable_retention=mean([q['correct'] for q in stable]),
        adaptation_lag=mean([q['lag_queries'] for q in episodes]),adaptation_score=mean([q['score'] for q in episodes]),
        wrong_fact_error=mean([q['wrong_fact'] for q in scored]),resident_correct_coverage=mean([q['resident_correct_available'] for q in scored]),
        pre_reminder_recall=mean([q['correct'] for q in scored if q['pre_reminder']]),mismerge_rate=mismerge/max(1,merges),semantic_merges=merges,
        eviction_count=p.counts['resident_evictions'],policy_seconds=elapsed,
        time_weighted_resident_age=p.age_integral/max(1,p.slot_time),initial_sha256=empty,environment_sha256=row['environment_sha256'],
        queries=len(scored),peak_slots=max(len(t['resident']) for t in memory.trace),**p.counts)
    result=dict(metrics=metrics,trace=memory.trace,queries=queries,episodes=episodes,lifetimes=p.finish_lifetimes(),origins=origins)
    audit(row,result);return result

def audit(row,result):
    live={};vectors={};origins={};truth={};queries=iter(result['queries']);nquery=0;last_correct=None
    for w,f in zip(row['events'],result['trace'],strict=True):
        e=f['event'];kind=e['kind']
        assert not any(k in e for k in ('target_value','changes_truth','stable','gold_key'))
        if kind=='observe':
            label=w['evaluation'];origins[e['item_id']]=dict(key=label['key'],value=label['value'])
            if label['changes_truth']:truth[label['key']]=label['value']
            vectors[e['item_id']]=np.asarray(w['input']['vector'],dtype=np.float32)
            live[e['item_id']]=e['content'];victim=f['action']['victim']
            if victim is not None:assert victim in live;del live[victim];del vectors[victim]
        if kind=='query':
            q=next(queries);nquery+=1
            if f['answer']:assert f['answer']['item_id'] in live
            assert q['target_value']==w['evaluation']['target_value']==truth[w['evaluation']['key']]
            v=np.asarray(w['input']['vector'],dtype=np.float32)
            nearest=max(vectors,key=lambda k:(float(vectors[k]@v),k)) if vectors else None
            states={i['item_id']:i for i in f['resident']}
            chosen=max((i for i in states.values() if i['key']==states[nearest]['key']),key=lambda i:i['version']) if nearest else None
            assert (chosen['item_id'] if chosen else None)==(f['answer']['item_id'] if f['answer'] else None)
            origin=origins.get(chosen['item_id']) if chosen else None
            last_correct=origin is not None and origin['key']==q['key'] and origin['value']==q['target_value']
            assert q['correct']==last_correct
            assert q['stale']==(origin is not None and origin['key']==q['key'] and origin['value']!=q['target_value'])
        if kind=='feedback':assert f['feedback']==float(last_correct)
        assert set(live)=={i['item_id'] for i in f['resident']} and len(live)<=row['capacity']
        assert all(i['content']==live[i['item_id']] and all(math.isfinite(i[z]) for z in ('I','C','U','T')) for i in f['resident'])
    assert nquery==len(result['queries'])
    return dict(passed=True,frames=len(result['trace']),gold_not_in_policy=True,resident_only_retrieval=True)
