"""Frozen-core adapters for B2-B7. No optimizer or additional neural parameters."""
import hashlib,math
import numpy as np
from src.memory.b1_attribution import AttributionPolicy
from src.memory.b1_environment import digest

BASELINES=('fixed_decay','frequency','lru','static_state_adaptive','replacement_matched','adaptive_rule','adaptive_evolution')


class MatrixPolicy(AttributionPolicy):
    def __init__(self,method,capacity,seed,head=None,admission=1.,minimum_age=0.):
        self.variant=method;self.suppress_flow=False;self.current_event={};self.last_access={}
        self.minimum_age=minimum_age;self.frozen_states={}
        base='fixed_decay' if method=='lru' else 'adaptive_evolution' if method.startswith(('no_','event_only','discrete_euler','query_conditioned')) else method
        super().__init__(base,capacity,seed,head,admission)

    @property
    def adaptive(self):return not self.suppress_flow and super().adaptive

    def mask(self):
        neutral={'no_importance':('I',1.),'no_consistency':('C',1.),'no_usage':('U',0.),'no_time':('T',0.)}
        if self.variant in neutral:
            field,value=neutral[self.variant]
            for i in self.items.values():setattr(i,field,value)

    def update_state(self,event):
        self.current_event=event
        if self.variant in ('event_only','discrete_euler'):
            dt=event['time']-self.now
            if self.variant=='discrete_euler':
                steps=max(1,math.ceil(dt/min(1.,.24/max(self.held_k,1e-8))))
                h=dt/steps
                for _ in range(steps):
                    for i in self.items.values():
                        i.I+=h*self.held_k*(.5+self.held_w*(i.topic==self.focus)-i.I)
                        i.C+=h*4*self.held_k*(float(self.ledger.get(i.key)==i.item_id)-i.C)
                        i.U*=1-h*self.held_k/20;i.T+=h
                self.counts['item_state_updates']+=steps*len(self.items)
            self.suppress_flow=True
            try:super().update_state(event)
            finally:self.suppress_flow=False
        else:super().update_state(event)
        if self.variant=='query_conditioned' and event['kind']=='query':self.focus=event['topic']
        self.mask()

    def observe(self,event):
        result=super().observe(event);self.last_access[event['item_id']]=self.now;self.mask();return result

    def retrieve(self,query):
        result=super().retrieve(query)
        if result:self.last_access[result['item_id']]=self.now
        return result

    def feedback(self,item_id,value):
        if self.current_event.get('feedback_flip',False):value=1-value
        super().feedback(item_id,value);self.mask()

    def score(self,item):
        if self.variant=='lru':return self.last_access.get(item.item_id,item.observed_at)
        if self.method=='replacement_matched' and item.item_id==self.pending:
            residents=[i for i in self.items.values() if i.item_id!=self.pending]
            victim=min(residents,key=lambda i:(-self.decay*(self.now-i.observed_at),self._tie(i)))
            if self.now-victim.birth<self.minimum_age:return -1e100
        return super().score(item)


def generate(stories,vectors,world_seed,trajectory_seed,capacity=8,noise_count=16,rounds=4,
             repetitions=1,cyclic_probe=False,feedback_noise=0.,misleading=False,versions=True):
    """Independent public-fact stream. Shared RNG tapes never use outcomes."""
    pools={}
    for story in stories.values():
        if story['split']=='train':
            for e in story['events']:
                if e['type']=='observation':pools.setdefault(e['key'],{})[e['value']]=e['text']
    subjects=sorted(k for k,v in pools.items() if len(v)>1)
    # Fixed master keys even as the requested noise pool changes.
    rng=np.random.default_rng(world_seed);order=list(map(int,rng.permutation(140)))
    stable,a,b=order[:4],order[4:8],order[8:12];noise_master=order[12:]
    if not 1<=noise_count<=128:raise ValueError('Noise count outside fixed master pool')
    noise=noise_master[:noise_count];topics={k:i%2 for i,k in enumerate(stable+noise_master)}
    topics.update({k:0 for k in a});topics.update({k:1 for k in b})
    subject={k:subjects[i%len(subjects)] for i,k in enumerate(order)}
    scope=np.random.default_rng(world_seed+500).normal(size=(140,32));scope/=np.linalg.norm(scope,axis=1,keepdims=True)
    current={k:str(rng.choice(sorted(pools[subject[k]]))) for k in order};version={k:1 for k in order}
    cache={}
    def vec(k,text=None):
        ck=(k,text)
        if ck not in cache:
            v=np.asarray(vectors[text]) if text else np.mean([vectors[t] for t in pools[subject[k]].values()],axis=0)
            v=v+.35*scope[k];cache[ck]=(v/max(np.linalg.norm(v),1e-8)).tolist()
        return cache[ck]
    def observation(k,t,revision=False):
        if revision:
            values=sorted(pools[subject[k]]);current[k]=values[(values.index(current[k])+1)%len(values)];version[k]+=1
        text=pools[subject[k]][current[k]]
        return dict(kind='observe',time=t,key=k,topic=topics[k],item_id=f'{k}:{version[k]}',version=version[k],content=text,vector=vec(k,text))
    initial=[observation(k,0.) for k in stable[:2]+a[:2]]
    known={e['key']:e['item_id'] for e in initial};events=[];cycle=0
    schedule=np.random.default_rng(trajectory_seed);nrng=np.random.default_rng(trajectory_seed+333)
    frng=np.random.default_rng(trajectory_seed+777)
    def emit(event,phase,**labels):
        event=dict(event,event_id=len(events));events.append(dict(input=event,evaluation=dict(phase=phase,**labels)));return event['event_id']
    def cue(focus,phase):
        shown=1-focus if misleading else focus
        emit(dict(kind='context',time=3*cycle+.01,topic=shown,vector=np.mean([vec(k) for k in (a if focus==0 else b)],axis=0).tolist()),phase,shifted=phase>0)
    def query(q,phase,time,probe=False):
        qid=emit(dict(kind='query',time=time,key=q,topic=topics[q],vector=vec(q)),phase,target_id=known[q],stable=q in stable,
                 target_query=q not in stable,shifted=phase>0,pre_reminder=probe)
        # Same random variates across policies and corruption strengths.
        flip=bool(frng.random()<feedback_noise)
        emit(dict(kind='feedback',time=time+.1,key=q,topic=topics[q],vector=vec(q),query_event_id=qid,feedback_flip=flip),phase)
    def triple(k,q,phase,revision=False):
        nonlocal cycle
        e=observation(k,3*cycle+.2,revision);known[k]=e['item_id']
        emit(e,phase,revision=revision,noise_observation=k in noise_master)
        query(q,phase,3*cycle+1.1);cycle+=1
    cue(0,-1)
    for n,k in enumerate(schedule.permutation(stable+a+b)):
        q=initial[n%4]['key']
        if q==k:q=initial[(n+1)%4]['key']
        triple(int(k),q,-1)
    for phase,focus in enumerate((0,1,0)):
        cue(focus,phase);active=stable+(a if focus==0 else b)
        if cyclic_probe and phase==2:
            # No observation until every returning topic fact has been probed.
            for q in a:
                query(q,phase,3*cycle+1.1,True);cycle+=1
        phase_noise=[]
        while len(phase_noise)<8*rounds*repetitions:phase_noise.extend(map(int,nrng.permutation(noise)))
        for round_index in range(rounds):
            observed=list(map(int,schedule.permutation(active)))*repetitions
            slots=list(map(int,schedule.permutation([0]*(8*repetitions)+[1]*(8*repetitions))))
            sq=list(map(int,schedule.permutation(stable)))*(2*repetitions)
            vq=list(map(int,schedule.permutation(a if focus==0 else b)))*(2*repetitions)
            qs=[v for pair in zip(sq,vq) for v in pair];ni=ci=0;changed=set()
            for j,is_noise in enumerate(slots):
                if is_noise:k=phase_noise[round_index*8*repetitions+ni];ni+=1
                else:k=observed[ci];ci+=1
                q=qs[j]
                if q==k:
                    pool=stable if q in stable else (a if focus==0 else b);q=pool[(pool.index(q)+1)%4]
                revision=bool(versions and phase>0 and not is_noise and k not in stable and round_index in (0,2) and k not in changed)
                if revision:changed.add(k)
                triple(k,q,phase,revision)
    row=dict(world_seed=world_seed,trajectory_seed=trajectory_seed,capacity=capacity,noise_memory_count=noise_count,active_memory_count=8,
             noise_ratio=noise_count/8,pressure='high',shift=True,phases=3,stage_topics=[0,1,0],stable_keys=stable,a_keys=a,b_keys=b,noise_keys=noise,
             initial_memory=initial,events=events,query_sequence=[e['input']['key'] for e in events if e['input']['kind']=='query'],
             feedback_noise=feedback_noise,misleading=misleading,cyclic_probe=cyclic_probe,versions=versions)
    row['environment_sha256']=digest(row);return row
