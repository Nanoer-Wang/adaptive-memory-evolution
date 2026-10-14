"""Rate-only interventions on frozen B1 dynamics; no training or new network."""
import copy, math
import numpy as np
from src.memory.validation_matrix import MatrixPolicy, generate
from src.memory.b1_environment import digest
from src.evaluation.b1_interleaved import run, audit

GRID=(0.,.25,.5,.75,1.,1.5,2.)
STATIC=('fixed_decay','frequency','lru','static_state_adaptive')

def policies(batch):
    families=('full',) if batch=='C1' else ('full','I','C','U') if batch=='C2' else ('full',)
    grid=GRID if batch in ('C1','C2') else (0.,.25,1.,2.)
    return list(STATIC)+[f'{family}_{a:g}' for family in families for a in grid]

class StrengthPolicy(MatrixPolicy):
    def __init__(self,name,capacity,seed,head=None):
        self.name=name
        self.strength=(1.,1.,1.)
        if name not in STATIC:
            family,amount=name.split('_');amount=float(amount)
            self.strength=tuple(amount if family in ('full',field) else 0. for field in ('I','C','U'))
        super().__init__(name if name in STATIC else 'adaptive_evolution',capacity,seed,
                         head if name not in ('fixed_decay','frequency','lru') else None)

    def topic_similarity(self,item):return float(item.topic==self.focus)

    def update_state(self,event):
        dt=float(event['time'])-self.now
        if dt<0 or not math.isfinite(dt):raise ValueError('Invalid event time')
        if self.adaptive:
            aI,aC,aU=self.strength
            for item in self.items.values():
                target=.5+self.held_w*self.topic_similarity(item)
                valid=float(self.ledger.get(item.key)==item.item_id)
                if aI:item.I=target+(item.I-target)*math.exp(-self.held_k*dt*aI)
                if aC:item.C=valid+(item.C-valid)*math.exp(-4*self.held_k*dt*aC)
                if aU:item.U*=math.exp(-self.held_k*dt/20*aU)
                item.T+=dt
            self.counts['item_state_updates']+=len(self.items)
        self.suppress_flow=True
        try:super().update_state(event)
        finally:self.suppress_flow=False

def environment(stories,vectors,world,capacity,setting,length=None):
    # Both settings mix four stable and four currently demanded volatile facts.
    rounds=4 if length is None else max(4,math.ceil(length/144))
    row=generate(stories,vectors,world,world+600000,capacity=capacity,
        noise_count=8 if setting=='mixed' else 16,rounds=rounds,versions=setting!='mixed')
    if length is not None:
        # Allocate the requested event budget over all three phases, retaining
        # their onset cues and complete observation/query/feedback triples.
        warm=[w for w in row['events'] if w['evaluation']['phase']==-1]
        budget=length-len(warm);per=budget//3
        chosen=warm[:]
        for phase in range(3):
            n=per+(phase<budget%3)
            phase_events=[w for w in row['events'] if w['evaluation']['phase']==phase]
            # A partial terminal triple is allowed; query truth still causal.
            chosen+=phase_events[:n]
        assert len(chosen)==length
        old_to_new={w['input']['event_id']:j for j,w in enumerate(chosen)}
        for j,w in enumerate(chosen):
            e=w['input'];e['event_id']=j;e['time']=float(j)
            if e['kind']=='feedback':e['query_event_id']=old_to_new[e['query_event_id']]
        row['events']=chosen
        row['query_sequence']=[w['input']['key'] for w in chosen if w['input']['kind']=='query']
        # Truth in later phases must reflect only retained observations, never
        # skipped tail revisions from the longer source stream.
        ledger={e['key']:e['item_id'] for e in row['initial_memory']}
        for w in chosen:
            e=w['input']
            if e['kind']=='observe':ledger[e['key']]=e['item_id']
            if e['kind']=='query':w['evaluation']['target_id']=ledger[e['key']]
    row.update(setting=setting,length=length,condition=f'{setting}-c{capacity}'+(f'-n{length}' if length else ''))
    row['environment_sha256']=digest({k:v for k,v in row.items() if k!='environment_sha256'})
    return row

def execute(row,name,seed,head):
    policy=StrengthPolicy(name,row['capacity'],seed,head)
    result=run(row,name,seed,policy=policy);check=audit(row,result)
    shifted=[e for e in result['episodes'] if e['applicable']]
    m=result['metrics'];m.update(condition=row['condition'],setting=row['setting'],length=len(row['events']),
        adaptation_score=float(np.mean([e['score'] for e in shifted])),
        strengths=list(policy.strength) if name not in STATIC else None,
        auxiliary_ledger_entries=len(policy.ledger),auxiliary_query_history_entries=len(policy.demand),
        max_state={f:max(i[f] for t in result['trace'] for i in t['resident']) for f in ('I','C','U','T')})
    # Equal-width event windows, not selected on where performance improves.
    ledger={e['key']:e['item_id'] for e in row['initial_memory']}
    for frame in result['trace']:
        e=frame['event']
        if e['kind']=='observe':ledger[e['key']]=e['item_id']
        frame['obsolete_resident_fraction']=sum(ledger[i['key']]!=i['item_id'] for i in frame['resident'])/max(1,len(frame['resident']))
    bins=[]
    for start in range(0,len(row['events']),100):
        qq=[q for q in result['queries'] if start<=q['event_id']<start+100 and q['phase']>=0]
        if not qq:continue
        stable=[q for q in qq if q['stable']]
        frames=result['trace'][start:start+100]
        bins.append(dict(start=start,end=min(start+100,len(row['events'])),queries=len(qq),
            accuracy=float(np.mean([q['correct'] for q in qq])),stale_error=float(np.mean([q['stale'] for q in qq])),
            stable_retention=float(np.mean([q['correct'] for q in stable])) if stable else None,
            mean_states={f:float(np.mean([i[f] for t in frames for i in t['resident']])) for f in ('I','C','U','T')},
            stale_occupancy=float(np.mean([t['obsolete_resident_fraction'] for t in frames]))))
    result['window_metrics']=bins
    return result,check
