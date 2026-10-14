import copy
import numpy as np
from src.memory.semantic_characterization import SemanticMemory

def head():return {'weight':np.zeros((3,40),dtype=np.float32),'bias':np.zeros(3,dtype=np.float32)}
def vector(j):
    x=np.zeros(384,dtype=np.float32);x[j]=1;return x.tolist()
def event(text,j,t,kind='observe'):return dict(kind=kind,time=t,text=text,vector=vector(j))

def test_gold_fields_cannot_change_policy():
    a=SemanticMemory('adaptive_evolution',2,11,head());b=SemanticMemory('adaptive_evolution',2,11,head())
    e=event('Favorite writer is Le Guin',0,0);bad=dict(e,key='oracle',topic=42,version=999,target_value='Le Guin',stable=True)
    a.process(e);b.process(bad);assert a.trace==b.trace

def test_semantic_retrieval_has_no_evicted_vectors():
    m=SemanticMemory('fixed_decay',1,11,head());m.process(event('Old preference',0,0));m.process(event('Different topic',1,1))
    assert len(m.vectors)==len(m.policy.items)==1
    result=m.process(event('Old preference question',0,2,'query'))
    assert result['content']=='Different topic' # resident-only miss, never archive lookup

def test_identical_text_refresh_and_semantic_versions():
    m=SemanticMemory('fixed_decay',3,11,head());m.process(event('Browser Firefox',0,0));m.process(event('Browser Firefox',0,1))
    assert len(m.policy.items)==1
    m.process(event('Browser Edge',0,2));items=list(m.policy.items.values())
    assert items[0].key==items[1].key and items[1].version==2
    result=m.process(event('Which browser?',0,3,'query'));assert result['content']=='Browser Edge'

def test_new_context_is_not_integrated_retroactively():
    m=SemanticMemory('adaptive_evolution',3,11,head());m.process(event('Browser Firefox',0,0));m.process(event('Browser questions',0,0,'context'))
    n=copy.deepcopy(m)
    m.process(event('Now another topic',1,10,'context'));n.process(event('Still browser',0,10,'context'))
    assert m.policy.snapshot()==n.policy.snapshot() # changed cue applies only forward
    m.process(event('Next observation',1,20));n.process(event('Next observation',1,20))
    old=next(i for i in m.policy.items.values() if i.content=='Browser Firefox')
    other=next(i for i in n.policy.items.values() if i.content=='Browser Firefox')
    assert old.I<other.I

def test_complete_stream_counts_wrong_fact_separately_from_stale():
    from src.memory.semantic_characterization import execute
    events=[]
    def add(kind,text,j,phase,**labels):
        events.append(dict(input=event(text,j,len(events),kind),evaluation=dict(phase=phase,**labels)))
    for phase in range(3):
        add('context','Browser demand',0,phase)
        key='browser' if phase<2 else 'other';value='Firefox' if phase==0 else 'Edge' if phase==1 else 'unrelated'
        add('observe',value,0 if phase<2 else 1,phase,key=key,value=value,changes_truth=True,stable=False)
        for _ in range(6):
            add('query','Which browser?',0,phase,key='browser',target_value='Firefox' if phase==0 else 'Edge',stable=False,target_query=True,pre_reminder=False)
            add('feedback','Which browser?',0,phase)
    row=dict(capacity=1,world_seed=1,condition='test',task='preference_update',environment_sha256='test',events=events)
    result=execute(row,'fixed_decay',11,head())
    assert result['metrics']['rejected_writes']==0
    assert result['metrics']['stale_error']==0 and result['metrics']['wrong_fact_error']==1/3
    assert len(result['episodes'])==1 and result['episodes'][0]['phase']==1
    assert result['episodes'][0]['window']=='observed_new_values'
