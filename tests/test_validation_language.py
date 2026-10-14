import copy
import numpy as np
import pytest
from src.memory.validation_language import LanguageMemory
from src.memory.validation_matrix import BASELINES
from tests.test_b1_sanity import zero_head
from scripts.validation_language import chunks,messages

@pytest.mark.parametrize('method',BASELINES)
def test_language_capacity_no_archive_retrieval_and_no_gold_feedback(method):
    memory=LanguageMemory(method,11,zero_head(),capacity=2)
    for n in range(9):
        r=dict(key=n,item_id=f'{n}:1',role='user',text=f'Person {n} lives in city {n}.',time=float(n),
               origin=dict(source=str(n),date=str(n),turn=0,dia_id=None,start=0))
        v=np.zeros(384,dtype=np.float32);v[n]=1;memory.observe(r,v)
    assert memory.policy.calls['feedback']==0
    ids=set(memory.policy.items)
    q=memory.retrieve('Where does person 0 live?',np.eye(1,384,0,dtype=np.float32)[0],10.,top_k=5)
    assert {s['item_id'] for s in q['sources']}<=ids and len(ids)<=2
    assert set(memory.vectors)==ids

def test_observed_feedback_is_explicit_and_question_labels_never_used():
    records=chunks([dict(time=0.,id='session',date='date',turns=[dict(role='assistant',content='Known evidence',feedback=0.)])])
    p=LanguageMemory('adaptive_evolution',11,zero_head(),capacity=2);p.observe(records[0],np.ones(384,dtype=np.float32)/np.sqrt(384))
    assert p.policy.calls['feedback']==1 and p.policy.last_feedback==0
    case=dict(dataset='LongMemEval',question='Question?',question_date='date',evaluation={'answer':'SECRET_GOLD'})
    assert 'SECRET_GOLD' not in str(messages(case,dict(context='Evidence')))

def test_chunking_is_lossless_allowlisted_and_causal():
    t='x'*2501
    r=chunks([dict(time=9.,id='b',date='b',turns=[dict(role='user',content=t,has_answer=True)]),dict(time=2.,id='a',date='a',turns=[dict(role='assistant',content='early')])])
    assert ''.join(x['text'][6:] for x in r if x['role']=='user')==t
    assert r[0]['time']==0 and r[-1]['time']==7
    assert 'has_answer' not in str(r)
