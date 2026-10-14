import numpy as np
import pytest
from src.memory.agent_memory import AgentMemory

class Encoder:
    signature='test-fixed'
    def encode(self,texts):
        x=np.zeros((len(texts),384),np.float32);x[:,0]=1;return x

def test_versions_labels_snapshot_and_budget(tmp_path):
    a=AgentMemory(Encoder(),6000)
    a.observe('answer_secret','2024/01/01 12:00',[dict(role='user',content='I live in Paris.',has_answer='SECRET')])
    a.observe('other','2024/01/02 12:00',[dict(role='user',content='I moved to Rome.')])
    ctx=a.query('Where did I live?','2024/01/03')['context']
    assert 'Paris' in ctx and 'Rome' in ctx and 'SECRET' not in ctx and 'answer_secret' not in ctx
    a.checkpoint(tmp_path/'state.json');b=AgentMemory(Encoder(),6000);b.restore(tmp_path/'state.json')
    assert b.query('Where did I live?','2024/01/03')==a.query('Where did I live?','2024/01/03')
    for i in range(3,15):b.observe(str(i),f'2024/01/{i:02d}',[dict(role='user',content=f'New text {i}')])
    assert b.payload_bytes<=6000 and not b.store.evicted

def test_chronology_and_snapshot_mismatch(tmp_path):
    a=AgentMemory(Encoder(),4000);a.observe('s','2024/02/01',[dict(role='user',content='hello')])
    with pytest.raises(ValueError):a.observe('s','2024/01/01',[dict(role='user',content='hello')])
    a.checkpoint(tmp_path/'state.json')
    with pytest.raises(ValueError):AgentMemory(Encoder(),5000).restore(tmp_path/'state.json')
