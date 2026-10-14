import numpy as np
import pytest
from src.memory.consolidation_agent import ConsolidatingAgentMemory


class Encoder:
    signature='test-encoder'
    def __init__(self):self.calls=[]
    def encode(self,texts):
        self.calls.append(list(texts))
        return np.tile(np.ones((1,384),dtype=np.float32)/np.sqrt(384),(len(texts),1))


def test_query_never_reencodes_evicted_text():
    encoder=Encoder();agent=ConsolidatingAgentMemory(encoder,None,512)
    agent.observe('source|with|pipes','2023/01/01',[dict(role='user',content='A secret lost to eviction.',has_answer=True)])
    encoder.calls.clear();answer=agent.query('What remains?','2023/01/02')
    assert answer['context']=='' and encoder.calls==[['What remains?']]


def test_source_ids_and_checkpoint_encoder_contract(tmp_path):
    encoder=Encoder();agent=ConsolidatingAgentMemory(encoder,None,4096)
    agent.observe('source|pipes','2023/01/01',[dict(role='user',content='Tea is preferred.',has_answer=True)])
    result=agent.query('Preferred drink?','2023/01/02');assert result['sources'][0]['source_id']=='source|pipes'
    assert 'source|pipes' not in result['context']
    p=tmp_path/'snapshot.json';agent.checkpoint(p);other=ConsolidatingAgentMemory(encoder,None,4096);other.restore(p)
    assert other.query('Preferred drink?','2023/01/02')['context']==result['context']
    wrong=Encoder();wrong.signature='other'
    with pytest.raises(ValueError):ConsolidatingAgentMemory(wrong,None,4096).restore(p)
