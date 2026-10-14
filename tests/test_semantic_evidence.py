import numpy as np
import torch
from src.memory.semantic_evidence import extract_spans,materialize,SemanticLedger,SEMANTIC_FEATURES
from src.memory.evidence_dynamics import EvidenceDynamics


def evidence(text,source='s',time=1):
    vector=np.ones(384,dtype=np.float32);vector/=np.linalg.norm(vector)
    return materialize(extract_spans([dict(role='user',content=text)],source,time)[0],vector)


def test_extraction_is_exhaustive_role_aware_and_annotation_free():
    turns=[dict(role='user',content='abcdefghijk',has_answer=True),dict(role='assistant',content='not a user fact')]
    clean=[{k:t[k] for k in ('role','content')} for t in turns]
    spans=extract_spans(turns,'s',1,4)
    assert spans==extract_spans(clean,'s',1,4)
    for i,t in enumerate(turns):
        assert ''.join(s['quote'] for s in spans if s['turn']==i)==t['content']
        assert all(t['content'][s['start']:s['end']]==s['quote'] for s in spans if s['turn']==i)


def test_semantic_updates_do_not_assert_current_truth_and_charge_vectors():
    model=EvidenceDynamics('liquid',SEMANTIC_FEATURES)
    a=evidence('I might move to Paris.');b=evidence('Consider Paris.',source='other',time=2)
    ledger=SemanticLedger(controller=model,policy='learned');ledger.write(a)
    state=ledger.items[a.identity].state.clone();ledger.write(b)
    assert not torch.equal(state,ledger.items[a.identity].state)
    assert ledger.current(a.key) is None
    assert ledger.items[a.identity].status=='past'
    assert a.payload_bytes>1664
    sum(ledger.scores().values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_source_ids_not_in_features_and_bounded_read():
    a=evidence('Quoted content.');b=evidence('Quoted content.','another')
    first=SemanticLedger();second=SemanticLedger()
    first.write(a);second.write(b)
    assert torch.equal(first.features(first.items[a.identity]),second.features(second.items[b.identity]))
    bounded=SemanticLedger(2200)
    for i in range(12):bounded.write(evidence('Fact '+str(i),str(i),i+1))
    assert bounded.payload_bytes<=2200
    assert len(bounded.retrieve(np.ones(384)))<=len(bounded.items)
