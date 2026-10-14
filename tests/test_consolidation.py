import numpy as np
import pytest
import torch
from src.memory.consolidation import ConsolidationController,ConsolidationMemory,PrefixFeatures
from src.memory.evidence_dynamics import exact_flow


VECTOR=np.ones(32,dtype=np.float32)


def test_flow_zero_and_partition_consistency():
    h=torch.tensor([.2,.8]);e=torch.tensor([.9,.1]);r=torch.tensor([.3,.7])
    assert torch.equal(exact_flow(h,e,r,0),h)
    assert torch.allclose(exact_flow(exact_flow(h,e,r,2),e,r,3),exact_flow(h,e,r,5),atol=1e-7)


def test_ablations_enforce_beta_and_weight():
    for kind,key,value in [('no_forgetting','beta',0.),('no_importance','w',1.)]:
        model=ConsolidationController(kind)
        p,_=model.step(torch.randn(40),7.,model.initial())
        assert float(p[key])==value


def test_versions_do_not_resurrect_after_rejected_update():
    model=ConsolidationController('linear')
    with torch.no_grad():
        model.policy.weight.zero_();model.policy.bias.zero_();model.policy.bias[0]=10
    memory=ConsolidationMemory(model,2048,admission='threshold')
    memory.observe('a','Mary moved to the office.',VECTOR,0,'Mary','office')
    with torch.no_grad():model.policy.bias[0]=-10
    memory.observe('b','Mary moved to the kitchen.',VECTOR,1,'Mary','kitchen')
    assert not memory.retrieve(2,key='Mary')
    assert any(r.value=='office' for r in memory.items.values())


def test_duplicate_consolidation_counts_provenance_and_budget():
    memory=ConsolidationMemory(None,1024,'decay')
    first=memory.observe('s0','user: likes tea',VECTOR,0)
    before=memory.payload_bytes
    second=memory.observe('s1','user: likes tea',VECTOR,1)
    assert second['action']=='merge'
    assert len(memory.items)==1 and memory.payload_bytes>before
    for i in range(2,60):
        memory.observe('s'+str(i),'user: likes tea',VECTOR,float(i))
        assert memory.payload_bytes<=1024


def test_checkpoint_continues_controller_and_strength(tmp_path):
    torch.manual_seed(7);m=ConsolidationController();a=ConsolidationMemory(m,2048)
    a.observe('s0','user: an observation',VECTOR,0)
    a.checkpoint(tmp_path/'memory.json');b=ConsolidationMemory(m,2048);b.restore(tmp_path/'memory.json')
    ra=a.observe('s1','user: another observation',VECTOR,10)
    rb=b.observe('s1','user: another observation',VECTOR,10)
    assert ra==rb
    assert all(torch.equal(x,y) for x,y in zip(a.state,b.state))
    with pytest.raises(ValueError):b.observe('future','bad',VECTOR,9)


def test_training_prefix_does_not_depend_on_admission():
    m=ConsolidationMemory(None,512);prefix=PrefixFeatures()
    for i in range(10):
        text=f'user: item{i}'
        expected,dt=prefix.observe(text,VECTOR,float(i))
        m.observe(str(i),text,VECTOR,float(i))
        assert m.prefix.count==prefix.count and np.array_equal(m.prefix.previous,prefix.previous)
        assert m.payload_bytes==512


def test_similarity_is_not_supersession():
    m=ConsolidationMemory(None,4096)
    m.observe('a','user: I own3 bikes.',VECTOR,0)
    m.observe('b','user: I own4 bikes.',VECTOR,1)
    assert len(m.retrieve(1,vector=VECTOR))==2


def test_spare_capacity_does_not_discard_low_confidence_evidence():
    model=ConsolidationController('linear')
    with torch.no_grad():model.policy.weight.zero_();model.policy.bias.fill_(-10.)
    m=ConsolidationMemory(model,2048)
    m.observe('a','user: important unanticipated fact',VECTOR,0)
    assert len(m.items)==1


def test_constant_beta_control_preserves_other_actions():
    torch.manual_seed(17);model=ConsolidationController();x=torch.randn(40);state=model.initial()
    original,s=model.step(x,7.,state);model.beta_override=1/30;changed,t=model.step(x,7.,state)
    assert torch.equal(original['alpha'],changed['alpha']) and torch.equal(original['w'],changed['w'])
    assert all(torch.equal(a,b) for a,b in zip(s,t))
    assert float(changed['beta'])==pytest.approx(1/30)
