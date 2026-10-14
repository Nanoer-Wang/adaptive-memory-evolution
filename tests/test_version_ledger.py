from dataclasses import replace
import inspect
import numpy as np
import pytest
import torch
from src.memory.evidence_dynamics import EvidenceDynamics, exact_flow
from src.memory.version_ledger import Evidence, VersionLedger
from src.data.fact_evidence import extract_evidence, query_key


def fact(value, day, source=None, predicate='residence', mode='current'):
    quote=f'I live in {value}.'
    return Evidence('user',predicate,'',value,float(day),source or f's{day}',0,0,len(quote),quote,mode)


def test_exact_flow_zero_time_partition_and_boundedness():
    torch.manual_seed(1)
    z=torch.rand(13,8,dtype=torch.float64); b=torch.rand_like(z); rate=torch.rand_like(z)
    assert torch.equal(exact_flow(z,b,rate,0),z)
    assert torch.allclose(exact_flow(exact_flow(z,b,rate,.13),b,rate,4.7),exact_flow(z,b,rate,4.83),atol=1e-12,rtol=1e-12)
    values=exact_flow(z,b,rate,1e8)
    assert torch.isfinite(values).all() and (values>=0).all() and (values<=1).all()
    with pytest.raises(ValueError): exact_flow(z,b,rate,-1)
    with pytest.raises(ValueError): exact_flow(z,b,rate,float('nan'))


def test_new_version_changes_old_state_without_rewriting_evidence():
    torch.manual_seed(4); model=EvidenceDynamics('liquid'); ledger=VersionLedger(controller=model,policy='learned')
    old=fact('Paris',1); new=fact('London',10)
    ledger.write(old); original_state=ledger.items[old.identity].state.clone(); score=ledger.score(ledger.items[old.identity]).clone()
    ledger.write(new)
    assert ledger.current(old.key)==new
    assert ledger.items[old.identity].status=='superseded'
    assert ledger.items[old.identity].evidence==old
    assert not torch.allclose(original_state,ledger.items[old.identity].state)
    assert not torch.allclose(score,ledger.score(ledger.items[old.identity]))
    assert ledger.known_as_of(old.key,5)==old


@pytest.mark.parametrize('mode',['past','planned','tentative'])
def test_noncurrent_claim_does_not_replace_current(mode):
    ledger=VersionLedger(); current=fact('Paris',1); ledger.write(current)
    ledger.write(fact('London',5,mode=mode))
    assert ledger.current(current.key)==current


def test_retraction_only_revokes_matching_value():
    ledger=VersionLedger(); paris=fact('Paris',1);ledger.write(paris)
    ledger.write(fact('London',2,mode='retracted'))
    assert ledger.current(paris.key)==paris
    ledger.write(fact('Paris',3,mode='retracted'))
    assert ledger.current(paris.key) is None
    assert ledger.known_as_of(paris.key,2)==paris
    assert ledger.known_as_of(paris.key,3) is None


def test_losing_latest_payload_never_resurrects_old_value():
    ledger=VersionLedger(); old=fact('Paris',1);new=fact('London',2)
    ledger.write(old);ledger.write(new)
    ledger.score=lambda item: torch.tensor(0. if item.evidence==new else 1.)
    ledger.budget_bytes=ledger.payload_bytes-1;ledger._evict()
    assert old.identity in ledger.items and new.identity not in ledger.items
    assert ledger.current(old.key) is None
    assert ledger.known_as_of(old.key,2) is None
    assert ledger.known_as_of(old.key,1.5)==old
    assert 'status=superseded' in ledger.export_context(old.key)


def test_clock_source_identity_and_evidence_grounding_guards():
    ledger=VersionLedger(); item=fact('Paris',2); ledger.write(item)
    assert ledger.write(item) is False
    with pytest.raises(ValueError):ledger.write(fact('Rome',1))
    with pytest.raises(ValueError):ledger.write(replace(item,value='Rome'))
    with pytest.raises(ValueError):replace(item,end=500)
    with pytest.raises(ValueError):ledger.known_as_of(item.key,3)


def test_unrelated_key_does_not_trigger_a_local_evidence_impulse():
    torch.manual_seed(7); model=EvidenceDynamics(); a=VersionLedger(controller=model,policy='learned');b=VersionLedger(controller=model,policy='learned')
    item=fact('Paris',1)
    a.write(item);b.write(item)
    a.write(fact('30',5,predicate='age'))
    assert torch.equal(a.items[item.identity].state,b.items[item.identity].state)
    assert torch.allclose(a.latent_at(a.items[item.identity],5),b.latent_at(b.items[item.identity],5))


def test_write_has_no_question_argument_and_controller_has_gradients():
    assert 'question' not in inspect.signature(VersionLedger.write).parameters
    for kind in ('static','gru','liquid','fixed_rate','no_time'):
        model=EvidenceDynamics(kind);ledger=VersionLedger(controller=model,policy='learned')
        ledger.write(fact('Paris',1));ledger.write(fact('London',8))
        loss=torch.stack(list(ledger.scores().values())).square().mean();loss.backward()
        assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()


def test_budget_counts_watermarks_and_repeated_new_keys_stay_bounded():
    ledger=VersionLedger(900)
    for i in range(50):
        ledger.write(fact('北京'+str(i),i,predicate='p'+str(i)))
        assert ledger.payload_bytes<=900
        assert len(ledger.keys)<=len(ledger.items)
    oversized=fact('x'*5000,51)
    ledger.write(oversized)
    assert ledger.payload_bytes<=900 and oversized.identity not in ledger.items


def test_extraction_ignores_annotation_fields_and_preserves_exact_spans():
    turns=[dict(role='user',content='I live in Paris. I now have 1,300 followers on Instagram.')]
    a=extract_evidence(turns,'s',1)
    b=extract_evidence([dict(turns[0],has_answer=True,answer='LEAK',question='LEAK')],'s',1)
    assert a==b and len(a)==2
    for e in a: assert turns[e.turn]['content'][e.start:e.end]==e.quote
    assert a[-1].value=='1300'


def test_attribution_plans_negation_and_unsupported_queries():
    assert not extract_evidence([dict(role='user',content='Please rephrase: I live in Paris.')],'s',1)
    assert not extract_evidence([dict(role='assistant',content='I live in Paris.')],'s',1)
    e=extract_evidence([dict(role='user',content='I plan to move to London. I no longer live in Paris.')],'s',1)
    assert [x.mode for x in e]==['planned','retracted']
    assert query_key('Where does John live?') is None
    assert query_key('Where does my wife live?') is None
    assert query_key('Where did I live last year?') is None
    assert query_key('Where do I live now?')==('user','residence','')
