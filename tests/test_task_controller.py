import numpy as np
import torch
import pytest
from src.memory.task_controller import compact_context,parse_task_answer,TaskController
from src.data.longmemeval import History

def test_answer_contract_rejects_ambiguous_objects_and_non_boolean():
    assert parse_task_answer('{"answer":"four","abstain":false}')==dict(answer='four',abstain=False)
    assert parse_task_answer('```json\n{"answer":"unknown","abstain":true}\n```')['abstain']
    assert parse_task_answer('{"answer":"four","abstain":"false"}') is None
    assert parse_task_answer('{"answer":"four","abstain":false}{"answer":"five","abstain":false}') is None

def test_compact_evidence_preserves_attribution_and_whole_units():
    h=History('q','Who is the project owner?','2026/10/12',('s1','s2'),('2026/10/01','2026/10/10'),('user: The project owner is Alice.\nassistant: Acknowledged.\nuser: Coffee is nice.','user: The project owner is now Bob.\nassistant: This updates Alice.'))
    text,trace=compact_context(h,[1,0],250)
    assert len(text)<=250 and trace
    assert 'date=' in text and 'session=' in text
    original={line for t in h.texts for line in t.splitlines()}
    assert all(line in original for line in text.splitlines() if line and not line.startswith('[session='))

@pytest.mark.parametrize('kind',['linear','mlp','gru','liquid'])
def test_controller_has_gradients_without_task_answers(kind):
    torch.manual_seed(1);m=TaskController(kind);x=torch.randn(7,14);dt=torch.linspace(0,1,7)
    pred=m(x,dt);assert pred.ndim==0 and torch.isfinite(pred)
    (pred-.5).square().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
