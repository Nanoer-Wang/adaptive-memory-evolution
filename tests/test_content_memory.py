import inspect
import numpy as np
import torch
from src.memory.content_store import ContentStore, ContentWriter, bounded_text
from src.data.content_benchmark import locomo_examples, canonical_turn
from scripts.content_experiment import Reader, retained_indices
from src.data.longmemeval import History
from src.retrieval.retriever import pack_context
from src.evaluation.runner import parse_judge_response


def test_judge_requires_one_complete_boolean_verdict():
    assert parse_judge_response('Reasoning. ```json\n{"correct": true}\n```') == 1
    assert parse_judge_response('{"correct": "true"}') is None
    assert parse_judge_response('{"correct":true} {"correct":false}') is None
    assert parse_judge_response('Explanation without verdict or truncated {"correct":') is None


def test_context_headers_do_not_reveal_evidence_ids():
    history = History('q', 'What happened?', '2026/01/02', ('answer_secret_1','distractor_2'),
                      ('2026/01/01','2026/01/02'), ('a known fact','another fact'))
    context, trace = pack_context(history,[0,1])
    assert 'answer_secret_1' not in context and 'distractor_2' not in context
    assert context.count('[session=m_') == 2
    assert trace[0]['session_id'] == 'answer_secret_1'
    assert trace[0]['context_id'] in context


def test_fifo_storage_and_reader_exclude_evicted_text():
    store = ContentStore(2)
    for i, text in enumerate(('secret pineapple', 'mountain walking', 'ocean sailing')):
        store.insert(str(i), '2026/01/01', text, np.eye(3, 384, dtype=np.float32)[i], 0)
    assert [item.session_id for item in store.items] == ['1', '2']
    reader = Reader(store)
    assert 'secret' not in reader.vectorizer.vocabulary_
    assert len(reader.query('secret pineapple', np.ones(384))[0]) == 2
    assert 'question' not in inspect.signature(store.insert).parameters
    assert store.payload_bytes >= 2 * 384 * 4


def test_writer_prefix_causality_and_reset_control():
    torch.manual_seed(42)
    x = torch.randn(4, 384); dt = torch.tensor([0., 1., 2., 3.])
    for kind in ('liquid_content', 'gru_content', 'static_content'):
        model = ContentWriter(kind)
        assert torch.allclose(model(x[:2], dt[:2]), model(x, dt)[:2])
        model(x, dt).sum().backward()
        assert model.encoder[0].weight.grad.abs().sum() > 0
    reset = ContentWriter('static_content')
    changed = x.clone(); changed[:3] *= 5
    assert torch.allclose(reset(x, dt)[-1], reset(changed, dt)[-1])


def test_offline_retention_matches_online_tie_eviction():
    assert retained_indices([1, 1, 1, 1], 2) == [2, 3]
    text = 'A' * 3000 + 'B' * 3000 + 'C' * 3000
    clipped = bounded_text(text)
    assert len(clipped) <= 2400
    assert all(c in clipped for c in 'ABC')


def test_locomo_annotations_never_become_content():
    samples = [dict(sample_id='demo', conversation={
        'speaker_a': 'A', 'session_1_date_time': '1:56 pm on 8 May, 2023',
        'session_1': [dict(speaker='A', dia_id='D1:1', text='hello', blip_caption='a cat', has_answer=True)]},
        session_summary={'secret': 'DO NOT INCLUDE'}, qa=[
            dict(question='q', answer='gold secret', category=4, evidence=['D1: 1']),
            dict(question='missing', category=5, adversarial_answer='incorrect distractor', evidence=[]),
            dict(question='bad evidence', answer='x', category=1, evidence=['D9:9'])])]
    rows = locomo_examples(samples)
    assert canonical_turn('D01: 01') == 'D1:1'
    assert rows[0]['gold'] == ['demo:session_1']
    assert rows[0]['exclusion'] is None
    assert rows[0]['history'].texts == ('A: hello\nImage caption: a cat',)
    assert rows[1]['history'].question_id.endswith('_abs')
    assert 'distractor' not in rows[1]['answer']
    assert rows[2]['exclusion'] == 'unmapped_evidence'
