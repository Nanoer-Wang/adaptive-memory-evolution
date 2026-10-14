from src.memory.turn_compression import compress_turns, spread_order


def test_labels_and_unrelated_fields_do_not_change_content():
    turns = [{'role': 'user', 'content': 'I moved to Paris. I work remotely.'}]
    annotated = [dict(turns[0], has_answer=True, answer='LEAK', question='LEAK')]
    for mode in ('whole_spread', 'sentence_spread', 'sentence_user_first'):
        assert compress_turns(turns, 60, mode) == compress_turns(annotated, 60, mode)
        assert 'LEAK' not in compress_turns(annotated, 60, mode)


def test_utf8_budget_and_complete_utterances():
    turns = [{'role': 'user', 'content': '我住北京。'},
             {'role': 'assistant', 'content': 'An oversized response. ' * 100},
             {'role': 'user', 'content': '现在搬到了上海。'}]
    for budget in (0, 1, 20, 40, 80):
        packed = compress_turns(turns, budget)
        assert len(packed.encode('utf-8')) <= budget
        assert all(line in ('user: 我住北京。', 'user: 现在搬到了上海。')
                   for line in packed.splitlines())


def test_spread_order_covers_every_unit_once():
    for size in range(80):
        assert sorted(spread_order(size)) == list(range(size))


def test_unanswerable_stale_annotations_are_excluded():
    from scripts.compression_repair import evaluate
    row = {'question_id': 'q_abs', 'answer_session_ids': ['stale'],
           'haystack_sessions': [[{'role': 'user', 'content': 'Old fact.', 'has_answer': True}]]}
    assert evaluate([row], ('legacy',)) == []
