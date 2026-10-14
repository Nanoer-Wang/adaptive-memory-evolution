import numpy as np
from src.memory.budget_store import BudgetStore, byte_retained_indices
from src.retrieval.preserved_context import pack_preserved_context
from src.data.longmemeval import History


def test_payload_budget_and_no_resurrection():
    store = BudgetStore(2000)
    costs = []
    for i in range(5):
        store.insert(str(i), '2026/01/01', 'user: 北京', np.zeros(384), 0)
        assert store.payload_bytes <= 2000
        costs.append(store.items[-1].nbytes)
    assert [i.sequence for i in store.items] == byte_retained_indices(range(5), costs, 2000)
    assert [i.sequence for i in store.items] == [4]
    store.insert('large', '2026/01/02', 'x' * 5000, np.zeros(384), 0)
    assert not store.items  # Budget eviction does not resurrect old payloads.


def test_context_preserves_word_crossing_old_window_boundary():
    text = 'a' * 1595 + ' important_fact_here ' + 'z' * 700
    history = History('q', 'fact', '2026/01/02', ('answer_leak',), ('2026/01/01',), (text,))
    context, trace = pack_preserved_context(history, [0])
    assert text in context and 'answer_leak' not in context
    assert not trace[0]['truncated']


def test_versions_remain_distinct_until_budget_eviction():
    store = BudgetStore(10000)
    for i, city in enumerate(('Paris', 'London')):
        store.insert(str(i), f'2026/01/0{i + 1}', f'user: I live in {city}.', np.zeros(384), 1)
    assert len(store.items) == 2
    assert 'Paris' in store.items[0].text and 'London' in store.items[1].text
