import numpy as np
import torch
from src.data.longmemeval import load_rows, public_history
from src.retrieval.lexical_features import cached_features, build_features
from src.models.anchored_memory import AnchoredMemory


def test_anchor_survives_initialization_and_memory_learns():
    torch.manual_seed(42)
    model = AnchoredMemory()
    x = torch.rand(6, 5)
    delta = torch.arange(6).float()
    assert torch.equal(model(x, deltas=delta)[0], x[:, 0])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss = -torch.log_softmax(model(x, deltas=delta)[0], dim=0)[0]
        loss.backward()
        optimizer.step()
    assert model.cell.rate.weight.grad.abs().sum() > 0
    assert not torch.allclose(model(x, deltas=delta)[0], model(x, deltas=delta * 0)[0])


def test_ranking_cache_invalidation_and_no_gold(tiny_config):
    row = load_rows(tiny_config['data']['file'])[0]
    before, hit = cached_features(tiny_config, public_history(row), with_status=True)
    assert not hit
    row['answer'] = 'secret'
    row['answer_session_ids'] = ['secret']
    row['haystack_sessions'][0][0]['has_answer'] = False
    after, hit = cached_features(tiny_config, public_history(row), with_status=True)
    assert hit and np.array_equal(before, after)
    row['question'] = 'Which city was visited?'
    changed, hit = cached_features(tiny_config, public_history(row), with_status=True)
    assert not hit and not np.array_equal(before, changed)
    assert changed.shape == (2, 5) and np.isfinite(changed).all()
    row['haystack_sessions'] = [[{'role':'user', 'content':''}]] * 2
    assert np.isfinite(build_features(public_history(row))).all()


def test_reset_control_removes_history_without_changing_parameters():
    torch.manual_seed(42)
    full = AnchoredMemory()
    reset = AnchoredMemory(reset_history=True)
    torch.nn.init.normal_(full.readout.weight)
    reset.load_state_dict(full.state_dict())
    x = torch.rand(6, 5)
    changed = x.clone()
    changed[:-1] += 4
    delta = torch.ones(6)
    assert torch.equal(reset(x, deltas=delta)[0][-1], reset(changed, deltas=delta)[0][-1])
    assert not torch.allclose(full(x, deltas=delta)[0][-1], full(changed, deltas=delta)[0][-1])
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in reset.parameters())


def test_fixed_time_control_and_gru_parameter_budget():
    model = AnchoredMemory(fixed_time=True)
    torch.nn.init.normal_(model.readout.weight)
    x = torch.rand(6, 5)
    assert torch.equal(model(x, deltas=torch.zeros(6))[0], model(x, deltas=torch.ones(6)*10)[0])
    gru = AnchoredMemory(kind='gru_v2', memory_dim=13)
    sizes = [sum(p.numel() for p in m.parameters()) for m in (model, gru)]
    assert abs(sizes[0]-sizes[1])/sizes[0] < .02
