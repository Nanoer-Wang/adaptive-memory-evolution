import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.longmemeval import load_rows, prepare, make_splits, public_history, encode
from src.evaluation.metrics import retrieval_metrics
from src.evaluation.runner import evaluate, messages_for
from src.models.memory_model import MemoryRetriever
from src.memory.liquid_memory import LiquidMemory
from src.retrieval.retriever import pack_context
from src.training.train import train


def test_gold_labels_never_reach_features_or_prompt(tiny_config):
    row = load_rows(tiny_config["data"]["file"])[0]
    public = public_history(row)
    before = encode(public, tiny_config["model"])
    row["answer"] = "SECRET_ANSWER_SENTINEL"
    row["answer_session_ids"] = ["SECRET_EVIDENCE_SENTINEL"]
    row["haystack_sessions"][0][0]["has_answer"] = "SECRET_LABEL_SENTINEL"
    after = public_history(row)
    assert public == after
    assert all(np.array_equal(x, y) for x, y in zip(before, encode(after, tiny_config["model"])))
    context, trace = pack_context(after, [0, 1], budget=120, chunk_chars=60)
    assert len(context) <= 120
    assert "SECRET_" not in json.dumps(messages_for(after, context))


def test_evidence_overlap_stays_in_one_split(tiny_config):
    rows = load_rows(tiny_config["data"]["file"])
    rows[1]["haystack_sessions"][1] = rows[0]["haystack_sessions"][0]
    # Evidence from row zero appears as a distractor with a different ID in row one.
    split = make_splits(rows, 42, .6, .2)["splits"]
    assert next(k for k, v in split.items() if "q00" in v) == next(k for k, v in split.items() if "q01" in v)
    assert len(set(sum(split.values(), []))) == len(rows)


def test_duplicate_upstream_session_ids_are_preserved(tiny_config):
    row = load_rows(tiny_config["data"]["file"])[0]
    row["haystack_session_ids"][1] = row["haystack_session_ids"][0]
    history = public_history(row)
    assert len(history.texts) == 2
    assert retrieval_metrics(history.session_ids, [history.session_ids[0]])["session_recall_at_k"] == 1
    assert retrieval_metrics([], [], True)["session_recall_at_k"] is None


@pytest.mark.parametrize("kind", ["lcm", "gru", "transformer", "lcm_v2", "ranker", "gru_v2"])
def test_train_checkpoint_evaluate_resume_and_leak_guard(tiny_config, tmp_path, kind):
    prepare(tiny_config)
    result = train(tiny_config, kind, epochs=1, output=tmp_path / kind)
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.exists()
    run = evaluate(tiny_config, kind, "test", checkpoint=checkpoint, retrieval_only=True)
    again = evaluate(tiny_config, kind, "test", checkpoint=checkpoint, retrieval_only=True)
    assert run["metrics"] == again["metrics"]
    with pytest.raises(ValueError, match="overlaps training"):
        evaluate(tiny_config, kind, "all", checkpoint=checkpoint, retrieval_only=True)
    resumed = train(tiny_config, kind, epochs=2, output=tmp_path / kind, resume=True)
    assert resumed["epochs"] == 2


def test_time_delta_changes_liquid_state():
    torch.manual_seed(9)
    cell = LiquidMemory(8)
    state, experience = torch.randn(8), torch.randn(8)
    assert not torch.allclose(cell(state, experience, delta=0), cell(state, experience, delta=5))
    output = cell(state, experience, delta=2)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())


def test_prepared_config_mismatch_is_rejected(tiny_config):
    prepare(tiny_config)
    tiny_config["model"]["input_dim"] += 1
    with pytest.raises(ValueError, match="different settings|configuration changed"):
        prepare(tiny_config)


def test_partial_evaluation_resumes_without_repeating_completed(tiny_config, monkeypatch, tmp_path):
    prepare(tiny_config)
    import src.evaluation.runner as runner
    calls = []
    class FakeClient:
        model = "agnes-test"
        base_url = "https://example.invalid/v1"
        fail = True
        def __init__(self, config):
            pass
        def complete(self, messages, **kwargs):
            calls.append(messages[-1]["content"])
            if len(calls) == 2 and self.fail:
                raise RuntimeError("simulated interruption")
            return {"content": "hobby0", "usage": {"total_tokens": 10}, "cache_hit": False, "finish_reason": "stop", "attempts": 1}
        def close(self):
            pass
    monkeypatch.setattr(runner, "AgnesClient", FakeClient)
    output = tmp_path / "resume-run"
    with pytest.raises(RuntimeError, match="interruption"):
        evaluate(tiny_config, "tfidf", "test", output=output)
    FakeClient.fail = False
    result = evaluate(tiny_config, "tfidf", "test", output=output)
    assert calls.count(calls[0]) == 1
    assert len(calls) == result["metrics"]["completed"] + 1
