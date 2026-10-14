import json
from copy import deepcopy
import pytest
from src.config import load_config


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.waits = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    monkeypatch.delenv("AGNES_BASE_URL", raising=False)
    monkeypatch.delenv("AGNES_MODEL", raising=False)
    config = deepcopy(load_config())
    config["model"].update(input_dim=32, memory_dim=16)
    config["training"].update(epochs=1, threads=1)
    config["data"]["file"] = str(tmp_path / "data.json")
    config["data"]["prepared"] = str(tmp_path / "prepared")
    config["api"]["rate_db"] = str(tmp_path / "rate.sqlite3")
    config["api"]["cache_db"] = str(tmp_path / "cache.sqlite3")
    config["evaluation"]["output_root"] = str(tmp_path / "runs")
    rows = []
    for i in range(12):
        rows.append({"question_id": f"q{i:02}", "question": f"What is person {i}'s hobby?",
                     "question_date": "2023/06/01 (Thu) 12:00", "answer": f"hobby{i}",
                     "question_type": "single-session-user", "answer_session_ids": [f"s{i}a"],
                     "haystack_session_ids": [f"s{i}a", f"s{i}b"],
                     "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 11:00"],
                     "haystack_sessions": [[{"role": "user", "content": f"Person {i} likes hobby{i}", "has_answer": True}],
                                           [{"role": "user", "content": f"Person {i} visited city{i}"}]]})
    with open(config["data"]["file"], "w", encoding="utf-8") as stream:
        json.dump(rows, stream)
    return config
