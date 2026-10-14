import sqlite3
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest
from src.api.limiter import SlidingWindowLimiter
from src.api.agnes import AgnesClient, APIError, retry_after


def test_eleventh_attempt_waits_and_restart_keeps_window(tmp_path, clock):
    path = tmp_path / "rate.db"
    limiter = SlidingWindowLimiter(path, clock=clock.time, sleep=clock.sleep, paced=False)
    times = [limiter.acquire() for _ in range(10)]
    assert times == [1000.0] * 10
    restarted = SlidingWindowLimiter(path, clock=clock.time, sleep=clock.sleep, paced=False)
    times.append(restarted.acquire())
    assert times[-1] >= 1060
    assert sum(clock.waits) >= 60
    assert all(sum(t - 60 < earlier <= t for earlier in times) <= 10 for t in set(times))


def test_pacing_is_shared_between_clients(tmp_path, clock):
    a = SlidingWindowLimiter(tmp_path / "rate.db", clock=clock.time, sleep=clock.sleep)
    b = SlidingWindowLimiter(tmp_path / "rate.db", clock=clock.time, sleep=clock.sleep)
    sent = [a.acquire(), b.acquire(), a.acquire()]
    assert min(sent[i+1] - sent[i] for i in range(2)) >= 6
    b.defer(120)
    assert a.acquire() >= sent[-1] + 120


def test_concurrent_reservations_never_exceed_limit(tmp_path):
    import time
    path = tmp_path / "threads.db"
    limiters = [SlidingWindowLimiter(path, window=0.2, paced=False) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        times = sorted(executor.map(lambda i: limiters[i % 4].acquire(), range(24)))
    assert all(sum(t - 0.2 < earlier <= t for earlier in times) <= 10 for t in times)


def success():
    return httpx.Response(200, json={"model": "agnes-3.0-flash", "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})


def test_429_retries_after_wait_and_success_is_cached(tiny_config, clock):
    calls = []
    def handler(request):
        calls.append(clock.time())
        return httpx.Response(429, headers={"Retry-After": "75"}) if len(calls) == 1 else success()
    limiter = SlidingWindowLimiter(tiny_config["api"]["rate_db"], clock=clock.time, sleep=clock.sleep)
    client = AgnesClient(tiny_config["api"], transport=httpx.MockTransport(handler), limiter=limiter, api_key="test-only", sleep=clock.sleep)
    try:
        first = client.complete([{"role": "user", "content": "test"}])
        second = client.complete([{"role": "user", "content": "test"}])
        assert len(calls) == 2 and calls[1] - calls[0] >= 75
        assert not first["cache_hit"] and second["cache_hit"]
        assert first["attempts"] == 2
        with sqlite3.connect(tiny_config["api"]["cache_db"]) as db:
            assert [row[0] for row in db.execute("SELECT status FROM requests_log ORDER BY id")] == ["429", "200"]
    finally:
        client.close()


def test_invalid_key_is_not_retried_or_logged(tiny_config, clock, caplog):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="test-private-secret")
    limiter = SlidingWindowLimiter(tiny_config["api"]["rate_db"], clock=clock.time, sleep=clock.sleep)
    client = AgnesClient(tiny_config["api"], transport=httpx.MockTransport(handler), limiter=limiter, api_key="test-private-secret")
    try:
        with pytest.raises(APIError) as error:
            client.complete([{"role": "user", "content": "test"}])
        assert len(calls) == 1
        assert "test-private-secret" not in str(error.value) + caplog.text
    finally:
        client.close()


def test_transient_retry_budget_and_http_date(tiny_config, clock):
    tiny_config["api"]["max_retries"] = 1
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(503)
    limiter = SlidingWindowLimiter(tiny_config["api"]["rate_db"], clock=clock.time, sleep=clock.sleep)
    client = AgnesClient(tiny_config["api"], transport=httpx.MockTransport(handler), limiter=limiter, api_key="test-only")
    try:
        with pytest.raises(APIError, match="503"):
            client.complete([{"role": "user", "content": "test"}])
        assert len(calls) == 2
        assert retry_after("Thu, 01 Jan 1970 00:20:00 GMT", now=1000) == 200
    finally:
        client.close()


def test_multiple_rate_limits_do_not_exhaust_transient_retry_budget(tiny_config, clock):
    tiny_config["api"]["max_retries"] = 0
    calls = []
    def handler(request):
        calls.append(clock.time())
        return httpx.Response(429) if len(calls) <= 3 else success()
    limiter = SlidingWindowLimiter(tiny_config["api"]["rate_db"], clock=clock.time, sleep=clock.sleep)
    client = AgnesClient(tiny_config["api"], transport=httpx.MockTransport(handler), limiter=limiter, api_key="test-only")
    try:
        result = client.complete([{"role": "user", "content": "test"}])
        assert result["attempts"] == 4
        assert calls[1] - calls[0] >= 60
        assert calls[2] - calls[1] >= 120
        assert calls[3] - calls[2] >= 240
    finally:
        client.close()
