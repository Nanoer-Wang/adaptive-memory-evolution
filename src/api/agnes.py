from __future__ import annotations
import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
import httpx
from dotenv import load_dotenv
from src.api.limiter import SlidingWindowLimiter
from src.config import ROOT, resolve

LOG = logging.getLogger(__name__)


class APIError(RuntimeError):
    pass


def retry_after(value: str | None, now: float | None = None) -> float:
    if not value:
        return 60
    try:
        return max(1, float(value))
    except ValueError:
        try:
            return max(1, parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, OverflowError):
            return 60


class AgnesClient:
    def __init__(self, config: dict, *, transport=None, limiter=None, api_key=None, sleep=time.sleep):
        load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
        self.key = api_key or os.getenv("AGNES_AI_API") or os.getenv("AGNES_API_KEY")
        if not self.key:
            raise APIError("Missing AGNES_AI_API in the project .env")
        self.base_url = os.getenv("AGNES_BASE_URL", config["base_url"]).rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("API base_url must use HTTPS without embedded credentials")
        self.model = os.getenv("AGNES_MODEL", config["model"])
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0)
        self.max_retries = config.get("max_retries", 5)
        self.sleep = sleep
        self.cache_path = resolve(config["cache_db"])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests_log (id INTEGER PRIMARY KEY, request_hash TEXT, model TEXT, base_url TEXT, started REAL, status TEXT, elapsed REAL, usage_json TEXT)")
        self.limiter = limiter or SlidingWindowLimiter(resolve(config["rate_db"]), config["rpm"])
        self.http = httpx.Client(timeout=httpx.Timeout(config.get("timeout_seconds", 120), connect=20),
                                 transport=transport, follow_redirects=False)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def close(self):
        self.http.close()

    def _record_attempt(self, key, started, status, elapsed, usage=None):
        # Record metadata only; never headers, keys, prompts, or raw error bodies.
        with sqlite3.connect(self.cache_path) as db:
            db.execute("INSERT INTO requests_log (request_hash,model,base_url,started,status,elapsed,usage_json) VALUES (?,?,?,?,?,?,?)",
                       (key, self.model, self.base_url, started, str(status), elapsed, json.dumps(usage or {})))

    def complete(self, messages: list[dict], *, namespace: str = "default", max_tokens=None) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        key = hashlib.sha256(json.dumps([self.base_url, namespace, body], sort_keys=True).encode()).hexdigest()
        with sqlite3.connect(self.cache_path) as db:
            cached = db.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        if cached:
            return {**json.loads(cached[0]), "cache_hit": True, "request_hash": key}
        failures, rate_events, attempts = 0, 0, 0
        queue_wait, retry_wait = 0.0, 0.0
        total_start = time.monotonic()
        while True:
            before_queue = time.monotonic()
            self.limiter.acquire()
            queue_wait += time.monotonic() - before_queue
            attempts += 1
            start = time.monotonic()
            sent_at = time.time()
            try:
                response = self.http.post(self.base_url + "/chat/completions", json=body,
                                          headers={"Authorization": "Bearer " + self.key})
            except httpx.TransportError:
                self._record_attempt(key, sent_at, "transport_error", time.monotonic() - start)
                failures += 1
                if failures > self.max_retries:
                    raise APIError("API network retries exhausted; rerun to resume") from None
                delay = min(2 ** failures, 60) + random.random()
                LOG.warning("Temporary network failure; retry in %.1fs", delay)
                self.sleep(delay)
                retry_wait += delay
                continue
            status = response.status_code
            try:
                attempt_usage = response.json().get("usage", {}) if status == 200 else {}
            except (ValueError, AttributeError):
                attempt_usage = {}
            self._record_attempt(key, sent_at, status, time.monotonic() - start, attempt_usage)
            if status == 429:
                rate_events += 1
                delay = max(retry_after(response.headers.get("retry-after")), min(60 * 2 ** min(rate_events - 1, 4), 900))
                self.limiter.defer(delay)
                LOG.warning("Server rate limit; queued for %.1fs (Ctrl+C to pause)", delay)
                continue
            if status in (408, 409, 425) or status >= 500:
                failures += 1
                if failures > self.max_retries:
                    raise APIError(f"API HTTP {status}: temporary retries exhausted; rerun to resume")
                delay = max(min(2 ** failures, 60), retry_after(response.headers["retry-after"]) if "retry-after" in response.headers else 0)
                self.limiter.defer(delay)
                continue
            if status != 200:
                reasons = {401: "authentication rejected; check China/international endpoint and key pairing", 403: "access denied", 402: "account quota or billing restriction"}
                raise APIError(f"API HTTP {status}: {reasons.get(status, 'request rejected; check model and configuration')}")
            try:
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty model answer")
                result = {"content": content, "usage": data.get("usage", {}), "request_hash": key,
                          "model": data.get("model", self.model), "finish_reason": choice.get("finish_reason"),
                          "latency_seconds": time.monotonic() - start, "attempts": attempts,
                          "queue_wait_seconds": queue_wait, "retry_backoff_seconds": retry_wait,
                          "total_wall_seconds": time.monotonic() - total_start}
            except (ValueError, KeyError, IndexError, TypeError):
                raise APIError("API returned an invalid/empty completion; response not cached") from None
            with sqlite3.connect(self.cache_path) as db:
                db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(result)))
            return {**result, "cache_hit": False}
