from __future__ import annotations
import logging
import sqlite3
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """Shared SQLite rolling window, pacing and server cooldown across processes.

    Every HTTP attempt reserves a slot, including retries. All free keys share
    this database. Waits happen outside transactions and remain interruptible.
    """
    def __init__(self, path: str | Path, rpm: int = 10, window: float = 60,
                 clock=time.time, sleep=time.sleep, paced: bool = True):
        if rpm < 1 or rpm > 10 or window <= 0:
            raise ValueError("Invalid rate limit")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rpm, self.window = rpm, window
        self.clock, self.sleep = clock, sleep
        self.spacing = window / rpm + 0.05 if paced else 0
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (sent REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS cooldown (id INTEGER PRIMARY KEY, until REAL NOT NULL)")
            db.execute("INSERT OR IGNORE INTO cooldown VALUES (1, 0)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def defer(self, seconds: float) -> None:
        with self._connect() as db:
            db.execute("UPDATE cooldown SET until = MAX(until, ?) WHERE id = 1", (self.clock() + max(seconds, 0),))

    def acquire(self) -> float:
        while True:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                now = self.clock()
                db.execute("DELETE FROM requests WHERE sent <= ?", (now - self.window,))
                times = [row[0] for row in db.execute("SELECT sent FROM requests ORDER BY sent")]
                ready = db.execute("SELECT until FROM cooldown WHERE id = 1").fetchone()[0]
                if len(times) >= self.rpm:
                    ready = max(ready, times[-self.rpm] + self.window + 0.05)
                if times:
                    ready = max(ready, times[-1] + self.spacing)
                delay = ready - now
                if delay <= 0:
                    db.execute("INSERT INTO requests VALUES (?)", (now,))
                    return now
            LOG.info("API queue: waiting %.1fs for the next request slot", delay)
            self.sleep(min(delay, 30))
