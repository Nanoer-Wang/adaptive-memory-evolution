"""Export sanitized API attempt metadata for usage and rate-limit analysis."""
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "artifacts/api/responses.sqlite3"
if not path.exists():
    raise SystemExit("No API usage database yet")
with sqlite3.connect(path) as db:
    rows = db.execute("SELECT id,request_hash,model,base_url,started,status,elapsed,usage_json FROM requests_log ORDER BY id").fetchall()
destination = root / "runs/api_usage.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["id", "request_hash", "model", "base_url", "started_utc", "status", "http_seconds", "prompt_tokens", "completion_tokens", "total_tokens"])
    for identifier, request_hash, model, base, timestamp, status, elapsed, usage in rows:
        tokens = json.loads(usage)
        writer.writerow([identifier, request_hash, model, base, datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), status, elapsed,
                         tokens.get("prompt_tokens"), tokens.get("completion_tokens"), tokens.get("total_tokens")])
print(f"Exported {len(rows)} HTTP attempts to {destination}")
