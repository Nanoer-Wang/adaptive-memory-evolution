"""Download the official cleaned LongMemEval-S with resumable HTTP transfers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import httpx

LOG = logging.getLogger(__name__)
REPO = "xiaowu0162/longmemeval-cleaned"
FILENAME = "longmemeval_s_cleaned.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(root: str | Path = "data/longmemeval", revision: str = "main") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / FILENAME
    manifest = root / "source.json"
    if target.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if file_sha256(target) == saved["sha256"]:
            LOG.info("Dataset already verified: %s", target)
            return target
        raise ValueError("Dataset checksum changed; move the damaged file before downloading again.")
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        info = client.get(f"https://huggingface.co/api/datasets/{REPO}/revision/{revision}")
        info.raise_for_status()
        commit = info.json()["sha"]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{FILENAME}"
        part = target.with_suffix(".json.part")
        marker = root / "download_revision.txt"
        if part.exists() and (not marker.exists() or marker.read_text() != commit):
            raise ValueError("Partial download belongs to a different revision; move it before retrying.")
        marker.write_text(commit)
        for attempt in range(8):
            offset = part.stat().st_size if part.exists() else 0
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416 and offset:
                        break
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Invalid server resume range")
                    LOG.info("Downloading official LongMemEval-S (resume=%s bytes)", offset if append else 0)
                    with part.open("ab" if append else "wb") as stream:
                        for block in response.iter_bytes(1024 * 1024):
                            stream.write(block)
                break
            except (httpx.HTTPError, OSError) as exc:
                if attempt == 7:
                    raise RuntimeError("Dataset download failed; partial file kept for resume") from exc
                LOG.warning("Download interrupted (%s); retrying", type(exc).__name__)
                time.sleep(min(2 ** attempt, 30))
        with part.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if len(rows) != 500 or not all("haystack_sessions" in r and "question_id" in r for r in rows):
            raise ValueError("Downloaded data did not pass the 500-instance schema check")
        os.replace(part, target)
        metadata = {"repository": REPO, "revision": commit, "url": url,
                    "license": "MIT (upstream dataset card)", "filename": FILENAME,
                    "bytes": target.stat().st_size, "instances": len(rows), "sha256": file_sha256(target)}
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        LOG.info("Verified %s instances, %.1f MB, sha256=%s", len(rows), target.stat().st_size / 1e6, metadata["sha256"])
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    download()
