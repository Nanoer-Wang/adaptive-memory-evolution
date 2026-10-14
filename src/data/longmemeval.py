from __future__ import annotations
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from src.config import atomic_json, resolve
from src.data.download import file_sha256

LOG = logging.getLogger(__name__)
FEATURE_VERSION = 1


@dataclass(frozen=True)
class History:
    question_id: str
    question: str
    question_date: str
    session_ids: tuple[str, ...]
    dates: tuple[str, ...]
    texts: tuple[str, ...]


def session_text(turns: list[dict]) -> str:
    # Allowlist: answer labels/has_answer never enter embeddings, state or prompts.
    return "\n".join(f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns)


def date_days(text: str, precision: str = 'day') -> float:
    match = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", str(text))
    if not match:
        raise ValueError(f"Unsupported history date: {text}")
    date = datetime(*map(int, match.groups()))
    if precision == 'day':
        return date.toordinal()  # Explicit legacy mode for existing checkpoints.
    if precision != 'minute':
        raise ValueError('timestamp_precision must be day or minute')
    clock = re.search(r'(?:T|\s)(\d{2}):(\d{2})(?::(\d{2}))?', str(text)[match.end():])
    if clock:
        hour, minute, second = [int(value or 0) for value in clock.groups()]
        date = date.replace(hour=hour, minute=minute, second=second)
    return date.toordinal() + (date.hour * 3600 + date.minute * 60 + date.second) / 86400


def public_history(row: dict, timestamp_precision: str = 'day') -> History:
    ids, dates, sessions = row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]
    if not (len(ids) == len(dates) == len(sessions)) or not ids:
        raise ValueError("Invalid or empty history")
    # Upstream contains reused IDs with different content/dates. Keep every
    # record intact; metric code counts unique evidence session IDs.
    ordered = sorted(zip(ids, dates, sessions), key=lambda entry: date_days(entry[1], timestamp_precision))
    return History(str(row["question_id"]), row["question"], row["question_date"],
                   tuple(str(s[0]) for s in ordered), tuple(s[1] for s in ordered),
                   tuple(session_text(s[2]) for s in ordered))


def load_rows(path: str | Path) -> list[dict]:
    with resolve(path).open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must be a list with unique question IDs")
    return rows


def make_splits(rows: list[dict], seed: int, train_fraction: float, val_fraction: float) -> dict:
    """Keep shared evidence IDs/content and related _abs pairs in one partition.

    Reused distractor sessions can occur across splits; never use test QA labels
    or evidence flags at inference. This is a CUSTOM research split, not official.
    """
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # Protect evidence even if another question presents it as a distractor.
    evidence_ids, evidence_hashes = set(), set()
    for row in rows:
        gold = set(row.get("answer_session_ids", []))
        for sid, turns in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            if sid in gold:
                evidence_ids.add(str(sid))
                evidence_hashes.add(hashlib.sha256(session_text(turns).encode()).hexdigest())
    owners = {}
    for i, row in enumerate(rows):
        keys = ["question:" + str(row["question_id"]).removesuffix("_abs")]
        for sid, turns in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            content_hash = hashlib.sha256(session_text(turns).encode()).hexdigest()
            if str(sid) in evidence_ids:
                keys.append("sid:" + str(sid))
            if content_hash in evidence_hashes:
                keys.append("text:" + content_hash)
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(str(row["question_id"]))
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    splits = {"train": [], "val": [], "test": []}
    targets = {"train": len(rows) * train_fraction, "val": len(rows) * val_fraction,
               "test": len(rows) * (1 - train_fraction - val_fraction)}
    for group in sorted(groups, key=len, reverse=True):
        name = max(targets, key=lambda key: targets[key] - len(splits[key]))
        splits[name].extend(group)
    if not all(splits.values()):
        raise ValueError("Not enough independent evidence groups for three splits")
    return {"protocol": "custom LongMemEval-S evidence-grouped train/val/test; NOT official full-500 evaluation",
            "seed": seed, "group_count": len(groups), "splits": splits}


def encode(history: History, model_config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A lightweight reproducible lexical encoder, not a pretrained semantic model.
    vectorizer = HashingVectorizer(n_features=model_config["input_dim"], alternate_sign=False,
                                  ngram_range=(1, 2), norm="l2", dtype=np.float32)
    features = vectorizer.transform(history.texts).toarray()
    query = vectorizer.transform([history.question]).toarray()[0]
    # Absolute ordinals near 740,000 lose minute-scale differences in float32.
    days = np.array([date_days(d, model_config.get('timestamp_precision', 'day')) for d in history.dates], dtype=np.float64)
    if np.any(np.diff(days) < 0):
        raise ValueError('History must be sorted at the configured timestamp precision')
    gaps = np.r_[0, np.diff(days)] / model_config["time_scale_days"]
    deltas = np.clip(gaps, 0, model_config["max_delta"]).astype(np.float32)
    return features, query, deltas


def feature_path(root: Path, question_id: str) -> Path:
    return root / "features" / (hashlib.sha256(question_id.encode()).hexdigest() + ".npz")


def prepare(config: dict) -> dict:
    data_config, model_config = config["data"], config["model"]
    path, root = resolve(data_config["file"]), resolve(data_config["prepared"])
    rows = load_rows(path)
    source_hash = file_sha256(path)
    signature = {"sha256": source_hash, "features_version": 2 if model_config.get('timestamp_precision') == 'minute' else FEATURE_VERSION, "split_version": 2, "model": model_config,
                 "seed": data_config["seed"], "train_fraction": data_config["train_fraction"], "val_fraction": data_config["val_fraction"]}
    metadata = root / "manifest.json"
    building = root / "preparing.json"
    if building.exists() and json.loads(building.read_text(encoding="utf-8")) != signature:
        raise ValueError("Partial preparation uses different settings; choose a new data.prepared directory")
    if metadata.exists():
        existing = json.loads(metadata.read_text(encoding="utf-8"))
        if existing["signature"] != signature:
            raise ValueError("Prepared data configuration changed; use a different data.prepared directory")
    atomic_json(building, signature)
    split = make_splits(rows, data_config["seed"], data_config["train_fraction"], data_config["val_fraction"])
    (root / "features").mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        target = feature_path(root, str(row["question_id"]))
        if not target.exists():
            history = public_history(row, model_config.get('timestamp_precision', 'day'))
            features, query, deltas = encode(history, model_config)
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, features=features, query=query, deltas=deltas)
            temporary.replace(target)
        if (i + 1) % 50 == 0:
            LOG.info("Prepared %s/%s histories locally (no API calls)", i + 1, len(rows))
    result = {"signature": signature, **split}
    atomic_json(metadata, result)
    return result


def prepared(config: dict) -> tuple[list[dict], dict]:
    root = resolve(config["data"]["prepared"])
    meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if meta["signature"]["sha256"] != file_sha256(resolve(config["data"]["file"])):
        raise ValueError("Dataset differs from prepared manifest")
    if meta["signature"]["model"] != config["model"]:
        raise ValueError("Model feature configuration differs from prepared manifest")
    expected_version = 2 if config['model'].get('timestamp_precision') == 'minute' else FEATURE_VERSION
    if meta['signature']['features_version'] != expected_version:
        raise ValueError('Prepared feature version differs from configuration')
    for key in ('seed', 'train_fraction', 'val_fraction'):
        if meta['signature'][key] != config['data'][key]:
            raise ValueError('Prepared split configuration differs from manifest')
    return load_rows(config["data"]["file"]), meta


def features_for(config: dict, question_id: str):
    with np.load(feature_path(resolve(config["data"]["prepared"]), question_id), allow_pickle=False) as data:
        return data["features"], data["query"], data["deltas"]
