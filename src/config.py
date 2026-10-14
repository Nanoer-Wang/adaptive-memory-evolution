from __future__ import annotations
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    config = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    for section in ("data", "model", "training", "api", "retrieval", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    if not 1 <= config["api"]["rpm"] <= 10:
        raise ValueError("Agnes free-key rpm must be between 1 and 10")
    for key in ("top_k", "context_chars"):
        if config["retrieval"][key] <= 0:
            raise ValueError(f"retrieval.{key} must be positive")
    if config["retrieval"]["chunk_chars"] < 1:
        raise ValueError("retrieval.chunk_chars must be positive")
    if config["model"]["input_dim"] < 1 or config["model"]["memory_dim"] < 4 or config["model"]["memory_dim"] % 4:
        raise ValueError("memory_dim must be a positive multiple of four; input_dim must be positive")
    if config["model"]["time_scale_days"] <= 0 or config["training"]["temperature"] <= 0:
        raise ValueError("time scale and training temperature must be positive")
    if config['model'].get('timestamp_precision', 'day') not in ('day', 'minute'):
        raise ValueError('timestamp_precision must be day or minute')
    return config


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
