
"""Supervised evidence retrieval plus next-session proxy and stability losses.

No remote LLM gradients or fabricated task-success labels. Uses only the custom
training partition; validation selects the best checkpoint. Test is never used.
"""
from __future__ import annotations
import logging
import csv
import time
import random
from pathlib import Path
import numpy as np
import torch
from src.config import atomic_json, resolve
from src.data.longmemeval import prepared, features_for, public_history
from src.models.memory_model import MemoryRetriever
from src.models.anchored_memory import AnchoredMemory, FEATURE_VERSION
from src.retrieval.lexical_features import cached_features
from src.evaluation.metrics import retrieval_metrics
from src.retrieval.retriever import rank
from src.provenance import runtime_info, snapshot_source, rss_mb

LOG = logging.getLogger(__name__)


def train_step(loss, optimizer, parameters=None, grad_clip=1.0):
    optimizer.zero_grad()
    loss.backward()
    if parameters is not None:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    optimizer.step()


def tensors(config, row):
    return tuple(torch.from_numpy(array) for array in features_for(config, str(row["question_id"])))


def positive_indices(row, config=None):
    history = public_history(row, config['model'].get('timestamp_precision', 'day') if config else 'day')
    evidence = set(map(str, row.get("answer_session_ids", [])))
    return [i for i, sid in enumerate(history.session_ids) if sid in evidence]


def train(config, kind="lcm", epochs=None, output=None, resume=False, consolidation=True, fixed_time=False, reset_history=False):
    torch.set_num_threads(config["training"]["threads"])
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows, manifest = prepared(config)
    by_id = {str(row["question_id"]): row for row in rows}
    train_ids = manifest["splits"]["train"]
    val_ids = manifest["splits"]["val"]
    model_settings = {**config["model"], "kind": kind, "consolidation": consolidation, "fixed_time": fixed_time}
    anchored = kind in ('lcm_v2', 'ranker', 'gru_v2')
    if anchored:
        model_settings.update(memory_dim=13 if kind == 'gru_v2' else 16, feature_version=FEATURE_VERSION)
        if reset_history:
            model_settings['reset_history'] = True
    elif reset_history:
        raise ValueError('reset_history is supported only for anchored models')
    model = (AnchoredMemory if anchored else MemoryRetriever)(**model_settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    output = resolve(output or f"artifacts/checkpoints/{kind}")
    output.mkdir(parents=True, exist_ok=True)
    start_epoch, best, history = 0, -1.0, []
    if (output / "last.pt").exists() and not resume:
        raise ValueError("Checkpoint directory already exists; use --resume or a different --output")
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["model_settings"] != model_settings or checkpoint["data_signature"] != manifest["signature"] or checkpoint["training_config"] != config["training"]:
            raise ValueError("Resume configuration or dataset differs from checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best, history = checkpoint["epoch"], checkpoint["best_val_recall"], checkpoint["history"]
    atomic_json(output / "environment.json", {**runtime_info(), "source_archive": snapshot_source()[1],
                "config": config, "parameter_count": sum(p.numel() for p in model.parameters()), "model_settings": model_settings})
    cached = {}
    for qid in train_ids + val_ids:
        row = by_id[qid]
        if not qid.endswith("_abs") and positive_indices(row, config):
            features, query, deltas = tensors(config, row)
            if anchored:
                features = torch.from_numpy(cached_features(config, public_history(row, config['model'].get('timestamp_precision', 'day'))))
            cached[qid] = (features, query, deltas, positive_indices(row, config))
    usable_train = [qid for qid in train_ids if qid in cached]
    usable_val = [qid for qid in val_ids if qid in cached]
    if not usable_train or not usable_val:
        raise ValueError("No answerable examples in train/validation")
    total_epochs = epochs if epochs is not None else config["training"]["epochs"]
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.monotonic()
        model.train()
        order = usable_train.copy()
        random.Random(seed + epoch).shuffle(order)
        losses = []
        components = {"retrieval_loss": [], "future_proxy_loss": [], "stability_loss": []}
        for qid in order:
            features, query, deltas, positives = cached[qid]
            scores, future, stability = model(features, query, deltas)
            log_prob = torch.log_softmax(scores / config["training"]["temperature"], dim=0)
            retrieval_loss = -log_prob[positives].mean()
            loss = retrieval_loss + config["training"]["future_weight"] * future + config["training"]["stability_weight"] * stability
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            train_step(loss, optimizer, model.parameters(), config["training"]["grad_clip"])
            losses.append(loss.item())
            for name, value in zip(components, (retrieval_loss, future, stability)):
                components[name].append(value.item())
        model.eval()
        recalls = []
        with torch.no_grad():
            for qid in usable_val:
                features, query, deltas, positives = cached[qid]
                scores, _, _ = model(features, query, deltas)
                top = rank(scores.numpy(), config["retrieval"]["top_k"])
                row = by_id[qid]
                session_ids = public_history(row, config['model'].get('timestamp_precision', 'day')).session_ids
                recalls.append(retrieval_metrics([session_ids[i] for i in top],
                               list(map(str, row['answer_session_ids'])))['session_recall_at_k'])
        val_recall = float(np.mean(recalls))
        is_best = val_recall > best
        best = max(best, val_recall)
        stats = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_session_recall_at_k": val_recall,
                 "epoch_seconds": time.monotonic() - epoch_start, "process_rss_mb": rss_mb(), "lr": optimizer.param_groups[0]["lr"],
                 **{name: float(np.mean(values)) for name, values in components.items()}}
        history.append(stats)
        LOG.info("%s epoch %d/%d loss=%.4f val recall=%.4f", kind, epoch + 1, total_epochs, stats["loss"], val_recall)
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch + 1, "best_val_recall": best, "history": history,
                      "model_settings": model_settings, "data_signature": manifest["signature"],
                      "training_config": config["training"], "train_ids": train_ids, "val_ids": val_ids,
                      "protocol": manifest["protocol"]}
        for name in ["last.pt"] + (["best.pt"] if is_best else []):
            temporary = output / (name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output / name)
        atomic_json(output / "history.json", history)
        with (output / "history.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerows(history)
    return {"checkpoint": str(output / "best.pt"), "epochs": len(history), "best_val_recall_at_k": best,
            "train_answerable": len(usable_train), "val_answerable": len(usable_val)}


if __name__ == "__main__":
    from src.cli import main
    import sys
    raise SystemExit(main(["train", *sys.argv[1:]]))
