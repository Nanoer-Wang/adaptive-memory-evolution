
def forgetting_rate(before, after):
    return before-after


def forward_transfer(baseline, model):
    return model-baseline


def normalize(text):
    import re
    import string
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction, reference):
    """Lexical diagnostics, explicitly NOT the official LLM-judged accuracy."""
    from collections import Counter
    p, r = normalize(prediction), normalize(reference)
    common = sum((Counter(p.split()) & Counter(r.split())).values())
    f1 = 2 * common / (len(p.split()) + len(r.split())) if p or r else 1.0
    return {"exact_match": float(p == r), "token_f1": f1}


def retrieval_metrics(retrieved, evidence, abstention=False):
    if abstention or not evidence:
        return {"session_recall_at_k": None, "all_evidence_at_k": None}
    selected, gold = set(retrieved), set(evidence)
    return {"session_recall_at_k": len(selected & gold) / len(gold),
            "all_evidence_at_k": float(gold <= selected)}


def summarize(rows):
    def average(key, values):
        scores = [r[key] for r in values if r.get(key) is not None]
        return sum(scores) / len(scores) if scores else None
    keys = ("session_recall_at_k", "all_evidence_at_k", "exact_match", "token_f1", "custom_judge_correct")
    result = {"completed": len(rows), **{key: average(key, rows) for key in keys}}
    result["by_type"] = {kind: {"n": len(items), **{key: average(key, items) for key in keys}}
                         for kind in sorted({r["question_type"] for r in rows})
                         if (items := [r for r in rows if r["question_type"] == kind])}
    result["answer_calls_from_cache"] = sum(bool(r.get("cache_hit")) for r in rows)
    result["truncated_answers"] = sum(r.get("finish_reason") == "length" for r in rows)
    result["retrieval_scored_instances"] = sum(r.get("session_recall_at_k") is not None for r in rows)
    result["judge_scored_instances"] = sum(r.get("custom_judge_correct") is not None for r in rows)
    result["judge_parse_failures"] = sum(bool(r.get("judge_parse_failed")) for r in rows)
    result["api_tokens_recorded"] = {key: sum((r.get("usage") or {}).get(key, 0) or 0 for r in rows)
                                     for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    result["uncached_answer_tokens"] = sum((r.get("usage") or {}).get("total_tokens", 0) or 0 for r in rows if not r.get("cache_hit"))
    result["uncached_judge_tokens"] = sum((r.get("judge_usage") or {}).get("total_tokens", 0) or 0 for r in rows if not r.get("judge_cache_hit"))
    import numpy as np
    result["performance"] = {}
    for key in ("retrieval_seconds", "context_seconds", "generation_seconds", "judge_seconds", "queue_wait_seconds", "judge_queue_wait_seconds", "elapsed_seconds", "process_rss_mb", "context_chars"):
        values = [r[key] for r in rows if r.get(key) is not None]
        result["performance"][key] = {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
                                      "p95": float(np.percentile(values, 95)), "max": float(max(values)), "sum": float(sum(values))} if values else None
    return result
