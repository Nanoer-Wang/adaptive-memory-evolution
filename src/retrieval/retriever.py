
import torch


def opaque_session_id(session_id):
    import hashlib
    return 'm_' + hashlib.sha256(str(session_id).encode()).hexdigest()[:16]


class StateConditionedRetriever:

    def retrieve(self, query, memory, k=5):
        if k < 1:
            raise ValueError("k must be positive")
        if len(memory.storage)==0:
            return []

        scores=[]

        for item in memory.storage:
            score=torch.cosine_similarity(
                query,
                item["embedding"],
                dim=-1
            )
            scores.append(score.item())

        idx=sorted(
            range(len(scores)),
            key=lambda i:scores[i],
            reverse=True
        )[:k]

        return [memory.storage[i] for i in idx]


def lexical_scores(texts, query):
    """Per-history TF-IDF; fit history only, never labels or other users."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        return (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    except ValueError:
        return np.zeros(len(texts), dtype=np.float32)


def rank(scores, k):
    import numpy as np
    if k < 1:
        raise ValueError("k must be positive")
    return np.argsort(-np.asarray(scores), kind="stable")[:k].tolist()


def pack_context(history, selected, budget=16000, chunk_chars=1600):
    """Fair per-session budgets; select text windows by query, never evidence labels.

    Session recall and supplied-window coverage are different. This reports the
    exact excerpts in run traces so clipping is visible to reviewers.
    """
    if not selected:
        return "", []
    chunks, trace = [], []
    per_session = budget // len(selected)
    for index in selected:
        text = history.texts[index]
        # Public benchmark IDs can reveal labels (e.g. LongMemEval answer_*).
        # Retain original IDs only in the local audit trace, never in prompts.
        context_id = opaque_session_id(history.session_ids[index])
        header = f"[session={context_id} date={history.dates[index]}]\n"
        allowance = max(0, per_session - len(header) - 2)
        windows = [text[start:start + chunk_chars] for start in range(0, len(text), chunk_chars)] or [""]
        order = rank(lexical_scores(windows, history.question), len(windows))
        parts, used = [], 0
        for window in order:
            remaining = allowance - used
            if remaining <= 0:
                break
            snippet = windows[window][:remaining]
            parts.append((window, snippet))
            used += len(snippet) + 1
        excerpt = "\n".join(part for _, part in sorted(parts))
        block = (header + excerpt)[:max(0, per_session - 2)]
        chunks.append(block)
        trace.append({"session_id": history.session_ids[index], "context_id": context_id, "date": history.dates[index],
                      "original_chars": len(text), "excerpt_chars": len(excerpt), "truncated": len(excerpt) < len(text)})
    return "\n\n".join(chunks)[:budget], trace
