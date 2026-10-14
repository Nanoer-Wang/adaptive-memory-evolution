"""Causal scorer-call budgets; skipped sessions are still written and retrieved.

This is a research control, not a proven liquid-specific decision rule. Every
writer receives the identical observation-only gate. No queries or labels.
"""
import numpy as np
import torch


def sparse_scores(model, vectors, deltas, policy='full'):
    if policy not in ('full', 'stride2', 'stride4', 'novelty'):
        raise ValueError('Unknown scorer budget')
    state = vectors.new_zeros(model.hidden)
    scores, calls = [], []
    last = -1; pending = 0.; previous = vectors[0]
    last_score = vectors.new_zeros(())
    for i, (vector, delta) in enumerate(zip(vectors, deltas)):
        pending += float(delta)
        gap = i-last
        if policy == 'full':
            evaluate = True
        elif policy.startswith('stride'):
            evaluate = gap >= int(policy[-1])
        else:
            similarity = float(torch.nn.functional.cosine_similarity(vector, previous, dim=0))
            evaluate = gap >= 4 or similarity < .8
        evaluate = i == 0 or evaluate
        if evaluate:
            last_score, state = model.step(vector, state, min(12., pending))
            pending = 0.; last = i; previous = vector
            calls.append(i)
        # Hold the last insertion score; never resurrect or rewrite a past item.
        scores.append(last_score)
    return torch.stack(scores), np.asarray(calls, dtype=np.int64)
