"""Avoid introducing artificial word breaks into text that already fits."""
import re
from src.retrieval.retriever import opaque_session_id, lexical_scores, rank


def pack_preserved_context(history, selected, budget=16000):
    if not selected:
        return '', []
    blocks, trace = [], []
    allowance_per_session = budget // len(selected)
    for index in selected:
        header = f'[session={opaque_session_id(history.session_ids[index])} date={history.dates[index]}]\n'
        allowance = max(0, allowance_per_session - len(header) - 2)
        text = history.texts[index]
        if len(text) <= allowance:
            excerpt = text
        else:
            # Existing complete rendered units; never slice inside a unit.
            units = [part for part in re.split(r'\n+', text) if part]
            chosen, used = [], 0
            for position in rank(lexical_scores(units, history.question), len(units)):
                cost = len(units[position]) + bool(chosen)
                if used + cost <= allowance:
                    chosen.append(position)
                    used += cost
            excerpt = '\n'.join(units[i] for i in sorted(chosen))
        if len(header) + 2 > allowance_per_session:
            continue
        blocks.append(header + excerpt)
        trace.append(dict(session_id=history.session_ids[index], context_id=opaque_session_id(history.session_ids[index]),
                          original_chars=len(text), excerpt_chars=len(excerpt), truncated=excerpt != text))
    context = '\n\n'.join(blocks)
    assert len(context) <= budget
    return context, trace
