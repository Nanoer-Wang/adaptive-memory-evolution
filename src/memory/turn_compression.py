"""Query-independent, byte-bounded extractive packing of structured turns.

This development module does not replace the frozen content-memory-v1 writer.
No question, evidence IDs, or annotation fields enter packing.
"""
import re


def sentences(text):
    return [part.strip() for part in re.split(r'(?<=[.!?。！？])\s+|\n+', text)
            if part.strip()]


def spread_order(size):
    """Cover both ends, then recursively fill the largest uncovered intervals."""
    if size == 0:
        return []
    result = [0] if size == 1 else [0, size - 1]
    intervals = [(0, size - 1)]
    while intervals:
        left, right = intervals.pop(0)
        if right - left > 1:
            middle = (left + right) // 2
            result.append(middle)
            intervals.extend([(left, middle), (middle, right)])
    return result


def compress_turns(turns, budget_bytes, mode='whole_spread'):
    if budget_bytes < 0:
        raise ValueError('budget_bytes must be nonnegative')
    if mode not in ('whole_spread', 'sentence_spread', 'sentence_user_first'):
        raise ValueError('Unknown compression mode')
    # Explicit allowlist; upstream has_answer and other labels are ignored.
    units = []
    for turn_index, turn in enumerate(turns):
        role = str(turn.get('role', 'unknown'))
        content = str(turn.get('content', ''))
        parts = [content] if mode == 'whole_spread' else sentences(content)
        for part_index, part in enumerate(parts):
            units.append((turn_index, part_index, role, part))
    order = spread_order(len(units))
    if mode == 'sentence_user_first':
        order.sort(key=lambda i: units[i][2] != 'user')
    chosen = []
    used = 0
    for index in order:
        turn_index, part_index, role, content = units[index]
        # Each extracted unit is independently attributed, including its cost.
        rendered = f'{role}: {content}'
        cost = len(rendered.encode('utf-8')) + bool(chosen)
        if used + cost <= budget_bytes:
            chosen.append((turn_index, part_index, rendered))
            used += cost
    result = '\n'.join(unit[2] for unit in sorted(chosen))
    assert len(result.encode('utf-8')) <= budget_bytes
    return result
