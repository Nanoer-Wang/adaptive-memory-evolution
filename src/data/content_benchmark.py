"""Public content views are kept separate from evaluation annotations."""
import hashlib
import json
import re
from datetime import datetime
from src.config import resolve
from src.data.longmemeval import History, public_history, prepared, date_days


def canonical_turn(value):
    match = re.fullmatch(r'D\s*(\d+)\s*:\s*(\d+)', str(value).strip())
    return f'D{int(match[1])}:{int(match[2])}' if match else str(value).strip()


def locomo_examples(samples):
    examples = []
    for sample in samples:
        conversation = sample['conversation']
        group = str(sample['sample_id'])
        sessions, turn_map = [], {}
        for key, turns in conversation.items():
            if not re.fullmatch(r'session_\d+', key):
                continue
            date = datetime.strptime(conversation[key + '_date_time'], '%I:%M %p on %d %B, %Y').strftime('%Y/%m/%d %H:%M')
            sid = group + ':' + key
            text = '\n'.join(str(t.get('speaker', 'unknown')) + ': ' + str(t.get('text', '')) +
                             ('\nImage caption: ' + str(t['blip_caption']) if t.get('blip_caption') else '') for t in turns)
            sessions.append((sid, date, text))
            for turn in turns:
                turn_map[canonical_turn(turn['dia_id'])] = sid
        sessions.sort(key=lambda item: date_days(item[1], 'minute'))
        for index, qa in enumerate(sample['qa']):
            abstention = qa['category'] == 5
            qid = f'locomo:{group}:{index:04d}' + ('_abs' if abstention else '')
            evidence = [canonical_turn(value) for value in qa.get('evidence', [])]
            unknown = [value for value in evidence if value not in turn_map]
            gold = sorted({turn_map[value] for value in evidence if value in turn_map})
            reason = 'unanswerable' if abstention else 'unmapped_evidence' if unknown else 'missing_evidence' if not gold else None
            history = History(qid, qa['question'], sessions[-1][1], tuple(s[0] for s in sessions),
                              tuple(s[1] for s in sessions), tuple(s[2] for s in sessions))
            examples.append(dict(history=history, dataset='locomo', group=group,
                                 answer='Cannot determine from the available history.' if abstention else str(qa.get('answer', '')),
                                 category='locomo-' + str(qa['category']), gold=gold,
                                 exclusion=reason, unmapped_evidence=unknown,
                                 question_date_assumed=True))
    return examples


def load_examples(config):
    rows, manifest = prepared(config)
    split_by_id = {qid: split for split, ids in manifest['splits'].items() for qid in ids}
    train, validation = [], []
    for row in rows:
        split = split_by_id[str(row['question_id'])]
        if split not in ('train', 'val'):
            continue
        history = public_history(row, 'minute')
        gold = sorted(set(map(str, row.get('answer_session_ids', []))))
        example = dict(history=history, dataset='longmemeval', group=history.question_id,
                       answer=str(row['answer']), category=row['question_type'], gold=gold,
                       exclusion='unanswerable' if history.question_id.endswith('_abs') else 'missing_evidence' if not gold else None,
                       unmapped_evidence=[], question_date_assumed=False)
        (train if split == 'train' else validation).append(example)
    samples = json.loads(resolve('data/locomo_reserved/locomo10.json').read_text(encoding='utf-8'))
    return train, validation, locomo_examples(samples), manifest


def choose_qa(validation, external):
    key = lambda e: hashlib.sha256(e['history'].question_id.encode()).hexdigest()
    result = sorted([e for e in validation if e['category'] == 'knowledge-update'], key=key)[:10]
    for group in sorted({e['group'] for e in external}):
        result.extend(sorted([e for e in external if e['group'] == group], key=key)[:2])
    return [e['history'].question_id for e in result]
