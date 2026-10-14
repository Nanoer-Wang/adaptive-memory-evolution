"""Train-selected, validation-gated content preservation diagnostic; no API calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import hashlib
import re
import numpy as np  # Must precede torch in this Windows environment.
from src.config import resolve, atomic_json, load_config
from src.data.longmemeval import prepared, session_text
from src.memory.content_store import bounded_text
from src.memory.turn_compression import compress_turns, sentences
from src.provenance import snapshot_source, utc_now, runtime_info

OUT = resolve('runs/compression-repair-v2')
MODES = ('legacy', 'whole_spread', 'sentence_spread', 'sentence_user_first')


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def evaluate(rows, modes):
    records = []
    for row in rows:
        # Unanswerable variants can retain stale has_answer flags upstream.
        if str(row['question_id']).endswith('_abs') or not row.get('answer_session_ids'):
            continue
        targets = [(turns, [t for t in turns if t.get('has_answer') is True])
                   for turns in row['haystack_sessions']]
        targets = [(turns, evidence) for turns, evidence in targets if evidence]
        if not targets:
            continue
        for mode in modes:
            full, units, costs, budgets = [], [], [], []
            started = time.perf_counter()
            for turns, evidence in targets:
                legacy = bounded_text(session_text(turns))
                budget = len(legacy.encode('utf-8'))
                packed = legacy if mode == 'legacy' else compress_turns(turns, budget, mode)
                text = normalized(packed)
                # Repeated role prefixes may interrupt an otherwise intact turn.
                # Strip generated per-unit prefixes for content continuity scoring.
                content = normalized(re.sub(r'(?m)^(user|assistant|system|unknown): ', '', packed))
                for turn in evidence:
                    value = normalized(turn['content'])
                    if value:
                        full.append(float(value in text or value in content))
                    for sentence in sentences(turn['content']):
                        units.append(float(normalized(sentence) in text))
                costs.append(len(packed.encode('utf-8')))
                budgets.append(budget)
            records.append({'question_id': row['question_id'], 'category': row['question_type'],
                            'method': mode, 'turn_coverage': float(np.mean(full)) if full else 0.,
                            'sentence_coverage': float(np.mean(units)) if units else 0.,
                            'annotated_turns': len(full), 'annotated_sentences': len(units),
                            'text_bytes': sum(costs), 'budget_bytes': sum(budgets),
                            'elapsed_seconds': time.perf_counter() - started})
    return records


def aggregate(records):
    result = {}
    for mode in sorted({r['method'] for r in records}):
        subset = [r for r in records if r['method'] == mode]
        def metrics(values):
            return {'n': len(values), **{key: float(np.mean([r[key] for r in values]))
                    for key in ('turn_coverage', 'sentence_coverage')},
                    'byte_utilization': sum(r['text_bytes'] for r in values) / sum(r['budget_bytes'] for r in values)}
        result[mode] = {'overall': metrics(subset), 'categories': {
            category: metrics([r for r in subset if r['category'] == category])
            for category in sorted({r['category'] for r in subset})}}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Versioned run already exists; preserve it and create a new protocol version.')
    source, archive = snapshot_source()
    config = load_config('configs/minute_precision.yaml')
    rows, manifest = prepared(config)
    splits = manifest['splits']
    protocol = {'created_at': utc_now(), 'source_sha256': source, 'source_archive': archive,
                'correction': 'v1 accidentally scored residual has_answer flags in unanswerable variants. v2 excludes _abs and empty answer_session_ids; candidates, selection rule and gate unchanged. v1 retained as superseded diagnostic.',
                'runtime': runtime_info(), 'dataset_sha256': manifest['signature']['sha256'],
                'candidates': MODES, 'train_ids': splits['train'], 'validation_ids': splits['val'],
                'budget': 'Each session gets exactly the UTF-8 byte upper bound consumed by legacy bounded_text(2400 chars). Actual utilization reported; no embeddings changed.',
                'scope': 'Independent per-session packing, conditioned on annotated evidence sessions for scoring only. All roles allowed. Does not measure eviction, retrieval, generation or official benchmark accuracy. No external/test scores.',
                'selection': 'On train only, maximize mean of question-macro full-turn and sentence coverage; candidate order breaks ties.',
                'gate': 'Validation sentence coverage gain >= 0.05, full-turn gain >= 0, and no question-type sentence drop > 0.05.',
                'limitations': ['Existing validation has been exposed in earlier research.', 'Sentence overlap is a proxy, not semantic fact correctness.', 'Only answerable questions with has_answer=True turns can be scored.', 'No role-specific or latest-fact labels enter compression.']}
    atomic_json(OUT / 'protocol.json', protocol)
    train = evaluate([r for r in rows if r['question_id'] in set(splits['train'])], MODES)
    train_summary = aggregate(train)
    score = lambda mode: sum(train_summary[mode]['overall'][key] for key in ('turn_coverage', 'sentence_coverage')) / 2
    selected = max(MODES[1:], key=score)
    # Selection is persisted before any validation evaluation.
    atomic_json(OUT / 'train_selection.json', {'selected': selected, 'train': train_summary,
                'selected_score': score(selected), 'legacy_score': score('legacy'), 'selected_at': utc_now()})
    validation = evaluate([r for r in rows if r['question_id'] in set(splits['val'])], ('legacy', selected))
    val_summary = aggregate(validation)
    base, candidate = val_summary['legacy'], val_summary[selected]
    deltas = {key: candidate['overall'][key] - base['overall'][key]
              for key in ('turn_coverage', 'sentence_coverage')}
    category_deltas = {category: candidate['categories'][category]['sentence_coverage'] - values['sentence_coverage']
                       for category, values in base['categories'].items()}
    passed = deltas['sentence_coverage'] >= .05 and deltas['turn_coverage'] >= 0 and min(category_deltas.values()) >= -.05
    # Paired bootstrap at question level; descriptive, not independent evidence-group inference.
    rng = np.random.default_rng(42)
    baseline = {r['question_id']: r for r in validation if r['method'] == 'legacy'}
    difference = np.array([r['sentence_coverage'] - baseline[r['question_id']]['sentence_coverage']
                           for r in validation if r['method'] == selected])
    boot = rng.choice(difference, (5000, len(difference)), replace=True).mean(axis=1)
    result = {'selected': selected, 'gate_passed': bool(passed), 'validation': val_summary,
              'validation_deltas': deltas, 'category_sentence_deltas': category_deltas,
              'descriptive_question_bootstrap_sentence_delta_ci95': np.quantile(boot, [.025, .975]).tolist(),
              'api_calls': 0, 'finished_at': utc_now(),
              'next_decision': 'Proceed to matched downstream pilot; no liquid advantage claim.' if passed else 'Do not deploy automatically or launch large QA. Revisit representation using training data.'}
    for split, records in (('train', train), ('validation', validation)):
        (OUT / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    result['record_hashes'] = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                              for name in ('train.jsonl', 'validation.jsonl')}
    atomic_json(OUT / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
