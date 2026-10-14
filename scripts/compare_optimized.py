"""Freeze validation selection before test; save paired evidence and uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config import load_config, resolve, atomic_json
from src.data.download import file_sha256
from src.evaluation.runner import evaluate
from src.provenance import utc_now, snapshot_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/optimized.yaml')
    parser.add_argument('--output', default='runs/optimization-v1')
    parser.add_argument('--lcm-checkpoint', default='artifacts/checkpoints/lcm-v2-trial/best.pt')
    parser.add_argument('--ranker-checkpoint', default='artifacts/checkpoints/ranker-trial/best.pt')
    args = parser.parse_args()
    config, root = load_config(args.config), resolve(args.output)
    checkpoints = {'tfidf': None, 'ranker': args.ranker_checkpoint, 'lcm_v2': args.lcm_checkpoint}
    validation = {}
    for method, checkpoint in checkpoints.items():
        validation[method] = evaluate(config, method, 'val', checkpoint=checkpoint,
                                     retrieval_only=True, output=root / (method + '-val'))
    # Stable tie-break favors the simpler candidate. Test is not involved.
    selected = max(checkpoints, key=lambda m: validation[m]['metrics']['session_recall_at_k'])
    frozen = {'selected': selected, 'criterion': 'validation unique-session Recall@5; ties favor simpler model',
              'validation': validation, 'checkpoint_sha256': {m:file_sha256(resolve(p)) if p else None for m,p in checkpoints.items()},
              'source_sha256': snapshot_source()[0]}
    selection_path = root / 'selection.json'
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding='utf-8'))
        if {k:v for k,v in previous.items() if k != 'frozen_at_utc'} != frozen:
            raise ValueError('Frozen selection differs; use a new output directory')
    else:
        atomic_json(selection_path, {**frozen, 'frozen_at_utc': utc_now()})
    tests = {m: evaluate(config, m, 'test', checkpoint=p, retrieval_only=True,
                         output=root / (m + '-test')) for m,p in checkpoints.items()}
    records = {}
    for method in checkpoints:
        path = root / (method + '-test') / 'predictions.jsonl'
        records[method] = {r['question_id']:r for line in path.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))}
    pairs = []
    for qid, baseline in records['tfidf'].items():
        pairs.append({'question_id': qid, 'question_type': baseline['question_type'],
                      **{m: records[m][qid]['session_recall_at_k'] for m in checkpoints}})
    atomic_json(root / 'paired_test.json', pairs)
    intervals = {}
    for left, right in [('lcm_v2', 'tfidf'), ('lcm_v2', 'ranker')]:
        differences = np.array([r[left]-r[right] for r in pairs if r[left] is not None])
        rng = np.random.default_rng(42)
        boot = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
        intervals[left + '_minus_' + right] = {'mean':float(differences.mean()),
            'paired_question_bootstrap_95_percent': np.quantile(boot,[.025,.975]).tolist(),
            'improved':int((differences>0).sum()), 'worse':int((differences<0).sum()),
            'unchanged':int((differences==0).sum()),
            'caveat':'Exploratory question-level interval; shared histories create dependence. Not a significance claim.'}
    atomic_json(root / 'comparison.json', {'validation_selected':selected, 'test':tests, 'paired_differences':intervals,
        'protocol_note':'Custom split; test previously evaluated for v1. Current parameters selected only on validation. Retrieval, not QA accuracy.',
        'performance_note':'ranking_feature_cache_hit is saved per question. Cold feature extraction and warm inference are not comparable.'})
    print(json.dumps({'selected': selected, 'test_recall':{m:r['metrics']['session_recall_at_k'] for m,r in tests.items()},
                      'paired_differences':intervals}, indent=2))


if __name__ == '__main__':
    main()
