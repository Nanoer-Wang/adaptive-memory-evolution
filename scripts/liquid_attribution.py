"""Frozen language writer attribution; no training or API calls."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json, hashlib, time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from src.config import load_config, atomic_json
from src.data.longmemeval import prepared, public_history, date_days
from src.data.content_benchmark import locomo_examples
from src.memory.turn_compression import compress_turns
from src.memory.content_store import ContentWriter
from src.memory.budget_store import BudgetItem, byte_retained_indices
from src.models.content_encoder import ContentEncoder
from scripts.content_experiment import Reader, deltas
from src.provenance import snapshot_source, runtime_info

OUT = ROOT / 'runs/liquid-attribution-v1'
SEEDS = [11, 23, 42]
BUDGETS = [32000, 64000, 128000]

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def examples():
    rows, manifest = prepared(load_config('configs/minute_precision.yaml'))
    result = []
    for r in rows:
        if r['question_id'] not in manifest['splits']['test']:
            continue
        h = public_history(r, 'minute')
        ordered = sorted(zip(r['haystack_dates'], r['haystack_sessions']), key=lambda x: date_days(x[0], 'minute'))
        h = replace(h, texts=tuple(compress_turns(t, 2400, 'sentence_user_first') for _, t in ordered))
        result.append(dict(history=h, dataset='longmemeval', group=h.question_id,
                           category=r['question_type'], gold=list(map(str, r.get('answer_session_ids', []))),
                           eligible=not h.question_id.endswith('_abs')))
    for e in locomo_examples(read(ROOT / 'data/locomo_reserved/locomo10.json')):
        if e['exclusion'] not in (None, 'unanswerable'):
            continue
        h = e['history']
        h = replace(h, texts=tuple(compress_turns([dict(role='dialogue', content=t)], 2400, 'sentence_user_first') for t in h.texts))
        result.append(dict(history=h, dataset=e['dataset'], group=e['group'], category=e['category'], gold=e['gold'], eligible=e['exclusion'] is None))
    return result

def score(m, v, dt, mode):
    if mode == 'reset':
        return torch.stack([m.step(x, x.new_zeros(m.hidden), d)[0] for x, d in zip(v, dt)])
    if mode == 'zero_time':
        dt = torch.zeros_like(dt)
    if mode == 'long_gaps':
        dt = torch.clamp(dt * 10, max=12)
    return m(v, dt)

def run():
    if (OUT / 'complete.json').exists():
        return
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Partial run exists: preserve before retry')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    models, hashes, counts = {}, {}, {}
    for seed in SEEDS:
        for kind in ['static_content', 'gru_content', 'liquid_content', 'liquid_fixed_time']:
            p = ROOT / 'runs/preservation-study-v2/checkpoints' / f'{kind}-{seed}/best.pt'
            m = ContentWriter('liquid_content' if kind == 'liquid_fixed_time' else kind)
            if kind == 'liquid_fixed_time':
                m.cell.fixed_time = True
            m.load_state_dict(torch.load(p, weights_only=True)['state_dict'])
            m.eval()
            models[kind, seed] = m
            hashes[f'{kind}-{seed}'] = hashlib.sha256(p.read_bytes()).hexdigest()
            counts[kind] = sum(x.numel() for x in m.parameters())
    signature, archive = snapshot_source()
    atomic_json(OUT / 'protocol.json', dict(runtime=runtime_info(), source=signature, archive=archive,
        checkpoint_hashes=hashes, parameters=counts, seeds=SEEDS, budgets=BUDGETS,
        primary='LongMemEval evidence retrieval recall at 64000 bytes, average of three paired seeds per question. Existing exposed development data, not a fresh confirmatory test.',
        comparisons=['liquid vs GRU', 'liquid vs static', 'liquid vs same-weight reset', 'liquid vs same-weight zero-time'],
        success_rule='Mechanism-supporting accuracy signal requires all four primary paired differences positive with 95% history-cluster bootstrap lower bounds > 0; additionally report 98.75% intervals for four-comparison control. A positive result still requires unseen confirmation and downstream QA.',
        bootstrap='5000 paired history-cluster resamples, seed 20261012; aggregate seeds first, never count seeds as independent questions. Secondary budgets/categories are exploratory.',
        controls='Same 2400-byte extracted text, frozen MiniLM, hybrid reader top5, byte accounting; no query or labels during writes. Same-weight reset, zero-time and 10x clipped gaps are inference interventions, not retrained baselines. Independently trained fixed-time baseline also included.',
        timing='CPU one torch thread, cached input vectors; one warm-up, median of 3 writer-only repeats per history. Excludes encoding, storage eviction and retrieval. Not end-to-end speed.',
        limits='Actual cell is exponential gated discrete recurrence, not a numerical continuous-time ODE solver. Priorities fixed at insertion. Payload excludes shared weights, Python objects and reader index. Full retrieval is a reference, not an upper bound. No API usage.'))
    ee = examples(); groups = defaultdict(list)
    for e in ee:
        h = e['history']
        key = hashlib.sha256(json.dumps([h.session_ids, h.dates, h.texts], ensure_ascii=False).encode()).hexdigest()
        groups[key].append(e)
    encoder = ContentEncoder(); diagnostics = []; timings = []
    with (OUT / 'records.jsonl').open('w', encoding='utf-8') as f, torch.no_grad():
        for gi, (key, group) in enumerate(groups.items()):
            h = group[0]['history']; vec = encoder.encode(list(h.texts))
            v = torch.from_numpy(vec); dt = torch.from_numpy(deltas(h))
            costs = np.array([BudgetItem(s, d, t, z, 0, i).nbytes for i, (s, d, t, z) in enumerate(zip(h.session_ids, h.dates, h.texts, vec))])
            priorities = {('fifo', 0): np.arange(len(v), dtype=float)}
            states = {('fifo', 0): 0}
            for seed in SEEDS:
                for name, kind, mode in [
                    ('static', 'static_content', 'normal'), ('gru', 'gru_content', 'normal'),
                    ('liquid', 'liquid_content', 'normal'), ('fixed_trained', 'liquid_fixed_time', 'normal'),
                    ('reset', 'liquid_content', 'reset'), ('zero_time', 'liquid_content', 'zero_time'),
                    ('long_gaps', 'liquid_content', 'long_gaps')]:
                    m = models[kind, seed]
                    out = score(m, v, dt, mode)
                    ts = []
                    for _ in range(3):
                        start = time.perf_counter(); score(m, v, dt, mode); ts.append(time.perf_counter() - start)
                    priorities[name, seed] = out.numpy(); states[name, seed] = m.hidden * 4
                    timings.append(dict(history=key, dataset=group[0]['dataset'], method=name, seed=seed, sessions=len(v), seconds=float(np.median(ts)), repeats=ts))
                for ablation in ['reset', 'zero_time', 'long_gaps', 'fixed_trained']:
                    a, b = priorities['liquid', seed], priorities[ablation, seed]
                    diagnostics.append(dict(history=key, dataset=group[0]['dataset'], seed=seed, ablation=ablation,
                        max_score_change=float(np.max(np.abs(a-b))), mean_score_change=float(np.mean(np.abs(a-b))),
                        identical_ranking=bool(np.array_equal(np.argsort(a, kind='stable'), np.argsort(b, kind='stable')))))
            queries = encoder.encode([e['history'].question for e in group])
            # Cache identical surviving stores within a history. Labels never enter this cache key.
            retrieval_cache = {}
            conditions = [('full', 0, 0, list(range(len(v))), 0)]
            for (name, seed), sc in priorities.items():
                for budget in BUDGETS:
                    state = states[name, seed]
                    kept = byte_retained_indices(sc, costs, budget, state+16)
                    conditions.append((name, seed, budget, kept, state))
            for name, seed, budget, kept, state in conditions:
                cachekey = tuple(kept)
                if cachekey not in retrieval_cache:
                    items = [BudgetItem(h.session_ids[i], h.dates[i], h.texts[i], vec[i], 0, i) for i in kept]
                    reader = Reader(SimpleNamespace(items=items)) if items else None
                    retrieval_cache[cachekey] = [[items[i].session_id for i in reader.query(e['history'].question, q)[0]] if reader else [] for e, q in zip(group, queries)]
                used = int(costs[kept].sum()+state+16)
                assert budget == 0 or used <= budget
                retained = {h.session_ids[i] for i in kept}
                for e, selected in zip(group, retrieval_cache[cachekey]):
                    gold = set(e['gold']); eligible = e['eligible'] and bool(gold)
                    r = dict(question_id=e['history'].question_id, dataset=e['dataset'], cluster=key, category=e['category'], method=name, seed=seed, budget=budget,
                        retrieval_recall=len(gold & set(selected))/len(gold) if eligible else None,
                        evidence_retention=len(gold & retained)/len(gold) if eligible else None,
                        payload_bytes=used, retained_indices=kept, retrieved_sources=selected)
                    f.write(json.dumps(r, ensure_ascii=False)+'\n')
            f.flush()
            if gi % 5 == 0:
                print(f'Attribution histories {gi+1}/{len(groups)}', flush=True)
    encoder.close()
    atomic_json(OUT/'timing.json', timings); atomic_json(OUT/'diagnostics.json', diagnostics)
    atomic_json(OUT/'complete.json', dict(histories=len(groups), questions=len(ee), api_requests=0))

def analyze():
    records = [json.loads(s) for s in (OUT/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for r in records:
        if r['retrieval_recall'] is not None:
            groups[r['dataset'], r['budget'], r['method']].append(r)
    summary = []
    for (dataset, budget, method), rs in groups.items():
        summary.append(dict(dataset=dataset, budget=budget, method=method, questions=len({r['question_id'] for r in rs}),
            retrieval_recall=float(np.mean([r['retrieval_recall'] for r in rs])),
            evidence_retention=float(np.mean([r['evidence_retention'] for r in rs])),
            payload_bytes=float(np.mean([r['payload_bytes'] for r in rs]))))
    comparisons = []
    for dataset in ['longmemeval', 'locomo']:
        for budget in BUDGETS:
            byq = defaultdict(lambda: defaultdict(list)); clusters = {}
            for r in records:
                if r['dataset'] == dataset and r['budget'] == budget and r['retrieval_recall'] is not None:
                    byq[r['question_id']][r['method']].append(r['retrieval_recall']); clusters[r['question_id']] = r['cluster']
            for other in ['gru', 'static', 'reset', 'zero_time', 'fixed_trained', 'fifo', 'long_gaps']:
                cs = defaultdict(list)
                for q, methods in byq.items():
                    cs[clusters[q]].append(float(np.mean(methods['liquid']) - np.mean(methods[other])))
                sums = np.array([sum(x) for x in cs.values()]); ns = np.array([len(x) for x in cs.values()])
                rng = np.random.default_rng(20261012); ix = rng.integers(0, len(cs), size=(5000, len(cs)))
                bs = sums[ix].sum(axis=1)/ns[ix].sum(axis=1)
                comparisons.append(dict(dataset=dataset, budget=budget, comparison='liquid-'+other,
                    difference=float(sums.sum()/ns.sum()), ci95=np.quantile(bs,[.025,.975]).tolist(),
                    ci9875=np.quantile(bs,[.00625,.99375]).tolist(), clusters=len(cs), questions=int(ns.sum())))
    atomic_json(OUT/'summary.json', summary); atomic_json(OUT/'comparisons.json', comparisons)
    print(json.dumps([r for r in comparisons if r['dataset']=='longmemeval' and r['budget']==64000], indent=2))

if __name__ == '__main__':
    run(); analyze()
