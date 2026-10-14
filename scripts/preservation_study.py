"""Versioned continuation: context preservation, paired QA, byte-budget controls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from filelock import FileLock
from src.config import resolve, load_config, atomic_json
from src.data.longmemeval import prepared, public_history, date_days, session_text
from src.data.download import file_sha256
from src.memory.content_store import bounded_text, ContentWriter, ContentStore
from src.memory.turn_compression import compress_turns, sentences
from src.memory.budget_store import BudgetStore, BudgetItem, byte_retained_indices
from src.models.content_encoder import ContentEncoder
from src.retrieval.retriever import pack_context
from src.retrieval.preserved_context import pack_preserved_context
from src.evaluation.runner import messages_for, custom_judge
from src.evaluation.metrics import answer_metrics
from src.api.agnes import AgnesClient
from src.provenance import snapshot_source, utc_now, runtime_info, rss_mb
from scripts.content_experiment import Reader, deltas

ROOT = resolve('runs/preservation-study-v2')
KINDS = ('static_content', 'liquid_content', 'liquid_fixed_time', 'gru_content')
SEEDS = (11, 23, 42)
BUDGETS = (32000, 128000)


def log(text):
    print(f'{utc_now()} {text}', flush=True)


def dump_lines(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    temporary.replace(path)


def eligible(row):
    return not str(row['question_id']).endswith('_abs') and bool(row.get('answer_session_ids'))


def history(row, repaired):
    original = public_history(row, 'minute')
    if not repaired:
        return replace(original, texts=tuple(bounded_text(text) for text in original.texts))
    ordered = sorted(zip(row['haystack_session_ids'], row['haystack_dates'], row['haystack_sessions']),
                     key=lambda item: date_days(item[1], 'minute'))
    texts = tuple(compress_turns(turns, len(bounded_text(session_text(turns)).encode()), 'sentence_user_first')
                  for _, _, turns in ordered)
    return replace(original, texts=texts)


def coverage(row, text):
    normalized = ' '.join(text.split())
    values = [float(' '.join(sentence.split()) in normalized)
              for turns in row['haystack_sessions'] for turn in turns if turn.get('has_answer') is True
              for sentence in sentences(turn['content'])]
    return float(np.mean(values)) if values else None


def model_for(kind):
    model = ContentWriter('liquid_content' if kind == 'liquid_fixed_time' else kind)
    if kind == 'liquid_fixed_time':
        model.cell.fixed_time = True
    return model


def freeze(rows, manifest):
    api = load_config('configs/minute_precision.yaml')['api'].copy()
    api['rpm'] = 9
    validation = [row for row in rows if row['question_id'] in manifest['splits']['val'] and eligible(row)]
    # Deterministic stratification fixed before downstream scores, no failure selection.
    sample = []
    for category, count in [('knowledge-update', 8), ('single-session-assistant', 8), ('temporal-reasoning', 2)]:
        sample.extend(sorted([r['question_id'] for r in validation if r['question_type'] == category],
                             key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:count])
    signature, archive = snapshot_source()
    spec = dict(source_signature=signature, source_archive=archive, data_sha256=manifest['signature']['sha256'],
                parent_sha256=file_sha256(resolve('runs/content-memory-v1/results.sqlite3')),
                compression_selection_sha256=file_sha256(resolve('runs/compression-repair-v2/train_selection.json')),
                train_ids=manifest['splits']['train'], validation_ids=manifest['splits']['val'], qa_ids=sample, api=api,
                specification=dict(qa='18 predetermined validation questions x old/new preservation. Identical parent full-hybrid selected sessions, top5, 16000 chars, max answer/judge512. No oracle selection. Same-provider custom judge, not official.',
                mechanism='Re-encode repaired text, train 4 controls x 3 seeds x 3 epochs on train only; choose best checkpoint by mean validation byte-budget evidence retention. Development selection and scoring share validation; not confirmatory.',
                training='AdamW lr0.003 wd0.01, balanced BCE evidence-session labels, clip1, CPU4 threads. Fixed-time is independently trained, static resets hidden state.',
                storage='32k/128k decimal bytes: UTF8 text+source ID+date+384float32 vector+priority/sequence16B+hidden state; excludes shared weights and Python/index/trace artifacts; no deleted content read.',
                gates='QA preliminary: at least 2 net additional correct out of18; assistant subgroup no worse than1/8. Mechanism: liquid beats static and GRU by >=1pp mean context sentence coverage, >=2/3 seeds, paired question bootstrap lower bound>0; development gate only.',
                limitations=['Validation previously exposed; no independent/generalization claim.', 'Extractive units preserve version dates, but no semantic entity resolution or truth-changing event-time inference.', 'Session-level eviction remains; memory contributions distinguished from representation.', 'Mechanism diagnostic runs even if small QA gate fails, but is explicitly exploratory.'],
                kinds=KINDS, seeds=SEEDS, budgets=BUDGETS))
    path = ROOT / 'execution_manifest.json'
    # JSON canonicalization makes resumed tuple/list comparisons equivalent.
    spec = json.loads(json.dumps(spec))
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if any(saved[key] != value for key, value in spec.items()):
            raise RuntimeError('Frozen experiment changed; do not overwrite')
    else:
        atomic_json(path, dict(**spec, created_at=utc_now(), runtime=runtime_info()))
    return spec


def contexts(rows, spec):
    if (ROOT / 'qa_tasks.jsonl').exists():
        return
    parent = resolve('runs/content-memory-v1/results.sqlite3')
    with sqlite3.connect(f'file:{parent.as_posix()}?mode=ro', uri=True) as db:
        records = [json.loads(value) for (value,) in db.execute('SELECT payload FROM retrieval')]
    original = {r['question_id']: r for r in records if r['method'] == 'full_hybrid' and r['dataset'] == 'longmemeval'}
    tasks, metrics = [], []
    encoder = None
    for row in rows:
        if row['question_id'] not in spec['validation_ids'] or not eligible(row):
            continue
        source = original[row['question_id']]
        baseline_history = history(row, False)
        if len(set(baseline_history.session_ids)) != len(baseline_history.session_ids):
            # Upstream IDs can repeat with distinct texts/dates. Reproduce the
            # frozen parent's retrieval rather than guessing the first match.
            encoder = encoder or ContentEncoder()
            store = ContentStore()
            vectors = encoder.encode(list(baseline_history.texts))
            for sid, date, text, vector, delta in zip(baseline_history.session_ids, baseline_history.dates, baseline_history.texts, vectors, deltas(baseline_history)):
                store.insert(sid, date, text, vector, float(delta))
            selected, scores = Reader(store).query(baseline_history.question, encoder.encode([baseline_history.question])[0])
            if [baseline_history.session_ids[i] for i in selected] != source['selected_ids'] or not np.allclose([scores[i] for i in selected], source['selected_scores'], atol=1e-6):
                raise RuntimeError('Could not reproduce ambiguous parent retrieval')
        else:
            selected = [baseline_history.session_ids.index(sid) for sid in source['selected_ids']]
        for repaired in (False, True):
            h = history(row, repaired)
            packer = pack_preserved_context if repaired else pack_context
            context, trace = packer(h, selected)
            method = 'repaired' if repaired else 'legacy'
            metrics.append(dict(question_id=h.question_id, category=row['question_type'], method=method,
                                sentence_coverage=coverage(row, context), context_chars=len(context),
                                selected_ids=source['selected_ids'], selected_indices=selected, context_trace=trace))
            if h.question_id in spec['qa_ids']:
                tasks.append(dict(key=method+':'+h.question_id, method=method, question_id=h.question_id,
                                  question=h.question, question_date=h.question_date, answer=str(row['answer']),
                                  category=row['question_type'], context=context, context_trace=trace, selected_indices=selected))
    if encoder is not None:
        encoder.close()
    dump_lines(ROOT / 'context_metrics.jsonl', metrics)
    dump_lines(ROOT / 'qa_tasks.jsonl', tasks)
    log(f'Contexts prepared: {len(metrics)} diagnostics, {len(tasks)} QA tasks')


def qa(spec):
    tasks = [json.loads(line) for line in (ROOT / 'qa_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    result_dir = ROOT / 'qa_records'
    result_dir.mkdir(exist_ok=True)
    namespace = 'preservation-study-v2:' + file_sha256(ROOT / 'execution_manifest.json')
    def evaluate(row):
        path = result_dir / (hashlib.sha256(row['key'].encode()).hexdigest() + '.json')
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(row)
        if result.get('custom_judge_correct') is not None:
            return result
        client = AgnesClient(spec['api'])
        try:
            if 'generation' not in result:
                messages = messages_for(SimpleNamespace(question_date=row['question_date'], question=row['question']), row['context'])
                result['messages'] = messages
                result['generation'] = client.complete(messages, namespace=namespace)
                result['hypothesis'] = result['generation']['content']
                result.update(answer_metrics(result['hypothesis'], row['answer']))
                atomic_json(path, result)
            grade, response = custom_judge(client, row, result['hypothesis'], namespace, max_tokens=512)
            result.update(custom_judge_correct=grade, judge_response=response)
            result.pop('error_type', None)
            atomic_json(path, result)
            return result
        except Exception as exc:
            result['error_type'] = type(exc).__name__
            atomic_json(path, result)
            raise
        finally:
            client.close()
    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(evaluate, row): row['key'] for row in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                log(f'QA {index}/{len(tasks)} {result["key"]}: {result.get("custom_judge_correct")}')
            except Exception as exc:
                errors.append(dict(key=futures[future], error_type=type(exc).__name__))
                log(f'QA saved recoverable failure: {type(exc).__name__}')
            dump_lines(ROOT / 'qa.jsonl', [json.loads(path.read_text(encoding='utf-8')) for path in sorted(result_dir.glob('*.json'))])
    atomic_json(ROOT / 'qa_status.json', dict(errors=errors, finished_at=utc_now()))
    if errors:
        raise RuntimeError('QA failures saved; rerun unchanged phase to resume')


def mechanism(rows, spec):
    torch.set_num_threads(4)
    train = [r for r in rows if r['question_id'] in spec['train_ids'] and eligible(r)]
    validation = [r for r in rows if r['question_id'] in spec['validation_ids'] and eligible(r)]
    encoder = ContentEncoder()
    data = {}
    start = time.perf_counter()
    for index, row in enumerate(train + validation, 1):
        h = history(row, True)
        x = encoder.encode(list(h.texts))
        costs = [BudgetItem(sid, date, text, vector, 0., i).nbytes
                 for i, (sid, date, text, vector) in enumerate(zip(h.session_ids, h.dates, h.texts, x))]
        data[h.question_id] = (h, torch.from_numpy(x), torch.from_numpy(deltas(h)), costs)
        if index % 20 == 0:
            log(f'Repaired vectors {index}/{len(train)+len(validation)}')
    queries = dict(zip([r['question_id'] for r in validation], encoder.encode([r['question'] for r in validation])))
    atomic_json(ROOT / 'encoding.json', dict(seconds=time.perf_counter()-start, histories=len(data), signature=encoder.signature, rss_mb=rss_mb()))
    encoder.close()
    def retained_score(row, logits, costs, model):
        h = data[row['question_id']][0]
        gold = set(row['answer_session_ids'])
        return np.mean([len({h.session_ids[i] for i in byte_retained_indices(logits, costs, budget, model.hidden * 4)} & gold) / len(gold) for budget in BUDGETS])
    for kind in KINDS:
        for seed in SEEDS:
            directory = ROOT / 'checkpoints' / f'{kind}-{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'complete.json').exists():
                continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = model_for(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            best, epochs = -1., []
            for epoch in range(1, 4):
                began = time.perf_counter()
                order = list(train); random.shuffle(order)
                losses = []
                model.train()
                for row in order:
                    h, x, dt, costs = data[row['question_id']]
                    labels = torch.tensor([float(sid in row['answer_session_ids']) for sid in h.session_ids])
                    positives = labels.sum().clamp_min(1)
                    weights = torch.where(labels > 0, .5 / positives, .5 / (len(labels)-positives).clamp_min(1))
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(x, dt), labels, reduction='none') * weights).sum()
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    values = [retained_score(row, model(data[row['question_id']][1], data[row['question_id']][2]).numpy(), data[row['question_id']][3], model) for row in validation]
                score = float(np.mean(values))
                checkpoint = dict(state_dict=model.state_dict(), kind=kind, seed=seed, epoch=epoch, source_signature=spec['source_signature'], train_ids=[r['question_id'] for r in train])
                torch.save(checkpoint, directory / 'last.pt')
                if score > best:
                    best = score; torch.save(checkpoint, directory / 'best.pt')
                epochs.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_retention=score, seconds=time.perf_counter()-began))
                atomic_json(directory / 'epochs.json', epochs)
                log(f'{kind} seed{seed} epoch{epoch}: retention={score:.4f}')
            atomic_json(directory / 'complete.json', dict(best_validation_retention=best, parameters=sum(p.numel() for p in model.parameters())))
    records = []
    conditions = [('full', None, None)] + [('fifo', None, b) for b in BUDGETS] + [(kind, seed, b) for kind in KINDS for seed in SEEDS for b in BUDGETS]
    for kind, seed, budget in conditions:
        model = None
        if seed is not None:
            model = model_for(kind)
            model.load_state_dict(torch.load(ROOT / 'checkpoints' / f'{kind}-{seed}' / 'best.pt', weights_only=True)['state_dict'])
            model.eval()
        for row in validation:
            h, x, dt, costs = data[row['question_id']]
            began = time.perf_counter()
            store = BudgetStore(budget, model)
            for sid, date, text, vector, delta in zip(h.session_ids, h.dates, h.texts, x.numpy(), dt.numpy()):
                store.insert(sid, date, text, vector, float(delta))
            write_seconds = time.perf_counter() - began
            began = time.perf_counter()
            if store.items:
                reader = Reader(store)
                selected, scores = reader.query(h.question, queries[h.question_id])
                kept = replace(h, session_ids=tuple(i.session_id for i in store.items), dates=tuple(i.date for i in store.items), texts=tuple(i.text for i in store.items))
                context, trace = pack_preserved_context(kept, selected)
                selected_ids = [store.items[i].session_id for i in selected]
            else:
                context, trace, selected_ids = '', [], []
            retained_ids = [i.session_id for i in store.items]
            gold = set(row['answer_session_ids'])
            evidence = [(date_days(date, 'minute'), sid) for sid, date, turns in zip(row['haystack_session_ids'], row['haystack_dates'], row['haystack_sessions']) if any(t.get('has_answer') is True for t in turns)]
            latest = max(evidence)[1] if evidence else None
            records.append(dict(method=kind, seed=seed, budget=budget, question_id=h.question_id, category=row['question_type'],
                                retention=len(set(retained_ids) & gold)/len(gold), recall=len(set(selected_ids) & gold)/len(gold),
                                sentence_coverage=coverage(row, context), payload_bytes=store.payload_bytes,
                                model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()) if model else 0,
                                latest_marked_session_retained=latest in retained_ids if latest else None,
                                selected_ids=selected_ids, retained_ids=retained_ids, context_trace=trace,
                                write_seconds=write_seconds, index_read_pack_seconds=time.perf_counter()-began,
                                history_span_days=date_days(h.dates[-1], 'minute')-date_days(h.dates[0], 'minute')))
        dump_lines(ROOT / 'mechanism.jsonl', records)
        log(f'Mechanism evaluated {kind}/{seed}/{budget}')
    atomic_json(ROOT / 'mechanism_status.json', dict(records=len(records), models=12, completed_at=utc_now(), rss_mb=rss_mb()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('contexts', 'qa', 'mechanism', 'all'), default='all')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ROOT / 'runner.lock'), timeout=0):
        rows, manifest = prepared(load_config('configs/minute_precision.yaml'))
        spec = freeze(rows, manifest)
        contexts(rows, spec)
        if args.phase in ('qa', 'all'):
            qa(spec)
        if args.phase in ('mechanism', 'all'):
            mechanism(rows, spec)


if __name__ == '__main__':
    main()
