"""Frozen CPU content-memory study; repeat with --phase all to resume safely."""
import argparse
import hashlib
import json
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np  # Must precede torch in this system-site-packages environment.
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import resolve, load_config, atomic_json
from src.data.content_benchmark import load_examples, choose_qa
from src.data.longmemeval import date_days
from src.data.download import file_sha256
from src.memory.content_store import ContentWriter, ContentStore, bounded_text
from src.models.content_encoder import ContentEncoder
from src.provenance import snapshot_source, runtime_info, rss_mb, utc_now
from src.retrieval.retriever import rank, pack_context
from src.evaluation.runner import messages_for, custom_judge
from src.evaluation.metrics import answer_metrics
from src.api.agnes import AgnesClient

ROOT = resolve('runs/content-memory-v1')
KINDS = ('static_content', 'liquid_content', 'gru_content')
SEEDS = (11, 23, 42)


def log(message):
    print(f'{utc_now()} {message}', flush=True)


def deltas(history):
    values = np.array([date_days(d, 'minute') for d in history.dates], dtype=np.float64)
    return np.clip(np.diff(values, prepend=values[0]) / 30, 0, 12).astype(np.float32)


def history_key(history):
    return hashlib.sha256(json.dumps([history.session_ids, history.dates, history.texts], ensure_ascii=False).encode()).hexdigest()


def vectors_for(encoder, history):
    return encoder.encode([bounded_text(t) for t in history.texts])


def retained_indices(scores, capacity):
    # The online store evicts oldest when tied, so later sequence wins ties.
    return sorted(sorted(range(len(scores)), key=lambda i: (float(scores[i]), i), reverse=True)[:capacity])


def retention(history, indices, gold):
    return len({history.session_ids[i] for i in indices} & set(gold)) / len(set(gold))


def freeze(train, validation, external, manifest, encoder):
    qa = choose_qa(validation, external)
    signature, archive = snapshot_source()
    current = dict(source_signature=signature, source_archive=archive, data_signature=manifest['signature'],
                   external_sha256=file_sha256(resolve('data/locomo_reserved/locomo10.json')),
                   encoder_signature=encoder.signature, qa_ids=qa,
                   train_ids=[e['history'].question_id for e in train],
                   validation_ids=[e['history'].question_id for e in validation],
                   external_ids=[e['history'].question_id for e in external],
                   protocol_sha256=file_sha256(ROOT / 'protocol.json'),
                   specification=dict(epochs=3, seeds=list(SEEDS), lr=0.003, weight_decay=0.01,
                                      loss='per-history balanced BCE; labels used only during training',
                                      dense_readout='positive cosine / maximum positive cosine',
                                      lexical_readout='TFIDF / maximum TFIDF; average with dense',
                                      window_clarification='First and last 254 content tokens plus CLS/SEP; windows can be disjoint. This corrects the earlier overlapping wording before scoring.',
                                      storage='bounded UTF8 text, vectors, IDs, dates, scalar priority, persistent hidden state; excludes shared weights, Python overhead and offline artifact caches',
                                      qa='fixed seed42, 8 slots except unbounded full baseline; category5 standardized abstention reference; same-provider custom judge'))
    path = ROOT / 'execution_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        for key, value in current.items():
            if previous[key] != value:
                raise RuntimeError(f'Frozen experiment changed: {key}; use a new versioned experiment, do not overwrite results')
    else:
        atomic_json(path, dict(**current, environment=runtime_info(), frozen_at=utc_now()))
        atomic_json(ROOT / 'qa_sample.json', dict(question_ids=qa, selection='SHA256 question ID, before scores', frozen_at=utc_now()))
    return signature


def prepare(encoder, examples):
    histories = {}
    for e in examples:
        h = e['history']
        histories.setdefault(history_key(h), h)
    start = time.perf_counter()
    for index, h in enumerate(histories.values(), 1):
        vectors_for(encoder, h)
        if index % 10 == 0 or index == len(histories):
            log(f'content vectors: {index}/{len(histories)} histories, {time.perf_counter()-start:.1f}s')
    encoder.encode([e['history'].question for e in examples])
    atomic_json(ROOT / 'encoding.json', dict(histories=len(histories), questions=len(examples), seconds=time.perf_counter()-start,
                                           encoder=encoder.signature, process_rss_mb=rss_mb()))


def audit(train, validation, external):
    # Speaker prefixes omitted; exact text overlap is only one leakage diagnostic.
    def texts(examples):
        return {hashlib.sha256(' '.join(line.split(': ', 1)[-1].lower().split()).encode()).hexdigest()
                for e in examples for text in e['history'].texts for line in text.splitlines() if len(line) >= 40}
    a, b, c = texts(train), texts(validation), texts(external)
    exclusions = defaultdict(int)
    for e in external:
        exclusions[str(e['exclusion'])] += 1
    atomic_json(ROOT / 'data_audit.json', dict(train_long_line_hashes=len(a), external_long_line_hashes=len(c),
                exact_train_external_overlap=len(a & c), exact_validation_external_overlap=len(b & c),
                external_qa=len(external), external_groups=len({e['group'] for e in external}),
                external_retrieval_status=dict(exclusions),
                exclusions=[dict(question_id=e['history'].question_id, reason=e['exclusion'], unmapped=e['unmapped_evidence']) for e in external if e['exclusion']],
                limitations='Line-level exact matching ignores short utterances, does not prove semantic independence; LoCoMo is now inspected.'))


def train_models(encoder, train, validation, signature):
    eligible = [e for e in train if e['exclusion'] is None]
    dev = [e for e in validation if e['exclusion'] is None]
    data = {}
    for e in eligible + dev:
        h = e['history']
        data[h.question_id] = (torch.from_numpy(vectors_for(encoder, h)), torch.from_numpy(deltas(h)))
    for kind in KINDS:
        for seed in SEEDS:
            directory = ROOT / 'checkpoints' / f'{kind}-{seed}'
            if (directory / 'complete.json').exists():
                log(f'training already complete: {kind} seed{seed}')
                continue
            directory.mkdir(parents=True, exist_ok=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = ContentWriter(kind)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            best, records = -1, []
            for epoch in range(1, 4):
                start = time.perf_counter()
                model.train()
                order = list(eligible); random.shuffle(order)
                losses = []
                for e in order:
                    h = e['history']
                    x, dt = data[h.question_id]
                    labels = torch.tensor([float(s in e['gold']) for s in h.session_ids])
                    positive = labels.sum().clamp_min(1)
                    weights = torch.where(labels > 0, 0.5 / positive, 0.5 / (len(labels)-positive).clamp_min(1))
                    logits = model(x, dt)
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).sum()
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step(); losses.append(float(loss.detach()))
                model.eval(); values = []
                with torch.no_grad():
                    for e in dev:
                        h = e['history']; scores = model(*data[h.question_id]).numpy()
                        values.append(np.mean([retention(h, retained_indices(scores, cap), e['gold']) for cap in (8, 32)]))
                score = float(np.mean(values))
                record = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_retention=score,
                              seconds=time.perf_counter()-start, process_rss_mb=rss_mb())
                records.append(record)
                saved = dict(state_dict=model.state_dict(), kind=kind, seed=seed, epoch=epoch,
                             validation_retention=score, source_signature=signature,
                             train_ids=[e['history'].question_id for e in eligible])
                torch.save(saved, directory / 'last.pt')
                if score > best:
                    best = score; torch.save(saved, directory / 'best.pt')
                atomic_json(directory / 'epochs.json', records)
                log(f'train {kind} seed{seed} epoch{epoch}: loss={record["train_loss"]:.4f}, retention={score:.4f}')
            atomic_json(directory / 'complete.json', dict(best_validation_retention=best, parameters=sum(p.numel() for p in model.parameters()), source_signature=signature))


class Reader:
    def __init__(self, store):
        self.store = store
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            self.matrix = self.vectorizer.fit_transform([i.text for i in store.items])
        except ValueError:
            self.matrix = None
        self.vectors = np.stack([i.vector for i in store.items])

    def query(self, question, query_vector):
        lexical = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel() if self.matrix is not None else np.zeros(len(self.store.items))
        dense = np.maximum(0, self.vectors @ query_vector)
        scores = 0.5 * lexical / max(float(lexical.max()), 1e-8) + 0.5 * dense / max(float(dense.max()), 1e-8)
        return rank(scores, 5), scores


def conditions():
    yield 'full_hybrid', None, None
    for cap in (8, 32):
        yield 'fifo', None, cap
        for kind in KINDS:
            for seed in SEEDS:
                yield kind, seed, cap


def connection():
    db = sqlite3.connect(ROOT / 'results.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS retrieval (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS qa (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    return db


def evaluate(encoder, examples):
    db = connection()
    existing = {r[0] for r in db.execute('SELECT key FROM retrieval')}
    qa_ids = set(json.loads((ROOT / 'qa_sample.json').read_text(encoding='utf-8'))['question_ids'])
    queries = dict(zip([e['history'].question_id for e in examples], encoder.encode([e['history'].question for e in examples])))
    for kind, seed, cap in conditions():
        label = f'{kind}:{seed}:{cap}'
        model = None
        if seed is not None:
            model = ContentWriter(kind)
            saved = torch.load(ROOT / 'checkpoints' / f'{kind}-{seed}' / 'best.pt', weights_only=True)
            model.load_state_dict(saved['state_dict']); model.eval()
        stores = {}
        for e in examples:
            h = e['history']; key = label + ':' + h.question_id
            if key in existing:
                continue
            hk = history_key(h)
            if hk not in stores:
                x = vectors_for(encoder, h)
                store = ContentStore(cap, model)
                started = time.perf_counter()
                for sid, date, text, vector, dt in zip(h.session_ids, h.dates, h.texts, x, deltas(h)):
                    store.insert(sid, date, text, vector, float(dt))
                write_seconds = time.perf_counter()-started
                started = time.perf_counter(); reader = Reader(store)
                stores[hk] = (reader, write_seconds, time.perf_counter()-started)
            reader, write_seconds, index_seconds = stores[hk]
            store = reader.store
            started = time.perf_counter()
            selected, scores = reader.query(h.question, queries[h.question_id])
            read_seconds = time.perf_counter()-started
            ids = [store.items[i].session_id for i in selected]
            kept = [i.session_id for i in store.items]
            gold = set(e['gold'])
            span = date_days(h.dates[-1], 'minute') - date_days(h.dates[0], 'minute')
            valid = e['exclusion'] is None
            row = dict(key=key, method=kind, seed=seed, capacity=cap, question_id=h.question_id,
                       dataset=e['dataset'], group=e['group'], category=e['category'], exclusion=e['exclusion'],
                       history_span_days=span, span_bucket='<=30' if span <= 30 else '(30,180]' if span <= 180 else '>180',
                       recall=len(set(ids) & gold)/len(gold) if valid else None,
                       all_evidence=float(gold <= set(ids)) if valid else None,
                       retention=len(set(kept) & gold)/len(gold) if valid else None,
                       retained_ids=kept, selected_ids=ids, selected_scores=[float(scores[i]) for i in selected],
                       gold_ids=e['gold'], payload_bytes=store.payload_bytes, stored_sessions=len(kept), history_sessions=len(h.texts),
                       write_seconds=write_seconds, index_seconds=index_seconds, read_seconds=read_seconds,
                       latency_note='history write/index timings reused across QA on the same history; excludes cached embedding precomputation',
                       model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()) if model else 0,
                       process_rss_mb=rss_mb(), question_date_assumed=e['question_date_assumed'])
            if h.question_id in qa_ids and (kind == 'full_hybrid' or cap == 8 and seed in (None, 42)):
                retained = replace(h, session_ids=tuple(kept), dates=tuple(i.date for i in store.items), texts=tuple(i.text for i in store.items))
                context, trace = pack_context(retained, selected, budget=16000, chunk_chars=1600)
                row.update(context=context, context_trace=trace, question=h.question, question_date=h.question_date, answer=e['answer'])
            db.execute('INSERT INTO retrieval VALUES (?,?)', (key, json.dumps(row, ensure_ascii=False)))
            db.commit()
        log(f'offline evaluated {label}: {len(examples)} questions')
    db.close()


def qa(config):
    db = connection(); client = AgnesClient(config['api'])
    rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM retrieval ORDER BY key')]
    selected = [r for r in rows if 'context' in r]
    for index, row in enumerate(selected, 1):
        prior = db.execute('SELECT payload FROM qa WHERE key=?', (row['key'],)).fetchone()
        result = json.loads(prior[0]) if prior else dict(row)
        if 'judge_response' in result:
            continue
        h = type('PromptHistory', (), dict(question_date=row['question_date'], question=row['question']))()
        namespace = 'content-memory-v1:' + file_sha256(ROOT / 'execution_manifest.json')
        started = time.perf_counter()
        try:
            if 'generation' not in result:
                messages = messages_for(h, row['context'])
                result['messages'] = messages
                result['generation'] = client.complete(messages, namespace=namespace)
                result['hypothesis'] = result['generation']['content']
                result.update(answer_metrics(result['hypothesis'], row['answer']))
                db.execute('INSERT OR REPLACE INTO qa VALUES (?,?)', (row['key'], json.dumps(result, ensure_ascii=False))); db.commit()
            grade, response = custom_judge(client, row, result['hypothesis'], namespace)
            result.update(custom_judge_correct=grade, judge_response=response, qa_seconds=time.perf_counter()-started)
            result.pop('error_type', None)
            db.execute('INSERT OR REPLACE INTO qa VALUES (?,?)', (row['key'], json.dumps(result, ensure_ascii=False))); db.commit()
            log(f'Agnes QA {index}/{len(selected)} {row["method"]} {row["question_id"]}: {grade}')
        except Exception as exc:
            result['error_type'] = type(exc).__name__
            db.execute('INSERT OR REPLACE INTO qa VALUES (?,?)', (row['key'], json.dumps(result, ensure_ascii=False))); db.commit()
            log(f'Agnes stopped on {type(exc).__name__}; records safely saved, rerun --phase qa to resume')
            raise
    client.close(); db.close()


def export():
    db = connection()
    summaries = []
    for table in ('retrieval', 'qa'):
        rows = [json.loads(r[0]) for r in db.execute(f'SELECT payload FROM {table} ORDER BY key')]
        with (ROOT / f'{table}.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        groups = defaultdict(list)
        for row in rows:
            for subgroup, value in [('all', 'all'), ('category', row['category']), ('span', row['span_bucket'])]:
                groups[(row['dataset'], row['method'], row['seed'], row['capacity'], subgroup, value)].append(row)
        metrics = ('recall', 'all_evidence', 'retention', 'payload_bytes', 'write_seconds', 'read_seconds') if table == 'retrieval' else ('custom_judge_correct', 'exact_match', 'token_f1')
        for key, members in groups.items():
            item = dict(zip(('dataset', 'method', 'seed', 'capacity', 'subgroup', 'value'), key), table=table, n=len(members))
            for metric in metrics:
                values = [r[metric] for r in members if r.get(metric) is not None]
                item[metric] = float(np.mean(values)) if values else None
                item[metric + '_n'] = len(values)
            summaries.append(item)
    atomic_json(ROOT / 'summary.json', summaries)
    db.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('all', 'prepare', 'train', 'evaluate', 'qa', 'export'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config('configs/minute_precision.yaml')
    if args.phase == 'export':
        export(); return
    encoder = ContentEncoder()
    train, validation, external, manifest = load_examples(config)
    signature = freeze(train, validation, external, manifest, encoder)
    if args.phase in ('all', 'prepare'):
        audit(train, validation, external); prepare(encoder, train + validation + external)
    if args.phase in ('all', 'train'):
        train_models(encoder, train, validation, signature)
    if args.phase in ('all', 'evaluate'):
        evaluate(encoder, validation + external); export()
    if args.phase in ('all', 'qa'):
        try:
            qa(config)
        finally:
            export()
    encoder.close()


if __name__ == '__main__':
    main()
