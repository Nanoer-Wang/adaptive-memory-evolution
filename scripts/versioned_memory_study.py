"""CPU development pilot for explicit versions and per-version dynamics.

Public LongMemEval only. Extraction is a deliberately narrow closed-schema
adapter; missing coverage is reported, not hidden behind conditional accuracy.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import asdict
import numpy as np
import torch
from filelock import FileLock
from src.config import resolve,load_config,atomic_json
from src.data.longmemeval import prepared,date_days,session_text
from src.data.fact_evidence import extract_evidence,query_key,normalize_value
from src.memory.evidence_dynamics import EvidenceDynamics,KINDS
from src.memory.version_ledger import VersionLedger
from src.provenance import snapshot_source,utc_now,runtime_info,rss_mb

ROOT=resolve('runs/version-ledger-v1')
SEEDS=(11,23,42)
BUDGETS=(4096,16384)


def log(value):print(f'{utc_now()} {value}',flush=True)


def dump(name,rows):
    path=ROOT/name; temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');temporary.replace(path)


def example(row):
    events=[]; positive=set(); gold_turns=set(); source_map={}
    ordered=sorted(zip(row['haystack_session_ids'],row['haystack_dates'],row['haystack_sessions']),key=lambda x:date_days(x[1],'minute'))
    for sid,date,turns in ordered:
        fingerprint=hashlib.sha256((str(sid)+'\0'+date+'\0'+session_text(turns)).encode()).hexdigest()[:24]
        source='s_'+fingerprint
        source_map[source]=dict(original_id=sid,date=date)
        extracted=extract_evidence(turns,source,date_days(date,'minute'))
        events.extend(extracted)
        # Annotations used ONLY after extraction, for teacher labels/metrics.
        if not str(row['question_id']).endswith('_abs'):
            for index,turn in enumerate(turns):
                if turn.get('has_answer') is True:gold_turns.add((source,index))
            positive.update(e.identity for e in extracted if (source,e.turn) in gold_turns)
    return dict(row=row,events=events,positive=positive,gold_turns=gold_turns,source_map=source_map)


def scalar_reference(row,key):
    if key is None or str(row['question_id']).endswith('_abs'):return None
    value=str(row['answer']).strip()
    if key[1] in ('followers','age'):
        if not re.fullmatch(r'\d[\d,]*',value):return None
        return normalize_value(key[1],value)
    if key[1]=='gym_time':
        if not re.fullmatch(r'\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?',value,re.I):return None
        return normalize_value('gym_time',value)
    return value.rstrip('.').casefold()


def replay(e,budget=None,model=None,policy='rule'):
    ledger=VersionLedger(budget,model,'learned' if model is not None else policy)
    for event in e['events']:ledger.write(event)
    return ledger


def freeze(manifest):
    source,archive=snapshot_source()
    spec=dict(source_signature=source,source_archive=archive,data_sha256=manifest['signature']['sha256'],
              train_ids=manifest['splits']['train'],validation_ids=manifest['splits']['val'],
              kinds=list(KINDS),seeds=list(SEEDS),budgets=list(BUDGETS),
              design=dict(training='Train split only: up to64 histories with extracted positive evidence and32 all-negative histories, sorted by SHA256 qid. All extracted events, no question input. Balanced fact BCE, AdamW lr0.002 wd0.01, clip1, fixed5epochs; no validation checkpoint selection.',
              teacher='Gold source-turn evidence labels only in training loss. Broad evidence annotations are weak fact-utility labels; they do not label individual claim truth.',
              evaluation='All93 development questions including2 unanswerable. Report extraction ceiling,91 answerable-turn coverage, conditional retention, source grounding, stale-return audit, and narrow read-time routed scalar exact match. NOT full QA accuracy.',
              time='Observed-at knowledge order only. Current/past/planned/tentative/retracted assertions separated. Real-world validity intervals and arbitrary temporal question parsing unsupported.',
              budget='4kiB/16kiB effective payload; UTF8 evidence+provenance+version metadata+key watermark+96 state bytes/version reserved equally across controls. Shared weights and Python/allocator/index/cache overhead excluded.',
              primary='Per-question annotated-turn retention. Liquid vs static/GRU/no_time/fixed_rate; all3seeds. Development only: no novelty, generalization or accuracy claim from unit tests.',
              related_work={'A-MEM':'https://arxiv.org/abs/2502.12110','Mem0':'https://arxiv.org/abs/2504.19413','CfC':'https://www.nature.com/articles/s42256-022-00556-7'},
              distinction='Exact solution of piecewise-linear per-version evidence ODE with event impulses, not CfC/LTC. Truth/known-current status is a separate explicit ledger, never selected by learned score.'))
    path=ROOT/'execution_manifest.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in spec.items()):raise RuntimeError('Frozen study differs; create a new version')
    else:atomic_json(path,dict(**spec,created_at=utc_now(),environment=runtime_info()))
    return spec


def prepare(rows,spec):
    train=[example(r) for r in rows if r['question_id'] in spec['train_ids'] and not str(r['question_id']).endswith('_abs')]
    validation=[example(r) for r in rows if r['question_id'] in spec['validation_ids']]
    order=lambda e:hashlib.sha256(str(e['row']['question_id']).encode()).hexdigest()
    positives=sorted([e for e in train if e['positive']],key=order)[:64]
    negatives=sorted([e for e in train if e['events'] and not e['positive']],key=order)[:32]
    chosen=positives+negatives
    audit={}
    for split,examples in [('train',train),('validation',validation)]:
        answerable=[e for e in examples if not str(e['row']['question_id']).endswith('_abs')]
        annotated=[e for e in answerable if e['gold_turns']]
        extracted=[e for e in examples for e in e['events']]
        audit[split]=dict(histories=len(examples),events=len(extracted),positive_histories=sum(bool(e['positive']) for e in examples),
                         unique_source_events=len({e.identity for e in extracted}),
                         gold_turn_macro_extraction_coverage=float(np.mean([len({(v.source_id,v.turn) for v in e['events']} & e['gold_turns'])/len(e['gold_turns']) for e in annotated])),
                         routed_questions=sum(query_key(e['row']['question']) is not None for e in examples),
                         predicates=dict((p,sum(v.predicate==p for v in extracted)) for p in sorted({v.predicate for v in extracted})))
    atomic_json(ROOT/'extraction_audit.json',dict(**audit,training_positive_histories=len(positives),training_negative_histories=len(negatives),
        selected_train_ids=[e['row']['question_id'] for e in chosen],note='Exact quoted-span grounding does not establish extractor semantic precision. Conservative user-only closed schema omits assistant and unsupported facts.'))
    # Persist only extracted payloads and provenance, not another full raw dataset.
    dump('extracted_events.jsonl',[dict(question_id=e['row']['question_id'],split=split,events=[asdict(v) for v in e['events']],positive_ids=sorted(e['positive']),source_map=e['source_map']) for split,examples in [('train',chosen),('validation',validation)] for e in examples])
    return chosen,validation


def train_models(train,spec):
    if not any(e['positive'] for e in train):
        raise RuntimeError('No extracted training positives; audit saved, training refused')
    for kind in KINDS:
        for seed in SEEDS:
            directory=ROOT/'checkpoints'/f'{kind}-{seed}';directory.mkdir(parents=True,exist_ok=True)
            if (directory/'complete.json').exists():continue
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            model=EvidenceDynamics(kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            records=[]
            for epoch in range(1,6):
                start=time.perf_counter();order=list(train);random.shuffle(order);losses=[];model.train()
                for e in order:
                    ledger=replay(e,model=model)
                    scores=ledger.scores();ids=list(scores)
                    labels=torch.tensor([float(identity in e['positive']) for identity in ids])
                    logits=torch.stack([scores[identity] for identity in ids])
                    npos=labels.sum();nneg=len(labels)-npos
                    weights=torch.where(labels>0,.5/npos.clamp_min(1),(.5 if npos>0 else 1.)/nneg.clamp_min(1))
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).sum()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
                records.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.perf_counter()-start,rss_mb=rss_mb()))
                atomic_json(directory/'epochs.json',records)
                torch.save(dict(state_dict=model.state_dict(),kind=kind,seed=seed,epoch=epoch,source_signature=spec['source_signature'],train_ids=[e['row']['question_id'] for e in train]),directory/'last.pt')
                log(f'{kind}/{seed} epoch{epoch}: loss={records[-1]["train_loss"]:.4f}')
            atomic_json(directory/'complete.json',dict(epoch=5,allocated_parameters=sum(p.numel() for p in model.parameters()),source_signature=spec['source_signature']))


def evaluate(validation):
    conditions=[('full',None,None)]+[(p,None,b) for p in ('fifo','frozen_rule','rule') for b in BUDGETS]+[(k,s,b) for k in KINDS for s in SEEDS for b in BUDGETS]
    oracle={e['row']['question_id']:replay(e) for e in validation}
    results=[]
    for kind,seed,budget in conditions:
        model=None
        if seed is not None:
            model=EvidenceDynamics(kind);model.load_state_dict(torch.load(ROOT/'checkpoints'/f'{kind}-{seed}'/'last.pt',weights_only=True)['state_dict']);model.eval()
        for e in validation:
            row=e['row'];qid=row['question_id'];start=time.perf_counter()
            with torch.no_grad():ledger=replay(e,budget,model,kind if kind in ('fifo','frozen_rule','rule') else 'rule')
            elapsed=time.perf_counter()-start
            retained_turns={(v.evidence.source_id,v.evidence.turn) for v in ledger.items.values()}
            key=query_key(row['question']);expected=scalar_reference(row,key)
            prediction=ledger.current(key) if key is not None else None
            unbounded=oracle[qid].current(key) if key is not None else None
            stale=int(prediction is not None and (unbounded is None or prediction.value.casefold()!=unbounded.value.casefold()))
            result=dict(method=kind,seed=seed,budget=budget,question_id=qid,category=row['question_type'],unanswerable=str(qid).endswith('_abs'),
                extracted_events=len(e['events']),extracted_positive_events=len(e['positive']),gold_turns=len(e['gold_turns']),
                annotated_turn_retention=len(retained_turns&e['gold_turns'])/len(e['gold_turns']) if e['gold_turns'] else None,
                conditional_extracted_evidence_retention=len(set(ledger.items)&e['positive'])/len(e['positive']) if e['positive'] else None,
                routed=key is not None,scalar_reference_eligible=expected is not None,
                scalar_exact=float(prediction is not None and prediction.value.rstrip('.').casefold()==expected.casefold()) if expected is not None else None,
                predicted_value=prediction.value if prediction else None,reference=str(row['answer']) if key is not None else None,
                stale_returns=stale,payload_bytes=ledger.payload_bytes,retained_versions=len(ledger.items),retained_keys=len(ledger.keys),
                write_seconds=elapsed,parameters=sum(p.numel() for p in model.parameters()) if model is not None else 0,
                trace=[dict(identity=v.evidence.identity,key=v.evidence.key,value=v.evidence.value,status=v.status,observed_at=v.evidence.observed_at,known_until=v.known_until,utility=float(ledger.score(v).detach()),source_id=v.evidence.source_id,turn=v.evidence.turn) for v in ledger.items.values()])
            assert budget is None or ledger.payload_bytes<=budget
            assert stale==0,'Version invariant violated'
            results.append(result)
        dump('evaluation.jsonl',results);log(f'Evaluated {kind}/{seed}/{budget}')
    atomic_json(ROOT/'execution_status.json',dict(completed=True,records=len(results),trained_models=15,api_calls=0,finished_at=utc_now()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','train','evaluate','all'),default='all');args=parser.parse_args()
    torch.set_num_threads(4);ROOT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(ROOT/'runner.lock'),timeout=0):
        if args.phase=='all' and (ROOT/'execution_status.json').exists():
            status=json.loads((ROOT/'execution_status.json').read_text(encoding='utf-8'))
            if status.get('completed'):
                log('Completed study preserved; read report or choose an explicit phase to recompute.');return
        rows,manifest=prepared(load_config('configs/minute_precision.yaml'));spec=freeze(manifest)
        train,validation=prepare(rows,spec)
        if args.phase in ('all','train'):train_models(train,spec)
        if args.phase in ('all','evaluate'):evaluate(validation)


if __name__=='__main__':main()
