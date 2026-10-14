"""Public toy correctness check, not a learned-dynamics benefit experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import torch
from src.config import resolve, atomic_json
from src.provenance import snapshot_source, runtime_info, utc_now
from src.data.babi_events import stories, read_key
from src.memory.version_ledger import VersionLedger, Evidence


def main():
    root=resolve('runs/babi-ledger-v2');root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():
        print('Completed run preserved');return
    source,archive=snapshot_source()
    atomic_json(root/'protocol.json',dict(source_signature=source,source_archive=archive,
        created_at=utc_now(),environment=runtime_info(),
        dataset=json.loads(resolve('data/babi/provenance.json').read_text()),
        policies=['fifo','frozen_rule','rule'],budgets=[512,1024,2048,None],
        scope='Official en 1k qa1 train sanity then test. All stories, causal question prefixes. No learning, no API. NOT liquid superiority evidence.',
        time='Statement index is logical step time, not real elapsed days.',
        budget='UTF8 evidence, metadata, watermark and equal 96 latent bytes/version; Python runtime/shared code excluded.'))
    records=[];summary=[]
    with torch.no_grad():
        for split in ('train','test'):
            data=stories(resolve(f'data/babi/qa1_single-supporting-fact_{split}.txt'),split)
            for policy in ('fifo','frozen_rule','rule'):
                for budget in (512,1024,2048,None):
                    start=time.perf_counter();condition=[]
                    for index,story in enumerate(data):
                        ledger=VersionLedger(budget,policy=policy)
                        for item in story:
                            if isinstance(item,Evidence):
                                ledger.write(item)
                                assert budget is None or ledger.payload_bytes<=budget
                            else:
                                evidence=ledger.current(read_key(item['question']))
                                prediction=evidence.value if evidence else None
                                row=dict(split=split,story=index,line=item['line'],policy=policy,budget=budget,
                                    question=item['question'],answer=item['answer'],prediction=prediction,
                                    exact=prediction==item['answer'],abstained=prediction is None,
                                    wrong_return=prediction is not None and prediction!=item['answer'],
                                    payload_bytes=ledger.payload_bytes,
                                    evidence_id=evidence.identity if evidence else None,
                                    evidence_quote=evidence.quote if evidence else None)
                                condition.append(row)
                    assert not any(r['wrong_return'] for r in condition),'Stale/wrong current fact'
                    if budget is None: assert all(r['exact'] for r in condition)
                    stats=dict(split=split,policy=policy,budget=budget,questions=len(condition),
                        exact_match=sum(r['exact'] for r in condition)/len(condition),
                        abstentions=sum(r['abstained'] for r in condition),
                        wrong_returns=sum(r['wrong_return'] for r in condition),
                        seconds=time.perf_counter()-start,
                        max_payload_bytes=max(r['payload_bytes'] for r in condition))
                    records.extend(condition);summary.append(stats);print(stats,flush=True)
    (root/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    atomic_json(root/'summary.json',dict(conditions=summary,api_calls=0,
        conclusion='Version correctness check only. No trained liquid model evaluated.',finished_at=utc_now()))


if __name__=='__main__':main()
