"""Predeclared development-only mechanism screen; never evaluates the test split."""
import argparse
import copy
import csv
import json
import logging
import time
from pathlib import Path
import numpy as np
from src.config import atomic_json, load_config, resolve
from src.data.longmemeval import prepared
from src.provenance import snapshot_source, utc_now
from src.training.train import train
from src.evaluation.runner import evaluate

VARIANTS = {
    'liquid': {'kind':'lcm_v2'},
    'reset_state': {'kind':'lcm_v2', 'reset_history':True},
    'fixed_time': {'kind':'lcm_v2', 'fixed_time':True},
    'no_consolidation': {'kind':'lcm_v2', 'consolidation':False},
    'gru_matched': {'kind':'gru_v2'},
    'linear': {'kind':'ranker'},
}


def summarize(root, records, seeds, variants=VARIANTS):
    summary = {'completed_runs':len(records), 'planned_runs':len(seeds)*len(variants),
               'partition':'validation development only; checkpoint selection uses this partition',
               'variants':{}, 'paired_seed_differences':{}}
    for name in variants:
        items = [r for r in records if r['variant']==name]
        if not items:
            continue
        values = np.array([r['recall_at_5'] for r in items])
        summary['variants'][name] = {'n':len(items), 'mean':float(values.mean()),
            'seed_std':float(values.std(ddof=1)) if len(items)>1 else None,
            'parameters':items[0]['parameters'],
            'all_evidence_mean':float(np.mean([r['all_evidence_at_5'] for r in items]))}
    liquid = {r['seed']:r for r in records if r['variant']=='liquid'}
    for name in variants:
        if name=='liquid':
            continue
        control = {r['seed']:r for r in records if r['variant']==name}
        common = sorted(set(liquid)&set(control))
        if common:
            differences = [liquid[s]['recall_at_5']-control[s]['recall_at_5'] for s in common]
            summary['paired_seed_differences'][name] = {'seeds':common, 'differences':differences,
                'mean':float(np.mean(differences)), 'wins':sum(d>0 for d in differences)}
    complete = len(records)==summary['planned_runs']
    if complete:
        controls = [summary['paired_seed_differences'][name] for name in ('reset_state','gru_matched')]
        summary['screen_passed'] = all(r['mean']>=.01 and r['wins']>=4 for r in controls)
        summary['conclusion'] = 'Candidate for independent confirmation; not proof' if summary['screen_passed'] else 'Current architecture does not pass the predeclared mechanism screen'
    atomic_json(root/'summary.json',summary)
    if records:
        with (root/'runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/optimized.yaml')
    parser.add_argument('--output',default='runs/research-round-01')
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    args=parser.parse_args()
    variants={name:VARIANTS[name] for name in args.variants}
    if not {'liquid','reset_state','gru_matched'} <= set(variants):
        raise ValueError('The predeclared screen requires liquid, reset_state and gru_matched')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%H:%M:%S')
    config=load_config(args.config)
    root=resolve(args.output)
    root.mkdir(parents=True,exist_ok=True)
    seeds=[11,23,42,67,89]
    epochs=6
    _, manifest=prepared(config)
    source_hash,source_archive=snapshot_source()
    protocol={'seeds':seeds,'epochs_per_run':epochs,'variants':variants,'data_signature':manifest['signature'],
        'config':config,'source_sha256':source_hash,'source_archive':source_archive,
        'primary_metric':'macro unique-session Recall@5 on development validation',
        'secondary_metrics':['all-evidence@5','by_type','runtime','parameters'],
        'screen_rule':'Liquid mean gain >= 1 percentage point and wins >= 4/5 seeds against BOTH reset-state and parameter-matched GRU',
        'checkpoint_rule':'Best validation Recall@5 in six epochs; earliest tie. This is development screening, not an independent estimate.',
        'test_policy':'No test evaluation, no test-dependent tuning, report every variant/seed including negative findings',
        'downstream_rule':'If screen fails, analyze representation/task mismatch before proposing a new frozen round; do not extend seeds until positive',
        'api_calls':0}
    protocol_path=root/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text(encoding='utf-8'))
        if {k:v for k,v in previous.items() if k!='created_at_utc'}!=protocol:
            raise ValueError('Protocol/source changed; use a new round directory')
    else:
        atomic_json(protocol_path,{**protocol,'created_at_utc':utc_now()})
    records=[]
    for name,options in variants.items():
        for seed in seeds:
            run_root=root/f'{name}-seed{seed}'
            completion=run_root/'result.json'
            if completion.exists():
                records.append(json.loads(completion.read_text(encoding='utf-8')))
                continue
            conf=copy.deepcopy(config)
            conf['training'].update(seed=seed,epochs=epochs)
            started=time.monotonic()
            checkpoint_root=run_root/'checkpoints'
            try:
                result=train(conf,epochs=epochs,output=checkpoint_root,
                    resume=(checkpoint_root/'last.pt').exists(),**options)
                evaluated=evaluate(conf,options['kind'],'val',checkpoint=result['checkpoint'],
                    retrieval_only=True,output=run_root/'validation')
                environment=json.loads((checkpoint_root/'environment.json').read_text(encoding='utf-8'))
                metrics=evaluated['metrics']
                record={'variant':name,'seed':seed,'recall_at_5':metrics['session_recall_at_k'],
                    'all_evidence_at_5':metrics['all_evidence_at_k'],
                    'parameters':environment['parameter_count'],'wall_seconds':time.monotonic()-started,
                    'checkpoint':result['checkpoint'],'evaluation':evaluated['output']}
                atomic_json(completion,record)
                records.append(record)
                summary=summarize(root,records,seeds,variants)
                print(f"ROUND_PROGRESS {len(records)}/{len(seeds)*len(variants)} {name} seed={seed} recall={record['recall_at_5']:.6f}",flush=True)
            except Exception as exc:
                atomic_json(run_root/'failure.json',{'type':type(exc).__name__,'message':str(exc),'at':utc_now()})
                raise
    print(json.dumps(summarize(root,records,seeds,variants),indent=2))


if __name__=='__main__':
    main()
