"""Validation-only sensitivity and same-weight interventions; no model selection."""
import argparse
import numpy as np
import torch
from src.config import load_config, atomic_json, resolve
from src.data.longmemeval import prepared, public_history, features_for
from src.retrieval.lexical_features import cached_features
from src.models.anchored_memory import AnchoredMemory
from src.retrieval.retriever import rank
from src.evaluation.metrics import retrieval_metrics
from src.data.download import file_sha256
from src.provenance import snapshot_source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/minute_precision.yaml')
    parser.add_argument('--checkpoint',default='runs/research-round-02-timefix/liquid-seed42/checkpoints/best.pt')
    parser.add_argument('--output',default='runs/research-analysis/memory_diagnostics.json')
    args=parser.parse_args()
    config=load_config(args.config)
    torch.set_num_threads(config['training']['threads'])
    rows,manifest=prepared(config)
    byid={str(r['question_id']):r for r in rows}
    saved=torch.load(resolve(args.checkpoint),map_location='cpu',weights_only=True)
    if saved['data_signature']!=manifest['signature'] or saved['model_settings']['kind']!='lcm_v2':
        raise ValueError('Expected an lcm_v2 checkpoint for this data configuration')
    if set(saved['train_ids']) & set(manifest['splits']['val']):
        raise ValueError('Training/validation overlap')
    model=AnchoredMemory(**saved['model_settings'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    records=[]
    gaps=[]
    for qid in sorted(manifest['splits']['val']):
        row=byid[qid]
        h=public_history(row,config['model'].get('timestamp_precision','day'))
        x=torch.tensor(cached_features(config,h),requires_grad=True)
        deltas=torch.tensor(features_for(config,qid)[2])
        gaps.extend(deltas[1:].tolist())
        result={'question_id':qid,'question_type':row['question_type']}
        selections={}
        with torch.no_grad():
            for name,reset,fixed in [('native',False,False),('reset_state',True,False),('fixed_time',False,True)]:
                model.reset_history=reset
                model.cell.fixed_time=fixed
                scores=model(x,deltas=deltas)[0].numpy()
                selected=[h.session_ids[i] for i in rank(scores,5)]
                selections[name]=set(selected)
                result[name]=retrieval_metrics(selected,row['answer_session_ids'],qid.endswith('_abs'))['session_recall_at_k']
        result['reset_changes_top5']=selections['native']!=selections['reset_state']
        result['fixed_time_changes_top5']=selections['native']!=selections['fixed_time']
        model.reset_history=False
        model.cell.fixed_time=False
        encoded=model.encoder(x)
        state=torch.zeros_like(encoded[0])
        for experience,delta in zip(encoded,deltas):
            state=model.cell(state,experience,delta=delta)
        gradient=torch.autograd.grad(model.readout(state).sum(),x)[0].norm(dim=1).detach().numpy()
        total=float(gradient.sum())
        result.update(sessions=len(x),gradient_norm_sum=total,
            last_5_gradient_mass=float(gradient[-5:].sum()/total) if total else None,
            older_than_20_gradient_mass=float(gradient[:-20].sum()/total) if total and len(x)>20 else None)
        records.append(result)
    valid=[r for r in records if r['native'] is not None]
    summary={'scope':'validation only; seed42 diagnostic, same-weight interventions and local derivatives; NOT proof of causal retention or capacity',
        'checkpoint_sha256':file_sha256(resolve(args.checkpoint)),'source_archive':snapshot_source()[1],
        'means':{k:float(np.mean([r[k] for r in valid])) for k in ('native','reset_state','fixed_time')},
        'gap_zero_fraction':float(np.mean(np.asarray(gaps)==0)),
        'median_last_5_gradient_mass':float(np.median([r['last_5_gradient_mass'] for r in records if r['last_5_gradient_mass'] is not None])),
        'median_older_than_20_gradient_mass':float(np.median([r['older_than_20_gradient_mass'] for r in records if r['older_than_20_gradient_mass'] is not None])),
        'records':records}
    atomic_json(resolve(args.output),summary)
    print({k:v for k,v in summary.items() if k!='records'})


if __name__=='__main__':
    main()
