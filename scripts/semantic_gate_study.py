"""Frozen public semantic skip gate experiment; no API and no invented labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib,time,copy
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.data.semantic_gate import paws_rows,update_rows,lexical_features,overlap_audit
from src.models.semantic_gate import SemanticGate,calibrate
from src.models.content_encoder import ContentEncoder
from src.provenance import snapshot_source,runtime_info

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/semantic-gate-v2'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    if OUT.exists():raise SystemExit('Run exists; preserve frozen preparation')
    OUT.mkdir(parents=True)
    rows=paws_rows(ROOT/'data/paws');updates,audit=update_rows(ROOT/'data/memops/updates');rows+=updates
    signature,archive=snapshot_source()
    save('protocol.json',dict(signature=signature,source_archive=archive,runtime=runtime_info(),
        sources={'paws':json.loads((ROOT/'data/paws/provenance.json').read_text()),'memops':json.loads((ROOT/'data/memops/provenance.json').read_text())},
        splits=dict(Counter(r['split'] for r in rows)),sampling='SHA256(split:id) first6000 official train; ALL official dev grouped by shared-sentence connected components, group SHA modulo2 selects selection/calibration; FULL official8000 test. No label-based selection.',
        prior_attempt='v1 preparation audit found shared sentences between row-sampled selection/calibration. Stopped before any training/test metrics; v2 fixes group isolation, preserves v1.',
        labels='PAWS human paraphrase labels. MemOps generated gold confirmed updates with verbatim user triggers, matched predecessor and changed value. Non-operation turns never treated as negatives/duplicates.',
        challenge='All eligible public update-file chains; gold target binding and trigger selection supplied for pair construction, NOT end-to-end extraction. No MemOps fitting or calibration.',
        features='Pinned frozen MiniLM384 for each full sentence/trigger quote; 7 lexical features shared by all learned gates. No gold target, old/new value, operation name, session labels in model input.',
        kinds=['mlp','gru','input_flow','ltc'],width=64,seeds=[11,23,42],epochs=30,patience=8,batch=256,lr=.001,
        baselines=['always_write','exact_match','cosine','cosine_numeric_negation_guard','logistic_lexical'],
        threshold='Separate calibration subset: maximize skip count with score>=0.5 and empirical Wilson95 upper nonparaphrase false-skip <=1%. This selection heuristic is not a risk guarantee under selection or domain shift.',
        timing='Pair gate only CPU one thread; encoding measured separately; downstream write engine not executed so savings are skip opportunity, NOT actual end-to-end speedup.',
        scope='Pair-local semantic discrimination; recurrent state resets per pair, unit steps not physical time. No long-horizon liquid-memory superiority claim.',
        model_fairness='Same input vectors, lexical signals, projected width and reader form. Core parameter counts differ; not FLOP-matched.',
        test_policy='Freeze all checkpoints and thresholds before scoring PAWS test or MemOps challenge. No test-guided rerun.'))
    save('examples.json',rows);save('update_label_audit.json',audit);save('overlap_audit.json',overlap_audit(rows))
    texts=sorted({r[k] for r in rows for k in ('text1','text2')});index={t:i for i,t in enumerate(texts)}
    encoder=ContentEncoder();parts=[];start=time.perf_counter()
    for begin in range(0,len(texts),512):
        parts.append(encoder.encode(texts[begin:begin+512]));print('encoded',min(begin+512,len(texts)),'/',len(texts),flush=True)
    vectors=np.concatenate(parts);elapsed=time.perf_counter()-start;encoder_signature=encoder.signature;encoder.close()
    pairs=np.asarray([[index[r['text1']],index[r['text2']]] for r in rows]);lex=np.stack([lexical_features(r['text1'],r['text2']) for r in rows])
    np.savez_compressed(OUT/'features.npz',vectors=vectors,pairs=pairs,lexical=lex)
    save('preparation.json',dict(unique_texts=len(texts),encoder_signature=encoder_signature,encoding_seconds=elapsed,
                               vector_cache_included=True,update_audit=dict(Counter(r['status'] for r in audit)),feature_sha256=hashlib.sha256((OUT/'features.npz').read_bytes()).hexdigest()))
    print('Preparation completed',flush=True)


def metrics(scores,labels,threshold):
    scores=np.asarray(scores);labels=np.asarray(labels);skip=scores>=threshold;positive=labels==1;negative=labels==0
    return dict(n=len(labels),accuracy=float(np.mean(skip==positive)),skip_fraction=float(skip.mean()),
        duplicate_recall=float(skip[positive].mean()) if positive.any() else None,
        false_skip_rate=float(skip[negative].mean()) if negative.any() else None,
        false_skips=int((skip&negative).sum()),duplicate_count=int(positive.sum()),different_count=int(negative.sum()),
        auc=float(roc_auc_score(labels,scores)) if positive.any() and negative.any() else None,
        balanced_accuracy_at_half=float(balanced_accuracy_score(labels,scores>=.5)) if positive.any() and negative.any() else None)


def train():
    if (OUT/'complete.json').exists() or (OUT/'frozen_models.json').exists():raise SystemExit('Frozen models already exist')
    torch.set_num_threads(1)
    rows=json.loads((OUT/'examples.json').read_text(encoding='utf-8'));data=np.load(OUT/'features.npz')
    pair=torch.from_numpy(data['vectors'][data['pairs']]);lex=torch.from_numpy(data['lexical']);y=torch.tensor([r['label'] for r in rows],dtype=torch.float32)
    indices={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','selection','calibration','test','challenge')}
    def probabilities(model,ids):
        with torch.no_grad():return torch.cat([model(pair[b],lex[b]).sigmoid() for b in ids.split(512)]).numpy()
    frozen=[]
    # All fitting and thresholds precede ANY test/challenge scoring.
    for kind in ('mlp','gru','input_flow','ltc'):
        for seed in (11,23,42):
            torch.manual_seed(seed);model=SemanticGate(kind);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            best=float('inf');best_state=None;stale=0;history=[];start=time.perf_counter()
            for epoch in range(30):
                model.train();losses=[]
                for b in indices['train'][torch.randperm(len(indices['train']))].split(256):
                    opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(pair[b],lex[b]),y[b]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    b=indices['selection'];val=float(torch.nn.functional.binary_cross_entropy_with_logits(model(pair[b],lex[b]),y[b]))
                history.append(dict(epoch=epoch+1,train_nll=float(np.mean(losses)),selection_nll=val))
                if val<best:best=val;best_state=copy.deepcopy(model.state_dict());stale=0
                else:stale+=1
                if stale>=8:break
            elapsed=time.perf_counter()-start;model.load_state_dict(best_state);model.eval();name=f'{kind}-{seed}'
            torch.save(best_state,OUT/f'{name}.pt');save(f'{name}-epochs.json',history)
            calibration=calibrate(probabilities(model,indices['calibration']),y[indices['calibration']].numpy())
            frozen.append(dict(method=name,kind=kind,seed=seed,calibration=calibration,selection_nll=best,epochs=len(history),
                               train_seconds=elapsed,weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                               checkpoint_sha256=hashlib.sha256((OUT/f'{name}.pt').read_bytes()).hexdigest()))
            print(name,'trained',len(history),'epochs',round(elapsed,1),'s','calibration skips',calibration['skip_count'],flush=True)
    cosine=(pair[:,0]*pair[:,1]).sum(-1).numpy();features=np.c_[lex.numpy(),cosine]
    scaler=StandardScaler().fit(features[indices['train']]);logit=LogisticRegression(C=1,max_iter=500).fit(scaler.transform(features[indices['train']]),y[indices['train']].numpy())
    # Persist simple-model numeric coefficients without pickled executable objects.
    save('logistic.json',dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=logit.coef_.tolist(),intercept=logit.intercept_.tolist()))
    baseline={'always_write':np.zeros(len(rows)),'exact_match':lex[:,0].numpy(),'cosine':cosine,
              'cosine_numeric_negation_guard':np.where((lex[:,4].numpy()==0)&(lex[:,5].numpy()==0),cosine,0),
              'logistic_lexical':logit.predict_proba(scaler.transform(features))[:,1]}
    for name,scores in baseline.items():
        calibration={'threshold':.5,'note':'fixed exact/no-skip policy'} if name in ('always_write','exact_match') else calibrate(scores[indices['calibration']],y[indices['calibration']].numpy())
        frozen.append(dict(method=name,calibration=calibration))
    save('frozen_models.json',frozen)
    evaluate()


def evaluate():
    if (OUT/'complete.json').exists():raise SystemExit('Completed evaluation preserved')
    torch.set_num_threads(1)
    rows=json.loads((OUT/'examples.json').read_text(encoding='utf-8'));data=np.load(OUT/'features.npz')
    pair=torch.from_numpy(data['vectors'][data['pairs']]);lex=torch.from_numpy(data['lexical']);y=torch.tensor([r['label'] for r in rows],dtype=torch.float32)
    indices={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','selection','calibration','test','challenge')}
    frozen=json.loads((OUT/'frozen_models.json').read_text())
    for item in frozen:
        if 'kind' in item and hashlib.sha256((OUT/f'{item["method"]}.pt').read_bytes()).hexdigest()!=item['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
    cosine=(pair[:,0]*pair[:,1]).sum(-1).numpy();features=np.c_[lex.numpy(),cosine]
    logistic=json.loads((OUT/'logistic.json').read_text())
    z=((features-np.asarray(logistic['mean']))/np.asarray(logistic['scale']))@np.asarray(logistic['coef'])[0]+logistic['intercept'][0]
    baseline={'always_write':np.zeros(len(rows)),'exact_match':lex[:,0].numpy(),'cosine':cosine,
              'cosine_numeric_negation_guard':np.where((lex[:,4].numpy()==0)&(lex[:,5].numpy()==0),cosine,0),
              'logistic_lexical':1/(1+np.exp(-z))}
    def probabilities(model,ids):
        with torch.no_grad():return torch.cat([model(pair[b],lex[b]).sigmoid() for b in ids.split(512)]).numpy()
    signature,archive=snapshot_source()
    save('evaluation_source.json',dict(signature=signature,source_archive=archive,
        repair='Original evaluation call omitted threshold argument and stopped before summary. Evaluation-only fix reuses byte-identical frozen checkpoints and thresholds, no retraining or retuning.'))
    results=[];predictions=[];timings=[]
    for item in frozen:
        name=item['method'];threshold=item['calibration']['threshold']
        model=None
        if 'kind' in item:
            model=SemanticGate(item['kind']);model.load_state_dict(torch.load(OUT/f'{name}.pt',weights_only=True));model.eval()
            costs=[]
            with torch.no_grad():
                for i in range(550):
                    b=indices['selection'][i%len(indices['selection'])].item()
                    a=time.perf_counter_ns();_ = model(pair[b:b+1],lex[b:b+1]);duration=time.perf_counter_ns()-a
                    if i>=50:costs.append(duration/1000)
            timings.append(dict(method=name,gate_us_p50=float(np.median(costs)),gate_us_p95=float(np.percentile(costs,95)),raw_us=costs))
        for split in ('test','challenge'):
            ids=indices[split]
            if not len(ids):continue
            scores=probabilities(model,ids) if model else baseline[name][ids]
            result=dict(method=name,split=split,threshold=threshold,**metrics(scores,y[ids].numpy(),threshold));results.append(result)
            for i,s in zip(ids.tolist(),scores):
                row=rows[i];predictions.append(dict(id=row['id'],group=row.get('group'),split=split,method=name,label=row['label'],score=float(s),skip=bool(s>=threshold)))
    save('summary.json',results);save('timing.json',timings)
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    save('complete.json',dict(models=12,baselines=5,predictions=len(predictions),api_requests=0))
    print('All held-out scores saved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'train':train,'evaluate':evaluate}[args.stage]()
