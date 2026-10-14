"""Frozen bidirectional NLI verification, fresh calibration and fresh audits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,time
from collections import defaultdict,Counter
import numpy as np
import torch
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from src.data.semantic_gate import sentence_groups,normalize,stable_key,lexical_features,overlap_audit
from src.models.relation_verifier import PairNLI,relation_scores,choose_candidate,certify
from src.models.content_encoder import ContentEncoder
from src.models.semantic_gate import SemanticGate
from src.provenance import snapshot_source,runtime_info

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/semantic-gate-v2'
OUT=ROOT/'runs/relation-verifier-v1'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    prior=json.loads((OLD/'examples.json').read_text(encoding='utf-8'))
    old_train={int(r['id'].rsplit(':',1)[1]) for r in prior if r['split']=='train'}
    raw=pq.read_table(ROOT/'data/paws/train.parquet').to_pylist()
    groups=sentence_groups([dict(text1=r['sentence1'],text2=r['sentence2']) for r in raw])
    blocked={g for r,g in zip(raw,groups) if r['id'] in old_train}
    available=defaultdict(list)
    for r,g in zip(raw,groups):
        if g not in blocked:available[g].append(r)
    chosen=sorted(available,key=lambda g:stable_key('relation-v1:'+g))[:3000]
    if len(chosen)!=3000:raise ValueError('Insufficient isolated components')
    result=[]
    # ONE pair per component removes obvious repeated-sentence dependence from
    # calibration, rather than merely putting many correlated pairs together.
    for i,g in enumerate(chosen):
        r=min(available[g],key=lambda r:stable_key(str(r['id'])))
        result.append(dict(id=f'paws:train:{r["id"]}',group=g,split='certificate' if i<1500 else 'fresh_paws',
                           dataset='paws',text1=r['sentence1'],text2=r['sentence2'],label=int(r['label'])))
    result.extend(dict(r,split='selection') for r in prior if r['split']=='selection')
    result.extend(dict(r,split='revision') for r in prior if r['split']=='challenge')
    mrpc=pq.read_table(ROOT/'data/mrpc/test.parquet').to_pylist()
    for r in mrpc:
        if r['label'] not in (0,1):raise ValueError('MRPC test labels unavailable')
        result.append(dict(id=f'mrpc:test:{r["idx"]}',group=f'mrpc:{r["idx"]}',split='mrpc',dataset='mrpc',
                           text1=r['sentence1'],text2=r['sentence2'],label=r['label']))
    # Protect fresh pair contents against ALL text viewed by previous model study.
    seen={normalize(r[k]) for r in prior for k in ('text1','text2')}
    collisions=[r['id'] for r in result if r['split'] in ('certificate','fresh_paws') and any(normalize(r[k]) in seen for k in ('text1','text2'))]
    if collisions:raise ValueError('Fresh split overlaps previously used content')
    save('split_audit.json',dict(unused_components=len(available),selected_components=3000,one_pair_per_component=True,
        prior_content_collisions=collisions,counts=dict(Counter(r['split'] for r in result)),
        note='Fresh PAWS calibration/audit are custom held-out samples of originally unused official train components, not official test. Remaining article-level dependence possible. MRPC is new domain; MemOps challenge is reused.'))
    save('examples.json',result)
    return result


def score_block(rows,nli,encoder,old_models):
    texts=[(r['text1'],r['text2']) for r in rows]
    forward,meta_f=nli.predict(texts);backward,meta_b=nli.predict([(b,a) for a,b in texts])
    score=relation_scores(forward,backward,texts)
    valid=np.asarray(meta_f['valid'])&np.asarray(meta_b['valid'])
    for name in score:score[name]=np.where(valid,score[name],0)
    vectors=encoder.encode([t for pair in texts for t in pair]).reshape(len(rows),2,384)
    lex=np.stack([lexical_features(a,b) for a,b in texts]);cos=(vectors[:,0]*vectors[:,1]).sum(-1)
    logistic=json.loads((OLD/'logistic.json').read_text())
    x=np.c_[lex,cos];z=((x-np.asarray(logistic['mean']))/np.asarray(logistic['scale']))@np.asarray(logistic['coef'])[0]+logistic['intercept'][0]
    score['old_cosine']=cos;score['old_logistic']=1/(1+np.exp(-z))
    with torch.no_grad():
        for name,model in old_models.items():score[name]=model(torch.from_numpy(vectors),torch.from_numpy(lex)).sigmoid().numpy()
    records=[]
    for i,row in enumerate(rows):
        records.append(dict(id=row['id'],split=row['split'],group=row['group'],label=row['label'],
                            scores={k:float(v[i]) for k,v in score.items()},nli_forward=forward[i].tolist(),
                            nli_backward=backward[i].tolist(),untruncated=bool(valid[i])))
    return records,dict(forward=meta_f,backward=meta_b)


def measure(records,method,threshold):
    labels=np.asarray([r['label'] for r in records]);scores=np.asarray([r['scores'][method] for r in records]);skip=scores>=threshold
    n=int(skip.sum());errors=int((skip&(labels==0)).sum())
    return dict(n=len(labels),skips=n,errors=errors,skip_fraction=float(skip.mean()),
        wrong_among_skips=errors/n if n else None,
        different_content_false_skip=float(skip[labels==0].mean()) if (labels==0).any() else None,
        duplicate_recall=float(skip[labels==1].mean()) if (labels==1).any() else None,
        auc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None)


def main():
    if OUT.exists():raise SystemExit('Run exists; frozen results preserved')
    OUT.mkdir(parents=True);torch.set_num_threads(1)
    rows=prepare();signature,archive=snapshot_source()
    save('protocol.json',dict(signature=signature,source_archive=archive,runtime=runtime_info(),
        nli_source=json.loads((ROOT/'artifacts/encoders/nli-deberta-small/source.json').read_text()),
        mrpc_source=json.loads((ROOT/'data/mrpc/provenance.json').read_text()),
        primary='nli_bidirectional_guarded',
        hypothesis='Joint token-level NLI and mutual entailment improve relation discrimination before any liquid history modeling.',
        scoring='One-way entailment, min(forward,reverse), same min with numerical/negation mismatch veto. No truncation; >512 tokens forces retain.',
        selection='Reuse old PAWS selection only to pick fixed grid threshold with empirical wrong-among-skips<=0.5% and >=100 skips. No eligible threshold => retain all.',
        certification='Fresh 1500 sentence-component representatives: fixed threshold must have >=100 skips and one-sided Clopper-Pearson95 upper wrong-among-skips<=1%. Primary policy predeclared; other policies exploratory, not certified family-wide.',
        audit='Fresh1500 PAWS component representatives, full1725 MRPC official test with public labels, reused147 MemOps revisions. No retuning on any audit.',
        dependence='Components block identical/transitively shared sentence leakage; they do not prove independent articles. Certification is IID binomial diagnostic, not an unconditional guarantee.',
        baselines='Prior cosine/logistic and all3 frozen LTC seeds, recalibrated by SAME stricter selection rule; weights unchanged.',
        actions='Offline retain/skip proposal only. Not a fact-replacement or deletion controller. Failure to certify forces deployment-facing retain-all policy.',
        limitations='NLI pretrained on SNLI/MNLI per author; pretrained contamination cannot be ruled out. MRPC paraphrase labels may tolerate small additions; mutual entailment is stricter. No API, GPU, or claimed end-to-end savings.'))
    nli=PairNLI(ROOT/'artifacts/encoders/nli-deberta-small',ROOT/'artifacts/nli_pair_cache.sqlite3')
    encoder=ContentEncoder();old_models={}
    old_manifest=json.loads((OLD/'frozen_models.json').read_text())
    for seed in (11,23,42):
        name=f'ltc-{seed}';expected=next(r['checkpoint_sha256'] for r in old_manifest if r['method']==name)
        if hashlib.sha256((OLD/f'{name}.pt').read_bytes()).hexdigest()!=expected:raise ValueError('Old model changed')
        model=SemanticGate('ltc');model.load_state_dict(torch.load(OLD/f'{name}.pt',weights_only=True));model.eval();old_models['old_'+name]=model
    all_records=[];timing=[];candidates=None;certificates=None
    for split in ('selection','certificate','fresh_paws','mrpc','revision'):
        subset=[r for r in rows if r['split']==split];records=[]
        for begin in range(0,len(subset),128):
            t=time.perf_counter();batch,stats=score_block(subset[begin:begin+128],nli,encoder,old_models);records+=batch
            timing.append(dict(split=split,start=begin,rows=len(batch),wall_seconds=time.perf_counter()-t,stats=stats))
            # Save raw incremental scores for recovery/audit, never tune on audits.
            with (OUT/'scores.jsonl').open('a',encoding='utf-8') as f:
                for r in batch:f.write(json.dumps(r)+'\n')
            print(split,min(begin+128,len(subset)),'/',len(subset),flush=True)
        all_records+=records
        if split=='selection':
            candidates={name:choose_candidate([r['scores'][name] for r in records],[r['label'] for r in records]) for name in records[0]['scores']}
            save('frozen_candidates.json',candidates)
        elif split=='certificate':
            certificates={name:certify([r['scores'][name] for r in records],[r['label'] for r in records],info['threshold']) for name,info in candidates.items()}
            save('certificates.json',certificates)
            print('Primary calibration decision',certificates['nli_bidirectional_guarded'],flush=True)
    summaries=[]
    for split in ('selection','certificate','fresh_paws','mrpc','revision'):
        records=[r for r in all_records if r['split']==split]
        for method,info in candidates.items():
            summaries.append(dict(split=split,method=method,threshold=info['threshold'],policy='candidate',**measure(records,method,info['threshold'])))
            actual=info['threshold'] if certificates[method]['approved'] else 1.000001
            summaries.append(dict(split=split,method=method,threshold=actual,policy='fail_closed',**measure(records,method,actual)))
    save('summary.json',summaries);save('timing.json',dict(blocks=timing,cold_nli_batches=nli.batches,nli_signature=nli.signature))
    save('complete.json',dict(pairs=len(all_records),api_requests=0,methods=len(candidates),primary_approved=certificates['nli_bidirectional_guarded']['approved']))
    encoder.close();nli.close();print('Completed relation verification study',flush=True)


if __name__=='__main__':main()
