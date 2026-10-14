"""Training-only annotated evidence expansion, actual QA labels, fixed-size models."""
import sys, json, time, hashlib, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json, load_config
from src.provenance import snapshot_source
from src.data.longmemeval import prepared, public_history
from src.models.content_encoder import ContentEncoder
from src.memory.semantic_reading import semantic_views, parse_units, render
from src.memory.retained_observation import retained_observation
from scripts import semantic_fallback as base
from scripts import semantic_fallback_augmentation as aug
from scripts import context_utility as generator
OUT=ROOT/'runs/bidirectional-fallback-v1'
read=base.read


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():return
    source,archive=snapshot_source()
    ids=read(base.OUT/'protocol.json')['question_ids']
    atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),source=source,archive=archive,question_ids=ids,
        design='All20 existing train questions, annotation-assisted evidence view then expansion with original same-history text; no labels fabricated, no valid answers resampled. Natural validation and previously exposed12 test questions unchanged.',
        intervention='Answer-session IDs only select TRAIN evidence sources; no gold answer strings in excerpts/features. Query MiniLM extracts1800 characters from evidence sources, expanded view preserves these lines and adds authentic lines from evidence plus highest-query non-evidence sources up to16000 characters. Expanded input may exceed original compressed-memory coverage: training-only privileged augmentation, not same64KB deployment evidence.',
        budgets=[.1,.25,.5],budget='Causal prefix hard cap: extra calls <=floor(rate*queries_seen), no borrowing, initial allowance0. Independent streams per compact budget and hash-order seed11/23/42. No future predictions or labels used for admission.',
        training='Same18features,8hidden,3seeds,60epochs AdamW. Train target full_correct-compact_correct without soft token penalty. Compare previous60pairs to60plus20new pairs. Select epoch by validation MSE; threshold from validation gains only with at most requested fraction and positive score. All methods share hard limiter.',
        scope='Exploratory continuation on exposed test set, not new independent confirmation. Same-query source history; no state transfer between unrelated questions. Report all budgets/models, all three label counts even if a class remains absent.'))
    rows,_=prepared(load_config('configs/minute_precision.yaml'));by={r['question_id']:r for r in rows}
    enc=ContentEncoder();tasks=[];features={}
    for q in ids['train']:
        row=by[q];h=public_history(row,'minute');gold=set(map(str,row['answer_session_ids']))
        def context(indices):
            return '\n\n'.join(f'[session={h.session_ids[i]} date={h.dates[i]}]\n{h.texts[i]}' for i in sorted(indices))
        evidence=[i for i,s in enumerate(h.session_ids) if s in gold];assert evidence
        compact=semantic_views(h.question,context(evidence),enc,[1800])[1800]['context']
        # Rank source relevance without answer text or outcome labels.
        v=enc.encode(list(h.texts));query=enc.encode([h.question])[0]
        other=sorted((i for i in range(len(h.texts)) if i not in evidence),key=lambda i:-float(v[i]@query))[:5]
        units=parse_units(context(evidence+other));retained=set(parse_units(compact));chosen={i for i,u in enumerate(units) if u in retained}
        vv=enc.encode([t for _,t in units]);order=np.argsort(-(vv@query),kind='stable')
        for i in order:
            proposal=chosen|{int(i)}
            if len(render(units,proposal))<=16000:chosen=proposal
        full=render(units,chosen);assert set(parse_units(compact))<=set(parse_units(full))
        obs=retained_observation(h.question,full,enc,compact)
        features[q]=dict(question_id=q,split='train',category=row['question_type'],budget=1800,x=obs['x'],dt=obs['dt'])
        for method,c in [('evidence',compact),('expanded',full)]:
            tasks.append(dict(question_id=q,split='train',category=row['question_type'],method=method,question=h.question,question_date=h.question_date,answer=str(row['answer']),context=c,context_chars=len(c),context_sha256=hashlib.sha256(c.encode()).hexdigest()))
        print('Prepared training intervention',q,flush=True)
    enc.close();atomic_json(OUT/'features.json',features);atomic_json(OUT/'tasks.json',tasks);atomic_json(OUT/'prepared.json',dict(questions=20,answers=40))


def qa():
    prev=generator.OUT
    try:generator.OUT=OUT;generator.qa('development')
    finally:generator.OUT=prev
    from scripts.run_semantic_series import repair_invalid
    repair_invalid(OUT)


def prepare_stages():
    if (OUT/'stage_features.json').exists():return
    atomic_json(OUT/'stage_amendment.json',dict(registered_at=time.time(),reason='Existing TRAIN600->1800 records contain real harmful expansion overlooked by full-only endpoint. Add all20 train adjacent-stage pairs, not just harmful ones. Separate evaluation600->1800; test action outcomes already exposed, exploratory.',recipes=['previous','staged','bidirectional'],training_pairs=[60,80,100],unchanged_architecture=True))
    tasks=read(base.OUT/'tasks.json');by={(r['question_id'],r['method']):r for r in tasks};enc=ContentEncoder();features={}
    for q,method in by:
        if method!='compact600':continue
        c=by[q,method];f=by[q,'compact1800'];v=retained_observation(c['question'],f['context'],enc,c['context'])
        features[q]=dict(question_id=q,split=c['split'],category=c['category'],budget='600_to_1800',x=v['x'],dt=v['dt'])
    enc.close();atomic_json(OUT/'stage_features.json',features)


def stage_data(include_test=False):
    ds={}
    for q,f in read(OUT/'stage_features.json').items():
        if f['split']=='test' and not include_test:continue
        c=read(base.OUT/'qa'/f'{q}-compact600.json');full=read(base.OUT/'qa'/f'{q}-compact1800.json')
        assert all(r['evaluation_status'] in ['graded','explicit_abstention'] for r in [c,full])
        x=np.array(f['x']);x[:,16]=float(c['parsed']['abstain']);x[:,17]=min(len(c['hypothesis'])/1000,2)
        ds[q+':stage']={**f,'x':x,'dt':np.array(f['dt']),'compact':c,'full':full,'target':full['correct']-c['correct']}
    return ds


def data(include_test=False, extra=True, stages=True):
    ds=aug.augmented_data(include_test)
    if stages:
        # Stage validation is reserved for the separately reported stage endpoint.
        ds.update({k:d for k,d in stage_data(include_test).items() if d['split']=='train'})
    if extra:
        review=read(OUT/'reference_review.json')
        assert review['complete'], 'Review all new answers before fitting new supervision'
        for q,f in read(OUT/'features.json').items():
            c=read(OUT/'qa'/f'{q}-evidence.json');full=read(OUT/'qa'/f'{q}-expanded.json')
            assert all(r['evaluation_status'] in ['graded','explicit_abstention'] for r in [c,full])
            for method,r in [('evidence',c),('expanded',full)]:
                entry=review['records'][q+'-'+method]
                assert entry['sha256']==base.sha(OUT/'qa'/f'{q}-{method}.json')
                r['correct']=entry['reviewed_correct']
            x=np.array(f['x']);x[:,16]=float(c['parsed']['abstain']);x[:,17]=min(len(c['hypothesis'])/1000,2)
            ds[q+':expanded']={**f,'x':x,'dt':np.array(f['dt']),'compact':c,'full':full}
    for d in ds.values():d['target']=d['full']['correct']-d['compact']['correct']
    return ds


def train():
    from src.memory.task_controller import TaskController
    torch.set_num_threads(1)
    if (OUT/'training_complete.json').exists():return
    logs=[];counts={}
    for recipe in ['previous','staged','bidirectional']:
        ds=data(False,extra=recipe=='bidirectional',stages=recipe!='previous');tr=[d for d in ds.values() if d['split']=='train'];va=[d for d in ds.values() if d['split']=='val']
        counts[recipe]={str(k):sum(d['target']==k for d in tr) for k in [-1,0,1]}
        folder=OUT/recipe;folder.mkdir(exist_ok=True);xx=np.concatenate([d['x'] for d in tr]);norm=dict(mean=xx.mean(0),std=np.maximum(xx.std(0),.05));np.savez(folder/'normalization.npz',**norm)
        for kind in base.KINDS:
            for seed in base.SEEDS:
                torch.manual_seed(seed);m=TaskController('linear' if kind=='linear_no_omission' else kind,features=18,hidden=8);opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01);best=float('inf')
                for epoch in range(1,61):
                    m.train();loss=torch.stack([(m(*base.transform(d,norm,kind))-d['target'])**2 for d in tr]).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();m.eval()
                    with torch.no_grad():val=float(np.mean([(float(m(*base.transform(d,norm,kind)))-d['target'])**2 for d in va]))
                    logs.append(dict(recipe=recipe,kind=kind,seed=seed,epoch=epoch,train_mse=float(loss.detach()),val_mse=val))
                    if val<best:
                        best=val;torch.save(dict(state_dict=m.state_dict(),epoch=epoch,kind=kind,seed=seed,parameters=sum(p.numel() for p in m.parameters()),train_ids=sorted({d['question_id'] for d in tr}),val_ids=sorted({d['question_id'] for d in va})),folder/f'{kind}-{seed}.pt')
                print('Trained',recipe,kind,seed,flush=True)
    atomic_json(OUT/'training.json',logs);atomic_json(OUT/'training_complete.json',dict(label_counts=counts,models=36,completed_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='all',choices=['all','prepare','qa','train']);a=p.parse_args()
    if a.phase in ['all','prepare']:prepare();prepare_stages()
    if a.phase in ['all','qa']:qa()
    if a.phase in ['all','train']:train()
