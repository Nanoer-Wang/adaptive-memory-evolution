"""Train/calibrate only on train/validation; frozen multi-axis sparse-memory study."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,random,time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from src.config import atomic_json,load_config
from src.provenance import snapshot_source,runtime_info
from src.data.longmemeval import prepared,public_history,date_days
from src.memory.turn_compression import compress_turns,sentences
from src.memory.budget_store import BudgetItem,byte_retained_indices
from src.memory.risk_residual import RiskResidualWriter,risk_features,call_mask
from src.models.content_encoder import ContentEncoder
from src.retrieval.preserved_context import pack_preserved_context
from scripts.content_experiment import Reader,deltas
from scripts.writer_efficiency import examples,read
OUT=ROOT/'runs/risk-trajectory-v1';SEEDS=[11,23,42];KINDS=['liquid','gru','static'];REGIMES=['full','periodic','risk'];BUDGETS=[32000,64000,128000]

def history(r):
    h=public_history(r,'minute');ordered=sorted(zip(r['haystack_dates'],r['haystack_sessions']),key=lambda x:date_days(x[0],'minute'))
    return replace(h,texts=tuple(compress_turns(t,2400,'sentence_user_first') for _,t in ordered))

def row_weights(labels):
    return np.where(labels>0,.5/max(labels.sum(),1),.5/max(len(labels)-labels.sum(),1)).astype(np.float32)

def init(rows,manifest):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return
    sig,archive=snapshot_source();atomic_json(OUT/'protocol.json',dict(source=sig,archive=archive,runtime=runtime_info(),splits=manifest['splits'],data_signature=manifest['signature'],seeds=SEEDS,budgets=BUDGETS,
        model='Shared train-only linear content score. Writer predicts an additive residual on every event, not two incompatible score scales. All writers receive the same learned per-coordinate autonomous exponential decay between observations. On selected events, liquid/GRU/static core processes current vector. Skipped events preserve time-evolving residual and still get current-content cheap score.',
        gate='Shared384-vector + cheap score + abs score + cosine novelty + log1p elapsed + intercept, ridge alpha1 to class-balanced squared cheap prediction error. Targets only train labels. This is a fallibility proxy, not validated semantic contradiction or calibrated probability. Train-set residual targets are in-sample; validation measures generalization. Cheap classifier200full-batch Adam steps lr0.03. Fixed recipes.',
        budget='Causal token bucket initial1,capacity2,refill0.5/event, max floor(1+0.5*t) calls for first t events. One validation-only threshold calibrated to48% observed event call rate. Same gate/masks shared across all seeds/writers; no query/labels when deciding. Actual rates under shift explicitly reported.',
        training='27models=3architectures x3execution regimes x3seeds;3epochs AdamW0.003 wd0.01,balanced per-history BCE,clip1. Start from same random seed per architecture across regimes,zero residual head. Checkpoint selected by validation evidence retention at64k only. Train masks exactly match deployment for matched conditions.',
        primary='LME64k risk-trained/risk-executed liquid compared to shared cheap baseline, risk GRU, risk static, and full-trained/risk-executed liquid. History-cluster bootstrap5000,95% and98.75% intervals. Strict liquid advantage requires all corrected lower bounds>0. Noninferiority vs full-trained/full-executed liquid uses -1pp recall margin. Both previously exposed datasets: exploratory research, not fresh confirmation.',
        dimensions='Recall/retention,complete supporting-session retrieval,latest labeled session retention,knowledge-update/temporal categories,history span,pressure budgets,call savings,positive-evidence trigger sensitivity,false safety(non-triggered positive),calibration MAE,latency,checkpoint/seed consistency; context sentence coverage on QA; these are proxies not truth-state/agent success.',
        controls='cheap,FIFO; each trained regime in matching execution; full-trained models also run under risk to isolate training mismatch; risk liquid reset/zero-time ablations. No testing-to-training feedback.',
        qa='Same12fixed IDs from hybrid-validation-v1, four methods:cheap,liquid_full_full,liquid_risk_risk,gru_risk_risk;seed23,64k. Existing exposed QA set reused explicitly for paired development comparison.48answers plus custom judge. All cases kept.',qa_ids=read(ROOT/'runs/hybrid-validation-v1/protocol.json')['qa_ids']))

def prepare_data(rows,manifest):
    enc=ContentEncoder();result={};ids=set(manifest['splits']['train']+manifest['splits']['val'])
    for i,r in enumerate(r for r in rows if r['question_id'] in ids and not str(r['question_id']).endswith('_abs')):
        h=history(r);v=enc.encode(list(h.texts));labels=np.array([float(s in r['answer_session_ids']) for s in h.session_ids],dtype=np.float32)
        costs=np.array([BudgetItem(s,d,t,x,0,j).nbytes for j,(s,d,t,x) in enumerate(zip(h.session_ids,h.dates,h.texts,v))])
        result[r['question_id']]=dict(h=h,v=v,dt=deltas(h),y=labels,weights=row_weights(labels),costs=costs)
        if i%100==0:print('Risk features',i+1,flush=True)
    enc.close();return result

def fit_shared(data,manifest):
    path=OUT/'shared.npz'
    if path.exists():return np.load(path)
    train=[data[q] for q in manifest['splits']['train'] if q in data];val=[data[q] for q in manifest['splits']['val'] if q in data]
    x=torch.from_numpy(np.concatenate([d['v'] for d in train]));y=torch.from_numpy(np.concatenate([d['y'] for d in train]));weights=torch.from_numpy(np.concatenate([d['weights'] for d in train]))/len(train)
    torch.manual_seed(20261012);linear=torch.nn.Linear(384,1);optimizer=torch.optim.Adam(linear.parameters(),lr=.03);losses=[]
    for step in range(200):
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(linear(x).squeeze(-1),y,reduction='none')*weights).sum();optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    cheap=np.r_[linear.weight.detach().numpy().ravel(),linear.bias.detach().numpy()].astype(np.float32)
    scalar_history=(OUT/'compact_mode.json').exists()
    xx=[];yy=[];ww=[]
    for d in train:
        base=d['v']@cheap[:-1]+cheap[-1];features=risk_features(d['v'],d['dt'],base,scalar_history);p=1/(1+np.exp(-np.clip(base,-30,30)))
        xx.append(features);yy.append((p-d['y'])**2);ww.append(d['weights'])
    xx=np.concatenate(xx).astype(np.float64);yy=np.concatenate(yy);ww=np.concatenate(ww)
    mean=xx[:,:-1].mean(0);std=np.maximum(xx[:,:-1].std(0),1e-4);xx[:,:-1]=(xx[:,:-1]-mean)/std
    reg=np.eye(xx.shape[1]);reg[-1,-1]=0;risk=np.linalg.solve(xx.T@(xx*ww[:,None])+reg,xx.T@(yy*ww)).astype(np.float32)
    valrisks=[]
    for d in val:
        base=d['v']@cheap[:-1]+cheap[-1];z=risk_features(d['v'],d['dt'],base,scalar_history);z[:,:-1]=(z[:,:-1]-mean)/std;valrisks.append(z@risk)
    lower=min(float(x.min()) for x in valrisks)-1;upper=max(float(x.max()) for x in valrisks)+1
    for _ in range(40):
        midpoint=(lower+upper)/2;rate=sum(call_mask(x,midpoint).sum() for x in valrisks)/sum(len(x) for x in valrisks)
        if rate>.48:lower=midpoint
        else:upper=midpoint
    threshold=(lower+upper)/2
    np.savez(path,cheap=cheap,risk=risk,mean=mean.astype(np.float32),std=std.astype(np.float32),threshold=np.array(threshold),scalar_history=np.array(scalar_history))
    atomic_json(OUT/'shared_training.json',dict(cheap_losses=losses,train_histories=len(train),validation_histories=len(val),threshold=threshold,validation_rate=float(sum(call_mask(x,threshold).sum() for x in valrisks)/sum(len(x) for x in valrisks)),features=len(risk),risk_target='In-sample class-balanced cheap squared error; not a calibrated semantic risk probability'))
    return np.load(path)

def apply_shared(v,dt,shared):
    w=shared['cheap'];base=v@w[:-1]+w[-1];scalar_history=bool(shared['scalar_history']) if 'scalar_history' in shared else False;z=risk_features(v,dt,base,scalar_history);z[:,:-1]=(z[:,:-1]-shared['mean'])/shared['std'];risk=z@shared['risk']
    masks={r:call_mask(risk,float(shared['threshold']),r) for r in REGIMES}
    return base,risk,masks

def train():
    rows,manifest=prepared(load_config('configs/minute_precision.yaml'));init(rows,manifest)
    if (OUT/'training_complete.json').exists():return
    data=prepare_data(rows,manifest);shared=fit_shared(data,manifest)
    for d in data.values():
        base,risk,masks=apply_shared(d['v'],d['dt'],shared);d.update(x=torch.from_numpy(d['v']),delta=torch.from_numpy(d['dt']),base=torch.from_numpy(base),labels=torch.from_numpy(d['y']),weight=torch.from_numpy(d['weights']),masks=masks)
    trainids=[q for q in manifest['splits']['train'] if q in data];val=[data[q] for q in manifest['splits']['val'] if q in data]
    for kind in KINDS:
        for regime in REGIMES:
            for seed in SEEDS:
                directory=OUT/'checkpoints'/f'{kind}-{regime}-{seed}';directory.mkdir(parents=True,exist_ok=True)
                if (directory/'complete.json').exists():continue
                random.seed(seed);torch.manual_seed(seed);model=RiskResidualWriter(kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);best=-1.;logs=[]
                for epoch in range(1,4):
                    start=time.perf_counter();order=list(trainids);random.shuffle(order);losses=[];model.train()
                    for q in order:
                        d=data[q];out=model(d['x'],d['delta'],d['base'],d['masks'][regime]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(out,d['labels'],reduction='none')*d['weight']).sum()
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
                    model.eval();values=[]
                    with torch.no_grad():
                        for d in val:
                            scores=model(d['x'],d['delta'],d['base'],d['masks'][regime]).numpy();kept=byte_retained_indices(scores,d['costs'],64000,model.hidden*4+16);values.append(float(d['y'][kept].sum()/max(d['y'].sum(),1)))
                    metric=float(np.mean(values));ck=dict(state_dict=model.state_dict(),kind=kind,regime=regime,seed=seed,epoch=epoch,train_ids=trainids)
                    if metric>best:best=metric;torch.save(ck,directory/'best.pt')
                    logs.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_retention=metric,seconds=time.perf_counter()-start));atomic_json(directory/'epochs.json',logs)
                    print('Train',kind,regime,seed,epoch,round(metric,4),flush=True)
                atomic_json(directory/'complete.json',dict(best=best,parameters=sum(p.numel() for p in model.parameters())))
    atomic_json(OUT/'training_complete.json',dict(models=27,epochs_per_model=3))

def evaluate(destination=None,account_controller=False):
    target=OUT if destination is None else destination
    target.mkdir(parents=True,exist_ok=True)
    if (target/'offline_complete.json').exists():return
    if (target/'records.jsonl').exists():raise RuntimeError('Preserve partial evaluation before retry')
    shared=np.load(OUT/'shared.npz');models={};hashes={}
    for kind in KINDS:
        for regime in REGIMES:
            for seed in SEEDS:
                p=OUT/'checkpoints'/f'{kind}-{regime}-{seed}'/'best.pt';ck=torch.load(p,weights_only=True);m=RiskResidualWriter(kind);m.load_state_dict(ck['state_dict']);m.eval();models[kind,regime,seed]=m;hashes[f'{kind}-{regime}-{seed}']=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(target/'checkpoint_hashes.json',hashes)
    raw={str(r['question_id']):r for r in prepared(load_config('configs/minute_precision.yaml'))[0]};gg=defaultdict(list);enc=ContentEncoder();qa=[];gate_log=[];timing=[];qaids=read(OUT/'protocol.json')['qa_ids']
    for e in examples():
        h=e['history'];key=hashlib.sha256(json.dumps([h.session_ids,h.dates,h.texts],ensure_ascii=False).encode()).hexdigest();gg[key].append(e)
    with (target/'records.jsonl').open('w',encoding='utf-8') as f,torch.inference_mode():
        for gi,(key,ee) in enumerate(gg.items()):
            h=ee[0]['history'];v=enc.encode(list(h.texts));dt=deltas(h);base,risk,masks=apply_shared(v,dt,shared);vt=torch.from_numpy(v);dtt=torch.from_numpy(dt);bt=torch.from_numpy(base);qs=enc.encode([e['history'].question for e in ee]);cache={};costs=np.array([BudgetItem(s,d,t,z,0,i).nbytes for i,(s,d,t,z) in enumerate(zip(h.session_ids,h.dates,h.texts,v))])
            conditions=[('cheap',0,base,0,None),('fifo',0,np.arange(len(v)),0,None)]
            for (kind,regime,seed),m in models.items():
                modes=[regime]+(['risk'] if regime=='full' else [])
                for execution in modes:
                    start=time.perf_counter();score=m(vt,dtt,bt,masks[execution]).numpy();elapsed=time.perf_counter()-start;name=f'{kind}_{regime}_{execution}'
                    conditions.append((name,seed,score,m.hidden*4,execution));timing.append(dict(cluster=key,dataset=ee[0]['dataset'],method=name,seed=seed,seconds=elapsed,events=len(v),calls=int(masks[execution].sum())))
                if kind=='liquid' and regime=='risk':
                    for mode in ['reset','zero_time']:
                        score=m(vt,dtt,bt,masks['risk'],reset=mode=='reset',zero_time=mode=='zero_time').numpy();conditions.append(('liquid_risk_'+mode,seed,score,m.hidden*4,'risk'))
            for e in ee:
                gold=set(e['gold']);labels=np.array([s in gold for s in h.session_ids]);ok=e['eligible'] and bool(gold)
                probability=1/(1+np.exp(-np.clip(base,-30,30)))
                gate_log.append(dict(question_id=e['history'].question_id,dataset=e['dataset'],cluster=key,category=e['category'],events=len(v),calls=int(masks['risk'].sum()),threshold_requests=int(np.sum(risk>=float(shared['threshold']))),positive_events=int(labels.sum()) if ok else None,triggered_positive=int(np.sum(labels&masks['risk'])) if ok else None,risk_target_mae=float(np.mean(np.abs(risk-(probability-labels)**2))) if ok else None,mean_risk=float(risk.mean()),history_days=float(date_days(h.dates[-1],'minute')-date_days(h.dates[0],'minute'))))
            for name,seed,scores,state,execution in conditions:
                scalar_history=bool(shared['scalar_history']) if 'scalar_history' in shared else False
                controller=(16 if scalar_history else 1544) if account_controller and execution=='risk' else 0
                for budget in BUDGETS:
                    kept=byte_retained_indices(scores,costs,budget,state+16+controller);ids=[h.session_ids[i] for i in kept];used=int(costs[kept].sum()+state+16+controller);assert used<=budget;ck=tuple(kept)
                    if ck not in cache:
                        items=[BudgetItem(h.session_ids[i],h.dates[i],h.texts[i],v[i],0,i) for i in kept];reader=Reader(SimpleNamespace(items=items)) if items else None;cache[ck]=[reader.query(e['history'].question,q)[0] if reader else [] for e,q in zip(ee,qs)]
                    for e,selected in zip(ee,cache[ck]):
                        gold=set(e['gold']);ok=e['eligible'] and bool(gold);found={ids[i] for i in selected};latest=next((s for s in reversed(h.session_ids) if s in gold),None)
                        rec=dict(dataset=e['dataset'],question_id=e['history'].question_id,cluster=key,category=e['category'],method=name,seed=seed,budget=budget,recall=len(gold&found)/len(gold) if ok else None,retention=len(gold&set(ids))/len(gold) if ok else None,all_evidence_retrieved=gold<=found if ok else None,latest_evidence_retained=latest in ids if ok else None,payload_bytes=used,controller_bytes=controller,history_days=float(date_days(h.dates[-1],'minute')-date_days(h.dates[0],'minute')))
                        f.write(json.dumps(rec)+'\n')
                        if e['history'].question_id in qaids and budget==64000 and (name=='cheap' or (seed==23 and name in ['liquid_full_full','liquid_risk_risk','gru_risk_risk'])):
                            hh=replace(e['history'],session_ids=tuple(ids),dates=tuple(h.dates[i] for i in kept),texts=tuple(h.texts[i] for i in kept));ctx,trace=pack_preserved_context(hh,selected,16000);r=raw[hh.question_id];ss=[s for turns in r['haystack_sessions'] for turn in turns if turn.get('has_answer') is True for s in sentences(turn['content'])];norm=' '.join(ctx.split());coverage=float(np.mean([' '.join(s.split()) in norm for s in ss])) if ss else None
                            qa.append(dict(**rec,question=hh.question,question_date=hh.question_date,answer=str(r['answer']),context=ctx,trace=trace,context_sentence_coverage=coverage))
            f.flush()
            if gi%10==0:print('Risk evaluation histories',gi+1,'/',len(gg),flush=True)
    enc.close();atomic_json(target/'gates.json',gate_log);atomic_json(target/'timing.json',timing);atomic_json(target/'qa_tasks.json',qa);atomic_json(target/'offline_complete.json',dict(histories=len(gg),qa_tasks=len(qa),controller_accounted=account_controller))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','offline','qa','all'],default='all');a=p.parse_args();torch.set_num_threads(1)
    if a.phase in ['train','all']:train()
    if a.phase in ['offline','all']:evaluate()
    if a.phase in ['qa','all']:
        import scripts.writer_efficiency as runner
        runner.OUT=OUT;runner.qa_run()

