"""Public bAbI current-state estimation + explicitly synthetic channel stress."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,time,random,copy
import numpy as np
import torch
from filelock import FileLock
from src.config import resolve,atomic_json
from src.data.babi_events import stories,read_key
from src.memory.version_ledger import Evidence
from src.memory.liquid_belief import LiquidBelief,KINDS,INPUTS
from src.provenance import snapshot_source,runtime_info,utc_now,rss_mb

ROOT=resolve('runs/liquid-belief-v1')
SEEDS=(11,23,42)
CONDITIONS={'complete':(1.,1.),'missing':(.6,1.),'missing_noisy':(.6,.8)}
ROOMS=('bathroom','bedroom','garden','hallway','kitchen','office')


def log(x):print(utc_now(),x,flush=True)


def uniform(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:13],16)/16**13


def prepare():
    output=[];hashes={}
    for origin in ('train','test'):
        path=resolve(f'data/babi/qa1_single-supporting-fact_{origin}.txt')
        hashes[origin]=hashlib.sha256(path.read_bytes()).hexdigest()
        data=stories(path,origin)
        for sid,story in enumerate(data):
            digest=hashlib.sha256('\n'.join(x.quote if isinstance(x,Evidence) else x['question'] for x in story).encode()).hexdigest()
            split='test' if origin=='test' else ('validation' if int(digest[:8],16)%5==0 else 'train')
            for condition,(capture,reliability) in CONDITIONS.items():
                banks={};oracle={}
                for item in story:
                    if isinstance(item,Evidence):
                        owner=item.owner;truth=ROOMS.index(item.value);oracle[owner]=truth
                        key=f'{origin}:{sid}:{item.turn}'
                        # Same masking realization for missing and missing_noisy.
                        if uniform(key+':drop')>=capture:continue
                        reported=truth
                        if uniform(key+':corrupt')>=reliability:
                            alternatives=[i for i in range(6) if i!=truth]
                            reported=alternatives[min(int(uniform(key+':replacement')*5),4)]
                        likelihood=[(1-reliability)/5]*6;likelihood[reported]=reliability
                        previous=banks.get(owner,[])
                        delta=item.turn-(previous[-1]['time'] if previous else 0)
                        features=likelihood+[1.,0.,0.,reliability,capture,float(np.log1p(delta)/4)]
                        banks.setdefault(owner,[]).append(dict(time=float(item.turn),features=features,source_line=item.turn,reported=ROOMS[reported],source_quote=item.quote))
                    else:
                        owner=read_key(item['question'])[0]
                        if owner not in oracle or ROOMS[oracle[owner]]!=item['answer']:raise ValueError('Parser/oracle mismatch')
                        # Query chooses a pre-existing independent entity slot;
                        # the same writer runs for all entities, no query input.
                        output.append(dict(id=f'{origin}:{sid}:{item["line"]}:{condition}',story=f'{origin}:{sid}',story_hash=digest,
                            split=split,condition=condition,owner=owner,question=item['question'],label=ROOMS.index(item['answer']),
                            query_time=float(item['line']),capture=capture,reliability=reliability,observations=list(banks.get(owner,[]))))
    return output,hashes


def tensors(rows):
    length=max(1,max(len(r['observations']) for r in rows));n=len(rows)
    events=torch.zeros(n,length,INPUTS);events[:,:,:6]=1/6
    times=torch.zeros(n,length);valid=torch.zeros(n,length,dtype=torch.bool)
    for i,row in enumerate(rows):
        for j,event in enumerate(row['observations']):
            events[i,j]=torch.tensor(event['features']);times[i,j]=event['time'];valid[i,j]=True
    return dict(events=events,times=times,valid=valid,query_time=torch.tensor([r['query_time'] for r in rows]),
        capture=torch.tensor([r['capture'] for r in rows])),torch.tensor([r['label'] for r in rows])


def metrics(p,y):
    pred=p.argmax(-1);confidence=p.max(-1).values;correct=pred==y
    nll=-p[torch.arange(len(y)),y].clamp_min(1e-8).log().mean()
    brier=((p-torch.nn.functional.one_hot(y,6))**2).sum(-1).mean()
    ece=0.
    for i in range(10):
        mask=(confidence>=i/10)&((confidence<(i+1)/10) if i<9 else (confidence<=1))
        if mask.any():ece+=float(mask.float().mean()*(confidence[mask].mean()-correct[mask].float().mean()).abs())
    return dict(accuracy=float(correct.float().mean()),nll=float(nll),brier=float(brier),ece=ece)


def predict(model,rows,steps=None):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(rows),256):
            x,y=tensors(rows[start:start+256]);outputs.append(model(**x,steps=steps))
    return torch.cat(outputs)


def baseline(rows,rate,prior):
    output=[]
    for row in rows:
        # A report explicitly describes a movement, so it resets the
        # observation distribution. No hidden truth used after masking.
        observations=row['observations']
        p=np.array(observations[-1]['features'][:6]) if observations else prior.copy()
        since=row['query_time']-(observations[-1]['time'] if observations else 0)
        alpha=-np.expm1(-rate*since*(1-row['capture']))
        output.append(p+alpha*(prior-p))
    return torch.tensor(np.asarray(output),dtype=torch.float32)


def freeze(rows,hashes):
    signature,archive=snapshot_source()
    spec=dict(source_signature=signature,source_archive=archive,dataset_sha256=hashes,seeds=list(SEEDS),kinds=list(KINDS),rooms=list(ROOMS),conditions=CONDITIONS,
        splits={s:sorted({r['story'] for r in rows if r['split']==s}) for s in ('train','validation','test')},
        design=dict(task='Per-entity current-location belief. Immutable evidence remains source archive; no free-text generation or memory eviction. Does NOT solve the whole LongMemEval task.',
        data='Public bAbI en1k qa1, official test previously used for ledger checks: not pristine held-out evidence. Train stories SHA partition80/20, no story prefix overlap. Missing/noisy conditions are generated stress variants, NOT official benchmark scores.',
        channel='Known capture1 or0.6; reliability1 or0.8. Hash-selected omissions and wrong values independent of answer. Missing and noisy share omission mask. Model only sees observed reports; quoted original values stay in provenance file, not neural features. Same channel parameters supplied to all methods.',
        clock='Original numbered lines are logical steps, NOT physical time. A time-unit conversion must convert rates too; no real-world timescale generalization claim.',
        equation='State-coupled LTC dh/dt=-(leak+softplus(W[h,c]))h+softplus(W[h,c])*tanh(A), fused semi-implicit4substeps. Belief dp/dt=hazard(h,c)*(1-capture)*(destination(h,c)-p). Observation update uses full likelihood6+relation3+reliability+capture+log_gap. Movement report replaces likelihood; signed constraints supported by tested generic observation function.',
        train='All training questions/conditions, NLL, Adam lr0.003, fixed12epochs,batch256,gradclip1. Best validation NLL checkpoint per method/seed, no test selection. Hidden16. No_hflow holds h between observations but still propagates belief; input_decay removes h from conductance; time_gru holds GRU h and propagates belief.',
        baselines='Latest-report posterior and fixed-hazard probability filter; hazard selected from0,.01,.03,.1,.3,1,3 on TRAIN NLL. Prior estimated from train target labels. Full-information oracle is privileged upper bound, never a fair competitor.',
        primary='Test NLL and Brier for proper scoring; accuracy and10-bin ECE secondary. Pair comparisons bootstrap whole stories after averaging seeds. Report hflow removal with same weights plus16step integration sensitivity.',
        fairness='All methods share empirical train-label prior; all neural models use same inputs, hidden width, optimizer schedule and readout. Parameter counts and state bytes differ, disclosed; no equal-parameter or equal-FLOP claim. Fixed-rule filters have less state. No claim of novelty from using a standard hybrid filter.',
        limits='Synthetic public task, simple movement generator, perfect symbolic parsing. Negation/confirmation tested as unit probes, not native qa1 benchmark events. No real-language extraction validation, no finite archival-storage benefit claim.'))
    path=ROOT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=json.loads(json.dumps(v)) for k,v in spec.items()):raise RuntimeError('Frozen protocol changed')
    else:atomic_json(path,dict(**spec,created_at=utc_now(),environment=runtime_info()))
    return spec


def main():
    torch.set_num_threads(2);ROOT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(ROOT/'run.lock'),timeout=0):
        if (ROOT/'complete.json').exists():log('Completed frozen results preserved');return
        rows,hashes=prepare();spec=freeze(rows,hashes)
        (ROOT/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test']
        assert not ({r['story_hash'] for r in train}&{r['story_hash'] for r in val})
        atomic_json(ROOT/'split_audit.json',dict(rows={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
            train_test_exact_story_overlap=len({r['story_hash'] for r in train}&{r['story_hash'] for r in test})))
        prior=np.bincount([r['label'] for r in train],minlength=6).astype(float);prior/=prior.sum()
        ytrain=torch.tensor([r['label'] for r in train]);grid=[]
        for rate in (0.,.01,.03,.1,.3,1.,3.):grid.append(dict(rate=rate,**metrics(baseline(train,rate,prior),ytrain)))
        best_rate=min(grid,key=lambda r:r['nll'])['rate'];atomic_json(ROOT/'filter_fit.json',dict(prior=prior.tolist(),grid=grid,selected_rate=best_rate))
        outputs=[];history=[];diagnostics=[]
        def save_prediction(name,seed,p,seconds,params,state_bytes):
            for i,r in enumerate(test):outputs.append(dict(id=r['id'],story=r['story'],condition=r['condition'],label=r['label'],method=name,seed=seed,probabilities=p[i].tolist()))
            for condition in CONDITIONS:
                ids=[i for i,r in enumerate(test) if r['condition']==condition];y=torch.tensor([test[i]['label'] for i in ids])
                history.append(dict(method=name,seed=seed,condition=condition,questions=len(ids),**metrics(p[ids],y),
                    parameters=params,persistent_slot_bytes=state_bytes,test_seconds_all_conditions=seconds))
            atomic_json(ROOT/'summary.json',history)
            (ROOT/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs))
        for name,rate in [('latest_report',0.),('fixed_hazard',best_rate)]:
            start=time.perf_counter();p=baseline(test,rate,prior);save_prediction(name,None,p,time.perf_counter()-start,0,32)
        oracle=torch.nn.functional.one_hot(torch.tensor([r['label'] for r in test]),6).float()
        save_prediction('oracle_full_information',None,oracle,0.,0,0)
        for kind in KINDS:
            for seed in SEEDS:
                directory=ROOT/'checkpoints'/f'{kind}-{seed}';directory.mkdir(parents=True,exist_ok=True)
                model=LiquidBelief(kind)
                if not (directory/'complete.json').exists():
                    torch.manual_seed(seed);model=LiquidBelief(kind);model.prior.copy_(torch.tensor(prior,dtype=torch.float32));optimizer=torch.optim.Adam(model.parameters(),lr=.003)
                    rng=random.Random(seed);best=float('inf');logs=[]
                    for epoch in range(1,13):
                        start=time.perf_counter();order=list(train);rng.shuffle(order);losses=[];model.train()
                        for j in range(0,len(order),256):
                            x,y=tensors(order[j:j+256]);p=model(**x)
                            loss=-p[torch.arange(len(y)),y].clamp_min(1e-8).log().mean()
                            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
                        model.eval();pv=predict(model,val);score=metrics(pv,torch.tensor([r['label'] for r in val]))['nll']
                        if score<best:best=score;torch.save(model.state_dict(),directory/'best.pt');best_epoch=epoch
                        logs.append(dict(epoch=epoch,train_nll=float(np.mean(losses)),validation_nll=score,seconds=time.perf_counter()-start,rss_mb=rss_mb()))
                        atomic_json(directory/'epochs.json',logs)
                    atomic_json(directory/'complete.json',dict(best_epoch=best_epoch,best_validation_nll=best,parameters=sum(p.numel() for p in model.parameters())))
                model.load_state_dict(torch.load(directory/'best.pt',weights_only=True));model.eval()
                start=time.perf_counter();p=predict(model,test);elapsed=time.perf_counter()-start
                save_prediction(kind,seed,p,elapsed,sum(x.numel() for x in model.parameters()),(16+6+INPUTS)*4+8)
                if kind=='ltc':
                    fine=predict(model,test,steps=16);knockout=copy.deepcopy(model);knockout.kind='no_hflow';ablated=predict(knockout,test)
                    for condition in CONDITIONS:
                        ids=[i for i,r in enumerate(test) if r['condition']==condition];y=torch.tensor([test[i]['label'] for i in ids])
                        diagnostics.append(dict(seed=seed,condition=condition,fine16=metrics(fine[ids],y),same_weight_no_hflow=metrics(ablated[ids],y),
                            mean_probability_change_no_hflow=float((p[ids]-ablated[ids]).abs().mean()),max_probability_change_steps=float((p[ids]-fine[ids]).abs().max())))
                log(f'{kind}/{seed} completed')
        atomic_json(ROOT/'diagnostics.json',diagnostics)
        atomic_json(ROOT/'complete.json',dict(completed=True,trained_models=15,test_prediction_records=len(outputs),api_calls=0,finished_at=utc_now()))


if __name__=='__main__':main()
