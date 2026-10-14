"""Sequential B2-B7 mini -> calibration -> heldout validation. Frozen B1 core."""
import argparse,copy,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import runtime_info,snapshot_source
from src.memory.b1_policy import frozen_head
from src.memory.validation_matrix import MatrixPolicy,BASELINES,generate
from src.evaluation.b1_interleaved import run,audit
from scripts.ame_validation_batches import read,sha
from scripts.b1_attribution import zipped
from scripts.b1_sanity_check import read_gzip

SEEDS=(11,23,42)
CORE_FILES=['src/memory/b1_policy.py','src/memory/b1_attribution.py']


def specifications(batch):
    common=dict(capacity=8,noise_count=16,rounds=4,repetitions=1)
    if batch=='B2':return [(f'capacity{c}',dict(common,capacity=c,noise_count=2*c,repetitions=4)) for c in (4,8,16,32,64)]
    if batch=='B3':return [('cyclic',dict(common,cyclic_probe=True,versions=False))]
    if batch=='B4':return [(name,dict(common,feedback_noise=noise,misleading=misleading)) for name,noise,misleading in [('clean',0.,False),('noise10',.1,False),('noise25',.25,False),('noise50',.5,False),('misleading_importance',0.,True)]]
    if batch=='B5':return [('version_update',common)]
    if batch in ('B6','B7'):return [('rapid_high',common)]
    raise ValueError(batch)


def methods(batch):
    return BASELINES+({'B6':('no_importance','no_consistency','no_usage','no_time'),
                       'B7':('event_only','discrete_euler','query_conditioned')}.get(batch,()))


def execute(row,method,seed,heads,admission=1.,minimum_age=0.):
    needs_head=method not in ('fixed_decay','frequency','lru','replacement_matched','adaptive_rule')
    p=MatrixPolicy(method,row['capacity'],seed,heads[seed] if needs_head else None,admission,minimum_age)
    result=run(row,method,seed,policy=p)
    for f in result['trace']:
        if f['event']['kind']=='feedback':f['delivered_feedback']=1-f['feedback_value'] if f['event'].get('feedback_flip') else f['feedback_value']
    m=result['metrics'];lives=result['lifetimes'];qq=result['queries']
    pre=[q for q in qq if q.get('pre_reminder')];post=[q for q in qq if q['phase']==2 and q['target_query'] and not q.get('pre_reminder')]
    from src.evaluation.b1_interleaved import lag_episode
    first_reminder=next((f['event']['time'] for f in result['trace'] if f['event']['kind']=='observe' and f['evaluation']['phase']==2 and f['event']['key'] in row['a_keys']),None)
    after=[q for q in post if first_reminder is not None and q['time']>=first_reminder]
    m.update(condition=row['condition'],batch=row['batch'],mean_observed_lifetime=float(np.mean([l['lifetime'] for l in lives])),
        lifetime_uncensored_count=sum(not l['right_censored'] for l in lives),lifetime_censor_fraction=float(np.mean([l['right_censored'] for l in lives])),
        pre_reminder_recall=float(np.mean([q['correct'] for q in pre])) if pre else None,
        post_reminder_recovery_lag=lag_episode(after,first_reminder)['lag_queries'] if after else None,
        post_reminder_recovery_success=lag_episode(after,first_reminder)['recovered'] if after else None,
        state_payload_float64_bytes=32*len(p.items),state_payload_max_bytes=32*m['peak_slots'],
        model_parameter_bytes=int(sum(x.nbytes for x in p.coefficients.values())) if p.coefficients is not None else 0,
        policy_update_seconds=m['timing_seconds']['update_state'],admission=admission if method=='replacement_matched' else None,
        minimum_age=minimum_age if method=='replacement_matched' else None,
        auxiliary_ledger_entries=len(p.ledger),auxiliary_query_history_entries=len(p.demand))
    return result


def pooled_replacement(results):
    times=[];uncensored=[];evictions=[];means=[];censors=[]
    for r in results:
        times.extend(l['lifetime'] for l in r['lifetimes']);uncensored.extend(l['lifetime'] for l in r['lifetimes'] if not l['right_censored'])
        evictions.append(r['metrics']['eviction_count']);means.append(r['metrics']['mean_observed_lifetime']);censors.append(r['metrics']['lifetime_censor_fraction'])
    return dict(eviction_count=float(np.mean(evictions)),
                mean_observed_lifetime=float(np.mean(means)),
                lifetime_censor_fraction=float(np.mean(censors)),
                lifetimes=times,uncensored_lifetimes=uncensored)


def ks(a,b):
    if not a or not b:return 0. if not a and not b else 1.
    a=np.sort(a);b=np.sort(b);v=np.unique(np.r_[a,b])
    return float(np.max(np.abs(np.searchsorted(a,v,side='right')/len(a)-np.searchsorted(b,v,side='right')/len(b))))


def match_distance(a,b):
    e=abs(a['eviction_count']-b['eviction_count'])/max(b['eviction_count'],1.)
    l=abs(a['mean_observed_lifetime']-b['mean_observed_lifetime'])/max(b['mean_observed_lifetime'],1.)
    d=ks(a['lifetimes'],b['lifetimes']);u=ks(a['uncensored_lifetimes'],b['uncensored_lifetimes'])
    c=abs(a['lifetime_censor_fraction']-b['lifetime_censor_fraction'])
    return dict(eviction_relative_error=e,lifetime_mean_relative_error=l,lifetime_ks=d,uncensored_lifetime_ks=u,censor_fraction_error=c,
                loss=e*e+l*l+d*d+u*u+c*c,matched=bool(e<=.1 and l<=.1 and d<=.1 and u<=.1 and c<=.1))


def row_for(stories,vectors,batch,condition,config,world):
    r=generate(stories,vectors,world,world+400000,**config);r.update(batch=batch,condition=condition)
    from src.memory.b1_environment import digest
    r['environment_sha256']=digest(r);return r


def save_result(out,name,row,result,replay=False):
    check=audit(row,result);file='traces/'+name+'.json.gz';result['metrics']['trace_file']=file
    zipped(out/file,result)
    return dict(trace_file=file,sha256=sha(out/file),metrics=result['metrics'],audit=check,replayed=replay)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--batch',required=True,choices=['B2','B3','B4','B5','B6','B7']);ap.add_argument('--output');ap.add_argument('--mini-only',action='store_true');ap.add_argument('--calibrate-only',action='store_true');args=ap.parse_args()
    batch=args.batch;out=ROOT/(args.output or f'runs/ame-validation-v3/{batch}')
    if (out/'completion.json').exists():
        verify(out);print(batch,'completed evidence verified; no rerun');return
    out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    core={f:sha(ROOT/f) for f in CORE_FILES};source,archive=snapshot_source()
    old=read(ROOT/'runs/ame-validation-v2/B1-attribution-v1/protocol.json')
    import zipfile,hashlib
    with zipfile.ZipFile(ROOT/old['archive']) as z:
        assert all(hashlib.sha256(z.read(f)).hexdigest()==h for f,h in core.items())
    start=80000+1000*int(batch[1:]);config=specifications(batch);allmethods=methods(batch)
    protocol=dict(batch=batch,source=source,archive=archive,core_hashes=core,runtime=runtime_info(),conditions=config,methods=allmethods,
        world_seeds=list(range(start+100,start+124)),calibration_world_seeds=list(range(start+10,start+12)),mini_world_seeds=[start+1],policy_seeds=SEEDS,
        trajectory_seed_offset=400000,initial_memory='Same4observations for all methods/capacities; never truncated differently by policy.',
        matching='Independent calibration of admission probability and minimum resident age; select ONLY count/lifetime distribution/censoring errors. Exact distribution equality is NOT guaranteed; heldout gate requires each relative/KS/censor error<=.10.',
        match_grid=dict(admission=[.025,.05,.1,.2,.35,.5,.75,1.],minimum_age=[0.,24.,60.]),
        causality='All methods share observation/query/feedback-corruption tapes; labels only in evaluator. Feedback received after answering; correction may differ across methods because answers differ.',
        B2='Noise keys=2*capacity. Fixed12core facts, 128noise+128core arrivals per phase; same query/key clocks across capacities. This is joint capacity/working-set scaling, not fixed-workload scaling.',
        B3='Four returning-topic queries occur before any reminder in phase2; no factual revisions. Post-reminder lag starts at first topicA observation.',
        B4='Bernoulli correctness-feedback flips shared across methods; misleading importance flips topic context cue only. This does not corrupt observation content. Replacement control calibrated on CLEAN ONLY; same admission and minimum-age parameters used in every corruption condition.',
        B5='Explicit observed key/version conflicts; evaluator-only current truth. Acquisition records separate versions with no eligible query and right-censored failures.',
        B6='Remove a state by pinning it to neutral score value after every operation; no_time removes T score/state, not the time integration of I/C/U.',
        B7='Full exact-flow already reads query features. Event-only removes inter-event flow but keeps observation/feedback jumps. Euler integrates the same ODE with stability-bounded steps. Query-conditioned additionally holds the latest query topic forward; not a pure time-only comparison.',
        statistics='24worlds, three seeds averaged within world. Paired world bootstrap5000; sign-flip10000; Holm within batch across preregistered primary comparisons. No claim that plotting method operating points proves tunable Pareto frontier.',
        metrics=['accuracy','stale_error','stable_retention','adaptation_lag'],trace='Full events, flow_state, resident_state, true and delivered feedback, lifetimes with censor flags.',
        resource='CPU policy call/update timing excludes audit/serialization. 32bytes/item is theoretical4float64 payload, not measured incremental heap: baseline Item also allocates these fields. Auxiliary ledgers grow outside content capacity and are reported.',
        training_steps=0,api_calls=0,input_hashes=old['input_hashes'])
    if (out/'protocol.json').exists():
        previous=read(out/'protocol.json');assert previous['source']==source and previous['core_hashes']==core
        protocol=previous
    else:atomic_json(out/'protocol.json',protocol)
    for f,h in protocol['input_hashes'].items():assert sha(ROOT/f)==h
    stories=read(ROOT/'runs/liquid-consolidation-v2/stories.json');vectors=read(ROOT/'runs/liquid-consolidation-v2/vectors.json')
    heads={s:frozen_head(ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt') for s in SEEDS}
    if not (out/'mini_complete.json').exists():
        checks=[]
        for condition,c in config:
            row=row_for(stories,vectors,batch,condition,c,start+1)
            for method in allmethods:
                r=execute(row,method,11,heads,.2);rr=execute(copy.deepcopy(row),method,11,heads,.2)
                for f in ('trace','queries','lifetimes','episodes','acquisition'):assert r[f]==rr[f]
                checks.append(save_result(out,f'mini-{condition}-{method}',row,r,True))
        atomic_json(out/'mini_complete.json',dict(passed=True,records=checks));print(batch,'mini passed',len(checks),'policies',flush=True)
    if args.mini_only:return
    # Generate and save the paired environment matrix before inspecting outcomes.
    rows={split:[row_for(stories,vectors,batch,condition,c,w) for w in protocol[seedkey] for condition,c in config]
          for split,seedkey in [('calibration','calibration_world_seeds'),('evaluation','world_seeds')]}
    if not (out/'environments.json.gz').exists():zipped(out/'environments.json.gz',rows)
    if not (out/'calibration_lock.json').exists():
        selected={};records=[]
        for condition,_ in config:
            if batch=='B4' and condition!='clean':
                selected[condition]=dict(selected['clean'],calibration_match=None,transferred_from='clean')
                records.append(dict(condition=condition,selected=selected[condition],note='No noise-specific calibration; heldout matching audited separately.'))
                continue
            tapes=[r for r in rows['calibration'] if r['condition']==condition]
            target=pooled_replacement(execute(r,'adaptive_evolution',s,heads) for r in tapes for s in SEEDS);candidates=[]
            for probability in protocol['match_grid']['admission']:
                for age in protocol['match_grid']['minimum_age']:
                    results=(execute(r,'replacement_matched',s,heads,probability,age) for r in tapes for s in SEEDS)
                    candidate=pooled_replacement(results);distance=match_distance(candidate,target)
                    candidates.append(dict(admission=probability,minimum_age=age,statistics=candidate,**distance))
            winner=min(candidates,key=lambda c:(c['loss'],c['admission'],c['minimum_age']))
            selected[condition]=dict(admission=winner['admission'],minimum_age=winner['minimum_age'],calibration_match=winner['matched'])
            records.append(dict(condition=condition,target=target,candidates=candidates,selected=selected[condition]))
            print(batch,'calibration',condition,selected[condition],flush=True)
        atomic_json(out/'calibration_lock.json',dict(selected=selected,records=records,uses_task_metrics=False))
    if args.calibrate_only:
        print(batch,'mini and calibration locked; formal execution not started',flush=True);return
    selected=read(out/'calibration_lock.json')['selected'];path=out/'records.jsonl';done={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert sha(out/r['trace_file'])==r['sha256'];done[r['trace_file']]=r
    total=len(rows['evaluation'])*len(SEEDS)*len(allmethods);count=0
    for n,row in enumerate(rows['evaluation']):
        for seed in SEEDS:
            order=list(allmethods);j=(n+SEEDS.index(seed))%len(order);order=order[j:]+order[:j]
            for method in order:
                name=f'{row["world_seed"]}-{row["condition"]}-{seed}-{method}';filename='traces/'+name+'.json.gz';count+=1
                if filename in done:continue
                choice=selected[row['condition']];r=execute(row,method,seed,heads,choice['admission'],choice['minimum_age'])
                record=save_result(out,name,row,r)
                with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                done[filename]=record
                if count%72==0:print(batch,f'{count}/{total}',flush=True)
    metrics=[r['metrics'] for r in done.values()];assert len(metrics)==total
    for row in rows['evaluation']:
        rr=[m for m in metrics if m['environment_sha256']==row['environment_sha256']]
        for field in ('initial_sha256','input_sha256','query_sha256','capacity','observation_opportunities','queries'):
            assert len({m[field] for m in rr})==1
    atomic_json(out/'metrics.json',metrics)
    from scripts.validation_matrix_report import build
    build(out,rows['evaluation'])
    atomic_json(out/'completion.json',dict(batch=batch,runs=total,mini_passed=True,training_steps=0,api_calls=0,
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'}))
    print(batch,'COMPLETE',total,flush=True)


def verify(out):
    for f,h in read(out/'completion.json')['files'].items():assert sha(out/f)==h,f


if __name__=='__main__':main()
