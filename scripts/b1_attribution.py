"""B1-A calibration then locked B1 evaluation; no training or API calls.

python scripts/b1_attribution.py --stage all
Completed records are hash-verified and resumed, never silently replaced.
"""
import argparse,copy,gzip,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import atomic_json
from src.provenance import snapshot_source,runtime_info
from src.memory.b1_interleaved import generate,audit_environment,PRESSURES
from src.memory.b1_policy import frozen_head
from src.memory.b1_attribution import AttributionPolicy,METHODS,replacement_summary,matching_loss
from src.evaluation.b1_interleaved import run,audit
from scripts.ame_validation_batches import read,sha
from scripts.b1_sanity_check import read_gzip

DYNAMICS=('stable','slow_drift','rapid_shift')
SEEDS=(11,23,42)
GRID=tuple(round(i*.025,3) for i in range(1,21))+(.6,.8,1.)


def zipped(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt',encoding='utf-8',compresslevel=6) as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    temp.replace(path)


def execute(row,method,seed,heads,admission=1.):
    head=heads[seed] if method in ('adaptive_evolution','static_state_adaptive') else None
    policy=AttributionPolicy(method,row['capacity'],seed,head,admission)
    result=run(row,method,seed,head,policy=policy)
    result['metrics'].update(dynamics=row['dynamics'],**replacement_summary(result))
    # Diagnostics retain their separate interpretation: first-hit != durability.
    terminal=[q for q in result['queries'] if q['phase']==2 and q['target_query']][-8:]
    result['metrics']['final_eight_accuracy']=float(np.mean([q['correct'] for q in terminal]))
    return result


def register(out):
    if (out/'protocol.json').exists():
        p=read(out/'protocol.json')
        assert p['source']==snapshot_source()[0],'Source changed after registration; use a new output.'
        for name,h in p['input_hashes'].items():assert sha(ROOT/name)==h,name
        return p
    if out.exists() and any(out.iterdir()):raise FileExistsError('Nonempty output without protocol')
    out.mkdir(parents=True)
    source,archive=snapshot_source()
    inputs=[ROOT/'runs/liquid-consolidation-v2'/n for n in ('stories.json','vectors.json')]
    inputs += [ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt' for s in SEEDS]
    p=dict(source=source,archive=archive,registered_at=time.time(),runtime=runtime_info(),
        input_hashes={x.relative_to(ROOT).as_posix():sha(x) for x in inputs},
        capacity=8,capacity_scope='Primary B1 keeps validated capacity8. Capacity16 is not evaluated in this matrix.',
        pressures=dict(low=2,medium=8,high=16),dynamics=DYNAMICS,policy_seeds=SEEDS,methods=METHODS,
        calibration_worlds=list(range(71001,71005)),evaluation_worlds=list(range(72001,72025)),trajectory_seed_offset=300000,
        paired_design='Same world/query scheduler shared across pressures. Slow/rapid share topic shifts and clocks; stable holds topicA.',
        temporal_definition=dict(stable='No factual revisions or topic changes.',
            slow_drift='Shared A-B-A topic cues; one current-topic fact revised each round after a shift,8revisions total.',
            rapid_shift='Same A-B-A cues; all4current-topic facts revised in rounds0/2 after a shift,16revisions total.'),
        scope='Controlled factual revision speed, not a gradually drifting embedding or real natural-language agent.',
        admission_grid=GRID,match_selection='Per cell shared across seeds; minimize normalized squared errors in calibration mean eviction count and observed lifetime. Never use utility metrics.',
        match_tolerance=.10,match_evaluation='Both heldout cell mean relative errors<=10%; unmatched cells cannot support replacement-independent attribution.',
        lifetime='Includes right-censored observed episodes. With always-full storage, sum(lifetimes)=capacity*horizon, so mean lifetime and eviction count are algebraically coupled, not independent controls. Also report resident age and quantiles.',
        fixed_decay='Positive lambda changes scale only, never age order; replacement_matched adds an explicit seeded admission gate. This is NOT lambda-only tuning.',
        static_state='Same frozen head for insertion and same score; per-residency I/C/U/T frozen after insertion, including observation and feedback jumps. Reinsertion is a new residency. Additional static_uniform_state freezes identical I=1,C=1,U=0,T=0 for all bootstrap and new items to expose initialization lock-in artifacts.',
        metrics=['accuracy','stale_error','stable_retention','adaptation_lag','adaptation_success','eviction_count','mean_observed_lifetime','time_weighted_resident_age'],
        lag='Two consecutive overlapping4query windows>=75%; failure censored32; stable condition not applicable. First passage not sustained correctness.',
        statistics='Average 3policy seeds within each of24independent worlds, paired bootstrap5000 for95% CIs. Contrasts adaptive vs fixed/static/matched/uniform_static; sign-flip10000 with Holm correction across all primary contrasts/endpoints.',
        visualization='First heldout world, medium pressure, rapid shift, seed11, same preselected version1 records.',
        training_steps=0,api_calls=0,freeze='Use existing learned heads11/23/42 without refitting. Seeds include weight initialization and tie-breaking; deterministic controls do not create72independent worlds.',
        runs=9*24*3*len(METHODS))
    atomic_json(out/'protocol.json',p)
    return p


def environments(out,p,stories,vectors):
    path=out/'environments.json.gz'
    if path.exists():
        assert sha(path)==read(out/'environment_manifest.json')['sha256']
        return read_gzip(path)
    rows={}
    checks=[]
    for split,worlds in [('calibration',p['calibration_worlds']),('evaluation',p['evaluation_worlds'])]:
        rows[split]=[]
        for world in worlds:
            for dynamics in DYNAMICS:
                trio=[]
                for pressure in PRESSURES:
                    row=generate(stories,vectors,world,world+p['trajectory_seed_offset'],pressure,8,dynamics=dynamics)
                    check=audit_environment(row);checks.append(dict(split=split,dynamics=dynamics,**check))
                    rows[split].append(row);trio.append(row)
                stripped=lambda r:[e for e in r['events'] if not e['evaluation'].get('noise_observation',False)]
                assert stripped(trio[0])==stripped(trio[1])==stripped(trio[2])
    zipped(path,rows);atomic_json(out/'environment_manifest.json',dict(sha256=sha(path),audits=checks))
    return rows


def key(row):return row['dynamics']+'-'+row['pressure']


def calibration(out,p,rows,heads):
    locked=out/'calibration_lock.json'
    if locked.exists():
        result=read(locked)
        for f,h in result['files'].items():assert sha(out/f)==h
        return result['selected']
    folder=out/'calibration';folder.mkdir(exist_ok=True)
    selected={}
    for dynamics in DYNAMICS:
        for pressure in PRESSURES:
            cell=dynamics+'-'+pressure;path=folder/(cell+'.json')
            cellrows=[r for r in rows if key(r)==cell]
            if path.exists():
                record=read(path);assert record['protocol_sha256']==sha(out/'protocol.json')
                selected[cell]=record['selected'];continue
            raw=[];targets=[]
            for row in cellrows:
                for seed in SEEDS:
                    result=execute(row,'adaptive_evolution',seed,heads)
                    audit(row,result);targets.append(replacement_summary(result))
                    raw.append(dict(world_seed=row['world_seed'],policy_seed=seed,method='adaptive_evolution',**targets[-1]))
            average=lambda values:{k:float(np.mean([v[k] for v in values])) for k in targets[0]}
            target=average(targets);candidates=[]
            for probability in GRID:
                values=[]
                for row in cellrows:
                    for seed in SEEDS:
                        result=execute(row,'replacement_matched',seed,heads,probability)
                        audit(row,result);values.append(replacement_summary(result))
                        raw.append(dict(world_seed=row['world_seed'],policy_seed=seed,method='replacement_matched',admission=probability,**values[-1]))
                mean=average(values);candidates.append(dict(admission=probability,loss=matching_loss(mean,target),**mean))
            best=min(candidates,key=lambda c:(c['loss'],c['admission']))
            selected[cell]=dict(admission=best['admission'],target=target,calibrated=best)
            atomic_json(path,dict(protocol_sha256=sha(out/'protocol.json'),selected=selected[cell],candidates=candidates,raw_replacement_statistics=raw))
            print('Calibration locked:',cell,'admission=',best['admission'],flush=True)
    atomic_json(locked,dict(selected=selected,files={x.relative_to(out).as_posix():sha(x) for x in folder.glob('*.json')},selection_uses_task_metrics=False))
    return selected


def evaluate(out,p,rows,heads,selected):
    folder=out/'traces';folder.mkdir(exist_ok=True)
    manifest=out/'evaluation_records.jsonl';existing={}
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert sha(out/r['trace_file'])==r['sha256'];existing[r['trace_file']]=r
    total=p['runs'];done=0;start=time.time()
    for rownum,row in enumerate(rows):
        for seed in SEEDS:
            ordered=list(METHODS);rotation=(rownum+SEEDS.index(seed))%len(ordered);ordered=ordered[rotation:]+ordered[:rotation]
            for method in ordered:
                filename=f'traces/{row["world_seed"]}-{key(row)}-{seed}-{method}.json.gz';done+=1
                if filename in existing:continue
                path=out/filename
                if path.exists():raise FileExistsError('Uncommitted trace preserved; inspect before resuming: '+str(path))
                result=execute(row,method,seed,heads,selected[key(row)]['admission'])
                check=audit(row,result)
                # Fixed preselected world, all conditions/methods/seeds:189 exact replays.
                replay=row['world_seed']==p['evaluation_worlds'][0]
                if replay:
                    other=execute(copy.deepcopy(row),method,seed,heads,selected[key(row)]['admission'])
                    for field in ('trace','queries','lifetimes','episodes','acquisition'):assert result[field]==other[field],field
                    assert {k:v for k,v in result['metrics'].items() if k not in ('policy_seconds','timing_seconds')}=={k:v for k,v in other['metrics'].items() if k not in ('policy_seconds','timing_seconds')}
                result['metrics']['trace_file']=filename
                zipped(path,result)
                record=dict(trace_file=filename,sha256=sha(path),metrics=result['metrics'],audit=check,replayed=replay)
                with manifest.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                existing[filename]=record
                if done%63==0:print(f'Evaluation {done}/{total}; elapsed {time.time()-start:.1f}s',flush=True)
    records=list(existing.values());assert len(records)==total
    metrics=[r['metrics'] for r in records];fairness=[]
    for row in rows:
        rr=[m for m in metrics if m['environment_sha256']==row['environment_sha256']]
        assert len(rr)==len(METHODS)*len(SEEDS)
        for field in ('input_sha256','initial_sha256','query_sha256','capacity','observation_opportunities','queries'):
            assert len({m[field] for m in rr})==1,field
        fairness.append(dict(world_seed=row['world_seed'],cell=key(row),passed=True,policies=len(rr)))
    atomic_json(out/'metrics.json',metrics);atomic_json(out/'fairness_audit.json',fairness)
    atomic_json(out/'evaluation_complete.json',dict(runs=len(records),audited=len(records),exact_replays=sum(r['replayed'] for r in records),training_steps=0,api_calls=0,
        files={f:sha(out/f) for f in ['evaluation_records.jsonl','metrics.json','fairness_audit.json','calibration_lock.json','environments.json.gz']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='runs/ame-validation-v2/B1-attribution-v1')
    parser.add_argument('--stage',choices=['all','register','calibrate','evaluate','report'],default='all');args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output outside workspace')
    p=register(out)
    if args.stage=='register':return
    if args.stage=='report':
        from scripts.b1_attribution_report import build
        build(out);return
    stories=read(ROOT/'runs/liquid-consolidation-v2/stories.json');vectors=read(ROOT/'runs/liquid-consolidation-v2/vectors.json')
    rows=environments(out,p,stories,vectors);heads={s:frozen_head(ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt') for s in SEEDS}
    selected=calibration(out,p,rows['calibration'],heads)
    if args.stage=='calibrate':return
    evaluate(out,p,rows['evaluation'],heads,selected)
    from scripts.b1_attribution_report import build
    build(out)


if __name__=='__main__':main()
