"""C1/C2/C4: freeze, mini exact replay, shared worlds, resumable formal batch."""
import argparse,copy,json,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.memory.characterization import environment,execute,policies,StrengthPolicy
from src.memory.validation_matrix import MatrixPolicy
from src.memory.b1_policy import frozen_head
from src.evaluation.b1_interleaved import run
from src.config import atomic_json
from src.provenance import snapshot_source,runtime_info
from scripts.ame_validation_batches import read,sha
from scripts.b1_attribution import zipped
from scripts.b1_sanity_check import read_gzip
SEEDS=(11,23,42)
HEADS=None

def init(heads):
    global HEADS;HEADS=heads

def worker(out,row,method,seed):
    result,check=execute(row,method,seed,HEADS[seed]);p=Path(out)/'traces'/f"{row['world_seed']}-{row['condition']}-{seed}-{method}.json.gz"
    zipped(p,result)
    return dict(trace_file=p.relative_to(out).as_posix(),sha256=sha(p),metrics=result['metrics'],audit=check)

def spec(batch):
    if batch=='C4':return [(c,'rapid_high',n) for c in (8,16) for n in (500,1000,5000)]
    return [(c,s,None) for c in (8,16) for s in ('mixed','rapid_high')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',choices=['C1','C2','C4'],required=True);ap.add_argument('--mini-only',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    batch=args.batch;out=ROOT/'runs/ame-characterization-v1'/batch;out.mkdir(parents=True,exist_ok=True)
    for folder in ('mini','traces'):(out/folder).mkdir(exist_ok=True)
    if (out/'completion.json').exists():
        for p,h in read(out/'completion.json')['files'].items():assert sha(out/p)==h,p
        print(batch,'completed, verified');return
    old=read(ROOT/'runs/ame-validation-v3/B8/protocol.json')
    for p,h in old['core_hashes'].items():assert sha(ROOT/p)==h,p
    deps=['src/memory/characterization.py','scripts/characterization.py','scripts/characterization_report.py',
        'src/memory/b1_policy.py','src/memory/b1_attribution.py','src/memory/validation_matrix.py','src/evaluation/b1_interleaved.py']
    inputs={**old['checkpoints'],**{p:sha(ROOT/p) for p in deps}}
    start=91000 if batch=='C1' else 92000 if batch=='C2' else 94000
    if not (out/'protocol.json').exists():
        source,archive=snapshot_source()
        atomic_json(out/'protocol.json',dict(batch=batch,source=source,archive=archive,runtime=runtime_info(),frozen_files=inputs,
            world_seeds=list(range(start+100,start+124)),mini_world_seed=start+1,policy_seeds=SEEDS,methods=policies(batch),conditions=spec(batch),
            grid=[0,.25,.5,.75,1,1.5,2],training_steps=0,api_calls=0,
            intervention='Only multiply the I/C/U inter-event rates. T ages normally. Insertion, observation/feedback jumps, score, head weights, and retrieval unchanged. alpha0 is not frozen-state memory.',
            C2='Only one inter-event rate is enabled in I/C/U-only variants; all other event jumps and T remain. Full family swept at same alpha.',
            environments='Mixed:4stable+4active,8noise,no version revisions; rapid_high:4stable+4active,16noise,observed revisions. Both A-B-A context. Settings differ jointly, not an isolated volatility effect.',
            C4='Exact500/1000/5000events allocated over warmup and three phases. Event-index clock; skipped-tail truth recomputed from retained observations. Different horizons are not identical-prefix trajectories.',
            adaptation='1-capped first qualifying-window query lag/window length, mean over shifts. Two overlapping length4 windows each>=3correct; not sustained success guarantee.',
            statistics='24world paired inference, average3policyseeds within world; bootstrap5000,signflip10000,Holm within batch. All alpha-pair endpoint tests for C1; empirical front with bootstrap membership, not population Pareto proof.',
            capacity='Same4initial facts, same input/query stream per condition, same1retrieval and eviction opportunity. Full traces and lifetimes retained. Auxiliary ID ledger is outside content capacity.'))
    protocol=read(out/'protocol.json')
    for p,h in protocol['frozen_files'].items():assert sha(ROOT/p)==h,p
    stories=read(ROOT/'runs/liquid-consolidation-v2/stories.json');vectors=read(ROOT/'runs/liquid-consolidation-v2/vectors.json')
    heads={s:frozen_head(ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt') for s in SEEDS}
    if not (out/'mini_complete.json').exists():
        checks=[]
        for cap,setting,length in spec(batch):
            row=environment(stories,vectors,start+1,cap,setting,length)
            for name in policies(batch):
                a,check=execute(row,name,11,heads[11]);b,_=execute(copy.deepcopy(row),name,11,heads[11])
                for field in ('trace','queries','lifetimes','episodes','window_metrics'):assert a[field]==b[field],(name,field)
                if name=='full_1':
                    reference=run(row,'adaptive_evolution',11,policy=MatrixPolicy('adaptive_evolution',cap,11,heads[11]))
                    for x,y in zip(a['trace'],reference['trace']):
                        assert x['resident']==y['resident'] and x['action']==y['action'] and x['answer']==y['answer'],'alpha1 parity'
                file=f'mini/{row["condition"]}-{name}.json.gz';zipped(out/file,a)
                checks.append(dict(condition=row['condition'],method=name,trace_file=file,sha256=sha(out/file),**check))
            if cap==8 and setting=='rapid_high':zipped(out/'sample_environment.json.gz',row)
        atomic_json(out/'mini_complete.json',dict(passed=True,exact_replay=True,alpha1_frozen_parity=True,records=checks))
        print(batch,'mini passed',len(checks),flush=True)
    if args.mini_only:return
    envfile=out/'environments.json.gz'
    if not envfile.exists():zipped(envfile,[environment(stories,vectors,w,c,s,n) for w in protocol['world_seeds'] for c,s,n in spec(batch)])
    rows=read_gzip(envfile);manifest=out/'records.jsonl';done={}
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert sha(out/r['trace_file'])==r['sha256'];done[r['trace_file']]=r
    jobs=[]
    for row in rows:
        for seed in SEEDS:
            for name in policies(batch):
                filename=f"traces/{row['world_seed']}-{row['condition']}-{seed}-{name}.json.gz"
                if filename not in done:jobs.append((str(out),row,name,seed))
    total=len(rows)*len(SEEDS)*len(policies(batch));start_time=time.time();it=iter(jobs)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(heads,)) as pool:
        pending=set()
        def submit():
            job=next(it,None)
            if job is not None:pending.add(pool.submit(worker,*job))
        for _ in range(args.workers*2):submit()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.remove(future);record=future.result()
                with manifest.open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                done[record['trace_file']]=record;submit()
                if len(done)%72==0:print(batch,len(done),'/',total,round(time.time()-start_time),'seconds',flush=True)
    assert len(done)==total
    metrics=[r['metrics'] for r in done.values()]
    for row in rows:
        mm=[m for m in metrics if m['environment_sha256']==row['environment_sha256']]
        assert len(mm)==len(SEEDS)*len(policies(batch))
        for field in ('initial_sha256','input_sha256','query_sha256','capacity','queries','observation_opportunities'):
            assert len({m[field] for m in mm})==1,field
    atomic_json(out/'metrics.json',metrics)
    from scripts.characterization_report import build
    build(out)
    atomic_json(out/'completion.json',dict(batch=batch,runs=total,mini_passed=True,training_steps=0,
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'}))
    print(batch,'COMPLETE',total,flush=True)

if __name__=='__main__':main()
