"""C4 execution; immutable snapshots and fixed-frequency prefix controls."""
import argparse,copy,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.memory.stress_characterization import stream,execute
from src.memory.characterization import policies
from src.memory.b1_policy import frozen_head
from src.config import atomic_json
from src.provenance import snapshot_source,runtime_info
from scripts.ame_validation_batches import read,sha
from scripts.b1_attribution import zipped
from scripts.b1_sanity_check import read_gzip
OUT=ROOT/'runs/ame-characterization-v1/C4';SEEDS=(11,23,42);HEADS=None

def init(heads):
    global HEADS;HEADS=heads

def worker(row,method,seed):
    result,check=execute(row,method,seed,HEADS[seed]);path=OUT/'traces'/f"{row['world_seed']}-{row['condition']}-{seed}-{method}.json.gz";zipped(path,result)
    return dict(trace_file=path.relative_to(OUT).as_posix(),sha256=sha(path),metrics=dict(result['metrics'],trace_file=path.relative_to(OUT).as_posix()),audit=check)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mini-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('mini','traces'):(OUT/folder).mkdir(exist_ok=True)
    if (OUT/'completion.json').exists():
        for p,h in read(OUT/'completion.json')['files'].items():assert sha(OUT/p)==h
        print('C4 completed evidence verified');return
    old=read(ROOT/'runs/ame-validation-v3/B8/protocol.json')
    for p,h in old['core_hashes'].items():assert sha(ROOT/p)==h,p
    files=['src/memory/stress_characterization.py','src/memory/characterization.py','scripts/characterization_stress.py','scripts/characterization_report.py',
        'src/memory/b1_policy.py','src/memory/b1_attribution.py','src/memory/validation_matrix.py','src/evaluation/b1_interleaved.py']
    inputs={**old['checkpoints'],**{p:sha(ROOT/p) for p in files}}
    if not (OUT/'protocol.json').exists():
        source,archive=snapshot_source()
        atomic_json(OUT/'protocol.json',dict(batch='C4',source=source,archive=archive,runtime=runtime_info(),frozen_files=inputs,
            methods=policies('C4'),world_seeds=list(range(94100,94124)),mini_world_seed=94001,policy_seeds=SEEDS,capacities=[8,16],lengths=[500,1000,5000],
            events='Initial knowledge37events then alternate topic A/B every97events. Same world gives exact identical input prefixes across lengths and capacities. Interference16keys,4stable+4active. Each active fact switches to another public value at its first observation in every post-initial phase; continued version conflicts, no future truth exposure.',
            adaptation='Mean over fully observed shift episodes, terminal partial episode excluded and saved. Same length16target-query window per complete phase. Lag threshold unchanged.',
            collapse='Descriptive indicator: two consecutive scored100-event windows with accuracy<.1; raw windows always reported, not a universal failure definition.',
            drift='Finite states, maxI/C/U/T,100-event state means, true obsolete resident fraction. Auxiliary ledgers outside content slots reported.',
            training_steps=0,api_calls=0))
    for p,h in read(OUT/'protocol.json')['frozen_files'].items():assert sha(ROOT/p)==h,p
    stories=read(ROOT/'runs/liquid-consolidation-v2/stories.json');vectors=read(ROOT/'runs/liquid-consolidation-v2/vectors.json')
    heads={s:frozen_head(ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt') for s in SEEDS}
    if not (OUT/'mini_complete.json').exists():
        checks=[]
        for c in (8,16):
            long=stream(stories,vectors,94001,c,5000)
            for n in (500,1000,5000):
                row=stream(stories,vectors,94001,c,n);assert row['events']==long['events'][:n]
                for method in policies('C4'):
                    a,check=execute(row,method,11,heads[11]);b,_=execute(copy.deepcopy(row),method,11,heads[11])
                    for f in ('trace','queries','episodes','lifetimes'):assert a[f]==b[f]
                    path=OUT/'mini'/f'{c}-{n}-{method}.json.gz';zipped(path,a);checks.append(dict(file=path.relative_to(OUT).as_posix(),sha256=sha(path),**check))
        atomic_json(OUT/'mini_complete.json',dict(passed=True,exact_prefix=True,exact_replay=True,records=checks));print('C4 mini passed',len(checks),flush=True)
    if args.mini_only:return
    env=OUT/'environments.json.gz'
    if not env.exists():zipped(env,[stream(stories,vectors,w,c,n) for w in range(94100,94124) for c in (8,16) for n in (500,1000,5000)])
    rows=read_gzip(env);manifest=OUT/'records.jsonl';done={}
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert sha(OUT/r['trace_file'])==r['sha256'];done[r['trace_file']]=r
    jobs=[]
    for row in rows:
        for seed in SEEDS:
            for method in policies('C4'):
                if f"traces/{row['world_seed']}-{row['condition']}-{seed}-{method}.json.gz" not in done:jobs.append((row,method,seed))
    with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(heads,)) as pool:
        it=iter(jobs);pending=set()
        def submit():
            job=next(it,None)
            if job is not None:pending.add(pool.submit(worker,*job))
        for _ in range(8):submit()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.remove(future);r=future.result();done[r['trace_file']]=r
                with manifest.open('a',encoding='utf-8') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                submit()
                if len(done)%72==0:print('C4',len(done),'/3456',flush=True)
    assert len(done)==3456;atomic_json(OUT/'metrics.json',[r['metrics'] for r in done.values()])
    from scripts.characterization_report import build
    build(OUT)
    atomic_json(OUT/'completion.json',dict(runs=3456,mini_passed=True,files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}))
    print('C4 COMPLETE',flush=True)

if __name__=='__main__':main()
