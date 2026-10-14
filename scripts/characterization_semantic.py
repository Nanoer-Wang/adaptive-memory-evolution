"""C3 semantic-only addressing, public quote provenance, no API or tuning."""
import argparse,copy,itertools,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.memory.semantic_characterization import DOMAINS,generate,execute,audit
from src.models.content_encoder import ContentEncoder
from src.memory.b1_policy import frozen_head
from src.provenance import snapshot_source,runtime_info
from src.config import atomic_json
from scripts.ame_validation_batches import read,sha
from scripts.b1_attribution import zipped
from scripts.b1_sanity_check import read_gzip
OUT=ROOT/'runs/ame-characterization-v1/C3';SEEDS=(11,23,42)
METHODS=('fixed_decay','frequency','adaptive_evolution')
TASKS=('preference_update','contradictory_preference','preference_revival','temporary_task')
HEADS=None

def init(heads):
    global HEADS;HEADS=heads

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('mini','traces'):(OUT/folder).mkdir(exist_ok=True)
    old=read(ROOT/'runs/ame-validation-v3/B8/protocol.json')
    for p,h in old['core_hashes'].items():assert sha(ROOT/p)==h
    files=['src/memory/semantic_characterization.py','src/memory/characterization.py','scripts/characterization_semantic.py',
        'src/memory/b1_policy.py','src/memory/b1_attribution.py','src/memory/validation_matrix.py','src/models/content_encoder.py']
    inputs={p:sha(ROOT/p) for p in files};inputs.update(old['checkpoints'])
    provenance=read(ROOT/'data/memops/provenance.json');facts={};queries={};stable=[];noise=[]
    for p in sorted((ROOT/'data/memops/updates').glob('*.json')):
        source=next(r for r in provenance['files'] if r['file']==p.name);assert sha(p)==source['sha256']
        inputs[p.relative_to(ROOT).as_posix()]=sha(p);data=read(p);code=p.stem.split('_')[0]
        ops=[o for o in data['operations'] if o.get('validity')=='confirmed' and o['type'] in ('remember','update')]
        for op in ops:
            target=op['target']['target_id'];key=code if code in DOMAINS and target==DOMAINS[code][0] else code+':'+target
            facts.setdefault(key,[]).append(dict(text=op['trigger_span']['quote'],value=op['new_value'],source=p.relative_to(ROOT).as_posix(),operation=op['operation_id']))
            queries[key]=DOMAINS[code][1] if key==code else 'What is my current '+op['target']['target_name']+'?'
    for key,ops in facts.items():
        if key in DOMAINS:continue
        # Use observed never-revised controls; filter answer-containing queries.
        if len(ops)==1 and not any(str(o['value']).casefold() in queries[key].casefold() for o in ops):stable.append(key)
        noise.append(key)
    assert len(stable)>=20 and all(len(facts[k])>=3 for k in DOMAINS)
    if not (OUT/'protocol.json').exists():
        source,archive=snapshot_source()
        atomic_json(OUT/'protocol.json',dict(batch='C3',source=source,archive=archive,runtime=runtime_info(),frozen_files=inputs,
            methods=METHODS,tasks=TASKS,capacities=[8,16],world_seeds=list(range(93100,93124)),mini_world_seed=93001,policy_seeds=SEEDS,
            source_provenance=provenance['source'],source_commit=provenance['commit'],training_steps=0,api_calls=0,
            addressing='Frozen MiniLM384 cosine. Assign a new observation to nearest resident cluster if cosine>=.65, otherwise new ID. No evicted embeddings/text retained. Query nearest resident cosine then latest arrival within inferred cluster. Topic affinity q=max(0,cos(memory,current demand cue)); same flow equation and frozen head.',
            boundary='Controlled recombination of public annotated quotes, not official MemOps or natural conversation score. Only confirmed operations selected by generator; labels isolated in evaluator. Explicit online correctness feedback provided after answering. Public content reused across schedule worlds; not independent content generalization.',
            ambiguity='Historical distractors explicitly marked no longer current in text, but adapter has no semantic contradiction oracle. Mismerge and wrong-fact errors reported separately from same-fact stale error. Return task probes before reminders; new reminder may reintroduce knowledge.',
            adaptation='Preference update/contradiction:phase1 only, queried values must differ from pre-shift truth and already have been observed. Revival:both demand shifts. Temporary:phase1demand shift,phase2observed new-value queries. Unchanged phase2 is not an additional update episode.',
            statistics='24schedule worlds,3policyseeds averaged; paired bootstrap5000,signflip10000,Holm across8conditions*2baselines*5endpoints. No threshold tuning on formal or mini outcomes.'))
    for p,h in read(OUT/'protocol.json')['frozen_files'].items():assert sha(ROOT/p)==h,p
    if (OUT/'corpus.json.gz').exists():return read_gzip(OUT/'corpus.json.gz')
    texts=set(queries.values())
    for ops in facts.values():
        for op in ops:texts.add(op['text']);texts.add('Historical note, no longer current: '+op['text'])
    for keys in itertools.permutations(list(DOMAINS)[:6],4):texts.add('My current questions concern: '+' '.join(queries[k] for k in keys))
    texts.add('My current questions concern: '+' '.join(queries[k] for k in list(DOMAINS)[6:]))
    encoder=ContentEncoder()
    try:
        textlist=sorted(texts);vectors=encoder.encode(textlist);signature=encoder.signature
    finally:encoder.close()
    corpus=dict(facts=facts,queries=queries,stable=stable,noise=noise,vectors={t:v.tolist() for t,v in zip(textlist,vectors)},encoder_signature=signature)
    zipped(OUT/'corpus.json.gz',corpus);return corpus

def worker(row,method,seed):
    r=execute(row,method,seed,HEADS[seed]);path=OUT/'traces'/f"{row['world_seed']}-{row['condition']}-{seed}-{method}.json.gz";zipped(path,r)
    return dict(trace_file=path.relative_to(OUT).as_posix(),sha256=sha(path),metrics=r['metrics'],audit=audit(row,r))

def report(rows):
    from scripts.characterization_report import values,ci,PRIMARY
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(20261014);summary=[];comparisons=[]
    conditions=sorted({r['condition'] for r in rows})
    for condition in conditions:
        rr=[r for r in rows if r['condition']==condition]
        for m in METHODS:
            s=dict(condition=condition,method=m)
            for field in PRIMARY+('wrong_fact_error','resident_correct_coverage','mismerge_rate','eviction_count'):
                v=values(rr,m,field);s[field]=float(v.mean());s[field+'_ci95']=ci(v,rng)
            pr=[r['pre_reminder_recall'] for r in rr if r['method']==m and r['pre_reminder_recall'] is not None];s['pre_reminder_recall']=float(np.mean(pr)) if pr else None
            summary.append(s)
        for base in METHODS[:2]:
            for field in PRIMARY:
                v=values(rr,'adaptive_evolution',field)-values(rr,base,field);null=np.mean(v*rng.choice([-1,1],size=(10000,24)),axis=1)
                comparisons.append(dict(condition=condition,baseline=base,metric=field,delta=float(v.mean()),ci95=ci(v,rng),p=float((1+np.sum(abs(null)>=abs(v.mean())-1e-14))/10001)))
    last=0.
    for n,r in enumerate(sorted(comparisons,key=lambda x:x['p'])):last=max(last,min(1.,r['p']*(len(comparisons)-n)));r['p_holm']=last
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'comparisons.json',comparisons)
    import csv,html
    with (OUT/'paper_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    for ax,c in zip(axes.flat,conditions):
        rr=[r for r in summary if r['condition']==c];v=np.array([r['accuracy'] for r in rr]);interval=np.array([r['accuracy_ci95'] for r in rr]).T
        ax.bar(range(3),v,color=['#9daeb5','#6e9095','#137b89']);ax.errorbar(range(3),v,yerr=np.maximum([v-interval[0],interval[1]-v],0),fmt='none',ecolor='black',capsize=3)
        ax.set_xticks(range(3),['Decay','Frequency','Adaptive'],rotation=25);ax.set(title=c,ylim=(0,1),ylabel='Retrieval accuracy');ax.grid(axis='y',alpha=.15)
    for fmt in ('png','svg','pdf'):fig.savefig(OUT/f'semantic_migration.{fmt}',dpi=170)
    plt.close(fig)
    keys=['condition','method',*PRIMARY,'wrong_fact_error','resident_correct_coverage','mismerge_rate','pre_reminder_recall']
    table='<table><tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in summary)+'</table>'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px/1.7 system-ui;max-width:1450px;margin:30px auto}img{max-width:100%}td,th{border:1px solid #ddd;padding:5px}table{border-collapse:collapse}</style><h1>C3: Public-quote semantic migration</h1><p>24schedule worlds ×3seeds; two capacities. Frozen embedding-only addressing, no true key/version in policy. Controlled public MemOps quote recombination; not official benchmark QA. Identical no-training policy flow; q becomes cosine affinity. Online correctness feedback is explicit. No threshold tuning. Wrong-fact errors separated from stale same-fact values.</p><img src="semantic_migration.png">'+table+'<p><a href="protocol.json">Protocol</a> · <a href="comparisons.json">Paired tests</a> · <a href="paper_table.csv">CSV</a></p>',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mini-only',action='store_true');args=ap.parse_args()
    if (OUT/'completion.json').exists():
        for p,h in read(OUT/'completion.json')['files'].items():assert sha(OUT/p)==h
        print('C3 completed evidence verified');return
    corpus=prepare();heads={s:frozen_head(ROOT/f'runs/evolving-memory-v1/checkpoints/learned-{s}.pt') for s in SEEDS}
    if not (OUT/'mini_complete.json').exists():
        checks=[]
        for c in (8,16):
            for task in TASKS:
                row=generate(corpus,93001,c,task)
                for method in METHODS:
                    a=execute(row,method,11,heads[11]);b=execute(copy.deepcopy(row),method,11,heads[11])
                    for field in ('trace','queries','lifetimes'):assert a[field]==b[field]
                    path=OUT/'mini'/f'{task}-{c}-{method}.json.gz';zipped(path,a);checks.append(dict(file=path.relative_to(OUT).as_posix(),sha256=sha(path),**audit(row,a)))
        atomic_json(OUT/'mini_complete.json',dict(passed=True,exact_replay=True,checks=checks));print('C3 mini passed',len(checks),flush=True)
    if args.mini_only:return
    env=OUT/'environments.json.gz'
    if not env.exists():zipped(env,[generate(corpus,w,c,t) for w in range(93100,93124) for c in (8,16) for t in TASKS])
    rows=read_gzip(env);manifest=OUT/'records.jsonl';done={}
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);assert sha(OUT/r['trace_file'])==r['sha256'];done[r['trace_file']]=r
    jobs=[]
    for r in rows:
        for s in SEEDS:
            for m in METHODS:
                if f"traces/{r['world_seed']}-{r['condition']}-{s}-{m}.json.gz" not in done:jobs.append((r,m,s))
    with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(heads,)) as pool:
        iterator=iter(jobs);pending=set()
        def submit():
            job=next(iterator,None)
            if job is not None:pending.add(pool.submit(worker,*job))
        for _ in range(8):submit()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.remove(future);r=future.result();done[r['trace_file']]=r
                with manifest.open('a',encoding='utf-8') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                submit()
                if len(done)%72==0:print('C3',len(done),'/1728',flush=True)
    assert len(done)==1728;metrics=[r['metrics'] for r in done.values()];atomic_json(OUT/'metrics.json',metrics)
    report(metrics)
    atomic_json(OUT/'completion.json',dict(runs=1728,mini_passed=True,files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}))
    print('C3 COMPLETE',flush=True)

if __name__=='__main__':main()
