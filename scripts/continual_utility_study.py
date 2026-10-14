"""Long-horizon utility study; no model growth, no API calls, preserved old runs."""
import sys,json,time,hashlib,argparse,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json
from src.provenance import snapshot_source
from src.memory.consolidation import ConsolidationController
from src.memory.continual_utility import (make_trajectory,batch_trajectories,rollout,policy_loss,
                                        ENVIRONMENTS,METHODS,LAMBDA_RETENTION,LAMBDA_FORGETTING)
OUT=ROOT/'runs/continual-utility-v1'
SEEDS=[11,23,42];BUDGETS=[8,16];EPOCHS=8
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data.json').exists():return
    public=ROOT/'runs/liquid-consolidation-v2'
    stories=read(public/'stories.json');vectors=read(public/'vectors.json')
    source,archive=snapshot_source()
    protocol=dict(title='Long-horizon continual utility under stationary and changing workloads',
        source=source,archive=archive,registered_at=time.time(),seeds=SEEDS,capacities_slots=BUDGETS,
        arrivals=360,train_per_environment=6,val_per_environment=3,test_per_environment=6,epochs=EPOCHS,
        methods=METHODS,hidden=8,features=40,learning_rate=.003,batch_size=6,
        lambda_retention=LAMBDA_RETENTION,lambda_forgetting=LAMBDA_FORGETTING,
        objective='L_task=1-query accuracy; L_retention=1-accuracy for queries delayed>=30 time units; L_forgetting=mean obsolete-slot fraction. Minimize L_task+.25 L_retention+.05 L_forgetting. Task outcomes only, no proxy importance labels.',
        optimizer='REINFORCE on actual stochastic hard-capacity evictions, undiscounted returns; independent deterministic fixed-decay baseline, no critic network. Full-trajectory backprop, clip1, AdamW. Best validation utility checkpoint; no test-driven selection.',
        policy='Eviction categorical softmax(-log(strength)/.5), same train/eval kernel; strength=alpha*w and exp(-beta*elapsed); shared duplicate/feedback strengthening, observed-version penalty. No new model layers.',
        data='Controlled procedural trajectories from public bAbI split-specific facts and existing frozen vectors.24 scoped keys; conflicts, delayed queries, moving query popularity, capacity competition. Not official bAbI benchmark score or natural lifelong-agent evidence.',
        environments='Same exact regime counts and common per-arrival random tape: stationary shuffled balanced mixture, slow60-arrival blocks, rapid15-arrival blocks. Balanced shuffle is finite-sample stationary approximation. Regime changes alter arrival gaps, conflict probability, query demand and lag. Realized metrics may differ; report audit.',
        causal_inputs='32D content plus8 observable history fields; no regime identifier, requested lag, future queries, target version or evaluator reward in features. Query text is observed causally. Shared bounded per-key demand statistics for all learned controls.',
        budgets='Evidence SLOTS, not byte or RSS limits.24-key version ledger and history metadata shared. No comparison to prior byte-budget scores. No archive retrieval after eviction.',
        baselines='Fixed decay rate chosen only on validation from[.005,.02,.05,.2]. Same stochastic eviction and successful-query reinforcement. Linear/GRU/liquid each8epochs, same seeds/batches/data/checkpoint criterion. Hidden size equal, parameter counts not equal.',
        independence='Disjoint procedural seeds and source story splits; same source text/templates can recur. Prior public data was exposed; fresh generated trajectories are held out. Test trajectories averaged over training/policy seeds for paired bootstrap, not treating seeds as new data.',
        primary='Held-out query accuracy, delayed accuracy, utility, obsolete occupancy; pooled-training model tested separately on3environments. Test once after all model selection. No universal LNN superiority claim.')
    atomic_json(OUT/'protocol.json',protocol)
    data={}
    for split,count,offset in [('train',6,1000),('val',3,2000),('test',6,3000)]:
        data[split]=[make_trajectory(stories,vectors,split,env,offset+i) for env in ENVIRONMENTS for i in range(count)]
    atomic_json(OUT/'data.json',data)
    audit=[]
    for split,rows in data.items():
        for row in rows:
            ev=row['events'];q=[e for e in ev if e['type']==0];obs=[e for e in ev if e['type']==1]
            audit.append(dict(split=split,environment=row['environment'],seed=row['seed'],events=len(ev),
                              queries=len(q),conflicts=sum(e['conflict'] for e in obs),
                              duration=ev[-1]['time'],mean_query_delay=float(np.mean([e['delay'] for e in q])),
                              long_queries=sum(e['delay']>=30 for e in q),regime_counts=row['regime_counts']))
    atomic_json(OUT/'data_audit.json',audit)
    print('Prepared18train/9validation/18test long trajectories',flush=True)


def evaluate_set(rows,capacity,model=None,decay=1/30,seed=0,greedy=False,record=False):
    with torch.no_grad():return rollout(batch_trajectories(rows),capacity,model,decay,seed,greedy,record)


def train():
    prepare();data=read(OUT/'data.json');(OUT/'checkpoints').mkdir(exist_ok=True)
    if not (OUT/'fixed_decay.json').exists():
        candidates=[]
        for rate in [.005,.02,.05,.2]:
            metrics=[np.mean(evaluate_set(data['val'],c,decay=rate,seed=700)['metrics']['utility']) for c in BUDGETS]
            candidates.append(dict(rate=rate,validation_utility=float(np.mean(metrics))))
        best=max(candidates,key=lambda r:r['validation_utility'])
        atomic_json(OUT/'fixed_decay.json',dict(rate=best['rate'],candidates=candidates))
        print('Validation fixed decay',best,flush=True)
    baseline_rate=read(OUT/'fixed_decay.json')['rate']
    for kind in METHODS[1:]:
        for seed in SEEDS:
            checkpoint=OUT/'checkpoints'/f'{kind}-{seed}.pt';history_path=OUT/f'training-{kind}-{seed}.json'
            if history_path.exists() and read(history_path).get('complete'):continue
            torch.manual_seed(seed);model=ConsolidationController(kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            history=[];best=-float('inf');start=time.perf_counter()
            rng=np.random.default_rng(seed)
            for epoch in range(EPOCHS):
                order=rng.permutation(len(data['train']));losses=[];utilities=[]
                model.train()
                for batch_index,left in enumerate(range(0,len(order),6)):
                    rows=[data['train'][i] for i in order[left:left+6]];batch=batch_trajectories(rows)
                    capacity=BUDGETS[(epoch+batch_index)%2]
                    with torch.no_grad():baseline=rollout(batch,capacity,decay=baseline_rate,greedy=True)
                    result=rollout(batch,capacity,model,seed=seed*10000+epoch*100+batch_index)
                    loss=policy_loss(result,baseline)
                    optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                    optimizer.step();losses.append(float(loss.detach()));utilities+=result['metrics']['utility']
                model.eval()
                val=float(np.mean([np.mean(evaluate_set(data['val'],c,model,seed=700)['metrics']['utility']) for c in BUDGETS]))
                if val>best:
                    best=val;torch.save(dict(state_dict=model.state_dict(),kind=kind,seed=seed,epoch=epoch,validation_utility=val),checkpoint)
                history.append(dict(epoch=epoch,policy_gradient_loss=float(np.mean(losses)),train_utility=float(np.mean(utilities)),validation_utility=val,seconds=time.perf_counter()-start))
                atomic_json(history_path,dict(complete=False,history=history))
                print(kind,seed,'epoch',epoch,'val',round(val,4),flush=True)
            atomic_json(history_path,dict(complete=True,history=history,best_validation_utility=best,seconds=time.perf_counter()-start))


def evaluate():
    if (OUT/'evaluation_complete.json').exists():return
    data=read(OUT/'data.json');rate=read(OUT/'fixed_decay.json')['rate'];results=[]
    # Check all training complete BEFORE accessing held-out outcomes.
    for kind in METHODS[1:]:
        for seed in SEEDS:assert read(OUT/f'training-{kind}-{seed}.json')['complete']
    trace_dir=OUT/'traces';trace_dir.mkdir(exist_ok=True)
    for kind in METHODS:
        for seed in SEEDS:
            model=None
            if kind!='decay':
                model=ConsolidationController(kind);saved=torch.load(OUT/'checkpoints'/f'{kind}-{seed}.pt',weights_only=True)
                model.load_state_dict(saved['state_dict']);model.eval()
            for environment in ENVIRONMENTS:
                rows=[r for r in data['test'] if r['environment']==environment]
                for capacity in BUDGETS:
                    start=time.perf_counter();output=evaluate_set(rows,capacity,model,rate,seed=seed*100+9000,record=True)
                    elapsed=time.perf_counter()-start
                    for i,row in enumerate(rows):
                        result=dict(method=kind,seed=seed,environment=environment,capacity=capacity,trajectory_seed=row['seed'],seconds_per_trajectory=elapsed/len(rows))
                        result.update({k:v[i] for k,v in output['metrics'].items()});results.append(result)
                    atomic_json(trace_dir/f'{kind}-{seed}-{environment}-{capacity}.json',dict(trajectory_seeds=[r['seed'] for r in rows],events=output['trace']))
            print('Held-out evaluation',kind,seed,flush=True)
    atomic_json(OUT/'results.json',results);atomic_json(OUT/'evaluation_complete.json',dict(rows=len(results),at=time.time()))


def report():
    rows=read(OUT/'results.json');summary=[];comparisons=[]
    objectives=[]
    for r in rows:
        task=1-r['accuracy'];retention=1-r['delayed_accuracy'];forgetting=r['stale_occupancy']
        total=task+LAMBDA_RETENTION*retention+LAMBDA_FORGETTING*forgetting
        assert abs(total-(1+LAMBDA_RETENTION-r['utility']))<1e-6
        objectives.append({k:r[k] for k in ['method','seed','environment','capacity','trajectory_seed']}|dict(L_task=task,L_retention=retention,L_forgetting=forgetting,L=total))
    atomic_json(OUT/'objective_components.json',objectives)
    for env in ENVIRONMENTS:
        for capacity in BUDGETS:
            for method in METHODS:
                rr=[r for r in rows if r['environment']==env and r['capacity']==capacity and r['method']==method]
                summary.append(dict(environment=env,capacity=capacity,method=method,**{k:float(np.mean([r[k] for r in rr])) for k in ['accuracy','delayed_accuracy','utility','stale_occupancy','evictions','mean_beta','seconds_per_trajectory']}))
            for baseline in METHODS[:-1]:
                differences={metric:[] for metric in ['accuracy','utility']}
                for trajectory in sorted({r['trajectory_seed'] for r in rows if r['environment']==env}):
                    for metric in differences:
                        means={method:np.mean([r[metric] for r in rows if r['environment']==env and r['capacity']==capacity and r['trajectory_seed']==trajectory and r['method']==method]) for method in ['liquid',baseline]}
                        differences[metric].append(float(means['liquid']-means[baseline]))
                for metric,diff in differences.items():
                    rng=np.random.default_rng(123);samples=np.mean(rng.choice(diff,size=(3000,len(diff)),replace=True),axis=1)
                    comparisons.append(dict(environment=env,capacity=capacity,baseline=baseline,metric=metric,difference=float(np.mean(diff)),ci95=np.quantile(samples,[.025,.975]).tolist(),independent_trajectories=len(diff)))
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'paired_comparisons.json',comparisons)
    def table(rr,columns):
        return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[c]:.4f}' if isinstance(r[c],float) else str(r[c]))+'</td>' for c in columns)+'</tr>' for r in rr)+'</table>'
    doc='''<!doctype html><meta charset="utf-8"><title>Continual utility study</title><style>body{max-width:1200px;margin:32px auto;font:16px/1.7 system-ui}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:5px;text-align:left}</style><h1>Long-horizon continual utility study</h1><p>Hidden8/features40 unchanged. Public bAbI facts assembled into synthetic long trajectories, not an official benchmark score.18 training,9 validation,18 held-out test trajectories;3 model/policy seeds. Same discrete stochastic eviction in training and evaluation.</p><p>Objective: L_task +0.25 L_retention +0.05 L_forgetting, from realized future query success, delayed query success and obsolete occupancy. REINFORCE, no proxy importance labels or added critic. Budgets below are evidence slots, not bytes. Fixed decay tuned on validation only.</p>'''
    doc+=table(summary,['environment','capacity','method','accuracy','delayed_accuracy','utility','stale_occupancy','mean_beta'])
    doc+='<h2>Liquid minus baseline: paired exploratory bootstrap</h2><p>Average three seeds within each trajectory before resampling six trajectories per environment. Small sample, unadjusted multiple comparisons; intervals are exploratory. CPU rollout timing includes research harness and is not an engineering speed benchmark.</p>'
    doc+=table(comparisons,['environment','capacity','baseline','metric','difference','ci95'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,layout='constrained')
    for ax,capacity in zip(axes,BUDGETS):
        x=np.arange(3)
        for index,method in enumerate(METHODS):
            values=[100*next(r['accuracy'] for r in summary if r['environment']==env and r['capacity']==capacity and r['method']==method) for env in ENVIRONMENTS]
            ax.bar(x+(index-1.5)*.2,values,width=.19,label=method)
        ax.set_xticks(x,ENVIRONMENTS);ax.set_title(f'{capacity} evidence slots');ax.set_ylim(0,100)
        ax.set_ylabel('Held-out task accuracy (%)');ax.legend(fontsize=8)
    fig.savefig(OUT/'environment_comparison.png',dpi=170);plt.close(fig)
    doc+='<h2>Across temporal environments</h2><img src="environment_comparison.png" style="width:100%" alt="Task accuracy in three temporal environments and two capacities">'
    if (OUT/'learning_audit.json').exists():
        learning=read(OUT/'learning_audit.json');interactions=read(OUT/'environment_interactions.json')
        no_time=read(OUT/'no_time_posthoc.json')['results'];ablation_summary=[]
        for env in ENVIRONMENTS:
            for capacity in BUDGETS:
                full=next(r['accuracy'] for r in summary if r['environment']==env and r['capacity']==capacity and r['method']=='liquid')
                disabled=float(np.mean([r['accuracy'] for r in no_time if r['environment']==env and r['capacity']==capacity]))
                ablation_summary.append(dict(environment=env,capacity=capacity,liquid=full,no_time_same_weights=disabled,difference=full-disabled))
        doc+='<h2>Learning and continuous-time diagnostics</h2><p>Post-primary audits, no checkpoint changes. Initialization comparison uses the same validation trajectories and random tape; best-of-eight selection means these gains are not independent confirmation of optimization. One GRU run selected a trained epoch slightly below initialization; initialization was not a checkpoint candidate in the registered primary recipe.</p>'
        doc+=table(learning,['method','seed','initial_validation_utility','selected_validation_utility','gain','selected_epoch'])
        doc+=table(ablation_summary,['environment','capacity','liquid','no_time_same_weights','difference'])
        doc+='<h2>Does changing dynamics increase the liquid advantage?</h2><p>Difference of differences: (liquid minus baseline in changing environment) minus (liquid minus baseline in stationary environment). Positive values favor a changing-environment-specific benefit. Six paired procedural seeds, unadjusted exploratory intervals.</p>'
        doc+=table([r for r in interactions if r['metric']=='accuracy'],['environment','capacity','baseline','difference_of_differences','ci95'])
    doc+='''<h2>Interpretation and limits</h2><p>The long-horizon mismatch has been addressed in this controlled study: train, validation and test all contain360arrivals, delayed questions and persistent state. Nevertheless, validation-tuned fixed decay remains ahead in both changing environments. This does not support the claim that short-horizon training was the sole reason for the previous failure.</p><p>At16slots: fixed decay versus liquid is67.20%vs66.26%stationary,77.40%vs75.38%slow,69.93%vs67.90%rapid. Relative changing-environment improvements are not positive in the point estimates. Some liquid-versus-GRU point estimates improve, but the strong rule baseline remains the relevant comparison.</p><p>Liquid selected validation utility improves about0.0155–0.0162 over initialization, but eight epochs/24 optimizer steps and18 training trajectories do not establish convergence. Recurrent time-flow removal changes accuracy by less than0.6percentage points in these conditions. Learned average beta remains around0.024, compared with selected fixed beta0.05; aggregate rates alone do not prove absence of event-level adaptation.</p><p>Slow drift can make task demand more locally concentrated and thus easier for recency rules. More nonstationarity does not automatically create a task requiring liquid state. A shared observed-version penalty also performs much of the stale-record handling. Regime components have matched counts but realized gaps/conflicts/lag distributions differ; see workload audit.</p><p>Beta is assigned when a record arrives or merges, rather than globally recomputed for all surviving records at every demand change. Successful-query reinforcement is shared and does not itself change beta. This bounds what adaptation the current mechanism can express without changing model size. The experiment trains expected structured task utility, not an LLM or natural-language answer model.</p><p>Next scientifically justified tests: verify optimization with matched fixed-decay initialization and sufficient equal training budgets; distinguish benefit of history from benefit of reactive statistics; test how existing memories can react to observed demand shifts using the same controller dimensions. Predefine any new test streams before evaluation. Do not add network complexity or claim universal LNN superiority.</p><h2>Reproduction</h2><pre>python -m src.cli continual-utility --phase all</pre><p>Cached completion markers preserve primary model/answer records. A partial interrupted training run restarts that model deterministically; this is not optimizer-state mid-epoch resume. No Agnes calls are made by this study.</p>'''
    doc+='<p><a href="protocol.json">Protocol</a> · <a href="data_audit.json">Workload audit</a> · <a href="results.json">Per-trajectory results</a> · <a href="objective_components.json">Loss components</a> · <a href="fixed_decay.json">Validation rate selection</a> · <a href="tests.txt">158-test log</a> · <a href="implementation_replay_audit.json">Replay audit</a></p>'
    (OUT/'index.html').write_text(doc,encoding='utf-8')
    source,archive=snapshot_source();atomic_json(OUT/'delivery.json',dict(source=source,archive=archive,summary='summary.json',api_requests=0,trained_models=9,tests=158,test_trajectories=18,primary_metric_rows=len(rows),parameters={k:sum(p.numel() for p in ConsolidationController(k).parameters()) for k in METHODS[1:]}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','train','evaluate','report','all'],default='all');args=parser.parse_args()
    torch.set_num_threads(1)
    for name,fn in [('prepare',prepare),('train',train),('evaluate',evaluate),('report',report)]:
        if args.phase in [name,'all']:fn()
