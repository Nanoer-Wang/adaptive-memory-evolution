"""All operating points, paired uncertainty, and empirical frontier diagnostics."""
import csv,html,itertools,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import atomic_json
from scripts.ame_validation_batches import read
from scripts.b1_sanity_check import read_gzip
PRIMARY=('stable_retention','adaptation_score','accuracy','stale_error','adaptation_lag')

def values(rows,method,field):
    rr=[r for r in rows if r['method']==method]
    return np.asarray([np.mean([r[field] for r in rr if r['world_seed']==w]) for w in sorted({r['world_seed'] for r in rr})])

def ci(v,rng):
    boot=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]

def save(fig,out,name):
    for fmt in ('png','svg','pdf'):fig.savefig(out/f'{name}.{fmt}',dpi=170)
    plt.close(fig)

def build(out):
    protocol=read(out/'protocol.json');rows=read(out/'metrics.json');batch=protocol['batch'];rng=np.random.default_rng(20261014)
    methods=protocol['methods'];conditions=sorted({r['condition'] for r in rows});summary=[];comparisons=[];fronts=[]
    for condition in conditions:
        rr=[r for r in rows if r['condition']==condition]
        for method in methods:
            s=dict(batch=batch,condition=condition,method=method,worlds=24,seeds=3)
            for field in PRIMARY+('eviction_count','time_weighted_resident_age','rejected_writes','policy_seconds'):
                v=values(rr,method,field);s[field]=float(v.mean());s[field+'_ci95']=ci(v,rng)
            summary.append(s)
        if batch=='C1':pairs=[(a,b,f) for a,b in itertools.combinations([m for m in methods if m.startswith('full_')],2) for f in ('stable_retention','adaptation_score')]
        elif batch=='C2':pairs=[(m,'full_'+m.split('_')[1],f) for m in methods if m[0] in ('I','C','U') for f in ('stable_retention','adaptation_score','stale_error')]
        else:pairs=[('full_1',m,f) for m in ('fixed_decay','frequency','full_0') for f in PRIMARY]
        for a,b,field in pairs:
            v=values(rr,a,field)-values(rr,b,field)
            null=(v*rng.choice([-1,1],size=(10000,len(v)))).mean(axis=1)
            comparisons.append(dict(condition=condition,a=a,b=b,metric=field,delta=float(v.mean()),ci95=ci(v,rng),
                p=float((1+np.sum(abs(null)>=abs(v.mean())-1e-14))/10001)))
        if batch=='C1':
            xy=np.stack([np.stack([values(rr,m,'stable_retention'),values(rr,m,'adaptation_score')],axis=1) for m in methods],axis=1)
            means=xy.mean(axis=0);boot=xy[rng.integers(0,24,(5000,24))].mean(axis=1)
            for j,m in enumerate(methods):
                dominates=np.all(means>=means[j]-1e-12,axis=1)&np.any(means>means[j]+1e-12,axis=1)
                bd=np.all(boot>=boot[:,j:j+1]-1e-12,axis=2)&np.any(boot>boot[:,j:j+1]+1e-12,axis=2)
                fronts.append(dict(condition=condition,method=m,empirical_nondominated=not bool(dominates.any()),bootstrap_front_membership=float(np.mean(~bd.any(axis=1)))))
    last=0.
    for n,r in enumerate(sorted(comparisons,key=lambda x:x['p'])):
        last=max(last,min(1.,r['p']*(len(comparisons)-n)));r['p_holm']=last
    tradeoffs=[]
    if batch=='C1':
        for condition in conditions:
            for a,b in itertools.combinations([m for m in methods if m.startswith('full_')],2):
                rr=[r for r in comparisons if r['condition']==condition and r['a']==a and r['b']==b]
                x,y=rr
                tradeoffs.append(dict(condition=condition,a=a,b=b,opposite_point_directions=x['delta']*y['delta']<0,
                    both_endpoints_holm_significant=x['p_holm']<.05 and y['p_holm']<.05,
                    supported_tradeoff=x['delta']*y['delta']<0 and x['p_holm']<.05 and y['p_holm']<.05))
    atomic_json(out/'summary.json',summary);atomic_json(out/'comparisons.json',comparisons)
    atomic_json(out/'empirical_frontier.json',fronts);atomic_json(out/'tradeoff_pairs.json',tradeoffs)
    with (out/'paper_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    fig,axes=plt.subplots(2,2 if len(conditions)==4 else 3,figsize=(13,9),layout='constrained')
    for ax,condition in zip(axes.flat,conditions):
        rr=[r for r in summary if r['condition']==condition]
        if batch in ('C1','C2'):
            for family,color in [('full','#137b89'),('I','#ba563c'),('C','#8166a4'),('U','#57965b')]:
                aa=[r for r in rr if r['method'].startswith(family+'_')]
                aa.sort(key=lambda r:float(r['method'].split('_')[1]))
                if not aa:continue
                ax.plot([r['stable_retention'] for r in aa],[r['adaptation_score'] for r in aa],'-o',color=color,label=family,lw=1)
                for r in aa:
                    x=r['stable_retention'];y=r['adaptation_score'];xc=r['stable_retention_ci95'];yc=r['adaptation_score_ci95']
                    if batch=='C1':
                        ax.errorbar(x,y,xerr=[[max(0,x-xc[0])],[max(0,xc[1]-x)]],yerr=[[max(0,y-yc[0])],[max(0,yc[1]-y)]],color=color,alpha=.25)
                        ax.annotate(r['method'].split('_')[1],(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
            for r in rr:
                if r['method'] in ('fixed_decay','frequency','lru','static_state_adaptive'):
                    ax.scatter(r['stable_retention'],r['adaptation_score'],marker='x',s=65,label=r['method'])
            ax.set(xlabel='Stable retention',ylabel='Adaptation score (higher better)',title=condition)
        else:
            aa=[r for r in rr if r['method'].startswith('full_')]
            ax.bar(range(len(aa)),[r['accuracy'] for r in aa]);ax.set_xticks(range(len(aa)),[r['method'] for r in aa],rotation=30)
            ax.set(title=condition,ylabel='Accuracy')
        ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle(batch+': all prespecified operating points; lines follow rate, not a fitted frontier')
    save(fig,out,'operating_points')
    if batch=='C4':
        windows=[]
        for row in rows:
            data=read_gzip(out/row['trace_file']) if 'trace_file' in row else read_gzip(out/'traces'/f"{row['world_seed']}-{row['condition']}-{row['policy_seed']}-{row['method']}.json.gz")
            for w in data['window_metrics']:windows.append(dict(w,method=row['method'],condition=row['condition'],world_seed=row['world_seed'],seed=row['policy_seed']))
        atomic_json(out/'window_metrics.json',windows)
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for ax,condition in zip(axes.flat,conditions):
            for m in ('fixed_decay','frequency','full_0','full_1','full_2'):
                ww=[w for w in windows if w['condition']==condition and w['method']==m];xx=sorted({w['end'] for w in ww})
                ax.plot(xx,[np.mean([w['accuracy'] for w in ww if w['end']==x]) for x in xx],label=m,lw=1)
            ax.set(title=condition,xlabel='Event count',ylabel='Accuracy / 100-event window');ax.legend(fontsize=7);ax.grid(alpha=.15)
        save(fig,out,'performance_over_time')
    header=''.join('<th>'+html.escape(k)+'</th>' for k in ('condition','method',*PRIMARY))
    body=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in ('condition','method',*PRIMARY))+'</tr>' for r in summary)
    supported=sum(r['supported_tradeoff'] for r in tradeoffs)
    note=f'C1 rate pairs with opposite signs and both endpoints passing Holm: {supported}/{len(tradeoffs)}.' if batch=='C1' else 'All variants reported; no outcome-based model selection.'
    page='<!doctype html><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1350px;margin:30px auto}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}</style>'
    page+=f'<h1>{batch}: Memory evolution characterization</h1><p>{note}</p><p>24 independent schedule worlds × 3 policy seeds. Seed averages precede paired inference. Alpha scales I/C/U inter-event flow only; observation and feedback jumps and T remain. Empirical nondominance is not proof of a population Pareto frontier. No training or LLM calls.</p><img src="operating_points.png"><table><tr>'+header+'</tr>'+body+'</table>'
    if batch=='C4':page+='<img src="performance_over_time.png">'
    page+='<p><a href="comparisons.json">Paired comparisons</a> · <a href="protocol.json">Frozen protocol</a> · <a href="paper_table.csv">Full table</a> · <a href="tradeoff_pairs.json">Trade-off test</a> · <a href="empirical_frontier.json">Frontier uncertainty</a></p>'
    (out/'index.html').write_text(page,encoding='utf-8')
