"""Build manuscript tables and vector figures solely from frozen result files."""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper'
V=ROOT/'runs/ame-validation-v3';C=ROOT/'runs/ame-characterization-v1'
sources={}
def read(path):
    data=path.read_bytes();sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(data).hexdigest()
    return json.loads(data)
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%')
names={'fixed_decay':'Fixed decay','frequency':'Frequency','lru':'LRU','static_state_adaptive':'Frozen state',
       'replacement_matched':r'Replacement control$^{\dagger}$','adaptive_rule':'Adaptive rule','adaptive_evolution':'AME',
       'no_importance':'No importance','no_consistency':'No consistency','no_usage':'No usage','no_time':'No temporal state'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(P/'figures'/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
def box(ax,x,y,w,h,text,color='#e8f2f1',size=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec='#416d73',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(ax,a,b):ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',lw=1,color='#416d73'))
def table_file(file,rows,metrics,caption,label,extra=''):
    cols='l'+'r'*len(metrics);s='\\begin{table}[t]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
    s+='\\begin{tabular}{'+cols+'}\\toprule\nMethod & '+' & '.join(n for _,n in metrics)+r'\\\midrule'+'\n'
    for r in rows:
        vals=[]
        for key,_ in metrics:
            v=r[key];scale=100 if key in ('accuracy','stale_error','stable_retention') else 1
            vals.append(f'{v*scale:.2f}')
        s+=names.get(r['method'],tex(r['method']))+' & '+' & '.join(vals)+r'\\'+'\n'
    s+='\\bottomrule\\end{tabular}\n'+extra+'\n\\end{table}\n';(P/'tables'/file).write_text(s,encoding='utf-8')
def main():
    b2=read(V/'B2/summary.json');b6=read(V/'B6/summary.json');c1=read(C/'C1/summary.json');c2=read(C/'C2/summary.json')
    metrics=[('accuracy','Acc.$\\uparrow$'),('stale_error','Stale$\\downarrow$'),('stable_retention','Ret.$\\uparrow$'),('adaptation_lag','Lag$\\downarrow$')]
    table_file('main.tex',[r for r in b2 if r['capacity']==8],metrics,
        'Constrained structured memory: eight slots, sixteen interference keys. Percentages except lag (target queries). All seven policies share the same streams; 24 worlds, three policy seeds. Full intervals accompany the machine-readable evidence (Appendix~\\ref{app:tables}).',
        'tab:main',r'\par\vspace{2pt}\footnotesize $\dagger$ Calibrated admission/lifetime control; distributional matching failed. This row is not an exactly matched counterfactual.')
    table_file('ablation.tex',[r for r in b6 if r['method'] in ['adaptive_evolution','no_importance','no_consistency','no_usage','no_time']],metrics,
        'State interventions under rapid revision and high interference (eight slots). No retraining; the removed state is pinned to its neutral score value. This experiment differs from the single-flow intervention in Figure~\\ref{fig:components}.','tab:ablation')
    b1=read(ROOT/'runs/ame-validation-v2/B1-attribution-v1/summary.json')
    rows=[r for r in b1 if r['dynamics']=='rapid_shift' and r['pressure']=='high' and r['method'] in ['fixed_decay','replacement_matched','adaptive_evolution']]
    s=r'\begin{center}\small\begin{tabular}{lrrrr}\toprule Policy & Evictions & Mean lifetime & Acc. (\%) & Lag\\\midrule'+'\n'
    for r in rows:s+=names[r['method']].replace(r'$^{\dagger}$','')+f" & {r['eviction_count']:.2f} & {r['mean_observed_lifetime']:.2f} & {100*r['accuracy']:.2f} & {r['adaptation_lag']:.2f}"+r'\\'+'\n'
    s+=r'\bottomrule\end{tabular}\end{center}'
    (P/'tables/attribution.tex').write_text(s,encoding='utf-8')
    from draw_method_figures import main as draw_schematics
    draw_schematics()
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.5),layout='constrained');cmap=plt.get_cmap('viridis')
    marks={'fixed_decay':'x','frequency':'+','lru':'s','static_state_adaptive':'D'}
    for ax,condition in zip(axes,['mixed-c8','rapid_high-c8']):
        rr=[r for r in c1 if r['condition']==condition];aa=sorted([r for r in rr if r['method'].startswith('full_')],key=lambda r:float(r['method'][5:]))
        ax.plot([r['stable_retention'] for r in aa],[r['adaptation_score'] for r in aa],color='#a0aaa8',lw=.7)
        for r in aa:
            a=float(r['method'][5:]);x=r['stable_retention'];y=r['adaptation_score'];xc=r['stable_retention_ci95'];yc=r['adaptation_score_ci95']
            ax.errorbar(x,y,xerr=[[x-xc[0]],[xc[1]-x]],yerr=[[y-yc[0]],[yc[1]-y]],fmt='o',ms=4,color=cmap(a/2),capsize=2)
            if a in (0,.5,1,2):ax.annotate(f'{a:g}',(x,y),xytext=(-13,-13),textcoords='offset points',fontsize=7)
        for method,mark in marks.items():
            r=next(r for r in rr if r['method']==method);ax.scatter(r['stable_retention'],r['adaptation_score'],marker=mark,s=30,color='#b65347',label=names[method])
        ax.set(title='Mixed demand' if 'mixed' in condition else 'Rapid revision / high interference',xlabel='Stable retention',ylabel='Adaptation score',xlim=(.4,1.025),ylim=(-.03,.78));ax.grid(alpha=.16)
    axes[0].legend(loc='lower left',fontsize=6,framealpha=.95)
    save(fig,'operating_region')
    fig,ax=plt.subplots(figsize=(6.4,1.75),layout='constrained')
    for m in ['fixed_decay','frequency','lru','static_state_adaptive','adaptive_rule','adaptive_evolution']:
        rr=sorted([r for r in b2 if r['method']==m],key=lambda r:r['capacity']);ax.plot([r['capacity'] for r in rr],[r['accuracy'] for r in rr],'-o',ms=3,label=names[m])
    ax.set(xscale='log',xticks=[4,8,16,32,64],xticklabels=['4','8','16','32','64'],xlabel='Content slots K (interference keys = 2K)',ylabel='Accuracy',ylim=(0,1.04));ax.legend(ncol=3,fontsize=6,loc='lower right');ax.grid(alpha=.16);save(fig,'capacity')
    fig,axes=plt.subplots(1,3,figsize=(6.4,1.85),layout='constrained')
    for ax,field,name in zip(axes,['stable_retention','adaptation_score','stale_error'],['Stable retention','Adaptation score','Stale error']):
        for family,color in [('full','#1b827b'),('I','#bc6942'),('C','#8165a4'),('U','#708a41')]:
            rr=sorted([r for r in c2 if r['condition']=='rapid_high-c8' and r['method'].startswith(family+'_')],key=lambda r:float(r['method'].split('_')[1]))
            ax.plot([float(r['method'].split('_')[1]) for r in rr],[r[field] for r in rr],'-o',ms=2,label=family,color=color)
        ax.set(xlabel='Enabled flow strength',ylabel=name,xticks=[0,1,2]);ax.grid(alpha=.15)
    axes[0].legend(ncol=2,fontsize=6);save(fig,'components')
    for src,name in [(C/'selected_item_states.pdf','selected_item_states.pdf'),(C/'long_horizon_state_diagnostics.pdf','long_horizon_states.pdf'),(C/'unified_operating_diagram.pdf','all_operating_points.pdf'),(C/'C4/performance_over_time.pdf','long_horizon_performance.pdf')]:
        shutil.copy2(src,P/'figures'/name);sources[src.relative_to(ROOT).as_posix()]=hashlib.sha256(src.read_bytes()).hexdigest()
    # Full data tables retain every reported method; labels are descriptive, not selected winners.
    for batch in ['B2','B3','B4','B5','B6','B7']:
        rr=read(V/batch/'summary.json');s=r'\begin{longtable}{llrrrr}\caption{'+batch+r': complete structured results (percentages; lag in queries).}\label{tab:'+batch+r'}\\\toprule Condition & Policy & Acc. & Stale & Ret. & Lag\\\midrule\endfirsthead\toprule Condition & Policy & Acc. & Stale & Ret. & Lag\\\midrule\endhead'+'\n'
        for r in rr:s+=tex(r['condition'])+' & '+names.get(r['method'],tex(r['method']))+f" & {100*r['accuracy']:.2f} & {100*r['stale_error']:.2f} & {100*r['stable_retention']:.2f} & {r['adaptation_lag']:.2f}"+r'\\'+'\n'
        s+=r'\bottomrule\end{longtable}';(P/'tables'/f'{batch}_all.tex').write_text(s,encoding='utf-8')
    b8=read(V/'B8/summary.json');s=r'\begin{longtable}{llrrr}\caption{Fixed-subset language transfer. Quality is binary judged correctness for conversational QA and normalized rubric quality for NFCats; these scales must not be pooled.}\label{tab:language}\\\toprule Dataset & Policy & Quality & 95\% CI & Evidence hit\\\midrule\endfirsthead\toprule Dataset & Policy & Quality & 95\% CI & Evidence hit\\\midrule\endhead'+'\n'
    for r in b8:
        ci=r['quality_ci95'];hit=r.get('evidence_hit');s+=tex(r['dataset'].replace('MemoryBench-',''))+' & '+names[r['method']]+f" & {r['quality']:.3f} & [{ci[0]:.3f},{ci[1]:.3f}] & "+(f'{hit:.3f}' if hit is not None else '--')+r'\\'+'\n'
    s+=r'\bottomrule\end{longtable}';(P/'tables/language.tex').write_text(s,encoding='utf-8')
    for batch in ['C1','C2','C3','C4']:
        rr=read(C/batch/'summary.json');s=r'\begin{longtable}{llrrrr}\caption{'+batch+r': all frozen operating points. Percentages except adaptation score.}\label{tab:'+batch+r'}\\\toprule Condition & Policy & Acc. & Stale & Ret. & Adapt.\\\midrule\endfirsthead\toprule Condition & Policy & Acc. & Stale & Ret. & Adapt.\\\midrule\endhead'+'\n'
        for r in rr:s+=tex(r['condition'])+' & '+tex(r['method'])+f" & {100*r['accuracy']:.2f} & {100*r['stale_error']:.2f} & {100*r['stable_retention']:.2f} & {r['adaptation_score']:.3f}"+r'\\'+'\n'
        s+=r'\bottomrule\end{longtable}';(P/'tables'/f'{batch}_all.tex').write_text(s,encoding='utf-8')
    for batch in ['B2','B3','B4','B5','B6','B7','B8']:
        for name in ['summary.json','comparisons.json','protocol.json']:
            src=V/batch/name;read(src);dst=P/'evidence'/batch;dst.mkdir(exist_ok=True);shutil.copy2(src,dst/name)
    for batch in ['C1','C2','C3','C4']:
        for name in ['summary.json','comparisons.json','protocol.json']:
            src=C/batch/name;read(src);dst=P/'evidence'/batch;dst.mkdir(exist_ok=True);shutil.copy2(src,dst/name)
    for name in ['summary.json','paired_comparisons.json','protocol.json']:
        src=ROOT/'runs/ame-validation-v2/B1-attribution-v1'/name;read(src);dst=P/'evidence/B1A';dst.mkdir(exist_ok=True);shutil.copy2(src,dst/name)
    extras=[(C/name,'characterization/'+name) for name in [
        'final_validation.json','raw_evidence_index.json','semantic_routing_diagnostics.json',
        'stress_boundary_metrics.json','selected_item_states.json','C3_window_amendment_replay.json',
        'computational_and_replacement_summary.json']]
    extras += [(V/'serial_computational_audit.json','validation/serial_computational_audit.json'),
               (V/'final_validation.json','validation/final_validation.json')]
    for batch in ['C1','C2','C4']:
        extras.append((C/batch/'tradeoff_pairs.json',batch+'/tradeoff_pairs.json'))
    for src,target in extras:
        read(src);dst=P/'evidence'/target;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    src=C/'tests-final.txt'
    sources[src.relative_to(ROOT).as_posix()]=hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(src,P/'evidence/characterization/tests-final.txt')
    (P/'evidence/source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    print('Tables and vector figures generated from',len(sources),'frozen sources')
if __name__=='__main__':main()
