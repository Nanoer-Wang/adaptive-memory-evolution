"""Read-only B2-B7 world-level statistics and capability-boundary reports."""
import csv,html
import numpy as np
from src.config import atomic_json
from scripts.ame_validation_batches import read
from scripts.b1_sanity_check import read_gzip
from scripts.b1_attribution_report import interval,world_values

PRIMARY=('accuracy','stale_error','stable_retention','adaptation_lag')
EXTRA=('adaptation_success','pre_reminder_recall','post_reminder_recovery_lag','post_reminder_recovery_success','acquisition_success',
       'eviction_count','replacement_rate_per_overflow','mean_observed_lifetime','time_weighted_resident_age','policy_seconds','policy_update_seconds','item_state_updates','state_payload_max_bytes','auxiliary_ledger_entries')


def build(out,environments):
    from scripts.validation_matrix import pooled_replacement,match_distance
    protocol=read(out/'protocol.json');rows=read(out/'metrics.json');batch=protocol['batch'];rng=np.random.default_rng(9931)
    summary=[];contrasts=[];matching=[]
    for condition,_ in protocol['conditions']:
        cell=[r for r in rows if r['condition']==condition];by={m:[r for r in cell if r['method']==m] for m in protocol['methods']}
        for method,rr in by.items():
            record=dict(batch=batch,condition=condition,method=method,capacity=rr[0]['capacity'],noise_count=rr[0]['noise_memory_count'],worlds=24,seeds=3)
            for metric in PRIMARY+EXTRA:
                if all(r[metric] is None for r in rr):record[metric]=None;continue
                mean,low,high=interval(world_values(rr,metric),rng);record[metric]=mean;record[metric+'_ci']=[low,high]
            summary.append(record)
        target=pooled_replacement(read_gzip(out/r['trace_file']) for r in by['adaptive_evolution'])
        control=pooled_replacement(read_gzip(out/r['trace_file']) for r in by['replacement_matched'])
        matching.append(dict(condition=condition,**match_distance(control,target),target=target,control=control))
        for method in protocol['methods']:
            if method=='adaptive_evolution':continue
            for metric in PRIMARY+(('pre_reminder_recall','post_reminder_recovery_lag') if batch=='B3' else ()):
                a=world_values(by['adaptive_evolution'],metric);b=world_values(by[method],metric)
                if not np.all(np.isfinite(a-b)):continue
                values=a-b;mean,low,high=interval(values,rng)
                null=np.mean(values*rng.choice([-1,1],size=(10000,len(values))),axis=1)
                contrasts.append(dict(condition=condition,baseline=method,metric=metric,delta_adaptive_minus_baseline=mean,ci95=[low,high],
                    p=float((1+np.sum(np.abs(null)>=abs(mean)-1e-14))/10001),matched=matching[-1]['matched'] if method=='replacement_matched' else None))
    last=0.
    for i,r in enumerate(sorted(contrasts,key=lambda c:c['p'])):
        last=max(last,min(1.,r['p']*(len(contrasts)-i)));r['p_holm']=last
    atomic_json(out/'summary.json',summary);atomic_json(out/'comparisons.json',contrasts);atomic_json(out/'replacement_distribution_matching.json',matching)
    fields=['batch','condition','method','capacity','noise_count','worlds','seeds']+[f+s for f in PRIMARY+EXTRA for s in ('','_ci_low','_ci_high')]
    with (out/'paper_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in summary:
            r={k:row[k] for k in fields[:7]}
            for metric in PRIMARY+EXTRA:r[metric]=row[metric];r[metric+'_ci_low'],r[metric+'_ci_high']=row.get(metric+'_ci',[None,None])
            writer.writerow(r)
    noise=[]
    if batch=='B4':
        for condition,_ in protocol['conditions']:
            if condition=='clean':continue
            for method in protocol['methods']:
                clean=[r for r in rows if r['condition']=='clean' and r['method']==method];corrupt=[r for r in rows if r['condition']==condition and r['method']==method]
                vals=world_values(clean,'accuracy')-world_values(corrupt,'accuracy');mean,low,high=interval(vals,rng)
                state_errors=[];resident_overlap=[]
                for c in corrupt:
                    ref=next(r for r in clean if r['world_seed']==c['world_seed'] and r['policy_seed']==c['policy_seed'])
                    a=read_gzip(out/ref['trace_file']);b=read_gzip(out/c['trace_file']);errors=[];overlap=[]
                    for af,bf in zip(a['trace'],b['trace'],strict=True):
                        aa={i['item_id']:i for i in af['resident']};bb={i['item_id']:i for i in bf['resident']};shared=aa.keys()&bb.keys()
                        overlap.append(len(shared)/max(len(aa.keys()|bb.keys()),1))
                        for key in shared:errors.append([abs(aa[key][f]-bb[key][f]) for f in ('I','C','U','T')])
                    state_errors.append(np.mean(errors,axis=0).tolist() if errors else [None]*4);resident_overlap.append(float(np.mean(overlap)))
                noise.append(dict(condition=condition,method=method,accuracy_degradation=mean,ci95=[low,high],
                    matched_resident_state_mae=np.nanmean(np.asarray(state_errors,dtype=float),axis=0).tolist(),mean_resident_jaccard=float(np.mean(resident_overlap)),
                    warning='State deviation conditions on shared retained item IDs; occupancy divergence reported separately. Not an unbiased all-item trajectory distance.'))
        atomic_json(out/'feedback_degradation.json',noise)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    conditions=[c for c,_ in protocol['conditions']]
    for ax,metric in zip(axes.flat,PRIMARY):
        for method in protocol['methods']:
            rr=[next(r for r in summary if r['method']==method and r['condition']==c) for c in conditions]
            x=[r['capacity'] for r in rr] if batch=='B2' else list(range(len(rr)))
            ax.plot(x,[r[metric] for r in rr],'o-',label=method,lw=1)
        ax.set(title=metric,xlabel='Capacity (noise=2xcapacity)' if batch=='B2' else 'Condition');ax.grid(alpha=.2)
        if batch!='B2':ax.set_xticks(range(len(conditions)),conditions,rotation=20)
    axes[0,0].legend(fontsize=7);fig.suptitle(batch+' /24worlds x3seeds; full intervals in table')
    fig.savefig(out/'performance.png',dpi=160);fig.savefig(out/'performance.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    for method in protocol['methods']:
        rr=[r for r in summary if r['method']==method];ax.plot([r['stable_retention'] for r in rr],[r['adaptation_lag'] for r in rr],'o-',label=method)
    ax.set(xlabel='Stable retention (higher better)',ylabel='First qualifying-window lag (lower better)',title=batch+' stability-plasticity operating points');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'tradeoff.png',dpi=160);plt.close(fig)
    # Sample selected by first registered world/condition/seed, never by outcome.
    first=next(r for r in rows if r['world_seed']==protocol['world_seeds'][0] and r['condition']==conditions[0] and r['method']=='adaptive_evolution' and r['policy_seed']==11)
    sample=read_gzip(out/first['trace_file']);env=next(r for r in environments if r['environment_sha256']==first['environment_sha256'])
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,field in zip(axes.flat,('I','C','U','T')):
        for name in ('stable_keys','a_keys','b_keys'):
            item=f'{env[name][0]}:1';times=[];values=[]
            for frame in sample['trace']:
                for states in (frame['flow_state'],frame['resident']):
                    state=next((i for i in (states or []) if i['item_id']==item),None)
                    times.append(frame['event']['time']);values.append(state[field] if state else np.nan)
            ax.plot(times,values,label=name,lw=1)
        ax.set(title=field,xlabel='Time');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle(batch+' preselected actual version1 records; gaps=absent')
    fig.savefig(out/'state_evolution.png',dpi=160);plt.close(fig)
    atomic_json(out/'sample_trajectory.json',env);atomic_json(out/'sample_trace.json',sample)
    def display(v):return '—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
    def table(data,columns):return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(display(r.get(k)))+'</td>' for k in columns)+'</tr>' for r in data)+'</table>'
    page=f'<!doctype html><meta charset="utf-8"><title>{batch} validation</title><style>body{{max-width:1350px;margin:30px auto;padding:20px;font:15px/1.8 system-ui;color:#213748}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}img{{width:100%}}</style><h1>{batch} 能力边界验证</h1>'
    page+=f'<p>先mini，后独立校准并锁参，再24配对世界×3种子正式评估。共{len(rows)}次正式运行，所有输入和容量按条件共享，冻结B1核心与权重，没有训练。区间按世界配对计算，不能把3种子当成独立世界。</p>'
    page+='<h2>协议与归因边界</h2><p>'+html.escape(protocol[batch])+'</p><p>匹配检查增加全部观察寿命CDF、未截尾寿命CDF和截尾比例，不能承诺独立测试世界恰好一致。只有所有误差均≤0.10时才标记近似分布匹配；否则该对照不支持分布受控归因。成功匹配也不能证明控制了完整淘汰时机与对象。</p>'
    page+=table(matching,['condition','matched','eviction_relative_error','lifetime_mean_relative_error','lifetime_ks','uncensored_lifetime_ks','censor_fraction_error'])
    page+='<h2>所有方法与条件</h2>'+table(summary,['condition','method','capacity',*PRIMARY,'adaptation_success','pre_reminder_recall','post_reminder_recovery_lag','eviction_count','mean_observed_lifetime'])
    page+='<img src="performance.png"><h2>保留与适应</h2><img src="tradeoff.png"><p>这些是不同策略/条件的工作点，不是经过参数扫描证明的可调Pareto前沿。首次恢复指标包含失败截尾，不能当作持续正确。额外auxiliary ledger与query history会随不同键增长，不能将内容槽位预算说成完整内存上限。</p>'
    if noise:page+='<h2>反馈扰动与轨迹偏离</h2>'+table(noise,['condition','method','accuracy_degradation','ci95','matched_resident_state_mae','mean_resident_jaccard'])
    page+='<h2>计算成本</h2>'+table(summary,['condition','method','policy_seconds','policy_update_seconds','item_state_updates','state_payload_max_bytes','auxiliary_ledger_entries'])
    page+='<p>耗时实测于CPU，仅含策略操作，不含轨迹保存/审计。并行执行下的墙钟耗时只作描述；方法速度比较采用另存的串行测量。32字节/条目是四个float64的理论载荷，不是实测增量堆内存；公共Item结构在静态基线中也分配这些字段。不作未经测量的效率结论。Lag要求连续两个长度4的窗口各至少3次正确，最低为5次目标查询；失败按窗口末截尾，不能解释为此后持续正确。</p><h2>逐项状态</h2><img src="state_evolution.png"><h2>统计比较</h2><p>Δ=adaptive−baseline；错误/lag为负更好，准确率/保留为正更好。主要比较按本批统一Holm校正。</p>'+table(contrasts,['condition','baseline','metric','delta_adaptive_minus_baseline','ci95','p_holm','matched'])
    page+='<p><a href="paper_table.csv">论文表格与置信区间</a> · <a href="metrics.json">全部指标</a> · <a href="protocol.json">协议</a> · <a href="records.jsonl">轨迹索引与哈希</a> · <a href="sample_trace.json">完整样例trace</a> · <a href="mini_complete.json">mini审计</a> · <a href="replacement_distribution_matching.json">分布匹配检查</a></p>'
    (out/'index.html').write_text(page,encoding='utf-8')
