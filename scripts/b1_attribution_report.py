"""Paper tables, world-clustered paired inference, and explicit attribution gates."""
import csv,html,json
import numpy as np
from scripts.ame_validation_batches import read,sha
from scripts.b1_sanity_check import read_gzip
from scripts.b1_interleaved_sanity import full_trace_page
from src.config import atomic_json
from src.memory.b1_attribution import METHODS

PRIMARY=('accuracy','stale_error','stable_retention','adaptation_lag')
EXTRA=('adaptation_success','eviction_count','mean_observed_lifetime','time_weighted_resident_age','final_eight_accuracy','policy_seconds')


def world_values(rows,metric):
    return np.asarray([np.mean([r[metric] for r in rows if r['world_seed']==w and r[metric] is not None])
        for w in sorted({r['world_seed'] for r in rows})],dtype=float)


def interval(values,rng):
    boot=np.mean(values[rng.integers(0,len(values),(5000,len(values)))],axis=1)
    return float(np.mean(values)),*[float(x) for x in np.quantile(boot,[.025,.975])]


def build(out):
    complete=read(out/'evaluation_complete.json')
    for f,h in complete['files'].items():assert sha(out/f)==h
    protocol=read(out/'protocol.json');rows=read(out/'metrics.json');cal=read(out/'calibration_lock.json')['selected']
    cells=sorted({(r['dynamics'],r['pressure']) for r in rows});summary=[];contrasts=[];matches=[]
    rng=np.random.default_rng(85112)
    for dynamics,pressure in cells:
        cell=[r for r in rows if r['dynamics']==dynamics and r['pressure']==pressure]
        by={m:[r for r in cell if r['method']==m] for m in METHODS}
        a,b=by['adaptive_evolution'],by['replacement_matched']
        errors={k:abs(np.mean([r[k] for r in a])-np.mean([r[k] for r in b]))/max(abs(np.mean([r[k] for r in a])),1e-9)
                for k in ('eviction_count','mean_observed_lifetime')}
        # Distribution mismatch is reported even when the two mean targets pass.
        age_error=abs(np.mean([r['time_weighted_resident_age'] for r in a])-np.mean([r['time_weighted_resident_age'] for r in b]))/max(np.mean([r['time_weighted_resident_age'] for r in a]),1e-9)
        matches.append(dict(dynamics=dynamics,pressure=pressure,passed=all(v<=.10 for v in errors.values()),
            admission=cal[dynamics+'-'+pressure]['admission'],relative_errors=errors,resident_age_relative_error=age_error))
        for method,rr in by.items():
            record=dict(dynamics=dynamics,pressure=pressure,method=method,worlds=24,policy_seeds=3)
            for metric in PRIMARY+EXTRA:
                if all(r[metric] is None for r in rr):record[metric]=None;continue
                mean,low,high=interval(world_values(rr,metric),rng)
                record[metric]=mean;record[metric+'_ci']=[low,high]
            summary.append(record)
        for baseline in ('fixed_decay','static_state_adaptive','replacement_matched','static_uniform_state'):
            for metric in PRIMARY:
                if a[0][metric] is None:continue
                values=world_values(a,metric)-world_values(by[baseline],metric)
                mean,low,high=interval(values,rng)
                null=np.mean(values*rng.choice([-1,1],size=(10000,len(values))),axis=1)
                pval=float((1+np.sum(np.abs(null)>=abs(mean)-1e-14))/10001)
                contrasts.append(dict(dynamics=dynamics,pressure=pressure,baseline=baseline,metric=metric,
                    delta_adaptive_minus_baseline=mean,ci95=[low,high],p_unadjusted=pval,
                    replacement_means_matched=matches[-1]['passed'] if baseline=='replacement_matched' else None))
    ordered=sorted(contrasts,key=lambda c:c['p_unadjusted']);last=0.
    for rank,row in enumerate(ordered):
        last=max(last,min(1.,(len(ordered)-rank)*row['p_unadjusted']));row['p_holm']=last
    atomic_json(out/'summary.json',summary);atomic_json(out/'paired_comparisons.json',contrasts);atomic_json(out/'replacement_matching.json',matches)
    # Machine-readable full table; empty fields explicitly mean not applicable.
    with (out/'performance_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dynamics','pressure','method','worlds','policy_seeds']+list(PRIMARY+EXTRA));writer.writeheader()
        for r in summary:writer.writerow({k:r.get(k) for k in writer.fieldnames})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(METHODS,plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for col,dynamics in enumerate(('stable','slow_drift','rapid_shift')):
        for method in METHODS:
            rr=[next(r for r in summary if r['dynamics']==dynamics and r['pressure']==p and r['method']==method) for p in ('low','medium','high')]
            for ax,metric in [(axes[0,col],'accuracy'),(axes[1,col],'stable_retention')]:
                ax.errorbar([2,8,16],[r[metric] for r in rr],yerr=np.array([[r[metric]-r[metric+'_ci'][0] for r in rr],[r[metric+'_ci'][1]-r[metric] for r in rr]]),marker='o',label=method,color=colors[method],capsize=2,lw=1)
                ax.set(title=dynamics+' / '+metric,xlabel='Distinct noise facts',ylabel='Rate',ylim=(0,1.03),xticks=[2,8,16]);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('B1 heldout: 24 paired worlds, 3 policy seeds, capacity8; world-bootstrap95% intervals')
    fig.savefig(out/'performance.png',dpi=170);fig.savefig(out/'performance.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,dynamics in zip(axes,('slow_drift','rapid_shift')):
        for method in METHODS:
            rr=[r for r in summary if r['dynamics']==dynamics and r['method']==method]
            for r in rr:
                ax.scatter(r['stable_retention'],r['adaptation_lag'],color=colors[method],s={'low':35,'medium':70,'high':110}[r['pressure']])
            ax.plot([r['stable_retention'] for r in rr],[r['adaptation_lag'] for r in rr],color=colors[method],alpha=.4,label=method)
        ax.set(title=dynamics,xlabel='Stable retention (higher is better)',ylabel='First-hit adaptation lag (lower is better)',ylim=(4,33));ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Retention-adaptation trade-off; marker size = pressure. Lag includes failures censored at32.')
    fig.savefig(out/'tradeoff.png',dpi=170);fig.savefig(out/'tradeoff.svg');plt.close(fig)
    # First-world actual lifetime/age distributions; retain every censored episode.
    sample_name='traces/72001-rapid_shift-medium-11-adaptive_evolution.json.gz'
    sample=read_gzip(out/sample_name);environments=read_gzip(out/'environments.json.gz')['evaluation']
    tape=next(r for r in environments if r['world_seed']==72001 and r['dynamics']=='rapid_shift' and r['pressure']=='medium')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in METHODS:
        result=read_gzip(out/f'traces/72001-rapid_shift-medium-11-{method}.json.gz')
        for ax,values in [(axes[0],[r['lifetime'] for r in result['lifetimes']]),(axes[1],[v for f in result['trace'] for v in f['resident_ages']])]:
            values=np.sort(values);ax.step(values,np.arange(1,len(values)+1)/len(values),label=method,color=colors[method],where='post')
    axes[0].set(xlabel='Observed lifetime (censored included)',ylabel='Empirical CDF',title='Preselected sample: observed exposure, not survival estimate')
    axes[1].set(xlabel='Resident age at event',ylabel='Empirical CDF',title='Same sample: age distribution');axes[1].legend(fontsize=7)
    fig.savefig(out/'lifetime_age.png',dpi=170);fig.savefig(out/'lifetime_age.svg');plt.close(fig)
    # Reuse only the plot/trace rendering; it does not evaluate or train anything.
    atomic_json(out/'sample_full_trace.json',sample);atomic_json(out/'sample_trajectory.json',tape)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,field in zip(axes.flat,('I','C','U','T')):
        for k,label in zip(('stable_keys','a_keys','b_keys'),('stable','topicA','topicB')):
            item_id=f'{tape[k][0]}:1';times=[];values=[]
            for frame in sample['trace']:
                for states in (frame['flow_state'],frame['resident']):
                    state=next((i for i in (states or []) if i['item_id']==item_id),None)
                    times.append(frame['event']['time']);values.append(state[field] if state else np.nan)
            ax.plot(times,values,label=label,lw=1)
        for frame in sample['trace']:
            if frame['event']['kind']=='context':ax.axvline(frame['event']['time'],color='grey',ls='--',alpha=.35)
        ax.set(title=field,xlabel='Observed time');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Preselected heldout world72001 / rapid / medium / seed11: version1 records; gaps = absent')
    fig.savefig(out/'memory_evolution.png',dpi=170);fig.savefig(out/'memory_evolution.svg');plt.close(fig)
    full_trace_page(out,sample)
    def display(v):return '—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
    def table(data,fields):
        return '<table><tr>'+''.join('<th>'+html.escape(f)+'</th>' for f in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(display(r.get(f)))+'</td>' for f in fields)+'</tr>' for r in data)+'</table>'
    compact=[dict(dynamics=m['dynamics'],pressure=m['pressure'],admission=m['admission'],matched=m['passed'],eviction_relative_error=m['relative_errors']['eviction_count'],lifetime_relative_error=m['relative_errors']['mean_observed_lifetime'],age_relative_error=m['resident_age_relative_error']) for m in matches]
    matched=sum(m['passed'] for m in matches)
    page='''<!doctype html><meta charset="utf-8"><title>B1 attribution</title><style>body{max-width:1400px;margin:30px auto;padding:0 24px;font:15px/1.8 system-ui;color:#213748}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{background:#edf4f6;padding:15px}h2{margin-top:35px}</style><h1>B1：演化、替换与稳定保留的归因实验</h1>'''
    page+=f'<p class="note">正式矩阵已完成：3压力×3动态×24配对世界×3策略种子×8方法，共{complete["runs"]}次评估；全部逐事件审计，{complete["exact_replays"]}次预指定重放。容量8，冻结旧123参数头，无训练、无API调用。参数与方法在测试前登记，校准世界与测试世界独立。</p>'
    page+='<p>额外的static_uniform_state把初始与新增条目都冻结在I=1,C=1,U=T=0，检查原冻结对照是否仅因初始状态较高而锁死写入。原冻结对照保留同一插入公式；若初始优势造成拒收，需据此限制其归因强度。</p>'
    page+='''<h2>对照究竟控制了什么</h2><p>static_state_adaptive保留同一评分式和学习到的插入状态，之后冻结I/C/U/T，连重复观察和反馈跳变也关闭。因此它检验整个插入后演化包，而不单独识别连续时间积分的贡献。</p><p>正的衰减率λ不会改变按年龄的排序，不能靠调λ匹配替换。replacement_matched使用固定衰减排序加独立种子写入节流；每条件只按4个校准世界的淘汰计数和观察寿命选择概率，不参考任务准确率。它不是纯衰减参数优化，也不是使用测试答案的策略。</p><p>每阶段8条活跃事实，噪声键2/8/16；噪声观察比例固定50%。慢漂移与快切换共享A→B→A主题提示和时钟，区别为事实逐项更新与成批更新。这里的“慢”指事实修订速率，并非主题语义缓慢移动。stable无主题变化与事实更新。每条轨迹616事件、192计分查询；环境时长仍有限。</p>'''
    page+=f'<h2>替换匹配：{matched}/9条件通过预定的双均值10%容差</h2>'+table(compact,['dynamics','pressure','admission','matched','eviction_relative_error','lifetime_relative_error','age_relative_error'])
    page+='''<p>未匹配条件不能作“排除替换频率解释”的强结论。始终满容量时，总观察寿命=容量×总时间，因此平均观察寿命与淘汰次数在代数上耦合，不是两份独立证据。年龄分布仍可能不同；通过双均值匹配也不等于控制了淘汰时机与对象。寿命统计保留右截尾，不把观察时长当作无偏完整寿命。</p><h2>全部方法与条件</h2><p>下表是先对每个世界平均3种子、再汇总24世界的均值；机器表保留原始计数和置信区间。stale error以全部查询为分母，低值可能来自回答缺失，必须结合准确率和命中率。stable的lag不适用，显示为空。</p>'''
    page+=table(summary,['dynamics','pressure','method',*PRIMARY,'adaptation_success','eviction_count','mean_observed_lifetime'])
    page+='<img src="performance.png"><h2>保留—适应权衡</h2><img src="tradeoff.png"><p>lag是首次连续两个窗口达标，未恢复截尾32；并非持续正确。中途的新事实修订还会触发再次适应，末尾8查询指标单独保存，不能简单解释为遗忘。</p>'
    page+='<h2>预定配对比较</h2><p>Δ=adaptive−baseline；准确率/保留为正更好，stale/lag为负更好。95%区间为世界级配对bootstrap，未作同时区间调整；p值对全部主比较使用Holm校正。3策略种子不是72个独立世界。匹配失败的对照保留数值，但不支持独立归因。</p>'+table(contrasts,['dynamics','pressure','baseline','metric','delta_adaptive_minus_baseline','ci95','p_holm','replacement_means_matched'])
    page+='''<h2>实际状态与寿命分布</h2><img src="memory_evolution.png"><img src="lifetime_age.png"><p>样例在实验前指定为首个测试世界/medium/rapid/seed11。被淘汰条目留空；返回阶段有重观察，不能冒充无提醒回忆。</p><h2>论文可支持的范围</h2><p>这组实验评价冻结控制器在受控公开事实日程中的行为。对照可以检验插入后状态演化是否有贡献，以及优势是否仍存在于接近的平均替换强度下；无法单独证明连续积分必要、排除所有替换机制、证明LNN优越，或推出真实语言Agent效果。可靠主题提示、精确键和干净反馈仍是重要边界。</p><p><a href="performance_table.csv">完整性能表CSV</a> · <a href="summary.json">均值及95%区间</a> · <a href="paired_comparisons.json">全部配对统计</a> · <a href="replacement_matching.json">匹配检验</a> · <a href="metrics.json">全部原始指标</a> · <a href="protocol.json">登记协议</a> · <a href="calibration_lock.json">测试前校准锁</a> · <a href="event_trace.html">完整样例事件</a> · <a href="sample_full_trace.json">状态轨迹JSON</a> · <a href="evaluation_records.jsonl">全部trace索引及校验和</a></p>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    atomic_json(out/'report_manifest.json',dict(evaluation_complete_sha256=sha(out/'evaluation_complete.json'),files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.html','.png','.svg','.csv')},matched_cells=matched))
    print(json.dumps(dict(runs=len(rows),matching=compact),ensure_ascii=False,indent=2))
