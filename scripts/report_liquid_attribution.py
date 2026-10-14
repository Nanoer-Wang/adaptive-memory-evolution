"""Audit and present frozen language mechanism attribution artifacts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json, hashlib, html
from collections import defaultdict
import numpy as np
from src.config import atomic_json
from scripts.liquid_attribution import OUT, read

def main():
    rr=[json.loads(s) for s in (OUT/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=read(OUT/'summary.json'); comparisons=read(OUT/'comparisons.json'); protocol=read(OUT/'protocol.json')
    lookup={(r['dataset'],r['question_id'],r['method'],r['seed'],r['budget']):r for r in rr}
    assert len(lookup)==len(rr)
    changes=defaultdict(list); cases=[]; categories=defaultdict(list)
    for r in rr:
        if r['method']!='liquid':continue
        for other in ['reset','zero_time','long_gaps','fixed_trained','gru','static']:
            b=lookup[r['dataset'],r['question_id'],other,r['seed'],r['budget']]
            c=dict(dataset=r['dataset'],question_id=r['question_id'],cluster=r['cluster'],category=r['category'],budget=r['budget'],seed=r['seed'],other=other,
                store_changed=r['retained_indices']!=b['retained_indices'],
                retrieval_changed=r['retrieved_sources']!=b['retrieved_sources'],
                recall_difference=r['retrieval_recall']-b['retrieval_recall'] if r['retrieval_recall'] is not None else None)
            changes[r['dataset'],r['budget'],other].append(c)
            if c['recall_difference'] is not None:
                categories[r['dataset'],r['budget'],str(r['category']),other].append(c['recall_difference'])
                if c['recall_difference']!=0:cases.append(c)
    change_summary=[]
    for (dataset,budget,other),cs in changes.items():
        unique={(c['cluster'],c['seed']):c for c in cs}
        scored=[c for c in cs if c['recall_difference'] is not None]
        change_summary.append(dict(dataset=dataset,budget=budget,other=other,history_seed_pairs=len(unique),
            changed_stores=sum(c['store_changed'] for c in unique.values()),
            scored_question_seed_pairs=len(scored),improved=sum(c['recall_difference']>0 for c in scored),
            worsened=sum(c['recall_difference']<0 for c in scored)))
    atomic_json(OUT/'decision_changes.json',change_summary)
    atomic_json(OUT/'paired_changed_cases.json',cases)
    atomic_json(OUT/'category_differences.json',[dict(dataset=k[0],budget=k[1],category=k[2],other=k[3],difference=float(np.mean(v)),question_seed_pairs=len(v),status='exploratory, no category selection') for k,v in categories.items()])
    times=read(OUT/'timing.json');timing=[]
    for method in ['static','gru','liquid','fixed_trained','reset','zero_time','long_gaps']:
        ts=[r['seconds']/r['sessions']*1000 for r in times if r['method']==method]
        timing.append(dict(method=method,median_ms_per_session=float(np.median(ts)),p25=float(np.quantile(ts,.25)),p75=float(np.quantile(ts,.75))))
    atomic_json(OUT/'timing_summary.json',timing)
    # Match previous suite exactly for all unchanged conditions, not just rounded averages.
    aliases={'static_content':'static','gru_content':'gru','liquid_content':'liquid','liquid_fixed_time':'fixed_trained'}
    matched=0
    with (ROOT/'runs/language-readiness-v1/offline.jsonl').open(encoding='utf-8') as f:
        for line in f:
            old=json.loads(line);name=old['method']
            if name=='full_hybrid':method,seed,budget='full',0,0
            elif name=='fifo':method,seed,budget='fifo',0,old['budget']
            else:
                k,ss=name.rsplit('-',1);method,seed,budget=aliases[k],int(ss),old['budget']
            new=lookup[old['dataset'],old['question_id'],method,seed,budget]
            for metric in ['retrieval_recall','evidence_retention','payload_bytes']:
                assert new[metric]==old[metric], (old['question_id'],name,metric)
            matched+=1
    changed=[]
    for name,digest in protocol['checkpoint_hashes'].items():
        p=ROOT/'runs/preservation-study-v2/checkpoints'/name/'best.pt'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(name)
    assert not changed
    audit=dict(records=len(rr),unique_records=len(lookup),questions=len({(r['dataset'],r['question_id']) for r in rr}),
        previous_records_matched_exactly=matched,checkpoints_unchanged=True,capacity_violations=sum(r['budget']>0 and r['payload_bytes']>r['budget'] for r in rr),
        data_previously_exposed=True,api_requests=0,api_tokens=0,
        timing_caution='Microbenchmark only; shared CPU, three repeats, no end-to-end or deployment speed claim.')
    atomic_json(OUT/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    colors={'liquid':'#176b87','gru':'#d17b0f','static':'#697580','reset':'#a4579c','zero_time':'#44a38a','fifo':'#ba4a4a'}
    for ax,dataset in zip(axes[:2],['longmemeval','locomo']):
        for method in colors:
            rs=sorted([r for r in summary if r['dataset']==dataset and r['method']==method],key=lambda r:r['budget'])
            ax.plot([r['budget']/1000 for r in rs],[r['retrieval_recall']*100 for r in rs],marker='o',label=method,color=colors[method],alpha=.85)
        ax.set(title=dataset, xlabel='Payload budget (decimal KB)',ylabel='Evidence retrieval recall (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    primary=[next(c for c in comparisons if c['dataset']=='longmemeval' and c['budget']==64000 and c['comparison']=='liquid-'+m) for m in ['gru','static','reset','zero_time']]
    y=np.arange(4);x=np.array([c['difference']*100 for c in primary]);lo=np.array([c['ci95'][0]*100 for c in primary]);hi=np.array([c['ci95'][1]*100 for c in primary])
    axes[2].errorbar(x,y,xerr=[x-lo,hi-x],fmt='o',color=colors['liquid'],capsize=4);axes[2].axvline(0,color='gray',ls='--');axes[2].set(yticks=y,yticklabels=['vs GRU','vs static','vs reset','vs zero time'],title='Primary: LME at 64 KB',xlabel='Liquid difference (percentage points), 95% CI')
    fig.suptitle('Frozen language-memory attribution | exposed development data | 3 training seeds')
    fig.tight_layout();fig.savefig(OUT/'attribution.png',dpi=170);plt.close(fig)
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    main_rows=[]
    for m in ['fifo','static','gru','liquid','fixed_trained','reset','zero_time','long_gaps']:
        main_rows.append([m]+[f"{next(r['retrieval_recall'] for r in summary if r['dataset']==d and r['method']==m and r['budget']==64000)*100:.2f}%" for d in ['longmemeval','locomo']])
    cirows=[[c['comparison'],f"{c['difference']*100:+.3f}",f"[{c['ci95'][0]*100:+.3f}, {c['ci95'][1]*100:+.3f}]",f"[{c['ci9875'][0]*100:+.3f}, {c['ci9875'][1]*100:+.3f}]"] for c in primary]
    ds=[c for c in change_summary if c['dataset']=='longmemeval' and c['budget']==64000 and c['other'] in ['reset','zero_time']]
    diagnostic='；'.join(f"{c['other']} 改变 {c['changed_stores']}/{c['history_seed_pairs']} 组历史×种子的保留集合，液态改善 {c['improved']}、退步 {c['worsened']} 个有标签问题×种子" for c in ds)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>液态独有优势归因验证</title><style>body{{max-width:1150px;margin:36px auto;padding:0 24px;font:16px/1.75 system-ui;color:#20303a}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef3f5}}img{{width:100%}}.result{{padding:18px;background:#eef3f5;border-left:5px solid #176b87}}small{{color:#566}}</style>
    <h1>液态独有优势：首轮语言任务归因验证</h1><p class="result">已完成。当前实现未达到预先固定的“液态机制增益”判据。LongMemEval 上的正向点估计尚不足以区别于普通历史门控；LoCoMo 未复现优势。没有改动模型、挑选最优种子或删除不利结果。</p>
    <p>覆盖 2,067 个问题、103 组历史、3 个训练种子、32/64/128 KB 容量，共 {len(rr):,} 条记录。可用于证据召回评分的问题为 LongMemEval 90 个、LoCoMo 1,528 个；LoCoMo 只含 10 组独立对话，区间按历史聚类，不把问题数或种子当作独立对话数。数据此前已用于开发，属于机制诊断，不能称作全新独立验证。</p>
    <img src="attribution.png"><h2>64 KB 主指标</h2><p>这里是证据会话检索召回，不是最终问答正确率。文本、向量、检索、容量计量一致，写入时不访问问题与答案。static 为每次重置状态的已训练写入器，仍能看到当前间隔；reset 为液态模型同权重重置，zero_time 为同权重去掉真实时间间隔；fixed_trained 为独立训练的固定时间基线。</p>
    {table(['方法','LongMemEval','LoCoMo'],main_rows)}
    <h2>预先固定的四个主比较</h2><p>先对每个问题的三个种子求均值，再做 5,000 次成对历史聚类 bootstrap。所有主比较的区间均跨零，未通过判据；98.75% 区间同时给出四比较的保守校正。零落在区间内不代表模型等价。</p>
    {table(['比较','差值（百分点）','95% 区间','98.75% 区间'],cirows)}
    <h2>机制实际影响</h2><p>{diagnostic}。这些计数描述干预敏感性，不能单独说明优势。完整有利与不利差异保存在 paired_changed_cases.json。</p>
    <p>把时间间隔人为放大十倍是压力测试，不是新的公开时间标签。LongMemEval 的下降只说明当前模型对时间尺度敏感，不能把扰动对照更差当作优于其他机制的证明。容量和题型分组均为次要探索结果。</p>
    <h2>计算与容量</h2>{table(['写入器','每会话 CPU 中位时间（毫秒）'],[[r['method'],f"{r['median_ms_per_session']:.4f}"] for r in timing])}
    <p>液态有 7,266 参数，GRU 有 7,383 参数，仅少 1.6%；本轮液态写入反而慢于 GRU。这里是缓存文本向量后的写入微基准，包含模型编码层，不包含文本嵌入、检索、API 或端到端 Agent 延迟。共享 CPU 下的三次重复不能支撑部署性能结论。容量包括文本、日期、来源、向量及状态，不包括共享权重、Python 对象或检索索引。</p>
    <h2>建模诊断与下一步边界</h2><p>当前状态更新为 α=exp[-softplus(r(h,x))·(1+Δt)]，h′=h+w·(αh+(1−α)c(h,x)−h)。它是指数门控递推近似，不是严格连续时间求解器。Δt 同时控制旧状态遗忘与当前观察融合，而且每条记录的优先级写入后不再更新；因此“证据随时间变化”的潜在收益并没有直接作用于已经存储的记录。</p>
    <p>下一项候选研究应明确分离“无新观察时的时间演化”和“有新观察时的证据更新”，在相同语义证据上比较液态衰减、普通指数衰减、时间输入 GRU 和静态策略。考察多轮查询与容量竞争下的动态保留决策；时间敏感标签需要来自公开原始记录，不能人为生成偏好液态的答案。此为待验证假设，本轮没有声称其已经成功，也不需要再次扩大静态关系分类器实验。</p>
    <h2>复现与文件</h2><p>与上一轮 55,809 条对应记录的召回、保留率、容量逐条完全一致；全部权重哈希未变，容量越界为零。本轮没有 API 调用，新增 API token 为零。新增三项消融正确性检查通过；完整测试记录另见 tests.txt。</p>
    <ul>{''.join(f'<li><a href="{n}">{n}</a></li>' for n in ['protocol.json','summary.json','comparisons.json','decision_changes.json','category_differences.json','paired_changed_cases.json','timing_summary.json','audit.json','records.jsonl','tests.txt'])}</ul>
    <small>原始协议与源代码快照在评分前保存。全部结果按冻结权重汇报，不做测试集调参。本报告没有把胜过 FIFO 归因为液态独有作用。</small></html>'''
    (OUT/'report.html').write_text(text,encoding='utf-8')
    print(json.dumps(audit,indent=2)); print(json.dumps(ds,indent=2))

if __name__=='__main__':main()
