"""Paired quality/cost analysis, implementation timing and actual QA report."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,html,hashlib
from collections import defaultdict
import numpy as np
from src.config import atomic_json
from scripts.writer_efficiency import OUT,read

def interval(grouped):
    sums=np.array([sum(v) for v in grouped.values()]);ns=np.array([len(v) for v in grouped.values()])
    rng=np.random.default_rng(20261012);ix=rng.integers(0,len(ns),(5000,len(ns)))
    samples=sums[ix].sum(1)/ns[ix].sum(1)
    return dict(difference=float(sums.sum()/ns.sum()),ci95=np.quantile(samples,[.025,.975]).tolist(),clusters=len(ns),questions=int(ns.sum()))

def fallback_analysis(base):
    path=ROOT/'runs/content-fallback-v1'
    if not (path/'offline_complete.json').exists():return None
    rr=[json.loads(s) for s in (path/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=defaultdict(list);byq=defaultdict(lambda:defaultdict(list));clusters={}
    for r in rr:
        if r['recall'] is None:continue
        groups[r['dataset'],r['budget'],r['kind'],r['policy']].append(r)
        byq[r['dataset'],r['budget'],r['question_id']][r['kind'],r['policy']].append(r['recall']);clusters[r['dataset'],r['question_id']]=r['cluster']
    summaries=[];paired=[];attribution=[]
    for (dataset,budget,kind,policy),rs in groups.items():
        cs=defaultdict(list)
        for (ds,b,qid),v in byq.items():
            if (ds,b)!=(dataset,budget):continue
            cs[clusters[ds,qid]].append(float(np.mean(v[kind,policy])-np.mean(base[ds,b,qid][kind,'full'])))
        stat=interval(cs)
        summaries.append(dict(dataset=dataset,budget=budget,kind=kind,policy=policy,recall=float(np.mean([r['recall'] for r in rs]))))
        paired.append(dict(dataset=dataset,budget=budget,kind=kind,policy=policy,**stat,passes_recall_screen=stat['ci95'][0]>-.01))
    for dataset in ['longmemeval','locomo']:
        for budget in [64000,128000]:
            rows=[(qid,v) for (ds,b,qid),v in byq.items() if (ds,b)==(dataset,budget)]
            for other in ['gru','static','liquid_linear_only']:
                cs=defaultdict(list)
                for qid,v in rows:
                    if other=='liquid_linear_only':
                        delta=np.mean(v['liquid','hybrid2'])-np.mean(v['liquid','linear_only'])
                    else:
                        delta=(np.mean(v['liquid','hybrid2'])-np.mean(base[dataset,budget,qid]['liquid','full']))-(np.mean(v[other,'hybrid2'])-np.mean(base[dataset,budget,qid][other,'full']))
                    cs[clusters[dataset,qid]].append(float(delta))
                attribution.append(dict(dataset=dataset,budget=budget,other=other,**interval(cs)))
    atomic_json(path/'summary.json',summaries);atomic_json(path/'paired_differences.json',paired);atomic_json(path/'attribution.json',attribution)
    qa=[read(p) for p in (path/'qa').glob('*.json')] if (path/'qa').exists() else []
    qa_summary=[];seen={}
    for method in ['liquid_linear_only','liquid_hybrid2','gru_hybrid2']:
        rows=[r for r in qa if r['method']==method and 'judge_response' in r]
        qa_summary.append(dict(method=method,completed=len(rows),correct=sum(r['custom_judge_correct'] == 1.0 for r in rows)))
    for r in qa:
        for field in ['generation','judge_response']:
            if field in r:
                response=r[field];key=response['request_hash']
                if key not in seen or not response['cache_hit']:seen[key]=response
    usage=dict(unique_responses=len(seen),reported_uncached_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in seen.values() if not r['cache_hit']))
    atomic_json(path/'qa_summary.json',qa_summary);atomic_json(path/'api_usage.json',usage)
    atomic_json(path/'qa_failures.json',[r for r in qa if 'judge_response' in r and r['custom_judge_correct'] != 1.0])
    return dict(summary=summaries,paired=paired,attribution=attribution,qa=qa_summary,usage=usage)

def main():
    rr=[json.loads(s) for s in (OUT/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=defaultdict(list); q=defaultdict(lambda:defaultdict(list));clusters={}
    for r in rr:
        if r['recall'] is None:continue
        groups[r['dataset'],r['budget'],r['kind'],r['policy']].append(r)
        q[r['dataset'],r['budget'],r['question_id']][r['kind'],r['policy']].append(r['recall'])
        clusters[r['dataset'],r['question_id']]=r['cluster']
    summary=[dict(dataset=k[0],budget=k[1],kind=k[2],policy=k[3],recall=float(np.mean([r['recall'] for r in v])),retention=float(np.mean([r['retention'] for r in v]))) for k,v in groups.items()]
    diffs=[];attribution=[]
    for dataset in ['longmemeval','locomo']:
        for budget in [64000,128000]:
            rows=[(key[2],v) for key,v in q.items() if key[:2]==(dataset,budget)]
            for policy in ['stride2','stride4','novelty']:
                for kind in ['liquid','gru','static']:
                    cs=defaultdict(list)
                    for qid,v in rows:cs[clusters[dataset,qid]].append(float(np.mean(v[kind,policy])-np.mean(v[kind,'full'])))
                    stat=interval(cs);diffs.append(dict(dataset=dataset,budget=budget,kind=kind,policy=policy,**stat,passes_recall_screen=stat['ci95'][0]>-.01))
                for other in ['gru','static']:
                    cs=defaultdict(list)
                    for qid,v in rows:
                        a=np.mean(v['liquid',policy])-np.mean(v['liquid','full'])
                        b=np.mean(v[other,policy])-np.mean(v[other,'full'])
                        cs[clusters[dataset,qid]].append(float(a-b))
                    attribution.append(dict(dataset=dataset,budget=budget,other=other,policy=policy,**interval(cs)))
    costs=read(OUT/'costs.json');costsummary=[]
    for dataset in ['longmemeval','locomo']:
        for kind in ['liquid','gru','static']:
            for policy in ['full','stride2','stride4','novelty']:
                rows=[r for r in costs if (r['dataset'],r['kind'],r['policy'])==(dataset,kind,policy)]
                events=sum(r['events'] for r in rows);calls=sum(r['scorer_calls'] for r in rows)
                costsummary.append(dict(dataset=dataset,kind=kind,policy=policy,events=events,scorer_calls=calls,saving=1-calls/events,seconds=sum(r['seconds'] for r in rows)))
    for d in diffs:
        c=next(c for c in costsummary if (c['dataset'],c['kind'],c['policy'])==(d['dataset'],d['kind'],d['policy']))
        d['savings']=c['saving'];d['passes_joint_screen']=d['passes_recall_screen'] and c['saving']>=.45
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'paired_differences.json',diffs);atomic_json(OUT/'attribution.json',attribution);atomic_json(OUT/'cost_summary.json',costsummary)
    prior={}
    with (ROOT/'runs/liquid-attribution-v1/records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['method'] in ['liquid','gru','static'] and r['budget'] in [64000,128000]:prior[r['dataset'],r['question_id'],r['method'],r['seed'],r['budget']]=r
    matched=0
    for r in rr:
        if r['policy']=='full':
            old=prior[r['dataset'],r['question_id'],r['kind'],r['seed'],r['budget']]
            assert old['retrieval_recall']==r['recall'] and old['evidence_retention']==r['retention'] and old['payload_bytes']==r['payload_bytes'];matched+=1
    for name,digest in read(OUT/'protocol.json')['checkpoints'].items():
        assert hashlib.sha256((ROOT/'runs/preservation-study-v2/checkpoints'/name/'best.pt').read_bytes()).hexdigest()==digest
    eq=read(OUT/'equivalence.json');audit=dict(records=len(rr),full_matches=matched,checkpoint_hashes_unchanged=True,equivalence_histories_models=len(eq),max_abs_error=max(r['max_abs_error'] for r in eq),retention_unchanged=all(r['retention_identical'] for r in eq),capacity_violations=sum(r['payload_bytes']>r['budget'] for r in rr))
    atomic_json(OUT/'audit.json',audit)
    qa=[read(p) for p in (OUT/'qa').glob('*.json')] if (OUT/'qa').exists() else []
    qa_summary=[]
    for method in ['liquid_full','liquid_stride2','gru_full','gru_stride2']:
        rows=[r for r in qa if r['method']==method and 'judge_response' in r]
        qa_summary.append(dict(method=method,completed=len(rows),correct=sum(r['custom_judge_correct'] == 1.0 for r in rows),invalid_judges=sum(r['custom_judge_correct'] is None for r in rows)))
    seen={}
    for r in qa:
        for field in ['generation','judge_response']:
            if field in r:
                response=r[field];key=response['request_hash']
                if key not in seen or not response['cache_hit']:seen[key]=response
    usage=dict(unique_response_hashes=len(seen),uncached_responses=sum(not r['cache_hit'] for r in seen.values()),reported_uncached_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in seen.values() if not r['cache_hit']),http_attempts_in_successful_responses=sum(r.get('attempts',0) for r in seen.values() if not r['cache_hit']))
    atomic_json(OUT/'qa_summary.json',qa_summary);atomic_json(OUT/'api_usage.json',usage)
    failures=[dict(question_id=r['question_id'],method=r['method'],question=r['question'],answer=r['answer'],hypothesis=r.get('hypothesis'),recall=r['recall'],retention=r['retention'],correct=r.get('custom_judge_correct')) for r in qa if 'judge_response' in r and r['custom_judge_correct'] != 1.0]
    atomic_json(OUT/'qa_failures.json',failures)
    engineering=read(OUT/'engineering.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4.4))
    for kind,color in [('liquid','#176b87'),('gru','#d17b0f')]:
        es=[next(r for r in engineering if r['kind']==kind and r['method']==m) for m in ['online_original','online_fused','online_scripted']]
        ax[0].plot(range(3),[r['ms_per_session'] for r in es],marker='o',label=kind,color=color)
    ax[0].set(xticks=range(3),xticklabels=['Original','Fused','Scripted'],ylabel='CPU milliseconds / event',title='Strict online write latency');ax[0].legend()
    for axis,dataset in zip(ax[1:],['longmemeval','locomo']):
        for kind,color in [('liquid','#176b87'),('gru','#d17b0f'),('static','#697580')]:
            rows=[next(r for r in summary if (r['dataset'],r['budget'],r['kind'],r['policy'])==(dataset,64000,kind,p)) for p in ['full','stride2','stride4','novelty']]
            xx=[next(c['saving']*100 for c in costsummary if (c['dataset'],c['kind'],c['policy'])==(dataset,kind,r['policy'])) for r in rows]
            axis.scatter(xx,[r['recall']*100 for r in rows],label=kind,color=color)
            if kind=='liquid':
                for x,r in zip(xx,rows):axis.annotate(r['policy'],(x,r['recall']*100),xytext=(3,5),textcoords='offset points',fontsize=8)
        axis.set(title=dataset+' | 64 KB',xlabel='Scorer calls saved (%)',ylabel='Evidence recall (%)');axis.legend()
    for axis in ax:axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'results.png',dpi=160);plt.close(fig)
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    fallback=fallback_analysis(q);fallback_html=''
    if fallback is not None:
        timing_path=ROOT/'runs/content-fallback-v1/online_timing_summary.json'
        fallback_timing=read(timing_path) if timing_path.exists() else []
        rows=[]
        for d in fallback['paired']:
            if d['budget']!=64000:continue
            rec=next(s['recall'] for s in fallback['summary'] if all(s[k]==d[k] for k in ['dataset','budget','kind','policy']))
            rows.append([d['dataset'],d['kind'],d['policy'],f'{rec*100:.2f}%',f"{d['difference']*100:+.2f}",f"[{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}]",str(d['passes_recall_screen'])])
        fallback_html='<h2>研究线第二步：内容敏感的便宜替代评分</h2><p>这是看到第一轮失效后另行冻结协议的探索扩展，不伪装为第一轮预注册结果。仅在原训练集上用冻结教师分数训练384维线性学生（ridge α=1，截距不惩罚），对九个教师使用同样配方，无测试集拟合或调参。hybrid2每两条调用一次教师，跳过时改用当前文本的学生分数；linear_only完全不用递推教师，作为必须保留的强对照。计算学生分数仍有成本，不能称为零计算。</p>'
        fallback_html+=table(['数据','教师','策略','召回','相对全教师差/百分点','95%区间','通过召回筛选'],rows)
        fallback_html+='<p><strong>液态hybrid2在LongMemEval从60.83%到61.82%，LoCoMo从48.62%到50.41%，减少约一半递推评分调用；但LongMemEval非劣筛选尚未通过，与GRU/静态和纯线性学生的差异区间也仍跨零。因此这是可继续验证的成本候选，不是液态独有优势证明。</strong></p>'
        if fallback_timing:
            fallback_html+=table(['教师','在线策略','每条毫秒（含控制器/学生）'],[[r['kind'],r['policy'],f"{r['median_ms_per_event']:.5f}"] for r in fallback_timing])
            fallback_html+='<p>使用同样脚本化教师、前8组公开历史、3次预热及15次交错重复，逐条输入，包含实际使用的当前文本线性评分；不含文本嵌入、检索与API。液态约0.0278→0.0165毫秒。GRU同样受益，纯线性模型更便宜，不能只与未优化教师比较。</p>'
            fig2,axes2=plt.subplots(1,3,figsize=(14,4))
            for axis,dataset in zip(axes2[:2],['longmemeval','locomo']):
                labels=['Full liquid','Hold score','Content fallback','Linear only']
                vals=[next(r['recall']*100 for r in summary if (r['dataset'],r['budget'],r['kind'],r['policy'])==(dataset,64000,'liquid',p)) for p in ['full','stride2']]
                vals += [next(r['recall']*100 for r in fallback['summary'] if (r['dataset'],r['budget'],r['kind'],r['policy'])==(dataset,64000,'liquid',p)) for p in ['hybrid2','linear_only']]
                axis.bar(range(4),vals,color=['#697580','#ba4a4a','#176b87','#44a38a']);axis.set(xticks=range(4),xticklabels=labels,title=dataset+' | 64 KB',ylabel='Evidence recall (%)');axis.tick_params(axis='x',rotation=20)
                for i,value in enumerate(vals):axis.text(i,value+1,f'{value:.2f}',ha='center',fontsize=9)
                axis.set_ylim(0,75)
            for kind,color in [('liquid','#176b87'),('gru','#d17b0f'),('static','#697580')]:
                vals=[next(r['median_ms_per_event'] for r in fallback_timing if (r['kind'],r['policy'])==(kind,p)) for p in ['full','hybrid2','linear_only']]
                axes2[2].plot(range(3),vals,marker='o',color=color,label=kind)
            axes2[2].set(xticks=range(3),xticklabels=['Full','Hybrid','Linear'],title='Same optimized teacher backend',ylabel='Online CPU ms/event');axes2[2].legend();fig2.tight_layout();fig2.savefig(OUT/'fallback.png',dpi=170);plt.close(fig2)
            fallback_html+='<img src="fallback.png">'
        fallback_html+='<p>比较液态是否独有时，还需看与GRU/静态的差分以及与linear_only的比较，完整区间在 content-fallback-v1/attribution.json。LoCoMo128KB几乎全部内容能装下，任何方法在那里通过都不能作为评分机制优势证据。</p>'
        fallback_html+=table(['追加QA方法','完成','正确'],[[r['method'],r['completed'],r['correct']] for r in fallback['qa']])
        fallback_html+=f"<p>追加问答采用同六题、相同容量与判分。新增报告未缓存tokens：{fallback['usage']['reported_uncached_tokens']:,}。协议、训练ID、学生权重和完整结果见 <a href='../content-fallback-v1/protocol.json'>第二步协议</a>、<a href='../content-fallback-v1/summary.json'>汇总</a>、<a href='../content-fallback-v1/paired_differences.json'>配对区间</a>。</p>"
    erows=[]
    for kind in ['liquid','gru','static']:
        values=[next(r['ms_per_session'] for r in engineering if r['kind']==kind and r['method']==m) for m in ['online_original','online_fused','online_scripted']]
        erows.append([kind]+[f'{v:.5f}' for v in values]+[f'{values[0]/values[-1]:.2f}x'])
    drows=[]
    for d in diffs:
        if d['budget']==64000 and d['policy']=='stride2':
            base=next(r['recall'] for r in summary if (r['dataset'],r['budget'],r['kind'],r['policy'])==(d['dataset'],64000,d['kind'],'full'))
            drows.append([d['dataset'],d['kind'],f"{d['savings']*100:.1f}%",f'{base*100:.2f}%',f"{(base+d['difference'])*100:.2f}%",f"[{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}]",str(d['passes_joint_screen'])])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>工程优化与评分预算研究</title><style>body{{max-width:1150px;margin:35px auto;padding:0 24px;font:16px/1.75 system-ui;color:#24323b}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#eef3f5}}img{{width:100%}}.box{{background:#eef3f5;padding:18px}}</style><h1>工程提速与液态计算预算：分开验证</h1>
    <p class="box">工程线已实现等价推理提速；研究线经过“沿用旧分数失败→训练集内容替代评分”完成两步实验，出现减少约一半递推调用且保持召回点估计的候选，但尚未证明液态独有或问答非劣。减少评分调用不等于减少存储写入、嵌入、检索或 API 请求。</p><img src="results.png">
    <h2>工程线：严格逐条在线耗时</h2>{table(['写入器','原实现 ms/条','合并操作 ms/条','脚本化 ms/条','相对自身提速'],erows)}
    <p>CPU、单线程、同权重、3次预热、15次交错重复；脚本构建时间另存。液态将速率/候选/重要性三个投影合并为一次投影。GRU也进行等价优化。独立流批次1/8/32的完整吞吐结果见 engineering.json；大批量吞吐不能当作单Agent延迟。脚本化使用本机可运行的 TorchScript，运行时提示弃用；作为本轮测量后端保留，尚未替换生产默认路径。</p>
    <p>全部103组真实历史、9个模型共{len(eq)}组验证，输出最大绝对误差{audit['max_abs_error']:.3g}，两档容量下保留集合完全一致。不是仅凭随机输入宣布等价。批量投影只作用于各自文本，递推仍按时间顺序。</p>
    <h2>研究线：固定每两条评分一次</h2><p>主预算64,000字节，三个种子先按题求均值，再对历史聚类做5,000次配对bootstrap。筛选标准预先定为至少45%评分调用减少且召回差值95%区间下界高于−1个百分点。这里的通过只指开发性证据召回筛选；并非最终QA非劣结论。LoCoMo只有10个对话簇。</p>
    {table(['数据','方法','调用减少','全评分召回','稀疏评分召回','差值95%区间/百分点','通过筛选'],drows)}
    <p>同时保留stride4和novelty（余弦低于0.8或间隔达4条则评分）结果，没有事后选阈值。所有策略保留每一条新记录参与容量竞争；略过评分时沿用最近分数，实际容量仍执行淘汰。时间累计到下一次真实评分，最大截断为12。新奇性门不能证明语义等义，更不用于删掉信息。完整控制器时间含门控成本，见 costs.json；单次观察耗时不能代替稳定的提速基准。</p>
    <h2>是否液态独有</h2><p>同一个门控规则分别用于液态、GRU与无历史模型。attribution.json 给出“液态稀疏−液态全评分”减去“基线稀疏−基线全评分”的配对区间。没有这个额外优势，就只能称作通用评分预算收益。当前实验是基线摸底，尚未实现以学习到的液态动力学选择调用时机的策略，也未实现动态重评分。</p>
    <h2>实际 Agnes 问答</h2>{table(['方法','完成','正确','无效判分'],[[r['method'],r['completed'],r['correct'],r['invalid_judges']] for r in qa_summary])}
    <p>使用此前按问题类型与哈希固定的6道LongMemEval问题，4条件，共24份回答；不是挑选本轮有利问题。相同64KB容量、seed23、相同阅读器和提示词，Agnes独立调用自定义判分，非官方评测器。六题结果只能发现实际链路退步，不能证明非劣。答复与错误均保存，API限速9次/分钟、支持缓存恢复。</p>
    <p>当前唯一已保存响应{usage['unique_response_hashes']}个，未缓存响应{usage['uncached_responses']}个，报告未缓存token共{usage['reported_uncached_tokens']:,}。失败请求额外费用未知；未把缓存响应重复计费。</p>
    {fallback_html}
    <h2>审计与复现</h2><p>首阶段共{len(rr):,}条评估，{matched:,}条完整评分结果与上一轮逐条一致；权重哈希未变，容量越界为零。使用既有暴露的开发数据，不能称新独立验证。Agent增加可选 --writer-backend fused|scripted，默认仍为original；优化版真实49条事件回放与恢复的上下文、来源、容量、淘汰数均与原版一致。稀疏评分未接入默认Agent策略。执行 writer_efficiency.py --phase all 可恢复首阶段，已完成阶段不重复请求。完整测试见 tests.txt。</p>
    <ul>{''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['protocol.json','engineering.json','summary.json','paired_differences.json','attribution.json','cost_summary.json','equivalence.json','qa_summary.json','qa_failures.json','api_usage.json','audit.json','tests.txt'])}</ul></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps([d for d in diffs if d['budget']==64000 and d['policy']=='stride2'],indent=2))
    print(json.dumps(qa_summary,indent=2))

if __name__=='__main__':main()

