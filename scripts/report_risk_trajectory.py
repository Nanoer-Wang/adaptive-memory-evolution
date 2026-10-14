"""Multi-dimensional evidence ledger and bounded claims for the staged study."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,html,hashlib,sqlite3
from collections import defaultdict,Counter
import numpy as np
from src.config import atomic_json,load_config,resolve
from scripts.report_hybrid_validation import interval
PARENT=ROOT/'runs/risk-trajectory-v1';COMPACT=ROOT/'runs/risk-trajectory-compact-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def analyze(path):
    groups=defaultdict(list);questions=defaultdict(lambda:defaultdict(list));clusters={};slices=defaultdict(list);seeds=defaultdict(list);count=0;violations=0
    with (path/'records.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads,f):
            count+=1;violations+=r['payload_bytes']>r['budget']
            if r['recall'] is None:continue
            groups[r['dataset'],r['budget'],r['method']].append([r['recall'],r['retention'],r['all_evidence_retrieved'],r['latest_evidence_retained']])
            seeds[r['dataset'],r['budget'],r['method'],r['seed']].append(r['recall'])
            questions[r['dataset'],r['budget'],r['question_id']][r['method']].append(r['recall']);clusters[r['dataset'],r['question_id']]=r['cluster']
            age='<=30days' if r['history_days']<=30 else ('31-180days' if r['history_days']<=180 else '>180days')
            for axis,label in [('category',str(r['category'])),('span',age)]:slices[r['dataset'],r['budget'],r['method'],axis,label].append([r['recall'],r['retention'],r['all_evidence_retrieved'],r['latest_evidence_retained']])
    summary=[]
    for k,values in groups.items():
        means=np.mean(values,axis=0);summary.append(dict(dataset=k[0],budget=k[1],method=k[2],recall=float(means[0]),retention=float(means[1]),complete_evidence=float(means[2]),latest_labeled_evidence_retained=float(means[3])))
    comparisons=[]
    for dataset in ['longmemeval','locomo']:
        for budget in [32000,64000,128000]:
            for other in ['cheap','gru_risk_risk','static_risk_risk','liquid_full_risk','liquid_full_full','liquid_periodic_periodic','liquid_risk_reset','liquid_risk_zero_time']:
                cs=defaultdict(list)
                for (ds,b,q),v in questions.items():
                    if (ds,b)==(dataset,budget):cs[clusters[ds,q]].append(float(np.mean(v['liquid_risk_risk'])-np.mean(v[other])))
                comparisons.append(dict(dataset=dataset,budget=budget,other=other,**interval(cs)))
    atomic_json(path/'summary.json',summary);atomic_json(path/'comparisons.json',comparisons)
    atomic_json(path/'seed_metrics.json',[dict(dataset=k[0],budget=k[1],method=k[2],seed=k[3],recall=float(np.mean(v))) for k,v in seeds.items()])
    atomic_json(path/'slices.json',[dict(dataset=k[0],budget=k[1],method=k[2],axis=k[3],value=k[4],observations=len(v),recall=float(np.mean(np.array(v)[:,0])),retention=float(np.mean(np.array(v)[:,1])),complete_evidence=float(np.mean(np.array(v)[:,2])),latest_labeled_evidence_retained=float(np.mean(np.array(v)[:,3])),scope='secondary exploratory; latest label is not verified latest truth') for k,v in slices.items()])
    gates=read(path/'gates.json');gate_summary=[]
    for dataset in ['longmemeval','locomo']:
        rs=[r for r in gates if r['dataset']==dataset];unique={r['cluster']:r for r in rs};events=sum(r['events'] for r in unique.values());calls=sum(r['calls'] for r in unique.values());eligible=[r for r in rs if r['positive_events'] is not None]
        gate_summary.append(dict(dataset=dataset,histories=len(unique),events=events,calls=calls,call_saving=1-calls/events,threshold_requests=sum(r.get('threshold_requests',0) for r in unique.values()),evidence_trigger_fraction=sum(r['triggered_positive'] for r in eligible)/sum(r['positive_events'] for r in eligible),risk_target_mae=float(np.mean([r['risk_target_mae'] for r in eligible if r.get('risk_target_mae') is not None]))))
    atomic_json(path/'gate_summary.json',gate_summary)
    qa=[read(p) for p in (path/'qa').glob('*.json')] if (path/'qa').exists() else [];qa_summary=[]
    for method in ['cheap','liquid_full_full','liquid_risk_risk','gru_risk_risk']:
        rs=[r for r in qa if r['method']==method and 'judge_response' in r]
        coverage=[r['context_sentence_coverage'] for r in rs if r.get('context_sentence_coverage') is not None]
        qa_summary.append(dict(method=method,completed=len(rs),correct=sum(r['custom_judge_correct']==1 for r in rs),invalid=sum(r['custom_judge_correct'] is None for r in rs),mean_sentence_coverage=float(np.mean(coverage)) if coverage else None))
    atomic_json(path/'qa_summary.json',qa_summary);atomic_json(path/'qa_failures.json',[r for r in qa if 'judge_response' in r and r['custom_judge_correct']!=1])
    pairs=[];lookup={(r['question_id'],r['method']):r for r in qa if 'judge_response' in r}
    for other in ['cheap','liquid_full_full','gru_risk_risk']:
        diffs=[]
        for q,method in lookup:
            if method!='liquid_risk_risk' or (q,other) not in lookup:continue
            a=lookup[q,method]['custom_judge_correct'];b=lookup[q,other]['custom_judge_correct']
            if a is not None and b is not None:diffs.append(a-b)
        if diffs:pairs.append(dict(other=other,wins=sum(x>0 for x in diffs),losses=sum(x<0 for x in diffs),ties=sum(x==0 for x in diffs),**interval({str(i):[x] for i,x in enumerate(diffs)})))
    atomic_json(path/'qa_pairs.json',pairs);atomic_json(path/'audit.json',dict(records=count,capacity_violations=violations,qa_complete=sum(r['completed'] for r in qa_summary)))
    return dict(summary=summary,comparisons=comparisons,gates=gate_summary,qa=qa_summary,pairs=pairs,records=count)

def main():
    results={}
    for name,path in [('vector',PARENT/'controller-accounted'),('compact',COMPACT)]:
        if (path/'offline_complete.json').exists():results[name]=analyze(path)
    if not results:return
    latest=results.get('compact',results['vector']);latencies={}
    for name,path in [('vector',PARENT),('compact',COMPACT)]:
        if (path/'online_latency_summary.json').exists():latencies[name]=read(path/'online_latency_summary.json')
    hashes=set();qa_sets=[PARENT,PARENT/'controller-accounted',COMPACT]
    for path in qa_sets:
        for p in (path/'qa').glob('*.json'):
            r=read(p)
            for field in ['generation','judge_response']:
                if field in r:hashes.add(r[field]['request_hash'])
    config=load_config('configs/minute_precision.yaml')['api']
    with sqlite3.connect(resolve(config['cache_db'])) as db:attempts=[dict(zip(['hash','started','status','usage'],r)) for r in db.execute('SELECT request_hash,started,status,usage_json FROM requests_log') if r[0] in hashes]
    for r in attempts:r['usage']=json.loads(r['usage'])
    times=sorted(r['started'] for r in attempts);left=0;maximum=0
    for i,t in enumerate(times):
        while times[left]<=t-60:left+=1
        maximum=max(maximum,i-left+1)
    api=dict(attempts=len(attempts),statuses=dict(Counter(r['status'] for r in attempts)),reported_tokens=sum(r['usage'].get('total_tokens',0) for r in attempts),rolling60_max=maximum,scope='Original,controller-accounted,compact stages; hashes deduplicated')
    atomic_json(PARENT/'combined_api.json',api);atomic_json(PARENT/'api_attempts.json',attempts)
    training=[]
    for stage,path in [('original',PARENT),('compact',COMPACT)]:
        for p in (path/'checkpoints').glob('*/epochs.json'):
            if stage=='compact' and '-risk-' not in p.parent.name:continue
            logs=read(p);training.append(dict(stage=stage,model=p.parent.name,epochs=logs,seconds=sum(r['seconds'] for r in logs),selected=read(p.parent/'complete.json')['best']))
    atomic_json(PARENT/'training_summary.json',training)
    claims=[dict(claim='Hard prefix call bound',status='implemented and tested',scope='Token bucket credit1,rate0.5,capacity2 gives calls(t)<=floor(1+0.5t); says nothing about quality or end-to-end cost.'),dict(claim='Train/deploy trajectory consistency',status='implemented',scope='Matched regimes share exact masks, recurrence and residual score scale; inferential performance benefit must still be shown.'),dict(claim='Risk is semantic contradiction probability',status='not supported',scope='It predicts a class-balanced cheap-score error proxy, not a semantic truth relation.'),dict(claim='Liquid-specific improvement',status='must satisfy saved comparisons; not presumed',scope='No success narrative may ignore cheap/GRU/static or time/reset ablations.'),dict(claim='Smaller controller memory',status='implemented;1544 to16 serialized dynamic bytes',scope='Shared coefficients,Python object overhead and text encoder excluded; controller feature also changes, so this is a resource-design comparison, not an identical-gate compression theorem.'),dict(claim='CCF-A-ready publication or new independent replication',status='not established',scope='Previously exposed data,12 reused QA items,custom same-provider judge;27+9trained models are experiments,not independent datasets.')]
    atomic_json(PARENT/'claim_ledger.json',claims)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    methods=['cheap','liquid_full_full','liquid_full_risk','liquid_risk_risk','gru_risk_risk','static_risk_risk']
    for axis,dataset in zip(axes[:2],['longmemeval','locomo']):
        for stage,color in [('vector','#a97777'),('compact','#176b87')]:
            if stage not in results:continue
            values=[next(r['recall']*100 for r in results[stage]['summary'] if (r['dataset'],r['budget'],r['method'])==(dataset,64000,m)) for m in methods]
            axis.plot(range(len(methods)),values,'o-',label=stage,color=color)
        axis.set(title=dataset+' | controller-inclusive64KB',xticks=range(len(methods)),xticklabels=['cheap','full','mismatch','liquid risk','GRU risk','static risk'],ylabel='Evidence recall (%)');axis.tick_params(axis='x',rotation=35);axis.legend();axis.grid(alpha=.2)
    for stage,values in latencies.items():
        ys=[next(r['ms_per_event'] for r in values if (r['kind'],r['policy'])==('liquid',p)) for p in ['full','periodic','risk','cheap']];axes[2].plot(range(4),ys,'o-',label=stage)
    axes[2].set(title='CPU online cost,including controller',xticks=range(4),xticklabels=['full','periodic','risk','cheap'],ylabel='ms/event');axes[2].legend();axes[2].grid(alpha=.2);fig.tight_layout();fig.savefig(PARENT/'series.png',dpi=170);plt.close(fig)
    def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    metrics=[]
    for stage,r in results.items():
        for dataset in ['longmemeval','locomo']:
            for method in methods:
                s=next(x for x in r['summary'] if (x['dataset'],x['budget'],x['method'])==(dataset,64000,method));metrics.append([stage,dataset,method]+[f"{s[k]*100:.2f}%" for k in ['recall','complete_evidence','latest_labeled_evidence_retained']])
    comp=[x for x in latest['comparisons'] if x['dataset']=='longmemeval' and x['budget']==64000]
    primary=[x for x in comp if x['other'] in ['cheap','gru_risk_risk','static_risk_risk','liquid_full_risk']];passed=all(x['ci9875'][0]>0 for x in primary)
    rows=[]
    for stage,r in results.items():
        for g in r['gates']:rows.append([stage,g['dataset'],f"{g['call_saving']*100:.2f}%",f"{g['evidence_trigger_fraction']*100:.2f}%",f"{g['risk_target_mae']:.3f}",g['calls'],g['threshold_requests']])
    qa_rows=[[stage,r['method'],r['completed'],r['correct'],None if r['mean_sentence_coverage'] is None else f"{r['mean_sentence_coverage']*100:.1f}%"] for stage,d in results.items() for r in d['qa']]
    latency_rows=[[stage,r['kind'],r['policy'],f"{r['ms_per_event']:.5f}"] for stage,values in latencies.items() for r in values]
    old=read(ROOT/'runs/hybrid-validation-v1/summary.json');anchors=[]
    for dataset in ['longmemeval','locomo']:
        for method in ['liquid_full','liquid_linear']:
            value=next(r['recall'] for r in old if (r['dataset'],r['budget'],r['method'])==(dataset,64000,method))
            anchors.append(dict(dataset=dataset,method='historical_'+method,recall=value,scope='Older frozen pipeline, same public histories/compression/retrieval budget, different training recipe; reference barrier, not controlled causal comparison'))
    atomic_json(PARENT/'historical_anchors.json',anchors)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>风险触发与稀疏轨迹一致性研究</title><style>body{{max-width:1250px;margin:34px auto;padding:0 24px;font:16px/1.75 system-ui;color:#23333b}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf3f5}}img{{width:100%}}.box{{background:#edf3f5;padding:20px}}</style>
    <h1>风险触发语言记忆：预算、执行轨迹与证据完整性</h1><p class="box">完成分阶段实验：共享便宜评分、风险预算门控、27组执行轨迹训练、控制器内存核算、紧凑门控后的9组重新训练、跨数据及实际问答。当前预设液态独有判据是否通过：{passed}。以下将可复现的工程事实、待验证机制与不能主张的结论分开保存，不把某个有利切片写成整体成功。</p><img src="series.png">
    <h2>研究主线与数学定义</h2><p>问题是：在固定内存和调用上限下，如何把昂贵的状态更新分配给更可能改变记忆价值的观察，同时对每条新内容维持有效评分？便宜分数b(x)每条计算；风险代理r(x,历史统计,Δt)决定是否允许更新；预算信用满足c←min(2,c+0.5)，只有风险超过验证集阈值且c≥1才更新，随后c←c−1。任意前缀t次输入的调用不超过floor(1+0.5t)。这是调用上限保证，不是质量保证。</p>
    <p>状态先按h⁻=exp[−softplus(λ)Δt]⊙h演化，再在被选中时做观察更新。所有事件使用同一尺度s=b(x)+head(h)，不再在教师分数与学生分数之间任意切换。液态、GRU与单次内容更新基线都获得相同的时间衰减、便宜评分和门控；这里的液态单元仍是指数门控递推近似，不冒充精确ODE或标准CfC。</p>
    <h2>训练、资源修正与公平性</h2><p>仅train拟合便宜模型与风险代理，val校准约48%调用率、选择检查点；test不参与拟合或调参。三个种子、三类模型、全量/周期/风险三种训练轨迹，共27模型各3epoch。全量训练后风险执行作为失配对照。风险目标是便宜预测的类平衡平方误差；训练目标为样本内残差，不能把它称为校准的矛盾概率。便宜评分和风险规则对各模型完全相同。</p>
    <p>初版漏计了上一条384维向量与调用信用的1544字节动态状态。原结果保留，controller-accounted中以总预算重新评估；没有重新选择检查点，原验证选择仍是核心状态预算代理，这限制了最优资源曲线主张。第二阶段将历史特征换为基础评分变化，只保留上一分数和信用共16字节，并重新拟合门控、重训9个风险模型；18个全量/周期模型逐字节复用。特征发生了变化，不能称为完全等价压缩。共享权重、Python对象和编码器未包含在序列化存储预算中。</p>
    <h2>多维质量：64KB总预算</h2>{table(['阶段','数据','方法','证据召回','全部支持会话取回','最新标注证据保留'],metrics)}
    <p>“最新标注证据”只指公开答案标注中日期最近的会话，不等于识别了真实世界最新事实。知识更新/时间推理题型、三个容量及历史跨度均完整保存在slices.json；这些切片是探索性分析，不能挑一个优势切片替代总体结论。LoCoMo只有10个独立历史簇，128KB几乎无淘汰压力。</p>
    <h2>机制与一致性：紧凑版本优先展示</h2>{table(['液态风险减去','差值/百分点','95%区间','98.75%区间'],[[x['other'],f"{x['difference']*100:+.3f}",str([round(v*100,3) for v in x['ci95']]),str([round(v*100,3) for v in x['ci9875']])] for x in comp])}
    <p>三个种子先按题求均值，再按历史聚类做5000次bootstrap。四主比较的98.75%区间用于保守多比较校正；与完整液态比较的非劣界限仍为−1个百分点。工程上的轨迹一致已实现，不代表实验证明它一定提升质量。zero_time消融只移除核心的自主时间衰减，风险门本身仍接收原时间特征，不能据此声称一切时间信息都无用。</p>
    <h2>保留历史强基线，避免降低参照门槛</h2>{table(['数据','历史方案','证据召回'],[[r['dataset'],r['method'],f"{r['recall']*100:.2f}%"] for r in anchors])}<p>新方案与本轮完整模型持平，并不等于超过项目已有最好方案。旧的蒸馏线性基线也应保留：同12道开发问答曾为8/12，本轮各新方案为4/12；提示与评测同类，但不同阶段调用不可当作完全确定的因果实验。新共享基础评分的训练配方较弱，是本轮的重要限制。当前结果不能写成综合SOTA或完整成本—质量前沿优势。</p>
    <h2>风险代理是否把调用用在证据上</h2>{table(['阶段','数据','调用减少','标注证据触发率','误差目标MAE','实际调用','超阈值请求'],rows)}
    <p>未触发递推的记录仍有当前内容评分并参与存储，因此“证据未触发”不等于被删除。触发率不是事实修订检测准确率。超阈值请求与实际调用的差异显示预算约束是否主导决策；调用统计按独立历史去重，标签相关指标按问题报告。</p>
    <h2>真实在线开销</h2>{table(['阶段','核心','策略','毫秒/事件'],latency_rows)}
    <p>CPU单线程，同一优化核心，前8组公开历史，3次预热7次轮换测量；包含实时便宜评分、风险计算、状态衰减及输出，不含编码、检索、存储淘汰和API。未优化风险特征分配的旧时延单独保留；等价投影折叠须验证所有历史的触发一致。少调用一半不是总耗时减半，也不是LLM token减半。</p>
    <h2>实际问答与最终上下文</h2>{table(['阶段','方法','完成','答对','证据句子覆盖'],qa_rows)}
    <p>复用此前12道固定开发问题，每阶段四条件、seed23/64KB；自定义同提供商判分，不是新独立样本或官方评测。逐题输赢、参考、原回答、上下文及失败全部保存。不因拒答、证据缺失或结果不利而删题。</p>
    <h2>创新点的边界与可写主线</h2><p>可组织为“证据完整性约束下的预算化语言记忆”：一条低成本内容通道保证每个观察得到评分；一条受调用预算约束的状态通道处理历史修正；训练显式重放部署策略；评价同时核算控制器自身的时间与空间。这是当前可检验的设计主线，尚不能写成已证实液态优于所有普通门控的论文摘要。</p>
    <p>仅“跳过RNN状态更新”已有<a href="https://arxiv.org/abs/1708.06834">Skip RNN</a>；周期时间门已有<a href="https://arxiv.org/abs/1610.09513">Phased LSTM</a>；无需常规ODE求解器的连续时间网络已有<a href="https://www.nature.com/articles/s42256-022-00556-7">CfC</a>。因此潜在贡献必须落在语言证据的风险分配、预算保证、轨迹一致性及完整系统验证，而不能只重新命名门控。这里是相关工作边界核查，不是穷尽检索或新颖性认证。</p>
    <p>支持继续研究的具体信号：紧凑状态释放预算后，保持约一半递推调用，并在本轮完整模型对照下达到证据召回非劣筛选；评分子系统约17%提速，非端到端Agent提速。未建立的关键结论：液态独有收益、训练一致性的额外性能收益、跨域风险选择能力及相对历史强基线的优势。下一步应固定或恢复强基础评分，针对真正的证据修订风险验证跨域触发，而不是继续降低基础评分质量或只调核心门控。</p><p>尚缺：未暴露的独立数据、覆盖事实修订的可靠风险标签、正式独立问答评分、更大样本与真实多轮Agent任务。若液态不超过简单基线，保留它作为一种实例而不是强行归因；目前不能保证CCF-A录用或宣称科研任务完成。</p>
    <h2>保存与费用</h2><p>训练、每轮验证、所有检查点、完整切片与原始问答均保留。当前跨阶段HTTP {api['attempts']}次，状态{html.escape(str(api['statuses']))}，报告tokens {api['reported_tokens']:,}，滚动60秒最多{api['rolling60_max']}次。已缓存结果按请求哈希去重，失败调用未知费用不计入成功响应token。</p>
    <ul>{''.join(f'<li><a href="{x}">{x}</a></li>' for x in ['protocol.json','controller-accounted/amendment.json','controller-accounted/summary.json','controller-accounted/comparisons.json','controller-accounted/slices.json','controller-accounted/qa_summary.json','training_summary.json','claim_ledger.json','online_latency_summary.json','combined_api.json','tests.txt'])}<li><a href="../risk-trajectory-compact-v1/protocol.json">紧凑阶段协议</a></li><li><a href="../risk-trajectory-compact-v1/comparisons.json">紧凑阶段全部区间</a></li><li><a href="../risk-trajectory-compact-v1/qa_pairs.json">紧凑阶段配对问答</a></li></ul></html>'''
    (PARENT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(dict(primary=primary,qa=latest['qa'],gates=latest['gates'],api=api),indent=2))

if __name__=='__main__':main()
