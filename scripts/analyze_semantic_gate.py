"""Grouped uncertainty, error audit and cost boundaries for frozen gate study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.semantic_gate import wilson_upper

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/semantic-gate-v2'


def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    summaries=load('summary.json');frozen=load('frozen_models.json');timings=load('timing.json');examples=load('examples.json')
    rows={r['id']:r for r in examples};predictions=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text().splitlines()]
    methods=('exact_match','cosine','cosine_numeric_negation_guard','logistic_lexical','mlp','gru','input_flow','ltc')
    aggregate=[]
    for method in methods:
        rs=[r for r in summaries if r['split']=='test' and (r['method']==method or r['method'].startswith(method+'-'))]
        ts=[r for r in timings if r['method'].startswith(method+'-')]
        challengers=[r for r in summaries if r['split']=='challenge' and (r['method']==method or r['method'].startswith(method+'-'))]
        result=dict(method=method,skip_fraction=float(np.mean([r['skip_fraction'] for r in rs])),
            duplicate_recall=float(np.mean([r['duplicate_recall'] for r in rs])),false_skip_rate=float(np.mean([r['false_skip_rate'] for r in rs])),
            auc=float(np.mean([r['auc'] for r in rs])),wrong_fraction_among_skips=float(np.mean([r['false_skips']/(r['skip_fraction']*r['n']) if r['skip_fraction'] else 0 for r in rs])),
            update_errors=[r['false_skips'] for r in challengers],update_n=challengers[0]['n'])
        if ts:
            result['gate_us']=float(np.mean([r['gate_us_p50'] for r in ts]))
            result['break_even_backend_us_excluding_encoder']=result['gate_us']/result['skip_fraction'] if result['skip_fraction'] else None
        aggregate.append(result)
    # Components share an actual sentence; resample whole components, all seeds
    # averaged within each test pair. Does not handle latent Wikipedia articles.
    test=[r for r in examples if r['split']=='test'];groups=sorted({r['group'] for r in test});gi={g:i for i,g in enumerate(groups)}
    byid=defaultdict(dict)
    for p in predictions:
        if p['split']=='test':byid[p['method']][p['id']]=p
    arrays={}
    for kind in ('mlp','gru','input_flow','ltc'):
        matrix=np.zeros((len(groups),3))
        for row in test:
            skips=np.mean([byid[f'{kind}-{s}'][row['id']]['skip'] for s in (11,23,42)])
            matrix[gi[row['group']]]+=[skips,skips*(1-row['label']),1]
        arrays[kind]=matrix
    rng=np.random.default_rng(12345);indices=rng.integers(0,len(groups),(3000,len(groups)));paired=[]
    for control in ('mlp','gru','input_flow'):
        diff=arrays['ltc']-arrays[control];denom=arrays['ltc'][indices,2].sum(1)
        delta=diff[indices,0].sum(1)/denom
        paired.append(dict(contrast=f'ltc - {control}',skip_fraction_difference=float(diff[:,0].sum()/len(test)),
                           ci95=np.quantile(delta,[.025,.975]).tolist(),groups=len(groups),
                           caveat='Unadjusted exploratory interval; sentence components, not article clusters. Different false-skip rates, not matched realized risk.'))
    # CPU lower-cost baselines on the same precomputed content feature interface.
    # Feature extraction and encoding are excluded just as for neural timings.
    data=np.load(OUT/'features.npz');pairs=data['vectors'][data['pairs']];lex=data['lexical']
    model=load('logistic.json');coef=np.asarray(model['coef'])[0];mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);intercept=model['intercept'][0]
    baseline_timing=[]
    for name in ('exact_match','cosine','cosine_numeric_negation_guard','logistic_lexical'):
        values=[]
        for step in range(1050):
            i=step%len(pairs);a=time.perf_counter_ns()
            if name=='exact_match':score=lex[i,0]
            else:
                cos=float(np.dot(pairs[i,0],pairs[i,1]))
                if name=='cosine':score=cos
                elif name=='cosine_numeric_negation_guard':score=cos if lex[i,4]==0 and lex[i,5]==0 else 0
                else:
                    x=np.r_[lex[i],cos];score=1/(1+np.exp(-(((x-mean)/scale)@coef+intercept)))
            elapsed=time.perf_counter_ns()-a
            if step>=50:values.append(elapsed/1000)
        baseline_timing.append(dict(method=name,precomputed_feature_gate_us_p50=float(np.median(values)),raw_us=values,
            caveat='Python/numpy, excludes lexical extraction/encoding; exact can avoid semantic encoder entirely in deployment. Not apples-to-apples end-to-end cost.'))
    audit=[]
    for p in predictions:
        if p['split']=='test' and p['label']==0 and p['skip'] and p['method'] in ('exact_match','ltc-11','logistic_lexical'):
            e=rows[p['id']];audit.append(dict(**p,text1=e['text1'],text2=e['text2'],raw_text_identical=e['text1']==e['text2']))
    analysis=dict(aggregate=aggregate,paired_bootstrap=paired,baseline_timing=baseline_timing,
        challenge_zero_error_wilson95_upper=wilson_upper(0,147),
        risk='Calibration upper-bound screening is not a post-selection/shift-robust guarantee; logistic exceeds the 1% target on held-out data. False-skip denominator is different-content pairs; also show errors among skipped pairs.',
        costs='Gate timings only. No expensive backend was run or removed. Skip fraction is not measured wall-clock/API saving; break-even values exclude encoder/retrieval/lexical costs.',
        scientific='LTC not shown superior to GRU/input flow. Learned lexical baseline has higher AUC but violates test risk target. Pair-local benchmark does not test adaptive timescales over real streams.')
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8');(OUT/'false_skips.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    names={'exact_match':'Exact','cosine':'Cosine','cosine_numeric_negation_guard':'Cosine + guard','logistic_lexical':'Lexical logistic','mlp':'MLP','gru':'GRU','input_flow':'Input flow','ltc':'LTC'}
    plot=[r for r in aggregate if r['method'] not in ('exact_match','cosine_numeric_negation_guard')]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5))
    colors=['#64748b','#d97706','#8b5cf6','#059669','#0891b2','#dc2626']
    for r,color in zip(plot,colors):
        axes[0].scatter(r['false_skip_rate']*100,r['skip_fraction']*100,s=70,color=color,label=names[r['method']])
    axes[0].axvline(1,color='#333',linestyle='--',linewidth=1)
    axes[0].set(xlabel='Different content incorrectly skipped (%)',ylabel='All inputs skipped (%)',title='Frozen thresholds, 8,000 held-out pairs')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].bar(np.arange(len(plot)),[r['auc'] for r in plot],color=colors)
    axes[1].set(xticks=np.arange(len(plot)),xticklabels=[names[r['method']] for r in plot],ylim=(.5,.85),ylabel='ROC AUC',title='Semantic separation (higher is better)')
    axes[1].tick_params(axis='x',rotation=25);axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    table=''.join(f'<tr><td>{names[r["method"]]}</td><td>{r["skip_fraction"]:.2%}</td><td>{r["false_skip_rate"]:.2%}</td><td>{r["wrong_fraction_among_skips"]:.1%}</td><td>{r["auc"]:.3f}</td><td>{r.get("gate_us",float("nan")):.1f}</td></tr>' for r in aggregate)
    interval=''.join(f'<li>{r["contrast"]}：跳过比例差 {r["skip_fraction_difference"]*100:.3f} 个百分点，95% 区间 [{r["ci95"][0]*100:.3f}, {r["ci95"][1]*100:.3f}]。</li>' for r in paired)
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>语义重复与关键修订：写入判别实验</title><style>body{{font:16px/1.7 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}.lead{{background:#edf3f8;padding:20px}}</style>
    <h1>语义重复与关键修订：写入判别实验</h1><p>2026-10-12 · 12 个神经模型 + 5 个基线 · CPU · 0 API 调用</p>
    <p class="lead">已经从四个位置事实推进到开放文本对判别。LTC 在保守阈值下跳过约 2% 输入，与 GRU / 输入条件化流接近；没有得到独有优势。更强的简单词序特征基线具有更高 AUC，但测试误删率超过预定 1% 目标，也不能直接部署。</p>
    <h2>数据与标签</h2><p><a href="https://github.com/google-research-datasets/paws">PAWS</a> 使用人工标注的等义关系，来源包含 Wikipedia 词序交换与回译，因此不是自然对话流。官方训练集按哈希取 6000 条；官方验证集按共享句子连通分量划为选模 3850、校准 4150；官方测试完整 8000 条。v1 按行划分发现重叠，在训练前停止并保留；v2 分组后各分区未发现规范化相同句子。没有宣称消除了预训练污染或所有文章层面的相关性。</p>
    <p><a href="https://github.com/MemTensor/MemOps">MemOps</a> 的公开生成式对话提供操作标注。77 个 update 文件中核实出 147 条已确认、值有变化、前驱值匹配、触发引文逐字存在于用户原文的修订；10 条值未改变和 8 条链不匹配未纳入，其他操作不属于该标签。完整审计保存。未把没有标注的句子当作重复，也未用助手回复自造标签。目标绑定、引文选择由 gold trace 提供，因此这是判别器检查，不是端到端抽取与记忆 benchmark。</p>
    <img src="comparison.png"><h2>冻结阈值后的完整测试</h2><table><tr><th>方法</th><th>跳过比例</th><th>非重复误删率</th><th>跳过样本中错误比例</th><th>AUC</th><th>神经判别 μs</th></tr>{table}</table>
    <p>神经结果为三个种子的平均；nan 表示该列不是神经内核计时，规则与逻辑回归的独立计时保存在 analysis.json。误删率分母是所有非重复对，不是所有已跳过对；两者不能混淆。选择的阈值在独立校准集上要求 Wilson 上界不超过 1%，但经过阈值选择和分布变化后不构成风险保证。测试逻辑回归达到约 1.19%，未满足目标，不能为增加跳过比例忽略该结果。</p>
    <h2>关键修订与统计边界</h2><p>所有方法在 147 条修订中均未跳过任何一条。这包括完全不跳过的基线，所以不能仅凭 0/147 排出模型优劣；单方法零错误的 Wilson95 上界仍约 {wilson_upper(0,147):.2%}，而且文件内样本有关联，独立样本假设偏乐观。</p><ul>{interval}</ul>
    <p>上述区间按共享句子连通分量配对重采样 3000 次、先对种子平均；未作多重比较校正，也没有按 Wikipedia 文章聚类。模型实际误删率不同，因此跳过比例差不是同实际风险下的纯优势。</p>
    <h2>错误审计与成本</h2><p>4 条 PAWS 测试样本的两句话逐字相同却标注为非重复，精确相等策略因此也计错。全部保留官方标签与主成绩，错误原文可追溯，未为改善结果删样本。所有神经方法共享冻结 MiniLM384 和词序、数字、否定等词法信号；输入流/GRU/LTC 每对重置状态，单位步不是物理时间。该轮不验证长时间跨度或连续记忆自适应。</p>
    <p>LTC 判别内核约 204 μs，GRU 约 78 μs；尚未包含编码、检索、词法处理或后台写入。没有实际运行昂贵写入后端，因此不能把 2% 跳过比例写成 2% 端到端加速，更不能声称已减少 API 费用。break-even 计算只是排除这些成本后的理论下界。下一步的证据指向更精确的语义关系表示和真实序列数据，而非继续扩大当前连续流。</p>
    <h2>可复现文件</h2><p><a href="protocol.json">协议与来源哈希</a> · <a href="analysis.json">统计及成本边界</a> · <a href="summary.json">全部指标</a> · <a href="frozen_models.json">冻结检查点与阈值</a> · <a href="false_skips.json">误删原文</a> · <a href="update_label_audit.json">修订标签审计</a> · <a href="predictions.jsonl">全部预测</a></p>
    <p>训练完成后，初次指标调用遗漏阈值参数，评分前停止；修复只作用于评价代码，逐文件验证冻结检查点不变。训练与评价源码快照分别保存在 protocol.json 和 evaluation_source.json。</p></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    (OUT/'analysis_manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(aggregate,indent=2));print(json.dumps(paired,indent=2))


if __name__=='__main__':main()
