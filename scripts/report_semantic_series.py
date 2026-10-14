"""Compare identical held-out actions under natural and augmented training."""
import sys,json,html,hashlib,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source
from scripts import semantic_fallback as base
from scripts import semantic_fallback_augmentation as extra
P=base.OUT;E=extra.OUT
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    ps,es=read(P/'summary.json'),read(E/'summary.json');pr,er=read(P/'policy_records.json'),read(E/'policy_records.json')
    by={name:{r['method']:r for r in s['methods'] if r['split']=='test' and r['budget']=='pooled'} for name,s in [('natural',ps),('augmented',es)]}
    ids=read(P/'protocol.json')['question_ids']['test'];rng=np.random.default_rng(20261012);effects=[]
    for method in base.KINDS:
        metrics={}
        for metric in ['correct','prompt_tokens','answer_calls','utility']:
            d=np.array([np.mean([r[metric] for r in er if r['question_id']==q and r['method']==method])-np.mean([r[metric] for r in pr if r['question_id']==q and r['method']==method]) for q in ids])
            boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1);metrics[metric]=dict(mean=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
        effects.append(dict(method=method,question_units=12,augmented_minus_natural=metrics))
    histories=[]
    for control in ['liquid_reset','liquid_zero_time','liquid_reverse']:
        a={(r['question_id'],r['budget'],r['seed']):r for r in er if r['split']=='test' and r['method']=='liquid'}
        b={(r['question_id'],r['budget'],r['seed']):r for r in er if r['split']=='test' and r['method']==control}
        histories.append(dict(control=control,decisions=len(a),questions=12,different_decisions=sum(a[k]['fallback']!=b[k]['fallback'] for k in a),
            correct_difference=float(np.mean([a[k]['correct']-b[k]['correct'] for k in a]))))
    ds=extra.augmented_data(False);train=[d for d in ds.values() if d['split']=='train'];positive=[d for d in train if d['full']['correct']>d['compact']['correct']]
    support=dict(train_pairs=len(train),positive_fallback_pairs=len(positive),positive_questions=len({d['question_id'] for d in positive}),
        positive_with_confident_wrong_answer=sum(not d['compact']['parsed']['abstain'] for d in positive),natural_pairs=sum(not d.get('intervention',False) for d in train),interventions=sum(d.get('intervention',False) for d in train))
    atomic_json(P/'series_comparison.json',dict(training_support=support,augmentation_effects=effects,history_controls=histories,
        inference_scope='Same-query source chronology, not persistent state across agent tasks. Full local context is already retrieved/encoded; fallback saves or adds an LLM call, not a database retrieval.',
        api=es['api'],limits='12different test questions; two budgets and model seeds clustered within question. One potential answer per action, no independent repeated generation. Development study, not formal superiority.'))
    names=[('full_only','直接完整读取'),('never_fallback','只读精简'),('abstention_fallback','拒答才补读'),('uncertainty_rule','拒答/不确定规则'),('linear_no_omission','线性：无缺失特征'),('linear','线性：含缺失特征'),('gru','GRU'),('liquid','液态')]
    table=''
    for method,label in names:
        a,b=by['natural'][method],by['augmented'][method]
        table+=f"<tr><td>{label}</td><td>{a['correct']:.1%}</td><td>{b['correct']:.1%}</td><td>{b['prompt_tokens']:.0f}</td><td>{b['answer_calls']:.2f}</td><td>{b['missed_helpful_fallback']:.1%}</td></tr>"
    htable=''.join(f"<tr><td>{r['control']}</td><td>{r['different_decisions']}/{r['decisions']}</td><td>{r['correct_difference']:+.1%}</td></tr>" for r in histories)
    concrete=[]
    for q in ['01493427','4d6b87c8','a4996e51']:
        r=read(E/'qa'/f'{q}-withheld1800.json');full=read(P/'qa'/f'{q}-full.json')
        concrete.append(f"<li>{html.escape(r['question'])}<br>干预后的回答：{html.escape(r['hypothesis'])}<br>完整回答：{html.escape(full['hypothesis'])}</li>")
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>语义缺失与真实补读收益：系列结果</title><style>body{{max-width:1100px;margin:36px auto;padding:0 18px;font:16px/1.75 system-ui;color:#223548}}h1,h2{{line-height:1.3}}.box{{background:#edf5f8;padding:18px;border-radius:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #dce3e8;text-align:left}}img{{width:100%}}pre{{white-space:pre-wrap}}a{{color:#176294}}</style>
    <h1>先补足语义观察与任务损失，再比较历史机制</h1><p class="box">已补上60组事实训练对照：40组自然预算、20组训练专用来源扣除干预。14组实测补读有益，覆盖10道题；其中6组精简回答没有拒答，却给出了错误事实。24个模型保持8维隐藏状态，全部冻结后才生成12道新QA测试题的36份动作回答。</p>
    <p>语义输入能观察被删去内容与查询的关系、与保留句子的相似度、数字及查询词的覆盖缺口，而不再只看全文长度和总体相关度。数值缺口单独保留，避免把近义但数值不同的句子视作完整覆盖。嵌入相似度不是逻辑蕴含，仍是风险代理。训练监督来自实际回答与参考答案的一致性，不再依赖偏好题裁判或召回比例。</p>
    <h2>相同自然测试题，比较补充前后的训练</h2><p>表中按两个预算和三个训练种子在问题内平均。每个预算先生成精简回答；补读时再付出一次完整LLM调用，所有token和调用均计入。直接完整读取是更强的成本参照，不要求先付精简调用成本。测试没有扣除来源的人为干预。</p><table><tr><th>方法</th><th>自然样本训练</th><th>加入干预训练</th><th>补充后输入token</th><th>回答调用</th><th>漏掉有益补读</th></tr>{table}</table>
    <img src="../semantic-fallback-augmentation-v1/results.png" alt="加入训练干预后的正确率与token成本">
    <h2>这次训练信号究竟是什么</h2><p>所有20道训练题都生成一次干预，不依据答对与否挑题；移除的是冻结编码器认为最相关的已选来源，参考答案不参与选择。它是明确标记的训练压力场景，不能当作自然分布上的成功证据。以下是三条不拒答但事实出错的训练实例：</p><ul>{''.join(concrete)}</ul>
    <h2>历史机制检查</h2><table><tr><th>对照</th><th>改变的动作</th><th>液态正确率减对照</th></tr>{htable}</table><p>72个动作比较来自12题×2预算×3种子，不是72道独立题。状态沿同一查询的已检索来源日期序列更新，不跨查询持续。反转顺序只是机制扰动，不是真实时间过程。当前本地全文已经检索和编码，“补读”指追加一次大模型调用，没有减少数据库检索，也没有证实端到端提速。</p>
    <h2>为什么补充数据反而增加了调用</h2><p>自然训练中完整读取更好的比例为4/40；加入扣除来源干预后为14/60，而自然测试为2/24。训练没有出现完整读取导致质量下降的案例，测试却出现3/24个这种动作对。干预补足了关键信息损失，但也改变了收益分布；这与补充模型更频繁调用、质量未改善的现象一致，不能把额外样本数量当作监督已经充分的证明。</p><p>当前一次完整输入约3244token，其惩罚仅为0.02×3244/4000≈0.0162个正确答案单位，目标本身容许为很小的预测收益追加调用。后续应明确调用约束并纳入“补读反而干扰”的反例，而不是继续增加网络容量。已追加<a href="calibration.html">固定模型的完整预算曲线</a>：阈值仅用验证分数设置，但分析是在看到主结果后提出，所有预算点均保留，不作为新的独立优势验证。</p>
    <h2>完整统计与复核</h2><p>首次模型和新增干预模型均在测试生成前冻结；训练与验证没有读取测试回答。12道测试题来自旧开发划分中此前未调用QA的题目，仍不是新独立数据集。自然训练/验证和干预训练分别保存，原始模型未覆盖。单次生成和小样本的区间只适于探索。</p>
    <pre>{html.escape(json.dumps(effects,ensure_ascii=False,indent=2))}</pre><p>本系列新请求及费用元数据：{html.escape(json.dumps(es['api'],ensure_ascii=False))}。还明确复用了30份旧开发集完整回答。限流会排队重试；一份格式无效的开发回答原样归档后单独重试，未对有效错误回答重新抽样。</p>
    <p>133项测试通过；预算和来源归属、训练划分、24个检查点冻结顺序已审计。原始回答、自动评分及格式失败均保留；本助手对全部新事实回答的参考一致性进行复核，未修改有效评分。这不是独立人工盲评或所有解释性陈述的穷尽事实核验。</p><p><a href="report.html">自然训练完整报告</a> · <a href="../semantic-fallback-augmentation-v1/report.html">加入干预的完整报告</a> · <a href="audit.json">预算、隔离与冻结审计</a> · <a href="series_comparison.json">全部比较数据</a> · <a href="resume_audit.json">恢复检查</a></p></html>'''
    (P/'index.html').write_text(doc,encoding='utf-8');source,archive=snapshot_source();atomic_json(P/'delivery.json',dict(source=source,archive=archive,training_support=support,api=es['api'],main_report='index.html'))
    print(json.dumps(dict(training_support=support,natural_test=by['natural'],augmented_test=by['augmented'],history=histories,api=es['api']),indent=2))

if __name__=='__main__':main()
